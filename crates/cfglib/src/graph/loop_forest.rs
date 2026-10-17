//! The loops of a graph as a forest, in near-linear space and time.
//!
//! [`detect_loops`](crate::detect_loops) returns each loop with an owned set
//! of its body blocks. Loops nested `n` deep share most of their blocks, so
//! those sets hold `n²/2` entries in all: 8,000 nested loops are 64 million
//! entries and nine seconds, and 100,000 would not fit in memory. The
//! quadratic size is in the answer's shape, not in the way it is computed.
//! [`LoopForest`] is the same answer without that shape: every block is stored
//! once, under the innermost loop it belongs to, and the loops form a forest
//! in which a loop's body is the blocks of its subtree.
//!
//! Natural loops are properly nested: two of them with different headers are
//! disjoint or one holds the other (a block in both has both headers among its
//! dominators, and the outer header reaches the inner latch through the shared
//! block). So the loops whose header dominates every latch, which is every
//! loop dominance finds, are discovered innermost first, each by a walk back
//! from its latches that jumps over a finished inner loop to the blocks that
//! enter it, with a union-find over loops making each jump a single step. The
//! cost is near-linear in the blocks and edges whatever the nesting.
//!
//! A tagged back edge whose header does not dominate its tail (an irreducible
//! cycle) can break proper nesting, so such a loop is kept as an explicit set
//! of blocks, as [`NaturalLoop`] has it, and every question about it is
//! answered from that set. The cost of such a loop is its own size, so a graph
//! with many of them still costs what [`detect_loops_tagged`] costs for them.
//!
//! [`detect_loops_tagged`]: crate::detect_loops_tagged

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use smallvec::SmallVec;

use super::dominator::{DominatorScratch, DominatorTree, complete_dominator_forest};
use super::edge_view::EdgeView;
use super::structure::{BackEdge, NaturalLoop, find_back_edges, find_back_edges_tagged};
use super::view::{DenseId, GraphView, RootedView};
use crate::block::BlockId;
use crate::edge::KindedEdge;

/// No loop, in a table of loop indices.
const NONE: u32 = u32::MAX;

/// A loop of one [`LoopForest`], by position.
///
/// Loops are numbered in ascending order of their header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoopId(u32);

impl LoopId {
    /// The position of the loop, in ascending header order.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// The loops of a graph, stored once per block.
///
/// See the [module documentation](self) for what the representation buys. A
/// loop is named by a [`LoopId`] and has a header, latches, a body, and a
/// depth: the number of loops it is nested in, where one loop is nested in
/// another when the other is the smallest loop with a strictly larger body
/// that contains its header. That is the rule of
/// [`LoopNestingTree`](crate::LoopNestingTree) and of
/// [`NaturalLoop::depth`].
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, DominatorTree, EdgeKind, LoopForest};
///
/// // entry -> outer; outer -> inner or exit; inner -> body, and body returns to
/// // inner; inner also returns to outer.
/// let mut cfg = Cfg::<u32>::new();
/// let outer = cfg.new_block();
/// let inner = cfg.new_block();
/// let body = cfg.new_block();
/// let exit = cfg.new_block();
/// cfg.add_edge(cfg.entry(), outer, EdgeKind::Fallthrough);
/// cfg.add_edge(outer, inner, EdgeKind::ConditionalTrue);
/// cfg.add_edge(outer, exit, EdgeKind::ConditionalFalse);
/// cfg.add_edge(inner, body, EdgeKind::ConditionalTrue);
/// cfg.add_edge(body, inner, EdgeKind::Back);
/// cfg.add_edge(inner, outer, EdgeKind::Back);
///
/// let dominators = DominatorTree::compute(&cfg);
/// let forest = LoopForest::compute_tagged(&cfg, &dominators);
///
/// let outer_loop = forest.loop_of_header(outer).expect("outer heads a loop");
/// let inner_loop = forest.loop_of_header(inner).expect("inner heads a loop");
/// assert_eq!(forest.parent(inner_loop), Some(outer_loop));
/// assert_eq!(forest.depth(inner_loop), 1);
/// assert!(forest.contains(outer_loop, body));
/// assert!(!forest.contains(outer_loop, exit));
/// assert_eq!(forest.body_len(outer_loop), 3);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopForest<N = BlockId> {
    /// Every loop, in ascending header order.
    loops: Vec<LoopRecord>,
    /// The headers of the loops, in the order of `loops`.
    headers: Vec<N>,
    /// The tails of the back edges into each header, ascending, one run per
    /// loop, in the order of `loops`.
    latches: Vec<N>,
    /// The blocks of every loop whose header does not dominate each of its
    /// latches, in the order of those loops.
    explicit: Vec<BTreeSet<N>>,
    /// The innermost properly nested loop each block belongs to, [`NONE`] for a
    /// block in none. Indexed by dense node index, and empty without loops.
    innermost: Vec<u32>,
    /// The blocks of the properly nested loops, grouped by the preorder
    /// position of their innermost loop, so that the body of a loop is one run.
    members: Vec<N>,
    /// Where the run of each preorder position starts in `members`, with one
    /// more entry for the end.
    member_starts: Vec<u32>,
    /// The node bound of the graph the loops were found in.
    node_bound: usize,
}

/// One loop of a [`LoopForest`], apart from what has a block type.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LoopRecord {
    /// Where the loop's latches start in [`LoopForest::latches`].
    latch_start: u32,
    /// Where they end.
    latch_end: u32,
    /// The index in [`LoopForest::explicit`] of the loop's blocks, [`NONE`] for
    /// a properly nested loop, whose body is the blocks of its subtree.
    explicit: u32,
    /// Which dominator tree the loop was found with. Building reads it; the
    /// finished forest does not.
    family: u32,
    /// The loop this one is nested in, [`NONE`] for an outermost loop.
    parent: u32,
    /// The number of loops this one is nested in.
    depth: usize,
    /// The properly nested loop directly around this one, [`NONE`] for an
    /// outermost one. This is the nesting of the bodies, which can differ from
    /// `parent` when an explicit loop lies between. Properly nested loops only.
    enclosing: u32,
    /// The preorder position of the loop among the properly nested loops.
    /// Properly nested loops only.
    first: u32,
    /// The greatest preorder position in its subtree. Properly nested loops
    /// only.
    last: u32,
    /// The greatest depth of this loop and of the properly nested loops around
    /// it. Properly nested loops only.
    deepest: usize,
}

impl LoopRecord {
    fn is_nested(&self) -> bool {
        self.explicit == NONE
    }
}

/// The back edges of one dominator tree, and the tree that classifies them.
struct Family<'a, N> {
    dominators: &'a DominatorTree<N>,
    back_edges: Vec<BackEdge<N>>,
}

impl<N: DenseId> LoopForest<N> {
    /// Find the loops dominance finds in `graph`: one per header that is the
    /// target of an edge from a block it dominates.
    ///
    /// The forest holds the loops of [`detect_loops`](crate::detect_loops).
    /// `dominators` must be the dominator tree of `graph`.
    #[must_use]
    pub fn compute<G>(graph: &G, dominators: &DominatorTree<N>) -> Self
    where
        G: GraphView<NodeId = N>,
    {
        let back_edges = find_back_edges(graph, dominators);
        build(
            graph,
            &[Family {
                dominators,
                back_edges,
            }],
        )
    }

    /// Find the loops of `graph`, honoring the [`EdgeKind::Back`] tags a
    /// builder put on edges that close a cycle dominance cannot prove.
    ///
    /// The forest holds the loops of
    /// [`detect_loops_tagged`](crate::detect_loops_tagged). `dominators` must
    /// be the dominator tree of `graph`.
    ///
    /// [`EdgeKind::Back`]: crate::EdgeKind::Back
    #[must_use]
    pub fn compute_tagged<G>(graph: &G, dominators: &DominatorTree<N>) -> Self
    where
        G: EdgeView<NodeId = N>,
        G::EdgeData: KindedEdge,
    {
        let back_edges = find_back_edges_tagged(graph, dominators);
        build(
            graph,
            &[Family {
                dominators,
                back_edges,
            }],
        )
    }

    /// Find the loops of all the code, tagged, including the code the root
    /// does not reach.
    ///
    /// The loops of [`compute_tagged`](Self::compute_tagged) are the live
    /// program's, and a consumer that also lays out the code the root misses (a
    /// handler pad no exception edge leads to, a filter funclet) wants the
    /// loops inside that code too: a loop there is as much a loop as one in live
    /// code. Those come from dominance within the dead code, which
    /// [`complete_dominator_forest`] provides by rooting a tree at each source
    /// component of it.
    ///
    /// The forest of dominators is a whole-graph answer, so the loops it gives
    /// for live headers are not the live program's: dead code that jumps into a
    /// live block takes the root away from it as dominator. Only the loops
    /// whose header the root misses are kept from it, and those are exact. A
    /// live block never has a dead successor, so a cycle of dead blocks, every
    /// tail of a back edge to a dead header, and every block that reaches such a
    /// tail without passing the header, are all dead, whatever the dead code
    /// feeds into. A dead loop is therefore nested only in dead loops, and a
    /// live loop only in live ones.
    pub(crate) fn compute_all_tagged<G>(graph: &G, dominators: &DominatorTree<N>) -> Self
    where
        G: RootedView<NodeId = N> + EdgeView,
        G::EdgeData: KindedEdge,
    {
        let live = Family {
            dominators,
            back_edges: find_back_edges_tagged(graph, dominators),
        };
        let mut roots = Vec::new();
        let Some(forest) =
            complete_dominator_forest(&mut DominatorScratch::new(), &mut roots, graph, dominators)
        else {
            return build(graph, &[live]);
        };
        let dead = Family {
            dominators: &forest,
            back_edges: find_back_edges_tagged(graph, &forest)
                .into_iter()
                .filter(|back_edge| !dominators.is_reachable(back_edge.header))
                .collect(),
        };
        build(graph, &[live, dead])
    }

    /// The number of loops.
    #[must_use]
    pub fn len(&self) -> usize {
        self.loops.len()
    }

    /// Whether there is no loop.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.loops.is_empty()
    }

    /// Every loop, in ascending header order.
    pub fn loops(&self) -> impl ExactSizeIterator<Item = LoopId> + use<N> {
        (0..self.loops.len()).map(|index| LoopId(word(index)))
    }

    /// The loop headed by `header`, if there is one.
    #[must_use]
    pub fn loop_of_header(&self, header: N) -> Option<LoopId> {
        self.headers
            .binary_search(&header)
            .ok()
            .map(|index| LoopId(word(index)))
    }

    /// The header of `lp`.
    ///
    /// # Panics
    ///
    /// Panics when `lp` is not a loop of this forest.
    #[must_use]
    pub fn header(&self, lp: LoopId) -> N {
        self.headers[lp.index()]
    }

    /// The blocks that jump back to the header of `lp`, in ascending order.
    ///
    /// # Panics
    ///
    /// Panics when `lp` is not a loop of this forest.
    #[must_use]
    pub fn latches(&self, lp: LoopId) -> &[N] {
        let record = self.record(lp);
        &self.latches[slot(record.latch_start)..slot(record.latch_end)]
    }

    /// The loop `lp` is nested in: the smallest loop with a strictly larger
    /// body that contains its header, and `None` for an outermost loop.
    ///
    /// # Panics
    ///
    /// Panics when `lp` is not a loop of this forest.
    #[must_use]
    pub fn parent(&self, lp: LoopId) -> Option<LoopId> {
        let parent = self.record(lp).parent;
        (parent != NONE).then_some(LoopId(parent))
    }

    /// The number of loops `lp` is nested in, `0` for an outermost loop.
    ///
    /// # Panics
    ///
    /// Panics when `lp` is not a loop of this forest.
    #[must_use]
    pub fn depth(&self, lp: LoopId) -> usize {
        self.record(lp).depth
    }

    /// The depth of the deepest loop counted from `1`, `0` when there is no
    /// loop: the greatest number of loops any block is in.
    #[must_use]
    pub fn max_nesting_depth(&self) -> usize {
        self.loops
            .iter()
            .map(|record| record.depth + 1)
            .max()
            .unwrap_or(0)
    }

    /// The number of blocks in `lp`, its header included.
    ///
    /// # Panics
    ///
    /// Panics when `lp` is not a loop of this forest.
    #[must_use]
    pub fn body_len(&self, lp: LoopId) -> usize {
        let record = self.record(lp);
        if record.is_nested() {
            self.run(record).len()
        } else {
            self.explicit[slot(record.explicit)].len()
        }
    }

    /// Whether `block` is in `lp`.
    ///
    /// This is constant time for a loop whose header dominates its latches,
    /// which is every loop dominance finds, and a set lookup for the others.
    /// A block beyond the node bound of the graph is in no loop.
    ///
    /// # Panics
    ///
    /// Panics when `lp` is not a loop of this forest.
    #[must_use]
    pub fn contains(&self, lp: LoopId, block: N) -> bool {
        let record = self.record(lp);
        if !record.is_nested() {
            return self.explicit[slot(record.explicit)].contains(&block);
        }
        let innermost = self.innermost.get(block.index()).copied().unwrap_or(NONE);
        innermost != NONE && {
            let inside = self.loops[slot(innermost)].first;
            record.first <= inside && inside <= record.last
        }
    }

    /// Whether every block of `inner` is a block of `outer`. A loop encloses
    /// itself.
    ///
    /// This is exact, which [`parent`](Self::parent) is not: the parent of a
    /// loop is the smallest larger loop that holds its *header*, and when that
    /// is an irreducible loop it can hold only part of the loop's blocks.
    ///
    /// The answer is constant time when both loops have a header that
    /// dominates its latches: such loops are properly nested, so one holds the
    /// other exactly when its run of the forest holds the other's. Otherwise
    /// the blocks of `inner` are looked up in `outer`, which costs at most the
    /// size of the smaller of the two, since a larger loop cannot be held by a
    /// smaller one and that is settled first.
    ///
    /// # Panics
    ///
    /// Panics when `outer` or `inner` is not a loop of this forest.
    #[must_use]
    pub fn encloses(&self, outer: LoopId, inner: LoopId) -> bool {
        let (outer_record, inner_record) = (self.record(outer), self.record(inner));
        if outer == inner {
            return true;
        }
        if outer_record.is_nested() && inner_record.is_nested() {
            return outer_record.first <= inner_record.first
                && inner_record.last <= outer_record.last;
        }
        self.body_len(inner) <= self.body_len(outer)
            && self.body(inner).all(|block| self.contains(outer, block))
    }

    /// The blocks of `lp`, header included, in no particular order.
    ///
    /// The walk costs the size of the body and allocates nothing.
    ///
    /// # Panics
    ///
    /// Panics when `lp` is not a loop of this forest.
    pub fn body(&self, lp: LoopId) -> impl Iterator<Item = N> + '_ {
        let record = self.record(lp);
        let (run, set) = if record.is_nested() {
            (self.run(record), None)
        } else {
            (&[][..], Some(&self.explicit[slot(record.explicit)]))
        };
        run.iter()
            .copied()
            .chain(set.into_iter().flatten().copied())
    }

    /// The depth of the deepest loop each block is in, counted from `1`, and
    /// `0` for a block in no loop: one entry for every node below the node
    /// bound the loops were found in.
    #[must_use]
    pub fn node_depths(&self) -> Vec<usize> {
        let mut depths = vec![0; self.node_bound];
        for (depth, &innermost) in depths.iter_mut().zip(&self.innermost) {
            if innermost != NONE {
                *depth = self.loops[slot(innermost)].deepest + 1;
            }
        }
        for record in self.loops.iter().filter(|record| !record.is_nested()) {
            for block in &self.explicit[slot(record.explicit)] {
                if let Some(depth) = depths.get_mut(block.index()) {
                    *depth = (*depth).max(record.depth + 1);
                }
            }
        }
        depths
    }

    /// The loops as [`NaturalLoop`]s, outermost first and by header within a
    /// depth, which is what [`detect_loops`](crate::detect_loops) and
    /// [`detect_loops_tagged`](crate::detect_loops_tagged) return.
    ///
    /// Every loop gets an owned set of its blocks, so the result is as large as
    /// the bodies of all the loops together, which is quadratic in the
    /// nesting depth of a deeply nested graph. A caller that only asks
    /// questions of the loops should ask the forest.
    #[must_use]
    pub fn natural_loops(&self) -> Vec<NaturalLoop<N>> {
        let mut loops: Vec<NaturalLoop<N>> = self
            .loops()
            .map(|lp| NaturalLoop {
                header: self.header(lp),
                body: self.body(lp).collect(),
                latches: self.latches(lp).iter().copied().collect(),
                depth: self.depth(lp),
            })
            .collect();
        loops.sort_by(|a, b| a.depth.cmp(&b.depth).then(a.header.cmp(&b.header)));
        loops
    }

    fn record(&self, lp: LoopId) -> &LoopRecord {
        &self.loops[lp.index()]
    }

    /// The blocks of a properly nested loop.
    fn run(&self, record: &LoopRecord) -> &[N] {
        let start = slot(self.member_starts[slot(record.first)]);
        let end = slot(self.member_starts[slot(record.last) + 1]);
        &self.members[start..end]
    }

    /// Whether `lp` is kept as an explicit set of blocks.
    #[cfg(test)]
    fn is_explicit(&self, lp: LoopId) -> bool {
        !self.record(lp).is_nested()
    }
}

/// The word that stores a dense index.
fn word(index: usize) -> u32 {
    u32::try_from(index).expect("a loop or node index fits in u32")
}

/// The dense index a stored word names.
const fn slot(stored: u32) -> usize {
    stored as usize
}

/// The outermost loop found so far that holds loop `x`, with every loop on the
/// way pointed straight at it.
fn find(representative: &mut [u32], mut x: u32) -> u32 {
    while representative[slot(x)] != x {
        let up = representative[slot(representative[slot(x)])];
        representative[slot(x)] = up;
        x = up;
    }
    x
}

/// Find the body of a loop whose header does not dominate every latch.
///
/// The body is the set of reachable nodes that can reach any latch without
/// going through `header`, plus `header` itself. A tagged latch the header
/// does not dominate (an irreducible cycle) lets that walk run on to blocks
/// that merely lead to the cycle, such as the entry, so the body is cut down
/// to the walked blocks the header reaches: the blocks on a path from the
/// header to a latch.
fn explicit_body<G: GraphView>(
    graph: &G,
    dominators: &DominatorTree<G::NodeId>,
    header: G::NodeId,
    latches: &[G::NodeId],
) -> BTreeSet<G::NodeId> {
    let mut walked = BTreeSet::new();
    walked.insert(header);
    let mut stack = Vec::new();
    for &latch in latches {
        if walked.insert(latch) {
            stack.push(latch);
        }
    }
    while let Some(node) = stack.pop() {
        for before in graph
            .predecessors(node)
            .filter(|&before| dominators.is_reachable(before))
        {
            if walked.insert(before) {
                stack.push(before);
            }
        }
    }

    let mut cycle = BTreeSet::from([header]);
    cycle.extend(latches.iter().copied());
    stack.push(header);
    while let Some(node) = stack.pop() {
        for next in graph.successors(node) {
            if walked.contains(&next) && cycle.insert(next) {
                stack.push(next);
            }
        }
    }
    cycle
}

/// A back edge as the build sorts it: header, tail, and the index of the family
/// that found it.
type Row<N> = (N, N, usize);

/// Place the loops of `families` in a forest.
fn build<G: GraphView>(graph: &G, families: &[Family<'_, G::NodeId>]) -> LoopForest<G::NodeId> {
    let mut forest = LoopForest::find(graph, families);
    if forest.loops.is_empty() {
        return forest;
    }
    let order = forest.innermost_first(families);
    forest.claim_blocks(graph, families, &order);
    forest.number_subtrees(&order);
    forest.group_members();
    forest.nest(&order);
    forest
}

impl<N: DenseId> LoopForest<N> {
    /// The loops of `families`, one per header, with their latches and the
    /// blocks of those that are not properly nested; nothing is placed yet.
    fn find<G: GraphView<NodeId = N>>(graph: &G, families: &[Family<'_, N>]) -> Self {
        let mut edges: SmallVec<[Row<N>; 8]> = families
            .iter()
            .enumerate()
            .flat_map(|(family, source)| {
                source
                    .back_edges
                    .iter()
                    .map(move |edge| (edge.header, edge.tail, family))
            })
            .collect();
        edges.sort_unstable();

        let mut forest = Self {
            loops: Vec::new(),
            headers: Vec::new(),
            latches: Vec::with_capacity(edges.len()),
            explicit: Vec::new(),
            innermost: Vec::new(),
            members: Vec::new(),
            member_starts: Vec::new(),
            node_bound: graph.node_bound(),
        };
        for run in edges.chunk_by(|a, b| a.0 == b.0) {
            let (header, _, family) = run[0];
            debug_assert!(
                run.iter().all(|edge| edge.2 == family),
                "one header belongs to one family of back edges"
            );
            let start = forest.latches.len();
            forest.latches.extend(run.iter().map(|edge| edge.1));
            let dominators = families[family].dominators;
            let blocks = &forest.latches[start..];
            let explicit = if blocks
                .iter()
                .all(|&latch| dominators.dominates(header, latch))
            {
                NONE
            } else {
                forest
                    .explicit
                    .push(explicit_body(graph, dominators, header, blocks));
                word(forest.explicit.len() - 1)
            };
            forest.headers.push(header);
            forest.loops.push(LoopRecord {
                latch_start: word(start),
                latch_end: word(forest.latches.len()),
                explicit,
                family: word(family),
                parent: NONE,
                depth: 0,
                enclosing: NONE,
                first: NONE,
                last: NONE,
                deepest: 0,
            });
        }
        forest
    }

    /// The properly nested loops, innermost first: an inner header is
    /// dominated by the outer one, so a deeper header comes first.
    fn innermost_first(&self, families: &[Family<'_, N>]) -> SmallVec<[usize; 8]> {
        let mut order: SmallVec<[usize; 8]> = (0..self.loops.len())
            .filter(|&k| self.loops[k].is_nested())
            .collect();
        order.sort_unstable_by_key(|&k| {
            let family = slot(self.loops[k].family);
            let level = families[family].dominators.level(self.headers[k]);
            (family, Reverse(level.unwrap_or(0)), self.headers[k])
        });
        order
    }

    /// Walk each properly nested loop's body, innermost loops first, giving
    /// every block to the innermost loop it is in and every loop to the loop
    /// directly around it.
    fn claim_blocks<G: GraphView<NodeId = N>>(
        &mut self,
        graph: &G,
        families: &[Family<'_, N>],
        order: &[usize],
    ) {
        let mut innermost = vec![NONE; self.node_bound];
        let mut representative: SmallVec<[u32; 8]> = (0..self.loops.len()).map(word).collect();
        let mut stack: SmallVec<[usize; 16]> = SmallVec::new();
        for &k in order {
            let this = word(k);
            let header = self.headers[k];
            let record = &self.loops[k];
            let dominators = families[slot(record.family)].dominators;
            innermost[header.index()] = this;
            stack.clear();
            stack.extend(
                self.latches[slot(record.latch_start)..slot(record.latch_end)]
                    .iter()
                    .map(|latch| latch.index()),
            );
            while let Some(node) = stack.pop() {
                if node == header.index() {
                    continue;
                }
                let entering = match innermost[node] {
                    NONE => {
                        // First time here: the block is this loop's own, and
                        // what reaches it is in the loop too.
                        innermost[node] = this;
                        N::from_index(node)
                    }
                    held => {
                        // A loop already found holds the block, and so is
                        // inside this one. Jump over it: the blocks that reach
                        // it do so through its header, and the others are
                        // already in it.
                        let inner = find(&mut representative, held);
                        if inner == this {
                            continue;
                        }
                        self.loops[slot(inner)].enclosing = this;
                        representative[slot(inner)] = this;
                        self.headers[slot(inner)]
                    }
                };
                for before in graph.predecessors(entering) {
                    if dominators.is_reachable(before) && innermost[before.index()] != this {
                        stack.push(before.index());
                    }
                }
            }
        }
        self.innermost = innermost;
    }

    /// A preorder numbering of the properly nested loops that gives each loop's
    /// subtree a run of positions. Children come before parents in `order`, so
    /// the sizes add up in it and the positions are handed out in its reverse.
    fn number_subtrees(&mut self, order: &[usize]) {
        let mut subtree: SmallVec<[u32; 8]> = SmallVec::from_elem(1, self.loops.len());
        for &k in order {
            let up = self.loops[k].enclosing;
            if up != NONE {
                subtree[slot(up)] += subtree[k];
            }
        }
        let mut next: SmallVec<[u32; 8]> = SmallVec::from_elem(0, self.loops.len());
        let mut next_outermost = 0_u32;
        for &k in order.iter().rev() {
            let up = self.loops[k].enclosing;
            let cursor = if up == NONE {
                &mut next_outermost
            } else {
                &mut next[slot(up)]
            };
            let first = *cursor;
            *cursor += subtree[k];
            next[k] = first + 1;
            self.loops[k].first = first;
            self.loops[k].last = first + subtree[k] - 1;
        }
        self.member_starts = vec![0; slot(next_outermost) + 1];
    }

    /// The blocks of each position, as one run apiece, in ascending order.
    /// `member_starts` holds the number of positions plus one zeros.
    fn group_members(&mut self) {
        let positions = self.member_starts.len() - 1;
        for &held in self.innermost.iter().filter(|&&held| held != NONE) {
            self.member_starts[slot(self.loops[slot(held)].first)] += 1;
        }
        let mut total = 0;
        for count in &mut self.member_starts[..positions] {
            total += *count;
            *count = total;
        }
        self.member_starts[positions] = total;
        self.members = vec![N::from_index(0); slot(total)];
        for (node, &held) in self.innermost.iter().enumerate().rev() {
            if held != NONE {
                let end = &mut self.member_starts[slot(self.loops[slot(held)].first)];
                *end -= 1;
                self.members[slot(*end)] = N::from_index(node);
            }
        }
    }
}

impl<N: DenseId> LoopForest<N> {
    /// Work out the nesting rule's parent and depth of every loop, and the
    /// greatest depth along each properly nested loop's chain of enclosing
    /// loops. `order` is the properly nested loops, innermost first.
    ///
    /// The parent of a loop is the smallest loop with a strictly larger body
    /// that contains its header, the one with the lowest header among equals.
    /// Among the properly nested loops alone, that is the loop directly around
    /// it, so a forest of them needs no search.
    fn nest(&mut self, order: &[usize]) {
        if self.explicit.is_empty() {
            for record in &mut self.loops {
                record.parent = record.enclosing;
            }
            self.assign_depths(order.iter().rev().copied());
        } else {
            let (parents, outermost_first) = self.parents_among_explicit_loops();
            for (record, parent) in self.loops.iter_mut().zip(parents) {
                record.parent = parent;
            }
            self.assign_depths(outermost_first.into_iter());
        }

        for &k in order.iter().rev() {
            let around = self.loops[k].enclosing;
            let outer = if around == NONE {
                0
            } else {
                self.loops[slot(around)].deepest
            };
            self.loops[k].deepest = self.loops[k].depth.max(outer);
        }
    }

    /// Give every loop the depth below its parent, handing out loops
    /// `outermost_first` so that a parent has its depth before its children.
    fn assign_depths(&mut self, outermost_first: impl Iterator<Item = usize>) {
        for k in outermost_first {
            self.loops[k].depth = match self.loops[k].parent {
                NONE => 0,
                parent => self.loops[slot(parent)].depth + 1,
            };
        }
    }

    /// The parent of every loop, found by the rule when explicit loops may be
    /// the smallest enclosing one, and the loops largest first.
    ///
    /// The properly nested loops that contain a header are a chain, nearest
    /// first, and their bodies grow along it, so the first one larger than the
    /// loop is the best of them. An explicit loop is found by asking each one.
    fn parents_among_explicit_loops(&self) -> (Vec<u32>, Vec<usize>) {
        let sizes: Vec<usize> = self.loops().map(|lp| self.body_len(lp)).collect();
        let explicit: Vec<usize> = (0..self.loops.len())
            .filter(|&k| !self.loops[k].is_nested())
            .collect();

        let parents = (0..self.loops.len())
            .map(|k| {
                let header = self.headers[k];
                let mut best: Option<(usize, N, usize)> = None;
                let mut consider = |candidate: usize| {
                    let key = (sizes[candidate], self.headers[candidate]);
                    if best.is_none_or(|(size, header, _)| key < (size, header)) {
                        best = Some((key.0, key.1, candidate));
                    }
                };
                let mut chain = self.innermost[header.index()];
                while chain != NONE {
                    let candidate = slot(chain);
                    if sizes[candidate] > sizes[k] {
                        consider(candidate);
                        break;
                    }
                    chain = self.loops[candidate].enclosing;
                }
                for &candidate in &explicit {
                    if candidate != k
                        && sizes[candidate] > sizes[k]
                        && self.contains(LoopId(word(candidate)), header)
                    {
                        consider(candidate);
                    }
                }
                best.map_or(NONE, |(_, _, candidate)| word(candidate))
            })
            .collect();

        // A parent's body is strictly larger than its child's.
        let mut largest_first: Vec<usize> = (0..self.loops.len()).collect();
        largest_first.sort_unstable_by_key(|&k| Reverse(sizes[k]));
        (parents, largest_first)
    }
}

#[cfg(test)]
mod tests;
