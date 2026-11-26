//! Single-entry single-exit regions and the program structure tree.
//!
//! A single-entry single-exit (SESE) region is an ordered pair of edges `(a, b)`
//! such that `a` dominates `b`, `b` post-dominates `a`, and every cycle
//! through one of them passes through the other. Its blocks are those control
//! reaches over `a` before it crosses `b`: they are entered only over `a` and
//! left only over `b`. The canonical regions are the smallest ones, those
//! whose two edges are consecutive in dominance order. They never partly
//! overlap, so they nest into a tree, the program structure tree of Johnson,
//! Pearson and Pingali ("The Program Structure Tree: Computing Control Regions
//! in Linear Time", PLDI 1994). A pass can then work one region at a time,
//! innermost first.
//!
//! These regions are a fact about the shape of the graph. The exception
//! regions of [`region`](crate::region) are something else: protected blocks
//! and their handlers, which the consumer declares.
//!
//! # The closed graph
//!
//! The analysis runs on the blocks the root reaches, closed into one strongly
//! connected graph:
//!
//! - every exit block, a reachable block without successors, has a virtual
//!   edge to a virtual exit;
//! - so does every block the [`PostDominatorTree`](crate::PostDominatorTree)
//!   cut joins to its virtual exit: in a region no path leaves, an infinite
//!   loop, the blocks that close each loop;
//! - one virtual edge leads from the virtual exit into the root.
//!
//! A region boundary is therefore a [`SeseEdge`]: an edge of the graph, the
//! virtual edge into the root ([`SeseEdge::Entry`]), or the virtual edge out of
//! an exit block or a cut root ([`SeseEdge::Exit`]). A loop whose header is the
//! root is entered over [`SeseEdge::Entry`], and a region that ends in a
//! `return` is left over [`SeseEdge::Exit`]. Blocks the root does not reach
//! are in no region.
//!
//! # Algorithm
//!
//! [`CycleEquivalence`] partitions the edges by one undirected depth-first
//! search with bracket lists. The edges of one class are totally ordered by
//! dominance, and each two consecutive ones bound a canonical region; a
//! directed depth-first search meets them in that order, so the same search
//! pairs them up and places every block in its innermost region. Both
//! searches run on explicit stacks, so a nest 10⁵ deep needs no call stack,
//! and the whole computation is O(n + m) in time and space for n blocks and m
//! edges, on irreducible graphs too. Parallel edges stay distinct edges.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::block::BlockId;
use crate::edge::EdgeId;
use crate::graph::edge_view::EdgeView;
use crate::graph::fanout::DenseFanout;
use crate::graph::view::{DenseId, RootedView};
use crate::identity::{NO_INDEX, define_dense_id, slot, word};
use closed::{ClosedGraph, slot_edge};
pub use cycle_equivalence::CycleEquivalence;
use cycle_equivalence::cycle_classes;

mod closed;
mod cycle_equivalence;
#[cfg(test)]
mod tests;

/// An edge of the closed graph a region boundary can be.
///
/// See the [module documentation](self) for the virtual edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SeseEdge<N = BlockId, E = EdgeId> {
    /// The virtual edge from the virtual exit into the root, through which
    /// every path from the root begins.
    Entry,
    /// An edge of the graph.
    Edge(E),
    /// The virtual edge from a block to the virtual exit: from an exit block,
    /// or from a block that closes a loop no path leaves.
    Exit(N),
}

define_dense_id! {
    /// A region of one [`ProgramStructureTree`], by position.
    ///
    /// Regions are numbered in preorder, so a region's number is below the
    /// numbers of the regions inside it.
    pub struct SeseRegionId(u32);
}

/// The canonical single-entry single-exit regions of a rooted graph, nested
/// into a tree.
///
/// See the [module documentation](self) for the definitions. Each region has
/// an [`entry_edge`](Self::entry_edge), an [`exit_edge`](Self::exit_edge) and
/// a [`parent`](Self::parent), the smallest region around it; the outermost
/// regions have none. Every block the root reaches is in at most one
/// [`innermost`](Self::innermost) region, and a block in no region belongs to
/// the procedure as a whole.
///
/// Every block is stored once, under its innermost region, and the regions
/// are numbered in preorder, so the [`blocks`](Self::blocks) of a region are
/// one slice and [`contains`](Self::contains) is an O(1) interval test, as in
/// [`LoopForest`](crate::LoopForest). The set of regions depends only on the
/// graph; their numbering, the order of siblings, follows adjacency order.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, ProgramStructureTree, SeseEdge};
///
/// // entry -> branch; branch -> then | else -> merge; merge -> exit.
/// let mut cfg = Cfg::<u32>::new();
/// let branch = cfg.new_block();
/// let then_block = cfg.new_block();
/// let else_block = cfg.new_block();
/// let merge = cfg.new_block();
/// let into_branch = cfg.add_edge(cfg.entry(), branch, EdgeKind::Fallthrough);
/// let into_then = cfg.add_edge(branch, then_block, EdgeKind::ConditionalTrue);
/// cfg.add_edge(branch, else_block, EdgeKind::ConditionalFalse);
/// let out_of_then = cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);
/// cfg.add_edge(else_block, merge, EdgeKind::Fallthrough);
///
/// let tree = ProgramStructureTree::compute(&cfg);
///
/// // The `then` arm is a region of its own, inside the diamond.
/// let arm = tree.innermost(then_block).expect("the arm is a region");
/// assert_eq!(tree.entry_edge(arm), SeseEdge::Edge(into_then));
/// assert_eq!(tree.exit_edge(arm), SeseEdge::Edge(out_of_then));
/// assert_eq!(tree.blocks(arm), [then_block]);
///
/// // The diamond runs from the edge into the branch to the return.
/// let diamond = tree.parent(arm).expect("the arm is nested");
/// assert_eq!(tree.entry_edge(diamond), SeseEdge::Edge(into_branch));
/// assert_eq!(tree.exit_edge(diamond), SeseEdge::Exit(merge));
/// assert!(tree.contains(diamond, then_block));
/// assert!(!tree.contains(diamond, cfg.entry()));
/// assert_eq!(tree.parent(diamond), None);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramStructureTree<N = BlockId, E = EdgeId> {
    /// Every region, in preorder.
    regions: Vec<RegionRecord>,
    /// The innermost region of each block by dense index, [`NO_INDEX`] for a
    /// block in none.
    innermost: Vec<u32>,
    /// The blocks of every region, grouped by the preorder position of their
    /// innermost region, in ascending dense index within a group.
    members: Vec<N>,
    /// Where the group of each preorder position starts in `members`, with one
    /// more entry for the end.
    member_starts: Vec<u32>,
    /// The edge bound of the graph, which is also the slot of the virtual edge
    /// into the root.
    edge_bound: usize,
    /// The blocks with a virtual edge to the virtual exit, ascending.
    outlets: Vec<N>,
    /// The edge identity of the boundaries.
    edge: PhantomData<E>,
}

/// One region of a [`ProgramStructureTree`], apart from what has a block type.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RegionRecord {
    /// The slot of the entry edge in the closed graph.
    entry: u32,
    /// The slot of the exit edge.
    exit: u32,
    /// The region directly around this one, [`NO_INDEX`] for an outermost one.
    parent: u32,
    /// The greatest preorder position in its subtree.
    last: u32,
}

impl<N: DenseId, E: DenseId> ProgramStructureTree<N, E> {
    /// Compute the canonical regions of the blocks the root of `graph`
    /// reaches.
    ///
    /// # Panics
    ///
    /// Panics when the root is not a node of `graph`, or when the nodes or
    /// the edges of the closed graph exceed the range of a dense `u32` table.
    #[must_use]
    pub fn compute<G>(graph: &G) -> Self
    where
        G: RootedView<NodeId = N> + EdgeView<EdgeId = E>,
    {
        let closed = ClosedGraph::new(graph);
        let (classes, class_count) = cycle_classes(&closed);
        let found = Found::search(&closed, &classes, class_count);
        Self::number(closed, &found)
    }

    /// The number of regions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Whether there is no region.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Every region, in preorder: each region before the regions inside it.
    pub fn regions(&self) -> impl ExactSizeIterator<Item = SeseRegionId> + use<N, E> {
        (0..self.regions.len()).map(|index| SeseRegionId(word(index)))
    }

    /// The edge over which control enters `region`.
    ///
    /// # Panics
    ///
    /// Panics when `region` is not a region of this tree.
    #[must_use]
    pub fn entry_edge(&self, region: SeseRegionId) -> SeseEdge<N, E> {
        self.boundary(self.record(region).entry)
    }

    /// The edge over which control leaves `region`.
    ///
    /// # Panics
    ///
    /// Panics when `region` is not a region of this tree.
    #[must_use]
    pub fn exit_edge(&self, region: SeseRegionId) -> SeseEdge<N, E> {
        self.boundary(self.record(region).exit)
    }

    /// The smallest region around `region`, `None` for an outermost one.
    ///
    /// # Panics
    ///
    /// Panics when `region` is not a region of this tree.
    #[must_use]
    pub fn parent(&self, region: SeseRegionId) -> Option<SeseRegionId> {
        let parent = self.record(region).parent;
        (parent != NO_INDEX).then_some(SeseRegionId(parent))
    }

    /// The smallest region `block` is in, `None` for a block in no region, a
    /// block the root does not reach, or one beyond the node bound of the
    /// graph.
    #[must_use]
    pub fn innermost(&self, block: N) -> Option<SeseRegionId> {
        match self.innermost.get(block.index()).copied() {
            None | Some(NO_INDEX) => None,
            Some(region) => Some(SeseRegionId(region)),
        }
    }

    /// Whether `block` is in `region`, in O(1).
    ///
    /// # Panics
    ///
    /// Panics when `region` is not a region of this tree.
    #[must_use]
    pub fn contains(&self, region: SeseRegionId, block: N) -> bool {
        let last = self.record(region).last;
        self.innermost(block)
            .is_some_and(|inner| region.0 <= inner.0 && inner.0 <= last)
    }

    /// The blocks of `region`, those of the regions inside it included: first
    /// the blocks whose innermost region it is, then those of each region
    /// inside it in preorder, each group in ascending dense index.
    ///
    /// # Panics
    ///
    /// Panics when `region` is not a region of this tree.
    #[must_use]
    pub fn blocks(&self, region: SeseRegionId) -> &[N] {
        let record = self.record(region);
        let start = slot(self.member_starts[region.index()]);
        let end = slot(self.member_starts[slot(record.last) + 1]);
        &self.members[start..end]
    }

    /// The record of `region`.
    fn record(&self, region: SeseRegionId) -> &RegionRecord {
        &self.regions[region.index()]
    }

    /// The edge the slot `edge` names.
    fn boundary(&self, edge: u32) -> SeseEdge<N, E> {
        slot_edge(self.edge_bound, &self.outlets, slot(edge))
    }

    /// Number the regions `found` holds in preorder, and lay their blocks out
    /// by it.
    fn number(closed: ClosedGraph<N>, found: &Found) -> Self {
        let count = found.entry.len();
        // The children of each region in the order the search entered them,
        // and the outermost regions as the children of `count`.
        let group = |parent: u32| {
            if parent == NO_INDEX {
                count
            } else {
                slot(parent)
            }
        };
        let mut child_starts = Vec::new();
        let mut children = Vec::new();
        DenseFanout::group_into(
            count + 1,
            || {
                found
                    .parent
                    .iter()
                    .enumerate()
                    .map(|(region, &parent)| (group(parent), word(region)))
            },
            &mut child_starts,
            &mut children,
        );
        let children_of =
            |group: usize| &children[slot(child_starts[group])..slot(child_starts[group + 1])];

        let mut number = vec![NO_INDEX; count];
        let mut preorder = Vec::with_capacity(count);
        let mut stack: Vec<u32> = children_of(count).iter().rev().copied().collect();
        while let Some(region) = stack.pop() {
            number[slot(region)] = word(preorder.len());
            preorder.push(region);
            stack.extend(children_of(slot(region)).iter().rev());
        }
        let mut last = number.clone();
        for &region in preorder.iter().rev() {
            let parent = found.parent[slot(region)];
            if parent != NO_INDEX {
                last[slot(parent)] = last[slot(parent)].max(last[slot(region)]);
            }
        }
        let renumber = |region: u32| {
            if region == NO_INDEX {
                NO_INDEX
            } else {
                number[slot(region)]
            }
        };

        let regions = preorder
            .iter()
            .map(|&region| {
                let region = slot(region);
                RegionRecord {
                    entry: found.entry[region],
                    exit: found.exit[region],
                    parent: renumber(found.parent[region]),
                    last: last[region],
                }
            })
            .collect();
        let innermost: Vec<u32> = found.innermost[..closed.bound]
            .iter()
            .map(|&region| renumber(region))
            .collect();
        let mut member_starts = Vec::new();
        let mut members = Vec::new();
        DenseFanout::group_into(
            count,
            || {
                innermost
                    .iter()
                    .enumerate()
                    .filter(|&(_, &region)| region != NO_INDEX)
                    .map(|(block, &region)| (slot(region), N::from_index(block)))
            },
            &mut member_starts,
            &mut members,
        );

        Self {
            regions,
            innermost,
            members,
            member_starts,
            edge_bound: closed.edge_bound,
            outlets: closed.outlets,
            edge: PhantomData,
        }
    }
}

/// The canonical regions in the order a directed search enters them, and the
/// innermost region of every node.
struct Found {
    /// The slot of each region's entry edge.
    entry: Vec<u32>,
    /// The slot of each region's exit edge.
    exit: Vec<u32>,
    /// The region directly around each one, [`NO_INDEX`] for an outermost one.
    parent: Vec<u32>,
    /// The innermost region of each node, [`NO_INDEX`] for a node in none.
    innermost: Vec<u32>,
}

impl Found {
    /// Pair up the edges of each class and place every node, in one directed
    /// depth-first search from the virtual exit.
    ///
    /// The search meets the edges of a class in dominance order: a path from
    /// the virtual exit to an edge passes through every edge that dominates
    /// it, and the search's stack is such a path. So the edge that ends one
    /// region is the next edge of its class the search meets after the one
    /// that began it. An entry edge dominates the region's first block, so it
    /// is the tree edge that discovers it; the regions open and close along
    /// the search's stack, and a block discovered over an edge is in the
    /// region current after that edge.
    fn search<N: DenseId>(closed: &ClosedGraph<N>, classes: &[u32], class_count: usize) -> Self {
        let mut class_size = vec![0_u32; class_count];
        for &class in classes.iter().filter(|&&class| class != NO_INDEX) {
            class_size[slot(class)] += 1;
        }
        let regions = class_size
            .iter()
            .map(|&size| slot(size).saturating_sub(1))
            .sum();
        let nodes = closed.node_count();
        let mut found = Self {
            entry: Vec::with_capacity(regions),
            exit: Vec::with_capacity(regions),
            parent: Vec::with_capacity(regions),
            innermost: vec![NO_INDEX; nodes],
        };
        let mut seen = vec![0_u32; class_count];
        let mut open = vec![NO_INDEX; class_count];
        let mut visited = vec![false; nodes];
        let root = closed.bound;
        visited[root] = true;
        let mut stack = vec![(root, 0_usize)];
        while let Some(frame) = stack.last_mut() {
            let (node, position) = *frame;
            let Some(&edge) = closed.outgoing(node).get(position) else {
                stack.pop();
                continue;
            };
            frame.1 += 1;
            let edge = slot(edge);
            let class = slot(classes[edge]);
            seen[class] += 1;
            let rank = seen[class];
            let to = closed.target(edge);
            let discovers = !visited[to];
            let mut region = found.innermost[node];
            if rank > 1 {
                let closing = open[class];
                found.exit[slot(closing)] = word(edge);
                if discovers {
                    debug_assert_eq!(region, closing, "an exit edge leaves the current region");
                    region = found.parent[slot(closing)];
                }
            }
            if rank < class_size[class] {
                debug_assert!(
                    discovers,
                    "an entry edge discovers its region's first block"
                );
                let opened = word(found.entry.len());
                found.entry.push(word(edge));
                found.exit.push(NO_INDEX);
                found.parent.push(region);
                open[class] = opened;
                region = opened;
            }
            if discovers {
                visited[to] = true;
                found.innermost[to] = region;
                stack.push((to, 0));
            }
        }
        found
    }
}
