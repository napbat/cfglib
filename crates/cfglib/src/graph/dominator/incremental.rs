//! Dominator trees brought up to date after edge edits.
//!
//! [`DominatorTree::apply_updates_in`] takes the tree of a graph before a
//! batch of edge insertions and deletions and the graph after them, and
//! replays the batch one edit at a time, following Georgiadis, Italiano,
//! Laura and Santaroni, "An Experimental Study of Dynamic Dominators" (ESA
//! 2012, full version arXiv 1604.02711), section 2 and the depth-based search
//! of section 3.2. For an edge `(x, y)` from a reachable `x`:
//!
//! - **Insertion, `y` reachable.** With `c` the nearest common dominator of
//!   `x` and `y`, nothing changes when `c` is `y` or its immediate dominator.
//!   Otherwise exactly the nodes `v` deeper than `c`'s children that `y`
//!   reaches along nodes deeper than `v`'s immediate dominator move up to
//!   `c` (Lemma 2.5). A search from `y` that takes the deepest candidate
//!   first finds them, each node scanned once.
//! - **Insertion, `y` unreachable.** `y` and everything it newly reaches hang
//!   below the nearest common dominator `r` of `x` and of every node those
//!   new nodes lead back into, and nothing outside the subtree of `r` moves.
//!   The subtree of `r` with the new nodes is recomputed.
//! - **Deletion.** Nothing changes when `y` dominates `x`, or when another
//!   copy of the edge is left. When `y` keeps an edge from a node it does not
//!   dominate, every node that moves is in the subtree of `y`'s immediate
//!   dominator (Lemma 2.6), which is recomputed. When it keeps none, `y` and
//!   its subtree become unreachable, and the subtree recomputed is the one of
//!   the shallowest immediate dominator of `y` or of a node an edge out of
//!   `y`'s subtree enters (Lemma 2.7).
//!
//! A recomputed subtree is the simple Lengauer-Tarjan run over the subgraph
//! that subtree induces, from its root: every path from the root to a node it
//! dominates stays among the nodes it dominates, so that subgraph has the
//! same dominators. Each edit that changes the tree renumbers it, so the next
//! edit reads levels and O(1) dominance from a numbered tree.
//!
//! The graph a caller passes already holds the whole batch, so each edit is
//! replayed on a view of the graph with the later edits of the batch undone.
//! Dominance does not depend on how many parallel edges join two nodes, so
//! the view only has to get the set of edges right.

extern crate alloc;
use alloc::vec::Vec;
use core::mem;

use super::numbering::{Numbered, NumberingBuffers};
use super::{DominatorScratch, DominatorTree};
use crate::graph::epoch::grow_to;
use crate::graph::view::{DenseId, GraphView, Induced, RootedView};
use crate::identity::{NO_INDEX, check_bound, slot, word};

/// When a batch stops being replayed and is finished with one full
/// construction, because that has become the cheaper way.
#[derive(Debug, Clone, Copy)]
pub(super) struct Gates {
    /// The most edits a batch may hold. Replaying an edit reads the later
    /// edits of the batch at every node they name.
    pub(super) edits: usize,
    /// How many nodes of the tree buy one pass of work: an edit that changes
    /// the tree renumbers it, which is one pass, and a recomputed subtree
    /// adds its share of the tree. Once a batch would spend more and edits
    /// remain, it is finished from scratch. 0 sets no limit.
    pub(super) pass_nodes: usize,
    /// A subtree to recompute that holds more than this share of the tree,
    /// as a divisor of its size, is left to the full construction; 0 lets
    /// every subtree through.
    pub(super) region_divisor: usize,
}

/// The gates [`DominatorTree::apply_updates_in`] uses, from the benchmark
/// rows `graph_dominators_*` (2¹² nodes) and `sparse_dominators_*` (2²⁰).
///
/// Renumbering costs about a third of a construction at 2¹² nodes and about
/// a ninetieth at 2²⁰, where the construction's random accesses miss the
/// cache. So a tree may spend one pass per 2¹⁴ nodes: a small tree finishes
/// a batch from scratch as soon as one edit changes it, and a tree of 2²⁰
/// nodes may replay 64. A single edit is always replayed. A deletion
/// recomputes a subtree of the node it cuts off, which on the sparse graph
/// is most of the tree; past a quarter of it a construction is cheaper.
pub(super) const GATES: Gates = Gates {
    edits: 64,
    pass_nodes: 1 << 14,
    region_divisor: 4,
};

/// What replaying one edit did.
enum Step {
    /// The tree is as it was.
    Kept,
    /// The tree changed; the work done, in nodes visited.
    Changed(usize),
    /// The edit is better left to a full construction.
    Abandoned,
}

/// One edit already applied to the graph a dominator tree is being updated
/// for. See [`DominatorTree::apply_updates_in`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DominatorUpdate<N> {
    /// The edge `from -> to` was added, beside any copies already there.
    Insert {
        /// The tail of the edge.
        from: N,
        /// The head of the edge.
        to: N,
    },
    /// One copy of the edge `from -> to` was removed.
    Delete {
        /// The tail of the edge.
        from: N,
        /// The head of the edge.
        to: N,
    },
}

/// The buffers an update pass fills, owned by the caller.
#[derive(Debug, Clone, Default)]
pub(super) struct UpdateBuffers {
    /// Whether each node's parent was set during the batch, and its parent
    /// before the batch.
    touched: Vec<bool>,
    original: Vec<u32>,
    /// The nodes `touched` marks.
    touched_list: Vec<u32>,
    /// Whether each node was found affected by an insertion, and whether it
    /// was scanned, cleared after each insertion through `marked`.
    affected: Vec<bool>,
    scanned: Vec<bool>,
    marked: Vec<u32>,
    /// The affected nodes still to scan, by level, as linked lists.
    bucket_head: Vec<u32>,
    bucket_next: Vec<u32>,
    /// Nodes a search has yet to expand.
    stack: Vec<u32>,
    /// Whether each node is in the region being recomputed, and its nodes.
    member: Vec<bool>,
    region: Vec<u32>,
    /// The parent each node of the region gets from the recomputation.
    fresh: Vec<u32>,
}

impl UpdateBuffers {
    /// Give every table room for a graph of `bound` nodes.
    ///
    /// Every pass leaves the flags cleared, the buckets empty and the fresh
    /// parents [`NO_INDEX`], so a table only has to grow, and an edit that
    /// changes nothing costs nothing here.
    fn start(&mut self, bound: usize) {
        for table in [
            &mut self.touched,
            &mut self.affected,
            &mut self.scanned,
            &mut self.member,
        ] {
            grow_to(table, bound, false);
        }
        for table in [
            &mut self.original,
            &mut self.bucket_head,
            &mut self.bucket_next,
            &mut self.fresh,
        ] {
            grow_to(table, bound, NO_INDEX);
        }
        self.touched_list.clear();
        self.marked.clear();
    }

    /// Clear the marks of every node whose parent the pass set.
    fn untouch(&mut self) {
        for &node in &self.touched_list {
            self.touched[slot(node)] = false;
        }
    }
}

/// What one batch works with besides the tree.
struct Pass<'a> {
    numbering: &'a mut NumberingBuffers,
    buffers: &'a mut UpdateBuffers,
    /// The root of the tree.
    root: usize,
    /// How many nodes the tree holds.
    reached: usize,
    /// The most nodes a subtree may hold to be recomputed on its own.
    region_limit: usize,
}

impl<N: DenseId> DominatorTree<N> {
    /// Bring this tree, the dominator tree of the graph before `updates`, up
    /// to date with `graph`, which already contains them, and write to
    /// `changed` every node whose immediate dominator changed, ascending by
    /// dense index, each once.
    ///
    /// The edits are replayed in order, each on the graph as it stood after
    /// it (see the `incremental` module for the rules):
    ///
    /// - An edit from an unreachable node, the insertion of an edge that
    ///   cannot shorten a dominator chain, and the deletion of an edge into a
    ///   node that dominates its tail, or of one copy of a repeated edge,
    ///   leave the tree alone and cost O(degree + depth).
    /// - An insertion into reachable code moves every node it affects below
    ///   the nearest common dominator of its ends, found by a search of what
    ///   it reaches through deeper nodes.
    /// - Any other edit recomputes one subtree.
    ///
    /// An edit that changes the tree renumbers it, O(n), so the next edit
    /// reads levels and dominance from a numbered tree. The result is the
    /// tree [`compute`](Self::compute) builds, `==` to it.
    ///
    /// Replaying stops paying at some point, and the batch is then finished
    /// with one full construction, the cost of [`compute`](Self::compute)
    /// plus an O(n) scan for the nodes that moved: when it holds more than 64
    /// edits, when a subtree to recompute holds more than a quarter of the
    /// tree, or when, with edits left, it would renumber the tree more than
    /// once per 2¹⁴ of its nodes. So a single edit is always replayed, and a
    /// tree under 2¹⁴ nodes finishes a batch from scratch once an edit
    /// changes it. On the benchmark graphs one insertion takes about half
    /// the time of that full construction at 4,096 nodes and about a
    /// ninetieth at 2²⁰, a deletion and a batch of 16 mixed edits about the
    /// same.
    ///
    /// A node beyond the tree's node bound joins the tree as an unreachable
    /// node first, so the edits may name nodes the graph gained after the
    /// tree was computed. `changed` is cleared first; a node that became
    /// reachable or unreachable is in it, as [`idom`](Self::idom) reads.
    ///
    /// # Panics
    ///
    /// Panics when `graph` has a smaller node bound than the tree, when an
    /// edit names a node beyond `graph`'s node bound, when the tree was not
    /// computed from `graph`'s root, and under the conditions of
    /// [`compute`](Self::compute). An edit that the graph does not reflect,
    /// such as the deletion of an edge that was not there, gives an
    /// unspecified tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, DominatorScratch, DominatorTree, DominatorUpdate, EdgeKind};
    ///
    /// // entry -> a -> b
    /// let mut cfg = Cfg::<u32>::new();
    /// let entry = cfg.entry();
    /// let a = cfg.new_block();
    /// let b = cfg.new_block();
    /// cfg.add_edge(entry, a, EdgeKind::Unconditional);
    /// cfg.add_edge(a, b, EdgeKind::Unconditional);
    /// let mut tree = DominatorTree::compute(&cfg);
    /// assert_eq!(tree.idom(b), Some(a));
    ///
    /// // A shortcut entry -> b lifts b to the entry.
    /// cfg.add_edge(entry, b, EdgeKind::ConditionalTrue);
    /// let mut changed = Vec::new();
    /// tree.apply_updates_in(
    ///     &mut DominatorScratch::default(),
    ///     &cfg,
    ///     &[DominatorUpdate::Insert { from: entry, to: b }],
    ///     &mut changed,
    /// );
    /// assert_eq!(tree.idom(b), Some(entry));
    /// assert_eq!(changed, [b]);
    /// assert_eq!(tree, DominatorTree::compute(&cfg));
    /// ```
    pub fn apply_updates_in<G>(
        &mut self,
        scratch: &mut DominatorScratch,
        graph: &G,
        updates: &[DominatorUpdate<N>],
        changed: &mut Vec<N>,
    ) where
        G: RootedView<NodeId = N>,
    {
        self.apply_updates_with(scratch, graph, updates, changed, GATES);
    }

    /// [`apply_updates_in`](Self::apply_updates_in) behind `gates`.
    pub(super) fn apply_updates_with<G>(
        &mut self,
        scratch: &mut DominatorScratch,
        graph: &G,
        updates: &[DominatorUpdate<N>],
        changed: &mut Vec<N>,
        gates: Gates,
    ) where
        G: RootedView<NodeId = N>,
    {
        changed.clear();
        let bound = graph.node_bound();
        check_bound(bound, "node bound of a dominator tree");
        assert!(
            bound >= self.idom.len(),
            "the graph has fewer node slots than the dominator tree"
        );
        let root = graph.root();
        assert!(
            self.order.first().is_none_or(|&first| first == root),
            "the dominator tree was not computed from the graph's root"
        );
        for update in updates {
            let (DominatorUpdate::Insert { from, to } | DominatorUpdate::Delete { from, to }) =
                *update;
            assert!(
                from.index() < bound && to.index() < bound,
                "an edit names a node beyond the graph's node bound"
            );
        }
        self.grow(bound);
        scratch.updates.start(bound);

        if updates.len() <= gates.edits && self.replay(scratch, graph, updates, gates) {
            let buffers = &mut scratch.updates;
            buffers.untouch();
            let UpdateBuffers {
                original,
                touched_list,
                ..
            } = buffers;
            touched_list.retain(|&node| original[slot(node)] != self.idom[slot(node)]);
            touched_list.sort_unstable();
            changed.extend(touched_list.iter().map(|&node| N::from_index(slot(node))));
            return;
        }

        let fresh = Self::construct(scratch, graph, root, None);
        let buffers = &mut scratch.updates;
        for node in 0..bound {
            let before = if buffers.touched[node] {
                buffers.original[node]
            } else {
                self.idom[node]
            };
            if before != fresh.idom[node] {
                changed.push(N::from_index(node));
            }
        }
        buffers.untouch();
        *self = fresh;
    }

    /// Replay `updates` in order. Returns `false` when the gates stop the
    /// batch, which a full construction must then finish.
    fn replay<G>(
        &mut self,
        scratch: &mut DominatorScratch,
        graph: &G,
        updates: &[DominatorUpdate<N>],
        gates: Gates,
    ) -> bool
    where
        G: RootedView<NodeId = N>,
    {
        let DominatorScratch {
            construction,
            numbering,
            updates: buffers,
            ..
        } = scratch;
        let reached = self.order.len();
        let mut pass = Pass {
            numbering,
            buffers,
            root: graph.root().index(),
            reached,
            region_limit: match gates.region_divisor {
                0 => usize::MAX,
                divisor => reached / divisor,
            },
        };
        let budget = match gates.pass_nodes {
            0 => usize::MAX,
            nodes => reached.saturating_mul(reached / nodes),
        };
        let mut spent = 0_usize;
        for (index, &update) in updates.iter().enumerate() {
            let view = Pending {
                graph,
                later: &updates[index + 1..],
            };
            let step = match update {
                DominatorUpdate::Insert { from, to } => {
                    self.insert(&view, from.index(), to.index(), construction, &mut pass)
                }
                DominatorUpdate::Delete { from, to } => {
                    self.delete(&view, from.index(), to.index(), construction, &mut pass)
                }
            };
            match step {
                Step::Kept => {}
                Step::Changed(work) => {
                    // Past the budget with edits left, the tree is not worth
                    // renumbering: a construction replaces it.
                    spent = spent.saturating_add(work).saturating_add(pass.reached);
                    if spent > budget && index + 1 < updates.len() {
                        return false;
                    }
                    self.renumber(&mut pass);
                }
                Step::Abandoned => return false,
            }
        }
        true
    }

    /// Give every table room for `bound` nodes, the new ones unreachable.
    fn grow(&mut self, bound: usize) {
        for table in [
            &mut self.idom,
            &mut self.pre,
            &mut self.last,
            &mut self.level,
        ] {
            table.resize(bound, NO_INDEX);
        }
    }

    /// Number the tree again from its parent table, in its own storage.
    fn renumber(&mut self, pass: &mut Pass<'_>) {
        let mut numbered = Numbered {
            pre: mem::take(&mut self.pre),
            last: mem::take(&mut self.last),
            level: mem::take(&mut self.level),
            order: mem::take(&mut self.order),
        };
        pass.numbering
            .number_into(&self.idom, &[word(pass.root)], pass.reached, &mut numbered);
        let Numbered {
            pre,
            last,
            level,
            order,
        } = numbered;
        (self.pre, self.last, self.level, self.order) = (pre, last, level, order);
    }

    /// Set the parent of `node`, remembering the one it had before the batch.
    fn set_parent(&mut self, node: usize, parent: u32, buffers: &mut UpdateBuffers) {
        if !buffers.touched[node] {
            buffers.touched[node] = true;
            buffers.original[node] = self.idom[node];
            buffers.touched_list.push(word(node));
        }
        self.idom[node] = parent;
    }

    /// Whether the tree holds the node at `index`.
    fn holds(&self, index: usize) -> bool {
        self.pre[index] != NO_INDEX
    }

    /// Whether `ancestor` dominates `node`, both by dense index.
    fn above(&self, ancestor: usize, node: usize) -> bool {
        self.dominates(N::from_index(ancestor), N::from_index(node))
    }

    /// The nearest common dominator of two nodes the tree holds.
    fn meet(&self, first: usize, second: usize) -> usize {
        self.nearest_common_dominator(N::from_index(first), N::from_index(second))
            .expect("two nodes of one tree share its root")
            .index()
    }

    /// Replay the insertion of `from -> to` on `view`.
    fn insert<G: GraphView<NodeId = N>>(
        &mut self,
        view: &Pending<'_, G>,
        from: usize,
        to: usize,
        construction: &mut super::construction::ConstructionBuffers,
        pass: &mut Pass<'_>,
    ) -> Step {
        if !self.holds(from) {
            return Step::Kept;
        }
        if self.holds(to) {
            let common = self.meet(from, to);
            if common == to || self.idom[to] == word(common) {
                return Step::Kept;
            }
            return Step::Changed(self.raise(view, to, common, pass.buffers));
        }

        // `to` and the nodes it newly reaches hang below `from`; the nodes
        // they lead back into decide how far up the change can reach.
        let buffers = &mut *pass.buffers;
        buffers.region.clear();
        buffers.stack.clear();
        buffers.member[to] = true;
        buffers.region.push(word(to));
        buffers.stack.push(word(to));
        let mut top = from;
        while let Some(node) = buffers.stack.pop() {
            for next in view.successors(N::from_index(slot(node))) {
                let next = next.index();
                if self.holds(next) {
                    top = self.meet(top, next);
                } else if !buffers.member[next] {
                    buffers.member[next] = true;
                    buffers.region.push(word(next));
                    buffers.stack.push(word(next));
                }
            }
        }
        self.recompute(view, top, construction, pass)
    }

    /// The depth-based search of an insertion into reachable `to` whose
    /// tail meets it at `common`: move every affected node below `common`.
    /// Returns how many nodes the search visited.
    fn raise<G: GraphView<NodeId = N>>(
        &mut self,
        view: &Pending<'_, G>,
        to: usize,
        common: usize,
        buffers: &mut UpdateBuffers,
    ) -> usize {
        // An affected node is deeper than `common`'s children.
        let floor = self.level[common] + 1;
        let mut found = Vec::new();
        mem::swap(&mut found, &mut buffers.region);
        found.clear();
        let mark = |node: usize, buffers: &mut UpdateBuffers, found: &mut Vec<u32>| {
            buffers.affected[node] = true;
            buffers.marked.push(word(node));
            found.push(word(node));
        };
        mark(to, buffers, &mut found);
        let top = self.level[to];
        buffers.bucket_next[to] = buffers.bucket_head[slot(top)];
        buffers.bucket_head[slot(top)] = word(to);

        let mut level = top;
        while level > floor {
            let next = buffers.bucket_head[slot(level)];
            if next == NO_INDEX {
                level -= 1;
                continue;
            }
            buffers.bucket_head[slot(level)] = buffers.bucket_next[slot(next)];
            if buffers.scanned[slot(next)] {
                continue;
            }
            // Scan everything `next` reaches through nodes deeper than it;
            // a shallower node met on the way, still below the floor's
            // children, is affected too and waits in its level's bucket.
            buffers.scanned[slot(next)] = true;
            buffers.marked.push(next);
            buffers.stack.clear();
            buffers.stack.push(next);
            while let Some(node) = buffers.stack.pop() {
                for successor in view.successors(N::from_index(slot(node))) {
                    let successor = successor.index();
                    let depth = self.level[successor];
                    if depth > level {
                        if !buffers.scanned[successor] {
                            buffers.scanned[successor] = true;
                            buffers.marked.push(word(successor));
                            buffers.stack.push(word(successor));
                        }
                    } else if depth > floor && !buffers.affected[successor] {
                        mark(successor, buffers, &mut found);
                        buffers.bucket_next[successor] = buffers.bucket_head[slot(depth)];
                        buffers.bucket_head[slot(depth)] = word(successor);
                    }
                }
            }
        }
        let visited = buffers.marked.len();
        for &node in &buffers.marked {
            buffers.affected[slot(node)] = false;
            buffers.scanned[slot(node)] = false;
        }
        buffers.marked.clear();
        for &node in &found {
            self.set_parent(slot(node), word(common), buffers);
        }
        mem::swap(&mut found, &mut buffers.region);
        visited
    }

    /// Replay the deletion of one copy of `from -> to` on `view`.
    fn delete<G: GraphView<NodeId = N>>(
        &mut self,
        view: &Pending<'_, G>,
        from: usize,
        to: usize,
        construction: &mut super::construction::ConstructionBuffers,
        pass: &mut Pass<'_>,
    ) -> Step {
        if !self.holds(from) || self.above(to, from) {
            return Step::Kept;
        }
        if view
            .successors(N::from_index(from))
            .any(|next| next.index() == to)
        {
            return Step::Kept;
        }
        let stays = view
            .predecessors(N::from_index(to))
            .any(|before| self.holds(before.index()) && !self.above(to, before.index()));
        let mut top = slot(self.idom[to]);
        if !stays {
            // `to` and its subtree become unreachable. The nodes their edges
            // leave them for hang from dominators of `to`, and the shallowest
            // of those bounds what can move.
            for &node in self.subtree(N::from_index(to)) {
                for next in view.successors(node) {
                    let next = next.index();
                    if self.holds(next) && !self.above(to, next) && !self.above(next, to) {
                        let parent = slot(self.idom[next]);
                        if self.level[parent] < self.level[top] {
                            top = parent;
                        }
                    }
                }
            }
        }
        pass.buffers.region.clear();
        self.recompute(view, top, construction, pass)
    }

    /// Recompute the parents of the subtree of `top`, together with the
    /// unreachable nodes already listed in the region, on `view` restricted
    /// to them, from `top`. Nodes the run does not reach leave the tree.
    ///
    /// A region past the pass's limit is abandoned instead, its marks
    /// cleared.
    fn recompute<G: GraphView<NodeId = N>>(
        &mut self,
        view: &Pending<'_, G>,
        top: usize,
        construction: &mut super::construction::ConstructionBuffers,
        pass: &mut Pass<'_>,
    ) -> Step {
        let buffers = &mut *pass.buffers;
        let span = self.span(N::from_index(top));
        if buffers.region.len() + span.len() > pass.region_limit {
            for &node in &buffers.region {
                buffers.member[slot(node)] = false;
            }
            buffers.region.clear();
            return Step::Abandoned;
        }
        for &node in &self.order[span] {
            buffers.member[node.index()] = true;
            buffers.region.push(word(node.index()));
        }
        let held = buffers
            .region
            .iter()
            .filter(|&&node| self.holds(slot(node)))
            .count();
        construction.run(&Induced::new(view, &buffers.member), top, false);
        let mut reached = 1;
        for (node, parent) in construction.reached_parents() {
            buffers.fresh[node] = parent;
            reached += 1;
        }
        let mut region = mem::take(&mut buffers.region);
        for &node in &region {
            let node = slot(node);
            buffers.member[node] = false;
            if node == top {
                continue;
            }
            let parent = mem::replace(&mut buffers.fresh[node], NO_INDEX);
            if parent != self.idom[node] {
                self.set_parent(node, parent, buffers);
            }
        }
        let work = region.len();
        region.clear();
        buffers.region = region;
        pass.reached = pass.reached + reached - held;
        Step::Changed(work)
    }
}

/// The graph a caller passed with the edits of the batch after the current
/// one undone: an inserted edge taken out, a deleted one put back.
///
/// Only the presence of an edge matters to dominance, so an edge the graph
/// still has is listed as often as the graph lists it, and an edge put back
/// is listed once.
struct Pending<'a, G: GraphView> {
    graph: &'a G,
    later: &'a [DominatorUpdate<G::NodeId>],
}

impl<G: GraphView> Pending<'_, G> {
    /// How many copies of `from -> to` the later edits insert and delete.
    fn edits(&self, from: G::NodeId, to: G::NodeId) -> (usize, usize) {
        self.later
            .iter()
            .fold((0, 0), |(inserted, deleted), update| match *update {
                DominatorUpdate::Insert {
                    from: tail,
                    to: head,
                } if (tail, head) == (from, to) => (inserted + 1, deleted),
                DominatorUpdate::Delete {
                    from: tail,
                    to: head,
                } if (tail, head) == (from, to) => (inserted, deleted + 1),
                _ => (inserted, deleted),
            })
    }

    /// Whether the graph of this edit has `from -> to`, which the graph
    /// lists `copies` times.
    fn has(&self, from: G::NodeId, to: G::NodeId, copies: usize) -> bool {
        let (inserted, deleted) = self.edits(from, to);
        copies + deleted > inserted
    }

    /// The heads of the edges a later edit deletes from `node` that the
    /// graph no longer lists, each once, when the graph of this edit has
    /// them; or the tails, read backwards.
    fn restored(&self, node: G::NodeId, backwards: bool) -> impl Iterator<Item = G::NodeId> + '_ {
        self.later
            .iter()
            .enumerate()
            .filter_map(move |(index, update)| {
                let DominatorUpdate::Delete { from, to } = *update else {
                    return None;
                };
                let (near, far) = if backwards { (to, from) } else { (from, to) };
                let first = !self.later[..index].iter().any(|earlier| earlier == update);
                (near == node && first).then_some(far)
            })
            .filter(move |&far| {
                let (from, to) = if backwards { (far, node) } else { (node, far) };
                let copies = self
                    .graph
                    .successors(from)
                    .filter(|&next| next == to)
                    .count();
                copies == 0 && self.has(from, to, 0)
            })
    }

    /// Whether a later edit names `node` as the tail, or as the head when
    /// read `backwards`.
    fn edited(&self, node: G::NodeId, backwards: bool) -> bool {
        self.later.iter().any(|update| {
            let (DominatorUpdate::Insert { from, to } | DominatorUpdate::Delete { from, to }) =
                *update;
            node == if backwards { to } else { from }
        })
    }
}

impl<G: GraphView> GraphView for Pending<'_, G> {
    type NodeId = G::NodeId;

    fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = G::NodeId> + '_ {
        self.graph.node_ids()
    }

    fn successors(&self, node: G::NodeId) -> impl Iterator<Item = G::NodeId> + '_ {
        let edited = self.edited(node, false);
        self.graph
            .successors(node)
            .filter(move |&next| {
                !edited || {
                    let copies = self
                        .graph
                        .successors(node)
                        .filter(|&other| other == next)
                        .count();
                    self.has(node, next, copies)
                }
            })
            .chain(self.restored(node, false))
    }

    fn predecessors(&self, node: G::NodeId) -> impl Iterator<Item = G::NodeId> + '_ {
        let edited = self.edited(node, true);
        self.graph
            .predecessors(node)
            .filter(move |&before| {
                !edited || {
                    let copies = self
                        .graph
                        .predecessors(node)
                        .filter(|&other| other == before)
                        .count();
                    self.has(before, node, copies)
                }
            })
            .chain(self.restored(node, true))
    }
}
