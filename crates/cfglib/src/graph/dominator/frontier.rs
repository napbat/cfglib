//! Dominance frontiers, answered per query, and the iterated-frontier search
//! every frontier query shares.
//!
//! The frontier of a node `x` is every node `y` where `x` dominates a
//! predecessor of `y` but does not strictly dominate `y`: the first nodes
//! beyond the region `x` controls. A variable written in a set of nodes needs
//! a phi function, or a block parameter, at every node of the *iterated*
//! frontier of that set, which is the set closed under taking the frontiers of
//! its own members. Run over the reversed edges of a graph, the frontier of a
//! node is the set of nodes it is control dependent on.
//!
//! Nothing is tabulated, because a table of every frontier can be quadratic in
//! the node count. [`DominatorTree::frontier`] evaluates the definition over
//! the subtree of one node. [`DominatorTree::iterated_frontier`] is the
//! level-guided search of Sreedhar and Gao in the form LLVM's
//! `IDFCalculatorBase` takes: no explicit DJ graph, roots handled deepest level
//! first, a successor accepted only when its level is no deeper than the
//! root's, and an optional live-in test that prunes the result. Roots wait in
//! buckets by level rather than in a heap, so a query costs O(n + m) for n
//! nodes and m edges, plus the sort that puts the answer in ascending order.
//!
//! The worklist of that search, [`FrontierBuffers`], is also the one
//! [`DominanceFrontiers::iterated_frontier`](super::DominanceFrontiers::iterated_frontier)
//! runs over its table: a candidate is judged once per query, added when the
//! live-in test takes it, and then queued once as a root of its own.
//!
//! # Direction
//!
//! Both queries follow the [`successors`](GraphView::successors) of the view
//! they are given. Pass the view the tree was computed over. A tree computed
//! over a [`Reversed`](crate::Reversed) view takes that same reversed view.
//!
//! # Forests
//!
//! A tree computed from one root has one tree, and the forests that SSA
//! construction and post-dominance build have several. Both queries work on a
//! forest. A node in no tree is ignored wherever it appears, as a definition
//! and as a successor. Levels count from the root of each tree, which has level
//! 0, and an edge into another tree of the forest is a member of the frontier
//! when the level test of the search accepts it. A forest numbered from a
//! virtual root, as both of those are, only ever has edges between trees that
//! end at the root of the target tree, which has level 0 and so always passes.

extern crate alloc;
use alloc::vec::Vec;

use super::{DominatorScratch, DominatorTree};
use crate::graph::epoch::{EpochMarks, grow_to};
use crate::graph::view::{DenseId, GraphView};
use crate::identity::{NO_INDEX, slot, word};

/// The worklist of an iterated-frontier query, kept in a
/// [`DominatorScratch`]: the marks of the nodes queued and judged, and the
/// roots waiting by level.
///
/// The marks are epochs, so the per-query clearing is a bump rather than a
/// pass over the graph, and a query allocates nothing once the buffers have
/// grown to the graph.
#[derive(Debug, Clone, Default)]
pub(super) struct FrontierBuffers {
    /// Nodes that are, or have been, a root of the search: the definitions,
    /// and every node added to the answer. A queued node is walked once.
    queued: EpochMarks,
    /// Nodes already judged as candidates for the answer, whether the live-in
    /// test took them or not, so a node is judged once per query.
    judged: EpochMarks,
    /// The first root waiting at each level, [`NO_INDEX`] when none waits.
    bucket_head: Vec<u32>,
    /// The root after each queued node in its level's bucket.
    bucket_next: Vec<u32>,
    /// The deepest level that may still hold a waiting root.
    deepest: usize,
}

impl FrontierBuffers {
    /// Empty every buffer and grow the node-indexed ones to `node_bound`.
    ///
    /// This is a correctness step, not only hygiene: nothing a previous query
    /// left behind, including one that panicked, may reach this one's answer.
    pub(super) fn begin(&mut self, node_bound: usize) {
        for marks in [&mut self.queued, &mut self.judged] {
            marks.grow_to(node_bound);
            marks.reset();
        }
        grow_to(&mut self.bucket_next, node_bound, NO_INDEX);
        self.bucket_head.clear();
        self.deepest = 0;
    }

    /// Queue `node`, of tree level `level`, as a root to walk, unless it has
    /// been queued already.
    pub(super) fn enqueue(&mut self, node: usize, level: usize) {
        if self.queued.is_marked(node) {
            return;
        }
        self.queued.mark(node);
        if self.bucket_head.len() <= level {
            self.bucket_head.resize(level + 1, NO_INDEX);
        }
        self.bucket_next[node] = self.bucket_head[level];
        self.bucket_head[level] = word(node);
        self.deepest = self.deepest.max(level);
    }

    /// Take the next root to walk: one of the deepest level that holds one.
    pub(super) fn next_root(&mut self) -> Option<usize> {
        loop {
            let head = *self.bucket_head.get(self.deepest)?;
            if head != NO_INDEX {
                let node = slot(head);
                self.bucket_head[self.deepest] = self.bucket_next[node];
                return Some(node);
            }
            self.deepest = self.deepest.checked_sub(1)?;
        }
    }

    /// Judge `member`, of tree level `level`, as a member of the answer:
    /// once per query, it is added to `out` and queued as a root when
    /// `live_in` accepts it or there is no `live_in`.
    #[inline]
    pub(super) fn offer<N: DenseId>(
        &mut self,
        member: N,
        level: usize,
        live_in: Option<&dyn Fn(N) -> bool>,
        out: &mut Vec<N>,
    ) {
        let index = member.index();
        if self.judged.is_marked(index) {
            return;
        }
        self.judged.mark(index);
        if live_in.is_some_and(|accepts| !accepts(member)) {
            return;
        }
        out.push(member);
        self.enqueue(index, level);
    }

    /// Whether `node` is or has been a root of this query.
    fn is_queued(&self, node: usize) -> bool {
        self.queued.is_marked(node)
    }
}

impl<N: DenseId> DominatorTree<N> {
    /// Collect the dominance frontier of `node` into `out`.
    ///
    /// The frontier is every node `y` where `node` dominates a predecessor of
    /// `y` and does not strictly dominate `y`. It is found the way the
    /// definition reads: every successor of every node `node` dominates is
    /// kept unless `node` strictly dominates it. That includes `node` itself
    /// when an edge leads back to it from its own subtree, as one does for an
    /// entry that heads a loop. A node in no tree is never a member, and an
    /// unreachable `node` has an empty frontier.
    ///
    /// `out` is cleared first and receives each member once, in ascending
    /// dense index, the order [`DominanceFrontiers::frontier`](super::DominanceFrontiers::frontier)
    /// gives.
    ///
    /// `graph` must be the view the tree was computed over, because the walk
    /// follows its [`successors`](GraphView::successors). A tree computed over
    /// [`Reversed`](crate::Reversed) takes that reversed view, which makes
    /// the frontier of a node the set of nodes it is control dependent on. On
    /// a forest, an edge from the subtree to a node of another tree makes that
    /// node a member; see the module documentation.
    ///
    /// The cost is O(|subtree| + successors scanned), plus sorting the
    /// qualifying edges; `out` is the only buffer it grows.
    ///
    /// # Panics
    ///
    /// Panics when `node`, or a successor of a node in its subtree, has an
    /// index beyond the node bound the tree was computed with.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, DominatorTree, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let entry = cfg.entry();
    /// let left = cfg.new_block();
    /// let right = cfg.new_block();
    /// let merge = cfg.new_block();
    /// cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(entry, right, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    /// cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    /// cfg.add_edge(merge, entry, EdgeKind::Back);
    /// let dominators = DominatorTree::compute(&cfg);
    ///
    /// let mut frontier = Vec::new();
    /// // Both arms hand control to `merge`, which neither dominates.
    /// dominators.frontier(&cfg, left, &mut frontier);
    /// assert_eq!(frontier, [merge]);
    /// // The entry dominates everything, but the back edge returns to it.
    /// dominators.frontier(&cfg, entry, &mut frontier);
    /// assert_eq!(frontier, [entry]);
    /// ```
    pub fn frontier<G: GraphView<NodeId = N>>(&self, graph: &G, node: N, out: &mut Vec<N>) {
        out.clear();
        let span = self.span(node);
        let strictly_dominated = span.start + 1..span.end;
        for &member in &self.order[span] {
            for successor in graph.successors(member) {
                let position = self.pre[successor.index()];
                if position != NO_INDEX && !strictly_dominated.contains(&slot(position)) {
                    out.push(successor);
                }
            }
        }
        out.sort_unstable_by_key(|member| member.index());
        out.dedup();
    }

    /// Collect the iterated dominance frontier of `defs` into `out`.
    ///
    /// The result is the smallest set that holds the frontier of every
    /// definition and the frontier of each of its own members. It is where a
    /// variable written in `defs` needs a phi function or a block parameter.
    /// A definition that sits in no tree is ignored, and so is one named twice.
    ///
    /// With `live_in`, the result is pruned the way LLVM prunes phi placement:
    /// a node is added, and its own frontier explored, only when `live_in`
    /// accepts it. A rejected node never enters the answer, and a definition
    /// is a root of the search whether `live_in` accepts it or not. Without
    /// one, every node of the plain iterated frontier is returned. Each node
    /// is offered to `live_in` at most once per call.
    ///
    /// `out` is cleared first and receives each member once, in ascending
    /// dense index.
    ///
    /// `graph` must be the view the tree was computed over, because the search
    /// follows its [`successors`](GraphView::successors); see
    /// [`frontier`](Self::frontier). A tree computed over
    /// [`Reversed`](crate::Reversed) takes that reversed view.
    ///
    /// The search takes the roots deepest level first from buckets, and walks
    /// the dominator subtree of each, skipping the subtrees an earlier root
    /// already walked. It follows an edge only when the level of its target
    /// is no deeper than the level of the root being walked, which is exactly
    /// when the target is in the frontier of that root. Every node and edge
    /// is visited at most once, so the cost is O(n + m) plus the sort of the
    /// answer. A pass that asks for many frontiers hands one scratch to
    /// [`iterated_frontier_in`](Self::iterated_frontier_in) instead, which
    /// allocates nothing once the scratch and `out` have grown.
    ///
    /// A graph that takes many queries, or queries whose definitions sit high
    /// in the tree, reads the same answer faster off a
    /// [`DominanceFrontiers`](super::DominanceFrontiers) table with
    /// [`DominanceFrontiers::iterated_frontier`](super::DominanceFrontiers::iterated_frontier).
    ///
    /// # Panics
    ///
    /// Panics when a definition, or a successor of a node the search walks,
    /// has an index beyond the node bound the tree was computed with.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, DominatorTree, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let entry = cfg.entry();
    /// let head = cfg.new_block();
    /// let body = cfg.new_block();
    /// let exit = cfg.new_block();
    /// cfg.add_edge(entry, head, EdgeKind::Fallthrough);
    /// cfg.add_edge(head, body, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(head, exit, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(body, head, EdgeKind::Back);
    /// let dominators = DominatorTree::compute(&cfg);
    ///
    /// // A variable written in the loop body is merged at the loop header.
    /// let mut placed = Vec::new();
    /// dominators.iterated_frontier(&cfg, [body], None, &mut placed);
    /// assert_eq!(placed, [head]);
    ///
    /// // Pruned by liveness: the variable is dead on entry to the header.
    /// let live_in = |block| block != head;
    /// dominators.iterated_frontier(&cfg, [body], Some(&live_in), &mut placed);
    /// assert!(placed.is_empty());
    /// ```
    pub fn iterated_frontier<G: GraphView<NodeId = N>>(
        &self,
        graph: &G,
        defs: impl IntoIterator<Item = N>,
        live_in: Option<&dyn Fn(N) -> bool>,
        out: &mut Vec<N>,
    ) {
        self.iterated_frontier_in(&mut DominatorScratch::default(), graph, defs, live_in, out);
    }

    /// [`iterated_frontier`](Self::iterated_frontier) over caller-owned
    /// working storage.
    ///
    /// A pass that places phi functions asks for one iterated frontier per
    /// variable, and most variables are written in a handful of blocks. At
    /// that size the search is not the cost, the *call* is: two visit flags
    /// the size of the graph and the buckets of waiting roots, each a malloc
    /// and a free for an answer of two nodes. One [`DominatorScratch`] handed
    /// to every query of the pass moves them out of the call, and the answer
    /// is the same.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of
    /// [`iterated_frontier`](Self::iterated_frontier).
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, DominatorScratch, DominatorTree, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let entry = cfg.entry();
    /// let then_block = cfg.new_block();
    /// let merge = cfg.new_block();
    /// cfg.add_edge(entry, then_block, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(entry, merge, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);
    /// let dominators = DominatorTree::compute(&cfg);
    ///
    /// let mut scratch = DominatorScratch::default();
    /// let mut fresh = Vec::new();
    /// let mut reused = Vec::new();
    /// dominators.iterated_frontier(&cfg, [then_block], None, &mut fresh);
    /// dominators.iterated_frontier_in(&mut scratch, &cfg, [then_block], None, &mut reused);
    /// dominators.iterated_frontier_in(&mut scratch, &cfg, [then_block], None, &mut reused);
    /// assert_eq!(reused, fresh);
    /// assert_eq!(reused, [merge]);
    /// ```
    pub fn iterated_frontier_in<G: GraphView<NodeId = N>>(
        &self,
        scratch: &mut DominatorScratch,
        graph: &G,
        defs: impl IntoIterator<Item = N>,
        live_in: Option<&dyn Fn(N) -> bool>,
        out: &mut Vec<N>,
    ) {
        out.clear();
        let search = &mut scratch.frontier;
        search.begin(self.pre.len());
        for def in defs {
            let index = def.index();
            if self.pre[index] != NO_INDEX {
                search.enqueue(index, slot(self.level[index]));
            }
        }
        while let Some(root) = search.next_root() {
            self.walk_root(graph, search, live_in, root, out);
        }
        out.sort_unstable_by_key(|member| member.index());
    }

    /// Walk the subtree of `root`, and offer every successor that lies in the
    /// frontier of `root`.
    ///
    /// The walk is a scan of the preorder positions of the subtree. A queued
    /// node met on the way, other than `root`, was a root of an earlier walk:
    /// roots leave the buckets deepest first, and a root that is deeper than
    /// `root` is the only kind that can lie inside its subtree. Walks nest
    /// like subtrees do, so that earlier walk covered the whole subtree of the
    /// node, and the scan jumps past it.
    fn walk_root<G: GraphView<NodeId = N>>(
        &self,
        graph: &G,
        search: &mut FrontierBuffers,
        live_in: Option<&dyn Fn(N) -> bool>,
        root: usize,
        out: &mut Vec<N>,
    ) {
        let root_level = self.level[root];
        let start = slot(self.pre[root]);
        let end = slot(self.last[root]) + 1;
        let mut position = start;
        while position < end {
            let node = self.order[position];
            if position != start && search.is_queued(node.index()) {
                position = slot(self.last[node.index()]) + 1;
                continue;
            }
            position += 1;
            for successor in graph.successors(node) {
                let index = successor.index();
                if self.pre[index] == NO_INDEX || self.level[index] > root_level {
                    continue;
                }
                search.offer(successor, slot(self.level[index]), live_in, out);
            }
        }
    }
}
