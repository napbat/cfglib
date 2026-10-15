//! Dominance frontiers, answered per query.
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
//! nodes and m edges, plus the sort that puts the answer in preorder.
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

use super::{ABSENT, DominatorTree, slot, word};
use crate::graph::epoch::EpochMarks;
use crate::graph::view::{DenseId, GraphView};

/// Every buffer [`DominatorTree::iterated_frontier`] fills, owned by the
/// caller.
///
/// A pass that places phi functions asks for one iterated frontier per
/// variable, and most variables are written in a handful of blocks. At that
/// size the search is not the cost, the *call* is: two visit flags the size of
/// the graph and the buckets of waiting roots, each a malloc and a free for an
/// answer of two nodes. This moves all of them out of the call, and the
/// per-query clearing of the flags is a bump of an epoch rather than a pass over
/// the graph.
///
/// Hand one scratch to every [`iterated_frontier`](DominatorTree::iterated_frontier)
/// of the pass. The buffers grow to the largest graph the pass meets and are
/// then reused by every graph after it, so a query allocates nothing once the
/// scratch has seen a graph that size.
///
/// # One scratch, any graph
///
/// Nodes are held as dense indices rather than as a node-id type, so the
/// scratch carries no type parameter: one instance serves a [`Cfg`](crate::Cfg),
/// a [`Graph`](crate::Graph), and a consumer-defined [`GraphView`], in any
/// order, and it is [`Send`], so a worker thread can own one for the whole
/// corpus it is handed.
///
/// # Sizing
///
/// Nothing is fixed at construction and nothing is released by a call. Every
/// buffer is cleared and grown on entry, so a scratch used on a large graph
/// and then on a small one is correct and allocation-free, and the space it
/// holds is the high-water mark of the sequence.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, DominatorTree, EdgeKind, FrontierScratch};
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
/// let mut scratch = FrontierScratch::new();
/// let mut fresh = Vec::new();
/// let mut reused = Vec::new();
/// dominators.iterated_frontier(&cfg, &mut FrontierScratch::new(), [then_block], None, &mut fresh);
/// dominators.iterated_frontier(&cfg, &mut scratch, [then_block], None, &mut reused);
/// dominators.iterated_frontier(&cfg, &mut scratch, [then_block], None, &mut reused);
/// assert_eq!(reused, fresh);
/// assert_eq!(reused, [merge]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct FrontierScratch {
    /// Nodes that are, or have been, a root of the search: the definitions,
    /// and every node added to the answer. A queued node is walked once.
    queued: EpochMarks,
    /// Nodes already judged as candidates for the answer, whether the live-in
    /// test took them or not, so a node is judged once per query.
    judged: EpochMarks,
    /// The first root waiting at each level, [`ABSENT`] when none waits.
    bucket_head: Vec<u32>,
    /// The root after each queued node in its level's bucket.
    bucket_next: Vec<u32>,
}

impl FrontierScratch {
    /// Scratch holding nothing yet.
    ///
    /// Every buffer is sized by the first graph it sees, so there is no node
    /// count to state here.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Empty every buffer and grow the node-indexed ones to `node_bound`.
    ///
    /// This is a correctness step, not only hygiene: nothing a previous query
    /// left behind, including one that panicked, may reach this one's answer.
    fn begin(&mut self, node_bound: usize) {
        for marks in [&mut self.queued, &mut self.judged] {
            marks.grow_to(node_bound);
            marks.reset();
        }
        if self.bucket_next.len() < node_bound {
            self.bucket_next.resize(node_bound, ABSENT);
        }
        self.bucket_head.clear();
    }

    /// Queue `node`, of tree level `level`, as a root to walk, unless it has
    /// been queued already.
    fn enqueue(&mut self, node: usize, level: usize) {
        if self.queued.is_marked(node) {
            return;
        }
        self.queued.mark(node);
        if self.bucket_head.len() <= level {
            self.bucket_head.resize(level + 1, ABSENT);
        }
        self.bucket_next[node] = self.bucket_head[level];
        self.bucket_head[level] = word(node);
    }

    /// Take a root off the bucket of `level`, if one waits there.
    fn dequeue(&mut self, level: usize) -> Option<usize> {
        let head = self.bucket_head[level];
        if head == ABSENT {
            return None;
        }
        let node = slot(head);
        self.bucket_head[level] = self.bucket_next[node];
        Some(node)
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
    /// `out` is cleared first and receives each member once, in preorder.
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
                if position != ABSENT && !strictly_dominated.contains(&slot(position)) {
                    out.push(successor);
                }
            }
        }
        out.sort_unstable_by_key(|member| self.pre[member.index()]);
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
    /// `out` is cleared first and receives each member once, in preorder.
    ///
    /// `graph` must be the view the tree was computed over, because the search
    /// follows its [`successors`](GraphView::successors); see
    /// [`frontier`](Self::frontier). A tree computed over
    /// [`Reversed`](crate::Reversed) takes that reversed view.
    ///
    /// The search takes the roots deepest level first from buckets in
    /// `scratch`, and walks the dominator subtree of each, skipping the
    /// subtrees an earlier root already walked. It follows an edge only when
    /// the level of its target is no deeper than the level of the root being
    /// walked, which is exactly when the target is in the frontier of that
    /// root. Every node and edge is visited at most once, so the cost is
    /// O(n + m) plus the sort of the answer, and a call allocates nothing once
    /// `scratch` and `out` have grown to the size the graph needs.
    ///
    /// # Panics
    ///
    /// Panics when a definition, or a successor of a node the search walks,
    /// has an index beyond the node bound the tree was computed with.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, DominatorTree, EdgeKind, FrontierScratch};
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
    /// let mut scratch = FrontierScratch::new();
    /// let mut placed = Vec::new();
    /// dominators.iterated_frontier(&cfg, &mut scratch, [body], None, &mut placed);
    /// assert_eq!(placed, [head]);
    ///
    /// // Pruned by liveness: the variable is dead on entry to the header.
    /// let live_in = |block| block != head;
    /// dominators.iterated_frontier(&cfg, &mut scratch, [body], Some(&live_in), &mut placed);
    /// assert!(placed.is_empty());
    /// ```
    pub fn iterated_frontier<G: GraphView<NodeId = N>>(
        &self,
        graph: &G,
        scratch: &mut FrontierScratch,
        defs: impl IntoIterator<Item = N>,
        live_in: Option<&dyn Fn(N) -> bool>,
        out: &mut Vec<N>,
    ) {
        out.clear();
        scratch.begin(self.pre.len());
        for def in defs {
            let index = def.index();
            if self.pre[index] != ABSENT {
                scratch.enqueue(index, slot(self.level[index]));
            }
        }
        for level in (0..scratch.bucket_head.len()).rev() {
            while let Some(root) = scratch.dequeue(level) {
                self.walk_root(graph, scratch, live_in, root, out);
            }
        }
        out.sort_unstable_by_key(|member| self.pre[member.index()]);
    }

    /// Walk the subtree of `root`, and add to `out` and to the queue every
    /// successor that lies in the frontier of `root`.
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
        scratch: &mut FrontierScratch,
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
            if position != start && scratch.queued.is_marked(node.index()) {
                position = slot(self.last[node.index()]) + 1;
                continue;
            }
            position += 1;
            for successor in graph.successors(node) {
                let index = successor.index();
                if self.pre[index] == ABSENT
                    || self.level[index] > root_level
                    || scratch.judged.is_marked(index)
                {
                    continue;
                }
                scratch.judged.mark(index);
                if live_in.is_some_and(|accepts| !accepts(successor)) {
                    continue;
                }
                out.push(successor);
                scratch.enqueue(index, slot(self.level[index]));
            }
        }
    }
}
