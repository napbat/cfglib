//! Dominance frontiers as one flat run table.
//!
//! The runner walk of Cooper, Harvey, and Kennedy collects `(runner, frontier
//! block)` pairs, and grouping them by runner makes every block's frontier
//! one contiguous, ascending run. [`DominanceFrontiers`] is that table: it
//! hands a consumer one block's frontier, or the iterated frontier of a set
//! of blocks.
//!
//! It reads any [`GraphView`], so a [`Cfg`](crate::Cfg) and a consumer-owned
//! function view share the one algorithm, and any [`FrontierTree`]: a
//! [`DominatorTree`] over the graph it was computed on, or a
//! [`PostDominatorTree`](crate::PostDominatorTree), whose frontiers are the
//! control-dependence sources of its blocks.
//!
//! The rule itself is the one [`DominatorTree::frontier`] states, which
//! evaluates it per query where this table walks every join once: block `y`
//! is in the frontier of `x` when `x` dominates a predecessor of `y` and does
//! not strictly dominate `y`. This includes a tree root that is a branch
//! target, such as an entry with a back edge into it: the root is in its own
//! frontier and in the frontier of every block on the path back to it. A
//! block outside every tree, such as unreachable code, has an empty frontier,
//! and an edge from it contributes nothing to the frontier of live code. The
//! tests check the table against the per-query frontier on every small
//! digraph.

extern crate alloc;

use alloc::vec::Vec;

use super::{DominatorScratch, DominatorTree};
use crate::block::BlockId;
use crate::graph::view::{DenseId, GraphView};

/// A tree whose frontiers [`DominanceFrontiers`] tabulates.
///
/// A [`DominatorTree`] gives its dominance frontiers over the graph it was
/// computed on. A [`PostDominatorTree`](crate::PostDominatorTree) gives the
/// frontiers of [`PostDominatorTree::frontier`](crate::PostDominatorTree::frontier),
/// which it reads over the reverse of the graph it was computed on: the
/// control-dependence sources of each block, where a block of a sink class
/// has the sources of its class. The trait is sealed; those two are its only
/// implementations.
pub trait FrontierTree<N>: sealed::Tabulate<N> {}

impl<N: DenseId> FrontierTree<N> for DominatorTree<N> {}

pub(super) mod sealed {
    use super::DominanceFrontiers;
    use crate::graph::view::GraphView;

    /// How a tree fills a frontier table, which only the crate implements.
    pub trait Tabulate<N> {
        /// Fill `table` with the frontier of every block of `graph`.
        fn tabulate<G: GraphView<NodeId = N>>(&self, graph: &G, table: &mut DominanceFrontiers<N>);
    }
}

impl<N: DenseId> sealed::Tabulate<N> for DominatorTree<N> {
    fn tabulate<G: GraphView<NodeId = N>>(&self, graph: &G, table: &mut DominanceFrontiers<N>) {
        table.collect(graph, self);
        table.group(graph.node_bound());
    }
}

/// The frontier of every block, in one flat run table.
///
/// The block type `N` is the node identity of the view the frontiers were
/// computed from. It defaults to [`BlockId`], the identity of a
/// [`Cfg`](crate::Cfg).
///
/// For a reachable block, the frontier holds the members of the per-query
/// frontier of its tree ([`DominatorTree::frontier`] or
/// [`PostDominatorTree::frontier`](crate::PostDominatorTree::frontier)), in
/// the same ascending dense index. The runner walk emits each `(runner,
/// frontier block)` pair once, and the table groups them by runner with a
/// counting pass instead of a sort: one buffer of members and one of run
/// starts however many blocks there are.
#[derive(Debug, Clone)]
pub struct DominanceFrontiers<N = BlockId> {
    /// The `(runner, frontier block)` pairs of the last walk, in the order
    /// the walk emitted them. A table that is kept drops them; one that is
    /// recomputed keeps them for the next walk.
    pub(super) pairs: Vec<(N, N)>,
    /// Where each block's run starts in `blocks`. One longer than the block
    /// bound, so a run is `starts[b]..starts[b + 1]`.
    starts: Vec<usize>,
    /// The frontier members themselves, grouped by runner, each run strictly
    /// ascending by dense index.
    blocks: Vec<N>,
}

impl<N: DenseId> DominanceFrontiers<N> {
    /// Compute the frontier of every block of `graph` under `tree`, with the
    /// runner walk of Cooper, Harvey, and Kennedy.
    ///
    /// `tree` is the dominator tree or the post-dominator tree computed from
    /// `graph`. The predecessors of `graph`, or its successors for a
    /// post-dominator tree, are what the algorithm walks, so a view whose
    /// adjacency breaks the [`GraphView`] contract gives an unspecified
    /// answer.
    ///
    /// # Examples
    ///
    /// The control-dependence frontiers of a diamond: both arms depend on the
    /// branch, and the merge on nothing.
    ///
    /// ```
    /// use cfglib::{Cfg, DominanceFrontiers, EdgeKind, PostDominatorTree};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let branch = cfg.entry();
    /// let left = cfg.new_block();
    /// let right = cfg.new_block();
    /// let merge = cfg.new_block();
    /// cfg.add_edge(branch, left, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(branch, right, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    /// cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    /// let post = PostDominatorTree::compute(&cfg);
    ///
    /// let sources = DominanceFrontiers::compute(&cfg, &post);
    /// assert_eq!(sources.frontier(left), [branch]);
    /// assert_eq!(sources.frontier(right), [branch]);
    /// assert!(sources.frontier(merge).is_empty());
    /// ```
    #[must_use]
    pub fn compute<G, T>(graph: &G, tree: &T) -> Self
    where
        G: GraphView<NodeId = N>,
        T: FrontierTree<N>,
    {
        let mut table = Self::default();
        tree.tabulate(graph, &mut table);
        // A recomputed table keeps the pair list for the next procedure, but
        // a table that is kept has no use for it.
        table.pairs = Vec::new();
        table
    }

    /// Recompute the table for `graph` under `tree` in the storage of this
    /// table.
    ///
    /// The result equals [`compute`](Self::compute) of the same graph and
    /// tree. The buffers of the table, and the pair list of the walk, stay
    /// from one call to the next, so a pass that recomputes the frontiers of
    /// each procedure into one table allocates only while the table grows to
    /// the largest procedure.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, DominanceFrontiers, DominatorTree, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let entry = cfg.entry();
    /// let left = cfg.new_block();
    /// let merge = cfg.new_block();
    /// cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(entry, merge, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    /// let dominators = DominatorTree::compute(&cfg);
    ///
    /// let mut frontiers = DominanceFrontiers::default();
    /// frontiers.recompute(&cfg, &dominators);
    /// assert_eq!(frontiers.frontier(left), [merge]);
    /// assert!(frontiers.frontier(entry).is_empty());
    /// ```
    pub fn recompute<G, T>(&mut self, graph: &G, tree: &T)
    where
        G: GraphView<NodeId = N>,
        T: FrontierTree<N>,
    {
        tree.tabulate(graph, self);
    }

    /// Return the frontier of `block`, in ascending dense index.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the frontiers.
    #[must_use]
    pub fn frontier(&self, block: N) -> &[N] {
        let start = self.starts[block.index()];
        let end = self.starts[block.index() + 1];
        &self.blocks[start..end]
    }

    /// Collect the iterated frontier of `defs` into `out`, read off this
    /// table.
    ///
    /// The answer is the one the tree that computed the table gives
    /// ([`DominatorTree::iterated_frontier`] or
    /// [`PostDominatorTree::iterated_frontier`](crate::PostDominatorTree::iterated_frontier)),
    /// with the same `live_in` rule: a node is added, and its own frontier
    /// explored, only when `live_in` accepts it, a definition is a root of
    /// the search whether `live_in` accepts it or not, and each node is
    /// offered to `live_in` at most once per call. A definition outside every
    /// tree has an empty frontier, so it adds nothing. `out` is cleared first
    /// and receives each member once, in ascending dense index.
    ///
    /// # Choosing between the two queries
    ///
    /// This query visits only the frontiers of the definitions and of the
    /// nodes it adds, so its cost is the size of those frontiers, but the
    /// table costs one pass over the graph to build, and its size can be
    /// quadratic in the node count. [`DominatorTree::iterated_frontier`] needs
    /// no table, but each query scans the dominator subtree of each
    /// definition, which is the whole tree once a definition is a tree root.
    /// Build the table for a graph that takes many queries, such as one query
    /// for each variable or each register of a function, and above all when
    /// a definition sits high in the tree. Use the tree for a few queries.
    ///
    /// # Panics
    ///
    /// Panics when a definition is not below the node bound of the view that
    /// computed the table.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, DominanceFrontiers, DominatorScratch, DominatorTree, EdgeKind};
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
    /// let frontiers = DominanceFrontiers::compute(&cfg, &dominators);
    ///
    /// // The entry defines the variable and the loop body writes it again.
    /// let mut placed = Vec::new();
    /// frontiers.iterated_frontier([entry, body], None, &mut placed);
    /// assert_eq!(placed, [head]);
    ///
    /// // Pruned by liveness: the variable is dead on entry to the header.
    /// let mut scratch = DominatorScratch::default();
    /// let live_in = |block| block != head;
    /// frontiers.iterated_frontier_in(&mut scratch, [entry, body], Some(&live_in), &mut placed);
    /// assert!(placed.is_empty());
    /// ```
    pub fn iterated_frontier(
        &self,
        defs: impl IntoIterator<Item = N>,
        live_in: Option<&dyn Fn(N) -> bool>,
        out: &mut Vec<N>,
    ) {
        self.iterated_frontier_in(&mut DominatorScratch::default(), defs, live_in, out);
    }

    /// [`iterated_frontier`](Self::iterated_frontier) over caller-owned
    /// working storage, the same answer: a pass that asks for many frontiers
    /// hands one [`DominatorScratch`] to every query, and a query allocates
    /// nothing once the scratch and `out` have grown.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of
    /// [`iterated_frontier`](Self::iterated_frontier).
    pub fn iterated_frontier_in(
        &self,
        scratch: &mut DominatorScratch,
        defs: impl IntoIterator<Item = N>,
        live_in: Option<&dyn Fn(N) -> bool>,
        out: &mut Vec<N>,
    ) {
        out.clear();
        // The table knows no levels, and needs none: every root waits on one
        // level, so the buckets are a stack.
        let search = &mut scratch.frontier;
        search.begin(self.starts.len().saturating_sub(1));
        for def in defs {
            search.enqueue(def.index(), 0);
        }
        while let Some(node) = search.next_root() {
            for &member in self.frontier(N::from_index(node)) {
                search.offer(member, 0, live_in, out);
            }
        }
        out.sort_unstable_by_key(|member| member.index());
    }

    /// Collect the `(runner, frontier block)` pairs of `graph` under `dom`.
    pub(super) fn collect<G>(&mut self, graph: &G, dom: &DominatorTree<N>)
    where
        G: GraphView<NodeId = N>,
    {
        let bound = graph.node_bound();
        self.pairs.clear();
        // `starts` is not read until the pairs are grouped, so the walk uses
        // it to remember, per runner, the block whose walk reached it last.
        self.starts.clear();
        self.starts.resize(bound + 1, usize::MAX);
        for block in graph.node_ids() {
            // Unreachable code has no frontier, and an edge out of it adds
            // none to live code: a runner is always a block the tree holds.
            if !dom.is_reachable(block) {
                continue;
            }

            // A frontier block is a join: two incoming paths at least. A tree
            // root has one path that is not an edge, the start of its tree,
            // so one predecessor edge already makes it a join.
            let stop = dom.idom(block);
            let joins_at = if stop.is_none() { 1 } else { 2 };
            if graph.predecessors(block).nth(joins_at - 1).is_none() {
                continue;
            }

            // The walk stops at the immediate dominator of `block`. A tree
            // root has none, so the walk goes up to the root and adds it as
            // well: the root dominates the predecessor and does not strictly
            // dominate itself. A runner that an earlier predecessor's walk
            // reached for this block continues exactly as that walk did, so
            // the walk stops there and each pair is emitted once.
            for predecessor in graph
                .predecessors(block)
                .filter(|&predecessor| dom.is_reachable(predecessor))
            {
                let mut runner = predecessor;
                while Some(runner) != stop && self.starts[runner.index()] != block.index() {
                    self.starts[runner.index()] = block.index();
                    self.pairs.push((runner, block));
                    let Some(parent) = dom.idom(runner) else {
                        break;
                    };
                    runner = parent;
                }
            }
        }
    }

    /// Group the collected pairs by runner into the runs of a table over
    /// `bound` blocks.
    pub(super) fn group(&mut self, bound: usize) {
        // Group the pairs by runner without comparing them. `starts` first
        // holds the run lengths as running totals, so `starts[b]` is where
        // run `b` begins; the scatter then advances each cursor to the end of
        // its run, which is where the next run begins. The pairs of one block
        // come together, so the runs come out ascending when the walk met its
        // blocks in ascending order, as it does for every store here, and
        // only then.
        let mut ascending = true;
        let mut previous = 0;
        self.starts.clear();
        self.starts.resize(bound + 1, 0);
        for &(runner, block) in &self.pairs {
            self.starts[runner.index() + 1] += 1;
            ascending &= previous <= block.index();
            previous = block.index();
        }
        for index in 1..self.starts.len() {
            self.starts[index] += self.starts[index - 1];
        }
        self.blocks.clear();
        self.blocks.resize(self.pairs.len(), N::from_index(0));
        for &(runner, block) in &self.pairs {
            let cursor = &mut self.starts[runner.index()];
            self.blocks[*cursor] = block;
            *cursor += 1;
        }
        self.starts.rotate_right(1);
        self.starts[0] = 0;

        if !ascending {
            for run in self.starts.windows(2) {
                self.blocks[run[0]..run[1]].sort_unstable_by_key(|block| block.index());
            }
        }
    }
}

impl<N> Default for DominanceFrontiers<N> {
    /// The table of the empty graph, which allocates nothing. It holds no
    /// frontier until [`recompute`](DominanceFrontiers::recompute) fills it.
    fn default() -> Self {
        Self {
            pairs: Vec::new(),
            starts: Vec::new(),
            blocks: Vec::new(),
        }
    }
}
