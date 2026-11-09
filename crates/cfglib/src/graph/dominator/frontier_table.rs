//! Dominance frontiers as one flat run table.
//!
//! The runner walk of Cooper, Harvey, and Kennedy collects `(runner, frontier
//! block)` pairs, and grouping them by runner makes every block's frontier
//! one contiguous, ascending run. [`FrontierRuns`] is that table, and
//! [`DominanceFrontiers`] hands it to a consumer that asks for one block's
//! frontier, or for the iterated frontier of a set of blocks.
//!
//! Both read any [`RootedView`], so a [`Cfg`](crate::Cfg) and a
//! consumer-owned function view share the one algorithm.
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

use super::frontier::FrontierScratch;
use super::word;
use crate::block::BlockId;
use crate::graph::dominator::DominatorTree;
use crate::graph::view::{DenseId, RootedView};

/// The dominance frontier of every block.
///
/// The block type `N` is the node identity of the view the frontiers were
/// computed from. It defaults to [`BlockId`], the identity of a
/// [`Cfg`](crate::Cfg).
///
/// For a reachable block, the frontier holds the members of
/// [`DominatorTree::frontier`] of that block, in ascending order. This type
/// computes every frontier at once and keeps them in one flat table, which
/// is two buffers however many blocks there are.
#[derive(Debug, Clone)]
pub struct DominanceFrontiers<N = BlockId> {
    /// Every block's frontier, flat.
    runs: FrontierRuns<N>,
}

impl<N: DenseId> DominanceFrontiers<N> {
    /// Compute dominance frontiers using the algorithm from Cooper, Harvey,
    /// and Kennedy.
    ///
    /// `dom` is the dominator tree of `graph`. The predecessors of `graph`
    /// are what the algorithm walks, so a view whose adjacency breaks the
    /// [`GraphView`](crate::graph::view::GraphView) contract gives an
    /// unspecified answer.
    #[must_use]
    pub fn compute<G>(graph: &G, dom: &DominatorTree<N>) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        let mut runs = FrontierRuns::default();
        runs.rebuild(graph, dom);
        // A scratch keeps the pair list for the next procedure, but a table
        // that is kept has no use for it.
        runs.pairs = Vec::new();
        Self { runs }
    }

    /// Recompute the table for `graph` under `dom` in the storage of this
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
    pub fn recompute<G>(&mut self, graph: &G, dom: &DominatorTree<N>)
    where
        G: RootedView<NodeId = N>,
    {
        self.runs.rebuild(graph, dom);
    }

    /// Return the dominance frontier of `block`, in ascending order.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the frontiers.
    #[must_use]
    pub fn frontier(&self, block: N) -> &[N] {
        self.runs.frontier(block)
    }

    /// Collect the iterated dominance frontier of `defs` into `out`, read off
    /// this table.
    ///
    /// The answer is the one [`DominatorTree::iterated_frontier`] gives for
    /// the tree and graph that computed the table, with the same `live_in`
    /// rule: a node is added, and its own frontier explored, only when
    /// `live_in` accepts it, a definition is a root of the search whether
    /// `live_in` accepts it or not, and each node is offered to `live_in` at
    /// most once per call. A definition outside every tree has an empty
    /// frontier, so it adds nothing. `out` is cleared first and receives each
    /// member once, in ascending order.
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
    /// use cfglib::{Cfg, DominanceFrontiers, DominatorTree, EdgeKind, FrontierScratch};
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
    /// let mut scratch = FrontierScratch::new();
    /// let mut placed = Vec::new();
    /// frontiers.iterated_frontier(&mut scratch, [entry, body], None, &mut placed);
    /// assert_eq!(placed, [head]);
    ///
    /// // Pruned by liveness: the variable is dead on entry to the header.
    /// let live_in = |block| block != head;
    /// frontiers.iterated_frontier(&mut scratch, [entry, body], Some(&live_in), &mut placed);
    /// assert!(placed.is_empty());
    /// ```
    pub fn iterated_frontier(
        &self,
        scratch: &mut FrontierScratch,
        defs: impl IntoIterator<Item = N>,
        live_in: Option<&dyn Fn(N) -> bool>,
        out: &mut Vec<N>,
    ) {
        out.clear();
        scratch.begin(self.runs.starts.len().saturating_sub(1));
        for def in defs {
            let index = def.index();
            if !scratch.queued.is_marked(index) {
                scratch.queued.mark(index);
                scratch.pending.push(word(index));
            }
        }
        while let Some(node) = scratch.pending.pop() {
            for &member in self.frontier(N::from_index(node as usize)) {
                let index = member.index();
                if scratch.judged.is_marked(index) {
                    continue;
                }
                scratch.judged.mark(index);
                if live_in.is_some_and(|accepts| !accepts(member)) {
                    continue;
                }
                out.push(member);
                if !scratch.queued.is_marked(index) {
                    scratch.queued.mark(index);
                    scratch.pending.push(word(index));
                }
            }
        }
        out.sort_unstable();
    }
}

impl<N> Default for DominanceFrontiers<N> {
    /// The table of the empty graph, which allocates nothing. It holds no
    /// frontier until [`recompute`](DominanceFrontiers::recompute) fills it.
    fn default() -> Self {
        Self {
            runs: FrontierRuns::default(),
        }
    }
}

/// Every block's dominance frontier in one flat run table.
///
/// The runner walk emits each `(runner, frontier block)` pair once, and the
/// table groups them by runner with a counting pass instead of a sort: the
/// table is one buffer of members and one of run starts however many blocks
/// there are, and a frontier is read back as a slice in ascending order. The
/// pair list is the walk's own scratch, and a scratch keeps it between
/// procedures.
#[derive(Debug, Clone)]
pub(crate) struct FrontierRuns<N = BlockId> {
    /// The `(runner, frontier block)` pairs of the last walk, in the order
    /// the walk emitted them.
    pairs: Vec<(N, N)>,
    /// Where each block's run starts in [`blocks`](Self::blocks). One longer
    /// than the block bound, so a run is `starts[b]..starts[b + 1]`.
    starts: Vec<usize>,
    /// The frontier members themselves, grouped by runner, each run strictly
    /// ascending.
    blocks: Vec<N>,
}

impl<N> Default for FrontierRuns<N> {
    fn default() -> Self {
        Self {
            pairs: Vec::new(),
            starts: Vec::new(),
            blocks: Vec::new(),
        }
    }
}

impl<N: DenseId> FrontierRuns<N> {
    /// Recompute the table for `graph` under `dom`, reusing every buffer.
    pub(crate) fn rebuild<G>(&mut self, graph: &G, dom: &DominatorTree<N>)
    where
        G: RootedView<NodeId = N>,
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

        // Group the pairs by runner without comparing them. `starts` first
        // holds the run lengths as running totals, so `starts[b]` is where
        // run `b` begins; the scatter then advances each cursor to the end of
        // its run, which is where the next run begins. The pairs of one block
        // come together, so the runs come out ascending when the walk met its
        // blocks in ascending order, as it does for every store here, and
        // only then.
        let mut ascending = true;
        let mut previous = None;
        self.starts.fill(0);
        for &(runner, block) in &self.pairs {
            self.starts[runner.index() + 1] += 1;
            ascending &= previous <= Some(block);
            previous = Some(block);
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
                self.blocks[run[0]..run[1]].sort_unstable();
            }
        }
    }

    /// The dominance frontier of `block`, in ascending order.
    pub(crate) fn frontier(&self, block: N) -> &[N] {
        let start = self.starts[block.index()];
        let end = self.starts[block.index() + 1];
        &self.blocks[start..end]
    }
}
