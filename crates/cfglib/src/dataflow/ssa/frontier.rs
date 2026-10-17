//! Dominance frontiers as one flat run table.
//!
//! The runner walk of Cooper, Harvey, and Kennedy collects `(runner, frontier
//! block)` pairs, and grouping them by runner makes every block's frontier
//! one contiguous, ascending run. [`FrontierRuns`] is that table, which an
//! SSA scratch rebuilds in place for each procedure, and
//! [`DominanceFrontiers`] is the same table handed to a consumer that asks
//! for one block's frontier.
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

use crate::block::BlockId;
use crate::graph::dominator::DominatorTree;
use crate::graph::view::{DenseId, RootedView};

#[cfg(test)]
mod tests;

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
pub(super) struct FrontierRuns<N = BlockId> {
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
    pub(super) fn rebuild<G>(&mut self, graph: &G, dom: &DominatorTree<N>)
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
    pub(super) fn frontier(&self, block: N) -> &[N] {
        let start = self.starts[block.index()];
        let end = self.starts[block.index() + 1];
        &self.blocks[start..end]
    }
}
