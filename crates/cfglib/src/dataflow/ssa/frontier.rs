//! Dominance frontiers as one flat run table.
//!
//! The runner walk of Cooper, Harvey, and Kennedy collects `(runner, frontier
//! block)` pairs, and sorting them makes every block's frontier one
//! contiguous, ascending run. [`FrontierRuns`] is that table, which an SSA
//! scratch rebuilds in place for each procedure, and [`DominanceFrontiers`]
//! is the same table handed to a consumer that asks for one block's frontier.
//! There is one statement of the rule rather than two.
//!
//! Both read any [`RootedView`], so a [`Cfg`](crate::Cfg) and a
//! consumer-owned function view share the one algorithm.
//!
//! For every reachable block `x`, the frontier is the one that
//! [`DominatorTree::frontier`] states: block `y` is a member when `x`
//! dominates a predecessor of `y` and does not strictly dominate `y`. This
//! includes a tree root that is a branch target, such as an entry with a back
//! edge into it: the root is in its own frontier and in the frontier of every
//! block on the path back to it.

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
        // The pair list is the walk's scratch. A scratch keeps it for the
        // next procedure, but a table that is kept has no use for it.
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
/// A sorted pair list collapsed into runs: the table is one buffer of
/// members and one of run starts however many blocks there are, and a
/// frontier is read back as a slice in ascending order. The pair list is
/// what the walk sorts to get there, and a scratch keeps it between
/// procedures.
#[derive(Debug, Clone)]
pub(super) struct FrontierRuns<N = BlockId> {
    /// `(runner, frontier block)` pairs, sorted and deduplicated, which is
    /// what makes the runs contiguous and each one ascending.
    pairs: Vec<(N, N)>,
    /// Where each block's run starts in [`blocks`](Self::blocks). One longer
    /// than the block bound, so a run is `starts[b]..starts[b + 1]`.
    starts: Vec<usize>,
    /// The frontier members themselves, grouped by runner.
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
        self.pairs.clear();
        for block in graph.node_ids() {
            // A frontier block is a join: two incoming paths at least. A tree
            // root has one path that is not an edge, the start of its tree,
            // so one predecessor edge already makes it a join.
            let tree_root = dom.idom(block).is_none() && dom.is_reachable(block);
            let joins_at = if tree_root { 1 } else { 2 };
            if graph.predecessors(block).nth(joins_at - 1).is_none() {
                continue;
            }

            // The walk stops at the immediate dominator of `block`. A tree
            // root has none, so the walk goes up to the root and adds it as
            // well: the root dominates the predecessor and does not strictly
            // dominate itself.
            let stop = (!tree_root).then(|| dom.idom(block).unwrap_or(block));
            for predecessor in graph.predecessors(block) {
                let mut runner = predecessor;
                while Some(runner) != stop {
                    self.pairs.push((runner, block));
                    let Some(parent) = dom.idom(runner) else {
                        break;
                    };
                    runner = parent;
                }
            }
        }
        self.pairs.sort_unstable();
        self.pairs.dedup();

        let bound = graph.node_bound();
        self.starts.clear();
        self.starts.resize(bound + 1, 0);
        for &(runner, _) in &self.pairs {
            self.starts[runner.index() + 1] += 1;
        }
        for index in 1..self.starts.len() {
            self.starts[index] += self.starts[index - 1];
        }
        self.blocks.clear();
        self.blocks
            .extend(self.pairs.iter().map(|&(_, block)| block));
    }

    /// The dominance frontier of `block`, in ascending order.
    pub(super) fn frontier(&self, block: N) -> &[N] {
        let start = self.starts[block.index()];
        let end = self.starts[block.index() + 1];
        &self.blocks[start..end]
    }
}
