//! Dominance frontiers, in the two shapes their two consumers want.
//!
//! One algorithm — the runner walk of Cooper, Harvey, and Kennedy — feeds
//! both: [`FrontierRuns`] is the flat run table SSA construction reads and a
//! scratch reuses, and [`DominanceFrontiers`] is the per-block set a consumer
//! asking for one block's frontier reads. The public type is built from the
//! flat one, so there is one statement of the rule rather than two.
//!
//! Both read any [`RootedView`], so a [`Cfg`](crate::Cfg) and a
//! consumer-owned function view share the one algorithm.

extern crate alloc;

use alloc::collections::BTreeSet;
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
#[derive(Debug, Clone)]
pub struct DominanceFrontiers<N = BlockId> {
    /// `frontiers[b]` is the dominance-frontier set of `b`.
    frontiers: Vec<BTreeSet<N>>,
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
        let frontiers = (0..graph.node_bound())
            .map(|index| {
                // Grown by insertion rather than collected: a set built from
                // a sorted iterator is bulk-loaded, which costs a second node
                // for the one-member frontiers most blocks have.
                let mut frontier = BTreeSet::new();
                frontier.extend(runs.frontier(N::from_index(index)).iter().copied());
                frontier
            })
            .collect();
        Self { frontiers }
    }

    /// Return the dominance-frontier set for `block`.
    #[must_use]
    pub fn frontier(&self, block: N) -> &BTreeSet<N> {
        &self.frontiers[block.index()]
    }
}

/// Every block's dominance frontier in one flat run table.
///
/// A [`BTreeSet`] per block is one allocation per block that has a frontier
/// and a tree node per member, rebuilt for every procedure of a codebase.
/// The same answer is a sorted pair list collapsed into runs: three buffers
/// however many blocks there are, and a frontier read back as a slice in
/// ascending order, which is the order the set iterated in.
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
            // A frontier block is a join: it has two incoming edges at least.
            if graph.predecessors(block).nth(1).is_none() {
                continue;
            }

            let frontier_root = dom.idom(block).unwrap_or(block);
            for predecessor in graph.predecessors(block) {
                let mut runner = predecessor;
                while runner != frontier_root {
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
