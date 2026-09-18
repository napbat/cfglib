//! Dominance frontiers, in the two shapes their two consumers want.
//!
//! One algorithm — the runner walk of Cooper, Harvey, and Kennedy — feeds
//! both: [`FrontierRuns`] is the flat run table SSA construction reads and a
//! scratch reuses, and [`DominanceFrontiers`] is the per-block set a consumer
//! asking for one block's frontier reads. The public type is built from the
//! flat one, so there is one statement of the rule rather than two.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::dominator::DominatorTree;

/// The dominance frontier of every block.
#[derive(Debug, Clone)]
pub struct DominanceFrontiers {
    /// `frontiers[b]` is the dominance-frontier set of `b`.
    frontiers: Vec<BTreeSet<BlockId>>,
}

impl DominanceFrontiers {
    /// Compute dominance frontiers using the algorithm from Cooper, Harvey,
    /// and Kennedy.
    #[must_use]
    pub fn compute<I, E>(cfg: &Cfg<I, E>, dom: &DominatorTree) -> Self {
        let mut runs = FrontierRuns::default();
        runs.rebuild(cfg, dom);
        let frontiers = (0..cfg.block_bound())
            .map(|index| {
                // Grown by insertion rather than collected: a set built from
                // a sorted iterator is bulk-loaded, which costs a second node
                // for the one-member frontiers most blocks have.
                let mut frontier = BTreeSet::new();
                frontier.extend(runs.frontier(BlockId::from_index(index)).iter().copied());
                frontier
            })
            .collect();
        Self { frontiers }
    }

    /// Return the dominance-frontier set for `block`.
    #[must_use]
    pub fn frontier(&self, block: BlockId) -> &BTreeSet<BlockId> {
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
#[derive(Debug, Clone, Default)]
pub(super) struct FrontierRuns {
    /// `(runner, frontier block)` pairs, sorted and deduplicated, which is
    /// what makes the runs contiguous and each one ascending.
    pairs: Vec<(BlockId, BlockId)>,
    /// Where each block's run starts in [`blocks`](Self::blocks). One longer
    /// than the block bound, so a run is `starts[b]..starts[b + 1]`.
    starts: Vec<usize>,
    /// The frontier members themselves, grouped by runner.
    blocks: Vec<BlockId>,
}

impl FrontierRuns {
    /// Recompute the table for `cfg` under `dom`, reusing every buffer.
    pub(super) fn rebuild<I, E>(&mut self, cfg: &Cfg<I, E>, dom: &DominatorTree) {
        self.pairs.clear();
        for block_id in cfg.block_ids() {
            if cfg.incoming(block_id).count() < 2 {
                continue;
            }

            let frontier_root = dom.idom(block_id).unwrap_or(block_id);
            for predecessor in cfg.predecessors(block_id) {
                let mut runner = predecessor;
                while runner != frontier_root {
                    self.pairs.push((runner, block_id));
                    let Some(parent) = dom.idom(runner) else {
                        break;
                    };
                    runner = parent;
                }
            }
        }
        self.pairs.sort_unstable();
        self.pairs.dedup();

        let bound = cfg.block_bound();
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
    pub(super) fn frontier(&self, block: BlockId) -> &[BlockId] {
        let start = self.starts[block.index()];
        let end = self.starts[block.index() + 1];
        &self.blocks[start..end]
    }
}
