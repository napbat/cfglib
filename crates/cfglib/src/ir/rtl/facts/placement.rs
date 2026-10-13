//! Block-parameter placement by pruned iterated dominance frontier.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::liveness::RootLiveness;
use super::roots::RootTable;
use super::summary::BlockSummaries;
use crate::dataflow::ssa::DominanceFrontiers;
use crate::graph::dominator::DominatorTree;
use crate::graph::view::DenseId;
use crate::ir::rtl::{Dialect, FunctionView};

/// The roots that need a block parameter at each block.
///
/// A root needs a parameter at a block when two definitions of the root can
/// meet there and the root is live at the start of the block. The first
/// condition is the iterated dominance frontier of the blocks that define the
/// root. The second condition prunes the parameters that nothing reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamPlacement<D: Dialect> {
    /// The roots of each block in ascending order, one list per block
    /// identity below the node bound of the view.
    blocks: Vec<Vec<D::Root>>,
    /// How many parameters the placement holds.
    param_count: usize,
}

impl<D: Dialect> ParamPlacement<D> {
    /// Places the block parameters of `function`.
    ///
    /// The blocks that define a root are the entry block, which defines every
    /// root as the state on entry, and each reachable block that can write
    /// the root. An unreachable block defines nothing. A parameter at a
    /// block also defines its root there, so the placement iterates over the
    /// dominance frontiers.
    ///
    /// `dominators` is the dominator tree of `function`, `frontiers` are its
    /// dominance frontiers, and `liveness` is the liveness that prunes the
    /// result.
    ///
    /// # Panics
    ///
    /// Panics when a fact was not computed from `function`.
    #[must_use]
    pub fn compute<F: FunctionView<D>>(
        function: &F,
        dominators: &DominatorTree<F::NodeId>,
        frontiers: &DominanceFrontiers<F::NodeId>,
        roots: &RootTable<D>,
        summaries: &BlockSummaries,
        liveness: &RootLiveness,
    ) -> Self {
        let bound = function.node_bound();
        let entry = function.root();
        let mut writers: Vec<(usize, F::NodeId)> = Vec::new();
        for block in function.node_ids() {
            if dominators.is_reachable(block) {
                writers.extend(
                    summaries
                        .block(block)
                        .writes()
                        .ones()
                        .map(|root| (root, block)),
                );
            }
        }
        writers.sort_unstable();

        let mut blocks = vec![Vec::new(); bound];
        let mut param_count = 0;
        // A mark holds the index of the root that last set it, so nothing
        // clears the marks between two roots.
        let mut queued = vec![usize::MAX; bound];
        let mut visited = vec![usize::MAX; bound];
        let mut work = Vec::new();
        let mut next_writer = 0;
        for root in 0..roots.len() {
            queued[entry.index()] = root;
            work.push(entry);
            while let Some(&(writer_root, writer)) = writers.get(next_writer)
                && writer_root == root
            {
                next_writer += 1;
                if queued[writer.index()] != root {
                    queued[writer.index()] = root;
                    work.push(writer);
                }
            }

            while let Some(block) = work.pop() {
                for &join in frontiers.frontier(block) {
                    if visited[join.index()] == root {
                        continue;
                    }
                    visited[join.index()] = root;
                    if liveness.live_in(join).get(root) {
                        blocks[join.index()].push(roots.root(root));
                        param_count += 1;
                    }
                    if queued[join.index()] != root {
                        queued[join.index()] = root;
                        work.push(join);
                    }
                }
            }
        }
        Self {
            blocks,
            param_count,
        }
    }

    /// Returns the roots that need a parameter at `block`, in ascending
    /// order.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the placement.
    #[must_use]
    pub fn roots_at<B: DenseId>(&self, block: B) -> &[D::Root] {
        &self.blocks[block.index()]
    }

    /// Returns how many parameters the placement holds over all blocks.
    #[must_use]
    pub const fn param_count(&self) -> usize {
        self.param_count
    }
}
