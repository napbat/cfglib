//! Whole-root liveness.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::roots::RootTable;
use super::summary::BlockSummaries;
use crate::DenseBits;
use crate::graph::traverse::{TraversalDirection, depth_first_postorder};
use crate::graph::view::DenseId;
use crate::ir::rtl::{Dialect, FunctionView};

/// The roots that are live after one exit block, as the caller states them.
///
/// The liveness solver gives one seed to the closure of
/// [`RootLiveness::compute_seeded`] for every exit block.
#[derive(Debug)]
pub struct ExitSeed<'a, D: Dialect> {
    /// The table that maps roots to dense indices.
    roots: &'a RootTable<D>,
    /// The live-out set of the exit block.
    live: &'a mut DenseBits,
}

impl<D: Dialect> ExitSeed<'_, D> {
    /// Marks `root` live after the exit block.
    ///
    /// Returns whether the root was not live before. A root that the function
    /// does not name is never live, so the call ignores it and returns
    /// `false`.
    pub fn insert(&mut self, root: D::Root) -> bool {
        self.roots
            .index_of(root)
            .is_some_and(|index| self.live.insert(index))
    }

    /// Marks every root of the function live after the exit block.
    pub fn insert_all(&mut self) {
        for index in 0..self.roots.len() {
            self.live.insert(index);
        }
    }
}

/// Which roots are live at the start and at the end of each block.
///
/// Liveness is whole-root: a root is live when some path reads any bit of
/// it before it is overwritten completely. Precision below a root is the work
/// of a later analysis. A set holds dense indices of the [`RootTable`] that
/// the facts were computed from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootLiveness {
    /// The roots live at the start of each block.
    live_in: Vec<DenseBits>,
    /// The roots live at the end of each block.
    live_out: Vec<DenseBits>,
}

impl RootLiveness {
    /// Solves liveness with every root of the function live after every
    /// exit block.
    ///
    /// This is the safe seed for a function whose caller can read any root
    /// that the function names, which is the set T(F).
    ///
    /// # Panics
    ///
    /// Panics when `roots` or `summaries` was not computed from `function`.
    #[must_use]
    pub fn compute<D: Dialect, F: FunctionView<D>>(
        function: &F,
        roots: &RootTable<D>,
        summaries: &BlockSummaries,
    ) -> Self {
        Self::compute_seeded(function, roots, summaries, |_, seed| seed.insert_all())
    }

    /// Solves liveness with the roots that `seed` states live after each
    /// exit block.
    ///
    /// An exit block is a block without successors. The solver calls `seed`
    /// once for each exit block, and the seed starts empty.
    ///
    /// # Panics
    ///
    /// Panics when `roots` or `summaries` was not computed from `function`.
    #[must_use]
    pub fn compute_seeded<D, F, S>(
        function: &F,
        roots: &RootTable<D>,
        summaries: &BlockSummaries,
        mut seed: S,
    ) -> Self
    where
        D: Dialect,
        F: FunctionView<D>,
        S: FnMut(F::NodeId, &mut ExitSeed<'_, D>),
    {
        let bound = function.node_bound();
        let empty = DenseBits::new(roots.len());
        let mut live_in = vec![empty.clone(); bound];
        let mut live_out = vec![empty; bound];
        for block in function.node_ids() {
            if function.successors(block).next().is_none() {
                let mut exit = ExitSeed {
                    roots,
                    live: &mut live_out[block.index()],
                };
                seed(block, &mut exit);
            }
        }

        let mut queued = vec![false; bound];
        let mut work = Vec::new();
        for block in solving_order(function).into_iter().rev() {
            queued[block.index()] = true;
            work.push(block);
        }
        while let Some(block) = work.pop() {
            let index = block.index();
            queued[index] = false;
            for successor in function.successors(block) {
                live_out[index].union_with(&live_in[successor.index()]);
            }
            let summary = summaries.block(block);
            let row = &mut live_in[index];
            let mut changed = row.union_with(summary.reads_before_write());
            for root in live_out[index].ones() {
                if !summary.kills().get(root) {
                    changed |= row.insert(root);
                }
            }
            if changed {
                for predecessor in function.predecessors(block) {
                    if !queued[predecessor.index()] {
                        queued[predecessor.index()] = true;
                        work.push(predecessor);
                    }
                }
            }
        }
        Self { live_in, live_out }
    }

    /// Returns the roots live at the start of `block`, as dense indices.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the facts.
    #[must_use]
    pub fn live_in<B: DenseId>(&self, block: B) -> &DenseBits {
        &self.live_in[block.index()]
    }

    /// Returns the roots live at the end of `block`, as dense indices.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the facts.
    #[must_use]
    pub fn live_out<B: DenseId>(&self, block: B) -> &DenseBits {
        &self.live_out[block.index()]
    }
}

/// Returns every live block, the blocks reachable from the root first and in
/// postorder, so a backward solve meets the successors of a block first.
fn solving_order<D: Dialect, F: FunctionView<D>>(function: &F) -> Vec<F::NodeId> {
    let mut order = depth_first_postorder(function, function.root(), TraversalDirection::Outgoing);
    let mut reached = vec![false; function.node_bound()];
    for block in &order {
        reached[block.index()] = true;
    }
    order.extend(function.node_ids().filter(|block| !reached[block.index()]));
    order
}
