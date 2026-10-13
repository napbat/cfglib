//! The roots that each block reads, writes, and overwrites.

extern crate alloc;

use alloc::vec::Vec;

use super::roots::{RootTable, selection_roots, target_roots};
use crate::DenseBits;
use crate::graph::view::DenseId;
use crate::ir::rtl::{Action, Dialect, FunctionView, RecordView, Statement, WriteTarget};

/// The root sets of one block. A set holds dense indices of a [`RootTable`].
///
/// A root is a whole unit here: a read of any bit reads the root, and a write
/// of any bit writes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSummary {
    /// The roots that the block reads before it overwrites them.
    reads_before_write: DenseBits,
    /// The roots that the block can write.
    writes: DenseBits,
    /// The roots that the block overwrites on every execution.
    kills: DenseBits,
}

impl BlockSummary {
    fn new(root_count: usize) -> Self {
        Self {
            reads_before_write: DenseBits::new(root_count),
            writes: DenseBits::new(root_count),
            kills: DenseBits::new(root_count),
        }
    }

    /// Returns the upward-exposed roots: the roots whose value from before
    /// the block the block can read.
    #[must_use]
    pub const fn reads_before_write(&self) -> &DenseBits {
        &self.reads_before_write
    }

    /// Returns the roots that the block can write. This set includes the
    /// roots of every may-write and every kill.
    #[must_use]
    pub const fn writes(&self) -> &DenseBits {
        &self.writes
    }

    /// Returns the roots that the block overwrites completely on every
    /// execution, so that their value from before the block is gone.
    #[must_use]
    pub const fn kills(&self) -> &DenseBits {
        &self.kills
    }

    /// Records a read of root `index` at a point where `kills` holds the
    /// roots that earlier instructions overwrote.
    fn read(&mut self, index: usize) {
        if !self.kills.get(index) {
            self.reads_before_write.insert(index);
        }
    }

    /// Records the effect of one write statement.
    ///
    /// A kill goes to `pending` and takes effect after the instruction,
    /// because every read of the instruction sees the state before it.
    fn write<D: Dialect, V: Copy>(
        &mut self,
        roots: &RootTable<D>,
        target: WriteTarget<'_, D, V>,
        pending: &mut Vec<usize>,
    ) {
        let kills = match target {
            WriteTarget::Root { plan, .. } => !plan
                .iter()
                .any(|span| matches!(span.action, Action::PreservePreState)),
            WriteTarget::FiniteIndexed { .. } | WriteTarget::SparseIndexed { .. } => false,
        };
        for root in target_roots(target) {
            let index = roots.index_of_named(root);
            self.writes.insert(index);
            if kills {
                pending.push(index);
            } else {
                self.read(index);
            }
        }
    }
}

/// The summary of every block of a function.
///
/// The summary of a block follows from its instructions in order:
///
/// - A read of a root is upward-exposed when no earlier instruction of the
///   block kills the root. Every read of one instruction sees the state
///   before that instruction.
/// - A [`WriteTarget::Root`] write kills its root when its plan holds no
///   [`Action::PreservePreState`] span.
/// - A write that keeps some bits of its root reads the root and writes it.
///   A partial [`WriteTarget::Root`] write, every candidate of an indexed
///   write, and the bank of a sparse write are such roots. An indexed or a
///   sparse write is a may-write: it never kills.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSummaries {
    /// One summary per block identity below the node bound of the view.
    blocks: Vec<BlockSummary>,
}

impl BlockSummaries {
    /// Summarizes every live block of `function`.
    ///
    /// # Panics
    ///
    /// Panics when `roots` was not collected from `function`.
    #[must_use]
    pub fn compute<D: Dialect, F: FunctionView<D>>(function: &F, roots: &RootTable<D>) -> Self {
        let mut blocks = alloc::vec![BlockSummary::new(roots.len()); function.node_bound()];
        let mut pending = Vec::new();
        for block in function.node_ids() {
            let summary = &mut blocks[block.index()];
            for &inst in function.instructions(block) {
                pending.clear();
                let record = function.record(inst);
                for statement in record.statements() {
                    match statement {
                        Statement::Read { source, .. } => {
                            for root in selection_roots(source) {
                                summary.read(roots.index_of_named(root));
                            }
                        }
                        Statement::Write { target, .. } => {
                            summary.write(roots, target, &mut pending);
                        }
                        _ => {}
                    }
                }
                for &index in &pending {
                    summary.kills.insert(index);
                }
            }
        }
        Self { blocks }
    }

    /// Returns the summary of `block`. A removed block has an empty summary.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the summaries.
    #[must_use]
    pub fn block<B: DenseId>(&self, block: B) -> &BlockSummary {
        &self.blocks[block.index()]
    }
}
