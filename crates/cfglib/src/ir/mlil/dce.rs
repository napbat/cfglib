//! Dead-code marking.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::uses::UseIndex;
use crate::graph::view::DenseId;
use crate::ir::mlil::{AnalysisDialect, FunctionView, ValueDef};

/// The instructions that a pass can delete without changing what the
/// function does.
///
/// An instruction is dead when its operation is pure and no live code uses
/// any of its results. A use on a successor edge counts as a use. Removing
/// one dead instruction can make its operands dead, so the marking follows
/// the chains: it marks an instruction whose only users are dead
/// instructions. A pure instruction without results is dead.
///
/// The marking does not look through block parameters. A value that only
/// feeds an unused parameter stays alive, because it is used on an edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeadInstructions<I> {
    /// The dead instructions in ascending order.
    dead: Vec<I>,
}

impl<I: DenseId> DeadInstructions<I> {
    /// Marks the dead instructions of `function`. The call changes nothing.
    ///
    /// `uses` is the use index of `function`.
    ///
    /// # Panics
    ///
    /// Panics when `uses` was not built from `function`.
    #[must_use]
    pub fn compute<D, F>(function: &F, uses: &UseIndex<I, F::Block>) -> Self
    where
        D: AnalysisDialect,
        F: FunctionView<D, Inst = I>,
    {
        let mut remaining: Vec<usize> = (0..function.value_bound())
            .map(|index| uses.use_count(F::Value::from_index(index)))
            .collect();
        let mut dead = vec![false; function.inst_bound()];
        let mut work = Vec::new();

        let removable = |inst: I, remaining: &[usize]| {
            D::effects(function.op(inst)).is_pure()
                && function
                    .results(inst)
                    .iter()
                    .all(|result| remaining[result.index()] == 0)
        };
        for block in function.node_ids() {
            for &inst in function.insts(block) {
                if removable(inst, &remaining) {
                    dead[inst.index()] = true;
                    work.push(inst);
                }
            }
        }
        while let Some(inst) = work.pop() {
            for &operand in function.operands(inst) {
                remaining[operand.index()] -= 1;
                if remaining[operand.index()] != 0 {
                    continue;
                }
                if let ValueDef::Result { inst: producer, .. } = function.def(operand)
                    && !dead[producer.index()]
                    && removable(producer, &remaining)
                {
                    dead[producer.index()] = true;
                    work.push(producer);
                }
            }
        }

        let dead = (0..dead.len())
            .filter(|&index| dead[index])
            .map(I::from_index)
            .collect();
        Self { dead }
    }

    /// Returns the dead instructions in ascending order.
    #[must_use]
    pub fn as_slice(&self) -> &[I] {
        &self.dead
    }

    /// Returns whether `inst` is dead.
    #[must_use]
    pub fn contains(&self, inst: I) -> bool {
        self.dead.binary_search(&inst).is_ok()
    }

    /// Returns how many instructions are dead.
    #[must_use]
    pub fn len(&self) -> usize {
        self.dead.len()
    }

    /// Returns whether no instruction is dead.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dead.is_empty()
    }
}
