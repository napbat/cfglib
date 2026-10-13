//! The use index: for each value, where the function uses it.

use super::csr::Csr;
use crate::graph::view::DenseId;
use crate::ir::mlil::{Dialect, FunctionView};

/// One use of a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Use<I, B> {
    /// The value is operand `position` of instruction `inst`.
    Operand {
        /// The instruction that reads the value.
        inst: I,
        /// The position among the operands of `inst`.
        position: u32,
    },
    /// The value is argument `position` of successor record `successor` of
    /// the terminator of `block`. The use is on the edge, at the end of
    /// `block`.
    Edge {
        /// The block whose terminator holds the successor record.
        block: B,
        /// The position among the successor records of the terminator.
        successor: u32,
        /// The position among the arguments of the successor record.
        position: u32,
    },
}

/// The uses of every value of a function, in one flat table.
///
/// The index is a fact about one state of the function. It is not a use
/// list that the function keeps: a consumer builds it when it needs it and
/// builds it again after an edit.
///
/// The uses of one value are in the order of the live blocks of the view.
/// Within a block they follow the instructions and their operands, and then
/// the successor records of the terminator and their arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UseIndex<I, B> {
    /// The uses of each value index.
    rows: Csr<Use<I, B>>,
}

impl<I: DenseId, B: DenseId> UseIndex<I, B> {
    /// Builds the use index of `function`.
    #[must_use]
    pub fn build<D, F>(function: &F) -> Self
    where
        D: Dialect,
        F: FunctionView<D, Inst = I, Block = B>,
    {
        let rows = Csr::build(function.value_bound(), |emit| {
            for block in function.node_ids() {
                for &inst in function.insts(block) {
                    for (position, operand) in function.operands(inst).iter().enumerate() {
                        emit(
                            operand.index(),
                            Use::Operand {
                                inst,
                                position: narrow(position),
                            },
                        );
                    }
                }
                let records = FunctionView::<D>::successors(function, block);
                for (successor, record) in records.enumerate() {
                    for (position, arg) in record.args.iter().enumerate() {
                        emit(
                            arg.index(),
                            Use::Edge {
                                block,
                                successor: narrow(successor),
                                position: narrow(position),
                            },
                        );
                    }
                }
            }
        });
        Self { rows }
    }

    /// Returns the uses of `value`.
    ///
    /// # Panics
    ///
    /// Panics when `value` is not below the value bound of the view that
    /// built the index.
    #[must_use]
    pub fn uses<V: DenseId>(&self, value: V) -> &[Use<I, B>] {
        self.rows.row(value.index())
    }

    /// Returns how many times the function uses `value`.
    ///
    /// # Panics
    ///
    /// Panics when `value` is not below the value bound of the view that
    /// built the index.
    #[must_use]
    pub fn use_count<V: DenseId>(&self, value: V) -> usize {
        self.uses(value).len()
    }
}

/// Narrows a position inside one instruction or terminator.
fn narrow(position: usize) -> u32 {
    u32::try_from(position).expect("a position inside one instruction fits in 32 bits")
}
