//! Read-only views over sealed RTL records.

use crate::graph::view::{DenseId, RootedView};

use super::bits::Bits;
use super::dialect::Dialect;
use super::statement::{Control, Statement};

/// One sealed instruction record.
pub trait RecordView<D: Dialect> {
    /// A handle to one value of the record.
    type Value: DenseId;

    /// Iterates over the statements in the order that they were built.
    fn statements(&self) -> impl ExactSizeIterator<Item = Statement<'_, D, Self::Value>> + '_;

    /// Returns the control boundary that ends the instruction.
    fn control(&self) -> Control<Self::Value>;

    /// Returns the type of a value of this record.
    fn width(&self, value: Self::Value) -> Bits;
}

/// A function as a graph of blocks of instructions, each with a sealed
/// record.
///
/// The graph is the decoded machine control flow. The root of the view is the
/// entry block.
pub trait FunctionView<D: Dialect>: RootedView {
    /// An instruction of the function, dense within the consumer's storage.
    type Inst: DenseId;

    /// A value of any record of the function, dense within the consumer's
    /// storage.
    type Value: DenseId;

    /// The view of one record.
    type Record<'a>: RecordView<D, Value = Self::Value>
    where
        Self: 'a;

    /// An exclusive upper bound on every instruction index.
    fn inst_bound(&self) -> usize;

    /// An exclusive upper bound on every value index.
    fn value_bound(&self) -> usize;

    /// Returns the instructions of `block` in execution order.
    fn instructions(&self, block: Self::NodeId) -> &[Self::Inst];

    /// Returns the record of `inst`.
    fn record(&self, inst: Self::Inst) -> Self::Record<'_>;
}
