//! The read-only view of one medium-level function.

use crate::graph::view::{DenseId, RootedView};

use super::dialect::Dialect;

/// Where a value is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValueDef<I, B> {
    /// Result `index` of instruction `inst`.
    Result {
        /// The defining instruction.
        inst: I,
        /// The position among the results of `inst`.
        index: u32,
    },
    /// Parameter `index` of block `block`.
    Param {
        /// The block that receives the value.
        block: B,
        /// The position among the parameters of `block`.
        index: u32,
    },
}

/// One successor record of a terminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Successor<'a, B, V, E> {
    /// The target block.
    pub target: B,
    /// The values passed to the parameters of `target`, in parameter order.
    pub args: &'a [V],
    /// The edge metadata.
    pub edge: &'a E,
}

/// A medium-level function: blocks of flat instructions in static single
/// assignment form.
///
/// The view is a [`RootedView`] over its blocks; the root is the entry block.
/// The adjacency of the graph view agrees with [`Self::successors`].
///
/// # Contract
///
/// - Each live block ends with exactly one terminator, which is the last
///   instruction of [`Self::insts`]; only a terminator has successors.
/// - Every value has exactly one definition, reported by [`Self::def`].
/// - The arguments of each successor record match the parameters of its
///   target in count and type.
/// - Identities below the bounds that the view does not yield are removed
///   entities.
pub trait FunctionView<D: Dialect>: RootedView<NodeId = Self::Block> {
    /// A block, dense within the consumer's storage.
    type Block: DenseId;
    /// An instruction, dense within the consumer's storage.
    type Inst: DenseId;
    /// A value, dense within the consumer's storage.
    type Value: DenseId;

    /// An exclusive upper bound on every instruction index.
    fn inst_bound(&self) -> usize;

    /// An exclusive upper bound on every value index.
    fn value_bound(&self) -> usize;

    /// Returns the parameters of `block` in order.
    fn params(&self, block: Self::Block) -> &[Self::Value];

    /// Returns the instructions of `block` in execution order. The last one
    /// is the terminator.
    fn insts(&self, block: Self::Block) -> &[Self::Inst];

    /// Returns the operation of `inst`.
    fn op(&self, inst: Self::Inst) -> &D::Op;

    /// Returns the operands of `inst` in operation order. Successor
    /// arguments are not operands; [`Self::successors`] reports them.
    fn operands(&self, inst: Self::Inst) -> &[Self::Value];

    /// Returns the results of `inst` in operation order.
    fn results(&self, inst: Self::Inst) -> &[Self::Value];

    /// Iterates over the successor records of the terminator of `block`, in
    /// terminator order.
    fn successors(
        &self,
        block: Self::Block,
    ) -> impl ExactSizeIterator<Item = Successor<'_, Self::Block, Self::Value, D::Edge>> + '_;

    /// Returns the block that holds `inst`.
    fn block_of(&self, inst: Self::Inst) -> Self::Block;

    /// Returns where `value` is defined.
    fn def(&self, value: Self::Value) -> ValueDef<Self::Inst, Self::Block>;

    /// Returns the type of `value`.
    fn ty(&self, value: Self::Value) -> &D::Type;
}
