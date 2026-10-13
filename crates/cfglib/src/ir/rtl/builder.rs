//! The write-only protocol through which instruction semantics emit RTL.

use crate::graph::view::DenseId;

use super::bits::Bits;
use super::dialect::Dialect;
use super::statement::{Control, Raise, Selection, ValueRange, WriteTarget};

/// Emits the record of one instruction into consumer-owned storage.
///
/// The producer calls the methods in statement order and ends with
/// [`finish`](Self::finish). Each call appends one statement; the protocol
/// offers no way to read a statement back or to change it. A producer that
/// simplifies its output does so in its own scratch space before it calls the
/// builder.
///
/// A value handle stands for its expression evaluated where the handle was
/// created. Every read observes the state before the instruction, and writes
/// take effect only at the instruction boundary, so a value that a later
/// statement needs is passed by its handle instead of being read back.
///
/// # Contract
///
/// - Every operand is a handle that this builder returned earlier.
/// - A [`WriteTarget::Root`] plan covers its whole root (see
///   [`WriteTarget`]); two writes of one instruction never cover the same bit.
/// - A one-bit guard or condition has type [`Bits::BOOL`].
/// - An unconditional [`Raise`] is followed by [`Control::Terminating`].
///
/// The consumer checks the contract when the record is sealed and reports a
/// violation through [`Self::Error`].
pub trait InstructionBuilder<D: Dialect> {
    /// A handle to one value of the record, dense within the consumer's
    /// storage.
    type Value: DenseId;

    /// Why the consumer refused to seal the record.
    type Error;

    /// Reads bits of the state before the instruction.
    fn read(&mut self, source: Selection<'_, D, Self::Value>) -> Self::Value;

    /// Creates a constant from little-endian 64-bit words.
    ///
    /// `words` holds [`Bits::words`] words; bits past the width are zero.
    fn constant(&mut self, words: &[u64], width: Bits) -> Self::Value;

    /// Applies a pure operator and returns its results in operator order.
    fn apply(
        &mut self,
        operator: &D::Operator,
        args: &[Self::Value],
        results: &[Bits],
    ) -> ValueRange<Self::Value>;

    /// Creates one fresh nondeterministic choice.
    fn nondet(&mut self, width: Bits, domain: D::NondetDomain) -> Self::Value;

    /// Performs an ordered effect and returns its results in effect order.
    ///
    /// `outcomes` lists the outcomes that the environment can raise at this
    /// effect, such as a page fault of a load.
    fn effect(
        &mut self,
        operation: &D::EffectOp,
        args: &[Self::Value],
        results: &[Bits],
        outcomes: &[D::Outcome],
    ) -> ValueRange<Self::Value>;

    /// Raises an outcome when its guard is one, or unconditionally.
    fn raise(&mut self, raise: Raise<'_, D, Self::Value>);

    /// Adds a write to the delta that commits at the instruction boundary.
    fn write(&mut self, target: WriteTarget<'_, D, Self::Value>, policy: D::WritePolicy);

    /// Ends the instruction with its control boundary and seals the record.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the record breaks the contract of this
    /// trait or a rule of the consumer, such as a control boundary that
    /// disagrees with the decoded flow of the instruction.
    fn finish(self, control: Control<Self::Value>) -> Result<(), Self::Error>;
}
