//! The low-level contract: one instruction in the literal form of its
//! instruction set, with its semantics.
//!
//! LLIL is an edit surface. An edit changes an instruction: its form and its
//! operands. The consumer encodes the edited instruction to bytes, decodes it
//! again, and makes the levels above it again. LLIL has no storage of its own:
//! the bytes of the image hold it.
//!
//! - [`Instruction`] is what one instruction tells: its address, its length,
//!   its bytes, and its [`Flow`](crate::Flow) through the
//!   [`AddressInstruction`] supertrait.
//! - [`Semantics`] emits the register-transfer record of one instruction
//!   through an [`InstructionBuilder`] for one [`rtl::Dialect`], or captures it
//!   as owned data.
//!
//! The lift from a native decoding and the lowering back to bytes need the
//! architecture of a disassembler, so they are not part of this contract.

use core::convert::Infallible;
use core::fmt;

use crate::AddressInstruction;
use crate::effects::Captured;
use crate::ir::rtl::{self, InstructionBuilder};

/// One instruction in the literal form of its instruction set.
///
/// [`AddressInstruction`] gives the address and the [`Flow`](crate::Flow).
/// This trait adds the encoding. The operation and the encoding are separate
/// facts: two spellings of one operation are equal operations and different
/// encodings.
pub trait Instruction: AddressInstruction {
    /// The length of the encoding in bytes.
    fn length(&self) -> usize;

    /// The encoding of the instruction.
    fn bytes(&self) -> &[u8];
}

/// Why one instruction had no register-transfer record.
///
/// `E` is the error of the builder that the consumer supplied. It reports
/// that the consumer refused to seal the record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TranslateError<E> {
    /// The instruction set does not model the semantics of this instruction.
    ///
    /// A producer can be incomplete. This variant says so, instead of an
    /// emission that is not what the machine does.
    Unsupported,
    /// The operands of the instruction are not a shape that the translation
    /// admits. An example is a memory destination where the encoding
    /// promised a register.
    MalformedOperands,
    /// The instruction names storage that the dialect has no root for.
    UnmodeledStorage,
    /// The operand widths of the instruction do not agree with each other.
    WidthMismatch,
    /// The consumer refused to seal the record.
    ///
    /// The builder reports the violation of the record contract, or of a
    /// rule of the consumer, when the semantics call
    /// [`finish`](InstructionBuilder::finish).
    Seal(E),
}

impl<E: fmt::Display> fmt::Display for TranslateError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => formatter.write_str("no modeled semantics for the instruction"),
            Self::MalformedOperands => {
                formatter.write_str("operands are not a shape the translation admits")
            }
            Self::UnmodeledStorage => {
                formatter.write_str("instruction names storage the dialect cannot express")
            }
            Self::WidthMismatch => formatter.write_str("operand widths disagree"),
            Self::Seal(cause) => write!(formatter, "the record was not sealed: {cause}"),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for TranslateError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Seal(cause) => Some(cause),
            _ => None,
        }
    }
}

/// The semantics of the instructions of one instruction set.
///
/// The semantics own no storage. They emit each instruction through the
/// [`InstructionBuilder`] of the consumer, and the consumer decides where the
/// record lives. The semantics can hold the context of a decoding, so they are
/// a value beside the instruction, not a method of it.
pub trait Semantics {
    /// The instruction that the semantics translate.
    type Instruction: Instruction;

    /// The register-transfer dialect of the records.
    type Rtl: rtl::Dialect;

    /// Emits the record of one instruction and seals it.
    ///
    /// One instruction is one atomic state update. Every read observes the
    /// state before the instruction, and the writes take effect together at
    /// its end. The semantics call the methods of `builder` in statement
    /// order and end with [`finish`](InstructionBuilder::finish), which names
    /// the one control boundary of the instruction. The control boundary must
    /// agree with the [`Flow`](crate::Flow) of the instruction. The consumer
    /// checks that agreement when it seals the record.
    ///
    /// # Errors
    ///
    /// Returns [`TranslateError`] when the instruction set has no faithful
    /// translation, or when the consumer refuses to seal the record. After
    /// an error the builder is dropped without `finish`, and the consumer
    /// discards the partial record.
    fn translate<B: InstructionBuilder<Self::Rtl>>(
        &self,
        instruction: &Self::Instruction,
        builder: B,
    ) -> Result<(), TranslateError<B::Error>>;

    /// Captures the record of one instruction into `captured`, as owned data
    /// with no consumer storage.
    ///
    /// The capture holds the statements that [`translate`](Self::translate)
    /// emits, in order, with their control boundary. `captured` keeps its
    /// buffers, so a caller that captures many instructions allocates once.
    /// The default runs `translate` into [`Captured::begin`].
    ///
    /// # Errors
    ///
    /// Returns [`TranslateError`] as [`translate`](Self::translate) does,
    /// except `Seal`: the capture refuses no record. After an error the
    /// capture is incomplete.
    fn capture(
        &self,
        instruction: &Self::Instruction,
        captured: &mut Captured<Self::Rtl>,
    ) -> Result<(), TranslateError<Infallible>> {
        self.translate(instruction, captured.begin())
    }
}

#[cfg(test)]
mod tests {
    use core::error::Error as _;
    use core::fmt;
    use std::string::ToString;

    use super::TranslateError;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Refused;

    impl fmt::Display for Refused {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("refused")
        }
    }

    impl core::error::Error for Refused {}

    #[test]
    fn translation_failures_explain_themselves() {
        assert_eq!(
            TranslateError::<Refused>::Unsupported.to_string(),
            "no modeled semantics for the instruction"
        );
        assert_ne!(
            TranslateError::<Refused>::MalformedOperands,
            TranslateError::WidthMismatch
        );
    }

    #[test]
    fn a_refused_seal_names_the_consumer_error_as_its_source() {
        let error = TranslateError::Seal(Refused);
        assert_eq!(error.to_string(), "the record was not sealed: refused");
        assert_eq!(
            error.source().map(ToString::to_string),
            Some("refused".to_string())
        );
        assert!(TranslateError::<Refused>::Unsupported.source().is_none());
    }
}
