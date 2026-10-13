//! Constant evaluation of instruction-set operators.

use super::bits::Bits;
use super::dialect::Dialect;

/// A known constant operand: little-endian 64-bit words and a type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstBits<'a> {
    /// The words of the value. Bits past the width are zero.
    pub words: &'a [u64],
    /// The type of the value.
    pub width: Bits,
}

/// Evaluates the operators of a dialect over constant operands.
///
/// RTL analyses, such as the resolution of an indexed selection, call this
/// hook. A generator normally derives it from the evaluator of the instruction
/// set.
pub trait Fold: Dialect {
    /// Evaluates `operator` over `args`.
    ///
    /// On success the results are written to `out` in operator order; result
    /// `i` takes [`Bits::words`] words of `results[i]`, and bits past each
    /// width are zero. Returns `false`, leaving `out` unspecified, when the
    /// operator has no constant evaluation for these operands.
    fn fold(
        operator: &Self::Operator,
        args: &[ConstBits<'_>],
        results: &[Bits],
        out: &mut [u64],
    ) -> bool;
}
