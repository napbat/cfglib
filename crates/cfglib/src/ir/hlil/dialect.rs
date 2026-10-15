//! The vocabulary of one structured dialect and the hooks that read it.

extern crate alloc;

use alloc::vec::Vec;
use core::fmt::{self, Debug};
use core::hash::Hash;

use super::verify::VerificationIssue;
use super::view::FunctionView;

/// The vocabulary of one structured dialect.
///
/// The statement shapes (assignments, conditionals, loops, switches,
/// exception regions) are universal and owned by the library; see
/// [`StmtShape`](super::StmtShape). Everything semantic is the consumer's:
/// the operations that expressions apply, the constants, the value types, and
/// the metadata of a variable. Every associated type is plain data that the
/// consumer's storage holds.
///
/// A marker implements this trait next to [`mlil::Dialect`] when one
/// consumer owns both levels. The two traits both name an `Op`, so such a
/// marker qualifies the name: `<Marker as hlil::Dialect>::Op`.
///
/// [`mlil::Dialect`]: crate::ir::mlil::Dialect
pub trait Dialect: Copy + Eq + Ord + Hash + Debug + 'static {
    /// An operation applied to ordered operand expressions: calls, loads,
    /// arithmetic, and the place operations (dereference, field, element)
    /// that an assignment can target. A [`Region`] statement carries one too.
    ///
    /// [`Region`]: super::StmtShape::Region
    type Op: Clone + Debug + Eq;

    /// The constant domain of literal expressions and of switch-case values.
    type Constant: Clone + Debug + Eq;

    /// The type of one expression occurrence. A variable holds differently
    /// typed values at different points, so each occurrence has its own type.
    type Type: Clone + Debug + Eq + Hash;

    /// The metadata of one declared variable: the consumer's role and native
    /// identity of it.
    type Variable: Clone + Debug + Eq;
}

/// Dialect-specific semantic and typing verification.
///
/// [`verify`](fn@super::verify) runs the structural checks first and then calls
/// [`Self::verify`], so a dialect only states what its own vocabulary requires.
pub trait VerifyDialect: Dialect {
    /// Appends every dialect invariant violation to `issues`, in
    /// deterministic order.
    ///
    /// Every handle that `function` yields is in range when this runs, so an
    /// implementation can index with it. The structure can still be invalid
    /// in other ways: a node can have no parent or several.
    fn verify<F: FunctionView<Self>>(function: &F, issues: &mut Vec<VerificationIssue>);
}

/// The spelling hooks of the built-in pseudocode writer.
pub trait Pseudocode: Dialect {
    /// Returns the compact stable mnemonic of an operation.
    fn mnemonic(op: &Self::Op) -> &str;

    /// Writes one constant.
    ///
    /// The default writes the `Debug` form of the constant. A dialect
    /// overrides it for presentation-quality literals.
    ///
    /// # Errors
    ///
    /// Propagates the formatter's error.
    fn fmt_constant(formatter: &mut fmt::Formatter<'_>, constant: &Self::Constant) -> fmt::Result {
        write!(formatter, "{constant:?}")
    }
}
