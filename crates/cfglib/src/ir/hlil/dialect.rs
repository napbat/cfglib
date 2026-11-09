//! The vocabulary of one structured dialect.

use core::fmt::Debug;
use core::hash::Hash;

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
