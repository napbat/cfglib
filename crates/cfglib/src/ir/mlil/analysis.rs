//! The semantic hooks that the generic medium-level analyses need.

use core::fmt::Debug;

use super::dialect::Dialect;

/// What executing one operation can do besides defining its results.
///
/// An operation with no flag set is pure: removing it when no result is used,
/// or merging two equal ones, changes nothing that can be observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag is one independent property of an operation"
)]
pub struct Effects {
    /// The operation must keep its order relative to other ordered
    /// operations, for example a load, a store, or a call.
    pub ordered: bool,
    /// The operation can end execution with a non-normal outcome.
    pub may_trap: bool,
    /// Each execution makes a fresh nondeterministic choice, so two equal
    /// operations are never merged and one is never duplicated.
    pub nondeterministic: bool,
    /// The operation transfers control. Only a terminator sets this flag.
    pub terminator: bool,
}

impl Effects {
    /// Returns whether the operation is pure.
    #[must_use]
    pub const fn is_pure(self) -> bool {
        !(self.ordered || self.may_trap || self.nondeterministic || self.terminator)
    }
}

/// The value lattice of sparse constant propagation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Lattice<C> {
    /// No definition has been seen yet.
    Undefined,
    /// Every execution produces this constant.
    Constant(C),
    /// The value can vary.
    Varying,
}

/// The semantics of a dialect that the generic analyses consult.
pub trait AnalysisDialect: Dialect {
    /// A constant of the dialect, which may describe only some bits.
    type Constant: Clone + Eq + Debug;

    /// Returns what `op` can do besides defining its results.
    fn effects(op: &Self::Op) -> Effects;

    /// Evaluates `op` over the lattice values of its operands.
    ///
    /// `results` has one slot per result of the operation and enters as
    /// [`Lattice::Varying`]. A dialect that cannot evaluate `op` leaves the
    /// slots unchanged.
    fn fold(
        op: &Self::Op,
        args: &[Lattice<Self::Constant>],
        results: &mut [Lattice<Self::Constant>],
    );

    /// Decides which successor record of the terminator `op` is taken.
    ///
    /// Returns the position of the taken successor among `successors`
    /// records, or `None` when the operands do not decide it.
    fn fold_branch(
        op: &Self::Op,
        args: &[Lattice<Self::Constant>],
        successors: usize,
    ) -> Option<usize>;

    /// Returns the operand position that `op` copies unchanged into its only
    /// result, or `None` when `op` is not a copy.
    fn copy_of(op: &Self::Op) -> Option<usize>;
}
