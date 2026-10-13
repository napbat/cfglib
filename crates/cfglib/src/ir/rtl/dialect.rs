//! The vocabulary that one instruction set supplies to RTL.

use core::fmt::Debug;
use core::hash::Hash;

use super::bits::{BitSlice, Bits};

/// The vocabulary of one instruction set.
///
/// A generator normally implements this trait on a unit marker type. Every
/// associated type is plain data: an RTL record names a root, an operator, or
/// an outcome by value, so the vocabulary needs no runtime registry. The
/// marker bounds let generic RTL data derive its standard traits.
pub trait Dialect: Copy + Eq + Ord + Hash + Debug + 'static {
    /// A canonical storage root, such as a 64-bit general-purpose register,
    /// one flag, or one physical x87 data register.
    ///
    /// Roots never alias: a register view is a [`BitSlice`] of one root, and
    /// a packed view is several reads joined by operators.
    type Root: Copy + Eq + Ord + Hash + Debug;

    /// A pure operator.
    type Operator: Clone + Eq + Hash + Debug;

    /// An ordered effect: a load, a store, an atomic update, or an intrinsic
    /// with an observable effect.
    type EffectOp: Clone + Eq + Hash + Debug;

    /// A named write policy. The policy is provenance only: the write itself
    /// states its complete coverage of each root it changes.
    type WritePolicy: Copy + Eq + Hash + Debug;

    /// A kind of non-normal outcome, such as an architectural fault.
    type Outcome: Copy + Eq + Hash + Debug;

    /// The domain of a nondeterministic choice, such as the undefined flags of
    /// one instruction.
    type NondetDomain: Copy + Eq + Hash + Debug;

    /// Returns the width of a root.
    fn root_width(root: Self::Root) -> Bits;
}

/// A bit range of one storage root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Place<D: Dialect> {
    /// The root that holds the range.
    pub root: D::Root,
    /// The bits of the root.
    pub slice: BitSlice,
}

impl<D: Dialect> Place<D> {
    /// Returns the place that covers the whole of `root`.
    #[must_use]
    pub fn whole(root: D::Root) -> Self {
        Self {
            root,
            slice: BitSlice::whole(D::root_width(root)),
        }
    }
}
