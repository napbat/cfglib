//! The shared shape of the effects of an instruction, a block, or a function.

extern crate alloc;

use alloc::vec::Vec;

use crate::ir::rtl::{Control, Place};
use crate::memory::AccessMode;

use super::exprs::{ExprId, Exprs};

/// The effects of one instruction, one block, or one function.
///
/// `S` is a storage root, such as a register root or a stack slot, and a
/// [`Place`] names bits of one root. `L` is a memory location, such as an
/// address space with an address expression. `O` is the kind of a non-normal
/// outcome. The vocabulary names no instruction set: every root, location,
/// and outcome is opaque here.
///
/// Every [`ExprId`] in the effect points into [`exprs`](Self::exprs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Effect<S, L, O> {
    /// The places whose values from before the code the code can read.
    ///
    /// [`Effect::compute`] lists each root once, with its read bits in
    /// ascending order and without overlap.
    pub reads: Vec<Place<S>>,
    /// Each write, with its kind, in the order of the code.
    pub writes: Vec<Write<S>>,
    /// Each memory access, in the order of the code.
    pub memory: Vec<Access<L>>,
    /// The outcomes that the code can raise, with their guards, in the order
    /// of the code.
    pub raises: Vec<Raised<O>>,
    /// How control leaves the code: the kind of its control boundary,
    /// without the values.
    pub exit: Control<()>,
    /// Whether the code makes a nondeterministic choice.
    pub nondeterministic: bool,
    /// The machine state that an intrinsic of the code can change and that no
    /// place names.
    pub intrinsic_state: StateTouch,
    /// The expressions that the guards and the addresses use.
    pub exprs: Exprs<S>,
}

/// Bits of one place that code writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Write<S> {
    /// The written bits.
    pub place: Place<S>,
    /// When and with what the bits change.
    pub kind: WriteKind,
}

/// When and with what a write changes its bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WriteKind {
    /// The code assigns a value to the bits.
    Always,
    /// The bits change only when the guard is one. Otherwise they keep their
    /// value from before the code.
    ///
    /// The write reads the place, and [`Effect::reads`] lists that read.
    Conditional(ExprId),
    /// The bits receive a nondeterministic value, such as a flag that the
    /// architecture leaves undefined.
    Undefined,
    /// The bits are one candidate of an indexed write: the code can write
    /// them, and it can keep them.
    Indexed,
}

/// One memory access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Access<L> {
    /// Whether the access reads, writes, or reads and writes the location,
    /// and whether it is atomic.
    pub mode: AccessMode,
    /// The accessed location.
    pub location: L,
    /// The number of accessed bytes.
    pub bytes: u32,
    /// The one-bit guard of a conditional access. The access happens only
    /// when the guard is one. `None` is an unconditional access.
    pub guard: Option<ExprId>,
}

/// A memory location as an address space and an address expression.
///
/// This is the location that [`Effect::compute`] gives. `P` is the address
/// space of the dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryAddress<P> {
    /// The address space.
    pub space: P,
    /// The address, as an expression of [`Effect::exprs`].
    pub address: ExprId,
}

/// One outcome that code can raise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Raised<O> {
    /// The kind of outcome.
    pub outcome: O,
    /// The one-bit guard of the outcome. See [`RaiseOrigin`] for the meaning
    /// of `None`.
    pub guard: Option<ExprId>,
    /// What raises the outcome.
    pub origin: RaiseOrigin,
}

/// What raises an outcome.
///
/// A reader that treats the outcomes of one effect apart, such as the check
/// that a dialect runs at the start of each instruction, finds them by the
/// position of the effect: [`Environment::effect`](Self::Environment) names
/// the effect that declares an outcome, and an [`Opaque::Effect`] in a guard
/// names the effect that a guard depends on.
///
/// [`Opaque::Effect`]: super::Opaque::Effect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseOrigin {
    /// The code raises the outcome when its guard is one. Without a guard,
    /// the code always raises it, and control does not continue.
    Raise,
    /// The environment can raise the outcome at an ordered effect, such as a
    /// page fault of a load. The guard is the guard of a conditional access.
    /// Without a guard, the environment can raise the outcome whenever the
    /// effect runs.
    Environment {
        /// The position of the effect among the effects of the code, from
        /// zero.
        effect: u32,
    },
}

/// The machine state that an intrinsic can change and that no place names.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateTouch {
    /// The intrinsic changes no state that its operands and results do not
    /// show.
    #[default]
    Preserved,
    /// The intrinsic can change any state.
    Unrestricted,
}

impl StateTouch {
    /// Returns the touch of code that runs both `self` and `other`.
    #[must_use]
    pub fn join(self, other: Self) -> Self {
        self.max(other)
    }
}
