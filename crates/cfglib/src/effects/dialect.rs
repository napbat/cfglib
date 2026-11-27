//! The knowledge about effects that one register-transfer dialect supplies.

use core::fmt::Debug;
use core::hash::Hash;

use crate::ir::rtl::{Bits, Dialect};
use crate::memory::AccessMode;

use super::vocabulary::StateTouch;

/// The memory access that one effect operation makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AccessShape<P> {
    /// Whether the operation reads, writes, or reads and writes memory, and
    /// whether the access is atomic.
    pub mode: AccessMode,
    /// The address space.
    pub space: P,
    /// The number of accessed bytes.
    pub bytes: u32,
    /// The position of the address among the operands.
    pub address: usize,
    /// The position of the one-bit guard among the operands of a conditional
    /// access, or `None` for an unconditional access.
    pub guard: Option<usize>,
}

/// The operands of a select operator: the result is `taken` when `guard` is
/// one, and `otherwise` when it is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SelectShape {
    /// The position of the one-bit guard among the operands.
    pub guard: usize,
    /// The position of the value for a guard of one.
    pub taken: usize,
    /// The position of the value for a guard of zero.
    pub otherwise: usize,
}

/// What one register-transfer dialect tells about the effects of its
/// operations.
///
/// This is the only knowledge of a dialect that [`Effect::compute`](super::Effect::compute)
/// and [`StatementTree::compute`](super::StatementTree::compute) use. A
/// generator or a hand-written table implements it beside the dialect.
///
/// The trait asks nothing that only one consumer knows. A reader that knows
/// the value of an operator without operands, such as a value that moves
/// with its module after the reader has placed the module, substitutes the
/// value in the computed expressions. A reader that treats the outcomes of
/// one effect apart finds them by the position of the effect (see
/// [`RaiseOrigin`](super::RaiseOrigin)).
pub trait DialectEffects<D: Dialect> {
    /// An address space of a memory access, such as the data space of a
    /// machine or the shared memory of a shader.
    type Space: Copy + Eq + Hash + Debug;

    /// Returns the memory access of `operation`, or `None` for an intrinsic
    /// that accesses no memory.
    ///
    /// `args` holds the widths of the operands and `results` the widths of
    /// the results, so a load can take its size from its result and a store
    /// from its value. The positions in the shape must be below
    /// `args.len()`.
    fn access(
        &self,
        operation: &D::EffectOp,
        args: &[Bits],
        results: &[Bits],
    ) -> Option<AccessShape<Self::Space>>;

    /// Returns the machine state that the intrinsic `operation` can change
    /// and that no operand or result shows.
    ///
    /// The default is [`StateTouch::Unrestricted`], which is always sound.
    fn intrinsic_state(&self, operation: &D::EffectOp) -> StateTouch {
        let _ = operation;
        StateTouch::Unrestricted
    }

    /// Returns whether `operator` selects one of two values, and where its
    /// operands are.
    ///
    /// A write whose value is a select of the old value of the written bits
    /// is a conditional write. The default is `None`: no operator selects.
    fn select(&self, operator: &D::Operator) -> Option<SelectShape> {
        let _ = operator;
        None
    }

    /// Returns the name of `operator`, for expressions.
    fn operator_name(&self, operator: &D::Operator) -> &'static str;
}
