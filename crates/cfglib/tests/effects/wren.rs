//! Wren: a toy dialect and its effect knowledge.

use cfglib::ir::rtl::{
    Action, BitSlice, Bits, Control, Dialect, InstructionBuilder, Place, SpanAction, WriteTarget,
};
use cfglib::{
    AccessMode, AccessShape, CaptureBuilder, Captured, CapturedValue, DialectEffects, Effect,
    MemoryAccessKind, MemoryAtomicity, RtlEffect, SelectShape, StateTouch,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Wren;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Reg {
    /// A 32-bit register.
    A,
    /// A second 32-bit register.
    B,
    /// The zero flag.
    Z,
    /// The carry flag.
    C,
    /// The check register: one when the instruction check must run.
    Check,
    /// A 16-bit slot of a file that an index selects.
    Slot(u8),
    /// A bank of 64-bit configuration registers.
    Bank,
}

/// The pure operators. The type is `Clone` but not `Copy`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Op {
    Add,
    Sub,
    And,
    Eq,
    Not,
    /// `args[0]` picks `args[1]` when one and `args[2]` when zero.
    Select,
    /// `args[0]` picks `args[2]` when one and `args[1]` when zero.
    Mux,
    /// Two results: the quotient and the remainder.
    DivMod,
    /// No operand: the base of the module, which the reader places at
    /// [`BASE`].
    Base,
}

/// Where the reader of a Wren instruction places the module.
pub const BASE: u64 = 0x1000;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Space {
    Data,
    Stack,
}

/// The ordered effects. The type is `Clone` but not `Copy`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Eff {
    /// `args[0]` is the address; the result width is the size.
    Load(Space),
    /// `args[0]` is the address, `args[1]` the value.
    Store(Space),
    /// `args[0]` is the guard, `args[1]` the address.
    LoadIf,
    /// `args[0]` is the address, `args[1]` the value, `args[2]` the guard.
    StoreIf,
    /// An atomic add: `args[0]` is the address, `args[1]` the addend.
    AtomicAdd,
    /// The instruction check: `args[0]` is the check register; the result
    /// is one when the instruction must fault.
    InstructionCheck,
    /// An ordering fence that changes no state.
    Fence,
    /// An intrinsic that can change anything.
    Opaque,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Policy {
    Merge,
    Zero,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Fault {
    Page,
    Divide,
    Check,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Undef {
    Flags,
}

impl Dialect for Wren {
    type Root = Reg;
    type Operator = Op;
    type EffectOp = Eff;
    type WritePolicy = Policy;
    type Outcome = Fault;
    type NondetDomain = Undef;

    fn root_width(root: Reg) -> Bits {
        match root {
            Reg::A | Reg::B => bits(32),
            Reg::Z | Reg::C | Reg::Check => Bits::BOOL,
            Reg::Slot(_) => bits(16),
            Reg::Bank => bits(64),
        }
    }
}

/// The effect knowledge of Wren.
pub struct WrenEffects;

fn shape(
    kind: MemoryAccessKind,
    atomicity: MemoryAtomicity,
    space: Space,
    bytes: Bits,
    address: usize,
    guard: Option<usize>,
) -> AccessShape<Space> {
    AccessShape {
        mode: AccessMode { kind, atomicity },
        space,
        bytes: u32::from(bytes.width() / 8),
        address,
        guard,
    }
}

impl DialectEffects<Wren> for WrenEffects {
    type Space = Space;

    fn access(
        &self,
        operation: &Eff,
        args: &[Bits],
        results: &[Bits],
    ) -> Option<AccessShape<Space>> {
        use MemoryAccessKind::{Read, ReadModifyWrite, Write};
        use MemoryAtomicity::{Atomic, NonAtomic};
        Some(match operation {
            Eff::Load(space) => shape(Read, NonAtomic, *space, results[0], 0, None),
            Eff::Store(space) => shape(Write, NonAtomic, *space, args[1], 0, None),
            Eff::LoadIf => shape(Read, NonAtomic, Space::Data, results[0], 1, Some(0)),
            Eff::StoreIf => shape(Write, NonAtomic, Space::Data, args[1], 0, Some(2)),
            Eff::AtomicAdd => shape(ReadModifyWrite, Atomic, Space::Data, args[1], 0, None),
            Eff::InstructionCheck | Eff::Fence | Eff::Opaque => return None,
        })
    }

    fn intrinsic_state(&self, operation: &Eff) -> StateTouch {
        match operation {
            Eff::Opaque => StateTouch::Unrestricted,
            _ => StateTouch::Preserved,
        }
    }

    fn select(&self, operator: &Op) -> Option<SelectShape> {
        match operator {
            Op::Select => Some(SelectShape {
                guard: 0,
                taken: 1,
                otherwise: 2,
            }),
            Op::Mux => Some(SelectShape {
                guard: 0,
                taken: 2,
                otherwise: 1,
            }),
            _ => None,
        }
    }

    fn operator_name(&self, operator: &Op) -> &'static str {
        match operator {
            Op::Add => "add",
            Op::Sub => "sub",
            Op::And => "and",
            Op::Eq => "eq",
            Op::Not => "not",
            Op::Select => "select",
            Op::Mux => "mux",
            Op::DivMod => "divmod",
            Op::Base => "base",
        }
    }
}

/// A bit-vector type whose width the test knows to be valid.
pub fn bits(width: u16) -> Bits {
    Bits::new(width).expect("the test width is in 1..=Bits::MAX_WIDTH")
}

/// The place that covers the whole of `root`.
pub fn whole(root: Reg) -> Place<Reg> {
    Place::whole::<Wren>(root)
}

/// The place of `width` bits of `root` from `offset`.
pub fn part(root: Reg, offset: u16, width: u16) -> Place<Reg> {
    Place {
        root,
        slice: BitSlice::new(offset, bits(width)),
    }
}

/// A span that takes `width` bits of `value` from `offset`.
pub fn take<V>(at: u16, width: u16, value: V, offset: u16) -> SpanAction<Wren, V> {
    SpanAction {
        span: BitSlice::new(at, bits(width)),
        action: Action::TakeInput { value, offset },
    }
}

/// A span that keeps `width` bits.
pub fn keep<V>(at: u16, width: u16) -> SpanAction<Wren, V> {
    SpanAction {
        span: BitSlice::new(at, bits(width)),
        action: Action::PreservePreState,
    }
}

/// A span that receives `width` undefined bits.
pub fn undefined<V>(at: u16, width: u16) -> SpanAction<Wren, V> {
    SpanAction {
        span: BitSlice::new(at, bits(width)),
        action: Action::Nondeterministic(Undef::Flags),
    }
}

/// Writes `plan` to `root`.
pub fn write_plan<B: InstructionBuilder<Wren>>(
    builder: &mut B,
    root: Reg,
    plan: &[SpanAction<Wren, B::Value>],
) {
    builder.write(WriteTarget::Root { root, plan }, Policy::Merge);
}

/// Writes all of `root` from `value`.
pub fn write_whole<B: InstructionBuilder<Wren>>(builder: &mut B, root: Reg, value: B::Value) {
    let width = Wren::root_width(root).width();
    write_plan(builder, root, &[take(0, width, value, 0)]);
}

/// A constant of `width` bits.
pub fn constant<B: InstructionBuilder<Wren>>(builder: &mut B, value: u64, width: u16) -> B::Value {
    builder.constant(&[value], bits(width))
}

/// Captures the instruction that `emit` produces.
pub fn capture(
    emit: impl FnOnce(&mut CaptureBuilder<'_, Wren>) -> Control<CapturedValue>,
) -> Captured<Wren> {
    let mut captured = Captured::new();
    let mut builder = captured.begin();
    let control = emit(&mut builder);
    builder
        .finish(control)
        .expect("a capture accepts every record");
    captured
}

/// Captures the instruction that `emit` produces and computes its effect.
pub fn effect_of(
    emit: impl FnOnce(&mut CaptureBuilder<'_, Wren>) -> Control<CapturedValue>,
) -> RtlEffect<Wren, WrenEffects> {
    Effect::compute(&WrenEffects, &capture(emit))
}
