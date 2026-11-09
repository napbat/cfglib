//! Kestrel: a flag-setting register ISA with partial-width registers.
//!
//! Four 32-bit registers (their low and high halves are bit slices of one
//! root), four one-bit status flags, one 130-bit accumulator that spans three
//! words, and a bank of model-specific registers.

use cfglib::ir::rtl::{Bits, Dialect};

use crate::bits;
use crate::model::Target;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Kestrel;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Reg {
    R0,
    R1,
    R2,
    R3,
    Zero,
    Neg,
    Carry,
    Over,
    Wide,
    Csr,
}

impl Reg {
    pub const ALL: [Self; 10] = [
        Self::R0,
        Self::R1,
        Self::R2,
        Self::R3,
        Self::Zero,
        Self::Neg,
        Self::Carry,
        Self::Over,
        Self::Wide,
        Self::Csr,
    ];
}

/// The pure operators. Operand widths come from the operands themselves.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum KOp {
    /// The wrapping sum, then the carry out of the top bit.
    AddCarry,
    Xor,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum KEffect {
    Load,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Policy {
    Merging,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Fault {
    PageFault,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Undef {
    AllFlags,
}

impl Dialect for Kestrel {
    type Root = Reg;
    type Operator = KOp;
    type EffectOp = KEffect;
    type WritePolicy = Policy;
    type Outcome = Fault;
    type NondetDomain = Undef;

    fn root_width(root: Reg) -> Bits {
        match root {
            Reg::R0 | Reg::R1 | Reg::R2 | Reg::R3 | Reg::Csr => bits(32),
            Reg::Zero | Reg::Neg | Reg::Carry | Reg::Over => Bits::BOOL,
            Reg::Wide => bits(130),
        }
    }
}

impl Target for Kestrel {
    fn root_of(id: u8) -> Reg {
        Reg::ALL[usize::from(id)]
    }

    fn operator() -> KOp {
        KOp::Xor
    }

    fn effect() -> KEffect {
        KEffect::Load
    }

    fn policy() -> Policy {
        Policy::Merging
    }

    fn outcome() -> Fault {
        Fault::PageFault
    }

    fn domain() -> Undef {
        Undef::AllFlags
    }
}
