//! Mock instruction for value-numbering and PRE tests.

extern crate alloc;

use alloc::vec::Vec;

use crate::analysis::value_numbering::ValueNumberInfo;
use crate::dataflow::InstrInfo;
use crate::flow::{FlowControl, FlowEffect};

/// A mock instruction carrying an operator identity for value numbering:
/// `defs = op(uses)`, pure unless marked otherwise.
#[derive(Debug, Clone)]
pub struct VnInst {
    /// Operation identity for hash-consing.
    pub op: u32,
    /// Variables read by this instruction.
    pub uses: Vec<u16>,
    /// Variables written by this instruction.
    pub defs: Vec<u16>,
    /// Whether this instruction is pure (value-numberable).
    pub pure_: bool,
    /// Whether execution can unwind before this instruction completes.
    pub unwinds: bool,
}

impl FlowControl for VnInst {
    fn flow_effect(&self) -> FlowEffect {
        FlowEffect::Fallthrough
    }
}

impl InstrInfo for VnInst {
    type Variable = u16;

    fn uses(&self) -> &[u16] {
        &self.uses
    }
    fn defs(&self) -> &[u16] {
        &self.defs
    }

    fn may_unwind(&self) -> bool {
        self.unwinds
    }
}

impl ValueNumberInfo for VnInst {
    type Operator = u32;

    fn operator(&self) -> u32 {
        self.op
    }
    fn is_pure(&self) -> bool {
        self.pure_
    }
}

/// Create a pure [`VnInst`] (`defs = op(uses)`).
pub fn vn_inst(op: u32, uses: &[u16], defs: &[u16]) -> VnInst {
    VnInst {
        op,
        uses: uses.to_vec(),
        defs: defs.to_vec(),
        pure_: true,
        unwinds: false,
    }
}

/// Create an impure [`VnInst`] — skipped by value numbering, but its
/// definitions still invalidate dependent value numbers.
pub fn vn_impure(op: u32, uses: &[u16], defs: &[u16]) -> VnInst {
    VnInst {
        pure_: false,
        ..vn_inst(op, uses, defs)
    }
}

/// Create an impure [`VnInst`] that can unwind before it completes, such
/// as a call inside a protected region.
pub fn vn_throwing(op: u32, uses: &[u16], defs: &[u16]) -> VnInst {
    VnInst {
        unwinds: true,
        ..vn_impure(op, uses, defs)
    }
}
