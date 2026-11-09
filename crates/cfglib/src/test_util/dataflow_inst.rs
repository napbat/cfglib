//! Full-featured mock instruction for dataflow tests.

extern crate alloc;

use alloc::borrow::Cow;
use alloc::vec::Vec;

use crate::InstrInfo;
use crate::display::DisplayInstr;
use crate::flow::{FlowControl, FlowEffect};

/// A mock instruction that carries control-flow, data-flow, and call
/// semantics.
///
/// Used across all analysis and transform test modules.
#[derive(Debug, Clone)]
pub struct DfInst {
    /// Control-flow classification.
    pub effect: FlowEffect,
    /// Mnemonic label.
    pub name: &'static str,
    /// Variables read by this instruction.
    pub uses: Vec<u16>,
    /// Variables written by this instruction.
    pub defs: Vec<u16>,
    /// If set, this instruction calls the named function.
    pub callee: Option<&'static str>,
    /// Whether this call is a tail call.
    pub tail: bool,
}

impl FlowControl for DfInst {
    fn flow_effect(&self) -> FlowEffect {
        self.effect
    }
}

impl DisplayInstr for DfInst {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.name)
    }
}

impl InstrInfo for DfInst {
    type Variable = u16;

    fn uses(&self) -> &[u16] {
        &self.uses
    }
    fn defs(&self) -> &[u16] {
        &self.defs
    }

    fn may_unwind(&self) -> bool {
        self.effect == FlowEffect::MayThrow
    }
}

impl crate::flow::CallInfo for DfInst {
    type Callee = &'static str;

    fn callee(&self) -> Option<&'static str> {
        self.callee
    }

    fn is_tail_call(&self) -> bool {
        self.tail
    }
}

/// Default fields for a `DfInst` (no defs, uses, effects, or callee).
fn df_base(name: &'static str) -> DfInst {
    DfInst {
        effect: FlowEffect::Fallthrough,
        name,
        uses: Vec::new(),
        defs: Vec::new(),
        callee: None,
        tail: false,
    }
}

/// Create a [`DfInst`] that defines a single variable.
pub fn df_def(name: &'static str, loc: u16) -> DfInst {
    DfInst {
        defs: alloc::vec![loc],
        ..df_base(name)
    }
}

/// Create a [`DfInst`] that uses a single variable.
pub fn df_use(name: &'static str, loc: u16) -> DfInst {
    DfInst {
        uses: alloc::vec![loc],
        ..df_base(name)
    }
}

/// Create a plain [`DfInst`] with no defs, uses, or side effects.
pub fn df_ff(name: &'static str) -> DfInst {
    df_base(name)
}

/// Override the flow effect of a [`DfInst`].
pub fn df_with_effect(mut inst: DfInst, effect: FlowEffect) -> DfInst {
    inst.effect = effect;
    inst
}

/// Make `inst` an instruction that may unwind, as a call does.
pub fn df_throwing(inst: DfInst) -> DfInst {
    df_with_effect(inst, FlowEffect::MayThrow)
}

/// Create a call instruction targeting `callee`.
pub fn df_call(name: &'static str, callee: &'static str, tail: bool) -> DfInst {
    DfInst {
        effect: FlowEffect::Call,
        callee: Some(callee),
        tail,
        ..df_base(name)
    }
}
