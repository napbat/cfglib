//! Full-featured mock instruction for dataflow tests.

extern crate alloc;

use alloc::borrow::Cow;
use alloc::vec::Vec;

use crate::InstrInfo;
use crate::display::DisplayInstr;
use crate::flow::{Flow, FlowControl, FlowEffect};

/// A mock instruction that carries control-flow, data-flow, and call
/// semantics.
///
/// Used across all analysis and transform test modules.
#[derive(Debug, Clone)]
pub struct DfInst {
    /// Control-flow classification; a call names its callee here.
    pub effect: FlowEffect<&'static str, ()>,
    /// Mnemonic label.
    pub name: &'static str,
    /// Variables read by this instruction.
    pub uses: Vec<u16>,
    /// Variables written by this instruction.
    pub defs: Vec<u16>,
    /// Whether execution can unwind before this instruction completes.
    pub throws: bool,
}

impl FlowControl for DfInst {
    type Target = &'static str;
    type CaseKey = ();

    fn flow_effect(&self) -> FlowEffect<&'static str, ()> {
        self.effect.clone()
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
        self.throws
    }
}

/// Default fields for a `DfInst` (no defs, uses, effects, or callee).
fn df_base(name: &'static str) -> DfInst {
    DfInst {
        effect: FlowEffect::Flow(Flow::Next),
        name,
        uses: Vec::new(),
        defs: Vec::new(),
        throws: false,
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

/// Make `inst` an instruction that may unwind, as a call does.
pub fn df_throwing(mut inst: DfInst) -> DfInst {
    inst.throws = true;
    inst
}

/// Create a call instruction targeting `callee`, as a tail call when `tail`.
pub fn df_call(name: &'static str, callee: &'static str, tail: bool) -> DfInst {
    let flow = if tail {
        Flow::TailCall { target: callee }
    } else {
        Flow::Call { target: callee }
    };
    DfInst {
        effect: FlowEffect::Flow(flow),
        ..df_base(name)
    }
}
