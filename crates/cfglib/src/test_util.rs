//! Shared test helpers for cfglib.
//!
//! Provides mock instruction types used across all test modules:
//!
//! - [`MockInst`] — minimal, flow-effect only (for graph/transform tests).
//! - [`DfInst`] — full-featured: flow + defs/uses + effects + optional
//!   copy semantics, expression decomposition, and constant values.
//!
//! `DfInst` implements **all** instruction traits (`FlowControl`,
//! `DisplayInstr`, `InstrInfo`, `EffectInfo`, `Predicated`, `CopySource`,
//! `ExprInstr`, `ConstantFolder`, `CallInfo`) so test modules don't need
//! to define their own instruction types.

extern crate alloc;
use alloc::borrow::Cow;
use alloc::vec::Vec;

use crate::analysis::expr::ExprInstr;
use crate::dataflow::copyprop::CopySource;
use crate::dataflow::{EffectInfo, InstrInfo};
use crate::display::DisplayInstr;
use crate::flow::{FlowControl, FlowEffect};

/// Test-local side-effect vocabulary (the library imposes none).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TestEffect {
    /// Reads from memory / global state.
    MemoryRead,
    /// Writes to memory / global state.
    MemoryWrite,
}

// ── MockInst (flow-only) ────────────────────────────────────────────

/// A minimal mock instruction carrying only flow-effect and mnemonic.
#[derive(Debug, Clone)]
pub struct MockInst(pub FlowEffect, pub &'static str);

impl FlowControl for MockInst {
    fn flow_effect(&self) -> FlowEffect {
        self.0
    }
}

impl DisplayInstr for MockInst {
    fn mnemonic(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.1)
    }
}

/// Shorthand for a [`MockInst`] with [`FlowEffect::Fallthrough`].
pub fn ff(name: &'static str) -> MockInst {
    MockInst(FlowEffect::Fallthrough, name)
}

// ── DfInst (full-featured mock) ─────────────────────────────────────

/// A mock instruction that carries control-flow, data-flow, and
/// optional higher-level semantics (copy, expression, constant).
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
    /// Side effects (memory, I/O, etc.).
    pub side_effects: Vec<TestEffect>,
    /// If `true`, this instruction is a simple copy (`defs[0] := uses[0]`).
    pub is_copy: bool,
    /// Expression operator name (e.g. `"add"`, `"mul"`). `None` for
    /// instructions that can't be decomposed into expressions.
    pub op: Option<&'static str>,
    /// If set, this instruction loads a constant value.
    pub constant: Option<i64>,
    /// If set, this instruction calls the named function.
    pub callee: Option<&'static str>,
    /// Whether this call is a tail call.
    pub tail: bool,
    /// If set, this instruction executes only under `(variable, polarity)`.
    pub pred: Option<(u16, bool)>,
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
}

impl EffectInfo for DfInst {
    type Effect = TestEffect;

    fn effects(&self) -> &[TestEffect] {
        &self.side_effects
    }
}

impl CopySource for DfInst {
    fn as_copy(&self) -> Option<(u16, u16)> {
        if self.is_copy && self.defs.len() == 1 && self.uses.len() == 1 {
            Some((self.defs[0], self.uses[0]))
        } else {
            None
        }
    }
    fn rewrite_use(&mut self, old: &u16, new: &u16) {
        for u in &mut self.uses {
            if u == old {
                *u = *new;
            }
        }
    }
}

impl ExprInstr for DfInst {
    type Operator = &'static str;
    type Const = i64;

    fn as_expr(&self) -> Option<(&'static str, &[u16])> {
        self.op.map(|op| (op, self.uses.as_slice()))
    }
    fn as_const(&self) -> Option<i64> {
        self.constant
    }
}

impl crate::dataflow::Predicated for DfInst {
    fn predicate(&self) -> Option<(u16, bool)> {
        self.pred
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

impl crate::dataflow::constprop::ConstantFolder for DfInst {
    type Const = i64;

    fn fold_constant(&self, _known: &alloc::collections::BTreeMap<u16, i64>) -> Option<(u16, i64)> {
        // If this instruction is a constant load, report it.
        if let (Some(val), Some(&dst)) = (self.constant, self.defs.first()) {
            return Some((dst, val));
        }
        None
    }
}

// ── DfInst constructors ─────────────────────────────────────────────

/// Default fields for a `DfInst` (no copy, no expr, no constant).
fn df_base(name: &'static str) -> DfInst {
    DfInst {
        effect: FlowEffect::Fallthrough,
        name,
        uses: Vec::new(),
        defs: Vec::new(),
        side_effects: Vec::new(),
        is_copy: false,
        op: None,
        constant: None,
        callee: None,
        tail: false,
        pred: None,
    }
}

/// Create a [`DfInst`] predicated on `(variable, polarity)`. Per the
/// [`Predicated`](crate::dataflow::Predicated) contract, the predicate
/// variable is also a use.
pub fn df_pred(name: &'static str, variable: u16, when_true: bool) -> DfInst {
    DfInst {
        pred: Some((variable, when_true)),
        uses: alloc::vec![variable],
        ..df_base(name)
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

/// Create a pure [`DfInst`] (no side effects, no defs/uses).
pub fn df_pure(name: &'static str) -> DfInst {
    df_base(name)
}

/// Create an impure [`DfInst`] with a single side effect.
pub fn df_impure(name: &'static str, e: TestEffect) -> DfInst {
    DfInst {
        side_effects: alloc::vec![e],
        ..df_base(name)
    }
}

/// Create a copy instruction (`dst := src`).
pub fn df_copy(name: &'static str, dst: u16, src: u16) -> DfInst {
    DfInst {
        defs: alloc::vec![dst],
        uses: alloc::vec![src],
        is_copy: true,
        ..df_base(name)
    }
}

/// Create an expression instruction (`dst = op(srcs...)`).
pub fn df_op(name: &'static str, op: &'static str, dst: u16, srcs: &[u16]) -> DfInst {
    DfInst {
        defs: alloc::vec![dst],
        uses: srcs.to_vec(),
        op: Some(op),
        ..df_base(name)
    }
}

/// Create a constant-load instruction (`dst = constant`).
pub fn df_const(name: &'static str, dst: u16, val: i64) -> DfInst {
    DfInst {
        defs: alloc::vec![dst],
        constant: Some(val),
        ..df_base(name)
    }
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
