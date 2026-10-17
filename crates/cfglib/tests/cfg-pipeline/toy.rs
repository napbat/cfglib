//! The toy three-address instruction set the pipeline tests run on.
//!
//! One generic instruction type implements every consumer trait the pipeline
//! reads: `InstrInfo`, `EffectInfo`, `ConstantFolder`, `CopySource`,
//! `DisplayInstr`, and `FlowControl` for the structured builder. It is
//! generic over the variable identity so the same instruction can name a
//! source variable (`Var`) or an SSA value (`SsaValue<Var>`), which is what
//! SSA destruction needs to emit a program over renamed values.

use std::borrow::Cow;
use std::collections::BTreeMap;

use cfglib::{
    AliasPairs, ConstantFolder, CopySource, DisplayInstr, EffectInfo, FlowControl, FlowEffect,
    InstrInfo, SsaValue,
};

/// A source variable, named the way a frontend would intern it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Var(pub &'static str);

/// What an instruction computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// `def = constant`.
    Const(i64),
    /// `def = use0`.
    Copy,
    /// `def = use0 + use1`, wrapping.
    Add,
    /// `def = use0 - use1`, wrapping.
    Sub,
    /// `def = use0 * use1`, wrapping.
    Mul,
    /// `def = use0 < use1` as `0` or `1`.
    Less,
    /// `def = use0 / use1`; a zero divisor throws before `def` is written.
    Divide,
    /// Emit `use0` as observable output.
    Print,
    /// Conditional terminator: `use0 != 0` takes the `ConditionalTrue` edge.
    Branch,
    /// Terminator that returns `use0`.
    Return,
    /// Structural marker with no data semantics (`else`, `loop`, `break`).
    Marker,
    /// Every `defs[i] = uses[i]` at once: all reads happen before any write.
    ParallelCopy,
}

/// A side effect beyond the def/use sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Effect {
    /// Observable output.
    Output,
    /// A terminator decides where control goes.
    Control,
}

/// One instruction over variable identity `V`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inst<V> {
    /// What the instruction computes.
    pub op: Op,
    /// Variables written, in order.
    pub defs: Vec<V>,
    /// Variables read, in order.
    pub uses: Vec<V>,
    effects: Vec<Effect>,
    /// How the structured builder reads the instruction.
    pub flow: FlowEffect,
}

/// The value of a pure operation, or `None` when it has no value (a zero
/// divisor). The interpreter and the constant folder both read this one
/// definition of the language.
pub fn compute(op: Op, arguments: &[i64]) -> Option<i64> {
    Some(match (op, arguments) {
        (Op::Const(value), []) => value,
        (Op::Copy, [source]) => *source,
        (Op::Add, [left, right]) => left.wrapping_add(*right),
        (Op::Sub, [left, right]) => left.wrapping_sub(*right),
        (Op::Mul, [left, right]) => left.wrapping_mul(*right),
        (Op::Less, [left, right]) => i64::from(left < right),
        (Op::Divide, [_, 0]) => return None,
        (Op::Divide, [left, right]) => left.wrapping_div(*right),
        _ => return None,
    })
}

impl<V> Inst<V> {
    fn make(op: Op, defs: Vec<V>, uses: Vec<V>) -> Self {
        let (effects, flow) = match op {
            Op::Print => (vec![Effect::Output], FlowEffect::Fallthrough),
            Op::Branch => (vec![Effect::Control], FlowEffect::ConditionalOpen),
            Op::Return => (vec![Effect::Control], FlowEffect::Return),
            _ => (Vec::new(), FlowEffect::Fallthrough),
        };
        Self {
            op,
            defs,
            uses,
            effects,
            flow,
        }
    }

    /// `defs[i] = uses[i]` for every `i`, with all reads before any write.
    pub fn parallel_copy(defs: Vec<V>, uses: Vec<V>) -> Self {
        assert_eq!(defs.len(), uses.len(), "a parallel copy pairs its operands");
        Self::make(Op::ParallelCopy, defs, uses)
    }

    /// The same instruction over other variable identities.
    pub fn renamed<W>(&self, defs: Vec<W>, uses: Vec<W>) -> Inst<W> {
        Inst {
            op: self.op,
            defs,
            uses,
            effects: self.effects.clone(),
            flow: self.flow,
        }
    }

    /// The same instruction classified differently for the structured builder.
    #[must_use]
    pub fn with_flow(mut self, flow: FlowEffect) -> Self {
        self.flow = flow;
        self
    }
}

impl Inst<Var> {
    /// `def = value`.
    pub fn konst(def: &'static str, value: i64) -> Self {
        Self::make(Op::Const(value), vec![Var(def)], Vec::new())
    }

    /// `def = source`.
    pub fn copy(def: &'static str, source: &'static str) -> Self {
        Self::make(Op::Copy, vec![Var(def)], vec![Var(source)])
    }

    fn binary(op: Op, def: &'static str, left: &'static str, right: &'static str) -> Self {
        Self::make(op, vec![Var(def)], vec![Var(left), Var(right)])
    }

    /// `def = left + right`.
    pub fn add(def: &'static str, left: &'static str, right: &'static str) -> Self {
        Self::binary(Op::Add, def, left, right)
    }

    /// `def = left - right`.
    pub fn sub(def: &'static str, left: &'static str, right: &'static str) -> Self {
        Self::binary(Op::Sub, def, left, right)
    }

    /// `def = left * right`.
    pub fn mul(def: &'static str, left: &'static str, right: &'static str) -> Self {
        Self::binary(Op::Mul, def, left, right)
    }

    /// `def = left < right`.
    pub fn less(def: &'static str, left: &'static str, right: &'static str) -> Self {
        Self::binary(Op::Less, def, left, right)
    }

    /// `def = left / right`, throwing on a zero divisor.
    pub fn divide(def: &'static str, left: &'static str, right: &'static str) -> Self {
        Self::binary(Op::Divide, def, left, right)
    }

    /// Emit `source`.
    pub fn print(source: &'static str) -> Self {
        Self::make(Op::Print, Vec::new(), vec![Var(source)])
    }

    /// Branch on `condition`.
    pub fn branch(condition: &'static str) -> Self {
        Self::make(Op::Branch, Vec::new(), vec![Var(condition)])
    }

    /// Return `source`.
    pub fn ret(source: &'static str) -> Self {
        Self::make(Op::Return, Vec::new(), vec![Var(source)])
    }

    /// A structural marker the builder classifies as `flow`.
    pub fn marker(flow: FlowEffect) -> Self {
        Self::make(Op::Marker, Vec::new(), Vec::new()).with_flow(flow)
    }
}

impl<V: Clone + Ord> InstrInfo for Inst<V> {
    type Variable = V;

    fn uses(&self) -> &[V] {
        &self.uses
    }

    fn defs(&self) -> &[V] {
        &self.defs
    }

    fn may_unwind(&self) -> bool {
        self.op == Op::Divide
    }
}

impl<V: Clone + Ord> EffectInfo for Inst<V> {
    type Effect = Effect;

    fn effects(&self) -> &[Effect] {
        &self.effects
    }
}

impl<V> FlowControl for Inst<V> {
    fn flow_effect(&self) -> FlowEffect {
        self.flow
    }
}

impl<V: Clone + Ord> ConstantFolder for Inst<V> {
    type Const = i64;

    fn fold_constant(&self, known: &BTreeMap<V, i64>) -> Option<(V, i64)> {
        let arguments = self
            .uses
            .iter()
            .map(|used| known.get(used).copied())
            .collect::<Option<Vec<_>>>();
        let value = match self.op {
            Op::Const(value) => value,
            // Only the first pair is folded: the others of a multi-def
            // instruction must fall to bottom.
            Op::ParallelCopy => *arguments?.first()?,
            Op::Print | Op::Branch | Op::Return | Op::Marker => return None,
            op => compute(op, &arguments?)?,
        };
        Some((self.defs.first()?.clone(), value))
    }

    fn fold_branch(&self, known: &BTreeMap<V, i64>) -> Option<bool> {
        if self.op != Op::Branch {
            return None;
        }
        known
            .get(self.uses.first()?)
            .map(|condition| *condition != 0)
    }
}

impl<V: Clone + Ord> CopySource for Inst<V> {
    fn as_copy(&self) -> Option<(V, V)> {
        (self.op == Op::Copy).then(|| (self.defs[0].clone(), self.uses[0].clone()))
    }

    fn as_aliases(&self) -> Option<AliasPairs<'_, V>> {
        matches!(self.op, Op::Copy | Op::ParallelCopy).then(|| (&*self.defs, &*self.uses))
    }

    fn rewrite_use(&mut self, old: &V, new: &V) {
        for used in &mut self.uses {
            if used == old {
                *used = new.clone();
            }
        }
    }
}

/// How a variable identity prints in the toy assembly.
pub trait Label {
    /// The printed name.
    fn label(&self) -> String;
}

impl Label for Var {
    fn label(&self) -> String {
        self.0.to_owned()
    }
}

impl Label for SsaValue<Var> {
    fn label(&self) -> String {
        format!("{}.{}", self.variable.0, self.version)
    }
}

impl<V: Label> DisplayInstr for Inst<V> {
    fn mnemonic(&self) -> Cow<'_, str> {
        let def = self.defs.first().map(Label::label).unwrap_or_default();
        let operands: Vec<String> = self.uses.iter().map(Label::label).collect();
        let infix = |symbol: &str| format!("{def} = {} {symbol} {}", operands[0], operands[1]);
        Cow::Owned(match self.op {
            Op::Const(value) => format!("{def} = {value}"),
            Op::Copy => format!("{def} = {}", operands[0]),
            Op::Add => infix("+"),
            Op::Sub => infix("-"),
            Op::Mul => infix("*"),
            Op::Less => infix("<"),
            Op::Divide => infix("/"),
            Op::Print => format!("print {}", operands[0]),
            Op::Branch => format!("branch {}", operands[0]),
            Op::Return => format!("return {}", operands[0]),
            Op::Marker => String::from("nop"),
            Op::ParallelCopy => {
                let targets: Vec<String> = self.defs.iter().map(Label::label).collect();
                format!("({}) = ({})", targets.join(", "), operands.join(", "))
            }
        })
    }
}
