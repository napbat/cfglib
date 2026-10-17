//! The statement language the analyses read: three-address statements over
//! named locals, with declared effects, calls, and unwinding.

use cfglib::{
    BlockId, CallInfo, Cfg, EdgeId, EdgeKind, EffectInfo, ExprInstr, FlowControl, FlowEffect,
    InstrInfo, ValueNumberInfo,
};

pub type Local = &'static str;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Effect {
    Memory,
    Output,
    Network,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Op {
    Add,
    Multiply,
    Compare,
    Negate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Compute(Op),
    Constant(i64),
    /// Defines a local from outside the language, so nothing can be said about it.
    Opaque,
    Call,
    ConditionalCall,
    Return,
    Effect,
    Plain,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stmt {
    pub kind: Kind,
    uses: Vec<Local>,
    operands: Vec<Local>,
    defs: Vec<Local>,
    effects: Vec<Effect>,
    unwinds: bool,
    tail: bool,
    callee: Option<&'static str>,
}

impl Stmt {
    fn new(kind: Kind) -> Self {
        Self {
            kind,
            uses: Vec::new(),
            operands: Vec::new(),
            defs: Vec::new(),
            effects: Vec::new(),
            unwinds: false,
            tail: false,
            callee: None,
        }
    }

    pub fn compute(op: Op, def: Local, operands: &[Local]) -> Self {
        Self {
            uses: operands.to_vec(),
            operands: operands.to_vec(),
            defs: vec![def],
            ..Self::new(Kind::Compute(op))
        }
    }

    pub fn add(def: Local, left: Local, right: Local) -> Self {
        Self::compute(Op::Add, def, &[left, right])
    }

    pub fn multiply(def: Local, left: Local, right: Local) -> Self {
        Self::compute(Op::Multiply, def, &[left, right])
    }

    pub fn constant(def: Local, value: i64) -> Self {
        Self {
            defs: vec![def],
            ..Self::new(Kind::Constant(value))
        }
    }

    pub fn opaque(def: Local) -> Self {
        Self {
            defs: vec![def],
            ..Self::new(Kind::Opaque)
        }
    }

    /// An effectful statement with no result: a store, a print, a send.
    pub fn effect(effects: &[Effect], uses: &[Local]) -> Self {
        Self {
            uses: uses.to_vec(),
            effects: effects.to_vec(),
            ..Self::new(Kind::Effect)
        }
    }

    pub fn call(callee: &'static str, defs: &[Local], effects: &[Effect]) -> Self {
        Self {
            defs: defs.to_vec(),
            effects: effects.to_vec(),
            callee: Some(callee),
            ..Self::new(Kind::Call)
        }
    }

    pub fn conditional_call(callee: &'static str) -> Self {
        Self {
            callee: Some(callee),
            ..Self::new(Kind::ConditionalCall)
        }
    }

    pub fn tail_call(callee: &'static str) -> Self {
        Self {
            tail: true,
            ..Self::call(callee, &[], &[])
        }
    }

    pub fn indirect_tail_call() -> Self {
        Self {
            tail: true,
            callee: None,
            ..Self::new(Kind::Call)
        }
    }

    pub fn ret() -> Self {
        Self::new(Kind::Return)
    }

    pub fn plain() -> Self {
        Self::new(Kind::Plain)
    }

    /// The same statement, but it may unwind before it completes.
    pub fn throwing(self) -> Self {
        Self {
            unwinds: true,
            ..self
        }
    }

    /// Reads `extra` without it being an operand of the expression.
    pub fn also_reading(mut self, extra: Local) -> Self {
        self.uses.push(extra);
        self
    }
}

impl InstrInfo for Stmt {
    type Variable = Local;

    fn uses(&self) -> &[Local] {
        &self.uses
    }

    fn defs(&self) -> &[Local] {
        &self.defs
    }

    fn may_unwind(&self) -> bool {
        self.unwinds
    }
}

impl EffectInfo for Stmt {
    type Effect = Effect;

    fn effects(&self) -> &[Effect] {
        &self.effects
    }
}

impl FlowControl for Stmt {
    fn flow_effect(&self) -> FlowEffect {
        match self.kind {
            Kind::Call => FlowEffect::Call,
            Kind::ConditionalCall => FlowEffect::ConditionalCall,
            Kind::Return => FlowEffect::Return,
            _ => FlowEffect::Fallthrough,
        }
    }
}

impl CallInfo for Stmt {
    type Callee = &'static str;

    fn callee(&self) -> Option<&'static str> {
        self.callee
    }

    fn is_tail_call(&self) -> bool {
        self.tail
    }
}

impl ValueNumberInfo for Stmt {
    type Operator = Kind;

    fn operator(&self) -> Kind {
        self.kind
    }

    fn is_pure(&self) -> bool {
        matches!(self.kind, Kind::Compute(_) | Kind::Constant(_))
            && self.effects.is_empty()
            && !self.unwinds
    }
}

impl ExprInstr for Stmt {
    type Operator = Op;
    type Const = i64;

    fn as_expr(&self) -> Option<(Op, &[Local])> {
        match self.kind {
            Kind::Compute(op) => Some((op, &self.operands)),
            _ => None,
        }
    }

    fn as_const(&self) -> Option<i64> {
        match self.kind {
            Kind::Constant(value) => Some(value),
            _ => None,
        }
    }
}

/// Edge payload for the payload-bearing runs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Route(pub u32);

pub fn block<E>(cfg: &mut Cfg<Stmt, E>, statements: impl IntoIterator<Item = Stmt>) -> BlockId {
    let block = cfg.new_block();
    cfg.block_mut(block).instructions_mut().extend(statements);
    block
}

pub fn link<E: Default>(
    cfg: &mut Cfg<Stmt, E>,
    source: BlockId,
    target: BlockId,
    kind: EdgeKind,
) -> EdgeId {
    cfg.add_edge_with_payload(source, target, kind, E::default())
}

pub fn fill<E>(cfg: &mut Cfg<Stmt, E>, at: BlockId, statements: impl IntoIterator<Item = Stmt>) {
    cfg.block_mut(at).instructions_mut().extend(statements);
}
