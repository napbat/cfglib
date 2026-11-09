//! The statement language the analyses read: three-address statements over
//! named locals, with declared effects and returns.

use cfglib::{BlockId, Cfg, EdgeId, EdgeKind, EffectInfo, FlowControl, FlowEffect, InstrInfo};

pub type Local = &'static str;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Effect {
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Compute,
    Return,
    Effect,
    Plain,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stmt {
    pub kind: Kind,
    uses: Vec<Local>,
    defs: Vec<Local>,
    effects: Vec<Effect>,
}

impl Stmt {
    fn new(kind: Kind) -> Self {
        Self {
            kind,
            uses: Vec::new(),
            defs: Vec::new(),
            effects: Vec::new(),
        }
    }

    /// `def = op(operands)`.
    pub fn compute(def: Local, operands: &[Local]) -> Self {
        Self {
            uses: operands.to_vec(),
            defs: vec![def],
            ..Self::new(Kind::Compute)
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

    pub fn ret() -> Self {
        Self::new(Kind::Return)
    }

    pub fn plain() -> Self {
        Self::new(Kind::Plain)
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
            Kind::Return => FlowEffect::Return,
            Kind::Compute | Kind::Effect | Kind::Plain => FlowEffect::Fallthrough,
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
