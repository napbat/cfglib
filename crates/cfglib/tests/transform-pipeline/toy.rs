//! A toy register machine: the instruction set the passes transform, and two
//! interpreters that execute it. The interpreters are the semantic oracle of
//! this target: a pass that claims to preserve behavior is checked by running
//! the program before and after on the same inputs.

use std::collections::BTreeMap;

use cfglib::{
    BlockId, Cfg, CopySource, EdgeId, EdgeKind, EffectInfo, Emitter, InstrInfo, LinearInst,
    ValueNumberInfo,
};

pub type Reg = u8;

const STEP_LIMIT: usize = 20_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Less,
}

impl BinaryOp {
    fn apply(self, left: i64, right: i64) -> i64 {
        match self {
            Self::Add => left.wrapping_add(right),
            Self::Subtract => left.wrapping_sub(right),
            Self::Multiply => left.wrapping_mul(right),
            Self::Less => i64::from(left < right),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Constant(i64),
    Move,
    Binary(BinaryOp),
    Print,
    /// Two-way branch on its operand: nonzero takes the edge that is not
    /// `ConditionalFalse`, zero takes the `ConditionalFalse` edge.
    Branch,
    /// Ends the program; an operand, when present, is printed first.
    Return,
    /// The next three exist only in linearized streams. They carry the index
    /// of the block they name.
    Label(usize),
    Jump(usize),
    BranchIf(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Effect {
    Output,
    Control,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inst {
    pub kind: Kind,
    uses: Vec<Reg>,
    defs: Vec<Reg>,
    effects: Vec<Effect>,
}

impl Inst {
    fn new(kind: Kind, uses: &[Reg], defs: &[Reg], effects: &[Effect]) -> Self {
        Self {
            kind,
            uses: uses.to_vec(),
            defs: defs.to_vec(),
            effects: effects.to_vec(),
        }
    }

    pub fn constant(destination: Reg, value: i64) -> Self {
        Self::new(Kind::Constant(value), &[], &[destination], &[])
    }

    pub fn mov(destination: Reg, source: Reg) -> Self {
        Self::new(Kind::Move, &[source], &[destination], &[])
    }

    pub fn binary(op: BinaryOp, destination: Reg, left: Reg, right: Reg) -> Self {
        Self::new(Kind::Binary(op), &[left, right], &[destination], &[])
    }

    pub fn add(destination: Reg, left: Reg, right: Reg) -> Self {
        Self::binary(BinaryOp::Add, destination, left, right)
    }

    pub fn less(destination: Reg, left: Reg, right: Reg) -> Self {
        Self::binary(BinaryOp::Less, destination, left, right)
    }

    pub fn print(source: Reg) -> Self {
        Self::new(Kind::Print, &[source], &[], &[Effect::Output])
    }

    pub fn branch(condition: Reg) -> Self {
        Self::new(Kind::Branch, &[condition], &[], &[Effect::Control])
    }

    pub fn ret() -> Self {
        Self::new(Kind::Return, &[], &[], &[Effect::Control])
    }

    pub fn operands(&self) -> &[Reg] {
        &self.uses
    }

    /// The same instruction with every register renamed.
    pub fn renamed(&self, mut rename: impl FnMut(Reg) -> Reg) -> Self {
        Self {
            kind: self.kind,
            uses: self.uses.iter().map(|&register| rename(register)).collect(),
            defs: self.defs.iter().map(|&register| rename(register)).collect(),
            effects: self.effects.clone(),
        }
    }
}

impl InstrInfo for Inst {
    type Variable = Reg;

    fn uses(&self) -> &[Reg] {
        &self.uses
    }

    fn defs(&self) -> &[Reg] {
        &self.defs
    }
}

impl EffectInfo for Inst {
    type Effect = Effect;

    fn effects(&self) -> &[Effect] {
        &self.effects
    }
}

impl ValueNumberInfo for Inst {
    type Operator = Kind;

    fn operator(&self) -> Kind {
        self.kind
    }

    fn is_pure(&self) -> bool {
        self.effects.is_empty()
    }
}

impl CopySource for Inst {
    fn as_copy(&self) -> Option<(Reg, Reg)> {
        (self.kind == Kind::Move).then(|| (self.defs[0], self.uses[0]))
    }

    fn rewrite_use(&mut self, old: &Reg, new: &Reg) {
        for used in &mut self.uses {
            if used == old {
                *used = *new;
            }
        }
    }
}

/// Edge payload a payload-bearing consumer would attach. Edges the test
/// builds are `Tagged`; edges a pass invents get the `Default`.
pub trait Payload: Clone + Default + PartialEq + core::fmt::Debug {
    fn tagged(tag: u32) -> Self;
}

impl Payload for () {
    fn tagged(_: u32) -> Self {}
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Route {
    #[default]
    Synthetic,
    Tagged(u32),
}

impl Payload for Route {
    fn tagged(tag: u32) -> Self {
        Self::Tagged(tag)
    }
}

/// Builds a program, giving every edge a distinct payload tag.
pub struct Builder<E: Payload> {
    pub cfg: Cfg<Inst, E>,
    next_tag: u32,
}

impl<E: Payload> Builder<E> {
    pub fn new() -> Self {
        Self {
            cfg: Cfg::with_edge_payload(),
            next_tag: 0,
        }
    }

    pub fn entry(&self) -> BlockId {
        self.cfg.entry()
    }

    pub fn block(&mut self, instructions: impl IntoIterator<Item = Inst>) -> BlockId {
        let block = self.cfg.new_block();
        self.cfg
            .block_mut(block)
            .instructions_mut()
            .extend(instructions);
        block
    }

    pub fn push(&mut self, block: BlockId, instruction: Inst) {
        self.cfg.block_mut(block).push(instruction);
    }

    pub fn edge(&mut self, source: BlockId, target: BlockId, kind: EdgeKind) -> EdgeId {
        let tag = self.next_tag;
        self.next_tag += 1;
        let edge = self
            .cfg
            .add_edge_with_payload(source, target, kind, E::tagged(tag));
        self.cfg
            .edge_mut(edge)
            .set_weight(Some(f64::from(tag % 4) / 4.0));
        edge
    }

    pub fn finish(self) -> Cfg<Inst, E> {
        self.cfg
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub output: Vec<i64>,
    pub observed: Vec<Option<i64>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stuck {
    Undefined(Reg),
    Ambiguous(BlockId),
    MissingLabel(usize),
    StepLimit,
}

#[derive(PartialEq, Eq)]
enum Flow {
    Next,
    Halt,
}

struct Machine {
    registers: BTreeMap<Reg, i64>,
    output: Vec<i64>,
    steps: usize,
    /// What the last `Branch` latched; the next two-way block consumes it,
    /// so a block can be split between its branch and its exits.
    flag: Option<i64>,
}

impl Machine {
    fn new(inputs: &[(Reg, i64)]) -> Self {
        Self {
            registers: inputs.iter().copied().collect(),
            output: Vec::new(),
            steps: 0,
            flag: None,
        }
    }

    fn read(&self, register: Reg) -> Result<i64, Stuck> {
        self.registers
            .get(&register)
            .copied()
            .ok_or(Stuck::Undefined(register))
    }

    fn execute(&mut self, inst: &Inst) -> Result<Flow, Stuck> {
        self.steps += 1;
        if self.steps > STEP_LIMIT {
            return Err(Stuck::StepLimit);
        }
        match inst.kind {
            Kind::Constant(value) => {
                self.registers.insert(inst.defs[0], value);
            }
            Kind::Move => {
                let value = self.read(inst.uses[0])?;
                self.registers.insert(inst.defs[0], value);
            }
            Kind::Binary(op) => {
                let value = op.apply(self.read(inst.uses[0])?, self.read(inst.uses[1])?);
                self.registers.insert(inst.defs[0], value);
            }
            Kind::Print => {
                let value = self.read(inst.uses[0])?;
                self.output.push(value);
            }
            Kind::Return => {
                if let Some(&register) = inst.uses.first() {
                    let value = self.read(register)?;
                    self.output.push(value);
                }
                return Ok(Flow::Halt);
            }
            Kind::Branch => self.flag = Some(self.read(inst.uses[0])?),
            Kind::Label(_) | Kind::Jump(_) | Kind::BranchIf(_) => {}
        }
        Ok(Flow::Next)
    }

    fn finish(self, observe: &[Reg]) -> Outcome {
        Outcome {
            observed: observe
                .iter()
                .map(|register| self.registers.get(register).copied())
                .collect(),
            output: self.output,
        }
    }
}

/// Executes the CFG from its entry. Exceptional edges are never followed: the
/// toy machine cannot throw.
pub fn run<E>(
    cfg: &Cfg<Inst, E>,
    inputs: &[(Reg, i64)],
    observe: &[Reg],
) -> Result<Outcome, Stuck> {
    let mut machine = Machine::new(inputs);
    let mut block = cfg.entry();
    loop {
        let instructions = cfg.block(block).instructions();
        for inst in instructions {
            if machine.execute(inst)? == Flow::Halt {
                return Ok(machine.finish(observe));
            }
        }
        let edges: Vec<EdgeId> = cfg
            .outgoing(block)
            .filter(|&edge| !cfg.edge(edge).kind().is_exceptional())
            .collect();
        let chosen = match edges.as_slice() {
            [] => return Ok(machine.finish(observe)),
            [only] => Some(*only),
            [first, second] => machine.flag.take().and_then(|condition| {
                let wanted = |edge: &EdgeId| {
                    (cfg.edge(*edge).kind() != EdgeKind::ConditionalFalse) == (condition != 0)
                };
                [*first, *second].into_iter().find(wanted)
            }),
            _ => None,
        };
        let Some(chosen) = chosen else {
            return Err(Stuck::Ambiguous(block));
        };
        block = cfg.edge(chosen).target();
    }
}

/// Executes a linearized stream from its first instruction.
pub fn run_linear(
    stream: &[LinearInst<Inst>],
    inputs: &[(Reg, i64)],
    observe: &[Reg],
) -> Result<Outcome, Stuck> {
    let labels: BTreeMap<usize, usize> = stream
        .iter()
        .enumerate()
        .filter_map(|(position, entry)| match entry.inst.kind {
            Kind::Label(block) => Some((block, position)),
            _ => None,
        })
        .collect();
    let target = |block: usize| {
        labels
            .get(&block)
            .copied()
            .ok_or(Stuck::MissingLabel(block))
    };
    let mut machine = Machine::new(inputs);
    let mut position = 0;
    while let Some(entry) = stream.get(position) {
        position += 1;
        match entry.inst.kind {
            Kind::Jump(block) => position = target(block)?,
            Kind::BranchIf(block) => {
                if machine.read(entry.inst.uses[0])? != 0 {
                    position = target(block)?;
                }
            }
            _ => {
                if machine.execute(&entry.inst)? == Flow::Halt {
                    break;
                }
            }
        }
    }
    Ok(machine.finish(observe))
}

/// The emitter a consumer of this toy ISA would write.
pub struct ToyEmitter;

impl Emitter<Inst> for ToyEmitter {
    fn emit_jump(&self, target: BlockId) -> Inst {
        Inst::new(Kind::Jump(target.index()), &[], &[], &[Effect::Control])
    }

    fn emit_conditional_branch(&self, condition: &Inst, target: BlockId) -> Inst {
        Inst::new(
            Kind::BranchIf(target.index()),
            condition.operands(),
            &[],
            &[Effect::Control],
        )
    }

    fn emit_block_start(&self, block: BlockId) -> Option<Inst> {
        Some(Inst::new(
            Kind::Label(block.index()),
            &[],
            &[],
            &[Effect::Control],
        ))
    }
}

/// Input vectors every semantic comparison runs: register 0 is data, register
/// 1 bounds every loop, registers 2 and 3 are more data.
pub const INPUTS: [[(Reg, i64); 4]; 6] = [
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 1), (1, 1), (2, 5), (3, -2)],
    [(0, 2), (1, 2), (2, -3), (3, 7)],
    [(0, -4), (1, 3), (2, 11), (3, 1)],
    [(0, 9), (1, 1), (2, 0), (3, 0)],
    [(0, 3), (1, 0), (2, 8), (3, 8)],
];

/// The registers whose final values a comparison also observes.
pub const OBSERVED: [Reg; 3] = [0, 2, 3];

/// Asserts that `after` behaves exactly like `before` on every input vector,
/// and that `before` actually terminates normally, so the comparison cannot be
/// vacuous.
pub fn assert_same_behavior<E1, E2>(before: &Cfg<Inst, E1>, after: &Cfg<Inst, E2>, context: &str) {
    assert_same_behavior_observing(before, after, &OBSERVED, context);
}

/// [`assert_same_behavior`] observing only `observe` at the end of the run.
/// Passes that delete dead stores must be compared on the printed output
/// alone unless they were told which registers the caller reads.
pub fn assert_same_behavior_observing<E1, E2>(
    before: &Cfg<Inst, E1>,
    after: &Cfg<Inst, E2>,
    observe: &[Reg],
    context: &str,
) {
    for inputs in INPUTS {
        let expected = run(before, &inputs, observe);
        assert!(
            expected.is_ok(),
            "{context}: the reference program is stuck on {inputs:?}: {expected:?}"
        );
        assert_eq!(
            run(after, &inputs, observe),
            expected,
            "{context}: behavior differs on {inputs:?}"
        );
    }
}
