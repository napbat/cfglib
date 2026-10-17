//! The consumer's flat bytecode: a `Cfg` of register instructions with
//! typed occurrences, the `FlatBuilder` that `lower` drives, and the traits
//! that let `lift` read it back.

use std::fmt;

use cfglib::ir::hlil::{
    Appended, FlatBuilder, FlatFunction, InstructionShape, LiftEdge, LiftInstruction, LowerEdge,
    Lowered, Occurrence, VariableOrigin,
};
use cfglib::{
    BlockId, Cfg, EdgeId, EdgeKind, EffectInfo, FlowControl, FlowEffect, Id, IdTag, InstrInfo,
    KindedEdge, Predicated, ProgramPoint, Region, RegionId, verify,
};

use crate::dialect::{Calc, Effect, Lit, Local, Op, Ty};

/// Marks a register handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct RegTag;

impl IdTag for RegTag {
    const PREFIX: &'static str = "r";
}

/// A named local of the bytecode.
pub(crate) type Reg = Id<RegTag>;

/// What one bytecode instruction does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Form {
    Const(Lit),
    Copy,
    Apply(Op),
    /// Writes the last use into the memory cell that the `Cell` operation
    /// forms over the preceding uses.
    Store(Op),
    /// Moves use `i` into definition `i`, reading every use first.
    Move,
    Branch,
    /// Branches on the operation applied to every use.
    BranchOn(Op),
    Switch,
    Return,
    Caught,
    Enter(Op),
    Exit(Op),
    /// Does nothing; a jump trampoline when it is a block's only content.
    Nop,
}

/// One bytecode instruction with its typed occurrences.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Instr {
    pub(crate) form: Form,
    pub(crate) uses: Vec<Reg>,
    pub(crate) defs: Vec<Reg>,
    pub(crate) use_types: Vec<Ty>,
    pub(crate) def_types: Vec<Ty>,
    /// The register that guards the instruction and the polarity under
    /// which it runs.
    pub(crate) guard: Option<(Reg, bool)>,
    /// The use position that reads the previous value of the definition.
    pub(crate) previous_value: Option<usize>,
    /// Whether the definition must stay a visible statement.
    pub(crate) materialize: bool,
}

impl Instr {
    pub(crate) fn new(
        form: Form,
        uses: &[Occurrence<Reg, Ty>],
        defs: &[Occurrence<Reg, Ty>],
    ) -> Self {
        Self {
            form,
            uses: uses.iter().map(|occurrence| occurrence.variable).collect(),
            defs: defs.iter().map(|occurrence| occurrence.variable).collect(),
            use_types: uses.iter().map(|occurrence| occurrence.ty).collect(),
            def_types: defs.iter().map(|occurrence| occurrence.ty).collect(),
            guard: None,
            previous_value: None,
            materialize: false,
        }
    }

    pub(crate) fn may_throw(&self) -> bool {
        matches!(&self.form, Form::Apply(op) if op.may_throw())
    }

    pub(crate) fn is_terminator(&self) -> bool {
        matches!(
            self.form,
            Form::Branch | Form::BranchOn(_) | Form::Switch | Form::Return
        )
    }
}

/// The payload of one bytecode edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Link {
    Entry,
    Next,
    Goto,
    Yes,
    No,
    Case(Vec<Lit>),
    Otherwise,
    Unwind,
}

impl KindedEdge for Link {
    fn kind(&self) -> EdgeKind {
        match self {
            Self::Entry | Self::Next | Self::Otherwise => EdgeKind::Fallthrough,
            Self::Goto => EdgeKind::Jump,
            Self::Yes => EdgeKind::ConditionalTrue,
            Self::No => EdgeKind::ConditionalFalse,
            Self::Case(_) => EdgeKind::SwitchCase,
            Self::Unwind => EdgeKind::ExceptionUnwind,
        }
    }
}

impl InstrInfo for Instr {
    type Variable = Reg;

    fn uses(&self) -> &[Reg] {
        &self.uses
    }

    fn defs(&self) -> &[Reg] {
        &self.defs
    }

    fn may_unwind(&self) -> bool {
        self.may_throw()
    }
}

impl EffectInfo for Instr {
    type Effect = Effect;

    fn effects(&self) -> &[Effect] {
        match &self.form {
            Form::Apply(op) => op.effects(),
            Form::Store(_) => &[Effect::MemoryWrite],
            Form::Enter(_) | Form::Exit(_) => &[Effect::Output],
            Form::Caught => &[Effect::ExceptionState],
            _ => &[],
        }
    }
}

impl FlowControl for Instr {
    fn flow_effect(&self) -> FlowEffect {
        match self.form {
            Form::Branch | Form::BranchOn(_) => FlowEffect::ConditionalJump,
            Form::Switch => FlowEffect::IndirectJump,
            Form::Return => FlowEffect::Return,
            _ => FlowEffect::Fallthrough,
        }
    }
}

impl Predicated for Instr {
    fn predicate(&self) -> Option<(Reg, bool)> {
        self.guard
    }
}

impl LiftInstruction<Calc> for Instr {
    fn lift(&self) -> InstructionShape<Op, Lit> {
        match &self.form {
            Form::Const(literal) => InstructionShape::Constant(*literal),
            Form::Copy => InstructionShape::Copy,
            Form::Apply(op) => InstructionShape::Operation(*op),
            Form::Store(location) => InstructionShape::Store {
                location: *location,
            },
            Form::Move => InstructionShape::ParallelCopy,
            Form::Branch => InstructionShape::Branch,
            Form::BranchOn(op) => InstructionShape::BranchOperation(*op),
            Form::Switch => InstructionShape::Switch,
            Form::Return => InstructionShape::Return,
            Form::Caught => InstructionShape::Operation(Op::Caught),
            Form::Enter(_) => InstructionShape::Operation(Op::Acquire),
            Form::Exit(_) => InstructionShape::Operation(Op::Release),
            Form::Nop => InstructionShape::ControlFlow,
        }
    }

    fn use_types(&self) -> &[Ty] {
        &self.use_types
    }

    fn def_types(&self) -> &[Ty] {
        &self.def_types
    }

    fn previous_value_operand(&self) -> Option<usize> {
        self.previous_value
    }

    fn materialize_value(&self) -> bool {
        self.materialize
    }
}

impl LiftEdge<Calc> for Link {
    fn case_values(&self) -> &[Lit] {
        match self {
            Self::Case(values) => values,
            _ => &[],
        }
    }
}

/// A flat function: the graph, the register table, and the signature.
#[derive(Debug)]
pub(crate) struct Code {
    pub(crate) cfg: Cfg<Instr, Link>,
    pub(crate) locals: Vec<Local>,
    pub(crate) declared: Vec<Option<Ty>>,
    pub(crate) parameters: Vec<Reg>,
    pub(crate) returns: Vec<Ty>,
}

impl FlatFunction<Calc> for Code {
    type Var = Reg;
    type Inst = Instr;
    type Edge = Link;

    fn cfg(&self) -> &Cfg<Instr, Link> {
        &self.cfg
    }

    fn variable_count(&self) -> usize {
        self.locals.len()
    }

    fn variable(&self, variable: Reg) -> &Local {
        &self.locals[variable.index()]
    }

    fn parameters(&self) -> &[Reg] {
        &self.parameters
    }

    fn returns(&self) -> &[Ty] {
        &self.returns
    }
}

impl Code {
    /// The first block that holds code: the target of the entry edge.
    pub(crate) fn first_block(&self) -> BlockId {
        let root = self.cfg.entry();
        let entry = self
            .cfg
            .outgoing(root)
            .next()
            .expect("a sealed function has an entry edge");
        self.cfg.edge(entry).target()
    }

    /// The program points of every instruction, in block order.
    pub(crate) fn points(&self) -> Vec<ProgramPoint> {
        self.cfg
            .block_ids()
            .flat_map(|block| {
                (0..self.cfg.block(block).instructions().len())
                    .map(move |inst_idx| ProgramPoint { block, inst_idx })
            })
            .collect()
    }

    pub(crate) fn instruction(&self, point: ProgramPoint) -> &Instr {
        &self.cfg.block(point.block).instructions()[point.inst_idx]
    }
}

/// Why the builder refused to take or seal a function.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum CodeError {
    /// The bytecode has no instruction for this construct.
    Refused(&'static str),
    /// The sealed function broke the contract of the flat builder.
    Malformed(Vec<String>),
}

impl fmt::Display for CodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(what) => write!(formatter, "the bytecode has no form for {what}"),
            Self::Malformed(problems) => {
                write!(formatter, "malformed bytecode: {}", problems.join("; "))
            }
        }
    }
}

impl std::error::Error for CodeError {}

/// Builds a [`Code`], through `FlatBuilder` when `lower` drives it and
/// through the inherent methods when a test writes bytecode by hand.
#[derive(Debug)]
pub(crate) struct CodeBuilder {
    code: Code,
    temporaries: usize,
    regions: bool,
}

impl CodeBuilder {
    pub(crate) fn new() -> Self {
        Self {
            code: Code {
                cfg: Cfg::with_edge_payload(),
                locals: Vec::new(),
                declared: Vec::new(),
                parameters: Vec::new(),
                returns: Vec::new(),
            },
            temporaries: 0,
            regions: true,
        }
    }

    /// A builder whose bytecode has no region protocol.
    pub(crate) fn refusing_regions(mut self) -> Self {
        self.regions = false;
        self
    }

    /// Declares a typed register named `name`.
    pub(crate) fn register(&mut self, name: &str, ty: Ty) -> Reg {
        self.code.locals.push(Local::named(name));
        self.code.declared.push(Some(ty));
        Reg::from_index(self.code.locals.len() - 1)
    }

    pub(crate) fn block(&mut self) -> BlockId {
        self.code.cfg.new_block()
    }

    /// Appends an instruction to `block`; its occurrences take the declared
    /// types of the registers.
    pub(crate) fn push(&mut self, block: BlockId, form: Form, uses: &[Reg], defs: &[Reg]) {
        let occurrence = |&register: &Reg| Occurrence {
            variable: register,
            ty: self.code.declared[register.index()].expect("a hand-written register has a type"),
        };
        let uses: Vec<_> = uses.iter().map(occurrence).collect();
        let defs: Vec<_> = defs.iter().map(occurrence).collect();
        self.code
            .cfg
            .block_mut(block)
            .push(Instr::new(form, &uses, &defs));
    }

    pub(crate) fn link(&mut self, source: BlockId, target: BlockId, link: Link) -> EdgeId {
        let kind = link.kind();
        self.code
            .cfg
            .add_edge_with_payload(source, target, kind, link)
    }

    pub(crate) fn parameters(&mut self, parameters: &[Reg], returns: &[Ty]) {
        self.code.parameters = parameters.to_vec();
        self.code.returns = returns.to_vec();
    }

    /// Edits the instruction most recently pushed to `block`.
    pub(crate) fn amend_last(&mut self, block: BlockId, edit: impl FnOnce(&mut Instr)) {
        let instructions = self.code.cfg.block_mut(block).instructions_mut();
        edit(
            instructions
                .last_mut()
                .expect("the block holds an instruction"),
        );
    }

    /// The entry block, with the entry edge to `first` added.
    pub(crate) fn start_at(&mut self, first: BlockId) {
        let root = self.code.cfg.entry();
        self.link(root, first, Link::Entry);
    }

    /// Seals the function without checking it.
    pub(crate) fn seal(self) -> Code {
        self.code
    }

    fn form(&self, lowered: &Lowered<'_, Calc>) -> Result<Form, CodeError> {
        Ok(match *lowered {
            Lowered::Operation(op) => Form::Apply(*op),
            Lowered::Constant(literal) => Form::Const(*literal),
            Lowered::Copy => Form::Copy,
            Lowered::Store { location } => Form::Store(*location),
            Lowered::Branch => Form::Branch,
            Lowered::Switch => Form::Switch,
            Lowered::Return => Form::Return,
            Lowered::CaughtException => Form::Caught,
            Lowered::RegionEnter(op) if self.regions => Form::Enter(*op),
            Lowered::RegionExit(op) if self.regions => Form::Exit(*op),
            Lowered::RegionEnter(_) | Lowered::RegionExit(_) => {
                return Err(CodeError::Refused("a region"));
            }
        })
    }

    /// The contract that `finish` holds a sealed function to.
    fn problems(&self) -> Vec<String> {
        let cfg = &self.code.cfg;
        let mut problems: Vec<String> = verify(cfg)
            .errors
            .into_iter()
            .map(|error| error.message)
            .collect();
        let root = cfg.entry();
        let entries: Vec<EdgeId> = cfg.outgoing(root).collect();
        if !cfg.block(root).is_empty() || cfg.incoming(root).next().is_some() {
            problems.push("the root holds code or has a predecessor".into());
        }
        if entries.len() != 1 || *cfg.edge(entries[0]).payload() != Link::Entry {
            problems.push("the root needs exactly one entry edge".into());
        }
        for block in cfg.block_ids().filter(|&block| block != root) {
            let instructions = cfg.block(block).instructions();
            let links: Vec<&Link> = cfg
                .outgoing(block)
                .map(|id| cfg.edge(id).payload())
                .collect();
            let count =
                |wanted: fn(&Link) -> bool| links.iter().filter(|link| wanted(link)).count();
            let normal = count(|link| *link != Link::Unwind);
            if let Some(position) = instructions
                .iter()
                .position(Instr::is_terminator)
                .filter(|&position| position + 1 != instructions.len())
            {
                problems.push(format!(
                    "{block}: instruction {position} ends the block early"
                ));
            }
            match instructions.last().map(|last| &last.form) {
                Some(Form::Branch | Form::BranchOn(_)) => {
                    if count(|link| *link == Link::Yes) != 1 || count(|link| *link == Link::No) != 1
                    {
                        problems.push(format!("{block}: a branch needs one yes and one no edge"));
                    }
                }
                Some(Form::Switch) => {
                    if count(|link| *link == Link::Otherwise) != 1 {
                        problems.push(format!("{block}: a switch needs one default edge"));
                    }
                }
                Some(Form::Return) if normal != 0 => {
                    problems.push(format!("{block}: a return continues"));
                }
                Some(Form::Return) => {}
                _ if normal != 1 => {
                    problems.push(format!("{block}: a plain block needs exactly one way on"));
                }
                _ => {}
            }
            if links.contains(&&Link::Unwind) && !instructions.iter().any(Instr::may_throw) {
                problems.push(format!(
                    "{block}: an unwind edge leaves a block that cannot throw"
                ));
            }
        }
        problems
    }
}

impl FlatBuilder<Calc> for CodeBuilder {
    type Var = Reg;
    type Function = Code;
    type Error = CodeError;

    fn entry(&self) -> BlockId {
        self.code.cfg.entry()
    }

    fn declare_variable(&mut self, origin: VariableOrigin<'_, Calc>) -> Reg {
        let (local, declared) = match origin {
            VariableOrigin::Source {
                variable,
                declared_type,
            } => (variable.clone(), declared_type.copied()),
            VariableOrigin::Temporary { ty } => {
                self.temporaries += 1;
                let local = Local {
                    name: format!("%{}", self.temporaries),
                    temporary: true,
                };
                (local, Some(*ty))
            }
        };
        self.code.locals.push(local);
        self.code.declared.push(declared);
        Reg::from_index(self.code.locals.len() - 1)
    }

    fn set_signature(&mut self, parameters: &[Reg], returns: &[Ty]) {
        self.parameters(parameters, returns);
    }

    fn new_block(&mut self, label: Option<&str>) -> BlockId {
        let block = self.block();
        if let Some(label) = label {
            self.code.cfg.block_mut(block).set_label(label);
        }
        block
    }

    fn append(
        &mut self,
        block: BlockId,
        lowered: Lowered<'_, Calc>,
        uses: &[Occurrence<Reg, Ty>],
        defs: &[Occurrence<Reg, Ty>],
    ) -> Result<Appended, CodeError> {
        let instruction = Instr::new(self.form(&lowered)?, uses, defs);
        let may_throw = instruction.may_throw();
        let cfg = &mut self.code.cfg;
        let inst_idx = cfg.block(block).instructions().len();
        cfg.block_mut(block).push(instruction);
        Ok(Appended {
            point: ProgramPoint { block, inst_idx },
            may_throw,
        })
    }

    fn add_edge(&mut self, source: BlockId, target: BlockId, role: LowerEdge<'_, Lit>) -> EdgeId {
        let link = match role {
            LowerEdge::Entry => Link::Entry,
            LowerEdge::Fallthrough => Link::Next,
            LowerEdge::Jump => Link::Goto,
            LowerEdge::True => Link::Yes,
            LowerEdge::False => Link::No,
            LowerEdge::Case(value) => Link::Case(vec![*value]),
            LowerEdge::Default => Link::Otherwise,
            LowerEdge::Unwind => Link::Unwind,
        };
        self.link(source, target, link)
    }

    fn add_region(&mut self, region: Region) -> RegionId {
        self.code.cfg.add_region(region)
    }

    fn finish(self) -> Result<Code, CodeError> {
        let problems = self.problems();
        if problems.is_empty() {
            Ok(self.code)
        } else {
            Err(CodeError::Malformed(problems))
        }
    }
}
