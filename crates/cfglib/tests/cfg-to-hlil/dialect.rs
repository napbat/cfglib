//! The toy of a source-language frontend: statements over numbered
//! variables are the instructions of a `Cfg`, and the dialect names what
//! they mean in structured terms.
//!
//! Booleans are numbers (`less` gives 1 or 0), a condition holds when it is
//! not zero, and `quot` throws when it divides by zero.

use cfglib::ir::hlil::{
    Dialect, FlatFunction, FunctionView, InstructionShape, LiftDialect, LiftEdge, LiftInstruction,
    Pseudocode, VerificationIssue, VerifyDialect,
};
use cfglib::{Cfg, EdgeKind, EffectInfo, FlowControl, FlowEffect, InstrInfo, KindedEdge};

/// The marker of the dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Tiny;

/// The operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Prim {
    Plus,
    Less,
    /// Appends its operand to the output.
    Show,
    /// Throws `-1` for a zero divisor.
    Quot,
}

impl Prim {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Plus => "plus",
            Self::Less => "less",
            Self::Show => "show",
            Self::Quot => "quot",
        }
    }

    pub(crate) const fn arity(self) -> usize {
        match self {
            Self::Show => 1,
            _ => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Eff {
    Output,
}

impl Dialect for Tiny {
    type Op = Prim;
    type Constant = i64;
    type Type = ();
    type Variable = &'static str;
}

impl Pseudocode for Tiny {
    fn mnemonic(op: &Prim) -> &str {
        op.name()
    }
}

impl VerifyDialect for Tiny {
    /// Every operation has the arity of its primitive.
    fn verify<F: FunctionView<Self>>(function: &F, issues: &mut Vec<VerificationIssue>) {
        for index in 0..function.expr_count() {
            let expr = cfglib::DenseId::from_index(index);
            if let cfglib::ir::hlil::ExprShape::Operation { op, operands } = function.expr(expr)
                && operands.len() != op.arity()
            {
                issues.push(VerificationIssue::new(format!(
                    "e{index} gives {} {} operands",
                    op.name(),
                    operands.len()
                )));
            }
        }
    }
}

impl LiftDialect for Tiny {
    type Effect = Eff;

    fn void_type() {}
}

/// What one statement does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Form {
    Const(i64),
    Apply(Prim),
    /// Branches on the one use being non-zero.
    Test,
    /// Dispatches on the one use.
    Dispatch,
    Ret,
}

/// One statement of the source CFG, with the variables it reads and writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Stmt {
    pub(crate) form: Form,
    pub(crate) uses: Vec<usize>,
    pub(crate) defs: Vec<usize>,
}

pub(crate) fn konst(dst: usize, value: i64) -> Stmt {
    Stmt {
        form: Form::Const(value),
        uses: vec![],
        defs: vec![dst],
    }
}

pub(crate) fn set(dst: usize, prim: Prim, args: &[usize]) -> Stmt {
    Stmt {
        form: Form::Apply(prim),
        uses: args.to_vec(),
        defs: vec![dst],
    }
}

pub(crate) fn act(prim: Prim, args: &[usize]) -> Stmt {
    Stmt {
        form: Form::Apply(prim),
        uses: args.to_vec(),
        defs: vec![],
    }
}

pub(crate) fn test(condition: usize) -> Stmt {
    Stmt {
        form: Form::Test,
        uses: vec![condition],
        defs: vec![],
    }
}

pub(crate) fn dispatch(scrutinee: usize) -> Stmt {
    Stmt {
        form: Form::Dispatch,
        uses: vec![scrutinee],
        defs: vec![],
    }
}

pub(crate) fn ret(value: usize) -> Stmt {
    Stmt {
        form: Form::Ret,
        uses: vec![value],
        defs: vec![],
    }
}

impl Stmt {
    pub(crate) fn may_throw(&self) -> bool {
        self.form == Form::Apply(Prim::Quot)
    }
}

impl InstrInfo for Stmt {
    type Variable = usize;

    fn uses(&self) -> &[usize] {
        &self.uses
    }

    fn defs(&self) -> &[usize] {
        &self.defs
    }

    fn may_unwind(&self) -> bool {
        self.may_throw()
    }
}

impl EffectInfo for Stmt {
    type Effect = Eff;

    fn effects(&self) -> &[Eff] {
        if self.form == Form::Apply(Prim::Show) {
            &[Eff::Output]
        } else {
            &[]
        }
    }
}

impl FlowControl for Stmt {
    fn flow_effect(&self) -> FlowEffect {
        match self.form {
            Form::Test => FlowEffect::ConditionalJump,
            Form::Dispatch => FlowEffect::IndirectJump,
            Form::Ret => FlowEffect::Return,
            _ => FlowEffect::Fallthrough,
        }
    }
}

const UNITS: [(); 4] = [(); 4];

impl LiftInstruction<Tiny> for Stmt {
    fn lift(&self) -> InstructionShape<Prim, i64> {
        match self.form {
            Form::Const(value) => InstructionShape::Constant(value),
            Form::Apply(prim) => InstructionShape::Operation(prim),
            Form::Test => InstructionShape::Branch,
            Form::Dispatch => InstructionShape::Switch,
            Form::Ret => InstructionShape::Return,
        }
    }

    fn use_types(&self) -> &[()] {
        &UNITS[..self.uses.len()]
    }

    fn def_types(&self) -> &[()] {
        &UNITS[..self.defs.len()]
    }
}

/// The payload of an edge of the source CFG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Link {
    Next,
    Jump,
    True,
    False,
    Case(Vec<i64>),
    Default,
    Unwind,
}

impl KindedEdge for Link {
    fn kind(&self) -> EdgeKind {
        match self {
            Self::Next | Self::Default => EdgeKind::Fallthrough,
            Self::Jump => EdgeKind::Jump,
            Self::True => EdgeKind::ConditionalTrue,
            Self::False => EdgeKind::ConditionalFalse,
            Self::Case(_) => EdgeKind::SwitchCase,
            Self::Unwind => EdgeKind::ExceptionUnwind,
        }
    }
}

impl LiftEdge<Tiny> for Link {
    fn case_values(&self) -> &[i64] {
        match self {
            Self::Case(values) => values,
            _ => &[],
        }
    }
}

/// A source function: the CFG, the variable names, and the parameters.
pub(crate) struct Source {
    pub(crate) cfg: Cfg<Stmt, Link>,
    pub(crate) names: Vec<&'static str>,
    pub(crate) parameters: Vec<usize>,
}

impl FlatFunction<Tiny> for Source {
    type Var = usize;
    type Inst = Stmt;
    type Edge = Link;

    fn cfg(&self) -> &Cfg<Stmt, Link> {
        &self.cfg
    }

    fn variable_count(&self) -> usize {
        self.names.len()
    }

    fn variable(&self, variable: usize) -> &&'static str {
        &self.names[variable]
    }

    fn parameters(&self) -> &[usize] {
        &self.parameters
    }

    fn returns(&self) -> &[()] {
        &UNITS[..1]
    }
}
