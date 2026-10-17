//! `Vec`-backed flat storage for the toy dialects.
//!
//! [`ToyFlat`] is a [`Cfg`] of [`ToyInst`] and [`ToyEdge`] payloads plus a
//! variable table and a signature. [`ToyFlatBuilder`] builds one, through
//! [`FlatBuilder`] when lowering drives it and through its inherent methods
//! when a test authors a flat function by hand. The lift-input trait impls
//! live beside the lifter tests, in `flat_lift.rs`.
//!
//! The flat operation of an instruction is its [`ToyForm`], which carries the
//! dialect's own operations and constants unchanged. Nothing is renamed on the
//! way down, so a lowered function lifts back to the same operations.

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{self, Write};

use crate::block::BlockId;
use crate::dataflow::ProgramPoint;
use crate::edge::{EdgeId, EdgeKind, KindedEdge};
use crate::graph::view::DenseId;
use crate::ir::hlil::{
    Appended, Dialect, FlatBuilder, LiftDialect, LowerEdge, Lowered, Occurrence, Pseudocode,
    VariableOrigin, VerificationIssue, VerificationReport,
};
use crate::region::{HandlerBody, HandlerKind, Region, RegionId};
use crate::{Cfg, verify};

use super::{Imp, Kit, Prim, Sym, Ty};

dense_id!(
    /// A variable of a flat toy function.
    FlatVar
);

/// How an instruction touches memory. A toy dialect maps it to its own
/// effect vocabulary through [`FlatKit::effect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Access {
    Read,
    Write,
}

/// What the flat toy storage needs from a toy dialect, beyond [`Kit`].
pub(crate) trait FlatKit: Kit + LiftDialect {
    /// The effect that stands for `access`.
    fn effect(access: Access) -> Self::Effect;

    /// The kind of a synthesized [`Jump`](ToyEdge::Jump) edge. The two toys
    /// differ, so a test body over both covers both conventions.
    fn jump_kind() -> EdgeKind;

    /// The kind of an [`Except`](ToyEdge::Except) edge.
    fn unwind_kind() -> EdgeKind;

    /// Whether the flat operation `op` may transfer exceptionally. Calls do.
    fn throws(op: &Self::Op) -> bool {
        *op == Self::op(Prim::Call) || *op == Self::op(Prim::Expanded)
    }

    /// How the flat operation `op` touches memory, if it does. A throwing
    /// operation writes, and a load reads.
    fn access(op: &Self::Op) -> Option<Access> {
        if Self::throws(op) {
            Some(Access::Write)
        } else {
            (*op == Self::op(Prim::Load)).then_some(Access::Read)
        }
    }
}

impl FlatKit for Imp {
    fn effect(access: Access) -> Access {
        access
    }

    fn jump_kind() -> EdgeKind {
        EdgeKind::Jump
    }

    fn unwind_kind() -> EdgeKind {
        EdgeKind::ExceptionHandler
    }
}

impl FlatKit for Sym {
    fn effect(access: Access) -> &'static str {
        match access {
            Access::Read => "read",
            Access::Write => "write",
        }
    }

    fn jump_kind() -> EdgeKind {
        EdgeKind::Unconditional
    }

    fn unwind_kind() -> EdgeKind {
        EdgeKind::ExceptionUnwind
    }
}

/// What a flat instruction is. This is the shape `lift` classifies and
/// `lower` produces, with the dialect's operations carried verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToyForm<D: Dialect> {
    /// Materializes a constant into its definition.
    Constant(D::Constant),
    /// Copies its use into its definition.
    Copy,
    /// Applies an operation to the uses.
    Operation(D::Op),
    /// Stores the last use into the place that the operation forms over the
    /// preceding uses.
    Store(D::Op),
    /// Pairwise parallel move of uses to definitions.
    Exchange,
    /// Conditional branch on the one use.
    Branch,
    /// Conditional branch deciding on the operation applied to every use.
    BranchOperation(D::Op),
    /// Multi-way dispatch on the one use.
    Switch,
    /// Pure control transfer with no HLIL form.
    Jump,
    /// Return of the uses.
    Return,
    /// Defines a handler binding with the delivered exception.
    CaughtException,
    /// Enters the region that the operation names.
    RegionEnter(D::Op),
    /// Exits the region that the operation names.
    RegionExit(D::Op),
}

/// One flat toy instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToyInst<D: LiftDialect> {
    pub(crate) form: ToyForm<D>,
    pub(crate) uses: Vec<FlatVar>,
    pub(crate) defs: Vec<FlatVar>,
    /// Parallel to `uses`.
    pub(crate) use_types: Vec<D::Type>,
    /// Parallel to `defs`.
    pub(crate) def_types: Vec<D::Type>,
    pub(crate) effects: Vec<D::Effect>,
    pub(crate) may_throw: bool,
    /// The use position that reads the definition's previous value.
    pub(crate) previous_value: Option<usize>,
    /// Whether the definition must remain a visible statement.
    pub(crate) materialize: bool,
}

impl<D: FlatKit> ToyInst<D> {
    /// An instruction of `form` over the occurrences. Its effects and its
    /// ability to throw follow from the operation: a call writes and throws,
    /// a store writes.
    pub(crate) fn new(
        form: ToyForm<D>,
        uses: &[Occurrence<FlatVar, D::Type>],
        defs: &[Occurrence<FlatVar, D::Type>],
    ) -> Self {
        let (access, may_throw) = match &form {
            ToyForm::Operation(op) => (D::access(op), D::throws(op)),
            ToyForm::Store(_) => (Some(Access::Write), false),
            _ => (None, false),
        };
        Self {
            form,
            uses: uses.iter().map(|occurrence| occurrence.variable).collect(),
            defs: defs.iter().map(|occurrence| occurrence.variable).collect(),
            use_types: uses
                .iter()
                .map(|occurrence| occurrence.ty.clone())
                .collect(),
            def_types: defs
                .iter()
                .map(|occurrence| occurrence.ty.clone())
                .collect(),
            effects: access.into_iter().map(D::effect).collect(),
            may_throw,
            previous_value: None,
            materialize: false,
        }
    }

    /// Replaces the effects.
    pub(crate) fn with_effects(mut self, effects: Vec<D::Effect>) -> Self {
        self.effects = effects;
        self
    }

    /// Replaces whether the instruction may throw.
    pub(crate) fn with_may_throw(mut self, may_throw: bool) -> Self {
        self.may_throw = may_throw;
        self
    }

    /// Marks use `position` as the read of the definition's previous value.
    pub(crate) fn reading_previous_value(mut self, position: usize) -> Self {
        self.previous_value = Some(position);
        self
    }

    /// Marks the definition as one that must stay a visible statement.
    pub(crate) fn materialized(mut self) -> Self {
        self.materialize = true;
        self
    }
}

/// The payload of a flat toy edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToyEdge<D: Dialect> {
    /// The synthetic root to the first block.
    Entry,
    True,
    False,
    /// Sequential flow.
    Fall,
    /// A synthesized transfer. Its kind is [`FlatKit::jump_kind`].
    Jump,
    /// One or more switch case values.
    Case(Vec<D::Constant>),
    /// Taken when no switch case matches.
    Default,
    /// Exceptional, to a handler entry. Its kind is [`FlatKit::unwind_kind`].
    Except,
}

impl<D: FlatKit> KindedEdge for ToyEdge<D> {
    fn kind(&self) -> EdgeKind {
        match self {
            Self::Entry | Self::Fall | Self::Default => EdgeKind::Fallthrough,
            Self::True => EdgeKind::ConditionalTrue,
            Self::False => EdgeKind::ConditionalFalse,
            Self::Jump => D::jump_kind(),
            Self::Case(_) => EdgeKind::SwitchCase,
            Self::Except => D::unwind_kind(),
        }
    }
}

/// One variable of a flat toy function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToyVariable<D: Dialect> {
    pub(crate) variable: D::Variable,
    pub(crate) declared_type: Option<D::Type>,
    /// Whether lowering introduced it.
    pub(crate) temporary: bool,
}

/// A flat function: a [`Cfg`] plus the variable table and the signature.
#[derive(Debug)]
pub(crate) struct ToyFlat<D: LiftDialect> {
    pub(crate) cfg: Cfg<ToyInst<D>, ToyEdge<D>>,
    pub(crate) variables: Vec<ToyVariable<D>>,
    pub(crate) parameters: Vec<FlatVar>,
    pub(crate) returns: Vec<D::Type>,
}

/// The number that names the metadata of a lowering temporary in the
/// dialect's [`Kit::variable`].
const TEMPORARY: u8 = u8::MAX;

/// Builds a [`ToyFlat`].
pub(crate) struct ToyFlatBuilder<D: FlatKit> {
    cfg: Cfg<ToyInst<D>, ToyEdge<D>>,
    variables: Vec<ToyVariable<D>>,
    parameters: Vec<FlatVar>,
    returns: Vec<D::Type>,
    caught_exceptions: bool,
    regions: bool,
}

impl<D: FlatKit> ToyFlatBuilder<D> {
    /// A builder holding the synthetic root.
    pub(crate) fn new() -> Self {
        Self {
            cfg: Cfg::with_edge_payload(),
            variables: Vec::new(),
            parameters: Vec::new(),
            returns: Vec::new(),
            caught_exceptions: true,
            regions: true,
        }
    }

    /// A builder with no caught-exception form: lowering a handler binding
    /// fails.
    pub(crate) fn refusing_caught_exceptions(mut self) -> Self {
        self.caught_exceptions = false;
        self
    }

    /// A builder with no region protocol: lowering a region statement fails.
    pub(crate) fn refusing_regions(mut self) -> Self {
        self.regions = false;
        self
    }

    /// Declares an integer variable that stands for number `number`.
    pub(crate) fn int_variable(&mut self, number: u8) -> FlatVar {
        self.declare_variable(VariableOrigin::Source {
            variable: &D::variable(number),
            declared_type: Some(&D::ty(Ty::Integer)),
        })
    }

    /// Appends `instruction` to `block`.
    pub(crate) fn push(&mut self, block: BlockId, instruction: ToyInst<D>) -> Appended {
        let may_throw = instruction.may_throw;
        let inst_idx = self.cfg.block(block).instructions().len();
        self.cfg.block_mut(block).push(instruction);
        Appended {
            point: ProgramPoint { block, inst_idx },
            may_throw,
        }
    }

    /// Adds an edge with the payload `edge`.
    pub(crate) fn connect(&mut self, source: BlockId, target: BlockId, edge: ToyEdge<D>) -> EdgeId {
        let kind = edge.kind();
        self.cfg.add_edge_with_payload(source, target, kind, edge)
    }

    /// Seals the function without checking it.
    pub(crate) fn finish_unchecked(self) -> ToyFlat<D> {
        ToyFlat {
            cfg: self.cfg,
            variables: self.variables,
            parameters: self.parameters,
            returns: self.returns,
        }
    }

    fn form(&self, lowered: &Lowered<'_, D>) -> Result<ToyForm<D>, VerificationReport> {
        Ok(match *lowered {
            Lowered::Operation(op) => ToyForm::Operation(op.clone()),
            Lowered::Constant(constant) => ToyForm::Constant(constant.clone()),
            Lowered::Copy => ToyForm::Copy,
            Lowered::Store { location } => ToyForm::Store(location.clone()),
            Lowered::Branch => ToyForm::Branch,
            Lowered::Switch => ToyForm::Switch,
            Lowered::Return => ToyForm::Return,
            Lowered::CaughtException if self.caught_exceptions => ToyForm::CaughtException,
            Lowered::CaughtException => return Err(refusal("a caught exception")),
            Lowered::RegionEnter(op) if self.regions && *op == D::op(Prim::Acquire) => {
                ToyForm::RegionEnter(op.clone())
            }
            Lowered::RegionExit(op) if self.regions && *op == D::op(Prim::Acquire) => {
                ToyForm::RegionExit(op.clone())
            }
            Lowered::RegionEnter(_) | Lowered::RegionExit(_) => {
                return Err(refusal("a region enter or exit"));
            }
        })
    }
}

fn refusal(what: &str) -> VerificationReport {
    VerificationReport::new(alloc::vec![VerificationIssue::new(format!(
        "the toy flat builder has no form for {what}"
    ))])
}

impl<D: FlatKit> FlatBuilder<D> for ToyFlatBuilder<D> {
    type Var = FlatVar;
    type Function = ToyFlat<D>;
    type Error = VerificationReport;

    fn entry(&self) -> BlockId {
        self.cfg.entry()
    }

    fn declare_variable(&mut self, origin: VariableOrigin<'_, D>) -> FlatVar {
        let variable = match origin {
            VariableOrigin::Source {
                variable,
                declared_type,
            } => ToyVariable {
                variable: variable.clone(),
                declared_type: declared_type.cloned(),
                temporary: false,
            },
            VariableOrigin::Temporary { ty } => ToyVariable {
                variable: D::variable(TEMPORARY),
                declared_type: Some(ty.clone()),
                temporary: true,
            },
        };
        self.variables.push(variable);
        FlatVar::from_index(self.variables.len() - 1)
    }

    fn set_signature(&mut self, parameters: &[FlatVar], returns: &[D::Type]) {
        self.parameters = parameters.to_vec();
        self.returns = returns.to_vec();
    }

    fn new_block(&mut self, label: Option<&str>) -> BlockId {
        let block = self.cfg.new_block();
        if let Some(label) = label {
            self.cfg.block_mut(block).set_label(label);
        }
        block
    }

    fn append(
        &mut self,
        block: BlockId,
        lowered: Lowered<'_, D>,
        uses: &[Occurrence<FlatVar, D::Type>],
        defs: &[Occurrence<FlatVar, D::Type>],
    ) -> Result<Appended, VerificationReport> {
        let form = self.form(&lowered)?;
        Ok(self.push(block, ToyInst::new(form, uses, defs)))
    }

    fn add_edge(
        &mut self,
        source: BlockId,
        target: BlockId,
        role: LowerEdge<'_, D::Constant>,
    ) -> EdgeId {
        let edge = match role {
            LowerEdge::Entry => ToyEdge::Entry,
            LowerEdge::Fallthrough => ToyEdge::Fall,
            LowerEdge::Jump => ToyEdge::Jump,
            LowerEdge::True => ToyEdge::True,
            LowerEdge::False => ToyEdge::False,
            LowerEdge::Case(value) => ToyEdge::Case(alloc::vec![value.clone()]),
            LowerEdge::Default => ToyEdge::Default,
            LowerEdge::Unwind => ToyEdge::Except,
        };
        self.connect(source, target, edge)
    }

    fn add_region(&mut self, region: Region) -> RegionId {
        self.cfg.add_region(region)
    }

    fn finish(self) -> Result<ToyFlat<D>, VerificationReport> {
        let function = self.finish_unchecked();
        let report = function.check();
        if report.is_ok() {
            Ok(function)
        } else {
            Err(report)
        }
    }
}

impl<D: FlatKit> ToyFlat<D> {
    /// The invariants a sealed function holds: a well-formed graph, a
    /// synthetic root with one entry edge, declared variables, terminators
    /// last with the edges their kind needs, and exceptional edges only out
    /// of blocks that hold a throwing instruction.
    pub(crate) fn check(&self) -> VerificationReport {
        let mut issues = Vec::new();
        let mut report = |message: String| issues.push(VerificationIssue::new(message));
        for error in verify(&self.cfg).errors {
            report(format!("control-flow graph: {}", error.message));
        }
        let root = self.cfg.entry();
        let entries: Vec<EdgeId> = self.cfg.outgoing(root).collect();
        if !self.cfg.block(root).is_empty() {
            report("the synthetic root holds an instruction".into());
        }
        if self.cfg.incoming(root).next().is_some() {
            report("the synthetic root has an incoming edge".into());
        }
        if entries.len() != 1 || *self.cfg.edge(entries[0]).payload() != ToyEdge::Entry {
            report("the synthetic root needs exactly one entry edge".into());
        }
        for parameter in &self.parameters {
            if parameter.index() >= self.variables.len() {
                report(format!("parameter {parameter:?} is not declared"));
            }
        }
        for block in self.cfg.block_ids() {
            let instructions = self.cfg.block(block).instructions();
            let edges: Vec<_> = self
                .cfg
                .outgoing(block)
                .map(|id| self.cfg.edge(id))
                .collect();
            let normal = |payload: fn(&ToyEdge<D>) -> bool| {
                edges.iter().filter(|edge| payload(edge.payload())).count()
            };
            for (index, instruction) in instructions.iter().enumerate() {
                for variable in instruction.uses.iter().chain(&instruction.defs) {
                    if variable.index() >= self.variables.len() {
                        report(format!("{block}:{index} names undeclared {variable:?}"));
                    }
                }
                let terminates = matches!(
                    instruction.form,
                    ToyForm::Branch | ToyForm::Switch | ToyForm::Return
                );
                if terminates && index + 1 != instructions.len() {
                    report(format!("{block}:{index} terminates before the block ends"));
                }
            }
            match instructions.last().map(|last| &last.form) {
                Some(ToyForm::Branch)
                    if normal(|edge| *edge == ToyEdge::True) != 1
                        || normal(|edge| *edge == ToyEdge::False) != 1 =>
                {
                    report(format!(
                        "{block} branches without one true and one false edge"
                    ));
                }
                Some(ToyForm::Switch) if normal(|edge| *edge == ToyEdge::Default) != 1 => {
                    report(format!("{block} dispatches without one default edge"));
                }
                Some(ToyForm::Return) if normal(|edge| !matches!(edge, ToyEdge::Except)) != 0 => {
                    report(format!("{block} returns and then continues"));
                }
                _ => {}
            }
            if instructions.is_empty()
                && edges
                    .iter()
                    .filter(|edge| !edge.kind().is_exceptional())
                    .count()
                    > 1
            {
                report(format!("empty {block} decides between several edges"));
            }
            if edges.iter().any(|edge| edge.kind().is_exceptional())
                && !instructions.iter().any(|instruction| instruction.may_throw)
            {
                report(format!(
                    "{block} has an exceptional edge and nothing that throws"
                ));
            }
        }
        VerificationReport::new(issues)
    }

    /// A canonical text of the function, for structural comparison: the
    /// signature, then every block with its instructions and outgoing edges
    /// (target and payload), then every region. Variables and blocks are
    /// named by index. Constants print in the dialect's own spelling, which
    /// both toys share, so one text stands for the function in either toy.
    pub(crate) fn dump(&self) -> String {
        let mut text = String::new();
        let _ = writeln!(
            text,
            "signature ({}) returns {}",
            names(&self.parameters),
            self.returns.len()
        );
        for (index, variable) in self.variables.iter().enumerate() {
            let role = if variable.temporary {
                "temporary"
            } else {
                "source"
            };
            let _ = writeln!(text, "v{index} {role}");
        }
        for block in self.cfg.block_ids() {
            match self.cfg.block(block).label() {
                Some(label) => {
                    let _ = writeln!(text, "{block} {label:?}");
                }
                None => {
                    let _ = writeln!(text, "{block}");
                }
            }
            for instruction in self.cfg.block(block).instructions() {
                let _ = writeln!(text, "  {}", instruction.describe());
            }
            for id in self.cfg.outgoing(block) {
                let edge = self.cfg.edge(id);
                let _ = writeln!(text, "  -> {} {}", edge.target(), edge.payload().describe());
            }
        }
        for region in self.cfg.regions() {
            let _ = writeln!(text, "{} protects {:?}", region.id, region.protected_blocks);
            for handler in &region.handlers {
                let body = match &handler.body {
                    HandlerBody::Known(blocks) => format!("{blocks:?}"),
                    HandlerBody::Unknown => "unknown".into(),
                };
                let kind = match handler.kind {
                    HandlerKind::Filter { filter_block } => format!("Filter({filter_block})"),
                    other => format!("{other:?}"),
                };
                let _ = writeln!(text, "  handler {} {kind} {body}", handler.entry);
            }
        }
        text
    }
}

/// The names of `variables`, comma separated.
fn names(variables: &[FlatVar]) -> String {
    variables
        .iter()
        .map(|variable| format!("v{}", variable.index()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A constant in the dialect's own spelling.
struct Spelled<'a, D: Pseudocode>(&'a D::Constant);

impl<D: Pseudocode> fmt::Display for Spelled<'_, D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        D::fmt_constant(formatter, self.0)
    }
}

impl<D: FlatKit> ToyEdge<D> {
    /// The payload as one word, with the case values of a case edge.
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::Entry => "Entry".into(),
            Self::True => "True".into(),
            Self::False => "False".into(),
            Self::Fall => "Fall".into(),
            Self::Jump => "Jump".into(),
            Self::Default => "Default".into(),
            Self::Except => "Except".into(),
            Self::Case(values) => {
                let values: Vec<String> = values
                    .iter()
                    .map(|value| Spelled::<D>(value).to_string())
                    .collect();
                format!("Case {}", values.join(" "))
            }
        }
    }
}

impl<D: FlatKit> ToyInst<D> {
    /// One line: the definitions, the form, the uses, and the flags that
    /// are not the default: `!` may throw, `prev=n` reads the previous value
    /// at use `n`, `keep` must stay a statement. The effects are
    /// dialect-specific and stay out of the text.
    pub(crate) fn describe(&self) -> String {
        let form = match &self.form {
            ToyForm::Constant(constant) => format!("const {}", Spelled::<D>(constant)),
            ToyForm::Copy => "copy".into(),
            ToyForm::Operation(op) => D::mnemonic(op).into(),
            ToyForm::Store(op) => format!("store@{}", D::mnemonic(op)),
            ToyForm::Exchange => "exchange".into(),
            ToyForm::Branch => "branch".into(),
            ToyForm::BranchOperation(op) => format!("branch@{}", D::mnemonic(op)),
            ToyForm::Switch => "switch".into(),
            ToyForm::Jump => "jump".into(),
            ToyForm::Return => "return".into(),
            ToyForm::CaughtException => "caught".into(),
            ToyForm::RegionEnter(op) => format!("enter@{}", D::mnemonic(op)),
            ToyForm::RegionExit(op) => format!("exit@{}", D::mnemonic(op)),
        };
        // A constant has no operands to list.
        let operands = if matches!(self.form, ToyForm::Constant(_)) {
            String::new()
        } else {
            format!("({})", names(&self.uses))
        };
        let mut text = if self.defs.is_empty() {
            format!("{form}{operands}")
        } else {
            format!("{} = {form}{operands}", names(&self.defs))
        };
        if self.may_throw {
            text.push_str(" !");
        }
        if let Some(position) = self.previous_value {
            let _ = write!(text, " prev={position}");
        }
        if self.materialize {
            text.push_str(" keep");
        }

        text
    }
}
