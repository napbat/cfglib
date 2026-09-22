//! A managed-language test dialect: verifier-style reference
//! constraints over a class hierarchy, distinct per-level edge
//! vocabularies with exact throw-site identities, dispatch, semantic
//! multi-instruction emission, and the MLIL → RTL lowering direction.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::EdgeKind;
use crate::ir::dialect::Vocabulary;
use crate::ir::hlil::{self, Lifted, StatementKind, lift_function as lift_hlil};
use crate::ir::mlil::{self, InstructionId, InstructionMetadata, TypedVariable};
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::toy::{self, Span};

use super::super::{
    Constraint, Dialect, EdgeContext, Emission, Expr, FunctionBuilder, Lift, LiftedStatement,
    Lower, LowerContext, LowerEdgeContext, MlilBridge, Place, Placement, Result, ScalarType, Shape,
    Signature, Statement, StatementId, VarExpr, lift, lower,
};

/// Exceptional-flow tests: throw-site ownership, continuation splits,
/// emission validation, and cross-domain edge remapping.
mod exceptional;
/// The lowering direction for an effect that declares its writes.
mod writes;

/// A two-level class hierarchy: `parents` maps each class to its
/// superclass; class 0 is the root.
#[derive(Debug, Clone, Default)]
struct Hierarchy {
    parents: BTreeMap<u8, u8>,
}

impl Hierarchy {
    fn ancestors(&self, class: u8) -> Vec<u8> {
        let mut chain = vec![class];
        let mut cursor = class;
        while let Some(&parent) = self.parents.get(&cursor) {
            chain.push(parent);
            cursor = parent;
        }
        chain
    }

    /// The nearest common ancestor of two classes.
    fn join(&self, a: u8, b: u8) -> u8 {
        let ancestors = self.ancestors(b);
        self.ancestors(a)
            .into_iter()
            .find(|class| ancestors.contains(class))
            .unwrap_or(0)
    }
}

fn hierarchy() -> Hierarchy {
    Hierarchy {
        parents: [(1, 0), (2, 0)].into_iter().collect(),
    }
}

/// Verifier-style lane constraints over a two-level class hierarchy:
/// class 0 is the root, classes 1 and 2 extend it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum JvmConstraint {
    /// A numeric lane, delegating to the provided scalar domain.
    Word(ScalarType),
    /// An exact reference type.
    Reference(u8),
    /// The null literal — merges into any reference.
    Null,
    /// No constraint yet.
    Unknown,
    /// Irreconcilable observations.
    Conflict,
}

impl Constraint for JvmConstraint {
    type Context = Hierarchy;

    fn free() -> Self {
        Self::Unknown
    }

    fn conflicted() -> Self {
        Self::Conflict
    }

    fn merge(&self, other: &Self, context: &Hierarchy) -> Option<Self> {
        match (self, other) {
            (a, b) if a == b => Some(*a),
            (Self::Word(a), Self::Word(b)) => Constraint::merge(a, b, &()).map(Self::Word),
            (Self::Reference(a), Self::Reference(b)) => Some(Self::Reference(context.join(*a, *b))),
            (Self::Reference(class), Self::Null) | (Self::Null, Self::Reference(class)) => {
                Some(Self::Reference(*class))
            }
            _ => None,
        }
    }

    fn width(&self) -> Option<u32> {
        match self {
            Self::Word(scalar) => Constraint::width(scalar),
            _ => None,
        }
    }
}

type JvmShape = Shape<JvmConstraint>;

/// RTL-level edge metadata: an exceptional edge names its owning throw
/// in the statement identity domain, self-contained at this level.
#[derive(Debug, Clone, PartialEq, Eq)]
enum JvmRtlEdge {
    Entry,
    Fall,
    True,
    False,
    Case(i64),
    Except { site: Option<StatementId> },
}

/// MLIL-level edge metadata: an exceptional edge names its exact
/// emitted throw-site instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
enum JvmMlilEdge {
    Entry,
    Fall,
    True,
    False,
    Case(i64),
    Except { site: Option<InstructionId> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operator {
    /// Numeric addition — the managed dialect *expands* assignments of
    /// this operator into two MLIL instructions to exercise semantic
    /// emission.
    Add,
    /// An allocation of one exact class.
    New(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectOp {
    Invoke,
    Throw,
    /// Deliberately drops the statement's exceptional behavior — the
    /// emission validation must reject it.
    DropThrow,
    /// Deliberately references a variable outside the statement's reads.
    Smuggle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Effect {
    Call,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Managed;

impl Vocabulary for Managed {
    type ValueType = JvmShape;
    type Effect = Effect;
    type Source = String;
    type SourceSpan = Span;
    type SourcePoint = u32;
    type VariableRole = u8;
    type NativeVariable = u8;

    fn span_is_empty(span: &Self::SourceSpan) -> bool {
        toy::span_is_empty(*span)
    }

    fn span_contains(span: &Self::SourceSpan, point: &Self::SourcePoint) -> bool {
        toy::span_contains(*span, *point)
    }
}

impl Dialect for Managed {
    type Constraint = JvmConstraint;
    type Operator = Operator;
    type EffectOp = EffectOp;
    type Edge = JvmRtlEdge;

    fn mnemonic(operator: &Self::Operator) -> &str {
        match operator {
            Operator::Add => "add",
            Operator::New(_) => "new",
        }
    }

    fn effect_mnemonic(operation: &Self::EffectOp) -> &str {
        match operation {
            EffectOp::Invoke => "invoke",
            EffectOp::Throw => "throw",
            EffectOp::DropThrow => "drop-throw",
            EffectOp::Smuggle => "smuggle",
        }
    }

    fn edge_kind(edge: &Self::Edge) -> EdgeKind {
        match edge {
            JvmRtlEdge::Entry | JvmRtlEdge::Fall => EdgeKind::Fallthrough,
            JvmRtlEdge::True => EdgeKind::ConditionalTrue,
            JvmRtlEdge::False => EdgeKind::ConditionalFalse,
            JvmRtlEdge::Case(_) => EdgeKind::SwitchCase,
            JvmRtlEdge::Except { .. } => EdgeKind::ExceptionHandler,
        }
    }

    fn is_entry_edge(edge: &Self::Edge) -> bool {
        *edge == JvmRtlEdge::Entry
    }
}

impl mlil::Dialect for Managed {
    type Operation = LiftedStatement<Managed>;
    type Edge = JvmMlilEdge;

    fn instruction_metadata(
        operation: &Self::Operation,
        may_throw: bool,
    ) -> InstructionMetadata<Self::Effect> {
        toy::lifted_statement_metadata(operation, may_throw)
    }

    fn mnemonic(operation: &Self::Operation) -> &str {
        operation.mnemonic()
    }

    fn edge_kind(edge: &Self::Edge) -> EdgeKind {
        match edge {
            JvmMlilEdge::Entry | JvmMlilEdge::Fall => EdgeKind::Fallthrough,
            JvmMlilEdge::True => EdgeKind::ConditionalTrue,
            JvmMlilEdge::False => EdgeKind::ConditionalFalse,
            JvmMlilEdge::Case(_) => EdgeKind::SwitchCase,
            JvmMlilEdge::Except { .. } => EdgeKind::ExceptionHandler,
        }
    }

    fn is_entry_edge(edge: &Self::Edge) -> bool {
        *edge == JvmMlilEdge::Entry
    }
}

impl MlilBridge for Managed {
    type Mlil = Self;
}

impl mlil::AnalysisDialect for Managed {
    type Constant = Vec<u64>;
    type ExpressionOperator = Operator;
    type Callee = u32;

    fn is_copy(_operation: &Self::Operation) -> bool {
        false
    }

    fn expression_operator(_operation: &Self::Operation) -> Option<Self::ExpressionOperator> {
        None
    }

    fn constant(_operation: &Self::Operation) -> Option<Self::Constant> {
        None
    }

    fn fold_constant(
        _instruction: &mlil::Instruction<Self>,
        _known: &BTreeMap<mlil::VariableId, Self::Constant>,
    ) -> Option<(mlil::VariableId, Self::Constant)> {
        None
    }

    fn callee(_operation: &Self::Operation) -> Option<Self::Callee> {
        None
    }
}

impl mlil::VerifyDialect for Managed {
    fn verify(_function: &mlil::Function<Self>, _issues: &mut Vec<mlil::VerificationIssue>) {}
}

impl hlil::Dialect for Managed {
    type Operation = LiftedStatement<Managed>;
    type Constant = Vec<u64>;

    fn mnemonic(operation: &Self::Operation) -> &str {
        operation.mnemonic()
    }
}

impl hlil::VerifyDialect for Managed {
    fn verify(_function: &hlil::Function<Self>, _issues: &mut Vec<hlil::VerificationIssue>) {}
}

impl hlil::LiftDialect for Managed {
    fn lift_operation(operation: &LiftedStatement<Self>) -> Lifted<LiftedStatement<Self>> {
        operation.lifted()
    }

    fn case_values(edge: &JvmMlilEdge) -> Vec<Vec<u64>> {
        match edge {
            #[expect(clippy::cast_sign_loss, reason = "case value as raw pattern")]
            JvmMlilEdge::Case(value) => vec![vec![*value as u64]],
            _ => Vec::new(),
        }
    }

    fn void_type() -> JvmShape {
        JvmShape::vector(JvmConstraint::Unknown, 0)
    }

    fn previous_value_operand(operation: &LiftedStatement<Self>) -> Option<usize> {
        operation.merge_operand()
    }
}

impl Lift for Managed {
    fn value_type(shape: JvmShape) -> JvmShape {
        shape
    }

    fn web_role(storage: Option<&u8>) -> u8 {
        u8::from(storage.is_none())
    }

    fn native_variable(storage: &u8, _source: &String) -> Option<u8> {
        Some(*storage)
    }

    fn emit(context: &mut Emission<'_, '_, Self>, statement: LiftedStatement<Self>) -> Result<()> {
        match &statement {
            // Deliberate misbehavior markers for validation tests.
            LiftedStatement::Effect {
                operation: EffectOp::DropThrow,
                ..
            } => {
                let reads = context.reads().to_vec();
                context.append(statement.clone(), reads, Vec::new(), false)?;
                Ok(())
            }
            LiftedStatement::Effect {
                operation: EffectOp::Smuggle,
                ..
            } => {
                let foreign = TypedVariable::new(
                    mlil::VariableId::from_raw(999),
                    JvmShape::scalar(JvmConstraint::Unknown),
                );
                context.append(statement.clone(), vec![foreign], Vec::new(), false)?;
                Ok(())
            }
            // Semantic expansion: an addition computes into a dialect
            // temporary, then commits to the native target. A locally handled
            // throw uses a fresh continuation block so native state commits
            // only on the normal path. An unhandled throw can define the
            // target directly because no in-function handler observes its
            // pre-state.
            LiftedStatement::Assign {
                positions,
                width,
                value:
                    VarExpr::Apply {
                        operator: Operator::Add,
                        ..
                    },
                ..
            } => {
                if context.may_throw() && !context.has_exceptional_successors() {
                    context.single(statement)?;
                    return Ok(());
                }
                let target = context
                    .target()
                    .expect("assignments define a target")
                    .clone();
                let temporary = context.temporary(9, target.value_type)?;
                let compute = statement.clone();
                let reads = context.reads().to_vec();
                let may_throw = context.may_throw();
                context.append(compute, reads, vec![temporary.clone()], may_throw)?;
                context.continuation(JvmMlilEdge::Fall)?;
                let commit = LiftedStatement::Assign {
                    positions: positions.clone(),
                    width: *width,
                    merges: false,
                    value: VarExpr::Read {
                        positions: (0..*width).collect(),
                        scalar: JvmConstraint::Unknown,
                    },
                    effects: Vec::new(),
                };
                context.append(commit, vec![temporary], vec![target], false)?;
                Ok(())
            }
            _ => {
                context.single(statement)?;
                Ok(())
            }
        }
    }

    fn lift_edge(edge: &JvmRtlEdge, context: &EdgeContext<'_>) -> Result<JvmMlilEdge> {
        Ok(match edge {
            JvmRtlEdge::Entry => JvmMlilEdge::Entry,
            JvmRtlEdge::Fall => JvmMlilEdge::Fall,
            JvmRtlEdge::True => JvmMlilEdge::True,
            JvmRtlEdge::False => JvmMlilEdge::False,
            JvmRtlEdge::Case(value) => JvmMlilEdge::Case(*value),
            // The exceptional payload crosses into the MLIL identity
            // domain: the owning statement's emitted throw site.
            JvmRtlEdge::Except { site } => JvmMlilEdge::Except {
                site: site
                    .or_else(|| context.owner())
                    .map(|statement| {
                        context.throw_site(statement).ok_or_else(|| {
                            super::super::Error::Lifting(
                                "exception edge names a statement without a throw site".into(),
                            )
                        })
                    })
                    .transpose()?,
            },
        })
    }
}

impl Lower for Managed {
    fn plan(function: &mlil::Function<Self>) -> Result<Placement<Self>> {
        // A coordinated whole-function pass: every touched variable gets
        // a slot in one sweep.
        let mut placement = Placement::new();
        for block in function.cfg().blocks() {
            for instruction in block.instructions() {
                for &variable in instruction.uses().iter().chain(instruction.defs()) {
                    placement.assign(
                        variable,
                        Place {
                            storage: u8::try_from(variable.raw()).unwrap_or(u8::MAX),
                            lanes: vec![0],
                        },
                    );
                }
            }
        }
        Ok(placement)
    }

    fn lower_instruction(
        context: &mut LowerContext<'_, Self>,
        instruction: &mlil::Instruction<Self>,
    ) -> Result<()> {
        let reads = |context: &LowerContext<'_, Self>| -> Result<Vec<Expr<Self>>> {
            instruction
                .uses()
                .iter()
                .map(|&variable| context.read(variable, JvmConstraint::Unknown))
                .collect()
        };
        match instruction.operation() {
            LiftedStatement::Assign { .. } => {
                let target = context.place(instruction.defs()[0])?.clone();
                let operands = reads(context)?;
                let value = if operands.is_empty() {
                    Expr::Const {
                        bits: vec![0],
                        shape: JvmShape::scalar(JvmConstraint::Word(ScalarType::I32)),
                    }
                } else {
                    Expr::Apply {
                        operator: Operator::Add,
                        operands,
                        shape: JvmShape::scalar(JvmConstraint::Unknown),
                    }
                };
                context.emit(Statement::Transfer {
                    assignments: vec![(target, value)],
                    effects: Vec::new(),
                    may_throw: instruction.may_throw(),
                })?;
                Ok(())
            }
            LiftedStatement::Effect { operation, .. } => {
                let operands = reads(context)?;
                // An effect's definitions are the places it writes
                // without expressing a value; each lowers to its
                // variable's planned place.
                let writes = instruction
                    .defs()
                    .iter()
                    .map(|&variable| context.place(variable).cloned())
                    .collect::<Result<Vec<_>>>()?;
                context.emit(Statement::Effect {
                    operation: *operation,
                    operands,
                    writes,
                    effects: instruction.effects().to_vec(),
                    may_throw: instruction.may_throw(),
                })?;
                Ok(())
            }
            LiftedStatement::Branch { .. } => {
                let mut operands = reads(context)?;
                let condition = operands.pop().ok_or_else(|| {
                    super::super::Error::Lowering("branch without a condition".into())
                })?;
                context.emit(Statement::Branch { condition })?;
                Ok(())
            }
            LiftedStatement::Dispatch { .. } => {
                let mut operands = reads(context)?;
                let scrutinee = operands.pop().ok_or_else(|| {
                    super::super::Error::Lowering("dispatch without a scrutinee".into())
                })?;
                context.emit(Statement::Dispatch { scrutinee })?;
                Ok(())
            }
            LiftedStatement::Return { .. } => {
                let values = reads(context)?;
                context.emit(Statement::Return { values })?;
                Ok(())
            }
            LiftedStatement::Raise { .. } => {
                let operands = reads(context)?;
                context.emit(Statement::Raise {
                    operation: EffectOp::Throw,
                    operands,
                    effects: instruction.effects().to_vec(),
                })?;
                Ok(())
            }
        }
    }

    fn lower_edge(edge: &JvmMlilEdge, context: &LowerEdgeContext<'_>) -> Result<JvmRtlEdge> {
        Ok(match edge {
            JvmMlilEdge::Entry => JvmRtlEdge::Entry,
            JvmMlilEdge::Fall => JvmRtlEdge::Fall,
            JvmMlilEdge::True => JvmRtlEdge::True,
            JvmMlilEdge::False => JvmRtlEdge::False,
            JvmMlilEdge::Case(value) => JvmRtlEdge::Case(*value),
            // The exceptional payload crosses back into the RTL identity
            // domain: the owning instruction's lowered statement.
            JvmMlilEdge::Except { site } => JvmRtlEdge::Except {
                site: site
                    .or_else(|| context.owner())
                    .map(|instruction| {
                        context
                            .statements(instruction)
                            .first()
                            .copied()
                            .ok_or_else(|| {
                                super::super::Error::Lowering(
                                    "exception edge names an instruction without a statement"
                                        .into(),
                                )
                            })
                    })
                    .transpose()?,
            },
        })
    }
}

fn slot_read(storage: u8, constraint: JvmConstraint) -> Expr<Managed> {
    Expr::Read {
        storage,
        lanes: vec![0],
        scalar: constraint,
    }
}

fn slot_write(storage: u8, value: Expr<Managed>) -> Statement<Managed> {
    Statement::Transfer {
        assignments: vec![(
            Place {
                storage,
                lanes: vec![0],
            },
            value,
        )],
        effects: Vec::new(),
        may_throw: false,
    }
}

fn allocate(class: u8) -> Expr<Managed> {
    Expr::Apply {
        operator: Operator::New(class),
        operands: Vec::new(),
        shape: JvmShape::scalar(JvmConstraint::Reference(class)),
    }
}

fn word_const(value: u64) -> Expr<Managed> {
    Expr::Const {
        bits: vec![value],
        shape: JvmShape::scalar(JvmConstraint::Word(ScalarType::I32)),
    }
}

/// Reference lifetimes merge through the class hierarchy, null yields to
/// references, and a word/reference mix is a conflict.
#[test]
fn reference_constraints_merge_through_the_hierarchy() {
    let mut builder = FunctionBuilder::<Managed>::new("test".into());
    let entry = builder.entry();
    let head = builder.new_block("head");
    let then = builder.new_block("then");
    let other = builder.new_block("else");
    let join = builder.new_block("join");
    builder.add_edge(entry, head, JvmRtlEdge::Entry).unwrap();
    builder.add_edge(head, then, JvmRtlEdge::True).unwrap();
    builder.add_edge(head, other, JvmRtlEdge::False).unwrap();
    builder.add_edge(then, join, JvmRtlEdge::Fall).unwrap();
    builder.add_edge(other, join, JvmRtlEdge::Fall).unwrap();
    builder
        .append(
            head,
            Statement::Branch {
                condition: slot_read(9, JvmConstraint::Word(ScalarType::I32)),
            },
            None,
        )
        .unwrap();
    // Slot 0: new A(1) vs new B(2) — merges to the common superclass 0.
    builder
        .append(then, slot_write(0, allocate(1)), None)
        .unwrap();
    builder
        .append(other, slot_write(0, allocate(2)), None)
        .unwrap();
    // Slot 1: new A(1) vs null — null yields, the reference survives.
    builder
        .append(then, slot_write(1, allocate(1)), None)
        .unwrap();
    builder
        .append(
            other,
            slot_write(
                1,
                Expr::Const {
                    bits: vec![0],
                    shape: JvmShape::scalar(JvmConstraint::Null),
                },
            ),
            None,
        )
        .unwrap();
    // Slot 2: an integer vs a reference — a genuine conflict.
    builder
        .append(then, slot_write(2, word_const(7)), None)
        .unwrap();
    builder
        .append(other, slot_write(2, allocate(1)), None)
        .unwrap();
    for slot in 0..3u8 {
        builder
            .append(
                join,
                slot_write(10 + slot, slot_read(slot, JvmConstraint::Unknown)),
                None,
            )
            .unwrap();
    }
    builder
        .append(join, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let lifting = lift(&function, &hierarchy()).unwrap();
    let constraint = |storage: u8| {
        lifting
            .webs
            .iter()
            .find(|web| web.storage == Some(storage))
            .expect("slot web")
            .shape
            .scalar
    };
    assert_eq!(constraint(0), JvmConstraint::Reference(0));
    assert_eq!(constraint(1), JvmConstraint::Reference(1));
    assert_eq!(constraint(2), JvmConstraint::Conflict);
    lifting.builder.finish().unwrap();
}

/// A dispatch statement lifts through MLIL into a structured switch.
#[test]
fn dispatch_lifts_to_a_structured_switch() {
    let mut builder = FunctionBuilder::<Managed>::new("test".into());
    let entry = builder.entry();
    let head = builder.new_block("head");
    let one = builder.new_block("one");
    let two = builder.new_block("two");
    let fallback = builder.new_block("fallback");
    let merge = builder.new_block("merge");
    builder.add_edge(entry, head, JvmRtlEdge::Entry).unwrap();
    builder.add_edge(head, one, JvmRtlEdge::Case(1)).unwrap();
    builder.add_edge(head, two, JvmRtlEdge::Case(2)).unwrap();
    builder.add_edge(head, fallback, JvmRtlEdge::Fall).unwrap();
    builder.add_edge(one, merge, JvmRtlEdge::Fall).unwrap();
    builder.add_edge(two, merge, JvmRtlEdge::Fall).unwrap();
    builder.add_edge(fallback, merge, JvmRtlEdge::Fall).unwrap();
    builder
        .append(
            head,
            Statement::Dispatch {
                scrutinee: slot_read(9, JvmConstraint::Word(ScalarType::I32)),
            },
            None,
        )
        .unwrap();
    for block in [one, two, fallback] {
        builder
            .append(
                block,
                Statement::Effect {
                    writes: Vec::new(),
                    operation: EffectOp::Invoke,
                    operands: Vec::new(),
                    effects: vec![Effect::Call],
                    may_throw: false,
                },
                None,
            )
            .unwrap();
    }
    builder
        .append(merge, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let lifting = lift(&function, &hierarchy()).unwrap();
    let function = lifting.builder.finish().unwrap();
    let lifted = lift_hlil(&function).unwrap();
    assert!(lifted.report.is_fully_structured(), "{:?}", lifted.report);
    assert!(
        lifted
            .function
            .statements()
            .iter()
            .any(|statement| matches!(statement.kind(), StatementKind::Switch { .. })),
        "the dispatch structures as a switch"
    );
}

/// Lowering walks an MLIL function back onto target RTL with complete
/// rewrite maps and the retained storage plan.
fn configure_signature_and_region(
    builder: &mut FunctionBuilder<Managed>,
    protected: crate::BlockId,
    handler: crate::BlockId,
) {
    builder
        .set_signature(Signature::new(
            vec![Place {
                storage: 9,
                lanes: vec![0],
            }],
            vec![JvmShape::scalar(JvmConstraint::Word(ScalarType::I32))],
        ))
        .unwrap();
    builder
        .add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: BTreeSet::from([protected]),
            handlers: vec![Handler {
                entry: handler,
                body: HandlerBody::Known(BTreeSet::from([handler])),
                kind: HandlerKind::Catch,
            }],
            parent: None,
        })
        .unwrap();
}

#[test]
fn lowering_round_trips_with_rewrite_maps() {
    // Build RTL, lift it, then lower the MLIL back down.
    let mut builder = FunctionBuilder::<Managed>::new("test".into());
    let entry = builder.entry();
    let head = builder.new_block("head");
    let then = builder.new_block("then");
    let other = builder.new_block("else");
    let join = builder.new_block("join");
    builder.add_edge(entry, head, JvmRtlEdge::Entry).unwrap();
    builder.add_edge(head, then, JvmRtlEdge::True).unwrap();
    builder.add_edge(head, other, JvmRtlEdge::False).unwrap();
    builder.add_edge(then, join, JvmRtlEdge::Fall).unwrap();
    builder.add_edge(other, join, JvmRtlEdge::Fall).unwrap();
    configure_signature_and_region(&mut builder, then, other);
    builder
        .append(
            head,
            Statement::Branch {
                condition: slot_read(9, JvmConstraint::Word(ScalarType::I32)),
            },
            None,
        )
        .unwrap();
    builder
        .append(then, slot_write(0, word_const(1)), None)
        .unwrap();
    builder
        .append(other, slot_write(0, word_const(2)), None)
        .unwrap();
    builder
        .append(
            join,
            Statement::Return {
                values: vec![slot_read(0, JvmConstraint::Word(ScalarType::I32))],
            },
            None,
        )
        .unwrap();
    let function = builder.finish().unwrap();
    let lifting = lift(&function, &hierarchy()).unwrap();
    let mlil_function = lifting.builder.finish().unwrap();

    let lowered = lower::<Managed>(&mlil_function).unwrap();
    assert_eq!(
        lowered.function.cfg().block_count(),
        mlil_function.cfg().block_count(),
        "blocks mirror one-to-one"
    );
    for block_id in mlil_function.cfg().block_ids() {
        assert!(
            lowered.block(block_id).is_some(),
            "every block maps: {block_id}"
        );
        for instruction in mlil_function.cfg().block(block_id).instructions() {
            assert!(
                !lowered.statements(instruction.id()).is_empty(),
                "every instruction lowers to at least one statement"
            );
            assert!(
                instruction
                    .defs()
                    .iter()
                    .all(|&variable| lowered.placement.place(variable).is_some()),
                "the retained plan places every defined variable"
            );
        }
    }
    for edge in mlil_function.cfg().edges() {
        assert!(lowered.edge(edge.id()).is_some(), "every edge maps");
    }
    assert_eq!(lowered.function.signature().parameters.len(), 1);
    assert_eq!(lowered.function.signature().returns.len(), 1);
    assert_eq!(lowered.function.cfg().regions().len(), 1);

    let relifted = lift(&lowered.function, &hierarchy())
        .unwrap()
        .builder
        .finish()
        .unwrap();
    assert_eq!(relifted.signature().parameters.len(), 1);
    assert_eq!(
        relifted.signature().returns,
        mlil_function.signature().returns
    );
    assert_eq!(relifted.cfg().regions().len(), 1);
}

/// Fused source spans survive RTL → MLIL → RTL without selecting only
/// one "primary" location.
#[test]
fn lowering_round_trip_preserves_many_to_one_provenance() {
    let mut builder = FunctionBuilder::<Managed>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, JvmRtlEdge::Entry).unwrap();
    let statement = builder
        .append(
            body,
            slot_write(0, word_const(7)),
            Some(Span { start: 4, end: 5 }),
        )
        .unwrap();
    builder
        .map_statement(Span { start: 8, end: 10 }, statement)
        .unwrap();
    builder
        .append(body, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let lifting = lift(&function, &hierarchy()).unwrap();
    let instruction = lifting.maps.instructions(statement)[0];
    let mlil_function = lifting.builder.finish().unwrap();
    let mlil_spans: Vec<_> = mlil_function
        .provenance()
        .mappings_to(mlil::EntityId::Instruction(instruction))
        .map(|entry| entry.source)
        .collect();
    assert_eq!(
        mlil_spans,
        vec![Span { start: 4, end: 5 }, Span { start: 8, end: 10 }]
    );

    let lowered = lower::<Managed>(&mlil_function).unwrap();
    let lowered_statement = lowered.statements(instruction)[0];
    let rtl_spans: Vec<_> = lowered
        .function
        .provenance()
        .mappings_to(lowered_statement)
        .map(|entry| entry.source)
        .collect();
    assert_eq!(rtl_spans, mlil_spans);
}
