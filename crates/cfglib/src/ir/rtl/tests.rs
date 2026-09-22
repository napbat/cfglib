extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::EdgeKind;
use crate::ir::dialect::Vocabulary;
use crate::ir::hlil::{self, Lifted};
use crate::ir::mlil::{self, InstructionMetadata, VerificationIssue};
use crate::test_util::toy::{self, Span};

use super::{
    Dialect, Edge, Expr, Function, FunctionBuilder, Lift, LiftedStatement, MlilBridge, Place,
    ScalarType, Statement, ValueShape, VarExpr, lift,
};

/// Managed-language dialect tests: constraint domains, exceptional
/// ownership, dispatch, expansion, and lowering.
mod managed;
/// Golden pseudocode for representative RTL functions.
mod pseudocode;
/// Read-resolver tests, split out to respect the source-size policy.
mod resolver;
/// Return-value lowering tests, split out to respect the source-size
/// policy.
mod returns;
/// Construction and completion validation tests, split out to respect
/// the source-size policy.
mod validation;
/// Effect statements that declare the places they write.
mod writes;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Effect {
    Emit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operator {
    Add,
    Shr,
    Div,
    Rem,
    Less,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectOp {
    Emit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TestDialect;

/// A distinct semantic marker proving one RTL dialect need not also be
/// the MLIL dialect it raises into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SemanticDialect;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SemanticStorage(u8);

impl Vocabulary for TestDialect {
    type ValueType = ValueShape;
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

impl Vocabulary for SemanticDialect {
    type ValueType = ValueShape;
    type Effect = Effect;
    type Source = String;
    type SourceSpan = Span;
    type SourcePoint = u32;
    type VariableRole = u8;
    type NativeVariable = SemanticStorage;

    fn span_is_empty(span: &Self::SourceSpan) -> bool {
        toy::span_is_empty(*span)
    }

    fn span_contains(span: &Self::SourceSpan, point: &Self::SourcePoint) -> bool {
        toy::span_contains(*span, *point)
    }
}

impl Dialect for TestDialect {
    type Constraint = ScalarType;
    type Operator = Operator;
    type EffectOp = EffectOp;
    type Edge = Edge;

    fn mnemonic(operator: &Self::Operator) -> &str {
        match operator {
            Operator::Add => "add",
            Operator::Shr => "shr",
            Operator::Div => "div",
            Operator::Rem => "rem",
            Operator::Less => "less",
        }
    }

    fn effect_mnemonic(operation: &Self::EffectOp) -> &str {
        match operation {
            EffectOp::Emit => "emit",
        }
    }

    fn edge_kind(edge: &Self::Edge) -> EdgeKind {
        edge.kind()
    }

    fn is_entry_edge(edge: &Self::Edge) -> bool {
        edge.is_entry()
    }
}

impl mlil::Dialect for SemanticDialect {
    type Operation = LiftedStatement<TestDialect>;
    type Edge = Edge;

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
        edge.kind()
    }

    fn is_entry_edge(edge: &Self::Edge) -> bool {
        edge.is_entry()
    }
}

impl MlilBridge for TestDialect {
    type Mlil = SemanticDialect;
}

impl mlil::AnalysisDialect for SemanticDialect {
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

#[test]
fn canonical_edge_vocabulary_preserves_unwind_flow() {
    assert_eq!(Edge::Unwind.kind(), EdgeKind::ExceptionUnwind);
    assert!(!Edge::Unwind.is_entry());
}

impl hlil::Dialect for SemanticDialect {
    type Operation = LiftedStatement<TestDialect>;
    type Constant = Vec<u64>;

    fn mnemonic(operation: &Self::Operation) -> &str {
        operation.mnemonic()
    }
}

impl hlil::VerifyDialect for SemanticDialect {
    fn verify(_function: &hlil::Function<Self>, _issues: &mut Vec<hlil::VerificationIssue>) {}
}

impl hlil::LiftDialect for SemanticDialect {
    fn lift_operation(
        operation: &LiftedStatement<TestDialect>,
    ) -> Lifted<LiftedStatement<TestDialect>> {
        operation.lifted()
    }

    fn case_values(_edge: &Edge) -> Vec<Vec<u64>> {
        Vec::new()
    }

    fn void_type() -> ValueShape {
        ValueShape::vector(ScalarType::Bits, 0)
    }

    fn previous_value_operand(operation: &LiftedStatement<TestDialect>) -> Option<usize> {
        operation.merge_operand()
    }
}

impl mlil::VerifyDialect for SemanticDialect {
    fn verify(_function: &mlil::Function<Self>, _issues: &mut Vec<VerificationIssue>) {}
}

impl Lift for TestDialect {
    fn value_type(shape: ValueShape) -> ValueShape {
        shape
    }

    fn web_role(storage: Option<&u8>) -> u8 {
        u8::from(storage.is_none())
    }

    fn native_variable(storage: &u8, _source: &String) -> Option<SemanticStorage> {
        Some(SemanticStorage(*storage))
    }

    fn emit(
        context: &mut super::Emission<'_, '_, Self>,
        statement: LiftedStatement<Self>,
    ) -> super::Result<()> {
        context.single(statement)?;
        Ok(())
    }

    fn lift_edge(edge: &Edge, _context: &super::EdgeContext<'_>) -> super::Result<Edge> {
        Ok(*edge)
    }
}

fn read(storage: u8, lanes: &[u8], scalar: ScalarType) -> Expr<TestDialect> {
    Expr::Read {
        storage,
        lanes: lanes.to_vec(),
        scalar,
    }
}

fn constant(bits: u64, scalar: ScalarType) -> Expr<TestDialect> {
    Expr::Const {
        bits: vec![bits],
        shape: ValueShape::scalar(scalar),
    }
}

fn apply(
    operator: Operator,
    operands: Vec<Expr<TestDialect>>,
    shape: ValueShape,
) -> Expr<TestDialect> {
    Expr::Apply {
        operator,
        operands,
        shape,
    }
}

fn assign(storage: u8, lanes: &[u8], value: Expr<TestDialect>) -> Statement<TestDialect> {
    Statement::Transfer {
        assignments: vec![(
            Place {
                storage,
                lanes: lanes.to_vec(),
            },
            value,
        )],
        effects: Vec::new(),
        may_throw: false,
    }
}

fn vector_const(bits: &[u64], scalar: ScalarType) -> Expr<TestDialect> {
    Expr::Const {
        bits: bits.to_vec(),
        shape: ValueShape::vector(scalar, u8::try_from(bits.len()).unwrap()),
    }
}

/// Every MLIL instruction of a lifted function, in block order.
fn instructions(
    function: &mlil::Function<SemanticDialect>,
) -> Vec<&mlil::Instruction<SemanticDialect>> {
    function
        .cfg()
        .blocks()
        .flat_map(|block| block.instructions().iter())
        .collect()
}

/// Register reuse with different interpretations splits into typed webs.
#[test]
fn storage_reuse_splits_into_typed_webs() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    // r0.x ← f32 add; r1.x ← r0.x (float); r0.x ← u32 shr; r2.x ← r0.x (uint)
    builder
        .append(
            body,
            assign(
                0,
                &[0],
                apply(
                    Operator::Add,
                    vec![
                        constant(0x3f80_0000, ScalarType::F32),
                        constant(0x4000_0000, ScalarType::F32),
                    ],
                    ValueShape::scalar(ScalarType::F32),
                ),
            ),
            None,
        )
        .unwrap();
    builder
        .append(body, assign(1, &[0], read(0, &[0], ScalarType::F32)), None)
        .unwrap();
    builder
        .append(
            body,
            assign(
                0,
                &[0],
                apply(
                    Operator::Shr,
                    vec![constant(64, ScalarType::U32), constant(2, ScalarType::U32)],
                    ValueShape::scalar(ScalarType::U32),
                ),
            ),
            None,
        )
        .unwrap();
    builder
        .append(body, assign(2, &[0], read(0, &[0], ScalarType::U32)), None)
        .unwrap();
    builder
        .append(
            body,
            Statement::Effect {
                writes: Vec::new(),
                operation: EffectOp::Emit,
                operands: vec![read(2, &[0], ScalarType::U32)],
                effects: vec![Effect::Emit],
                may_throw: false,
            },
            None,
        )
        .unwrap();
    builder
        .append(body, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function: Function<TestDialect> = builder.finish().unwrap();

    let lifting = lift(&function, &()).unwrap();
    let r0_webs: Vec<_> = lifting
        .webs
        .iter()
        .filter(|web| web.storage == Some(0))
        .collect();
    assert_eq!(r0_webs.len(), 2, "reused storage splits into two webs");
    let scalars: Vec<ScalarType> = r0_webs.iter().map(|web| web.shape.scalar).collect();
    assert!(scalars.contains(&ScalarType::F32));
    assert!(scalars.contains(&ScalarType::U32));
    lifting.builder.finish().unwrap();
}

/// A loop-carried parallel transfer whose target is read by a sibling
/// serializes through a synthetic pre-state copy.
#[test]
fn parallel_hazard_pre_copies_when_webs_unite() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let header = builder.new_block("header");
    let exit = builder.new_block("exit");
    builder.add_edge(entry, header, Edge::Entry).unwrap();
    builder.add_edge(header, header, Edge::True).unwrap();
    builder.add_edge(header, exit, Edge::False).unwrap();
    // Parallel: r0.x ← r1.x / r0.x ; r2.x ← r1.x % r0.x. The loop back
    // edge φ-unites r0.x's versions, so the sibling read needs the
    // pre-state copy.
    builder
        .append(
            header,
            Statement::Transfer {
                assignments: vec![
                    (
                        Place {
                            storage: 0,
                            lanes: vec![0],
                        },
                        apply(
                            Operator::Div,
                            vec![
                                read(1, &[0], ScalarType::U32),
                                read(0, &[0], ScalarType::U32),
                            ],
                            ValueShape::scalar(ScalarType::U32),
                        ),
                    ),
                    (
                        Place {
                            storage: 2,
                            lanes: vec![0],
                        },
                        apply(
                            Operator::Rem,
                            vec![
                                read(1, &[0], ScalarType::U32),
                                read(0, &[0], ScalarType::U32),
                            ],
                            ValueShape::scalar(ScalarType::U32),
                        ),
                    ),
                ],
                effects: Vec::new(),
                may_throw: false,
            },
            None,
        )
        .unwrap();
    builder
        .append(
            header,
            Statement::Branch {
                condition: apply(
                    Operator::Less,
                    vec![read(2, &[0], ScalarType::U32), constant(8, ScalarType::U32)],
                    ValueShape::scalar(ScalarType::U32),
                ),
            },
            None,
        )
        .unwrap();
    builder
        .append(exit, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let lifting = lift(&function, &()).unwrap();
    let synthetic: Vec<_> = lifting
        .webs
        .iter()
        .filter(|web| web.storage.is_none())
        .collect();
    assert_eq!(synthetic.len(), 1, "one pre-state copy temporary");
    let function = lifting.builder.finish().unwrap();
    // header holds: copy, div, rem, branch.
    let header_len = function
        .cfg()
        .blocks()
        .find(|block| block.label() == Some("header"))
        .map(|block| block.instructions().len());
    assert_eq!(header_len, Some(4));
}

/// A straight-line swap has distinct old and new webs, but still needs
/// both native pre-state values while its assignments serialize.
#[test]
fn parallel_hazard_pre_copies_split_webs_in_straight_line_code() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    builder
        .append(
            body,
            Statement::Transfer {
                assignments: vec![
                    (
                        Place {
                            storage: 0,
                            lanes: vec![0],
                        },
                        read(1, &[0], ScalarType::U32),
                    ),
                    (
                        Place {
                            storage: 1,
                            lanes: vec![0],
                        },
                        read(0, &[0], ScalarType::U32),
                    ),
                ],
                effects: Vec::new(),
                may_throw: false,
            },
            None,
        )
        .unwrap();
    builder
        .append(body, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let lifting = lift(&function, &()).unwrap();
    let synthetic_count = lifting
        .webs
        .iter()
        .filter(|web| web.storage.is_none())
        .count();
    assert_eq!(synthetic_count, 2, "both old values are staged");
    let function = lifting.builder.finish().unwrap();
    assert_eq!(
        instructions(&function).len(),
        5,
        "two copies, two writes, return"
    );
}

/// A diamond join φ-unites both definitions into one typed web.
#[test]
fn join_unites_definitions_into_one_web() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let head = builder.new_block("head");
    let then = builder.new_block("then");
    let other = builder.new_block("else");
    let join = builder.new_block("join");
    builder.add_edge(entry, head, Edge::Entry).unwrap();
    builder.add_edge(head, then, Edge::True).unwrap();
    builder.add_edge(head, other, Edge::False).unwrap();
    builder.add_edge(then, join, Edge::Fall).unwrap();
    builder.add_edge(other, join, Edge::Fall).unwrap();
    builder
        .append(
            head,
            Statement::Branch {
                condition: read(9, &[0], ScalarType::U32),
            },
            None,
        )
        .unwrap();
    builder
        .append(
            then,
            assign(0, &[0], constant(0x3f80_0000, ScalarType::F32)),
            None,
        )
        .unwrap();
    builder
        .append(
            other,
            assign(0, &[0], constant(0x4000_0000, ScalarType::F32)),
            None,
        )
        .unwrap();
    builder
        .append(join, assign(1, &[0], read(0, &[0], ScalarType::F32)), None)
        .unwrap();
    builder
        .append(join, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let lifting = lift(&function, &()).unwrap();
    let r0_webs: Vec<_> = lifting
        .webs
        .iter()
        .filter(|web| web.storage == Some(0))
        .collect();
    assert_eq!(r0_webs.len(), 1, "both arms define one web");
    assert_eq!(r0_webs[0].shape.scalar, ScalarType::F32);
    assert!(!r0_webs[0].live_in);
    lifting.builder.finish().unwrap();
}

/// A dead loop-header φ — the loop overwrites the register before any
/// read — must not fuse the pre-loop and in-loop lifetimes: each keeps
/// its own web and its own honest type.
#[test]
fn dead_header_phi_keeps_lifetimes_apart() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let pre = builder.new_block("pre");
    let header = builder.new_block("header");
    let body = builder.new_block("body");
    let exit = builder.new_block("exit");
    builder.add_edge(entry, pre, Edge::Entry).unwrap();
    builder.add_edge(pre, header, Edge::Fall).unwrap();
    builder.add_edge(header, body, Edge::True).unwrap();
    builder.add_edge(header, exit, Edge::False).unwrap();
    builder.add_edge(body, header, Edge::Fall).unwrap();
    // Pre-loop: r0.x holds a float, read as one.
    builder
        .append(
            pre,
            assign(0, &[0], constant(0x3f80_0000, ScalarType::F32)),
            None,
        )
        .unwrap();
    builder
        .append(pre, assign(1, &[0], read(0, &[0], ScalarType::F32)), None)
        .unwrap();
    builder
        .append(
            header,
            Statement::Branch {
                condition: read(9, &[0], ScalarType::U32),
            },
            None,
        )
        .unwrap();
    // In-loop: r0.x is overwritten before any read — the header φ is
    // dead — and holds an integer.
    builder
        .append(body, assign(0, &[0], constant(7, ScalarType::U32)), None)
        .unwrap();
    builder
        .append(body, assign(2, &[0], read(0, &[0], ScalarType::U32)), None)
        .unwrap();
    builder
        .append(exit, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let lifting = lift(&function, &()).unwrap();
    let r0_webs: Vec<_> = lifting
        .webs
        .iter()
        .filter(|web| web.storage == Some(0))
        .collect();
    assert_eq!(r0_webs.len(), 2, "the dead φ must not merge the lifetimes");
    let scalars: Vec<ScalarType> = r0_webs.iter().map(|web| web.shape.scalar).collect();
    assert!(scalars.contains(&ScalarType::F32));
    assert!(scalars.contains(&ScalarType::U32));
    assert!(r0_webs.iter().all(|web| !web.live_in));
    lifting.builder.finish().unwrap();
}

/// Genuinely conflicting interpretations resolve to `Bits` no matter
/// the observation order — a later agreement never launders an earlier
/// conflict.
#[test]
fn conflicting_interpretations_resolve_to_bits() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    builder
        .append(
            body,
            assign(0, &[0], constant(0x3f80_0000, ScalarType::F32)),
            None,
        )
        .unwrap();
    builder
        .append(body, assign(1, &[0], read(0, &[0], ScalarType::F32)), None)
        .unwrap();
    builder
        .append(body, assign(2, &[0], read(0, &[0], ScalarType::U32)), None)
        .unwrap();
    builder
        .append(body, assign(3, &[0], read(0, &[0], ScalarType::F32)), None)
        .unwrap();
    builder
        .append(body, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let lifting = lift(&function, &()).unwrap();
    let r0_web = lifting
        .webs
        .iter()
        .find(|web| web.storage == Some(0))
        .unwrap();
    assert_eq!(
        r0_web.shape.scalar,
        ScalarType::Bits,
        "float and integer reads of one web are a conflict"
    );
    lifting.builder.finish().unwrap();
}

/// A straight-line partial rewrite is a fresh lifetime: the web splits,
/// a later full-width read composes both webs, and no merge arises.
#[test]
fn partial_rewrite_splits_the_web_and_reads_compose() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    builder
        .append(
            body,
            assign(
                0,
                &[0, 1],
                apply(
                    Operator::Add,
                    vec![
                        vector_const(&[1, 2], ScalarType::F32),
                        vector_const(&[3, 4], ScalarType::F32),
                    ],
                    ValueShape::vector(ScalarType::F32, 2),
                ),
            ),
            None,
        )
        .unwrap();
    builder
        .append(
            body,
            assign(0, &[0], constant(0x3f80_0000, ScalarType::F32)),
            None,
        )
        .unwrap();
    builder
        .append(
            body,
            assign(1, &[0, 1], read(0, &[0, 1], ScalarType::F32)),
            None,
        )
        .unwrap();
    builder
        .append(body, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let lifting = lift(&function, &()).unwrap();
    let r0_webs: Vec<_> = lifting
        .webs
        .iter()
        .filter(|web| web.storage == Some(0))
        .collect();
    assert_eq!(r0_webs.len(), 2, "the rewritten lane is a fresh lifetime");
    let function = lifting.builder.finish().unwrap();
    let composed = instructions(&function)
        .into_iter()
        .find_map(|instruction| match instruction.operation() {
            LiftedStatement::Assign {
                value: VarExpr::Compose { parts, .. },
                merges,
                ..
            } => Some((parts.len(), *merges, instruction.uses().len())),
            _ => None,
        })
        .expect("the full-width read composes two webs");
    assert_eq!(composed, (2, false, 2), "two parts, no merge, two uses");
}

/// A partial overwrite whose target web is genuinely wider — the other
/// lanes stay live through a join — merges with the prior state: the
/// instruction reads its target as the trailing operand.
#[test]
fn partial_overwrite_after_join_merges_prior_state() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let pre = builder.new_block("pre");
    let then = builder.new_block("then");
    let join = builder.new_block("join");
    builder.add_edge(entry, pre, Edge::Entry).unwrap();
    builder.add_edge(pre, then, Edge::True).unwrap();
    builder.add_edge(pre, join, Edge::False).unwrap();
    builder.add_edge(then, join, Edge::Fall).unwrap();
    builder
        .append(
            pre,
            assign(
                0,
                &[0, 1],
                apply(
                    Operator::Add,
                    vec![
                        vector_const(&[1, 2], ScalarType::F32),
                        vector_const(&[3, 4], ScalarType::F32),
                    ],
                    ValueShape::vector(ScalarType::F32, 2),
                ),
            ),
            None,
        )
        .unwrap();
    builder
        .append(
            pre,
            Statement::Branch {
                condition: read(9, &[0], ScalarType::U32),
            },
            None,
        )
        .unwrap();
    builder
        .append(
            then,
            assign(0, &[0], constant(0x3f80_0000, ScalarType::F32)),
            None,
        )
        .unwrap();
    builder
        .append(
            join,
            assign(1, &[0, 1], read(0, &[0, 1], ScalarType::F32)),
            None,
        )
        .unwrap();
    builder
        .append(join, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let lifting = lift(&function, &()).unwrap();
    let r0_webs: Vec<_> = lifting
        .webs
        .iter()
        .filter(|web| web.storage == Some(0))
        .collect();
    assert_eq!(r0_webs.len(), 1, "the live join φ unites the versions");
    assert_eq!(r0_webs[0].lanes, vec![0, 1]);
    let function = lifting.builder.finish().unwrap();
    let merge = instructions(&function)
        .into_iter()
        .find_map(|instruction| match instruction.operation() {
            LiftedStatement::Assign {
                positions,
                width,
                merges: true,
                ..
            } => Some((
                positions.clone(),
                *width,
                instruction.uses().to_vec(),
                instruction.defs().to_vec(),
            )),
            _ => None,
        })
        .expect("the partial overwrite merges");
    let (positions, width, uses, defs) = merge;
    assert_eq!(positions, vec![0]);
    assert_eq!(width, 2);
    assert_eq!(defs.len(), 1);
    assert_eq!(
        uses.as_slice(),
        &[defs[0]],
        "the trailing use reads the target's prior state"
    );
}

/// A merging assignment's previous-value slot trails every value read.
#[test]
fn merge_operand_trails_the_value_reads() {
    let value: VarExpr<TestDialect> = VarExpr::Apply {
        operator: Operator::Add,
        operands: vec![
            VarExpr::Read {
                positions: vec![0],
                scalar: ScalarType::F32,
            },
            VarExpr::Read {
                positions: vec![1],
                scalar: ScalarType::F32,
            },
            VarExpr::Const {
                bits: vec![0],
                shape: ValueShape::scalar(ScalarType::F32),
            },
        ],
        shape: ValueShape::scalar(ScalarType::F32),
    };
    assert_eq!(value.read_count(), 2);
    let merging = LiftedStatement::<TestDialect>::Assign {
        positions: vec![0],
        width: 2,
        merges: true,
        value: value.clone(),
        effects: Vec::new(),
    };
    assert_eq!(merging.merge_operand(), Some(2));
    let full = LiftedStatement::<TestDialect>::Assign {
        positions: vec![0, 1],
        width: 2,
        merges: false,
        value,
        effects: Vec::new(),
    };
    assert_eq!(full.merge_operand(), None);
}

/// Pre-order expression traversal visits every node once.
#[test]
fn for_each_expression_visits_pre_order() {
    let value: VarExpr<TestDialect> = VarExpr::Apply {
        operator: Operator::Add,
        operands: vec![
            VarExpr::Reinterpret {
                operand: alloc::boxed::Box::new(VarExpr::Const {
                    bits: vec![1],
                    shape: ValueShape::scalar(ScalarType::U32),
                }),
                shape: ValueShape::scalar(ScalarType::F32),
            },
            VarExpr::Compose {
                parts: vec![VarExpr::Read {
                    positions: vec![0],
                    scalar: ScalarType::F32,
                }],
                shape: ValueShape::scalar(ScalarType::F32),
            },
        ],
        shape: ValueShape::scalar(ScalarType::F32),
    };
    let mut mnemonics: Vec<String> = Vec::new();
    value.for_each_expression(&mut |expression| mnemonics.push(expression.mnemonic().into()));
    assert_eq!(mnemonics, vec!["add", "bitcast", "const", "compose", "mov"]);
}
