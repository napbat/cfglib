//! Exceptional-flow tests: throw-site ownership, continuation splits,
//! emission validation, and edge payloads crossing identity domains.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::ir::mlil;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};

use super::super::super::{
    Expr, Function, FunctionBuilder, Place, Statement, StatementId, lift, lower,
};
use super::{
    Effect, EffectOp, JvmConstraint, JvmMlilEdge, JvmRtlEdge, JvmShape, Managed, Operator,
    ScalarType, hierarchy, word_const,
};

/// A function whose body performs one throwing invoke-shaped addition
/// with a handler — the shared fixture for exceptional-flow tests.
fn throwing_function() -> (Function<Managed>, StatementId) {
    let mut builder = FunctionBuilder::<Managed>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    let exit = builder.new_block("exit");
    let handler = builder.new_block("handler");
    builder.add_edge(entry, body, JvmRtlEdge::Entry).unwrap();
    builder.add_edge(body, exit, JvmRtlEdge::Fall).unwrap();
    builder
        .add_edge(body, handler, JvmRtlEdge::Except { site: None })
        .unwrap();
    builder
        .add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: [body].into_iter().collect(),
            handlers: vec![Handler {
                entry: handler,
                body: HandlerBody::Unknown,
                kind: HandlerKind::Catch,
            }],
            parent: None,
        })
        .unwrap();
    // An invoke-shaped throwing assignment whose Add value triggers the
    // dialect's continuation-splitting two-instruction expansion.
    let invoke = builder
        .append(
            body,
            Statement::Transfer {
                assignments: vec![(
                    Place {
                        storage: 0,
                        lanes: vec![0],
                    },
                    Expr::Apply {
                        operator: Operator::Add,
                        operands: vec![word_const(1), word_const(2)],
                        shape: JvmShape::scalar(JvmConstraint::Word(ScalarType::I32)),
                    },
                )],
                effects: vec![Effect::Call],
                may_throw: true,
            },
            None,
        )
        .unwrap();
    builder
        .append(exit, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    builder
        .append(
            handler,
            Statement::Raise {
                operation: EffectOp::Throw,
                operands: Vec::new(),
                effects: vec![Effect::Call],
            },
            None,
        )
        .unwrap();
    (builder.finish().unwrap(), invoke)
}

/// A throwing invoke expands into two MLIL instructions across a
/// continuation split: the throw site is terminal in its block, native
/// state commits only in the continuation, and the exceptional edge
/// leaves the throw site's block with the exact emitted identity in its
/// MLIL-domain payload.
#[test]
fn exceptional_edge_carries_the_emitted_throw_site() {
    let (function, invoke) = throwing_function();
    let lifting = lift(&function, &hierarchy()).unwrap();
    let emitted = lifting.maps.instructions(invoke).to_vec();
    assert_eq!(
        emitted.len(),
        2,
        "the addition expands into two instructions"
    );
    let site = lifting
        .maps
        .throw_site(invoke)
        .expect("a designated throw site");
    assert_eq!(site, emitted[0], "the first instruction owns the throw");

    let function = lifting.builder.finish().unwrap();
    let exceptional = function
        .cfg()
        .edges()
        .find(|edge| matches!(edge.payload(), JvmMlilEdge::Except { .. }))
        .expect("the exceptional edge survives the lift");
    assert!(
        matches!(
            exceptional.payload(),
            JvmMlilEdge::Except { site: payload } if *payload == Some(site)
        ),
        "the edge payload names the throw site"
    );
    let instruction = function.instruction(site).expect("the throw site exists");
    assert!(instruction.may_throw());

    // Throw-terminal: the throw site is the last instruction of its
    // block, the commit lives in the continuation, and the normal path
    // continues from the commit block rather than the throw block.
    let throw_block = function.cfg().block(exceptional.source());
    assert_eq!(
        throw_block.instructions().last().map(mlil::Instruction::id),
        Some(site),
        "the throw site is terminal in its block"
    );
    let commit_block = function
        .cfg()
        .block_ids()
        .find(|&block| {
            function
                .cfg()
                .block(block)
                .instructions()
                .iter()
                .any(|instruction| instruction.id() == emitted[1])
        })
        .expect("the commit instruction has a block");
    assert_ne!(
        commit_block,
        exceptional.source(),
        "native state commits outside the throw block"
    );
    let region = &function.cfg().regions()[0];
    assert!(region.protected_blocks.contains(&exceptional.source()));
    assert!(
        region.protected_blocks.contains(&commit_block),
        "a continuation split remains inside the protected region"
    );
    assert!(
        function
            .cfg()
            .edges()
            .any(|edge| edge.source() == commit_block
                && edge.target() != commit_block
                && matches!(edge.payload(), JvmMlilEdge::Fall)),
        "the normal path continues from the commit block, not the throw block"
    );
}

/// Throw capability and local exceptional flow are separate facts: an
/// unhandled throw still marks the emitted instruction, but does not require
/// the native assignment to be staged behind a continuation.
#[test]
fn unhandled_throw_exposes_no_exceptional_successor() {
    let mut builder = FunctionBuilder::<Managed>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    let exit = builder.new_block("exit");
    builder.add_edge(entry, body, JvmRtlEdge::Entry).unwrap();
    builder.add_edge(body, exit, JvmRtlEdge::Fall).unwrap();
    let invoke = builder
        .append(
            body,
            Statement::Transfer {
                assignments: vec![(
                    Place {
                        storage: 0,
                        lanes: vec![0],
                    },
                    Expr::Apply {
                        operator: Operator::Add,
                        operands: vec![word_const(1), word_const(2)],
                        shape: JvmShape::scalar(JvmConstraint::Word(ScalarType::I32)),
                    },
                )],
                effects: vec![Effect::Call],
                may_throw: true,
            },
            None,
        )
        .unwrap();
    builder
        .append(exit, Statement::Return { values: Vec::new() }, None)
        .unwrap();

    let lifting = lift(&builder.finish().unwrap(), &hierarchy()).unwrap();
    let emitted = lifting.maps.instructions(invoke);
    assert_eq!(emitted.len(), 1, "no local handler requires no staging");
    let function = lifting.builder.finish().unwrap();
    let instruction = function
        .instruction(emitted[0])
        .expect("the throwing assignment was emitted");
    assert!(instruction.may_throw());
    assert_eq!(instruction.defs().len(), 1, "the native target is direct");
}

/// Lowering translates the exceptional edge back into the RTL identity
/// domain: the payload names a lowered statement, not an MLIL
/// instruction.
#[test]
fn lowering_remaps_the_throw_site_edge() {
    let (function, _invoke) = throwing_function();
    let lifting = lift(&function, &hierarchy()).unwrap();
    let mlil_function = lifting.builder.finish().unwrap();

    let lowered = lower::<Managed>(&mlil_function).unwrap();
    let remapped = lowered
        .function
        .cfg()
        .edges()
        .find_map(|edge| match edge.payload() {
            JvmRtlEdge::Except { site } => Some(*site),
            _ => None,
        })
        .expect("the exceptional edge survives the lowering");
    let statement = remapped.expect("the lowered payload names the owning throw statement");
    let throwing = lowered
        .function
        .cfg()
        .blocks()
        .flat_map(crate::BasicBlock::instructions)
        .find(|node| node.id() == statement)
        .expect("the named statement exists in the lowered function");
    assert!(
        throwing.statement().may_throw(),
        "the remapped site is the throwing statement"
    );
}

/// An emission that ignores the statement's exceptional behavior is
/// rejected instead of silently dropping it.
#[test]
fn dropped_exceptional_behavior_is_rejected() {
    let mut builder = FunctionBuilder::<Managed>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, JvmRtlEdge::Entry).unwrap();
    builder
        .append(
            body,
            Statement::Effect {
                operation: EffectOp::DropThrow,
                operands: Vec::new(),
                effects: vec![Effect::Call],
                may_throw: true,
            },
            None,
        )
        .unwrap();
    builder
        .append(body, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().unwrap();
    assert!(
        lift(&function, &hierarchy()).is_err(),
        "the dropped throw must be caught"
    );
}

/// An emission that references a variable outside the statement's reads
/// is rejected — read/definition alignment survives consumer expansion.
#[test]
fn foreign_operand_is_rejected() {
    let mut builder = FunctionBuilder::<Managed>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, JvmRtlEdge::Entry).unwrap();
    builder
        .append(
            body,
            Statement::Effect {
                operation: EffectOp::Smuggle,
                operands: Vec::new(),
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
    assert!(
        lift(&function, &hierarchy()).is_err(),
        "the foreign operand must be caught"
    );
}

/// A block with exceptional edges must contain exactly one throwing
/// statement to own them.
#[test]
fn exceptional_edges_need_one_owning_statement() {
    let mut builder = FunctionBuilder::<Managed>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    let exit = builder.new_block("exit");
    let handler = builder.new_block("handler");
    builder.add_edge(entry, body, JvmRtlEdge::Entry).unwrap();
    builder.add_edge(body, exit, JvmRtlEdge::Fall).unwrap();
    builder
        .add_edge(body, handler, JvmRtlEdge::Except { site: None })
        .unwrap();
    for _ in 0..2 {
        builder
            .append(
                body,
                Statement::Effect {
                    operation: EffectOp::Invoke,
                    operands: Vec::new(),
                    effects: vec![Effect::Call],
                    may_throw: true,
                },
                None,
            )
            .unwrap();
    }
    builder
        .append(exit, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    builder
        .append(
            handler,
            Statement::Raise {
                operation: EffectOp::Throw,
                operands: Vec::new(),
                effects: Vec::new(),
            },
            None,
        )
        .unwrap();
    assert!(
        builder.finish().is_err(),
        "two throwing statements cannot share the block's edges"
    );
}

/// A raise block must not continue normally.
#[test]
fn raise_with_normal_edge_is_rejected() {
    let mut builder = FunctionBuilder::<Managed>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    let after = builder.new_block("after");
    builder.add_edge(entry, body, JvmRtlEdge::Entry).unwrap();
    builder.add_edge(body, after, JvmRtlEdge::Fall).unwrap();
    builder
        .append(
            body,
            Statement::Raise {
                operation: EffectOp::Throw,
                operands: Vec::new(),
                effects: Vec::new(),
            },
            None,
        )
        .unwrap();
    builder
        .append(after, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    assert!(builder.finish().is_err());
}
