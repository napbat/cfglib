//! Golden pseudocode for RTL functions.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::ir::signature::Signature;
use crate::test_util::golden::assert_golden;

use super::super::{
    Edge, Expr, Function, FunctionBuilder, Place, ScalarType, Statement, ValueShape,
};
use super::{Effect, EffectOp, Operator, TestDialect, apply, assign, constant, read};

fn place(storage: u8, lanes: &[u8]) -> Place<TestDialect> {
    Place {
        storage,
        lanes: lanes.to_vec(),
    }
}

/// A straight-line function: one typed transfer, one reinterpretation, and
/// a return, under a declared signature.
fn linear_function() -> Function<TestDialect> {
    let mut builder = FunctionBuilder::<TestDialect>::new("blend".into());
    builder
        .set_signature(Signature::new(
            vec![place(0, &[0]), place(1, &[0])],
            vec![ValueShape::scalar(ScalarType::F32)],
        ))
        .unwrap();
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    builder
        .append(
            body,
            assign(
                2,
                &[0],
                apply(
                    Operator::Add,
                    vec![
                        read(0, &[0], ScalarType::F32),
                        constant(0x3f80_0000, ScalarType::F32),
                    ],
                    ValueShape::scalar(ScalarType::F32),
                ),
            ),
            None,
        )
        .unwrap();
    builder
        .append(
            body,
            Statement::Return {
                values: vec![Expr::Reinterpret {
                    operand: Box::new(read(2, &[0], ScalarType::U32)),
                    shape: ValueShape::scalar(ScalarType::F32),
                }],
            },
            None,
        )
        .unwrap();
    builder.finish().unwrap()
}

/// A branching function: a parallel transfer, a decided branch, a throwing
/// effect with its handler, and a raise that leaves the function.
fn branching_function() -> Function<TestDialect> {
    let mut builder = FunctionBuilder::<TestDialect>::new("divide".into());
    let entry = builder.entry();
    let head = builder.new_block("head");
    let body = builder.new_block("body");
    let handler = builder.new_block("handler");
    let exit = builder.new_block("exit");
    builder.add_edge(entry, head, Edge::Entry).unwrap();

    builder
        .append(
            head,
            Statement::Transfer {
                assignments: vec![
                    (place(0, &[0]), read(1, &[0], ScalarType::U32)),
                    (place(1, &[0]), read(0, &[0], ScalarType::U32)),
                ],
                effects: Vec::new(),
                may_throw: false,
            },
            None,
        )
        .unwrap();
    builder
        .append(
            head,
            Statement::Branch {
                condition: apply(
                    Operator::Less,
                    vec![
                        read(0, &[0], ScalarType::U32),
                        constant(0x10, ScalarType::U32),
                    ],
                    ValueShape::scalar(ScalarType::Bool),
                ),
            },
            None,
        )
        .unwrap();
    builder.add_edge(head, body, Edge::True).unwrap();
    builder.add_edge(head, exit, Edge::False).unwrap();

    builder
        .append(
            body,
            Statement::Effect {
                operation: EffectOp::Emit,
                operands: vec![read(0, &[0], ScalarType::U32)],
                effects: vec![Effect::Emit],
                may_throw: true,
            },
            None,
        )
        .unwrap();
    builder.add_edge(body, exit, Edge::Fall).unwrap();
    builder.add_edge(body, handler, Edge::Unwind).unwrap();

    builder
        .append(
            handler,
            Statement::Raise {
                operation: EffectOp::Emit,
                operands: Vec::new(),
                effects: vec![Effect::Emit],
            },
            None,
        )
        .unwrap();
    builder
        .append(exit, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    builder.finish().unwrap()
}

#[test]
fn a_linear_function_prints_its_signature_transfers_and_return() {
    assert_golden("rtl/linear.pseudo", &linear_function().to_pseudocode());
}

#[test]
fn a_branching_function_prints_parallel_transfers_effects_and_edge_kinds() {
    assert_golden(
        "rtl/branching.pseudo",
        &branching_function().to_pseudocode(),
    );
}
