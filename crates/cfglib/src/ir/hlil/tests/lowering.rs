//! HLIL → MLIL lowering tests, split out to respect the source-size policy.

extern crate alloc;

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use super::super::{ExpressionKind, FunctionBuilder, Signature, StatementKind};
use super::{Operation, Toy, Type};
use crate::ir::hlil::{lift_function, lower_function};
use crate::test_util::golden::assert_golden;

/// `while (lt(i, n)) { if (lt(i, 100)) { break; } i = add(i, 1); } return i;`
fn structured_counting_loop() -> crate::ir::hlil::Function<Toy> {
    let mut builder = FunctionBuilder::<Toy>::new("toy::structured".into());
    let i = builder
        .declare_variable(0, None, Some(Type::Integer))
        .unwrap();
    let n = builder
        .declare_variable(0, None, Some(Type::Integer))
        .unwrap();
    let read = |builder: &mut FunctionBuilder<Toy>, variable| {
        builder
            .add_expression(ExpressionKind::Variable(variable), Type::Integer)
            .unwrap()
    };

    let limit = builder
        .add_expression(ExpressionKind::Constant(100), Type::Integer)
        .unwrap();
    let read_i_break = read(&mut builder, i);
    let break_condition = builder
        .add_expression(
            ExpressionKind::Operation {
                operation: Operation::LessThan,
                operands: vec![read_i_break, limit],
            },
            Type::Boolean,
        )
        .unwrap();
    let break_statement = builder
        .add_statement(StatementKind::Break { label: None }, None)
        .unwrap();
    let break_if = builder
        .add_statement(
            StatementKind::If {
                condition: break_condition,
                then_body: vec![break_statement],
                else_body: Vec::new(),
            },
            None,
        )
        .unwrap();

    let read_i_sum = read(&mut builder, i);
    let one = builder
        .add_expression(ExpressionKind::Constant(1), Type::Integer)
        .unwrap();
    let sum = builder
        .add_expression(
            ExpressionKind::Operation {
                operation: Operation::Add,
                operands: vec![read_i_sum, one],
            },
            Type::Integer,
        )
        .unwrap();
    let target = read(&mut builder, i);
    let assign = builder
        .add_statement(StatementKind::Assign { target, value: sum }, None)
        .unwrap();

    let read_i_cond = read(&mut builder, i);
    let read_n = read(&mut builder, n);
    let condition = builder
        .add_expression(
            ExpressionKind::Operation {
                operation: Operation::LessThan,
                operands: vec![read_i_cond, read_n],
            },
            Type::Boolean,
        )
        .unwrap();
    let while_statement = builder
        .add_statement(
            StatementKind::While {
                condition,
                body: vec![break_if, assign],
            },
            None,
        )
        .unwrap();
    let result = read(&mut builder, i);
    let return_statement = builder
        .add_statement(
            StatementKind::Return {
                values: vec![result],
            },
            None,
        )
        .unwrap();
    builder
        .set_signature(Signature::<Toy>::new(vec![i, n], vec![Type::Integer]))
        .unwrap();
    builder
        .set_body(vec![while_statement, return_statement])
        .unwrap();
    builder.finish().unwrap()
}

#[test]
fn lowering_and_relifting_round_trips_a_while_loop() {
    let function = structured_counting_loop();
    let lowered = lower_function(&function).unwrap();
    assert!(lowered.function.verify().is_ok());
    assert!(!lowered.instructions.is_empty());

    let relifted = lift_function(&lowered.function).unwrap();
    assert!(
        relifted.report.is_fully_structured(),
        "{:?}",
        relifted.report
    );
    assert_golden(
        "hlil/relift-while-loop.pseudo",
        &relifted.function.to_pseudocode(),
    );
}

#[test]
fn lowering_and_relifting_round_trips_a_switch() {
    let mut builder = FunctionBuilder::<Toy>::new("toy::redispatch".into());
    let selector = builder
        .declare_variable(0, None, Some(Type::Integer))
        .unwrap();
    let result = builder
        .declare_variable(0, None, Some(Type::Integer))
        .unwrap();
    let arm = |builder: &mut FunctionBuilder<Toy>, value: i64| {
        let constant = builder
            .add_expression(ExpressionKind::Constant(value), Type::Integer)
            .unwrap();
        let target = builder
            .add_expression(ExpressionKind::Variable(result), Type::Integer)
            .unwrap();
        builder
            .add_statement(
                StatementKind::Assign {
                    target,
                    value: constant,
                },
                None,
            )
            .unwrap()
    };
    let first = arm(&mut builder, 10);
    let second = arm(&mut builder, 20);
    let fallback = arm(&mut builder, 30);
    let scrutinee = builder
        .add_expression(ExpressionKind::Variable(selector), Type::Integer)
        .unwrap();
    let switch = builder
        .add_statement(
            StatementKind::Switch {
                scrutinee,
                cases: vec![
                    crate::ir::hlil::SwitchArm {
                        values: vec![1, 2],
                        body: vec![first],
                    },
                    crate::ir::hlil::SwitchArm {
                        values: vec![3],
                        body: vec![second],
                    },
                ],
                default_body: vec![fallback],
            },
            None,
        )
        .unwrap();
    let read_result = builder
        .add_expression(ExpressionKind::Variable(result), Type::Integer)
        .unwrap();
    let return_statement = builder
        .add_statement(
            StatementKind::Return {
                values: vec![read_result],
            },
            None,
        )
        .unwrap();
    builder.set_body(vec![switch, return_statement]).unwrap();

    let lowered = lower_function(&builder.finish().unwrap()).unwrap();
    let relifted = lift_function(&lowered.function).unwrap();
    assert!(
        relifted.report.is_fully_structured(),
        "{:?}",
        relifted.report
    );
    assert_golden(
        "hlil/relift-switch.pseudo",
        &relifted.function.to_pseudocode(),
    );
}

#[test]
fn lowering_registers_declared_exception_regions() {
    let mut builder = FunctionBuilder::<Toy>::new("toy::reguard".into());
    let x = builder
        .declare_variable(0, None, Some(Type::Integer))
        .unwrap();
    let call = builder
        .add_expression(
            ExpressionKind::Operation {
                operation: Operation::Call,
                operands: Vec::new(),
            },
            Type::Integer,
        )
        .unwrap();
    let target = builder
        .add_expression(ExpressionKind::Variable(x), Type::Integer)
        .unwrap();
    let protected = builder
        .add_statement(
            StatementKind::Assign {
                target,
                value: call,
            },
            None,
        )
        .unwrap();
    let seven = builder
        .add_expression(ExpressionKind::Constant(7), Type::Integer)
        .unwrap();
    let handler_return = builder
        .add_statement(
            StatementKind::Return {
                values: vec![seven],
            },
            None,
        )
        .unwrap();
    let try_statement = builder
        .add_statement(
            StatementKind::Try {
                body: vec![protected],
                handlers: vec![crate::ir::hlil::Handler {
                    kind: crate::ir::hlil::HandlerKind::CatchAll,
                    binding: None,
                    caught_types: Vec::new(),
                    body: vec![handler_return],
                }],
                finally_body: Vec::new(),
            },
            None,
        )
        .unwrap();
    let read_x = builder
        .add_expression(ExpressionKind::Variable(x), Type::Integer)
        .unwrap();
    let return_statement = builder
        .add_statement(
            StatementKind::Return {
                values: vec![read_x],
            },
            None,
        )
        .unwrap();
    builder
        .set_body(vec![try_statement, return_statement])
        .unwrap();

    let lowered = lower_function(&builder.finish().unwrap()).unwrap();
    assert_eq!(lowered.function.cfg().regions().len(), 1);
    let region = &lowered.function.cfg().regions()[0];
    assert_eq!(region.handlers.len(), 1);
    assert!(region.handlers[0].body.is_known());

    let relifted = lift_function(&lowered.function).unwrap();
    assert_golden(
        "hlil/relift-try-catch.pseudo",
        &relifted.function.to_pseudocode(),
    );
}

#[test]
fn lowering_rejects_control_falling_off_the_end() {
    let mut builder = FunctionBuilder::<Toy>::new("toy::open-ended".into());
    let x = builder.declare_variable(0, None, None).unwrap();
    let one = builder
        .add_expression(ExpressionKind::Constant(1), Type::Integer)
        .unwrap();
    let target = builder
        .add_expression(ExpressionKind::Variable(x), Type::Integer)
        .unwrap();
    let assign = builder
        .add_statement(StatementKind::Assign { target, value: one }, None)
        .unwrap();
    builder.set_body(vec![assign]).unwrap();

    let error = lower_function(&builder.finish().unwrap())
        .unwrap_err()
        .to_string();
    assert!(error.contains("falls off the end"), "{error}");
}
