//! Structural recovery tests, split out to respect the source-size policy.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{Edge, MediumOperation, Operation, Toy, Type};
use crate::ir::hlil::{
    ExpressionId, ExpressionKind, FunctionBuilder, StatementId, StatementKind, VariableId,
    lift_function, recover_structure,
};
use crate::ir::mlil;
use crate::test_util::golden::assert_golden;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UpdateForm {
    Add,
    ExpandedOperand,
}

fn variable(builder: &mut FunctionBuilder<Toy>, id: VariableId) -> ExpressionId {
    builder
        .add_expression(ExpressionKind::Variable(id), Type::Integer)
        .unwrap()
}

fn counting_update(
    builder: &mut FunctionBuilder<Toy>,
    index: VariableId,
    form: UpdateForm,
) -> ExpressionId {
    let left = variable(builder, index);
    let one = builder
        .add_expression(ExpressionKind::Constant(1), Type::Integer)
        .unwrap();
    let (operation, operands) = match form {
        UpdateForm::Add => (Operation::Add, vec![left, one]),
        UpdateForm::ExpandedOperand => {
            let expanded = builder
                .add_expression(
                    ExpressionKind::Operation {
                        operation: Operation::Expanded,
                        operands: vec![left],
                    },
                    Type::Integer,
                )
                .unwrap();
            (Operation::Add, vec![expanded, one])
        }
    };
    builder
        .add_expression(
            ExpressionKind::Operation {
                operation,
                operands,
            },
            Type::Integer,
        )
        .unwrap()
}

/// `index = 0; while (index < bound) { total = total + index; index =
/// index + 1; } return total;` — and the loop statement's identity.
fn counting_while(update_form: UpdateForm) -> (crate::ir::hlil::Function<Toy>, StatementId) {
    let mut builder = FunctionBuilder::<Toy>::new("toy::count".into());
    let bound = builder.declare_variable(0, None, None).unwrap();
    let index = builder.declare_variable(0, None, None).unwrap();
    let total = builder.declare_variable(0, None, None).unwrap();
    let init_target = variable(&mut builder, index);
    let zero = builder
        .add_expression(ExpressionKind::Constant(0), Type::Integer)
        .unwrap();
    let init = builder
        .add_statement(
            StatementKind::Assign {
                target: init_target,
                value: zero,
            },
            None,
        )
        .unwrap();
    let condition_index = variable(&mut builder, index);
    let condition_bound = variable(&mut builder, bound);
    let condition = builder
        .add_expression(
            ExpressionKind::Operation {
                operation: Operation::LessThan,
                operands: vec![condition_index, condition_bound],
            },
            Type::Boolean,
        )
        .unwrap();
    let sum_target = variable(&mut builder, total);
    let sum_left = variable(&mut builder, total);
    let sum_right = variable(&mut builder, index);
    let sum_value = builder
        .add_expression(
            ExpressionKind::Operation {
                operation: Operation::Add,
                operands: vec![sum_left, sum_right],
            },
            Type::Integer,
        )
        .unwrap();
    let accumulate = builder
        .add_statement(
            StatementKind::Assign {
                target: sum_target,
                value: sum_value,
            },
            None,
        )
        .unwrap();
    let update_target = variable(&mut builder, index);
    let update_value = counting_update(&mut builder, index, update_form);
    let update = builder
        .add_statement(
            StatementKind::Assign {
                target: update_target,
                value: update_value,
            },
            None,
        )
        .unwrap();
    let looped = builder
        .add_statement(
            StatementKind::While {
                condition,
                body: vec![accumulate, update],
            },
            None,
        )
        .unwrap();
    let result = variable(&mut builder, total);
    let ret = builder
        .add_statement(
            StatementKind::Return {
                values: vec![result],
            },
            None,
        )
        .unwrap();
    builder.set_body(vec![init, looped, ret]).unwrap();
    (builder.finish().unwrap(), looped)
}

#[test]
fn counted_loop_recovers_as_for() {
    let (function, looped) = counting_while(UpdateForm::Add);
    let recovery = recover_structure(&function).unwrap();
    assert_eq!(
        recovery.for_loops,
        1,
        "{}",
        recovery.function.to_pseudocode()
    );
    assert!(recovery.function.verify().is_ok());
    assert_golden(
        "hlil/recover-for-loop.pseudo",
        &recovery.function.to_pseudocode(),
    );
    // The loop maps to the recovered statement; the init keeps its copy.
    let recovered_loop = recovery.statements[&looped];
    assert!(matches!(
        recovery.function.statement(recovered_loop).unwrap().kind(),
        StatementKind::For { .. }
    ));
}

#[test]
fn counted_loop_keeps_a_nested_multi_statement_update_in_the_body() {
    let (function, looped) = counting_while(UpdateForm::ExpandedOperand);
    let recovery = recover_structure(&function).unwrap();

    assert_eq!(recovery.for_loops, 0);
    let recovered_loop = recovery.statements[&looped];
    assert!(matches!(
        recovery.function.statement(recovered_loop).unwrap().kind(),
        StatementKind::While { .. }
    ));
}

#[test]
fn assigning_diamond_recovers_as_selection() {
    let mut builder = FunctionBuilder::<Toy>::new("toy::pick".into());
    let flag = builder.declare_variable(0, None, None).unwrap();
    let out = builder.declare_variable(0, None, None).unwrap();
    let condition = builder
        .add_expression(ExpressionKind::Variable(flag), Type::Boolean)
        .unwrap();
    let then_target = builder
        .add_expression(ExpressionKind::Variable(out), Type::Integer)
        .unwrap();
    let then_value = builder
        .add_expression(ExpressionKind::Constant(1), Type::Integer)
        .unwrap();
    let then_assign = builder
        .add_statement(
            StatementKind::Assign {
                target: then_target,
                value: then_value,
            },
            None,
        )
        .unwrap();
    let else_target = builder
        .add_expression(ExpressionKind::Variable(out), Type::Integer)
        .unwrap();
    let else_value = builder
        .add_expression(ExpressionKind::Constant(2), Type::Integer)
        .unwrap();
    let else_assign = builder
        .add_statement(
            StatementKind::Assign {
                target: else_target,
                value: else_value,
            },
            None,
        )
        .unwrap();
    let diamond = builder
        .add_statement(
            StatementKind::If {
                condition,
                then_body: vec![then_assign],
                else_body: vec![else_assign],
            },
            None,
        )
        .unwrap();
    let result = builder
        .add_expression(ExpressionKind::Variable(out), Type::Integer)
        .unwrap();
    let ret = builder
        .add_statement(
            StatementKind::Return {
                values: vec![result],
            },
            None,
        )
        .unwrap();
    builder.set_body(vec![diamond, ret]).unwrap();
    let function = builder.finish().unwrap();

    let recovery = recover_structure(&function).unwrap();
    assert_eq!(recovery.selects, 1, "{}", recovery.function.to_pseudocode());
    assert!(recovery.function.verify().is_ok());
    // The diamond became one selection: no `if` survives in the golden.
    assert_golden(
        "hlil/recover-select.pseudo",
        &recovery.function.to_pseudocode(),
    );
    // The diamond and both arms map to the single recovered assignment.
    assert_eq!(
        recovery.statements[&diamond],
        recovery.statements[&then_assign]
    );
    assert_eq!(
        recovery.statements[&diamond],
        recovery.statements[&else_assign]
    );
}

/// The statements a recovered region consumes, by role.
struct GuardIds {
    enter: StatementId,
    guarded: StatementId,
    normal_exit: StatementId,
    cleanup_exit: StatementId,
    rethrow: StatementId,
}

/// `acquire(lock); try { out = 9; release(lock) } catch-any {
/// release(lock); throw(caught) }`.
fn guarded_function() -> (crate::ir::hlil::Function<Toy>, GuardIds) {
    let mut builder = FunctionBuilder::<Toy>::new("toy::guard".into());
    let lock = builder.declare_variable(0, None, None).unwrap();
    let out = builder.declare_variable(0, None, None).unwrap();
    let expr = |builder: &mut FunctionBuilder<Toy>, kind, value_type| {
        builder.add_expression(kind, value_type).unwrap()
    };
    let subject = expr(&mut builder, ExpressionKind::Variable(lock), Type::Integer);
    let enter_expression = expr(
        &mut builder,
        ExpressionKind::Operation {
            operation: Operation::Acquire,
            operands: vec![subject],
        },
        Type::Void,
    );
    let enter = builder
        .add_statement(StatementKind::Expression(enter_expression), None)
        .unwrap();
    let work_target = expr(&mut builder, ExpressionKind::Variable(out), Type::Integer);
    let work_value = expr(&mut builder, ExpressionKind::Constant(9), Type::Integer);
    let work = builder
        .add_statement(
            StatementKind::Assign {
                target: work_target,
                value: work_value,
            },
            None,
        )
        .unwrap();
    let exit_subject = expr(&mut builder, ExpressionKind::Variable(lock), Type::Integer);
    let exit_expression = expr(
        &mut builder,
        ExpressionKind::Operation {
            operation: Operation::Release,
            operands: vec![exit_subject],
        },
        Type::Void,
    );
    let normal_exit = builder
        .add_statement(StatementKind::Expression(exit_expression), None)
        .unwrap();
    let cleanup_subject = expr(&mut builder, ExpressionKind::Variable(lock), Type::Integer);
    let cleanup_exit_expression = expr(
        &mut builder,
        ExpressionKind::Operation {
            operation: Operation::Release,
            operands: vec![cleanup_subject],
        },
        Type::Void,
    );
    let cleanup_exit = builder
        .add_statement(StatementKind::Expression(cleanup_exit_expression), None)
        .unwrap();
    let caught = expr(
        &mut builder,
        ExpressionKind::Operation {
            operation: Operation::Caught,
            operands: Vec::new(),
        },
        Type::Integer,
    );
    let rethrow_expression = expr(
        &mut builder,
        ExpressionKind::Operation {
            operation: Operation::Throw,
            operands: vec![caught],
        },
        Type::Void,
    );
    let rethrow = builder
        .add_statement(StatementKind::Expression(rethrow_expression), None)
        .unwrap();
    let guarded = builder
        .add_statement(
            StatementKind::Try {
                body: vec![work, normal_exit],
                handlers: vec![crate::ir::hlil::Handler {
                    kind: crate::ir::hlil::HandlerKind::CatchAll,
                    binding: None,
                    caught_types: Vec::new(),
                    body: vec![cleanup_exit, rethrow],
                }],
                finally_body: Vec::new(),
            },
            None,
        )
        .unwrap();
    builder.set_body(vec![enter, guarded]).unwrap();
    (
        builder.finish().unwrap(),
        GuardIds {
            enter,
            guarded,
            normal_exit,
            cleanup_exit,
            rethrow,
        },
    )
}

#[test]
fn paired_region_recovers_with_suppressed_exits() {
    let (function, ids) = guarded_function();
    let recovery = recover_structure(&function).unwrap();
    assert_eq!(recovery.regions, 1, "{}", recovery.function.to_pseudocode());
    assert!(recovery.function.verify().is_ok());
    // The pair became one region: its release and its `try` are gone.
    assert_golden(
        "hlil/recover-region.pseudo",
        &recovery.function.to_pseudocode(),
    );
    // Enter, try, cleanup, and the suppressed normal exit all map onto the
    // region.
    let region = recovery.statements[&ids.enter];
    assert_eq!(recovery.statements[&ids.guarded], region);
    assert_eq!(recovery.statements[&ids.normal_exit], region);
    assert_eq!(recovery.statements[&ids.cleanup_exit], region);
    assert_eq!(recovery.statements[&ids.rethrow], region);
    assert!(matches!(
        recovery.function.statement(region).unwrap().kind(),
        StatementKind::Region { .. }
    ));
}

#[test]
fn exit_on_true_loop_negates_by_operation_inversion() {
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::invert".into());
    let header = builder.new_block("header");
    let body = builder.new_block("body");
    let exit = builder.new_block("exit");
    let index = builder.declare_variable(0, None).unwrap();
    let bound = builder.declare_variable(0, None).unwrap();
    let typed = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Integer);
    builder
        .append_instruction(
            header,
            MediumOperation::CompareBranch,
            vec![typed(index), typed(bound)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            MediumOperation::Constant(1),
            Vec::new(),
            vec![typed(index)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            exit,
            MediumOperation::Return,
            vec![typed(index)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), header, Edge::Entry, None)
        .unwrap();
    builder.add_edge(header, exit, Edge::True, None).unwrap();
    builder.add_edge(header, body, Edge::False, None).unwrap();
    builder.add_edge(body, header, Edge::Fall, None).unwrap();
    builder
        .set_signature(mlil::Signature::<Toy>::new(
            vec![index, bound],
            vec![Type::Integer],
        ))
        .unwrap();
    let function = builder.finish().unwrap();

    let lifted = lift_function(&function).unwrap();
    assert!(lifted.function.verify().is_ok());
    // The exit-on-true test inverted in place: `while (at-least(...))`
    // instead of a wrapped negation or a degraded endless loop.
    assert_golden(
        "hlil/recover-inverted-loop.pseudo",
        &lifted.function.to_pseudocode(),
    );
}
