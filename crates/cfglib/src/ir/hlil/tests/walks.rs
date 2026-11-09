//! The walks over a function view.

extern crate alloc;

use alloc::vec::Vec;

use crate::ir::hlil::{
    ChildBodyKind, FunctionBuilder, FunctionView, Handler, HandlerKind, SwitchArm,
    for_each_child_body, for_each_statement_expression, visit_expression_tree, visit_statements,
};

use super::{Kit, Prim, StmtId, ToyHlil, ToyHlilBuilder, Ty};

over_toys!(
    statement_visiting_follows_for_execution_order,
    child_bodies_report_roles_in_execution_order,
    statement_expressions_are_the_direct_operands,
    expression_trees_visit_operands_before_parents,
    deep_expression_trees_do_not_recurse,
);

/// A `for` statement with one statement per section.
fn for_loop<D: Kit>() -> (ToyHlil<D>, [StmtId; 4]) {
    let mut builder = ToyHlilBuilder::<D>::new();
    let initializer = builder.noop();
    let body = builder.noop();
    let update = builder.noop();
    let looped = builder.for_(&[initializer], None, &[update], &[body]);
    let function = builder.finish(&[looped]).unwrap();
    (function, [looped, initializer, body, update])
}

fn statement_visiting_follows_for_execution_order<D: Kit>() {
    let (function, [looped, initializer, body, update]) = for_loop::<D>();
    let mut visited = Vec::new();

    visit_statements(&function, function.body(), &mut |statement| {
        visited.push(statement);
    });

    assert_eq!(visited, [looped, initializer, body, update]);
}

/// One statement of every shape that owns child bodies, each body holding
/// one leaf statement that names it.
fn child_bodies_report_roles_in_execution_order<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let then_leaf = builder.noop();
    let else_leaf = builder.noop();
    let initializer_leaf = builder.noop();
    let update_leaf = builder.noop();
    let loop_leaf = builder.noop();
    let first_arm_leaf = builder.noop();
    let second_arm_leaf = builder.noop();
    let default_leaf = builder.noop();
    let filter_leaf = builder.noop();
    let filtered_leaf = builder.noop();
    let catch_all_leaf = builder.noop();
    let protected_leaf = builder.noop();

    let condition = builder.condition();
    let conditional = builder.if_(condition, &[then_leaf], &[else_leaf]);
    let looped = builder.for_(&[initializer_leaf], None, &[update_leaf], &[loop_leaf]);
    let scrutinee = builder.number(0);
    let first_values = [D::constant(1)];
    let second_values = [D::constant(2)];
    let switch = builder.switch(
        scrutinee,
        &[
            SwitchArm {
                values: &first_values,
                body: &[first_arm_leaf],
            },
            SwitchArm {
                values: &second_values,
                body: &[second_arm_leaf],
            },
        ],
        &[default_leaf],
    );
    let filter = Handler {
        kind: HandlerKind::Filter {
            filter_body: &[filter_leaf],
        },
        binding: None,
        caught_types: &[],
        body: &[filtered_leaf],
    };
    let catch_all = Handler {
        kind: HandlerKind::CatchAll,
        binding: None,
        caught_types: &[],
        body: &[catch_all_leaf],
    };
    let guarded = builder.try_(&[protected_leaf], &[filter, catch_all], &[]);
    let function = builder
        .finish(&[conditional, looped, switch, guarded])
        .unwrap();

    let bodies = |statement: StmtId| {
        let mut bodies = Vec::new();
        for_each_child_body(&function, statement, |body, role| {
            bodies.push((body.to_vec(), role));
        });
        bodies
    };
    let lexical = |body: &[StmtId]| (body.to_vec(), ChildBodyKind::Lexical);
    let clause = |body: &[StmtId]| (body.to_vec(), ChildBodyKind::Clause);

    assert_eq!(
        bodies(conditional),
        [lexical(&[then_leaf]), lexical(&[else_leaf])]
    );
    assert_eq!(
        bodies(looped),
        [
            clause(&[initializer_leaf]),
            lexical(&[loop_leaf]),
            clause(&[update_leaf])
        ]
    );
    assert_eq!(
        bodies(switch),
        [
            lexical(&[first_arm_leaf]),
            lexical(&[second_arm_leaf]),
            lexical(&[default_leaf])
        ]
    );
    assert_eq!(
        bodies(guarded),
        [
            lexical(&[protected_leaf]),
            lexical(&[filter_leaf]),
            lexical(&[filtered_leaf]),
            lexical(&[catch_all_leaf]),
            lexical(&[])
        ]
    );
    assert!(
        bodies(then_leaf).is_empty(),
        "a leaf statement has no bodies"
    );
}

fn statement_expressions_are_the_direct_operands<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let counter = builder.int_variable(0);
    let one = builder.number(1);
    let read = builder.read(counter);
    let sum = builder.call(Prim::Add, &[read, one], Ty::Integer);
    let target = builder.read(counter);
    let assign = builder.assign(target, sum);
    let inner = builder.noop();
    let condition = builder.condition();
    let conditional = builder.if_(condition, &[assign, inner], &[]);
    let value = builder.number(7);
    let returned = builder.return_(&[value]);
    let function = builder
        .finish(&[conditional, returned])
        .expect("the function is well formed");

    let expressions = |statement: StmtId| {
        let mut expressions = Vec::new();
        for_each_statement_expression(&function, statement, |expression| {
            expressions.push(expression);
        });
        expressions
    };

    assert_eq!(expressions(assign), [target, sum]);
    assert_eq!(
        expressions(conditional),
        [condition],
        "expressions inside child bodies belong to the child statements"
    );
    assert_eq!(expressions(returned), [value]);
}

fn expression_trees_visit_operands_before_parents<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let counter = builder.int_variable(0);
    let left = builder.read(counter);
    let right = builder.number(3);
    let sum = builder.call(Prim::Add, &[left, right], Ty::Integer);
    let negated = builder.call(Prim::Not, &[sum], Ty::Integer);
    let statement = builder.expression_statement(negated);
    let function = builder.finish(&[statement]).unwrap();
    let mut visited = Vec::new();

    visit_expression_tree(&function, negated, &mut |expression| {
        visited.push(expression);
    });

    assert_eq!(visited, [left, right, sum, negated]);
}

/// The walks keep their own stack, so an operand chain as deep as generated
/// code can produce does not overflow the thread's.
fn deep_expression_trees_do_not_recurse<D: Kit>() {
    const DEPTH: usize = 20_000;

    let mut builder = ToyHlilBuilder::<D>::new();
    let counter = builder.int_variable(0);
    let chain = |builder: &mut ToyHlilBuilder<D>| {
        let mut sum = builder.read(counter);
        for _ in 0..DEPTH {
            let one = builder.number(1);
            sum = builder.call(Prim::Add, &[sum, one], Ty::Integer);
        }
        sum
    };
    let first = chain(&mut builder);
    let second = chain(&mut builder);
    let first_statement = builder.expression_statement(first);
    let second_statement = builder.expression_statement(second);
    let function = builder
        .finish(&[first_statement, second_statement])
        .expect("a deep chain is well formed");

    let mut visited = 0;
    visit_expression_tree(&function, first, &mut |_| visited += 1);
    assert_eq!(visited, 2 * DEPTH + 1);
    assert_eq!(function.expr_count(), 2 * (2 * DEPTH + 1));
}
