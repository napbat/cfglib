//! A function built through the builder reads back through the view, shape
//! for shape.

extern crate alloc;

use alloc::vec::Vec;

use crate::ir::hlil::{
    ExprShape, FunctionBuilder, FunctionView, Handler, HandlerKind, StmtShape, SwitchArm,
};

use super::{Kit, Prim, ToyHlilBuilder, Ty};

over_toys!(
    compound_statements_read_back_their_bodies,
    switch_arms_and_handlers_read_back_in_order,
    expression_trees_read_back_their_operands_and_types,
);

fn compound_statements_read_back_their_bodies<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let then_leaf = builder.noop();
    let else_leaf = builder.noop();
    let initializer = builder.noop();
    let update = builder.noop();
    let body = builder.noop();
    let condition = builder.condition();
    let conditional = builder.if_(condition, &[then_leaf], &[else_leaf]);
    let looped = builder.for_(&[initializer], None, &[update], &[body]);
    let function = builder
        .finish(&[conditional, looped])
        .expect("the function is well formed");

    assert_eq!(function.body(), [conditional, looped]);
    assert_eq!(function.stmt_count(), 7);
    assert_eq!(
        function.stmt(conditional),
        StmtShape::If {
            condition,
            then_body: &[then_leaf],
            else_body: &[else_leaf],
        }
    );
    assert_eq!(
        function.stmt(looped),
        StmtShape::For {
            initializer: &[initializer],
            condition: None,
            update: &[update],
            body: &[body],
        }
    );
    assert_eq!(function.switch_arms(conditional).len(), 0);
    assert_eq!(function.handlers(looped).len(), 0);
}

fn switch_arms_and_handlers_read_back_in_order<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let first_leaf = builder.noop();
    let second_leaf = builder.noop();
    let default_leaf = builder.noop();
    let filter_leaf = builder.noop();
    let filtered_leaf = builder.noop();
    let catch_all_leaf = builder.noop();
    let protected_leaf = builder.noop();
    let finally_leaf = builder.noop();

    let scrutinee = builder.number(0);
    let first_values = [D::constant(1)];
    let second_values = [D::constant(2), D::constant(3)];
    let arms = [
        SwitchArm {
            values: &first_values,
            body: &[first_leaf],
        },
        SwitchArm {
            values: &second_values,
            body: &[second_leaf],
        },
    ];
    let switch = builder.switch(scrutinee, &arms, &[default_leaf]);
    let handlers = [
        Handler {
            kind: HandlerKind::Filter {
                filter_body: &[filter_leaf],
            },
            binding: None,
            caught_types: &[],
            body: &[filtered_leaf],
        },
        Handler {
            kind: HandlerKind::CatchAll,
            binding: None,
            caught_types: &[],
            body: &[catch_all_leaf],
        },
    ];
    let guarded = builder.try_(&[protected_leaf], &handlers, &[finally_leaf]);
    let function = builder
        .finish(&[switch, guarded])
        .expect("the function is well formed");

    assert_eq!(
        function.stmt(switch),
        StmtShape::Switch {
            scrutinee,
            default_body: &[default_leaf],
        }
    );
    assert_eq!(function.switch_arms(switch).collect::<Vec<_>>(), arms);
    assert_eq!(
        function.stmt(guarded),
        StmtShape::Try {
            body: &[protected_leaf],
            finally_body: &[finally_leaf],
        }
    );
    assert_eq!(function.handlers(guarded).collect::<Vec<_>>(), handlers);
}

fn expression_trees_read_back_their_operands_and_types<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let counter = builder.int_variable(0);
    let left = builder.read(counter);
    let right = builder.number(3);
    let sum = builder.call(Prim::Add, &[left, right], Ty::Integer);
    let negated = builder.call(Prim::Not, &[sum], Ty::Boolean);
    let statement = builder.expression_statement(negated);
    let function = builder
        .finish(&[statement])
        .expect("the function is well formed");

    assert_eq!(function.stmt(statement), StmtShape::Expression(negated));
    assert_eq!(function.expr(left), ExprShape::Variable(counter));
    assert_eq!(function.expr(right), ExprShape::Constant(&D::constant(3)));
    assert_eq!(
        function.expr(negated),
        ExprShape::Operation {
            op: &D::op(Prim::Not),
            operands: &[sum],
        }
    );
    assert_eq!(function.ty(sum), &D::ty(Ty::Integer));
    assert_eq!(function.ty(negated), &D::ty(Ty::Boolean));
    assert_eq!(function.variable(counter), &D::variable(0));
    assert_eq!(function.declared_type(counter), Some(&D::ty(Ty::Integer)));
    assert_eq!(function.variable_count(), 1);
    assert_eq!(function.expr_count(), 4);
}
