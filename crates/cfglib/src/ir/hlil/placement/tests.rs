extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::ir::dialect::Vocabulary;
use crate::test_util::toy;

use super::super::{
    Dialect, ExpressionId, ExpressionKind, Function, FunctionBuilder, StatementId, StatementKind,
    VariableId, VariablePlacements, VerificationIssue, VerifyDialect,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Type {
    Integer,
    Boolean,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Toy;

impl Vocabulary for Toy {
    type ValueType = Type;
    type Effect = ();
    type Source = ();
    type SourceSpan = toy::Span;
    type SourcePoint = u32;
    type VariableRole = ();
    type NativeVariable = ();

    fn span_is_empty(span: &Self::SourceSpan) -> bool {
        toy::span_is_empty(*span)
    }

    fn span_contains(span: &Self::SourceSpan, point: &Self::SourcePoint) -> bool {
        toy::span_contains(*span, *point)
    }
}

impl Dialect for Toy {
    type Operation = ();
    type Constant = i32;

    fn mnemonic((): &Self::Operation) -> &'static str {
        "operation"
    }
}

impl VerifyDialect for Toy {
    fn verify(_function: &Function<Self>, _issues: &mut Vec<VerificationIssue>) {}
}

fn variable(builder: &mut FunctionBuilder<Toy>) -> VariableId {
    builder
        .declare_variable((), None, Some(Type::Integer))
        .expect("the test function has room for a variable")
}

fn read(builder: &mut FunctionBuilder<Toy>, variable: VariableId) -> StatementId {
    let expression = builder
        .add_expression(ExpressionKind::Variable(variable), Type::Integer)
        .expect("the test function has room for a variable expression");
    builder
        .add_statement(StatementKind::Expression(expression), None)
        .expect("the test function has room for an expression statement")
}

fn constant_statement(builder: &mut FunctionBuilder<Toy>) -> StatementId {
    let expression = builder
        .add_expression(ExpressionKind::Constant(0), Type::Integer)
        .expect("the test function has room for a constant expression");
    builder
        .add_statement(StatementKind::Expression(expression), None)
        .expect("the test function has room for an expression statement")
}

fn condition(builder: &mut FunctionBuilder<Toy>) -> ExpressionId {
    builder
        .add_expression(ExpressionKind::Constant(1), Type::Boolean)
        .expect("the test function has room for a condition")
}

fn finish(mut builder: FunctionBuilder<Toy>, body: Vec<StatementId>) -> Function<Toy> {
    builder
        .set_body(body)
        .expect("the test statements belong to the function");
    builder
        .finish()
        .expect("the test function satisfies HLIL invariants")
}

#[test]
fn shared_branch_variable_anchors_before_if() {
    let mut builder = FunctionBuilder::new(());
    let value = variable(&mut builder);
    let then_use = read(&mut builder, value);
    let else_use = read(&mut builder, value);
    let branch_condition = condition(&mut builder);
    let branch = builder
        .add_statement(
            StatementKind::If {
                condition: branch_condition,
                then_body: vec![then_use],
                else_body: vec![else_use],
            },
            None,
        )
        .expect("the test function has room for an if statement");
    let function = finish(builder, vec![branch]);

    let placements = VariablePlacements::compute(&function);

    assert_eq!(placements.placement(value), Some(branch));
    assert_eq!(placements.before(branch), [value]);
}

#[test]
fn branch_local_variable_stays_in_the_branch_body() {
    let mut builder = FunctionBuilder::new(());
    let value = variable(&mut builder);
    let unrelated = constant_statement(&mut builder);
    let first_use = read(&mut builder, value);
    let second_use = read(&mut builder, value);
    let branch_condition = condition(&mut builder);
    let branch = builder
        .add_statement(
            StatementKind::If {
                condition: branch_condition,
                then_body: vec![unrelated, first_use, second_use],
                else_body: Vec::new(),
            },
            None,
        )
        .expect("the test function has room for an if statement");
    let function = finish(builder, vec![branch]);

    let placements = VariablePlacements::compute(&function);

    assert_eq!(placements.placement(value), Some(first_use));
    assert_eq!(placements.before(branch), []);
    assert_eq!(placements.before(first_use), [value]);
}

#[test]
fn loop_local_variable_stays_in_the_loop_body() {
    let mut builder = FunctionBuilder::new(());
    let value = variable(&mut builder);
    let first_use = read(&mut builder, value);
    let second_use = read(&mut builder, value);
    let loop_condition = condition(&mut builder);
    let loop_statement = builder
        .add_statement(
            StatementKind::While {
                condition: loop_condition,
                body: vec![first_use, second_use],
            },
            None,
        )
        .expect("the test function has room for a while statement");
    let function = finish(builder, vec![loop_statement]);

    let placements = VariablePlacements::compute(&function);

    assert_eq!(placements.placement(value), Some(first_use));
    assert_eq!(placements.before(loop_statement), []);
}

#[test]
fn for_clause_variable_anchors_before_the_loop() {
    let mut builder = FunctionBuilder::new(());
    let value = variable(&mut builder);
    let initializer = read(&mut builder, value);
    let loop_condition = condition(&mut builder);
    let loop_statement = builder
        .add_statement(
            StatementKind::For {
                initializer: vec![initializer],
                condition: Some(loop_condition),
                update: Vec::new(),
                body: Vec::new(),
            },
            None,
        )
        .expect("the test function has room for a for statement");
    let function = finish(builder, vec![loop_statement]);

    let placements = VariablePlacements::compute(&function);

    assert_eq!(placements.placement(value), Some(loop_statement));
    assert_eq!(placements.before(loop_statement), [value]);
    assert_eq!(placements.before(initializer), []);
}

#[test]
fn statement_visiting_follows_for_execution_order() {
    let mut builder = FunctionBuilder::new(());
    let initializer = constant_statement(&mut builder);
    let body = constant_statement(&mut builder);
    let update = constant_statement(&mut builder);
    let loop_statement = builder
        .add_statement(
            StatementKind::For {
                initializer: vec![initializer],
                condition: None,
                update: vec![update],
                body: vec![body],
            },
            None,
        )
        .expect("the test function has room for a for statement");
    let function = finish(builder, vec![loop_statement]);
    let mut visited = Vec::new();

    function.visit_statements(function.body(), &mut |statement| {
        visited.push(statement.id());
    });

    assert_eq!(visited, [loop_statement, initializer, body, update]);
}
