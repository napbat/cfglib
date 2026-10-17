//! The walks that every consumer of the shapes needs, checked against
//! walks that the test writes over its own storage.

use cfglib::ir::hlil::{
    ChildBodyKind, ExprShape, FunctionView, compound_assignment, expressions_equal,
    for_each_child_body, for_each_statement_expression, visit_expression_tree, visit_statements,
};

use crate::describe;
use crate::dialect::{Calc, Op};
use crate::pipeline::{lifted, lowered, recovered, structured};
use crate::programs;
use crate::structured::{ExprId, ExprNode, Function, StmtId, StmtNode};
use crate::structured_run::{bodies, preorder};

/// The source functions, the lifted ones, and the recovered ones: the shapes
/// that a walk meets in practice.
fn functions() -> Vec<(String, Function)> {
    let mut all = Vec::new();
    for case in programs::all() {
        let source = structured(&case.program);
        let lifted = lifted(&lowered(&source).function).function;
        let recovered = recovered(&lifted).function;
        let name = case.program.name;
        all.push((format!("{name} (source)"), source));
        all.push((format!("{name} (lifted)"), lifted));
        all.push((format!("{name} (recovered)"), recovered));
    }
    all
}

#[test]
fn visit_statements_walks_in_preorder() {
    for (name, function) in functions() {
        let mut visited = Vec::new();
        visit_statements::<Calc, _>(&function, function.body(), &mut |stmt| visited.push(stmt));
        assert_eq!(visited, preorder(&function), "{name}");
    }
}

#[test]
fn for_each_child_body_lists_the_bodies_in_execution_order_with_their_scope_role() {
    for (name, function) in functions() {
        for index in 0..function.stmts.len() {
            let stmt = StmtId::from_index(index);
            let mut found: Vec<(&[StmtId], ChildBodyKind)> = Vec::new();
            for_each_child_body::<Calc, _>(&function, stmt, |body, kind| found.push((body, kind)));
            let expected = bodies(&function, stmt);
            let listed: Vec<&[StmtId]> = found.iter().map(|&(body, _)| body).collect();
            assert_eq!(listed, expected, "{name} s{index}");
            // Only the clauses of a `for` belong to the syntax of the parent.
            let clauses: Vec<usize> = found
                .iter()
                .enumerate()
                .filter(|&(_, &(_, kind))| kind == ChildBodyKind::Clause)
                .map(|(position, _)| position)
                .collect();
            let expected_clauses: &[usize] =
                if matches!(function.stmts[index], StmtNode::For { .. }) {
                    &[0, 2]
                } else {
                    &[]
                };
            assert_eq!(clauses, expected_clauses, "{name} s{index}");
        }
    }
}

/// The expressions that a statement references itself.
fn direct_expressions(statement: &StmtNode) -> Vec<ExprId> {
    match statement {
        StmtNode::Expression(expr) => vec![*expr],
        StmtNode::Assign { target, value } => vec![*target, *value],
        StmtNode::If { condition, .. }
        | StmtNode::While { condition, .. }
        | StmtNode::DoWhile { condition, .. } => vec![*condition],
        StmtNode::For { condition, .. } => condition.iter().copied().collect(),
        StmtNode::Switch { scrutinee, .. } => vec![*scrutinee],
        StmtNode::Return(values) => values.clone(),
        StmtNode::Region { operands, .. } => operands.clone(),
        _ => Vec::new(),
    }
}

#[test]
fn for_each_statement_expression_visits_what_the_statement_references_directly() {
    for (name, function) in functions() {
        for (index, statement) in function.stmts.iter().enumerate() {
            let mut visited = Vec::new();
            for_each_statement_expression::<Calc, _>(
                &function,
                StmtId::from_index(index),
                |expr| {
                    visited.push(expr);
                },
            );
            let mut expected = direct_expressions(statement);
            visited.sort();
            expected.sort();
            assert_eq!(visited, expected, "{name} s{index}");
        }
    }
}

fn postorder(function: &Function, root: ExprId, order: &mut Vec<ExprId>) {
    if let ExprNode::Op { operands, .. } = &function.exprs[root.index()].node {
        for &operand in operands {
            postorder(function, operand, order);
        }
    }
    order.push(root);
}

#[test]
fn visit_expression_tree_visits_operands_before_their_operation() {
    for (name, function) in functions() {
        for index in 0..function.exprs.len() {
            let root = ExprId::from_index(index);
            let mut visited = Vec::new();
            visit_expression_tree::<Calc, _>(&function, root, &mut |expr| visited.push(expr));
            let mut expected = Vec::new();
            postorder(&function, root, &mut expected);
            assert_eq!(visited, expected, "{name} e{index}");
        }
    }
}

#[test]
fn expressions_are_equal_exactly_when_their_trees_read_the_same() {
    let mut equal_pairs = 0;
    for (name, function) in functions() {
        let texts: Vec<String> = (0..function.exprs.len())
            .map(|index| describe::expression(&function, ExprId::from_index(index)))
            .collect();
        for (left, left_text) in texts.iter().enumerate() {
            for (right, right_text) in texts.iter().enumerate() {
                let same = expressions_equal::<Calc, _>(
                    &function,
                    ExprId::from_index(left),
                    ExprId::from_index(right),
                );
                assert_eq!(same, left_text == right_text, "{name} e{left} e{right}");
                equal_pairs += usize::from(same && left != right);
            }
        }
    }
    assert!(equal_pairs > 100, "only {equal_pairs} distinct equal pairs");
}

#[test]
fn a_compound_assignment_is_a_binary_operation_whose_first_operand_is_the_target() {
    let mut compound = 0;
    for (name, function) in functions() {
        for (index, statement) in function.stmts.iter().enumerate() {
            let found = compound_assignment::<Calc, _>(&function, StmtId::from_index(index));
            let expected = match statement {
                StmtNode::Assign { target, value } => match &function.exprs[value.index()].node {
                    ExprNode::Op { op, operands }
                        if operands.len() == 2
                            && describe::expression(&function, operands[0])
                                == describe::expression(&function, *target) =>
                    {
                        Some((*op, operands[1]))
                    }
                    _ => None,
                },
                _ => None,
            };
            assert_eq!(
                found.map(|(op, operand)| (*op, operand)),
                expected,
                "{name} s{index}"
            );
            compound += usize::from(found.is_some());
        }
    }
    assert!(compound > 50, "only {compound} compound assignments");
}

#[test]
fn a_commutative_operation_with_the_target_second_is_not_a_compound_assignment() {
    // The doc leaves that judgment to the caller: the walk answers only for
    // the target-first form.
    use crate::source::{Program, add, int, ret, set, var};
    let function = structured(&Program::new(
        "target_second",
        &["x"],
        &[],
        vec![set("x", add(int(1), var("x"))), ret(var("x"))],
    ));
    assert!(compound_assignment::<Calc, _>(&function, function.body()[0]).is_none());
    let shape = function.expr(match function.stmt(function.body()[0]) {
        cfglib::ir::hlil::StmtShape::Assign { value, .. } => value,
        other => panic!("{other:?}"),
    });
    assert!(matches!(shape, ExprShape::Operation { op: Op::Add, .. }));
}
