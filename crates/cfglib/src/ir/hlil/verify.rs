//! Structural verification shared by every HLIL dialect.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::statement::{child_bodies, expression_references};
use super::{
    Dialect, EntityId, ExpressionKind, Function, StatementId, StatementKind, VerificationIssue,
    VerificationReport, VerifyDialect,
};

pub(super) fn verify_function<D: VerifyDialect>(function: &Function<D>) -> VerificationReport {
    let mut issues = Vec::new();
    verify_identities(function, &mut issues);
    verify_signature(function, &mut issues);
    verify_expressions(function, &mut issues);
    verify_statements(function, &mut issues);
    let acyclic = issues.is_empty();
    if acyclic {
        // The context walk recurses through bodies; it only runs once the
        // flat passes prove children precede parents (no cycles).
        verify_context(function, &mut issues);
    }
    verify_provenance(function, &mut issues);
    D::verify(function, &mut issues);
    VerificationReport::new(super::error::LEVEL, issues)
}

fn verify_identities<D: Dialect>(function: &Function<D>, issues: &mut Vec<VerificationIssue>) {
    for (index, variable) in function.variables.iter().enumerate() {
        if variable.id.index() != index {
            issue(
                issues,
                format!(
                    "variable table slot {index} contains non-dense identity {}",
                    variable.id
                ),
            );
        }
    }
    for (index, expression) in function.expressions.iter().enumerate() {
        if expression.id.index() != index {
            issue(
                issues,
                format!(
                    "expression table slot {index} contains non-dense identity {}",
                    expression.id
                ),
            );
        }
    }
    for (index, statement) in function.statements.iter().enumerate() {
        if statement.id.index() != index {
            issue(
                issues,
                format!(
                    "statement table slot {index} contains non-dense identity {}",
                    statement.id
                ),
            );
        }
    }
}

fn verify_signature<D: Dialect>(function: &Function<D>, issues: &mut Vec<VerificationIssue>) {
    for message in function
        .signature
        .parameter_issues(|&parameter| function.variable(parameter).is_some())
    {
        issue(issues, message);
    }
}

fn verify_expressions<D: Dialect>(function: &Function<D>, issues: &mut Vec<VerificationIssue>) {
    let mut references: Vec<u32> = vec![0; function.expressions.len()];
    for expression in &function.expressions {
        match expression.kind() {
            ExpressionKind::Variable(variable) => {
                if function.variable(*variable).is_none() {
                    issue(
                        issues,
                        format!("{} reads undeclared variable {variable}", expression.id),
                    );
                }
            }
            ExpressionKind::Constant(_) => {}
            ExpressionKind::Operation { operands, .. } => {
                for &operand in operands {
                    if operand.index() >= function.expressions.len() {
                        issue(
                            issues,
                            format!("{} names missing operand {operand}", expression.id),
                        );
                    } else if operand >= expression.id {
                        issue(
                            issues,
                            format!(
                                "{} operand {operand} does not precede its parent",
                                expression.id
                            ),
                        );
                    } else {
                        references[operand.index()] += 1;
                    }
                }
            }
        }
    }
    for statement in &function.statements {
        expression_references(statement.kind(), &mut |expression| {
            if expression.index() >= function.expressions.len() {
                issue(
                    issues,
                    format!("{} names missing expression {expression}", statement.id),
                );
            } else {
                references[expression.index()] += 1;
            }
        });
    }
    for (index, count) in references.iter().enumerate() {
        if *count != 1 {
            issue(
                issues,
                format!(
                    "expression e{index} is referenced {count} times; every occurrence \
                     is one tree node with exactly one parent"
                ),
            );
        }
    }
}

fn verify_statements<D: Dialect>(function: &Function<D>, issues: &mut Vec<VerificationIssue>) {
    let mut references: Vec<u32> = vec![0; function.statements.len()];
    for statement in &function.statements {
        for body in child_bodies(statement.kind()) {
            for &child in body {
                if child.index() >= function.statements.len() {
                    issue(
                        issues,
                        format!("{} names missing statement {child}", statement.id),
                    );
                } else if child >= statement.id {
                    issue(
                        issues,
                        format!("{} child {child} does not precede its parent", statement.id),
                    );
                } else {
                    references[child.index()] += 1;
                }
            }
        }
        verify_statement_shape(function, statement.id, statement.kind(), issues);
    }
    for &root in &function.body {
        if root.index() >= function.statements.len() {
            issue(issues, format!("body names missing statement {root}"));
        } else {
            references[root.index()] += 1;
        }
    }
    for (index, count) in references.iter().enumerate() {
        if *count != 1 {
            issue(
                issues,
                format!(
                    "statement s{index} is referenced {count} times; the statement tree \
                     is rooted at the body with exactly one parent per statement"
                ),
            );
        }
    }
}

fn verify_statement_shape<D: Dialect>(
    function: &Function<D>,
    id: StatementId,
    kind: &StatementKind<D>,
    issues: &mut Vec<VerificationIssue>,
) {
    match kind {
        StatementKind::Assign { target, .. } => {
            if let Some(expression) = function.expression(*target)
                && matches!(expression.kind(), ExpressionKind::Constant(_))
            {
                issue(issues, format!("{id} assigns into constant {target}"));
            }
        }
        StatementKind::Switch { cases, .. } => {
            for case in cases {
                if case.values.is_empty() {
                    issue(issues, format!("{id} has an arm with no case values"));
                }
            }
        }
        StatementKind::Try { handlers, .. } => {
            for handler in handlers {
                if let Some(binding) = handler.binding
                    && function.variable(binding).is_none()
                {
                    issue(issues, format!("{id} binds undeclared variable {binding}"));
                }
            }
        }
        _ => {}
    }
}

/// Enclosing-construct facts for break/continue/label checking.
struct Context {
    loop_depth: usize,
    breakable_depth: usize,
    labels: Vec<String>,
    /// Labels whose immediate body is (or contains only) loop statements
    /// do not need distinguishing: a labeled continue is checked against
    /// the label stack alone, matching source languages that allow
    /// labels on any statement.
    defined_labels: BTreeMap<String, u32>,
}

fn verify_context<D: Dialect>(function: &Function<D>, issues: &mut Vec<VerificationIssue>) {
    let mut context = Context {
        loop_depth: 0,
        breakable_depth: 0,
        labels: Vec::new(),
        defined_labels: BTreeMap::new(),
    };
    // Label definitions are collected first so forward gotos resolve.
    function.visit_statements(&function.body, &mut |statement| {
        if let StatementKind::Labeled { label, .. } = statement.kind() {
            *context.defined_labels.entry(label.clone()).or_insert(0) += 1;
        }
    });
    for (label, count) in &context.defined_labels {
        if *count > 1 {
            issue(issues, format!("label {label} is defined {count} times"));
        }
    }
    walk_context(function, &function.body, &mut context, issues);
}

fn walk_context<D: Dialect>(
    function: &Function<D>,
    roots: &[StatementId],
    context: &mut Context,
    issues: &mut Vec<VerificationIssue>,
) {
    for &root in roots {
        let Some(statement) = function.statement(root) else {
            continue;
        };
        let id = statement.id;
        match statement.kind() {
            StatementKind::Break { label } => match label {
                None if context.breakable_depth == 0 => {
                    issue(issues, format!("{id} breaks outside any loop or switch"));
                }
                Some(label) if !context.labels.contains(label) => {
                    issue(issues, format!("{id} breaks unenclosing label {label}"));
                }
                _ => {}
            },
            StatementKind::Continue { label } => match label {
                None if context.loop_depth == 0 => {
                    issue(issues, format!("{id} continues outside any loop"));
                }
                Some(label) if !context.labels.contains(label) => {
                    issue(issues, format!("{id} continues unenclosing label {label}"));
                }
                _ => {}
            },
            StatementKind::Goto { label } => {
                if !context.defined_labels.contains_key(label) {
                    issue(issues, format!("{id} targets undefined label {label}"));
                }
            }
            StatementKind::While { .. }
            | StatementKind::DoWhile { .. }
            | StatementKind::Loop { .. }
            | StatementKind::For { .. } => {
                context.loop_depth += 1;
                context.breakable_depth += 1;
                descend(function, statement.kind(), context, issues);
                context.loop_depth -= 1;
                context.breakable_depth -= 1;
                continue;
            }
            StatementKind::Switch { .. } => {
                context.breakable_depth += 1;
                descend(function, statement.kind(), context, issues);
                context.breakable_depth -= 1;
                continue;
            }
            StatementKind::Labeled { label, .. } => {
                context.labels.push(label.clone());
                descend(function, statement.kind(), context, issues);
                context.labels.pop();
                continue;
            }
            _ => {}
        }
        descend(function, statement.kind(), context, issues);
    }
}

fn descend<D: Dialect>(
    function: &Function<D>,
    kind: &StatementKind<D>,
    context: &mut Context,
    issues: &mut Vec<VerificationIssue>,
) {
    for body in child_bodies(kind) {
        walk_context(function, body, context, issues);
    }
}

fn verify_provenance<D: Dialect>(function: &Function<D>, issues: &mut Vec<VerificationIssue>) {
    for entry in function.provenance.entries() {
        let valid = match entry.entity {
            EntityId::Statement(statement) => function.statement(statement).is_some(),
            EntityId::Expression(expression) => function.expression(expression).is_some(),
            EntityId::Variable(variable) => function.variable(variable).is_some(),
        };
        if !valid {
            issue(
                issues,
                format!("provenance names missing entity {:?}", entry.entity),
            );
        }
    }
}

fn issue(issues: &mut Vec<VerificationIssue>, message: impl Into<String>) {
    issues.push(VerificationIssue::new(message));
}
