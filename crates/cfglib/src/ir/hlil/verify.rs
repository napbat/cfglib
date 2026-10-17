//! Structural verification of a structured function, and its report.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::graph::view::DenseId;

use super::dialect::VerifyDialect;
use super::name::Name;
use super::shape::{ExprShape, StmtShape};
use super::transfer::Untargeted;
use super::view::{
    FunctionView, for_each_child_body, for_each_statement_expression, labeled_loop,
    visit_statements,
};

/// One independently actionable verification failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationIssue {
    /// A description of the violated invariant.
    pub message: String,
}

impl VerificationIssue {
    /// Creates one verification issue.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for VerificationIssue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

/// The complete, deterministic result of verifying one function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationReport {
    /// Every structural and dialect failure, in the order the checks found
    /// them.
    pub issues: Vec<VerificationIssue>,
}

impl VerificationReport {
    /// Creates a report of `issues`.
    #[must_use]
    pub const fn new(issues: Vec<VerificationIssue>) -> Self {
        Self { issues }
    }

    /// Returns whether the function satisfies every checked invariant.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.issues.is_empty()
    }

    /// Returns the number of independent failures.
    #[must_use]
    pub fn issue_count(&self) -> usize {
        self.issues.len()
    }
}

impl fmt::Display for VerificationReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HLIL verification failed")?;
        for issue in &self.issues {
            write!(formatter, "; {issue}")?;
        }
        Ok(())
    }
}

impl core::error::Error for VerificationReport {}

/// Verifies the structure of `function` and then the invariants of its
/// dialect.
///
/// The structural checks cover:
///
/// - every parameter is a declared variable and appears once;
/// - every variable read names a declared variable;
/// - every operand index is in range and below its parent;
/// - every child statement index is in range and below its parent;
/// - every expression and every statement has exactly one parent;
/// - no assignment targets a constant, no switch arm lacks case values, and no
///   handler binds an undeclared variable;
/// - `break` and `continue` have an enclosing target, a labeled `continue`
///   names a loop (see [`StmtShape::Labeled`]), and every label is defined
///   once and every `goto` finds its label.
///
/// The enclosing-target checks need a function without cycles, so they run
/// only when the other structural checks found nothing. The hook of the
/// dialect runs unless a handle was out of range.
#[must_use]
pub fn verify<H, F>(function: &F) -> VerificationReport
where
    H: VerifyDialect,
    F: FunctionView<H>,
{
    let mut check = Check {
        issues: Vec::new(),
        dangling: false,
    };
    verify_signature(function, &mut check);
    verify_expressions(function, &mut check);
    verify_statements(function, &mut check);
    if check.issues.is_empty() {
        // The context walk recurses through bodies, so it waits until the
        // flat passes prove that children precede parents.
        verify_context(function, &mut check);
    }
    if !check.dangling {
        H::verify(function, &mut check.issues);
    }
    VerificationReport::new(check.issues)
}

/// The failures found so far, and whether any handle was out of range.
struct Check {
    issues: Vec<VerificationIssue>,
    dangling: bool,
}

impl Check {
    fn issue(&mut self, message: impl Into<String>) {
        self.issues.push(VerificationIssue::new(message));
    }

    /// Records a failure that names a handle outside the function.
    fn dangle(&mut self, message: impl Into<String>) {
        self.dangling = true;
        self.issue(message);
    }
}

fn verify_signature<H, F>(function: &F, check: &mut Check)
where
    H: VerifyDialect,
    F: FunctionView<H>,
{
    let mut seen = BTreeSet::new();
    for &parameter in function.parameters() {
        if parameter.index() >= function.variable_count() {
            check.dangle(format!(
                "signature names undeclared parameter {}",
                Name::variable(parameter)
            ));
        }
        if !seen.insert(parameter) {
            check.issue(format!(
                "signature repeats parameter {}",
                Name::variable(parameter)
            ));
        }
    }
}

fn verify_expressions<H, F>(function: &F, check: &mut Check)
where
    H: VerifyDialect,
    F: FunctionView<H>,
{
    let expr_count = function.expr_count();
    let mut references: Vec<u32> = vec![0; expr_count];
    for index in 0..expr_count {
        let expression = F::Expr::from_index(index);
        match function.expr(expression) {
            ExprShape::Variable(variable) => {
                if variable.index() >= function.variable_count() {
                    check.dangle(format!(
                        "{} reads undeclared variable {}",
                        Name::expression(expression),
                        Name::variable(variable)
                    ));
                }
            }
            ExprShape::Constant(_) => {}
            ExprShape::Operation { operands, .. } => {
                for &operand in operands {
                    if operand.index() >= expr_count {
                        check.dangle(format!(
                            "{} names missing operand {}",
                            Name::expression(expression),
                            Name::expression(operand)
                        ));
                    } else if operand.index() >= index {
                        check.issue(format!(
                            "{} operand {} does not precede its parent",
                            Name::expression(expression),
                            Name::expression(operand)
                        ));
                    } else {
                        references[operand.index()] += 1;
                    }
                }
            }
        }
    }
    for index in 0..function.stmt_count() {
        let statement = F::Stmt::from_index(index);
        for_each_statement_expression(function, statement, |expression| {
            if expression.index() >= expr_count {
                check.dangle(format!(
                    "{} names missing expression {}",
                    Name::statement(statement),
                    Name::expression(expression)
                ));
            } else {
                references[expression.index()] += 1;
            }
        });
    }
    for (index, count) in references.into_iter().enumerate() {
        if count != 1 {
            check.issue(format!(
                "expression {} is referenced {count} times; every occurrence \
                 is one tree node with exactly one parent",
                Name::expression(F::Expr::from_index(index))
            ));
        }
    }
}

fn verify_statements<H, F>(function: &F, check: &mut Check)
where
    H: VerifyDialect,
    F: FunctionView<H>,
{
    let stmt_count = function.stmt_count();
    let mut references: Vec<u32> = vec![0; stmt_count];
    for index in 0..stmt_count {
        let statement = F::Stmt::from_index(index);
        for_each_child_body(function, statement, |body, _| {
            for &child in body {
                if child.index() >= stmt_count {
                    check.dangle(format!(
                        "{} names missing statement {}",
                        Name::statement(statement),
                        Name::statement(child)
                    ));
                } else if child.index() >= index {
                    check.issue(format!(
                        "{} child {} does not precede its parent",
                        Name::statement(statement),
                        Name::statement(child)
                    ));
                } else {
                    references[child.index()] += 1;
                }
            }
        });
        verify_statement_shape(function, statement, check);
    }
    for &root in function.body() {
        if root.index() >= stmt_count {
            check.dangle(format!(
                "body names missing statement {}",
                Name::statement(root)
            ));
        } else {
            references[root.index()] += 1;
        }
    }
    for (index, count) in references.into_iter().enumerate() {
        if count != 1 {
            check.issue(format!(
                "statement {} is referenced {count} times; the statement tree \
                 is rooted at the body with exactly one parent per statement",
                Name::statement(F::Stmt::from_index(index))
            ));
        }
    }
}

fn verify_statement_shape<H, F>(function: &F, statement: F::Stmt, check: &mut Check)
where
    H: VerifyDialect,
    F: FunctionView<H>,
{
    match function.stmt(statement) {
        StmtShape::Assign { target, .. } => {
            if target.index() < function.expr_count()
                && matches!(function.expr(target), ExprShape::Constant(_))
            {
                check.issue(format!(
                    "{} assigns into constant {}",
                    Name::statement(statement),
                    Name::expression(target)
                ));
            }
        }
        StmtShape::Switch { .. } => {
            for arm in function.switch_arms(statement) {
                if arm.values.is_empty() {
                    check.issue(format!(
                        "{} has an arm with no case values",
                        Name::statement(statement)
                    ));
                }
            }
        }
        StmtShape::Try { .. } => {
            for handler in function.handlers(statement) {
                if let Some(binding) = handler.binding
                    && binding.index() >= function.variable_count()
                {
                    check.dangle(format!(
                        "{} binds undeclared variable {}",
                        Name::statement(statement),
                        Name::variable(binding)
                    ));
                }
            }
        }
        _ => {}
    }
}

/// An enclosing `Labeled` statement.
struct EnclosingLabel<'a> {
    name: &'a str,
    /// Whether the label names the loop that it wraps.
    names_loop: bool,
}

/// The enclosing constructs of the statement under the walk.
struct Context<'a> {
    loop_depth: usize,
    breakable_depth: usize,
    labels: Vec<EnclosingLabel<'a>>,
    /// How many `Labeled` statements define each name. Collected before the
    /// walk so a forward `goto` finds its label.
    defined_labels: BTreeMap<&'a str, u32>,
}

impl Context<'_> {
    fn encloses(&self, label: &str) -> bool {
        self.labels.iter().any(|enclosing| enclosing.name == label)
    }

    fn encloses_loop(&self, label: &str) -> bool {
        self.labels
            .iter()
            .any(|enclosing| enclosing.name == label && enclosing.names_loop)
    }
}

fn verify_context<H, F>(function: &F, check: &mut Check)
where
    H: VerifyDialect,
    F: FunctionView<H>,
{
    let mut context = Context {
        loop_depth: 0,
        breakable_depth: 0,
        labels: Vec::new(),
        defined_labels: BTreeMap::new(),
    };
    visit_statements(function, function.body(), &mut |statement| {
        if let StmtShape::Labeled { label, .. } = function.stmt(statement) {
            *context.defined_labels.entry(label).or_insert(0) += 1;
        }
    });
    for (label, count) in &context.defined_labels {
        if *count > 1 {
            check.issue(format!("label {label} is defined {count} times"));
        }
    }
    walk_context(function, function.body(), &mut context, check);
}

fn walk_context<'a, H, F>(
    function: &'a F,
    roots: &'a [F::Stmt],
    context: &mut Context<'a>,
    check: &mut Check,
) where
    H: VerifyDialect,
    F: FunctionView<H>,
{
    for &statement in roots {
        let name = Name::statement(statement);
        match function.stmt(statement) {
            StmtShape::Break { label: None } if context.breakable_depth == 0 => {
                check.issue(Untargeted::Break.message(statement));
            }
            StmtShape::Break { label: Some(label) } if !context.encloses(label) => {
                check.issue(Untargeted::BreakUnenclosed(label).message(statement));
            }
            StmtShape::Continue { label: None } if context.loop_depth == 0 => {
                check.issue(Untargeted::Continue.message(statement));
            }
            StmtShape::Continue { label: Some(label) } if !context.encloses_loop(label) => {
                let cause = if context.encloses(label) {
                    Untargeted::ContinueBlock(label)
                } else {
                    Untargeted::ContinueUnenclosed(label)
                };
                check.issue(cause.message(statement));
            }
            StmtShape::Goto { label } if !context.defined_labels.contains_key(label) => {
                check.issue(format!("{name} targets undefined label {label}"));
            }
            StmtShape::While { .. }
            | StmtShape::DoWhile { .. }
            | StmtShape::Loop { .. }
            | StmtShape::For { .. } => {
                context.loop_depth += 1;
                context.breakable_depth += 1;
                descend(function, statement, context, check);
                context.loop_depth -= 1;
                context.breakable_depth -= 1;
                continue;
            }
            StmtShape::Switch { .. } => {
                context.breakable_depth += 1;
                descend(function, statement, context, check);
                context.breakable_depth -= 1;
                continue;
            }
            StmtShape::Labeled { label, body } => {
                context.labels.push(EnclosingLabel {
                    name: label,
                    names_loop: labeled_loop(function, body).is_some(),
                });
                descend(function, statement, context, check);
                context.labels.pop();
                continue;
            }
            _ => {}
        }
        descend(function, statement, context, check);
    }
}

fn descend<'a, H, F>(
    function: &'a F,
    statement: F::Stmt,
    context: &mut Context<'a>,
    check: &mut Check,
) where
    H: VerifyDialect,
    F: FunctionView<H>,
{
    for_each_child_body(function, statement, |body, _| {
        walk_context(function, body, context, check);
    });
}
