//! The read-only view of one structured function, and the walks over it.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::graph::view::DenseId;

use super::dialect::Dialect;
use super::shape::{ExprShape, Handler, HandlerKind, StmtShape, SwitchArm};

/// One entity of a structured function, named by a handle.
///
/// The three type parameters are the statement, expression, and variable
/// handles of one storage. A table that relates two functions, such as the
/// flat function and the lifted one, keys its rows with this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Entity<S, E, V> {
    /// A structured statement.
    Statement(S),
    /// An expression-tree node.
    Expression(E),
    /// A declared variable.
    Variable(V),
}

/// Whether a child statement sequence is a lexical body or belongs to the
/// syntax of its parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildBodyKind {
    /// A body that opens its own declaration scope, such as the arm of an
    /// `if` or the body of a loop.
    Lexical,
    /// A clause of the parent's own syntax, such as the initializer or the
    /// update of a `for`. It declares into the scope that holds the parent.
    Clause,
}

/// A read-only structured function: trees of statements and expressions over
/// declared variables.
///
/// A consumer implements this on its dense storage and keeps the statements,
/// expressions, operand lists, and bodies in pools that the view lends back
/// as slices. Every handle is dense within the storage.
///
/// # Contract
///
/// - Statements and expressions form strict forests. Every statement is
///   listed by exactly one parent body or by [`body`](Self::body), and every
///   expression is used by exactly one parent: a statement operand or an
///   enclosing operation.
/// - A child has a lower index than its parent. The operands of an operation
///   precede the operation, and the statements of a body precede the
///   compound statement that lists the body. The order among the siblings of
///   one body is free.
/// - Every index below a `*_count` is live. The view has no removed slots: a
///   rewrite builds a new function instead of editing this one.
/// - A handle that a view yields is below its count. The methods of the view
///   panic or return unspecified data for a handle outside the function, as an
///   index into a pool does; [`verify`](fn@super::verify) is the check that a
///   function meets this contract.
pub trait FunctionView<D: Dialect> {
    /// A statement, dense within the consumer's storage.
    type Stmt: DenseId;
    /// An expression, dense within the consumer's storage.
    type Expr: DenseId;
    /// A variable, dense within the consumer's storage.
    type Var: DenseId;

    /// Returns the number of statements. Every statement index is below it.
    fn stmt_count(&self) -> usize;

    /// Returns the number of expressions. Every expression index is below it.
    fn expr_count(&self) -> usize;

    /// Returns the number of declared variables. Every variable index is
    /// below it.
    fn variable_count(&self) -> usize;

    /// Returns the top-level statement sequence.
    fn body(&self) -> &[Self::Stmt];

    /// Returns the shape of `stmt`.
    fn stmt(&self, stmt: Self::Stmt) -> StmtShape<'_, D, Self::Stmt, Self::Expr>;

    /// Iterates over the arms of the switch `stmt` in order. Yields nothing
    /// for any other statement.
    fn switch_arms(
        &self,
        stmt: Self::Stmt,
    ) -> impl ExactSizeIterator<Item = SwitchArm<'_, D, Self::Stmt>> + '_;

    /// Iterates over the handlers of the try `stmt` in dispatch order. Yields
    /// nothing for any other statement.
    fn handlers(
        &self,
        stmt: Self::Stmt,
    ) -> impl ExactSizeIterator<Item = Handler<'_, D, Self::Stmt, Self::Var>> + '_;

    /// Returns the shape of `expr`.
    fn expr(&self, expr: Self::Expr) -> ExprShape<'_, D, Self::Expr, Self::Var>;

    /// Returns the type of the value that the occurrence `expr` produces.
    fn ty(&self, expr: Self::Expr) -> &D::Type;

    /// Returns the metadata of `variable`.
    fn variable(&self, variable: Self::Var) -> &D::Variable;

    /// Returns the declared type of `variable`, the presentation-level
    /// declaration when one is known. The type of an occurrence
    /// ([`ty`](Self::ty)) stays authoritative.
    fn declared_type(&self, variable: Self::Var) -> Option<&D::Type>;

    /// Returns the parameters in declaration order.
    fn parameters(&self) -> &[Self::Var];

    /// Returns the return types in result order. Empty for no return value.
    fn returns(&self) -> &[D::Type];
}

/// Visits every child statement body of `stmt` in execution order, with its
/// declaration-scope role.
///
/// A leaf statement has no body. An `if` has its then and else bodies. A `for`
/// has its initializer, body, and update. A `switch` has every arm and then
/// the default. A `try` has its body, then each handler's filter (when it has
/// one) and body, then the finally body. Code that walks or rewrites the
/// statement tree uses this function instead of matching every shape, so a
/// new shape reaches it in one place.
pub fn for_each_child_body<'a, D, F>(
    view: &'a F,
    stmt: F::Stmt,
    mut visit: impl FnMut(&'a [F::Stmt], ChildBodyKind),
) where
    D: Dialect,
    F: FunctionView<D>,
{
    match view.stmt(stmt) {
        StmtShape::Expression(_)
        | StmtShape::Assign { .. }
        | StmtShape::Break { .. }
        | StmtShape::Continue { .. }
        | StmtShape::Return { .. }
        | StmtShape::Goto { .. } => {}
        StmtShape::If {
            then_body,
            else_body,
            ..
        } => {
            visit(then_body, ChildBodyKind::Lexical);
            visit(else_body, ChildBodyKind::Lexical);
        }
        StmtShape::While { body, .. }
        | StmtShape::DoWhile { body, .. }
        | StmtShape::Loop { body }
        | StmtShape::Labeled { body, .. }
        | StmtShape::Region { body, .. } => visit(body, ChildBodyKind::Lexical),
        StmtShape::For {
            initializer,
            update,
            body,
            ..
        } => {
            visit(initializer, ChildBodyKind::Clause);
            visit(body, ChildBodyKind::Lexical);
            visit(update, ChildBodyKind::Clause);
        }
        StmtShape::Switch { default_body, .. } => {
            for arm in view.switch_arms(stmt) {
                visit(arm.body, ChildBodyKind::Lexical);
            }
            visit(default_body, ChildBodyKind::Lexical);
        }
        StmtShape::Try {
            body, finally_body, ..
        } => {
            visit(body, ChildBodyKind::Lexical);
            for handler in view.handlers(stmt) {
                if let HandlerKind::Filter { filter_body } = handler.kind {
                    visit(filter_body, ChildBodyKind::Lexical);
                }
                visit(handler.body, ChildBodyKind::Lexical);
            }
            visit(finally_body, ChildBodyKind::Lexical);
        }
    }
}

/// Visits every expression that `stmt` references directly: its condition,
/// scrutinee, assignment sides, return values, and region operands.
///
/// Expressions inside child bodies are not visited, and neither are the
/// operands nested inside a visited expression.
pub fn for_each_statement_expression<D, F>(view: &F, stmt: F::Stmt, mut visit: impl FnMut(F::Expr))
where
    D: Dialect,
    F: FunctionView<D>,
{
    match view.stmt(stmt) {
        StmtShape::Expression(expression) => visit(expression),
        StmtShape::Assign { target, value } => {
            visit(target);
            visit(value);
        }
        StmtShape::If { condition, .. }
        | StmtShape::While { condition, .. }
        | StmtShape::DoWhile { condition, .. }
        | StmtShape::Switch {
            scrutinee: condition,
            ..
        } => visit(condition),
        StmtShape::For { condition, .. } => {
            if let Some(condition) = condition {
                visit(condition);
            }
        }
        StmtShape::Return { values: operands } | StmtShape::Region { operands, .. } => {
            for &operand in operands {
                visit(operand);
            }
        }
        StmtShape::Loop { .. }
        | StmtShape::Break { .. }
        | StmtShape::Continue { .. }
        | StmtShape::Labeled { .. }
        | StmtShape::Goto { .. }
        | StmtShape::Try { .. } => {}
    }
}

/// Visits every statement under `roots` in preorder: a statement, then the
/// statements of its child bodies in [`for_each_child_body`] order.
pub fn visit_statements<'a, D, F>(
    view: &'a F,
    roots: &'a [F::Stmt],
    visit: &mut impl FnMut(F::Stmt),
) where
    D: Dialect,
    F: FunctionView<D>,
{
    let mut pending = vec![roots.iter()];
    while let Some(bodies) = pending.last_mut() {
        let Some(&stmt) = bodies.next() else {
            pending.pop();
            continue;
        };
        visit(stmt);
        let first_child = pending.len();
        for_each_child_body(view, stmt, |body, _| pending.push(body.iter()));
        // The first child body is the next one to run, so it goes on top.
        pending[first_child..].reverse();
    }
}

/// Visits the expression tree under `root` in postorder: the operands of an
/// operation before the operation.
///
/// A function that meets the [`FunctionView`] contract stores operands before
/// their parent, so this order is also a valid evaluation order.
pub fn visit_expression_tree<D, F>(view: &F, root: F::Expr, visit: &mut impl FnMut(F::Expr))
where
    D: Dialect,
    F: FunctionView<D>,
{
    // Each entry is an expression and the position of its next operand.
    let mut pending: Vec<(F::Expr, usize)> = vec![(root, 0)];
    while let Some((expr, next)) = pending.last_mut() {
        let expr = *expr;
        if let ExprShape::Operation { operands, .. } = view.expr(expr)
            && let Some(&operand) = operands.get(*next)
        {
            *next += 1;
            pending.push((operand, 0));
        } else {
            visit(expr);
            pending.pop();
        }
    }
}

/// Returns whether two expression trees are structurally identical: the same
/// variable reads, equal constants, and equal operations over pairwise
/// identical operands.
///
/// The types of the occurrences are not compared.
#[must_use]
pub fn expressions_equal<D, F>(view: &F, left: F::Expr, right: F::Expr) -> bool
where
    D: Dialect,
    F: FunctionView<D>,
{
    let mut pending = vec![(left, right)];
    while let Some((left, right)) = pending.pop() {
        match (view.expr(left), view.expr(right)) {
            (ExprShape::Variable(left), ExprShape::Variable(right)) if left == right => {}
            (ExprShape::Constant(left), ExprShape::Constant(right)) if left == right => {}
            (
                ExprShape::Operation {
                    op: left_op,
                    operands: left_operands,
                },
                ExprShape::Operation {
                    op: right_op,
                    operands: right_operands,
                },
            ) if left_op == right_op && left_operands.len() == right_operands.len() => {
                pending.extend(
                    left_operands
                        .iter()
                        .copied()
                        .zip(right_operands.iter().copied()),
                );
            }
            _ => return false,
        }
    }
    true
}

/// Returns the compound form of the assignment `stmt`, when it has one: the
/// operation and the second operand of `target = op(target, operand)`.
///
/// An assignment qualifies when its value is a binary operation whose first
/// operand is structurally identical to the target. A renderer spells the
/// result `target op= operand` with the dialect's compound spelling. Whether a
/// commutative operation with the target second also qualifies stays the
/// caller's judgment. Deciding by structure, never by comparing rendered text,
/// keeps the answer immune to parenthesization and casts.
#[must_use]
pub fn compound_assignment<D, F>(view: &F, stmt: F::Stmt) -> Option<(&D::Op, F::Expr)>
where
    D: Dialect,
    F: FunctionView<D>,
{
    let StmtShape::Assign { target, value } = view.stmt(stmt) else {
        return None;
    };
    let ExprShape::Operation { op, operands } = view.expr(value) else {
        return None;
    };
    let &[first, operand] = operands else {
        return None;
    };
    expressions_equal(view, first, target).then_some((op, operand))
}
