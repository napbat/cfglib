//! The read-only view of one structured function.

use crate::graph::view::DenseId;

use super::dialect::Dialect;
use super::shape::{ExprShape, Handler, StmtShape, SwitchArm};

/// One entity of a structured function, named by a handle.
///
/// The three type parameters are the statement, expression, and variable
/// handles of one storage. A table that relates two functions, such as a
/// flat function and the structured function lifted from it, keys its rows
/// with this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Entity<S, E, V> {
    /// A structured statement.
    Statement(S),
    /// An expression-tree node.
    Expression(E),
    /// A declared variable.
    Variable(V),
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
///   rewrite builds a new function through a
///   [`FunctionBuilder`](super::FunctionBuilder) instead of editing this one.
/// - A handle that a view yields is below its count. The methods of the view
///   panic or return unspecified data for a handle outside the function, as an
///   index into a pool does. A reader that does not trust its input checks
///   each handle against the counts before it reads it.
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
