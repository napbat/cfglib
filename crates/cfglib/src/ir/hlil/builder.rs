//! The write-only protocol through which an algorithm builds a structured
//! function.

use crate::graph::view::DenseId;

use super::dialect::Dialect;
use super::shape::{Handler, SwitchArm};
use super::view::FunctionView;

/// Builds a structured function bottom-up in consumer-owned storage.
///
/// An algorithm such as [`recover_structure`](super::recover_structure) calls
/// the methods in construction order and ends with [`finish`](Self::finish).
/// Each call appends one variable, expression, or statement and returns its
/// handle. The protocol offers no way to read a node back or to change one. An
/// algorithm that needs to inspect or simplify what it is about to emit does
/// so in its own scratch space, or reads its input function, before it calls
/// the builder.
///
/// # Contract
///
/// - Every operand, variable, and child statement is a handle that this
///   builder returned earlier, and each expression and statement handle is
///   used at most once. Handles are issued in call order, so a child always
///   has a lower index than its parent, as [`FunctionView`] requires.
/// - The target of [`assign`](Self::assign) is a variable read or an
///   [`apply`](Self::apply), never a [`constant`](Self::constant).
/// - A [`SwitchArm`] never carries an empty `values` slice. The default arm is
///   the `default_body` argument of [`switch`](Self::switch).
/// - [`declare_variable`](Self::declare_variable) comes before the first call
///   that names the variable it returns.
///
/// The consumer checks the contract when the function is sealed and reports a
/// violation through [`Self::Error`]. Calling [`verify`](fn@super::verify) in
/// [`finish`](Self::finish) checks all of it.
pub trait FunctionBuilder<D: Dialect> {
    /// A statement handle, dense within the function being built.
    type Stmt: DenseId;
    /// An expression handle, dense within the function being built.
    type Expr: DenseId;
    /// A variable handle, dense within the function being built.
    type Var: DenseId;
    /// The sealed function. It reads through the handles of this builder.
    type Function: FunctionView<D, Stmt = Self::Stmt, Expr = Self::Expr, Var = Self::Var>;
    /// Why the consumer refused to seal the function.
    type Error;

    /// Declares one variable and returns its handle. Handles are issued in
    /// call order.
    fn declare_variable(
        &mut self,
        variable: &D::Variable,
        declared_type: Option<&D::Type>,
    ) -> Self::Var;

    /// Declares the ordered signature: the parameters and the return types.
    fn set_signature(&mut self, parameters: &[Self::Var], returns: &[D::Type]);

    /// Adds a read occurrence of `variable` whose value has type `ty`.
    fn read_variable(&mut self, variable: Self::Var, ty: &D::Type) -> Self::Expr;

    /// Adds a literal of type `ty`.
    fn constant(&mut self, constant: &D::Constant, ty: &D::Type) -> Self::Expr;

    /// Adds an operation over `operands`, in operation order, whose value has
    /// type `ty`.
    fn apply(&mut self, op: &D::Op, operands: &[Self::Expr], ty: &D::Type) -> Self::Expr;

    /// Adds a statement that evaluates `expression` for its effects.
    fn expression_statement(&mut self, expression: Self::Expr) -> Self::Stmt;

    /// Adds a store of `value` into the place `target`.
    fn assign(&mut self, target: Self::Expr, value: Self::Expr) -> Self::Stmt;

    /// Adds an `if`. `else_body` is empty for no `else`.
    fn if_(
        &mut self,
        condition: Self::Expr,
        then_body: &[Self::Stmt],
        else_body: &[Self::Stmt],
    ) -> Self::Stmt;

    /// Adds a pre-tested loop.
    fn while_(&mut self, condition: Self::Expr, body: &[Self::Stmt]) -> Self::Stmt;

    /// Adds a post-tested loop.
    fn do_while(&mut self, body: &[Self::Stmt], condition: Self::Expr) -> Self::Stmt;

    /// Adds an endless loop.
    fn loop_(&mut self, body: &[Self::Stmt]) -> Self::Stmt;

    /// Adds a counted-style loop. `condition` is `None` for an unconditional
    /// loop.
    fn for_(
        &mut self,
        initializer: &[Self::Stmt],
        condition: Option<Self::Expr>,
        update: &[Self::Stmt],
        body: &[Self::Stmt],
    ) -> Self::Stmt;

    /// Adds a switch. `default_body` is empty when the default transfers
    /// straight to the continuation.
    fn switch(
        &mut self,
        scrutinee: Self::Expr,
        arms: &[SwitchArm<'_, D, Self::Stmt>],
        default_body: &[Self::Stmt],
    ) -> Self::Stmt;

    /// Adds a `break` of the innermost loop or switch, or of the labeled
    /// statement that `label` names.
    fn break_(&mut self, label: Option<&str>) -> Self::Stmt;

    /// Adds a `continue` of the innermost loop, or of the labeled loop that
    /// `label` names.
    fn continue_(&mut self, label: Option<&str>) -> Self::Stmt;

    /// Adds a return of `values`.
    fn return_(&mut self, values: &[Self::Expr]) -> Self::Stmt;

    /// Adds a labeled statement group.
    fn labeled(&mut self, label: &str, body: &[Self::Stmt]) -> Self::Stmt;

    /// Adds a transfer to the labeled statement that `label` names.
    fn goto(&mut self, label: &str) -> Self::Stmt;

    /// Adds a protected region. `finally_body` is empty for no `finally`.
    fn try_(
        &mut self,
        body: &[Self::Stmt],
        handlers: &[Handler<'_, D, Self::Stmt, Self::Var>],
        finally_body: &[Self::Stmt],
    ) -> Self::Stmt;

    /// Adds a dialect statement that wraps `body`.
    fn region(&mut self, op: &D::Op, operands: &[Self::Expr], body: &[Self::Stmt]) -> Self::Stmt;

    /// Roots the statement forest at `body` and seals the function.
    ///
    /// # Errors
    ///
    /// Returns the consumer's reason when the function breaks the contract.
    fn finish(self, body: &[Self::Stmt]) -> Result<Self::Function, Self::Error>;
}
