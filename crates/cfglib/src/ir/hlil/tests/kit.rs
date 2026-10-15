//! What the shared test bodies need from a toy dialect.
//!
//! Each test body is written once, over [`Kit`], and runs over both toy
//! dialects. The dialects represent every vocabulary differently, so a body
//! that passes on both cannot depend on one representation.

use crate::ir::hlil::{Pseudocode, VerifyDialect};

use super::storage::{ExprId, StmtId, ToyHlilBuilder, VarId};

/// The operations that the shared test bodies use. A toy dialect maps each
/// to its own operation type, and both spell them alike in pseudocode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Prim {
    Add,
    LessThan,
    Not,
    Call,
    Acquire,
    Release,
}

/// The value types that the shared test bodies use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ty {
    Integer,
    Boolean,
    Void,
}

/// A toy dialect that the shared test bodies run over.
pub(crate) trait Kit: VerifyDialect + Pseudocode {
    /// The type that a test names by `ty`.
    fn ty(ty: Ty) -> Self::Type;

    /// The constant that stands for `value`.
    fn constant(value: i64) -> Self::Constant;

    /// The variable metadata that stands for variable number `number`.
    fn variable(number: u8) -> Self::Variable;

    /// The operation that stands for `prim`.
    fn op(prim: Prim) -> Self::Op;
}

/// Shorthand over the dialect-neutral constructors of [`Kit`]. Every
/// expression it builds is integer-typed unless the method says otherwise.
impl<D: Kit> ToyHlilBuilder<D> {
    /// Declares an integer variable.
    pub(crate) fn int_variable(&mut self, number: u8) -> VarId {
        self.declare_variable(&D::variable(number), Some(&D::ty(Ty::Integer)))
    }

    /// Reads `variable` as an integer.
    pub(crate) fn read(&mut self, variable: VarId) -> ExprId {
        self.read_variable(variable, &D::ty(Ty::Integer))
    }

    /// An integer literal.
    pub(crate) fn number(&mut self, value: i64) -> ExprId {
        self.constant(&D::constant(value), &D::ty(Ty::Integer))
    }

    /// Applies `prim` to `operands`, producing a value of type `ty`.
    pub(crate) fn call(&mut self, prim: Prim, operands: &[ExprId], ty: Ty) -> ExprId {
        self.apply(&D::op(prim), operands, &D::ty(ty))
    }

    /// `variable = value;`
    pub(crate) fn store(&mut self, variable: VarId, value: ExprId) -> StmtId {
        let target = self.read(variable);
        self.assign(target, value)
    }

    /// `prim(operands);` as an expression statement.
    pub(crate) fn effect(&mut self, prim: Prim, operands: &[ExprId]) -> StmtId {
        let call = self.call(prim, operands, Ty::Void);
        self.expression_statement(call)
    }

    /// An expression statement that reads `variable`.
    pub(crate) fn use_of(&mut self, variable: VarId) -> StmtId {
        let read = self.read(variable);
        self.expression_statement(read)
    }

    /// An expression statement that evaluates a constant.
    pub(crate) fn noop(&mut self) -> StmtId {
        let zero = self.number(0);
        self.expression_statement(zero)
    }

    /// A boolean constant condition.
    pub(crate) fn condition(&mut self) -> ExprId {
        self.constant(&D::constant(1), &D::ty(Ty::Boolean))
    }
}
