//! The `Imp` toy dialect: closed enums and machine integers.
//!
//! Operations are a closed enum, constants are `i64`, types are a three-case
//! enum, and the metadata of a variable is a bare number. Its verification
//! hook checks the operand types of a comparison, which reads the operand
//! expressions through the view.

extern crate alloc;

use alloc::format;
use alloc::vec::Vec;

use crate::graph::view::DenseId;
use crate::ir::hlil::{
    Dialect, ExprShape, FunctionView, Pseudocode, VerificationIssue, VerifyDialect,
};

use super::kit::{Kit, Prim, Ty};

/// The marker of the `Imp` dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Imp;

/// The value types of `Imp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ImpType {
    Integer,
    Boolean,
    Void,
}

/// The operations of `Imp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImpOp {
    Add,
    LessThan,
    Not,
    Call,
    Acquire,
    Release,
}

impl Dialect for Imp {
    type Op = ImpOp;
    type Constant = i64;
    type Type = ImpType;
    type Variable = u8;
}

impl Pseudocode for Imp {
    fn mnemonic(op: &ImpOp) -> &str {
        match op {
            ImpOp::Add => "add",
            ImpOp::LessThan => "lt",
            ImpOp::Not => "not",
            ImpOp::Call => "call",
            ImpOp::Acquire => "acquire",
            ImpOp::Release => "release",
        }
    }
}

impl VerifyDialect for Imp {
    /// A comparison orders integers.
    fn verify<F: FunctionView<Self>>(function: &F, issues: &mut Vec<VerificationIssue>) {
        for index in 0..function.expr_count() {
            let expression = F::Expr::from_index(index);
            let ExprShape::Operation {
                op: ImpOp::LessThan,
                operands,
            } = function.expr(expression)
            else {
                continue;
            };
            for &operand in operands {
                if *function.ty(operand) != ImpType::Integer {
                    issues.push(VerificationIssue::new(format!(
                        "e{index} compares the non-integer operand e{}",
                        operand.index()
                    )));
                }
            }
        }
    }
}

impl Kit for Imp {
    fn ty(ty: Ty) -> ImpType {
        match ty {
            Ty::Integer => ImpType::Integer,
            Ty::Boolean => ImpType::Boolean,
            Ty::Void => ImpType::Void,
        }
    }

    fn constant(value: i64) -> i64 {
        value
    }

    fn variable(number: u8) -> u8 {
        number
    }

    fn op(prim: Prim) -> ImpOp {
        match prim {
            Prim::Add => ImpOp::Add,
            Prim::LessThan => ImpOp::LessThan,
            Prim::Not => ImpOp::Not,
            Prim::Call => ImpOp::Call,
            Prim::Acquire => ImpOp::Acquire,
            Prim::Release => ImpOp::Release,
        }
    }
}
