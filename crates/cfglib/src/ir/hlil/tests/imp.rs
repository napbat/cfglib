//! The `Imp` toy dialect: closed enums and machine integers.
//!
//! Operations are a closed enum, constants are `i64`, types are a three-case
//! enum, and the metadata of a variable is a bare number. Its verification
//! hook checks the operand types of a comparison, which reads the operand
//! expressions through the view. Its recovery hooks claim selection through
//! `select`, the `acquire` and `release` protocol, and a boolean target is
//! not a single-expression assignment.

extern crate alloc;

use alloc::format;
use alloc::vec::Vec;

use crate::graph::view::DenseId;
use crate::ir::hlil::{
    Dialect, ExprShape, FunctionView, Pseudocode, RecoverDialect, VerificationIssue, VerifyDialect,
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
    Below,
    AtLeast,
    Not,
    Load,
    Deref,
    Call,
    Select,
    Acquire,
    Release,
    Caught,
    Throw,
    Expanded,
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
            ImpOp::Below => "below",
            ImpOp::AtLeast => "at-least",
            ImpOp::Not => "not",
            ImpOp::Load => "load",
            ImpOp::Deref => "deref",
            ImpOp::Call => "call",
            ImpOp::Select => "select",
            ImpOp::Acquire => "acquire",
            ImpOp::Release => "release",
            ImpOp::Caught => "caught",
            ImpOp::Throw => "throw",
            ImpOp::Expanded => "expanded",
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

impl RecoverDialect for Imp {
    fn select() -> Option<ImpOp> {
        Some(ImpOp::Select)
    }

    fn single_expression_assignment(value_type: &ImpType) -> bool {
        *value_type != ImpType::Boolean
    }

    fn single_expression_operation(op: &ImpOp) -> bool {
        *op != ImpOp::Expanded
    }

    fn region_enter(op: &ImpOp) -> Option<ImpOp> {
        (*op == ImpOp::Acquire).then_some(ImpOp::Acquire)
    }

    fn releases(enter: &ImpOp, exit: &ImpOp) -> bool {
        *enter == ImpOp::Acquire && *exit == ImpOp::Release
    }

    fn is_exception_materialization(op: &ImpOp) -> bool {
        *op == ImpOp::Caught
    }

    fn is_throw(op: &ImpOp) -> bool {
        *op == ImpOp::Throw
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
            Prim::Below => ImpOp::Below,
            Prim::Not => ImpOp::Not,
            Prim::Load => ImpOp::Load,
            Prim::Deref => ImpOp::Deref,
            Prim::Call => ImpOp::Call,
            Prim::Acquire => ImpOp::Acquire,
            Prim::Release => ImpOp::Release,
            Prim::Caught => ImpOp::Caught,
            Prim::Throw => ImpOp::Throw,
            Prim::Expanded => ImpOp::Expanded,
        }
    }
}
