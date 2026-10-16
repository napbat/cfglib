//! The `Sym` toy dialect: symbolic operations and string constants.
//!
//! An operation is its own mnemonic, constants are owned strings, the type of
//! every value is `()`, and the metadata of a variable is a pair. Nothing
//! here is `Copy` except the marker, and no vocabulary is an enum. Its
//! verification hook rejects an empty constant, which it finds by reading
//! every expression through the view.

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use crate::graph::view::DenseId;
use crate::ir::hlil::{
    Dialect, ExprShape, FunctionView, Pseudocode, RecoverDialect, VerificationIssue, VerifyDialect,
};

use super::kit::{Kit, Prim, Ty};

/// The marker of the `Sym` dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Sym;

impl Dialect for Sym {
    type Op = &'static str;
    type Constant = String;
    type Type = ();
    type Variable = (u8, Option<u8>);
}

impl Pseudocode for Sym {
    fn mnemonic<'a>(op: &'a &'static str) -> &'a str {
        op
    }

    fn fmt_constant(formatter: &mut fmt::Formatter<'_>, constant: &String) -> fmt::Result {
        formatter.write_str(constant)
    }
}

impl VerifyDialect for Sym {
    /// A constant is never the empty string.
    fn verify<F: FunctionView<Self>>(function: &F, issues: &mut Vec<VerificationIssue>) {
        for index in 0..function.expr_count() {
            if let ExprShape::Constant(constant) = function.expr(F::Expr::from_index(index))
                && constant.is_empty()
            {
                issues.push(VerificationIssue::new(alloc::format!(
                    "e{index} is an empty constant"
                )));
            }
        }
    }
}

impl RecoverDialect for Sym {
    fn select() -> Option<&'static str> {
        Some("select")
    }

    fn single_expression_operation(op: &&'static str) -> bool {
        *op != "expanded"
    }

    fn region_enter(op: &&'static str) -> Option<&'static str> {
        (*op == "acquire").then_some("acquire")
    }

    fn releases(enter: &&'static str, exit: &&'static str) -> bool {
        *enter == "acquire" && *exit == "release"
    }

    fn is_exception_materialization(op: &&'static str) -> bool {
        *op == "caught"
    }

    fn is_throw(op: &&'static str) -> bool {
        *op == "throw"
    }
}

impl Kit for Sym {
    fn ty(_: Ty) {}

    fn constant(value: i64) -> String {
        value.to_string()
    }

    fn variable(number: u8) -> (u8, Option<u8>) {
        (number, number.checked_sub(1))
    }

    fn op(prim: Prim) -> &'static str {
        match prim {
            Prim::Add => "add",
            Prim::LessThan => "lt",
            Prim::Below => "below",
            Prim::Not => "not",
            Prim::Load => "load",
            Prim::Deref => "deref",
            Prim::Call => "call",
            Prim::Acquire => "acquire",
            Prim::Release => "release",
            Prim::Caught => "caught",
            Prim::Throw => "throw",
            Prim::Expanded => "expanded",
        }
    }
}
