//! The `Imp` toy dialect: closed enums and machine integers.
//!
//! Operations are a closed enum, constants are `i64`, types are a two-case
//! enum, and the metadata of a variable is a bare number.

use crate::ir::hlil::Dialect;

use super::kit::{Kit, Prim, Ty};

/// The marker of the `Imp` dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Imp;

/// The value types of `Imp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ImpType {
    Integer,
    Boolean,
}

/// The operations of `Imp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImpOp {
    Add,
    Not,
}

impl Dialect for Imp {
    type Op = ImpOp;
    type Constant = i64;
    type Type = ImpType;
    type Variable = u8;
}

impl Kit for Imp {
    fn ty(ty: Ty) -> ImpType {
        match ty {
            Ty::Integer => ImpType::Integer,
            Ty::Boolean => ImpType::Boolean,
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
            Prim::Not => ImpOp::Not,
        }
    }
}
