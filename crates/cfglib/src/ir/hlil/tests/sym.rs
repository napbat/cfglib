//! The `Sym` toy dialect: symbolic operations and string constants.
//!
//! An operation is its own mnemonic, constants are owned strings, the type of
//! every value is `()`, and the metadata of a variable is a pair. Nothing
//! here is `Copy` except the marker, and no vocabulary is an enum.

extern crate alloc;

use alloc::string::{String, ToString};

use crate::ir::hlil::Dialect;

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
            Prim::Not => "not",
        }
    }
}
