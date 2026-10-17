//! The wording of a `break` or `continue` that has no target.
//!
//! [`verify`](fn@super::verify) finds these statements in a function, and
//! [`lower`](fn@super::lower) meets them in a function that nobody verified.
//! Both name the failure through [`Untargeted`], so the same statement is
//! rejected in the same words.

extern crate alloc;

use alloc::format;
use alloc::string::String;

use crate::graph::view::DenseId;

use super::name::Name;

/// Why a `break` or `continue` has no enclosing target.
#[derive(Debug, Clone, Copy)]
pub(super) enum Untargeted<'a> {
    /// An unlabeled `break` outside every loop and switch.
    Break,
    /// An unlabeled `continue` outside every loop.
    Continue,
    /// A labeled `break` whose label encloses nothing.
    BreakUnenclosed(&'a str),
    /// A labeled `continue` whose label encloses nothing.
    ContinueUnenclosed(&'a str),
    /// A labeled `continue` whose label encloses it but labels a block, not a
    /// loop.
    ContinueBlock(&'a str),
}

impl Untargeted<'_> {
    /// The message that names `stmt` and what is wrong with its transfer.
    pub(super) fn message(self, stmt: impl DenseId) -> String {
        let stmt = Name::statement(stmt);
        match self {
            Self::Break => format!("{stmt} breaks outside any loop or switch"),
            Self::Continue => format!("{stmt} continues outside any loop"),
            Self::BreakUnenclosed(label) => format!("{stmt} breaks unenclosing label {label}"),
            Self::ContinueUnenclosed(label) => {
                format!("{stmt} continues unenclosing label {label}")
            }
            Self::ContinueBlock(label) => {
                format!("{stmt} continues label {label}, which labels a block and not a loop")
            }
        }
    }
}
