//! The conventional diagnostic names of handles: `s3`, `e7`, `v0`.

use core::fmt;

use crate::graph::view::DenseId;

/// A handle rendered as one letter and its dense index.
///
/// Verification messages and the pseudocode writer share these names, so a
/// message about `e7` points at the text that a reader sees.
#[derive(Debug, Clone, Copy)]
pub(super) struct Name {
    prefix: char,
    index: usize,
}

impl Name {
    /// The name of a statement handle.
    pub(super) fn statement(statement: impl DenseId) -> Self {
        Self {
            prefix: 's',
            index: statement.index(),
        }
    }

    /// The name of an expression handle.
    pub(super) fn expression(expression: impl DenseId) -> Self {
        Self {
            prefix: 'e',
            index: expression.index(),
        }
    }

    /// The name of a variable handle.
    pub(super) fn variable(variable: impl DenseId) -> Self {
        Self {
            prefix: 'v',
            index: variable.index(),
        }
    }
}

impl fmt::Display for Name {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}{}", self.prefix, self.index)
    }
}
