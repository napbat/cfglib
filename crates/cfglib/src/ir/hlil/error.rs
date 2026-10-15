//! The failures of the algorithms that drive a builder.
//!
//! Each error is generic over the error of the consumer's builder, which the
//! library cannot name. The consumer's [`finish`](super::FunctionBuilder::finish)
//! reports contract violations, and the algorithm wraps them unchanged.

extern crate alloc;

use alloc::string::String;
use core::fmt;

/// Why lifting a flat function into HLIL failed.
#[derive(Debug)]
pub enum LiftError<E> {
    /// The flat function has a shape that no HLIL form translates: a
    /// multi-result instruction, a conditional block whose terminator the
    /// instruction did not classify as a branch, or a switch case edge
    /// without values.
    Unsupported(String),
    /// The consumer's builder refused to seal the function.
    Builder(E),
}

impl<E: fmt::Display> fmt::Display for LiftError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(message) => write!(formatter, "unsupported HLIL lift: {message}"),
            Self::Builder(error) => write!(formatter, "HLIL builder failed: {error}"),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for LiftError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Unsupported(_) => None,
            Self::Builder(error) => Some(error),
        }
    }
}

/// Why lowering a structured function into a flat one failed.
#[derive(Debug)]
pub enum LowerError<E> {
    /// The structured function has a shape that the flat form cannot take:
    /// control that falls off the end of the function, or a construct that the
    /// builder has no flat protocol for.
    Unsupported(String),
    /// The consumer's builder refused an instruction or the function.
    Builder(E),
}

impl<E: fmt::Display> fmt::Display for LowerError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(message) => {
                write!(formatter, "unsupported HLIL lowering: {message}")
            }
            Self::Builder(error) => write!(formatter, "flat builder failed: {error}"),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for LowerError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Unsupported(_) => None,
            Self::Builder(error) => Some(error),
        }
    }
}

/// Why structure recovery failed.
#[derive(Debug)]
pub enum RecoverError<E> {
    /// The source function names a statement, expression, or variable that it
    /// does not have. [`verify`](fn@super::verify) reports such a function.
    Malformed(String),
    /// The consumer's builder refused to seal the function.
    Builder(E),
}

impl<E: fmt::Display> fmt::Display for RecoverError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(message) => write!(formatter, "malformed HLIL: {message}"),
            Self::Builder(error) => write!(formatter, "HLIL builder failed: {error}"),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for RecoverError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Malformed(_) => None,
            Self::Builder(error) => Some(error),
        }
    }
}
