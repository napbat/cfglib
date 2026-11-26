//! Runtime-neutral exception annotations for caller-owned edge payloads.
//!
//! Frame-scoped exception constructs (CLR clauses, Windows SEH `__try`
//! scopes, C++ and source `try` blocks) lower directly into
//! [`Region`](crate::Region) and [`Handler`](crate::Handler): an SEH
//! `__except` is a [`HandlerKind::Filter`](crate::HandlerKind::Filter), and a
//! CLR catch-type token goes in a [`HandlerMetadata`](crate::HandlerMetadata)
//! side table ([`HandlerMetadata::set_all`](crate::HandlerMetadata::set_all)).
//! Runtime registration state (an SEH frame chain, process-wide vectored
//! handlers) is not a lexical CFG region and stays with the consumer.
//! Platform-specific records stay consumer-owned generic payloads.

mod flow;

pub use flow::{ExceptionDisposition, ExceptionFlow, ExceptionPhase};
