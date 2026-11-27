//! The one error type of the exception-region invariants.

use crate::block::BlockId;

use super::{Continuation, HandlerRef, RegionId};

/// A broken exception-region invariant.
///
/// The region mutators of [`Cfg`](crate::Cfg) return it for the reference
/// they were handed, and [`Cfg::check_regions`](crate::Cfg::check_regions)
/// for the first reference of the whole metadata that breaks one. Decoding a
/// serialized CFG and [`verify`](crate::verify) report the same errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionError {
    /// A handler entry, filter block, cleanup exit block, or continuation
    /// target names no live block of the graph. These are single references
    /// with no stand-in, so [`Cfg::compact`](crate::Cfg::compact) could not
    /// renumber them.
    BlockNotLive(BlockId),
    /// A region names a parent that is not a region of the graph.
    ParentMissing(RegionId),
    /// A cleanup record, or a mutator, names a handler that no region holds.
    HandlerMissing(HandlerRef),
    /// A cleanup record lists one route twice.
    /// [`Cleanup::continuations`](super::Cleanup::continuations) holds each
    /// route once, so the second copy would be reported as a second path.
    ContinuationRepeated(HandlerRef, Continuation),
}

impl core::fmt::Display for RegionError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BlockNotLive(block) => write!(
                formatter,
                "exception metadata names {block}, which is not a live block of the graph"
            ),
            Self::ParentMissing(parent) => write!(
                formatter,
                "a region names parent {parent}, which does not exist"
            ),
            Self::HandlerMissing(handler) => write!(
                formatter,
                "cleanup data names handler {handler}, which no region holds"
            ),
            Self::ContinuationRepeated(handler, continuation) => write!(
                formatter,
                "the cleanup of handler {handler} lists the route {:?} to {} twice",
                continuation.reason, continuation.target
            ),
        }
    }
}

impl core::error::Error for RegionError {}
