//! Exception dispatch annotations suitable for caller-owned CFG edge payloads.

/// Runtime phase in which an exceptional transfer occurs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExceptionPhase {
    /// Handler/filter search before a handler has been selected.
    Search,
    /// Stack cleanup and transfer after a handler has been selected.
    Unwind,
}

/// Result of a CLR, SEH, or VEH exception filter/handler decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExceptionDisposition {
    /// Select this handler and transfer control to it.
    ExecuteHandler,
    /// Decline the exception and continue searching outward.
    ContinueSearch,
    /// Restore machine state and continue at the exceptional instruction.
    ContinueExecution,
}

/// Optional exception semantics plus consumer-owned platform metadata.
///
/// This type is designed to be used directly as `Cfg<I, ExceptionFlow<M>>`'s
/// edge payload. `M` can retain a CLR clause token, an SEH scope-table record,
/// an exception/context identifier, or source provenance without making those
/// platform types part of cfglib.
///
/// Whether an edge is exceptional is its [`EdgeKind`](crate::EdgeKind)'s to
/// say ([`is_exceptional`](crate::EdgeKind::is_exceptional)); the payload only
/// adds what the kind does not carry. An ordinary edge, and an
/// [`ExceptionLeave`](crate::EdgeKind::ExceptionLeave) — a normal transfer
/// out of a protected region — carry neither a phase nor a disposition.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExceptionFlow<M = ()> {
    /// Search or unwind phase of an exceptional transfer, when the frontend
    /// knows it.
    pub phase: Option<ExceptionPhase>,
    /// Filter/handler outcome, when this edge represents a decision.
    pub disposition: Option<ExceptionDisposition>,
    /// Consumer-owned platform metadata.
    pub metadata: M,
}

impl<M> ExceptionFlow<M> {
    /// An annotation carrying only `metadata`: no phase and no disposition.
    pub const fn new(metadata: M) -> Self {
        Self {
            phase: None,
            disposition: None,
            metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_annotation_retains_phase_disposition_and_platform_data() {
        let flow = ExceptionFlow {
            phase: Some(ExceptionPhase::Search),
            disposition: Some(ExceptionDisposition::ContinueExecution),
            metadata: 0xC000_0005_u32,
        };

        assert_eq!(flow.phase, Some(ExceptionPhase::Search));
        assert_eq!(
            flow.disposition,
            Some(ExceptionDisposition::ContinueExecution)
        );
        assert_eq!(flow.metadata, 0xC000_0005);
        assert_eq!(
            ExceptionFlow::new(7_u32),
            ExceptionFlow {
                phase: None,
                disposition: None,
                metadata: 7,
            }
        );
        assert_eq!(ExceptionFlow::<u32>::default(), ExceptionFlow::new(0));
    }
}
