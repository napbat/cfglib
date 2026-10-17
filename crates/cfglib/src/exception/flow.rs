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
/// The payload follows the edge's [`EdgeKind`](crate::EdgeKind): an edge whose
/// kind [`is_exceptional`](crate::EdgeKind::is_exceptional) carries
/// [`Exceptional`](Self::Exceptional), and every other edge — including
/// [`ExceptionLeave`](crate::EdgeKind::ExceptionLeave), a normal transfer out
/// of a protected region — carries [`Normal`](Self::Normal).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExceptionFlow<M = ()> {
    /// An ordinary, non-exceptional control-flow edge.
    Normal {
        /// Consumer-owned metadata.
        metadata: M,
    },
    /// An exceptional control-flow edge.
    Exceptional {
        /// Search or unwind phase.
        phase: ExceptionPhase,
        /// Filter/handler outcome, when this edge represents a decision.
        disposition: Option<ExceptionDisposition>,
        /// Consumer-owned platform metadata.
        metadata: M,
    },
}

impl<M> ExceptionFlow<M> {
    /// Construct ordinary flow carrying `metadata`.
    pub const fn normal(metadata: M) -> Self {
        Self::Normal { metadata }
    }

    /// Construct exceptional flow carrying phase, disposition, and metadata.
    pub const fn exceptional(
        phase: ExceptionPhase,
        disposition: Option<ExceptionDisposition>,
        metadata: M,
    ) -> Self {
        Self::Exceptional {
            phase,
            disposition,
            metadata,
        }
    }

    /// The exception phase, or `None` for ordinary flow.
    #[must_use]
    pub const fn phase(&self) -> Option<ExceptionPhase> {
        match self {
            Self::Normal { .. } => None,
            Self::Exceptional { phase, .. } => Some(*phase),
        }
    }

    /// The filter/handler disposition carried by this edge, if any.
    #[must_use]
    pub const fn disposition(&self) -> Option<ExceptionDisposition> {
        match self {
            Self::Normal { .. } => None,
            Self::Exceptional { disposition, .. } => *disposition,
        }
    }

    /// Access the consumer-owned metadata.
    #[must_use]
    pub const fn metadata(&self) -> &M {
        match self {
            Self::Normal { metadata } | Self::Exceptional { metadata, .. } => metadata,
        }
    }

    /// Mutably access the consumer-owned metadata.
    pub const fn metadata_mut(&mut self) -> &mut M {
        match self {
            Self::Normal { metadata } | Self::Exceptional { metadata, .. } => metadata,
        }
    }

    /// Consume the annotation and return its consumer-owned metadata.
    pub fn into_metadata(self) -> M {
        match self {
            Self::Normal { metadata } | Self::Exceptional { metadata, .. } => metadata,
        }
    }
}

impl<M: Default> Default for ExceptionFlow<M> {
    fn default() -> Self {
        Self::normal(M::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exceptional_flow_retains_phase_disposition_and_platform_data() {
        let flow = ExceptionFlow::exceptional(
            ExceptionPhase::Search,
            Some(ExceptionDisposition::ContinueExecution),
            0xC000_0005_u32,
        );

        assert_eq!(flow.phase(), Some(ExceptionPhase::Search));
        assert_eq!(
            flow.disposition(),
            Some(ExceptionDisposition::ContinueExecution)
        );
        assert_eq!(flow.metadata(), &0xC000_0005);
    }
}
