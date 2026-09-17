//! Edges connecting basic blocks in a control-flow graph.

use crate::graph::edge_view::EdgeRef;

pub use crate::graph::store::{EdgeId, EdgeTag};

/// The kind of a control-flow edge.
///
/// The vocabulary is universal, not machine-specific: every variant has both
/// a source-language and a machine reading (e.g. [`Jump`](Self::Jump) is a
/// `goto` or a `jmp`).
///
/// # Which kinds algorithms interpret
///
/// Dominance, loop detection, SCC, and traversals are purely structural —
/// they never read kinds, so consumers may choose kinds freely for their own
/// purposes. The kind-sensitive surfaces are: AST lifting (block
/// classification via [`Back`](Self::Back),
/// [`ConditionalTrue`](Self::ConditionalTrue) /
/// [`ConditionalFalse`](Self::ConditionalFalse),
/// [`SwitchCase`](Self::SwitchCase), [`Jump`](Self::Jump)), the `_tagged`
/// loop detectors ([`Back`](Self::Back)), linearization (fallthrough-like
/// kinds vs branches), switch recovery
/// ([`IndirectJump`](Self::IndirectJump) removal), the exception model
/// (the `Exception*` kinds), diffing (kind discriminants in fingerprints),
/// and DOT styling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EdgeKind {
    /// Sequential fallthrough to the next block.
    Fallthrough,
    /// Taken branch of a conditional (the "true" path).
    ConditionalTrue,
    /// Not-taken branch of a conditional (the "false" / merge path).
    ConditionalFalse,
    /// Unconditional jump (structured break/switch exit).
    Unconditional,
    /// Back-edge to a loop header.
    Back,
    /// Edge to a call target.
    Call,
    /// Return edge from a call site.
    CallReturn,
    /// Edge for a switch/case arm.
    SwitchCase,

    /// Direct explicit jump: a source `goto`, a machine `jmp` / `b`.
    ///
    /// Distinct from [`Unconditional`](Self::Unconditional): `Jump` records
    /// an explicit branch instruction, `Unconditional` a synthesized
    /// structured transfer (break, switch exit).
    Jump,
    /// Computed / indirect jump: a source computed goto or lowered `match`
    /// dispatch, a machine `jmp [rax]` through a jump table.
    IndirectJump,
    /// Indirect call: source dynamic dispatch or a function-pointer call, a
    /// machine `call [vtable]`.
    IndirectCall,
    /// Edge into an exception-handler entry block.
    ExceptionHandler,
    /// Edge from a potentially-throwing instruction to a handler.
    ExceptionUnwind,
    /// Edge from a protected region to the normal continuation.
    ExceptionLeave,
    /// Edge from a resume/rethrow point to the next exception dispatcher.
    ExceptionResume,
    /// Edge that resumes execution after an exception was handled in-place.
    ///
    /// Windows SEH and VEH use this for `EXCEPTION_CONTINUE_EXECUTION`: the
    /// target is the exact block at which execution resumes.
    ExceptionContinue,
}

impl EdgeKind {
    /// Whether the edge transfers control exceptionally rather than
    /// sequentially.
    ///
    /// [`ExceptionLeave`](Self::ExceptionLeave) is a normal transfer out
    /// of a protected region, so it stays sequential.
    #[must_use]
    pub const fn is_exceptional(self) -> bool {
        matches!(
            self,
            EdgeKind::ExceptionHandler
                | EdgeKind::ExceptionUnwind
                | EdgeKind::ExceptionResume
                | EdgeKind::ExceptionContinue
        )
    }
}

/// An edge payload that declares a control-flow [`EdgeKind`].
///
/// The kind-sensitive algorithms — back-edge detection honoring builder tags,
/// switch recovery, linearization — take this trait rather than a concrete
/// [`Cfg`](crate::Cfg), so a consumer whose own edge payload carries a kind
/// participates without a parallel implementation.
pub trait KindedEdge {
    /// The control-flow classification of this edge.
    fn kind(&self) -> EdgeKind;
}

impl EdgeKind {
    /// Every kind, in declaration order.
    ///
    /// This is what [`from_name`](Self::from_name) searches, so naming stays
    /// defined in exactly one place: [`name`](Self::name).
    pub const ALL: [Self; 16] = [
        Self::Fallthrough,
        Self::ConditionalTrue,
        Self::ConditionalFalse,
        Self::Unconditional,
        Self::Back,
        Self::Call,
        Self::CallReturn,
        Self::SwitchCase,
        Self::Jump,
        Self::IndirectJump,
        Self::IndirectCall,
        Self::ExceptionHandler,
        Self::ExceptionUnwind,
        Self::ExceptionLeave,
        Self::ExceptionResume,
        Self::ExceptionContinue,
    ];

    /// The canonical name of this kind, which is what
    /// [`Display`](core::fmt::Display) prints and what the text form of a
    /// control-flow graph writes.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fallthrough => "fallthrough",
            Self::ConditionalTrue => "true",
            Self::ConditionalFalse => "false",
            Self::Unconditional => "unconditional",
            Self::Back => "back",
            Self::Call => "call",
            Self::CallReturn => "call_return",
            Self::SwitchCase => "case",
            Self::Jump => "jump",
            Self::IndirectJump => "indirect_jump",
            Self::IndirectCall => "indirect_call",
            Self::ExceptionHandler => "handler",
            Self::ExceptionUnwind => "unwind",
            Self::ExceptionLeave => "leave",
            Self::ExceptionResume => "resume",
            Self::ExceptionContinue => "continue_exception",
        }
    }

    /// The kind with this canonical name, or `None` for an unknown name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }
}

impl core::fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::collections::BTreeSet;

    use super::EdgeKind;

    #[test]
    fn every_kind_has_a_distinct_name_that_reads_back() {
        let names: BTreeSet<_> = EdgeKind::ALL.iter().map(|kind| kind.name()).collect();
        assert_eq!(
            names.len(),
            EdgeKind::ALL.len(),
            "two kinds share a name, so one of them cannot be read back"
        );
        for kind in EdgeKind::ALL {
            assert_eq!(EdgeKind::from_name(kind.name()), Some(kind));
        }
        assert_eq!(EdgeKind::from_name("sideways"), None);
    }
}

/// A directed edge's payload: its classification, optional branch weight, and
/// consumer-defined metadata.
///
/// Identity and endpoints belong to the store, not to the payload;
/// [`Cfg::edge`](crate::Cfg::edge) hands out an [`EdgeRef`] that carries all
/// three together.
///
/// `E` is consumer-owned metadata. The default unit payload preserves the
/// compact `Cfg<I>` surface, while frontends that need switch labels, handler
/// identities, continuation tokens, or source provenance use `Cfg<I, E>`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Edge<E = ()> {
    /// Classification.
    pub(crate) kind: EdgeKind,
    /// Optional branch weight / probability (0.0–1.0).
    ///
    /// When set, this indicates the likelihood of this edge being taken
    /// relative to other outgoing edges of the same source block.
    /// Used by the linearizer for hot-path layout and by DOT output
    /// for visual emphasis.
    pub(crate) weight: Option<f64>,
    /// Consumer-defined metadata.
    pub(crate) payload: E,
}

impl<E: PartialEq> PartialEq for Edge<E> {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.weight.map(f64::to_bits) == other.weight.map(f64::to_bits)
            && self.payload == other.payload
    }
}

impl<E: Eq> Eq for Edge<E> {}

impl<E> Edge<E> {
    /// Create an edge payload.
    pub(crate) const fn new(kind: EdgeKind, weight: Option<f64>, payload: E) -> Self {
        Self {
            kind,
            weight,
            payload,
        }
    }

    /// The classification of this edge.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> EdgeKind {
        self.kind
    }

    /// Set the classification of this edge.
    #[inline]
    pub const fn set_kind(&mut self, kind: EdgeKind) {
        self.kind = kind;
    }

    /// The branch weight / probability, if set.
    #[inline]
    #[must_use]
    pub const fn weight(&self) -> Option<f64> {
        self.weight
    }

    /// Set the branch weight / probability.
    #[inline]
    pub const fn set_weight(&mut self, weight: Option<f64>) {
        self.weight = weight;
    }

    /// The consumer-defined edge metadata.
    #[inline]
    #[must_use]
    pub const fn payload(&self) -> &E {
        &self.payload
    }

    /// Mutable access to the consumer-defined edge metadata.
    #[inline]
    pub const fn payload_mut(&mut self) -> &mut E {
        &mut self.payload
    }

    /// Consume the edge and return its consumer-defined metadata.
    #[inline]
    #[must_use]
    pub fn into_payload(self) -> E {
        self.payload
    }
}

impl<E> KindedEdge for Edge<E> {
    fn kind(&self) -> EdgeKind {
        self.kind
    }
}

/// Control-flow accessors of a borrowed edge whose data is an [`Edge`].
///
/// [`Cfg::edge`](crate::Cfg::edge) and every view adapter over a CFG yield
/// `EdgeRef<'_, BlockId, EdgeId, Edge<E>>`, so identity, endpoints, kind,
/// weight, and payload all read off one value.
impl<'g, N: Copy, I: Copy, E> EdgeRef<'g, N, I, Edge<E>> {
    /// The classification of this edge.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> EdgeKind {
        self.data().kind
    }

    /// The branch weight / probability, if set.
    #[inline]
    #[must_use]
    pub const fn weight(&self) -> Option<f64> {
        self.data().weight
    }

    /// The consumer-defined edge metadata.
    #[inline]
    #[must_use]
    pub const fn payload(&self) -> &'g E {
        &self.data().payload
    }
}
