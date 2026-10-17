//! Edges connecting basic blocks in a control-flow graph.

use crate::graph::edge_view::EdgeRef;

pub use crate::graph::store::{EdgeId, EdgeTag};

/// Declares [`EdgeKind`] with its canonical names, so the variants, the list
/// of every kind, and the name of each are written once.
macro_rules! edge_kinds {
    (
        $(#[$enum_meta:meta])*
        pub enum EdgeKind {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident => $name:literal,
            )*
        }
    ) => {
        $(#[$enum_meta])*
        pub enum EdgeKind {
            $(
                $(#[$variant_meta])*
                $variant,
            )*
        }

        impl EdgeKind {
            /// Every kind, in declaration order.
            ///
            /// This is what [`from_name`](Self::from_name) searches, so naming
            /// stays defined in exactly one place: [`name`](Self::name).
            pub const ALL: [Self; [$(Self::$variant),*].len()] = [$(Self::$variant),*];

            /// The canonical name of this kind, which is what
            /// [`Display`](core::fmt::Display) prints and what the text form
            /// of a control-flow graph writes.
            #[must_use]
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)*
                }
            }
        }
    };
}

edge_kinds! {
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
        Fallthrough => "fallthrough",
        /// Taken branch of a conditional (the "true" path).
        ConditionalTrue => "true",
        /// Not-taken branch of a conditional (the "false" / merge path).
        ConditionalFalse => "false",
        /// Unconditional jump (structured break/switch exit).
        Unconditional => "unconditional",
        /// Back-edge to a loop header.
        Back => "back",
        /// Edge to a call target.
        Call => "call",
        /// Return edge from a call site.
        CallReturn => "call_return",
        /// Edge for a switch/case arm.
        SwitchCase => "case",

        /// Direct explicit jump: a source `goto`, a machine `jmp` / `b`.
        ///
        /// Distinct from [`Unconditional`](Self::Unconditional): `Jump` records
        /// an explicit branch instruction, `Unconditional` a synthesized
        /// structured transfer (break, switch exit).
        Jump => "jump",
        /// Computed / indirect jump: a source computed goto or lowered `match`
        /// dispatch, a machine `jmp [rax]` through a jump table.
        IndirectJump => "indirect_jump",
        /// Indirect call: source dynamic dispatch or a function-pointer call, a
        /// machine `call [vtable]`.
        IndirectCall => "indirect_call",
        /// Edge into an exception-handler entry block.
        ExceptionHandler => "handler",
        /// Edge from a potentially-throwing instruction to a handler.
        ExceptionUnwind => "unwind",
        /// Edge from a protected region to the normal continuation.
        ExceptionLeave => "leave",
        /// Edge from a resume/rethrow point to the next exception dispatcher.
        ExceptionResume => "resume",
        /// Edge that resumes execution after an exception was handled in-place.
        ///
        /// Windows SEH and VEH use this for `EXCEPTION_CONTINUE_EXECUTION`: the
        /// target is the exact block at which execution resumes.
        ExceptionContinue => "continue_exception",
    }
}

impl EdgeKind {
    /// Whether the edge transfers control exceptionally rather than
    /// sequentially.
    ///
    /// [`ExceptionLeave`](Self::ExceptionLeave) is a normal transfer out
    /// of a protected region, so it stays sequential. The same split decides
    /// the [`ExceptionFlow`](crate::ExceptionFlow) payload an edge carries.
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

    /// The kind with this canonical name, or `None` for an unknown name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
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

impl core::fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
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
    /// Optional branch probability (0.0–1.0).
    ///
    /// When set, this is the likelihood of this edge being taken relative to
    /// the other outgoing edges of the same source block.
    /// [`CfgProfile`](crate::CfgProfile) propagates it into block
    /// frequencies, and DOT output uses it for visual emphasis.
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

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::collections::BTreeSet;
    use alloc::vec::Vec;

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

    #[test]
    fn all_lists_every_kind_once_in_declaration_order() {
        // A fieldless enum's discriminants count up from zero in declaration
        // order, so `ALL` must too: that is its documented order, and the
        // order `from_name` and every enumeration of the kinds relies on.
        for (position, kind) in EdgeKind::ALL.into_iter().enumerate() {
            assert_eq!(kind as usize, position, "{kind}");
        }
    }

    #[test]
    fn only_the_handler_unwind_resume_and_continue_kinds_are_exceptional() {
        let exceptional: Vec<_> = EdgeKind::ALL
            .into_iter()
            .filter(|kind| kind.is_exceptional())
            .collect();
        assert_eq!(
            exceptional,
            [
                EdgeKind::ExceptionHandler,
                EdgeKind::ExceptionUnwind,
                EdgeKind::ExceptionResume,
                EdgeKind::ExceptionContinue,
            ],
            "ExceptionLeave is a normal transfer out of a protected region"
        );
    }
}
