//! The stored form of one directed edge.

use super::id::{Id, IdTag, NodeTag};

/// One directed edge as the store holds it: two endpoints and a payload.
///
/// The record carries no identity of its own — an edge's identity is its slot
/// index — and no adjacency links. Intrusive links would cost eight bytes on
/// every edge including the compacted majority that does not need them, so
/// the incremental delta keeps its links in a parallel array sized to the
/// delta alone.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "E: serde::Serialize",
        deserialize = "E: serde::Deserialize<'de>"
    ))
)]
pub struct EdgeRecord<E, NT: IdTag = NodeTag> {
    pub(super) source: Id<NT>,
    pub(super) target: Id<NT>,
    pub(super) payload: E,
}

impl<E, NT: IdTag> EdgeRecord<E, NT> {
    /// Create a record between two endpoints.
    pub(super) const fn new(source: Id<NT>, target: Id<NT>, payload: E) -> Self {
        Self {
            source,
            target,
            payload,
        }
    }

    /// Return the source node.
    #[must_use]
    pub const fn source(&self) -> Id<NT> {
        self.source
    }

    /// Return the target node.
    #[must_use]
    pub const fn target(&self) -> Id<NT> {
        self.target
    }

    /// Borrow the consumer-defined edge payload.
    #[must_use]
    pub const fn payload(&self) -> &E {
        &self.payload
    }

    /// Mutably borrow the consumer-defined edge payload.
    ///
    /// Endpoints have no mutable accessor: moving an edge would invalidate
    /// the adjacency index that reaches it. Remove the edge and add the
    /// replacement instead.
    pub const fn payload_mut(&mut self) -> &mut E {
        &mut self.payload
    }

    /// Consume the record and return its payload.
    #[must_use]
    pub fn into_payload(self) -> E {
        self.payload
    }
}
