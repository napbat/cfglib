//! [`KeyedGraph`]: a [`Graph`] whose nodes are minted from sparse consumer
//! keys.
//!
//! The dense-identity contract ([`DenseId`](super::view::DenseId)) is what
//! keeps the algorithms allocation-lean, but consumer graphs are usually
//! keyed by sparse identities (symbol ids, paths, coordinates). Every adopter
//! ends up writing the same `BTreeMap<K, NodeId>` interner; this wraps that
//! once and stays a plain [`Graph`] underneath — the view impls and
//! [`graph`](KeyedGraph::graph)/[`into_parts`](KeyedGraph::into_parts)
//! expose it, so every algorithm in the crate applies unchanged.

extern crate alloc;
use alloc::collections::BTreeMap;

use super::edge_view::{EdgeRef, EdgeView};
use super::store::{EdgeId, Graph, NodeId};
use super::view::{GraphView, NodeView};

/// A directed multigraph keyed by a consumer identity `K`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "K: serde::Serialize + Ord, N: serde::Serialize, E: serde::Serialize",
        deserialize = "K: serde::Deserialize<'de> + Ord, N: serde::Deserialize<'de>, \
                       E: serde::Deserialize<'de>"
    ))
)]
pub struct KeyedGraph<K: Ord, N, E> {
    graph: Graph<N, E>,
    ids: BTreeMap<K, NodeId>,
}

impl<K: Clone + Ord, N, E> KeyedGraph<K, N, E> {
    /// Create an empty keyed graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            ids: BTreeMap::new(),
        }
    }

    /// The dense node id for `key`, minting a node with `payload()` on
    /// first sight.
    pub fn ensure_node(&mut self, key: &K, payload: impl FnOnce() -> N) -> NodeId {
        if let Some(&id) = self.ids.get(key) {
            return id;
        }
        let id = self.graph.add_node(payload());
        self.ids.insert(key.clone(), id);
        id
    }

    /// The dense node id for `key`, when it has been minted.
    #[must_use]
    pub fn node_id(&self, key: &K) -> Option<NodeId> {
        self.ids.get(key).copied()
    }

    /// Add an edge between two minted nodes.
    ///
    /// # Panics
    ///
    /// Panics when either endpoint does not belong to this graph.
    pub fn add_edge(&mut self, source: NodeId, target: NodeId, payload: E) -> EdgeId {
        self.graph.add_edge(source, target, payload)
    }

    /// Add an edge between two keys, when both have been minted.
    pub fn add_edge_between(&mut self, source: &K, target: &K, payload: E) -> Option<EdgeId> {
        let source = self.node_id(source)?;
        let target = self.node_id(target)?;
        Some(self.graph.add_edge(source, target, payload))
    }

    /// Borrow the underlying graph (for algorithms and payload access).
    #[must_use]
    pub const fn graph(&self) -> &Graph<N, E> {
        &self.graph
    }

    /// Mutably borrow the underlying graph.
    pub const fn graph_mut(&mut self) -> &mut Graph<N, E> {
        &mut self.graph
    }

    /// Iterate over the minted keys and their dense ids.
    pub fn keys(&self) -> impl Iterator<Item = (&K, NodeId)> + '_ {
        self.ids.iter().map(|(key, &id)| (key, id))
    }

    /// Consume the wrapper, returning the graph and the key table.
    #[must_use]
    pub fn into_parts(self) -> (Graph<N, E>, BTreeMap<K, NodeId>) {
        (self.graph, self.ids)
    }
}

impl<Key: Clone + Ord, E> KeyedGraph<Key, Key, E> {
    /// The dense node id for `key` in a self-keyed graph (node payload =
    /// key), minting on first sight — the common case without the
    /// [`ensure_node`](Self::ensure_node) double-mention of the key.
    pub fn intern(&mut self, key: &Key) -> NodeId {
        self.ensure_node(key, || key.clone())
    }
}

impl<K: Clone + Ord, N, E> Default for KeyedGraph<K, N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord, N, E> GraphView for KeyedGraph<K, N, E> {
    type NodeId = NodeId;

    fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.node_ids()
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.successors(node)
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.predecessors(node)
    }
}

impl<K: Ord, N, E> NodeView for KeyedGraph<K, N, E> {
    type NodeData = N;

    fn node(&self, node: Self::NodeId) -> &Self::NodeData {
        self.graph.node(node)
    }
}

impl<K: Ord, N, E> EdgeView for KeyedGraph<K, N, E> {
    type EdgeId = EdgeId;
    type EdgeData = E;

    fn edge_bound(&self) -> usize {
        self.graph.edge_bound()
    }

    fn edge_ids(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.graph.edge_ids()
    }

    fn outgoing(&self, node: NodeId) -> impl Iterator<Item = EdgeId> + '_ {
        self.graph.outgoing(node)
    }

    fn incoming(&self, node: NodeId) -> impl Iterator<Item = EdgeId> + '_ {
        self.graph.incoming(node)
    }

    fn edge(&self, edge: EdgeId) -> EdgeRef<'_, NodeId, EdgeId, E> {
        EdgeView::edge(&self.graph, edge)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::scc::tarjan_scc;
    use alloc::string::String;
    use alloc::string::ToString;

    #[test]
    fn self_keyed_interning_is_idempotent() {
        let mut graph: KeyedGraph<String, String, ()> = KeyedGraph::new();
        let a = graph.intern(&"pkg::a".to_string());
        assert_eq!(a, graph.intern(&"pkg::a".to_string()));
        assert_eq!(graph.graph().node(a), "pkg::a");
    }

    #[test]
    fn interns_sparse_keys_and_runs_algorithms() {
        // Sparse symbol-id-like keys.
        let mut graph: KeyedGraph<String, &str, &str> = KeyedGraph::new();
        let a = graph.ensure_node(&"pkg::a".to_string(), || "a");
        let a_again = graph.ensure_node(&"pkg::a".to_string(), || unreachable!("already minted"));
        assert_eq!(a, a_again);
        let b = graph.ensure_node(&"pkg::b".to_string(), || "b");
        graph.add_edge(a, b, "extends");
        assert_eq!(
            graph.add_edge_between(&"pkg::b".to_string(), &"pkg::a".to_string(), "cycle"),
            Some(EdgeId::from_raw(1))
        );
        assert_eq!(
            graph.add_edge_between(&"missing".to_string(), &"pkg::a".to_string(), "x"),
            None
        );

        // The view impl feeds the generic algorithms directly.
        let sccs = tarjan_scc(&graph);
        assert_eq!(sccs.len(), 1, "a <-> b is one component");
        assert_eq!(graph.keys().count(), 2);
    }
}
