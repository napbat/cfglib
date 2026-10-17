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
//!
//! The key table and the graph describe the same nodes, so the wrapper never
//! hands out `&mut Graph`: a removal or a compaction through it would leave
//! the table naming removed nodes or, after a renumbering, the wrong ones.
//! [`remove_node`](KeyedGraph::remove_node) and
//! [`compact`](KeyedGraph::compact) keep the table in step instead.

extern crate alloc;
use alloc::collections::BTreeMap;

use super::edge_view::{EdgeRef, EdgeView};
use super::store::{EdgeId, Graph, NodeId, Renumbering};
use super::view::{GraphView, NodeView};

/// A directed multigraph keyed by a consumer identity `K`.
///
/// Every live node has exactly one key, and every key names a live node. A
/// deserialized graph is checked against that, and one whose key table
/// disagrees with its nodes is rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(
            serialize = "K: serde::Serialize + Ord, N: serde::Serialize, E: serde::Serialize",
            deserialize = "K: serde::Deserialize<'de> + Ord, N: serde::Deserialize<'de>, \
                           E: serde::Deserialize<'de>"
        ),
        try_from = "unchecked::Unchecked<K, N, E>"
    )
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

    /// Remove the node minted for `key`, with every edge that touches it, and
    /// forget the key.
    ///
    /// Returns whether `key` had a node. As with [`Graph::remove_node`], the
    /// slot keeps its identity until [`compact`](Self::compact), so no other
    /// node's id changes; a later [`ensure_node`](Self::ensure_node) of the
    /// same key mints a new node.
    pub fn remove_node(&mut self, key: &K) -> bool {
        self.ids
            .remove(key)
            .is_some_and(|node| self.graph.remove_node(node))
    }

    /// Compact the graph and renumber the key table with it.
    ///
    /// The key table follows [`Graph::compact`]: afterwards every key names
    /// the node it named before, under its new identity, which the returned
    /// [`Renumbering`] also reports for ids held elsewhere.
    pub fn compact(&mut self) -> Renumbering {
        let renumbering = self.graph.compact();
        // Every key names a live node and compaction keeps every live node,
        // so nothing is dropped here; a key that named a removed node could
        // not be renumbered, and has no node to name.
        self.ids.retain(|_, node| {
            renumbering.node(*node).is_some_and(|renumbered| {
                *node = renumbered;
                true
            })
        });
        renumbering
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

/// The stored form of a [`KeyedGraph`], checked on its way in.
#[cfg(feature = "serde")]
mod unchecked {
    extern crate alloc;
    use alloc::collections::BTreeMap;
    use alloc::vec;

    use super::{Graph, KeyedGraph, NodeId};

    #[derive(serde::Deserialize)]
    #[serde(bound(
        deserialize = "K: serde::Deserialize<'de> + Ord, N: serde::Deserialize<'de>, \
                       E: serde::Deserialize<'de>"
    ))]
    pub struct Unchecked<K: Ord, N, E> {
        graph: Graph<N, E>,
        ids: BTreeMap<K, NodeId>,
    }

    impl<K: Ord, N, E> TryFrom<Unchecked<K, N, E>> for KeyedGraph<K, N, E> {
        type Error = &'static str;

        /// Accepts a key table that names every live node exactly once.
        fn try_from(stored: Unchecked<K, N, E>) -> Result<Self, Self::Error> {
            let Unchecked { graph, ids } = stored;
            if ids.len() != graph.node_count() {
                return Err("the key table and the graph hold different numbers of nodes");
            }
            let mut named = vec![false; graph.node_bound()];
            for &node in ids.values() {
                if !graph.contains_node(node) {
                    return Err("the key table names a node the graph does not hold");
                }
                if core::mem::replace(&mut named[node.index()], true) {
                    return Err("the key table names one node under two keys");
                }
            }
            Ok(Self { graph, ids })
        }
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

    #[test]
    fn removing_a_key_forgets_its_node_and_the_key_can_be_minted_again() {
        let mut graph: KeyedGraph<&str, &str, ()> = KeyedGraph::new();
        let a = graph.ensure_node(&"a", || "a");
        let b = graph.ensure_node(&"b", || "b");
        graph.add_edge(a, b, ());

        assert!(graph.remove_node(&"a"));
        assert!(!graph.remove_node(&"a"), "a key is removed once");
        assert_eq!(graph.node_id(&"a"), None);
        assert_eq!(graph.graph().edge_count(), 0, "its edges go with it");

        let again = graph.ensure_node(&"a", || "a");
        assert_ne!(again, a, "a removed identity is never reissued");
        assert!(graph.graph().contains_node(again));
    }

    #[test]
    fn compaction_renumbers_the_key_table_with_the_graph() {
        let mut graph: KeyedGraph<&str, &str, ()> = KeyedGraph::new();
        let a = graph.ensure_node(&"a", || "a");
        let b = graph.ensure_node(&"b", || "b");
        let c = graph.ensure_node(&"c", || "c");
        graph.add_edge(b, c, ());
        graph.remove_node(&"a");

        let renumbering = graph.compact();

        assert_eq!(renumbering.node(a), None);
        assert_eq!(graph.node_id(&"b"), renumbering.node(b));
        assert_eq!(graph.node_id(&"c"), renumbering.node(c));
        assert!(graph.graph().is_compact());
        for (key, node) in graph.keys() {
            assert_eq!(graph.graph().node(node), key);
        }
    }
}
