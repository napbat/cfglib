//! Payload-generic directed multigraph storage.
//!
//! [`DirectedGraph`] is the graph substrate for code-intelligence products that
//! do not model basic blocks: value-flow graphs, call graphs, type-relation
//! graphs, import graphs, grammar dependencies, and similar structures. Nodes
//! and edges both retain consumer-defined payloads, parallel edges are valid,
//! and forward/reverse adjacency is maintained together.

extern crate alloc;
use alloc::vec::Vec;
use core::ops::Index;

use smallvec::SmallVec;

use crate::graph::view::{DenseNodeId, DirectedGraphView};

/// Dense identity of a node in a [`DirectedGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeId(u32);

impl NodeId {
    /// Construct an identity from a dense zero-based index.
    ///
    /// # Panics
    ///
    /// Panics when `index` exceeds `u32::MAX`.
    #[must_use]
    pub fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("node index exceeds u32::MAX"))
    }

    /// Construct an identity from its raw integer representation.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Return this identity's dense zero-based index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl core::fmt::Display for NodeId {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "n{}", self.0)
    }
}

impl DenseNodeId for NodeId {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }

    fn index(self) -> usize {
        self.index()
    }
}

/// Opaque identifier for an edge within a graph arena.
///
/// Shared by [`DirectedGraph`] and [`Cfg`](crate::Cfg): both arenas mint
/// dense edge identities that stay stable across tombstoned removals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EdgeId(pub(crate) u32);

impl EdgeId {
    pub(crate) fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("edge index exceeds u32::MAX"))
    }

    /// Construct an identity from its raw integer representation.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the raw index.
    #[inline]
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl core::fmt::Display for EdgeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "e{}", self.0)
    }
}

/// A directed edge carrying a consumer-defined payload.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DirectedEdge<E> {
    id: EdgeId,
    source: NodeId,
    target: NodeId,
    payload: E,
}

impl<E> DirectedEdge<E> {
    /// Return the edge identity.
    #[must_use]
    pub const fn id(&self) -> EdgeId {
        self.id
    }

    /// Return the source node.
    #[must_use]
    pub const fn source(&self) -> NodeId {
        self.source
    }

    /// Return the target node.
    #[must_use]
    pub const fn target(&self) -> NodeId {
        self.target
    }

    /// Borrow the consumer-defined edge payload.
    #[must_use]
    pub const fn payload(&self) -> &E {
        &self.payload
    }

    /// Mutably borrow the consumer-defined edge payload.
    pub const fn payload_mut(&mut self) -> &mut E {
        &mut self.payload
    }

    /// Consume the edge and return its payload.
    #[must_use]
    pub fn into_payload(self) -> E {
        self.payload
    }
}

/// An owned directed multigraph with stable node and edge identities.
///
/// Node identities are dense and never reused. Removing an edge leaves a
/// tombstone so every other [`EdgeId`] remains stable. Node removal is omitted
/// deliberately because it cannot preserve both dense algorithm indexes and
/// stable identities; consumers can build a compact replacement graph instead.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DirectedGraph<N, E> {
    nodes: Vec<N>,
    edges: Vec<Option<DirectedEdge<E>>>,
    outgoing: Vec<SmallVec<[EdgeId; 4]>>,
    incoming: Vec<SmallVec<[EdgeId; 4]>>,
}

impl<N, E> DirectedGraph<N, E> {
    /// Create an empty directed graph.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            outgoing: Vec::new(),
            incoming: Vec::new(),
        }
    }

    /// Create an empty graph with space for the requested nodes and edges.
    #[must_use]
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(nodes),
            edges: Vec::with_capacity(edges),
            outgoing: Vec::with_capacity(nodes),
            incoming: Vec::with_capacity(nodes),
        }
    }

    /// Add a node and return its stable identity.
    pub fn add_node(&mut self, payload: N) -> NodeId {
        let id = NodeId::from_index(self.nodes.len());
        self.nodes.push(payload);
        self.outgoing.push(SmallVec::new());
        self.incoming.push(SmallVec::new());
        id
    }

    /// Borrow a node payload.
    ///
    /// # Panics
    ///
    /// Panics when `node` does not belong to this graph.
    #[must_use]
    pub fn node(&self, node: NodeId) -> &N {
        &self.nodes[node.index()]
    }

    /// Mutably borrow a node payload.
    ///
    /// # Panics
    ///
    /// Panics when `node` does not belong to this graph.
    pub fn node_mut(&mut self, node: NodeId) -> &mut N {
        &mut self.nodes[node.index()]
    }

    /// Return all node payloads in identity order.
    #[must_use]
    pub fn nodes(&self) -> &[N] {
        &self.nodes
    }

    /// Iterate over every node identity in allocation order.
    pub fn node_ids(&self) -> impl ExactSizeIterator<Item = NodeId> + '_ {
        (0..self.nodes.len()).map(NodeId::from_index)
    }

    /// Return the number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return whether the graph contains no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Add a directed edge and return its stable identity.
    ///
    /// Parallel edges and self-edges are retained.
    ///
    /// # Panics
    ///
    /// Panics when either endpoint does not belong to this graph.
    pub fn add_edge(&mut self, source: NodeId, target: NodeId, payload: E) -> EdgeId {
        assert!(
            source.index() < self.nodes.len(),
            "source node is out of range"
        );
        assert!(
            target.index() < self.nodes.len(),
            "target node is out of range"
        );

        let id = EdgeId::from_index(self.edges.len());
        self.edges.push(Some(DirectedEdge {
            id,
            source,
            target,
            payload,
        }));
        self.outgoing[source.index()].push(id);
        self.incoming[target.index()].push(id);
        id
    }

    /// Borrow a live edge.
    ///
    /// # Panics
    ///
    /// Panics when the identity is out of range or the edge was removed.
    #[must_use]
    pub fn edge(&self, edge: EdgeId) -> &DirectedEdge<E> {
        self.edges[edge.index()]
            .as_ref()
            .expect("edge has been removed")
    }

    /// Mutably borrow a live edge.
    ///
    /// # Panics
    ///
    /// Panics when the identity is out of range or the edge was removed.
    pub fn edge_mut(&mut self, edge: EdgeId) -> &mut DirectedEdge<E> {
        self.edges[edge.index()]
            .as_mut()
            .expect("edge has been removed")
    }

    /// Iterate over all live edges in identity order.
    pub fn edges(&self) -> impl Iterator<Item = &DirectedEdge<E>> {
        self.edges.iter().filter_map(Option::as_ref)
    }

    /// Return the number of live edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.iter().filter(|edge| edge.is_some()).count()
    }

    /// Return the number of edge slots, including tombstones left by
    /// [`remove_edge`](Self::remove_edge).
    ///
    /// Every live [`EdgeId`]'s index is strictly below this, so it is the
    /// correct size for dense edge-indexed side tables (`seen` bitmaps,
    /// per-edge facts).
    #[must_use]
    pub fn edge_slot_count(&self) -> usize {
        self.edges.len()
    }

    /// Return outgoing edge identities for `node`.
    #[must_use]
    pub fn outgoing_edges(&self, node: NodeId) -> &[EdgeId] {
        &self.outgoing[node.index()]
    }

    /// Return incoming edge identities for `node`.
    #[must_use]
    pub fn incoming_edges(&self, node: NodeId) -> &[EdgeId] {
        &self.incoming[node.index()]
    }

    /// Iterate over outgoing neighbor identities, retaining parallel entries.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn successors(&self, node: NodeId) -> impl ExactSizeIterator<Item = NodeId> + '_ {
        self.outgoing[node.index()]
            .iter()
            .map(|edge| self.edge(*edge).target)
    }

    /// Iterate over incoming neighbor identities, retaining parallel entries.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn predecessors(&self, node: NodeId) -> impl ExactSizeIterator<Item = NodeId> + '_ {
        self.incoming[node.index()]
            .iter()
            .map(|edge| self.edge(*edge).source)
    }

    /// Remove an edge while preserving every other identity.
    pub fn remove_edge(&mut self, edge: EdgeId) -> Option<DirectedEdge<E>> {
        let removed = self.edges.get_mut(edge.index())?.take()?;
        self.outgoing[removed.source.index()].retain(|candidate| *candidate != edge);
        self.incoming[removed.target.index()].retain(|candidate| *candidate != edge);
        Some(removed)
    }
}

impl<N, E> Default for DirectedGraph<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> DirectedGraphView for DirectedGraph<N, E> {
    type NodeId = NodeId;

    fn node_count(&self) -> usize {
        self.node_count()
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.successors(node)
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.predecessors(node)
    }
}

impl<N, E> Index<NodeId> for DirectedGraph<N, E> {
    type Output = N;

    fn index(&self, node: NodeId) -> &Self::Output {
        self.node(node)
    }
}

impl<N, E> Index<EdgeId> for DirectedGraph<N, E> {
    type Output = DirectedEdge<E>;

    fn index(&self, edge: EdgeId) -> &Self::Output {
        self.edge(edge)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Provenance {
        line: u32,
        kind: &'static str,
    }

    #[test]
    fn payloads_and_bidirectional_adjacency_are_retained() {
        let mut graph = DirectedGraph::new();
        let source = graph.add_node("source");
        let target = graph.add_node("target");
        let edge = graph.add_edge(
            source,
            target,
            Provenance {
                line: 12,
                kind: "assign",
            },
        );

        assert_eq!(graph[source], "source");
        assert_eq!(graph[edge].payload().line, 12);
        assert_eq!(graph.successors(source).collect::<Vec<_>>(), vec![target]);
        assert_eq!(graph.predecessors(target).collect::<Vec<_>>(), vec![source]);
    }

    #[test]
    fn parallel_edges_and_stable_removal_are_supported() {
        let mut graph = DirectedGraph::new();
        let source = graph.add_node(());
        let target = graph.add_node(());
        let first = graph.add_edge(source, target, "read");
        let second = graph.add_edge(source, target, "call");

        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.successors(source).count(), 2);
        assert_eq!(graph.remove_edge(first).unwrap().into_payload(), "read");
        assert_eq!(graph[second].payload(), &"call");
        assert_eq!(graph.edge_count(), 1);
    }
}
