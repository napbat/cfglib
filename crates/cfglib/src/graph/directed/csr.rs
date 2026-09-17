//! Immutable directed multigraph storage with compressed sparse-row adjacency.
//!
//! [`CsrDirectedGraph`] serves large read-mostly graphs whose dense node and
//! edge identities no longer need mutation. Forward and reverse adjacency are
//! flat edge-id arrays addressed through per-node offsets, avoiding one
//! allocation or inline adjacency container per node.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Index;
use core::slice;

use crate::graph::edge_view::{EdgeGraphView, EdgeRef};
use crate::graph::view::{DirectedGraphView, NodeGraphView};

use super::NodeId;

/// A directed edge in immutable dense graph storage.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CsrDirectedEdge<E> {
    source: NodeId,
    target: NodeId,
    payload: E,
}

impl<E> CsrDirectedEdge<E> {
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
}

/// Immutable directed multigraph with dense identities and CSR adjacency.
///
/// Edge identities are their zero-based indexes in insertion order. Both
/// adjacency directions preserve that order, including parallel edges.
/// Mutation is intentionally confined to [`CsrDirectedGraphBuilder`]; the
/// completed graph has no tombstones, spare-capacity arenas, or per-node
/// adjacency containers.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CsrDirectedGraph<N, E> {
    nodes: Box<[N]>,
    edges: Box<[CsrDirectedEdge<E>]>,
    outgoing_offsets: Box<[u32]>,
    outgoing_edges: CsrAdjacency,
    incoming_offsets: Box<[u32]>,
    incoming_edges: Box<[u32]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum CsrAdjacency {
    Explicit(Box<[u32]>),
    Identity,
}

impl<N, E> CsrDirectedGraph<N, E> {
    /// Create an empty immutable graph.
    #[must_use]
    pub fn new() -> Self {
        CsrDirectedGraphBuilder::new().finish()
    }

    /// Borrow one node payload.
    ///
    /// # Panics
    ///
    /// Panics when `node` does not belong to this graph.
    #[must_use]
    pub fn node(&self, node: NodeId) -> &N {
        &self.nodes[node.index()]
    }

    /// Return all node payloads in identity order.
    #[must_use]
    pub const fn nodes(&self) -> &[N] {
        &self.nodes
    }

    /// Iterate over every node identity in allocation order.
    pub fn node_ids(&self) -> impl ExactSizeIterator<Item = NodeId> + '_ {
        (0..self.nodes.len()).map(NodeId::from_index)
    }

    /// Return the number of nodes.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return whether `node` belongs to this graph.
    #[must_use]
    pub fn contains_node(&self, node: NodeId) -> bool {
        node.index() < self.nodes.len()
    }

    /// Return whether this graph contains no nodes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Borrow one edge by its dense identity.
    ///
    /// # Panics
    ///
    /// Panics when `edge` does not belong to this graph.
    #[must_use]
    pub fn edge(&self, edge: u32) -> &CsrDirectedEdge<E> {
        &self.edges[edge as usize]
    }

    /// Return all edges in identity order.
    #[must_use]
    pub const fn edges(&self) -> &[CsrDirectedEdge<E>] {
        &self.edges
    }

    /// Return the number of edges.
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Return whether `edge` belongs to this graph.
    #[must_use]
    pub fn contains_edge(&self, edge: u32) -> bool {
        (edge as usize) < self.edges.len()
    }

    /// Return outgoing edge identities for `node` in insertion order.
    ///
    /// # Panics
    ///
    /// Panics when `node` does not belong to this graph.
    #[must_use]
    pub fn outgoing_edges(&self, node: NodeId) -> impl ExactSizeIterator<Item = u32> + '_ {
        adjacency(&self.outgoing_offsets, &self.outgoing_edges, node)
    }

    /// Return incoming edge identities for `node` in insertion order.
    ///
    /// # Panics
    ///
    /// Panics when `node` does not belong to this graph.
    #[must_use]
    pub fn incoming_edges(&self, node: NodeId) -> impl ExactSizeIterator<Item = u32> + '_ {
        explicit_adjacency(&self.incoming_offsets, &self.incoming_edges, node)
            .iter()
            .copied()
    }

    /// Iterate over outgoing neighbor identities, retaining parallel entries.
    #[must_use]
    pub fn successors(&self, node: NodeId) -> impl ExactSizeIterator<Item = NodeId> + '_ {
        self.outgoing_edges(node).map(|edge| self.edge(edge).target)
    }

    /// Iterate over incoming neighbor identities, retaining parallel entries.
    #[must_use]
    pub fn predecessors(&self, node: NodeId) -> impl ExactSizeIterator<Item = NodeId> + '_ {
        self.incoming_edges(node).map(|edge| self.edge(edge).source)
    }
}

impl<N, E> Default for CsrDirectedGraph<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> DirectedGraphView for CsrDirectedGraph<N, E> {
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

impl<N, E> NodeGraphView for CsrDirectedGraph<N, E> {
    type NodeData = N;

    fn node_ref(&self, node: Self::NodeId) -> &Self::NodeData {
        self.node(node)
    }
}

impl<N, E> EdgeGraphView for CsrDirectedGraph<N, E> {
    type EdgeId = u32;
    type EdgeData = E;

    fn edge_slot_count(&self) -> usize {
        self.edge_count()
    }

    fn edge_ids(&self) -> impl Iterator<Item = Self::EdgeId> + '_ {
        0..u32::try_from(self.edge_count()).expect("CSR edge count fits u32")
    }

    fn outgoing_edges(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.outgoing_edges(node)
    }

    fn incoming_edges(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.incoming_edges(node)
    }

    fn edge_ref(
        &self,
        edge: Self::EdgeId,
    ) -> EdgeRef<'_, Self::NodeId, Self::EdgeId, Self::EdgeData> {
        let value = self.edge(edge);
        EdgeRef::new(edge, value.source, value.target, &value.payload)
    }
}

impl<N, E> Index<NodeId> for CsrDirectedGraph<N, E> {
    type Output = N;

    fn index(&self, node: NodeId) -> &Self::Output {
        self.node(node)
    }
}

impl<N, E> Index<u32> for CsrDirectedGraph<N, E> {
    type Output = CsrDirectedEdge<E>;

    fn index(&self, edge: u32) -> &Self::Output {
        self.edge(edge)
    }
}

/// Mutable collector for an immutable [`CsrDirectedGraph`].
///
/// The builder retains only dense node and edge arenas. It creates both flat
/// adjacency indexes once, when [`finish`](Self::finish) is called.
///
/// # Superseded
///
/// [`graph::store::Graph`](crate::graph::store::Graph) covers this and more:
/// it is the same compressed representation, but it can be mutated after the
/// indexes exist and it does not need a second type to be built.
/// `Graph::new()`, `add_node`, `add_edge`, `compact()` is this builder plus
/// [`finish`](Self::finish), and the result stays appendable. New code should
/// use it; this type is kept until its callers move.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsrDirectedGraphBuilder<N, E> {
    nodes: Vec<N>,
    edges: Vec<CsrDirectedEdge<E>>,
}

impl<N, E> CsrDirectedGraphBuilder<N, E> {
    /// Create an empty builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Create a builder with space for the requested nodes and edges.
    #[must_use]
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(nodes),
            edges: Vec::with_capacity(edges),
        }
    }

    /// Add a node and return its stable identity.
    pub fn add_node(&mut self, payload: N) -> NodeId {
        let id = NodeId::from_index(self.nodes.len());
        self.nodes.push(payload);
        id
    }

    /// Borrow one node payload.
    ///
    /// # Panics
    ///
    /// Panics when `node` does not belong to this builder.
    #[must_use]
    pub fn node(&self, node: NodeId) -> &N {
        &self.nodes[node.index()]
    }

    /// Mutably borrow one node payload.
    ///
    /// # Panics
    ///
    /// Panics when `node` does not belong to this builder.
    pub fn node_mut(&mut self, node: NodeId) -> &mut N {
        &mut self.nodes[node.index()]
    }

    /// Return the number of collected nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Add an edge and return its stable dense identity.
    ///
    /// Parallel edges and self-edges are retained.
    ///
    /// # Panics
    ///
    /// Panics when either endpoint does not belong to this builder or the edge
    /// count exceeds `u32::MAX`.
    pub fn add_edge(&mut self, source: NodeId, target: NodeId, payload: E) -> u32 {
        assert!(
            source.index() < self.nodes.len(),
            "source node is out of range"
        );
        assert!(
            target.index() < self.nodes.len(),
            "target node is out of range"
        );
        let id = u32::try_from(self.edges.len()).expect("CSR edge count exceeds u32::MAX");
        self.edges.push(CsrDirectedEdge {
            source,
            target,
            payload,
        });
        id
    }

    /// Freeze the collected payloads and build forward/reverse CSR indexes.
    #[must_use]
    pub fn finish(self) -> CsrDirectedGraph<N, E> {
        self.finish_with_outgoing(false)
    }

    /// Freeze edges already grouped by source and omit their redundant
    /// outgoing edge-id array.
    ///
    /// Edge insertion order remains edge identity and outgoing adjacency
    /// order. Incoming adjacency is still indexed explicitly.
    ///
    /// # Panics
    ///
    /// Panics when edge sources are not in nondecreasing node-id order.
    #[must_use]
    pub fn finish_source_ordered(self) -> CsrDirectedGraph<N, E> {
        self.finish_with_outgoing(true)
    }

    fn finish_with_outgoing(self, source_ordered: bool) -> CsrDirectedGraph<N, E> {
        if source_ordered {
            assert!(
                self.edges
                    .windows(2)
                    .all(|pair| pair[0].source <= pair[1].source),
                "source-ordered CSR edges must be grouped by source"
            );
        }
        let (outgoing_offsets, outgoing_edges) = if source_ordered {
            (
                adjacency_offsets(&self.edges, self.nodes.len(), &|edge| edge.source),
                CsrAdjacency::Identity,
            )
        } else {
            let (offsets, edges) =
                compact_adjacency(&self.edges, self.nodes.len(), |edge| edge.source);
            (offsets, CsrAdjacency::Explicit(edges))
        };
        let incoming = compact_adjacency(&self.edges, self.nodes.len(), |edge| edge.target);
        CsrDirectedGraph {
            nodes: self.nodes.into_boxed_slice(),
            edges: self.edges.into_boxed_slice(),
            outgoing_offsets,
            outgoing_edges,
            incoming_offsets: incoming.0,
            incoming_edges: incoming.1,
        }
    }
}

impl<N, E> Default for CsrDirectedGraphBuilder<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

fn compact_adjacency<E>(
    edges: &[CsrDirectedEdge<E>],
    node_count: usize,
    endpoint: impl Fn(&CsrDirectedEdge<E>) -> NodeId,
) -> (Box<[u32]>, Box<[u32]>) {
    let offsets = adjacency_offsets(edges, node_count, &endpoint);
    let mut next = offsets[..node_count].to_vec();
    let mut adjacency = vec![0_u32; edges.len()];
    for (index, edge) in edges.iter().enumerate() {
        let endpoint = endpoint(edge).index();
        let slot = next[endpoint] as usize;
        adjacency[slot] = u32::try_from(index).expect("CSR edge count exceeds u32::MAX");
        next[endpoint] += 1;
    }
    (offsets, adjacency.into_boxed_slice())
}

fn adjacency_offsets<E>(
    edges: &[CsrDirectedEdge<E>],
    node_count: usize,
    endpoint: &impl Fn(&CsrDirectedEdge<E>) -> NodeId,
) -> Box<[u32]> {
    let mut offsets = vec![0_u32; node_count + 1];
    for edge in edges {
        offsets[endpoint(edge).index() + 1] += 1;
    }
    for index in 1..offsets.len() {
        offsets[index] += offsets[index - 1];
    }

    offsets.into_boxed_slice()
}

enum CsrEdgeIds<'a> {
    Explicit(core::iter::Copied<slice::Iter<'a, u32>>),
    Identity(core::ops::Range<u32>),
}

impl Iterator for CsrEdgeIds<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Explicit(edges) => edges.next(),
            Self::Identity(edges) => edges.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Explicit(edges) => edges.size_hint(),
            Self::Identity(edges) => edges.size_hint(),
        }
    }
}

impl ExactSizeIterator for CsrEdgeIds<'_> {}

fn adjacency<'a>(offsets: &[u32], edges: &'a CsrAdjacency, node: NodeId) -> CsrEdgeIds<'a> {
    let index = node.index();
    let start = offsets[index];
    let end = offsets[index + 1];
    match edges {
        CsrAdjacency::Explicit(edges) => {
            CsrEdgeIds::Explicit(edges[start as usize..end as usize].iter().copied())
        }
        CsrAdjacency::Identity => CsrEdgeIds::Identity(start..end),
    }
}

fn explicit_adjacency<'a>(offsets: &[u32], edges: &'a [u32], node: NodeId) -> &'a [u32] {
    let index = node.index();
    let start = offsets[index] as usize;
    let end = offsets[index + 1] as usize;
    &edges[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DirectedGraph, NodeGraphView, TraversalDirection, breadth_first_view_edges};

    fn node_label<G>(graph: &G, node: NodeId) -> &'static str
    where
        G: NodeGraphView<NodeId = NodeId, NodeData = &'static str>,
    {
        graph.node_ref(node)
    }

    #[test]
    fn node_payload_view_accepts_arena_and_csr_storage() {
        let mut arena = DirectedGraph::<&'static str, ()>::new();
        let arena_node = arena.add_node("arena");

        let mut builder = CsrDirectedGraphBuilder::<&'static str, ()>::new();
        let csr_node = builder.add_node("csr");
        let csr = builder.finish();

        assert_eq!(node_label(&arena, arena_node), "arena");
        assert_eq!(node_label(&csr, csr_node), "csr");
    }

    #[test]
    fn payloads_and_bidirectional_order_are_retained() {
        let mut builder = CsrDirectedGraphBuilder::new();
        let source = builder.add_node("source");
        let middle = builder.add_node("middle");
        let target = builder.add_node("target");
        let first = builder.add_edge(source, middle, "first");
        let second = builder.add_edge(source, target, "second");
        let parallel = builder.add_edge(source, middle, "parallel");
        let graph = builder.finish();

        assert_eq!(graph.nodes(), &["source", "middle", "target"]);
        assert_eq!(
            graph.outgoing_edges(source).collect::<Vec<_>>(),
            [first, second, parallel]
        );
        assert_eq!(
            graph.incoming_edges(middle).collect::<Vec<_>>(),
            [first, parallel]
        );
        assert_eq!(graph.edge(second).payload(), &"second");
        assert_eq!(graph.edge(second).source(), source);
        assert_eq!(graph.edge(second).target(), target);
    }

    #[test]
    fn generic_edge_traversal_reads_csr_storage() {
        let mut builder = CsrDirectedGraphBuilder::new();
        let source = builder.add_node(());
        let left = builder.add_node(());
        let right = builder.add_node(());
        let sink = builder.add_node(());
        let source_left = builder.add_edge(source, left, ());
        let source_right = builder.add_edge(source, right, ());
        let left_sink = builder.add_edge(left, sink, ());
        let right_sink = builder.add_edge(right, sink, ());
        let graph = builder.finish();

        let steps = breadth_first_view_edges(&graph, source, TraversalDirection::Outgoing);
        let edges: Vec<_> = steps.into_iter().map(|step| step.edge).collect();
        assert_eq!(edges, [source_left, source_right, left_sink, right_sink]);
    }

    #[test]
    fn empty_graph_has_empty_adjacency() {
        let graph = CsrDirectedGraph::<(), ()>::new();
        assert!(graph.is_empty());
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn source_ordered_graph_uses_implicit_outgoing_edge_ids() {
        let mut builder = CsrDirectedGraphBuilder::new();
        let source = builder.add_node(());
        let other = builder.add_node(());
        let target = builder.add_node(());
        let first = builder.add_edge(source, target, ());
        let second = builder.add_edge(source, other, ());
        let third = builder.add_edge(other, target, ());
        let graph = builder.finish_source_ordered();

        assert!(matches!(graph.outgoing_edges, CsrAdjacency::Identity));

        assert_eq!(
            graph.outgoing_edges(source).collect::<Vec<_>>(),
            [first, second]
        );
        assert_eq!(graph.outgoing_edges(other).collect::<Vec<_>>(), [third]);
        assert_eq!(
            graph.incoming_edges(target).collect::<Vec<_>>(),
            [first, third]
        );
    }
}
