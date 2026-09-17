//! Edge-aware graph views and zero-copy edge filtering.

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::Edge;

use super::directed::{DirectedGraph, EdgeId, NodeId};
use super::view::{DirectedGraphView, NodeGraphView, Reversed, Rooted, RootedGraphView};

/// A copyable, ordered edge identity backed by a dense arena index.
///
/// Live edges may have gaps because graph storage uses tombstones. The index
/// therefore addresses an edge slot rather than promising that every value in
/// `0..edge_slot_count()` is live.
pub trait DenseEdgeId: Copy + Ord {
    /// Construct an identity from a valid dense arena index.
    fn from_index(index: usize) -> Self;

    /// Return the dense arena index.
    fn index(self) -> usize;
}

impl DenseEdgeId for EdgeId {
    fn from_index(index: usize) -> Self {
        Self::from_raw(u32::try_from(index).expect("edge index exceeds u32::MAX"))
    }

    fn index(self) -> usize {
        self.index()
    }
}

impl DenseEdgeId for usize {
    fn from_index(index: usize) -> Self {
        index
    }

    fn index(self) -> usize {
        self
    }
}

impl DenseEdgeId for u32 {
    fn from_index(index: usize) -> Self {
        Self::try_from(index).expect("edge index exceeds u32::MAX")
    }

    fn index(self) -> usize {
        usize::try_from(self).expect("u32 edge index exceeds usize::MAX")
    }
}

/// One borrowed edge exposed by an [`EdgeGraphView`].
///
/// Endpoints are oriented as this view presents them. A [`Reversed`] view
/// therefore swaps source and target while retaining identity and data.
#[derive(Debug)]
pub struct EdgeRef<'g, N, E, D: ?Sized> {
    id: E,
    source: N,
    target: N,
    data: &'g D,
}

impl<'g, N: Copy, E: Copy, D: ?Sized> EdgeRef<'g, N, E, D> {
    /// Construct a borrowed edge reference.
    #[must_use]
    pub const fn new(id: E, source: N, target: N, data: &'g D) -> Self {
        Self {
            id,
            source,
            target,
            data,
        }
    }

    /// The edge identity.
    #[must_use]
    pub const fn id(&self) -> E {
        self.id
    }

    /// The stored source node, independent of traversal direction.
    #[must_use]
    pub const fn source(&self) -> N {
        self.source
    }

    /// The stored target node, independent of traversal direction.
    #[must_use]
    pub const fn target(&self) -> N {
        self.target
    }

    /// The graph-specific edge data.
    #[must_use]
    pub const fn data(&self) -> &'g D {
        self.data
    }
}

impl<N: Copy, E: Copy, D: ?Sized> Copy for EdgeRef<'_, N, E, D> {}

impl<N: Copy, E: Copy, D: ?Sized> Clone for EdgeRef<'_, N, E, D> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Read-only edge identity, endpoints, data, and adjacency.
///
/// This companion to [`DirectedGraphView`] is opt-in for stores that retain
/// explicit edges. Node-only algorithms keep depending on the smaller trait;
/// edge-sensitive traversals, filters, validation, and dataflow use this one.
pub trait EdgeGraphView: DirectedGraphView {
    /// Stable edge identity used by this view.
    type EdgeId: DenseEdgeId;

    /// Data exposed for each live edge.
    type EdgeData: ?Sized;

    /// Number of edge arena slots, including tombstones.
    fn edge_slot_count(&self) -> usize;

    /// Iterate over every live edge identity in stable order.
    fn edge_ids(&self) -> impl Iterator<Item = Self::EdgeId> + '_;

    /// Iterate over outgoing edge identities in adjacency order.
    fn outgoing_edges(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_;

    /// Iterate over incoming edge identities in adjacency order.
    fn incoming_edges(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_;

    /// Borrow one live edge.
    ///
    /// # Panics
    ///
    /// Panics when `edge` is out of range or names a tombstone.
    fn edge_ref(
        &self,
        edge: Self::EdgeId,
    ) -> EdgeRef<'_, Self::NodeId, Self::EdgeId, Self::EdgeData>;
}

impl<I, E> EdgeGraphView for Cfg<I, E> {
    type EdgeId = EdgeId;
    type EdgeData = Edge<E>;

    fn edge_slot_count(&self) -> usize {
        self.edge_slots()
    }

    fn edge_ids(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.edges().map(Edge::id)
    }

    fn outgoing_edges(&self, node: BlockId) -> impl Iterator<Item = EdgeId> + '_ {
        self.successor_edges(node).iter().copied()
    }

    fn incoming_edges(&self, node: BlockId) -> impl Iterator<Item = EdgeId> + '_ {
        self.predecessor_edges(node).iter().copied()
    }

    fn edge_ref(&self, edge: EdgeId) -> EdgeRef<'_, BlockId, EdgeId, Edge<E>> {
        let value = self.edge(edge);
        EdgeRef::new(edge, value.source(), value.target(), value)
    }
}

impl<N, E> EdgeGraphView for DirectedGraph<N, E> {
    type EdgeId = EdgeId;
    type EdgeData = E;

    fn edge_slot_count(&self) -> usize {
        self.edge_slot_count()
    }

    fn edge_ids(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.edges().map(super::directed::DirectedEdge::id)
    }

    fn outgoing_edges(&self, node: NodeId) -> impl Iterator<Item = EdgeId> + '_ {
        self.outgoing_edges(node).iter().copied()
    }

    fn incoming_edges(&self, node: NodeId) -> impl Iterator<Item = EdgeId> + '_ {
        self.incoming_edges(node).iter().copied()
    }

    fn edge_ref(&self, edge: EdgeId) -> EdgeRef<'_, NodeId, EdgeId, E> {
        let value = self.edge(edge);
        EdgeRef::new(edge, value.source(), value.target(), value.payload())
    }
}

impl<G: EdgeGraphView> EdgeGraphView for Rooted<'_, G> {
    type EdgeId = G::EdgeId;
    type EdgeData = G::EdgeData;

    fn edge_slot_count(&self) -> usize {
        self.graph().edge_slot_count()
    }

    fn edge_ids(&self) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph().edge_ids()
    }

    fn outgoing_edges(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph().outgoing_edges(node)
    }

    fn incoming_edges(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph().incoming_edges(node)
    }

    fn edge_ref(
        &self,
        edge: Self::EdgeId,
    ) -> EdgeRef<'_, Self::NodeId, Self::EdgeId, Self::EdgeData> {
        self.graph().edge_ref(edge)
    }
}

impl<G: EdgeGraphView> EdgeGraphView for Reversed<'_, G> {
    type EdgeId = G::EdgeId;
    type EdgeData = G::EdgeData;

    fn edge_slot_count(&self) -> usize {
        self.graph().edge_slot_count()
    }

    fn edge_ids(&self) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph().edge_ids()
    }

    fn outgoing_edges(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph().incoming_edges(node)
    }

    fn incoming_edges(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph().outgoing_edges(node)
    }

    fn edge_ref(
        &self,
        edge: Self::EdgeId,
    ) -> EdgeRef<'_, Self::NodeId, Self::EdgeId, Self::EdgeData> {
        let value = self.graph().edge_ref(edge);
        EdgeRef::new(edge, value.target(), value.source(), value.data())
    }
}

/// Borrowed graph view containing only edges accepted by `P`.
///
/// Nodes and edge identities are never cloned or renumbered. A predicate sees
/// the stable identity and graph-specific edge data, so it can select normal,
/// exceptional, switch, continuation, provenance, or any consumer-defined
/// edge class. Rejected edges do not contribute adjacency or reachability.
#[derive(Debug, Clone, Copy)]
pub struct FilteredEdges<'g, G, P> {
    graph: &'g G,
    predicate: P,
}

impl<'g, G, P> FilteredEdges<'g, G, P> {
    /// Borrow `graph` through `predicate`.
    #[must_use]
    pub const fn new(graph: &'g G, predicate: P) -> Self {
        Self { graph, predicate }
    }

    /// The unfiltered graph.
    #[must_use]
    pub const fn graph(&self) -> &'g G {
        self.graph
    }
}

impl<G, P> FilteredEdges<'_, G, P>
where
    G: EdgeGraphView,
    P: Fn(G::EdgeId, &G::EdgeData) -> bool,
{
    fn accepts(&self, edge: G::EdgeId) -> bool {
        let value = self.graph.edge_ref(edge);
        (self.predicate)(edge, value.data())
    }
}

impl<G, P> DirectedGraphView for FilteredEdges<'_, G, P>
where
    G: EdgeGraphView,
    P: Fn(G::EdgeId, &G::EdgeData) -> bool,
{
    type NodeId = G::NodeId;

    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph
            .outgoing_edges(node)
            .filter(|&edge| self.accepts(edge))
            .map(|edge| self.graph.edge_ref(edge).target())
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph
            .incoming_edges(node)
            .filter(|&edge| self.accepts(edge))
            .map(|edge| self.graph.edge_ref(edge).source())
    }
}

impl<G, P> NodeGraphView for FilteredEdges<'_, G, P>
where
    G: EdgeGraphView + NodeGraphView,
    P: Fn(G::EdgeId, &G::EdgeData) -> bool,
{
    type NodeData = G::NodeData;

    fn node_ref(&self, node: Self::NodeId) -> &Self::NodeData {
        self.graph.node_ref(node)
    }
}

impl<G, P> EdgeGraphView for FilteredEdges<'_, G, P>
where
    G: EdgeGraphView,
    P: Fn(G::EdgeId, &G::EdgeData) -> bool,
{
    type EdgeId = G::EdgeId;
    type EdgeData = G::EdgeData;

    fn edge_slot_count(&self) -> usize {
        self.graph.edge_slot_count()
    }

    fn edge_ids(&self) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph.edge_ids().filter(|&edge| self.accepts(edge))
    }

    fn outgoing_edges(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph
            .outgoing_edges(node)
            .filter(|&edge| self.accepts(edge))
    }

    fn incoming_edges(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph
            .incoming_edges(node)
            .filter(|&edge| self.accepts(edge))
    }

    fn edge_ref(
        &self,
        edge: Self::EdgeId,
    ) -> EdgeRef<'_, Self::NodeId, Self::EdgeId, Self::EdgeData> {
        assert!(
            self.accepts(edge),
            "edge is excluded from the filtered view"
        );
        self.graph.edge_ref(edge)
    }
}

impl<G, P> RootedGraphView for FilteredEdges<'_, G, P>
where
    G: EdgeGraphView + RootedGraphView,
    P: Fn(G::EdgeId, &G::EdgeData) -> bool,
{
    fn root(&self) -> Self::NodeId {
        self.graph.root()
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec;

    use crate::{DirectedGraph, DirectedGraphView, DominatorTree, Rooted, RootedGraphView};

    use super::{EdgeGraphView, FilteredEdges};

    #[test]
    fn filtered_views_keep_parallel_edge_identity_without_cloning() {
        let mut graph = DirectedGraph::new();
        let entry = graph.add_node("entry");
        let normal = graph.add_node("normal");
        let handler = graph.add_node("handler");
        let first = graph.add_edge(entry, normal, "normal");
        let second = graph.add_edge(entry, normal, "case");
        let exception = graph.add_edge(entry, handler, "exception");

        let rooted = Rooted::new(&graph, entry);
        let normal_view = FilteredEdges::new(&rooted, |_: crate::EdgeId, kind: &&'static str| {
            *kind != "exception"
        });
        assert_eq!(normal_view.root(), entry);
        assert_eq!(
            normal_view.edge_ids().collect::<alloc::vec::Vec<_>>(),
            vec![first, second]
        );
        assert_eq!(
            normal_view
                .successors(entry)
                .collect::<alloc::vec::Vec<_>>(),
            vec![normal, normal]
        );
        assert_eq!(normal_view.edge_slot_count(), graph.edge_slot_count());
        assert_eq!(graph.edge(exception).payload(), &"exception");

        let dominators = DominatorTree::compute(&normal_view);
        assert!(dominators.dominates(entry, normal));
        assert!(!dominators.is_reachable(handler));
    }
}
