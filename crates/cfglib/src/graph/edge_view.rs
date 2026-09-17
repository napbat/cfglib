//! Edge-aware graph views and zero-copy edge filtering.

use super::view::{DenseId, GraphView, NodeView, Reversed, Rooted, RootedView};

/// One borrowed edge exposed by an [`EdgeView`].
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
/// This companion to [`GraphView`] is opt-in for stores that retain explicit
/// edges. Node-only algorithms keep depending on the smaller trait;
/// edge-sensitive traversals, filters, validation, and dataflow use this one.
///
/// # Contract
///
/// - Every identity yielded by [`edge_ids`](Self::edge_ids), by
///   [`outgoing`](Self::outgoing), or by [`incoming`](Self::incoming) has an
///   [`index`](DenseId::index) below [`edge_bound`](Self::edge_bound).
/// - [`edge_ids`](Self::edge_ids) yields every live edge exactly once. The
///   stores in this crate yield ascending order, which is also insertion
///   order.
/// - The endpoints of every yielded edge are live nodes of the same view.
pub trait EdgeView: GraphView {
    /// Stable edge identity used by this view.
    type EdgeId: DenseId;

    /// Data exposed for each live edge.
    type EdgeData: ?Sized;

    /// An exclusive upper bound on every edge index this view yields.
    ///
    /// This is the correct size for an edge-indexed side table.
    fn edge_bound(&self) -> usize;

    /// Iterate over every live edge identity exactly once.
    fn edge_ids(&self) -> impl Iterator<Item = Self::EdgeId> + '_;

    /// Iterate over outgoing edge identities in adjacency order.
    fn outgoing(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_;

    /// Iterate over incoming edge identities in adjacency order.
    fn incoming(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_;

    /// Borrow one live edge.
    ///
    /// # Panics
    ///
    /// Panics when `edge` is out of range or names a removed edge.
    fn edge(&self, edge: Self::EdgeId) -> EdgeRef<'_, Self::NodeId, Self::EdgeId, Self::EdgeData>;
}

impl<G: EdgeView> EdgeView for Rooted<'_, G> {
    type EdgeId = G::EdgeId;
    type EdgeData = G::EdgeData;

    fn edge_bound(&self) -> usize {
        self.graph().edge_bound()
    }

    fn edge_ids(&self) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph().edge_ids()
    }

    fn outgoing(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph().outgoing(node)
    }

    fn incoming(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph().incoming(node)
    }

    fn edge(&self, edge: Self::EdgeId) -> EdgeRef<'_, Self::NodeId, Self::EdgeId, Self::EdgeData> {
        self.graph().edge(edge)
    }
}

impl<G: EdgeView> EdgeView for Reversed<'_, G> {
    type EdgeId = G::EdgeId;
    type EdgeData = G::EdgeData;

    fn edge_bound(&self) -> usize {
        self.graph().edge_bound()
    }

    fn edge_ids(&self) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph().edge_ids()
    }

    fn outgoing(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph().incoming(node)
    }

    fn incoming(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph().outgoing(node)
    }

    fn edge(&self, edge: Self::EdgeId) -> EdgeRef<'_, Self::NodeId, Self::EdgeId, Self::EdgeData> {
        let value = self.graph().edge(edge);
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
    G: EdgeView,
    P: Fn(G::EdgeId, &G::EdgeData) -> bool,
{
    fn accepts(&self, edge: G::EdgeId) -> bool {
        let value = self.graph.edge(edge);
        (self.predicate)(edge, value.data())
    }
}

impl<G, P> GraphView for FilteredEdges<'_, G, P>
where
    G: EdgeView,
    P: Fn(G::EdgeId, &G::EdgeData) -> bool,
{
    type NodeId = G::NodeId;

    fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.node_ids()
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph
            .outgoing(node)
            .filter(|&edge| self.accepts(edge))
            .map(|edge| self.graph.edge(edge).target())
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph
            .incoming(node)
            .filter(|&edge| self.accepts(edge))
            .map(|edge| self.graph.edge(edge).source())
    }
}

impl<G, P> NodeView for FilteredEdges<'_, G, P>
where
    G: EdgeView + NodeView,
    P: Fn(G::EdgeId, &G::EdgeData) -> bool,
{
    type NodeData = G::NodeData;

    fn node(&self, node: Self::NodeId) -> &Self::NodeData {
        self.graph.node(node)
    }
}

impl<G, P> EdgeView for FilteredEdges<'_, G, P>
where
    G: EdgeView,
    P: Fn(G::EdgeId, &G::EdgeData) -> bool,
{
    type EdgeId = G::EdgeId;
    type EdgeData = G::EdgeData;

    fn edge_bound(&self) -> usize {
        self.graph.edge_bound()
    }

    fn edge_ids(&self) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph.edge_ids().filter(|&edge| self.accepts(edge))
    }

    fn outgoing(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph.outgoing(node).filter(|&edge| self.accepts(edge))
    }

    fn incoming(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph.incoming(node).filter(|&edge| self.accepts(edge))
    }

    fn edge(&self, edge: Self::EdgeId) -> EdgeRef<'_, Self::NodeId, Self::EdgeId, Self::EdgeData> {
        assert!(
            self.accepts(edge),
            "edge is excluded from the filtered view"
        );
        self.graph.edge(edge)
    }
}

impl<G, P> RootedView for FilteredEdges<'_, G, P>
where
    G: EdgeView + RootedView,
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
    use alloc::vec::Vec;

    use crate::{DominatorTree, EdgeId, Graph, GraphView, Rooted, RootedView};

    use super::{EdgeView, FilteredEdges};

    #[test]
    fn filtered_views_keep_parallel_edge_identity_without_cloning() {
        let mut graph = Graph::new();
        let entry = graph.add_node("entry");
        let normal = graph.add_node("normal");
        let handler = graph.add_node("handler");
        let first = graph.add_edge(entry, normal, "normal");
        let second = graph.add_edge(entry, normal, "case");
        let exception = graph.add_edge(entry, handler, "exception");

        let rooted = Rooted::new(&graph, entry);
        let normal_view = FilteredEdges::new(&rooted, |_: EdgeId, kind: &&'static str| {
            *kind != "exception"
        });
        assert_eq!(normal_view.root(), entry);
        assert_eq!(
            normal_view.edge_ids().collect::<Vec<_>>(),
            vec![first, second]
        );
        assert_eq!(
            normal_view.successors(entry).collect::<Vec<_>>(),
            vec![normal, normal]
        );
        assert_eq!(normal_view.edge_bound(), graph.edge_bound());
        assert_eq!(graph.edge(exception).payload(), &"exception");

        let dominators = DominatorTree::compute(&normal_view);
        assert!(dominators.dominates(entry, normal));
        assert!(!dominators.is_reachable(handler));
    }
}
