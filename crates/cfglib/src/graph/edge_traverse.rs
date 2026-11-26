//! Edge-aware traversals over any [`EdgeView`].
//!
//! Every walk here is generic: an owned [`Graph`](crate::Graph), a
//! [`Cfg`](crate::Cfg), a [`FilteredEdges`](crate::FilteredEdges),
//! [`Induced`](crate::Induced) or reversed view, and consumer-owned storage
//! all take the same entry points. The walks report edge identities, so
//! parallel edges stay distinct; filter the edges a walk may take with
//! [`FilteredEdges`](crate::FilteredEdges). They are the edge instantiations
//! of the walk cores of [`traverse`](super::traverse).

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use super::edge_view::{EdgeRef, EdgeView};
use super::search::Visit;
use super::traverse::{
    Adjacency, DepthFirstStack, DepthFirstVisitor, Incoming, Outgoing, ShortestPathScratch,
    StepKind, Steps, breadth_first_core, by_axis, depth_first_core, edge_path_in,
};
use super::view::{DenseId, Direction, assert_in_bound};

/// An adjacency axis over edges: which edges a node exposes, and which end of
/// an exposed edge is the neighbor.
///
/// The edge-aware counterpart of [`Adjacency`], implemented on the same
/// zero-sized axes so one core serves both directions.
pub(crate) trait EdgeAdjacency: Adjacency {
    fn edges<G: EdgeView>(self, graph: &G, node: G::NodeId)
    -> impl Iterator<Item = G::EdgeId> + '_;

    fn next<N: Copy, E: Copy, D: ?Sized>(self, edge: EdgeRef<'_, N, E, D>) -> N;

    fn previous<N: Copy, E: Copy, D: ?Sized>(self, edge: EdgeRef<'_, N, E, D>) -> N;
}

impl EdgeAdjacency for Outgoing {
    #[inline]
    fn edges<G: EdgeView>(
        self,
        graph: &G,
        node: G::NodeId,
    ) -> impl Iterator<Item = G::EdgeId> + '_ {
        graph.outgoing(node)
    }

    #[inline]
    fn next<N: Copy, E: Copy, D: ?Sized>(self, edge: EdgeRef<'_, N, E, D>) -> N {
        edge.target()
    }

    #[inline]
    fn previous<N: Copy, E: Copy, D: ?Sized>(self, edge: EdgeRef<'_, N, E, D>) -> N {
        edge.source()
    }
}

impl EdgeAdjacency for Incoming {
    #[inline]
    fn edges<G: EdgeView>(
        self,
        graph: &G,
        node: G::NodeId,
    ) -> impl Iterator<Item = G::EdgeId> + '_ {
        graph.incoming(node)
    }

    #[inline]
    fn next<N: Copy, E: Copy, D: ?Sized>(self, edge: EdgeRef<'_, N, E, D>) -> N {
        edge.source()
    }

    #[inline]
    fn previous<N: Copy, E: Copy, D: ?Sized>(self, edge: EdgeRef<'_, N, E, D>) -> N {
        edge.target()
    }
}

/// The steps of an edge walk along an axis: a step is an edge, and it leads
/// to the edge's far end along the axis.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Edges<A>(pub(crate) A);

impl<G: EdgeView, A: EdgeAdjacency> Steps<G> for Edges<A> {
    type Node = G::NodeId;
    type Step = G::EdgeId;

    #[inline]
    fn steps(self, graph: &G, node: G::NodeId) -> impl Iterator<Item = G::EdgeId> + '_ {
        self.0.edges(graph, node)
    }

    #[inline]
    fn target(self, graph: &G, step: G::EdgeId) -> G::NodeId {
        self.0.next(graph.edge(step))
    }

    #[inline]
    fn index(node: G::NodeId) -> usize {
        node.index()
    }
}

/// Breadth-first edge traversal from `start` along `direction`.
///
/// Every edge leaving a reached node along `direction` is reported once, in
/// adjacency order, including parallel edges and edges to visited nodes.
/// Nodes are expanded in breadth-first order. Restrict the edges the walk
/// takes with [`FilteredEdges`](crate::FilteredEdges): a rejected edge is
/// neither reported nor traversed.
///
/// # Examples
///
/// ```
/// use cfglib::{Direction, Graph, breadth_first_edges};
///
/// let mut graph = Graph::<(), &str>::new();
/// let [a, b, c] = [(); 3].map(|()| graph.add_node(()));
/// let first = graph.add_edge(a, b, "first");
/// let parallel = graph.add_edge(a, b, "parallel");
/// let onward = graph.add_edge(b, c, "onward");
///
/// assert_eq!(
///     breadth_first_edges(&graph, a, Direction::Forward),
///     [first, parallel, onward]
/// );
/// ```
///
/// # Panics
///
/// Panics when `start` is outside the view or the view violates its dense
/// node/edge identity contract.
#[must_use]
pub fn breadth_first_edges<G: EdgeView>(
    graph: &G,
    start: G::NodeId,
    direction: Direction,
) -> Vec<G::EdgeId> {
    by_axis!(direction, breadth_first_edge_walk(graph, start))
}

fn breadth_first_edge_walk<G: EdgeView, A: EdgeAdjacency>(
    axis: A,
    graph: &G,
    start: G::NodeId,
) -> Vec<G::EdgeId> {
    assert_in_bound(graph, start, "start node");
    let mut edges = Vec::new();
    let mut visited = vec![false; graph.node_bound()];
    let mut queue = vec![start];
    visited[start.index()] = true;
    breadth_first_core::<_, _, ()>(
        Edges(axis),
        graph,
        &mut queue,
        |_, _| ControlFlow::Continue(Visit::Descend),
        |_, edge, to: G::NodeId, _| {
            edges.push(edge);
            !core::mem::replace(&mut visited[to.index()], true)
        },
    );
    edges
}

/// Depth-first edge traversal from `start` along `direction`.
///
/// Every edge leaving a reached node along `direction` is reported once,
/// including parallel edges and edges to visited nodes. A tree edge is
/// reported immediately before the edges of its target, preserving
/// depth-first adjacency order. Restrict the edges the walk takes with
/// [`FilteredEdges`](crate::FilteredEdges).
///
/// # Panics
///
/// Panics when `start` is outside the view or the view violates its dense
/// node/edge identity contract.
#[must_use]
pub fn depth_first_edges<G: EdgeView>(
    graph: &G,
    start: G::NodeId,
    direction: Direction,
) -> Vec<G::EdgeId> {
    by_axis!(direction, depth_first_edge_walk(graph, start))
}

/// The depth-first edge walk's visitor: every examined edge, in order.
struct EdgeOrder<E>(Vec<E>);

impl<N, E> DepthFirstVisitor<N, E, ()> for EdgeOrder<E> {
    #[inline]
    fn discover(&mut self, _node: N, _depth: usize) -> ControlFlow<(), Visit> {
        ControlFlow::Continue(Visit::Descend)
    }

    #[inline]
    fn step(&mut self, _from: N, edge: E, _to: N, _kind: StepKind) -> ControlFlow<()> {
        self.0.push(edge);
        ControlFlow::Continue(())
    }
}

fn depth_first_edge_walk<G: EdgeView, A: EdgeAdjacency>(
    axis: A,
    graph: &G,
    start: G::NodeId,
) -> Vec<G::EdgeId> {
    assert_in_bound(graph, start, "start node");
    let mut visited = vec![false; graph.node_bound()];
    let mut order = EdgeOrder(Vec::new());
    depth_first_core(
        Edges(axis),
        graph,
        [start],
        visited.as_mut_slice(),
        &mut DepthFirstStack::default(),
        &mut order,
    );
    order.0
}

/// The edges of a shortest path from `from` to `to` along `direction` edges.
///
/// The edge-identity form of [`shortest_path`](crate::shortest_path), with
/// the same bidirectional search, cost, tie-break and adjacency requirement,
/// over [`outgoing`](EdgeView::outgoing) and [`incoming`](EdgeView::incoming).
/// `from == to` is the empty path, and parallel edges stay distinct, so the
/// witness names the edge it took. Each call allocates its own
/// [`ShortestPathScratch`]; [`shortest_path_edges_in`] reuses one.
///
/// # Panics
///
/// Panics when either endpoint is outside the view, when the node or edge
/// bound exceeds `u32::MAX`, or when the view violates its dense node/edge
/// identity contract.
///
/// # Examples
///
/// ```
/// use cfglib::{Direction, Graph, shortest_path_edges};
///
/// let mut graph = Graph::<(), &str>::new();
/// let [a, b, c] = [(); 3].map(|()| graph.add_node(()));
/// let first = graph.add_edge(a, b, "first");
/// graph.add_edge(a, b, "parallel");
/// let second = graph.add_edge(b, c, "second");
///
/// assert_eq!(
///     shortest_path_edges(&graph, a, c, Direction::Forward),
///     Some(vec![first, second]),
/// );
/// ```
#[must_use]
pub fn shortest_path_edges<G: EdgeView>(
    graph: &G,
    from: G::NodeId,
    to: G::NodeId,
    direction: Direction,
) -> Option<Vec<G::EdgeId>> {
    edge_path_in(
        &mut ShortestPathScratch::default(),
        graph,
        from,
        to,
        direction,
    )
}

/// [`shortest_path_edges`] over a caller-owned [`ShortestPathScratch`], for a
/// pass that asks many point-to-point questions.
///
/// # Panics
///
/// Panics when either endpoint is outside the view, when the node or edge
/// bound exceeds `u32::MAX`, or when the view violates its dense node/edge
/// identity contract.
#[must_use]
pub fn shortest_path_edges_in<G: EdgeView>(
    scratch: &mut ShortestPathScratch,
    graph: &G,
    from: G::NodeId,
    to: G::NodeId,
    direction: Direction,
) -> Option<Vec<G::EdgeId>> {
    edge_path_in(scratch, graph, from, to, direction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FilteredEdges, Graph, NodeId, Rooted};

    fn fixture() -> (Graph<&'static str, &'static str>, [NodeId; 3]) {
        let mut graph = Graph::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        graph.add_edge(a, b, "x");
        graph.add_edge(b, c, "y");
        graph.add_edge(a, b, "z");
        graph.add_edge(c, a, "cycle");
        (graph, [a, b, c])
    }

    fn payloads<G: EdgeView<EdgeData = &'static str>>(
        graph: &G,
        edges: &[G::EdgeId],
    ) -> Vec<&'static str> {
        edges.iter().map(|&edge| *graph.edge(edge).data()).collect()
    }

    #[test]
    fn edge_bfs_reports_parallel_and_cycle_edges_once() {
        let (graph, [a, _, c]) = fixture();
        let edges = breadth_first_edges(&graph, a, Direction::Forward);
        assert_eq!(payloads(&graph, &edges), ["x", "z", "y", "cycle"]);

        let backward = breadth_first_edges(&graph, c, Direction::Backward);
        assert_eq!(payloads(&graph, &backward), ["y", "x", "z", "cycle"]);
    }

    #[test]
    fn filtered_view_walk_never_rebuilds_or_renumbers() {
        let (graph, [a, _, c]) = fixture();
        let rooted = Rooted::new(&graph, a);
        let filtered = FilteredEdges::new(&rooted, |_, payload: &&str| *payload != "z");
        let edges = breadth_first_edges(&filtered, a, Direction::Forward);
        assert_eq!(payloads(&filtered, &edges), ["x", "y", "cycle"]);
        assert_eq!(
            shortest_path_edges(&filtered, a, c, Direction::Forward)
                .unwrap()
                .len(),
            2
        );
        let incoming = shortest_path_edges(&filtered, c, a, Direction::Backward).unwrap();
        assert_eq!(payloads(&filtered, &incoming), ["y", "x"]);
    }

    #[test]
    fn depth_first_edges_descend_before_examining_the_next_sibling() {
        let (graph, [a, _, _]) = fixture();
        let edges = depth_first_edges(&graph, a, Direction::Forward);
        assert_eq!(payloads(&graph, &edges), ["x", "y", "cycle", "z"]);
    }

    #[test]
    fn a_filter_prunes_both_walks_alike() {
        let (graph, [a, _, _]) = fixture();
        let filtered = FilteredEdges::new(&graph, |_, payload: &&str| *payload != "y");
        let breadth = breadth_first_edges(&filtered, a, Direction::Forward);
        let depth = depth_first_edges(&filtered, a, Direction::Forward);
        assert_eq!(payloads(&graph, &breadth), ["x", "z"]);
        assert_eq!(payloads(&graph, &depth), ["x", "z"]);
    }
}
