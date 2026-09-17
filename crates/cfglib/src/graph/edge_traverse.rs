//! Edge-aware traversals over any [`EdgeView`].
//!
//! Every walk here is generic: an owned [`Graph`](crate::Graph), a
//! [`Cfg`](crate::Cfg), a filtered or reversed view, and consumer-owned
//! storage all take the same entry points.

extern crate alloc;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

use super::edge_view::{EdgeRef, EdgeView};
use super::traverse::{Incoming, Outgoing, TraversalDirection, by_axis};
use super::view::DenseId;

trait EdgeAdjacency: Copy {
    fn edges<G: EdgeView>(self, graph: &G, node: G::NodeId)
    -> impl Iterator<Item = G::EdgeId> + '_;

    fn next<N: Copy, E: Copy, D: ?Sized>(self, edge: EdgeRef<'_, N, E, D>) -> N;

    fn previous<N: Copy, E: Copy, D: ?Sized>(self, edge: EdgeRef<'_, N, E, D>) -> N;
}

impl EdgeAdjacency for Outgoing {
    fn edges<G: EdgeView>(
        self,
        graph: &G,
        node: G::NodeId,
    ) -> impl Iterator<Item = G::EdgeId> + '_ {
        graph.outgoing(node)
    }

    fn next<N: Copy, E: Copy, D: ?Sized>(self, edge: EdgeRef<'_, N, E, D>) -> N {
        edge.target()
    }

    fn previous<N: Copy, E: Copy, D: ?Sized>(self, edge: EdgeRef<'_, N, E, D>) -> N {
        edge.source()
    }
}

impl EdgeAdjacency for Incoming {
    fn edges<G: EdgeView>(
        self,
        graph: &G,
        node: G::NodeId,
    ) -> impl Iterator<Item = G::EdgeId> + '_ {
        graph.incoming(node)
    }

    fn next<N: Copy, E: Copy, D: ?Sized>(self, edge: EdgeRef<'_, N, E, D>) -> N {
        edge.source()
    }

    fn previous<N: Copy, E: Copy, D: ?Sized>(self, edge: EdgeRef<'_, N, E, D>) -> N {
        edge.target()
    }
}

/// One traversed edge, with endpoints as exposed by the traversed view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeStep<N = crate::NodeId, E = crate::EdgeId> {
    /// The edge traversed.
    pub edge: E,
    /// The edge source in this view.
    pub source: N,
    /// The edge target in this view.
    pub target: N,
}

/// Breadth-first edge traversal over any edge-aware graph view.
#[must_use]
pub fn breadth_first_edges<G: EdgeView>(
    graph: &G,
    start: G::NodeId,
    direction: TraversalDirection,
) -> Vec<EdgeStep<G::NodeId, G::EdgeId>> {
    breadth_first_edges_with(graph, start, direction, None, |_| true)
}

/// Breadth-first edge traversal with a predicate and optional depth bound.
///
/// Every accepted edge leaving a reached node in `direction` is reported once
/// in adjacency order, including parallel edges and edges to visited nodes.
/// Rejected edges are neither reported nor traversed. This operates directly
/// on [`crate::FilteredEdges`] and other borrowed views without rebuilding a
/// graph.
///
/// # Panics
///
/// Panics when `start` is outside the view or the view violates its dense
/// node/edge identity contract.
#[must_use]
pub fn breadth_first_edges_with<G: EdgeView>(
    graph: &G,
    start: G::NodeId,
    direction: TraversalDirection,
    max_depth: Option<usize>,
    filter: impl FnMut(EdgeRef<'_, G::NodeId, G::EdgeId, G::EdgeData>) -> bool,
) -> Vec<EdgeStep<G::NodeId, G::EdgeId>> {
    by_axis!(
        direction,
        breadth_first_edges_from(graph, start, max_depth, filter)
    )
}

fn breadth_first_edges_from<G: EdgeView, A: EdgeAdjacency>(
    axis: A,
    graph: &G,
    start: G::NodeId,
    max_depth: Option<usize>,
    mut filter: impl FnMut(EdgeRef<'_, G::NodeId, G::EdgeId, G::EdgeData>) -> bool,
) -> Vec<EdgeStep<G::NodeId, G::EdgeId>> {
    assert!(
        start.index() < graph.node_bound(),
        "start node is out of range"
    );
    let mut steps = Vec::new();
    let mut seen_node = vec![false; graph.node_bound()];
    let mut queue = VecDeque::new();
    seen_node[start.index()] = true;
    queue.push_back((start, 0));

    while let Some((node, depth)) = queue.pop_front() {
        if max_depth.is_some_and(|limit| depth >= limit) {
            continue;
        }
        for edge_id in axis.edges(graph, node) {
            let edge = graph.edge(edge_id);
            if !filter(edge) {
                continue;
            }
            steps.push(EdgeStep {
                edge: edge_id,
                source: edge.source(),
                target: edge.target(),
            });
            let next = axis.next(edge);
            if !seen_node[next.index()] {
                seen_node[next.index()] = true;
                queue.push_back((next, depth + 1));
            }
        }
    }
    steps
}

/// Depth-first edge traversal over any edge-aware graph view.
#[must_use]
pub fn depth_first_edges<G: EdgeView>(
    graph: &G,
    start: G::NodeId,
    direction: TraversalDirection,
) -> Vec<EdgeStep<G::NodeId, G::EdgeId>> {
    depth_first_edges_with(graph, start, direction, None, |_| true)
}

/// Depth-first edge traversal with a predicate and optional depth bound.
///
/// Every accepted edge leaving a reached node in `direction` is reported once,
/// including parallel edges and edges to visited nodes. A tree edge is
/// reported immediately before the target's edges, preserving depth-first
/// adjacency order. Rejected edges are neither reported nor traversed.
///
/// # Panics
///
/// Panics when `start` is outside the view or the view violates its dense
/// node/edge identity contract.
#[must_use]
pub fn depth_first_edges_with<G: EdgeView>(
    graph: &G,
    start: G::NodeId,
    direction: TraversalDirection,
    max_depth: Option<usize>,
    filter: impl FnMut(EdgeRef<'_, G::NodeId, G::EdgeId, G::EdgeData>) -> bool,
) -> Vec<EdgeStep<G::NodeId, G::EdgeId>> {
    by_axis!(
        direction,
        depth_first_edges_from(graph, start, max_depth, filter)
    )
}

fn depth_first_edges_from<G: EdgeView, A: EdgeAdjacency>(
    axis: A,
    graph: &G,
    start: G::NodeId,
    max_depth: Option<usize>,
    mut filter: impl FnMut(EdgeRef<'_, G::NodeId, G::EdgeId, G::EdgeData>) -> bool,
) -> Vec<EdgeStep<G::NodeId, G::EdgeId>> {
    assert!(
        start.index() < graph.node_bound(),
        "start node is out of range"
    );
    let mut steps = Vec::new();
    let mut seen_node = vec![false; graph.node_bound()];
    let mut arena = Vec::new();
    append_adjacency(axis, graph, start, &mut arena);
    let mut stack = vec![EdgeDfsFrame {
        depth: 0,
        start: 0,
        cursor: 0,
    }];
    seen_node[start.index()] = true;

    while let Some(frame) = stack.last_mut() {
        if max_depth.is_some_and(|limit| frame.depth >= limit) || frame.cursor >= arena.len() {
            let Some(finished) = stack.pop() else {
                break;
            };
            arena.truncate(finished.start);
            continue;
        }

        let edge_id = arena[frame.cursor];
        let depth = frame.depth;
        frame.cursor += 1;
        let edge = graph.edge(edge_id);
        if !filter(edge) {
            continue;
        }
        steps.push(EdgeStep {
            edge: edge_id,
            source: edge.source(),
            target: edge.target(),
        });
        let next = axis.next(edge);
        if seen_node[next.index()] {
            continue;
        }

        seen_node[next.index()] = true;
        let start = arena.len();
        append_adjacency(axis, graph, next, &mut arena);
        stack.push(EdgeDfsFrame {
            depth: depth + 1,
            start,
            cursor: start,
        });
    }

    steps
}

#[derive(Debug, Clone, Copy)]
struct EdgeDfsFrame {
    depth: usize,
    start: usize,
    cursor: usize,
}

fn append_adjacency<G: EdgeView, A: EdgeAdjacency>(
    axis: A,
    graph: &G,
    node: G::NodeId,
    arena: &mut Vec<G::EdgeId>,
) {
    arena.extend(axis.edges(graph, node));
}

/// The edges of one shortest path through an edge-aware view.
///
/// # Panics
///
/// Panics when either endpoint is outside the view or the view violates its
/// dense node/edge identity contract.
#[must_use]
pub fn shortest_path_edges<G: EdgeView>(
    graph: &G,
    from: G::NodeId,
    to: G::NodeId,
    direction: TraversalDirection,
) -> Option<Vec<G::EdgeId>> {
    by_axis!(direction, shortest_path_edges_from(graph, from, to))
}

fn shortest_path_edges_from<G: EdgeView, A: EdgeAdjacency>(
    axis: A,
    graph: &G,
    from: G::NodeId,
    to: G::NodeId,
) -> Option<Vec<G::EdgeId>> {
    assert!(
        from.index() < graph.node_bound(),
        "source node is out of range"
    );
    assert!(
        to.index() < graph.node_bound(),
        "target node is out of range"
    );
    if from == to {
        return Some(Vec::new());
    }
    let mut parent_edge = vec![G::EdgeId::from_index(0); graph.node_bound()];
    let mut seen = vec![false; graph.node_bound()];
    let mut queue = VecDeque::new();
    seen[from.index()] = true;
    queue.push_back(from);

    'search: while let Some(node) = queue.pop_front() {
        for edge_id in axis.edges(graph, node) {
            let edge = graph.edge(edge_id);
            let next = axis.next(edge);
            if seen[next.index()] {
                continue;
            }
            seen[next.index()] = true;
            parent_edge[next.index()] = edge_id;
            if next == to {
                break 'search;
            }
            queue.push_back(next);
        }
    }

    if !seen[to.index()] {
        return None;
    }
    let mut path = Vec::new();
    let mut current = to;
    while current != from {
        let edge_id = parent_edge[current.index()];
        path.push(edge_id);
        let edge = graph.edge(edge_id);
        current = axis.previous(edge);
    }
    path.reverse();
    Some(path)
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

    #[test]
    fn edge_bfs_reports_parallel_and_cycle_edges_once() {
        let (graph, [a, _, c]) = fixture();
        let steps = breadth_first_edges(&graph, a, TraversalDirection::Outgoing);
        assert_eq!(steps.len(), 4);
        let payloads: Vec<_> = steps
            .iter()
            .map(|step| *graph.edge(step.edge).payload())
            .collect();
        assert_eq!(payloads, ["x", "z", "y", "cycle"]);

        let backward = breadth_first_edges(&graph, c, TraversalDirection::Incoming);
        let backward_payloads: Vec<_> = backward
            .iter()
            .map(|step| *graph.edge(step.edge).payload())
            .collect();
        assert_eq!(backward_payloads, ["y", "x", "z", "cycle"]);
    }

    #[test]
    fn filtered_view_walk_never_rebuilds_or_renumbers() {
        let (graph, [a, _, c]) = fixture();
        let rooted = Rooted::new(&graph, a);
        let filtered = FilteredEdges::new(&rooted, |_, payload: &&str| *payload != "z");
        let steps = breadth_first_edges(&filtered, a, TraversalDirection::Outgoing);
        let payloads: Vec<_> = steps
            .iter()
            .map(|step| *filtered.edge(step.edge).data())
            .collect();
        assert_eq!(payloads, ["x", "y", "cycle"]);
        assert_eq!(
            shortest_path_edges(&filtered, a, c, TraversalDirection::Outgoing)
                .unwrap()
                .len(),
            2
        );
        let incoming = shortest_path_edges(&filtered, c, a, TraversalDirection::Incoming).unwrap();
        assert_eq!(incoming.len(), 2);
        assert_eq!(*filtered.edge(incoming[0]).data(), "y");
        assert_eq!(*filtered.edge(incoming[1]).data(), "x");
    }

    #[test]
    fn depth_and_predicate_filters_remain_compatible() {
        let (graph, [a, _, _]) = fixture();
        let steps =
            breadth_first_edges_with(&graph, a, TraversalDirection::Outgoing, Some(1), |edge| {
                *edge.data() != "z"
            });
        assert_eq!(steps.len(), 1);
        assert_eq!(*graph.edge(steps[0].edge).payload(), "x");
    }

    #[test]
    fn depth_first_edges_descend_before_examining_the_next_sibling() {
        let (graph, [a, _, _]) = fixture();
        let steps = depth_first_edges(&graph, a, TraversalDirection::Outgoing);
        let payloads: Vec<_> = steps
            .iter()
            .map(|step| *graph.edge(step.edge).payload())
            .collect();
        assert_eq!(payloads, ["x", "y", "cycle", "z"]);
    }

    #[test]
    fn depth_first_filters_and_bounds_match_breadth_first_contracts() {
        let (graph, [a, _, _]) = fixture();
        let steps =
            depth_first_edges_with(&graph, a, TraversalDirection::Outgoing, Some(1), |edge| {
                *edge.data() != "z"
            });
        assert_eq!(steps.len(), 1);
        assert_eq!(*graph.edge(steps[0].edge).payload(), "x");
    }
}
