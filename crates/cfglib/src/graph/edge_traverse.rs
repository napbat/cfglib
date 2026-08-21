//! Edge-aware traversals over [`DirectedGraph`].
//!
//! The node-only view traversals in [`traverse`](super::traverse) drop
//! parallel edges and carry no edge identities. Provenance-carrying graphs
//! (value flow, call graphs, typed relations) need the opposite: every
//! distinct edge visited once, with its identity and endpoints, and walks
//! that can filter on the edge payload or bound their depth. These
//! functions provide that, over the owned [`DirectedGraph`] storage where
//! edges exist.

extern crate alloc;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

use super::directed::{DirectedEdge, DirectedGraph, EdgeId, NodeId};
use super::traverse::TraversalDirection;

/// One traversed edge, with its real (direction-independent) endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeStep {
    /// The edge traversed.
    pub edge: EdgeId,
    /// The edge's source node (as stored, regardless of walk direction).
    pub source: NodeId,
    /// The edge's target node (as stored, regardless of walk direction).
    pub target: NodeId,
}

/// Breadth-first edge traversal from `start`.
///
/// Every edge **leaving a reached node** (in the walk direction) is
/// reported exactly once, in adjacency order — including parallel edges
/// and edges into already-visited nodes, which node-only traversal cannot
/// report. Edges entering the reached region from elsewhere are not walked
/// and not reported. Each node is expanded once, so the walk terminates on
/// cyclic graphs.
#[must_use]
pub fn breadth_first_edges<N, E>(
    graph: &DirectedGraph<N, E>,
    start: NodeId,
    direction: TraversalDirection,
) -> Vec<EdgeStep> {
    walk_edges(graph, start, direction, None, |_| true)
}

/// Breadth-first edge traversal with an edge filter and an optional depth
/// bound.
///
/// `filter` decides whether an edge is traversed at all: a rejected edge
/// produces no step and does not visit its far endpoint. The filter should be
/// pure so its decision depends only on the edge.
/// `max_depth` bounds the walk in hops from `start`: edges leaving a node
/// `max_depth` hops away are not taken (`Some(0)` reports nothing).
#[must_use]
pub fn walk_edges<N, E>(
    graph: &DirectedGraph<N, E>,
    start: NodeId,
    direction: TraversalDirection,
    max_depth: Option<usize>,
    mut filter: impl FnMut(&DirectedEdge<E>) -> bool,
) -> Vec<EdgeStep> {
    let forward = matches!(direction, TraversalDirection::Outgoing);
    let mut steps = Vec::new();
    let mut seen_node = vec![false; graph.node_count()];
    let mut queue: VecDeque<(NodeId, usize)> = VecDeque::new();
    seen_node[start.index()] = true;
    queue.push_back((start, 0));

    while let Some((node, depth)) = queue.pop_front() {
        if max_depth.is_some_and(|limit| depth >= limit) {
            continue;
        }
        let adjacency = if forward {
            graph.outgoing_edges(node)
        } else {
            graph.incoming_edges(node)
        };
        for &edge_id in adjacency {
            let edge = graph.edge(edge_id);
            if !filter(edge) {
                continue;
            }
            steps.push(EdgeStep {
                edge: edge_id,
                source: edge.source(),
                target: edge.target(),
            });
            let next = if forward {
                edge.target()
            } else {
                edge.source()
            };
            if !seen_node[next.index()] {
                seen_node[next.index()] = true;
                queue.push_back((next, depth + 1));
            }
        }
    }
    steps
}

/// The edges of one shortest path from `from` to `to`, or `None` when `to`
/// is unreachable. `from == to` yields `Some(vec![])`.
///
/// The node-yielding [`shortest_path`](super::traverse::shortest_path)
/// loses which of several parallel edges realized each hop; witness-path
/// consumers (value-flow provenance) need the edges themselves.
#[must_use]
pub fn shortest_path_edges<N, E>(
    graph: &DirectedGraph<N, E>,
    from: NodeId,
    to: NodeId,
    direction: TraversalDirection,
) -> Option<Vec<EdgeId>> {
    if from == to {
        return Some(Vec::new());
    }
    let forward = matches!(direction, TraversalDirection::Outgoing);
    // `seen` guards every read. A plain edge id halves the parent table on
    // targets where `Option<EdgeId>` cannot use a niche.
    let mut parent_edge = vec![EdgeId::from_raw(0); graph.node_count()];
    let mut seen = vec![false; graph.node_count()];
    let mut queue = VecDeque::new();
    seen[from.index()] = true;
    queue.push_back(from);

    'search: while let Some(node) = queue.pop_front() {
        let adjacency = if forward {
            graph.outgoing_edges(node)
        } else {
            graph.incoming_edges(node)
        };
        for &edge_id in adjacency {
            let edge = graph.edge(edge_id);
            let next = if forward {
                edge.target()
            } else {
                edge.source()
            };
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
        current = if forward {
            edge.source()
        } else {
            edge.target()
        };
    }
    path.reverse();
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a =x=> b =y=> c, plus a parallel a =z=> b and a cycle edge c => a.
    fn diamondish() -> (DirectedGraph<&'static str, &'static str>, [NodeId; 3]) {
        let mut graph = DirectedGraph::new();
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
        let (graph, [a, b, c]) = diamondish();
        let steps = breadth_first_edges(&graph, a, TraversalDirection::Outgoing);
        // All four edges appear exactly once; parallel a->b twice as
        // distinct edges; the cycle edge back into the visited region too.
        assert_eq!(steps.len(), 4);
        let payloads: Vec<&str> = steps
            .iter()
            .map(|s| *graph.edge(s.edge).payload())
            .collect();
        assert_eq!(payloads, ["x", "z", "y", "cycle"]);
        assert_eq!(steps[0].source, a);
        assert_eq!(steps[3].source, c);
        assert_eq!(steps[3].target, a);

        // Backward from c sees y then x/z then the cycle edge from a's side.
        let back = breadth_first_edges(&graph, c, TraversalDirection::Incoming);
        assert_eq!(back.len(), 4);
        assert_eq!(back[0].edge, steps[2].edge);
        let _ = b;
    }

    #[test]
    fn filtered_and_depth_bounded_walks() {
        let (graph, [a, _, _]) = diamondish();
        // Filter out the parallel "z" edge entirely.
        let steps = walk_edges(&graph, a, TraversalDirection::Outgoing, None, |edge| {
            *edge.payload() != "z"
        });
        let payloads: Vec<&str> = steps
            .iter()
            .map(|s| *graph.edge(s.edge).payload())
            .collect();
        assert_eq!(payloads, ["x", "y", "cycle"]);

        // Depth 1: only edges leaving the start node.
        let close = walk_edges(&graph, a, TraversalDirection::Outgoing, Some(1), |_| true);
        let payloads: Vec<&str> = close
            .iter()
            .map(|s| *graph.edge(s.edge).payload())
            .collect();
        assert_eq!(payloads, ["x", "z"]);
        assert_eq!(
            walk_edges(&graph, a, TraversalDirection::Outgoing, Some(0), |_| true).len(),
            0
        );
    }

    #[test]
    fn shortest_path_edges_returns_a_witness() {
        let (graph, [a, b, c]) = diamondish();
        let path = shortest_path_edges(&graph, a, c, TraversalDirection::Outgoing).unwrap();
        assert_eq!(path.len(), 2);
        assert_eq!(graph.edge(path[0]).source(), a);
        assert_eq!(graph.edge(path[0]).target(), b);
        assert_eq!(graph.edge(path[1]).target(), c);

        let incoming = shortest_path_edges(&graph, c, a, TraversalDirection::Incoming).unwrap();
        assert_eq!(incoming.len(), 2);
        assert_eq!(graph.edge(incoming[0]).source(), b);
        assert_eq!(graph.edge(incoming[0]).target(), c);
        assert_eq!(graph.edge(incoming[1]).source(), a);
        assert_eq!(graph.edge(incoming[1]).target(), b);

        assert_eq!(
            shortest_path_edges(&graph, a, a, TraversalDirection::Outgoing),
            Some(Vec::new())
        );

        // A node unreachable in the chosen direction.
        let mut disconnected: DirectedGraph<(), ()> = DirectedGraph::new();
        let lone = disconnected.add_node(());
        let other = disconnected.add_node(());
        assert_eq!(
            shortest_path_edges(&disconnected, lone, other, TraversalDirection::Outgoing),
            None
        );
    }
}
