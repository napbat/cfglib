//! Breadth-first traversal events over dense graph views.

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::vec;
use core::ops::ControlFlow;

use crate::graph::traverse::{Adjacency, TraversalDirection, by_axis};
use crate::graph::view::{DenseId, GraphView};

/// An event of a [`breadth_first_events`] walk.
///
/// Breadth-first traversal has no nested unwind and therefore no `Finish`
/// event. Its edges instead split into the edges that first discover a node
/// and edges whose target was already discovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BfsEvent<N> {
    /// The node was reached for the first time, at this depth.
    Discover(N, usize),
    /// The edge from the first node to the second discovered the second.
    TreeEdge(N, N),
    /// The edge reached a node already discovered by the walk.
    ///
    /// This includes self-edges, edges back toward an earlier level, edges
    /// within a level, and additional edges to a node already reached by
    /// another parent.
    NonTreeEdge(N, N),
}

/// Walk `graph` breadth-first from `start`, reporting every discovery and
/// classifying every examined edge, and return the value the callback broke
/// with (or `None` when the walk ran to completion).
///
/// [`search`](crate::graph::search::search) answers questions about nodes. This function is
/// for consumers whose output includes the breadth-first tree or the edges
/// that did not enter that tree.
///
/// # Order
///
/// The event order is deterministic and part of the API. The walk first emits
/// [`Discover`](BfsEvent::Discover) for `start`. When expanding a node, it
/// examines successors in adjacency order. A previously unseen successor
/// produces [`TreeEdge`](BfsEvent::TreeEdge), immediately followed by its
/// `Discover`; an already discovered successor produces
/// [`NonTreeEdge`](BfsEvent::NonTreeEdge). Nodes are expanded level by level
/// in discovery order.
///
/// # Examples
///
/// ```
/// use core::ops::ControlFlow;
///
/// use cfglib::{BfsEvent, Graph, TraversalDirection, breadth_first_events};
///
/// let mut graph = Graph::<&str, ()>::new();
/// let root = graph.add_node("root");
/// let left = graph.add_node("left");
/// let right = graph.add_node("right");
/// let merge = graph.add_node("merge");
/// graph.add_edge(root, left, ());
/// graph.add_edge(root, right, ());
/// graph.add_edge(left, merge, ());
/// graph.add_edge(right, merge, ());
///
/// let mut non_tree = Vec::new();
/// let outcome = breadth_first_events::<_, ()>(
///     &graph,
///     root,
///     TraversalDirection::Outgoing,
///     |event| {
///         if let BfsEvent::NonTreeEdge(from, to) = event {
///             non_tree.push((graph[from], graph[to]));
///         }
///         ControlFlow::Continue(())
///     },
/// );
///
/// assert_eq!(outcome, None);
/// assert_eq!(non_tree, vec![("right", "merge")]);
/// ```
///
/// # Panics
///
/// Panics when `start` is not a node in `graph`.
#[must_use]
pub fn breadth_first_events<G: GraphView, B>(
    graph: &G,
    start: G::NodeId,
    direction: TraversalDirection,
    on_event: impl FnMut(BfsEvent<G::NodeId>) -> ControlFlow<B>,
) -> Option<B> {
    by_axis!(direction, breadth_first_events_from(graph, start, on_event))
}

fn breadth_first_events_from<G: GraphView, A: Adjacency, B>(
    axis: A,
    graph: &G,
    start: G::NodeId,
    mut on_event: impl FnMut(BfsEvent<G::NodeId>) -> ControlFlow<B>,
) -> Option<B> {
    assert!(
        start.index() < graph.node_bound(),
        "start node is out of range"
    );

    let mut discovered = vec![false; graph.node_bound()];
    let mut queue = VecDeque::new();
    discovered[start.index()] = true;
    emit_or_break!(on_event, BfsEvent::Discover(start, 0));
    queue.push_back((start, 0));

    while let Some((node, depth)) = queue.pop_front() {
        for successor in axis.neighbors(graph, node) {
            if discovered[successor.index()] {
                emit_or_break!(on_event, BfsEvent::NonTreeEdge(node, successor));
                continue;
            }

            discovered[successor.index()] = true;
            emit_or_break!(on_event, BfsEvent::TreeEdge(node, successor));
            emit_or_break!(on_event, BfsEvent::Discover(successor, depth + 1));
            queue.push_back((successor, depth + 1));
        }
    }

    None
}
