//! Depth-first traversal events over dense graph views.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use crate::graph::traverse::{Adjacency, TraversalDirection, by_axis};
use crate::graph::view::{DenseId, GraphView};

/// An event of a [`depth_first_events`] walk.
///
/// Together these are the classic tri-color depth-first classification: the
/// edges of the walk split into the tree it builds, the back edges that close
/// a cycle, and the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DfsEvent<N> {
    /// The node was reached for the first time, at this depth.
    Discover(N, usize),
    /// The edge from the first node to the second discovered the second —
    /// an edge of the depth-first tree.
    TreeEdge(N, N),
    /// The edge from the first node to the second closed a cycle: the target
    /// is an ancestor on the current path (a self-edge is a back edge).
    BackEdge(N, N),
    /// The edge from the first node to the second reached an already
    /// finished node — a forward edge to a descendant, or a cross edge into
    /// a sibling subtree.
    ///
    /// The two are merged deliberately: telling them apart needs discovery
    /// timestamps, which a consumer that cares can record from
    /// [`Discover`](DfsEvent::Discover) itself.
    ForwardOrCross(N, N),
    /// Every edge out of the node has been classified and its subtree is
    /// complete.
    Finish(N),
}

/// One frame of the explicit depth-first stack.
///
/// A frame's successors live in the walk's single arena rather than in a `Vec`
/// of its own: frames are pushed and popped strictly last in, first out, so
/// the frame on top of the stack always owns the arena's tail and a pop
/// truncates the arena back to where that frame's successors began. One
/// allocation for the walk, instead of one per expanded node.
struct DfsFrame<N> {
    node: N,
    depth: usize,
    /// Where this frame's successors start in the arena; the pop truncates to
    /// it.
    start: usize,
    /// The next successor to examine, as an arena index. The frame's
    /// successors are exhausted when it reaches the arena's length, since the
    /// top frame's region runs to the end.
    cursor: usize,
}

/// The tri-color state of a node during a [`depth_first_events`] walk.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Color {
    /// Undiscovered.
    White,
    /// Discovered, on the current path, not yet finished.
    Gray,
    /// Finished.
    Black,
}

/// Walk `graph` depth-first from `start`, reporting every discovery, edge
/// classification, and finish, and returning the value the callback broke
/// with (or `None` when the walk ran to completion).
///
/// [`search`](crate::graph::search::search) answers questions about nodes; this answers questions whose
/// output order *is* the traversal — reporting a cycle at the edge that
/// closes it, emitting a nesting structure on discover/finish pairs,
/// classifying edges. Doing that on top of a node-yielding traversal is what
/// forces a consumer to hand-roll a walk.
///
/// # Order
///
/// The event order is deterministic and part of the API. Successors are
/// examined in adjacency order (equivalently, the rev-push convention of
/// [`depth_first_preorder`](crate::graph::traverse::depth_first_preorder)). For a
/// node `u` the walk emits [`Discover`](DfsEvent::Discover) once, then for
/// each successor `v` in adjacency order either
/// [`TreeEdge`](DfsEvent::TreeEdge) immediately followed by `v`'s own events,
/// or [`BackEdge`](DfsEvent::BackEdge) / one
/// [`ForwardOrCross`](DfsEvent::ForwardOrCross), and finally
/// [`Finish`](DfsEvent::Finish). Only nodes reachable from `start` produce
/// events.
///
/// # Examples
///
/// Reporting the cycles of a graph in the order the walk closes them:
///
/// ```
/// use core::ops::ControlFlow;
///
/// use cfglib::{DfsEvent, Graph, TraversalDirection, depth_first_events};
///
/// // a -> b -> c -> a, plus a chord a -> c.
/// let mut graph = Graph::<&str, ()>::new();
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let c = graph.add_node("c");
/// graph.add_edge(a, b, ());
/// graph.add_edge(a, c, ());
/// graph.add_edge(b, c, ());
/// graph.add_edge(c, a, ());
///
/// let mut cycles = Vec::new();
/// let outcome = depth_first_events::<_, ()>(
///     &graph,
///     a,
///     TraversalDirection::Outgoing,
///     |event| {
///         if let DfsEvent::BackEdge(from, to) = event {
///             cycles.push((graph[from], graph[to]));
///         }
///         ControlFlow::Continue(())
///     },
/// );
///
/// assert_eq!(outcome, None);
/// assert_eq!(cycles, vec![("c", "a")]);
/// ```
///
/// # Panics
///
/// Panics when `start` is not a node in `graph`.
#[must_use]
pub fn depth_first_events<G: GraphView, B>(
    graph: &G,
    start: G::NodeId,
    direction: TraversalDirection,
    on_event: impl FnMut(DfsEvent<G::NodeId>) -> ControlFlow<B>,
) -> Option<B> {
    by_axis!(direction, depth_first_events_from(graph, start, on_event))
}

fn depth_first_events_from<G: GraphView, A: Adjacency, B>(
    axis: A,
    graph: &G,
    start: G::NodeId,
    mut on_event: impl FnMut(DfsEvent<G::NodeId>) -> ControlFlow<B>,
) -> Option<B> {
    assert!(
        start.index() < graph.node_bound(),
        "start node is out of range"
    );

    let mut color = vec![Color::White; graph.node_bound()];
    // Every frame's successors, appended as the frame is pushed and truncated
    // away as it pops. The top frame owns `arena[frame.start..]`.
    let mut arena: Vec<G::NodeId> = Vec::new();
    let mut stack: Vec<DfsFrame<G::NodeId>> = Vec::new();

    // Push the frame for a node just discovered at `depth`, taking its
    // adjacency onto the arena's tail.
    macro_rules! descend {
        ($node:expr, $depth:expr) => {{
            let start = arena.len();
            arena.extend(axis.neighbors(graph, $node));
            stack.push(DfsFrame {
                node: $node,
                depth: $depth,
                start,
                cursor: start,
            });
        }};
    }

    color[start.index()] = Color::Gray;
    emit_or_break!(on_event, DfsEvent::Discover(start, 0));
    descend!(start, 0);

    // Read frames through `last_mut` and copy the Copy fields out; the
    // successors are no longer among them, so no frame is ever cloned.
    while let Some(frame) = stack.last_mut() {
        let node = frame.node;
        let depth = frame.depth;
        if frame.cursor < arena.len() {
            let successor = arena[frame.cursor];
            frame.cursor += 1;
            match color[successor.index()] {
                Color::White => {
                    emit_or_break!(on_event, DfsEvent::TreeEdge(node, successor));
                    color[successor.index()] = Color::Gray;
                    emit_or_break!(on_event, DfsEvent::Discover(successor, depth + 1));
                    descend!(successor, depth + 1);
                }
                Color::Gray => emit_or_break!(on_event, DfsEvent::BackEdge(node, successor)),
                Color::Black => {
                    emit_or_break!(on_event, DfsEvent::ForwardOrCross(node, successor));
                }
            }
            continue;
        }

        color[node.index()] = Color::Black;
        emit_or_break!(on_event, DfsEvent::Finish(node));
        let Some(finished) = stack.pop() else { break };
        arena.truncate(finished.start);
    }

    None
}
