//! Depth-first traversal events over dense graph views.

extern crate alloc;

use alloc::vec;
use core::marker::PhantomData;
use core::ops::ControlFlow;

use crate::block::BlockId;
use crate::graph::search::Visit;
use crate::graph::traverse::{
    Adjacency, DepthFirstStack, DepthFirstVisitor, Mark, StepKind, by_axis, depth_first_core,
};
use crate::graph::view::{Direction, GraphView, assert_in_bound};

/// An event of a [`depth_first_events`] walk.
///
/// Together these are the classic tri-color depth-first classification: the
/// edges of the walk split into the tree it builds, the back edges that close
/// a cycle, and the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DfsEvent<N = BlockId> {
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

/// Walk `graph` depth-first from `start`, reporting every discovery, edge
/// classification, and finish, and returning the value the callback broke
/// with (or `None` when the walk ran to completion).
///
/// [`search`](crate::graph::search::search) answers questions about nodes;
/// this answers questions whose output order *is* the traversal — reporting a
/// cycle at the edge that closes it, emitting a nesting structure on
/// discover/finish pairs, classifying edges. Doing that on top of a
/// node-yielding traversal is what forces a consumer to hand-roll a walk.
///
/// # Order
///
/// The event order is deterministic and part of the API. Successors are
/// examined in adjacency order, as in
/// [`depth_first_preorder`](crate::graph::traverse::depth_first_preorder). For
/// a node `u` the walk emits [`Discover`](DfsEvent::Discover) once, then for
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
/// use cfglib::{DfsEvent, Direction, Graph, depth_first_events};
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
/// let outcome = depth_first_events::<_, ()>(&graph, a, Direction::Forward, |event| {
///     if let DfsEvent::BackEdge(from, to) = event {
///         cycles.push((graph[from], graph[to]));
///     }
///     ControlFlow::Continue(())
/// });
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
    direction: Direction,
    on_event: impl FnMut(DfsEvent<G::NodeId>) -> ControlFlow<B>,
) -> Option<B> {
    by_axis!(direction, depth_first_event_walk(graph, start, on_event))
}

fn depth_first_event_walk<G: GraphView, A: Adjacency, B>(
    axis: A,
    graph: &G,
    start: G::NodeId,
    on_event: impl FnMut(DfsEvent<G::NodeId>) -> ControlFlow<B>,
) -> Option<B> {
    assert_in_bound(graph, start, "start node");
    let mut marks = vec![Mark::Unseen; graph.node_bound()];
    depth_first_core(
        axis,
        graph,
        [start],
        marks.as_mut_slice(),
        &mut DepthFirstStack::default(),
        &mut Events {
            on_event,
            node: PhantomData,
        },
    )
}

/// The core's callbacks, translated into [`DfsEvent`]s.
struct Events<F, N> {
    on_event: F,
    node: PhantomData<N>,
}

impl<N: Copy, B, F: FnMut(DfsEvent<N>) -> ControlFlow<B>> DepthFirstVisitor<N, N, B>
    for Events<F, N>
{
    #[inline]
    fn discover(&mut self, node: N, depth: usize) -> ControlFlow<B, Visit> {
        (self.on_event)(DfsEvent::Discover(node, depth))?;
        ControlFlow::Continue(Visit::Descend)
    }

    #[inline]
    fn step(&mut self, from: N, _step: N, to: N, kind: StepKind) -> ControlFlow<B> {
        (self.on_event)(match kind {
            StepKind::Tree => DfsEvent::TreeEdge(from, to),
            StepKind::Back => DfsEvent::BackEdge(from, to),
            StepKind::ForwardOrCross => DfsEvent::ForwardOrCross(from, to),
        })
    }

    #[inline]
    fn finish(&mut self, node: N) -> ControlFlow<B> {
        (self.on_event)(DfsEvent::Finish(node))
    }
}

#[cfg(test)]
mod tests;
