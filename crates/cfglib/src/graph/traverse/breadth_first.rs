//! The breadth-first walk core every breadth-first walk of the crate runs.

extern crate alloc;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use super::Steps;
use crate::graph::search::Visit;
use crate::graph::view::GraphView;

/// Walk `graph` breadth-first from the nodes already in `queue`, level by
/// level, and return the value a callback broke with.
///
/// `queue` holds nodes in the walk's [`Steps`] representation; the caller
/// seeds it with the depth-0 nodes, in the order they are to be expanded, and
/// has already recorded them in whatever marks it keeps. The queue is read through a head cursor rather
/// than drained, so on return it holds every node the walk enqueued, in
/// discovery order. Depths are not stored: a level ends where the queue ended
/// when the level began.
///
/// - `visit(node, depth)` runs when a node is dequeued. [`Visit::Skip`] keeps
///   its steps unexamined (a pruned node, a node at a depth bound),
///   [`ControlFlow::Break`] ends the walk.
/// - `step(from, step, to, depth)` runs for every step out of an expanded
///   node, in adjacency order, with `depth` the depth of `from`. It decides
///   whether `to` is new to the walk, marking it as it sees fit; `true`
///   enqueues `to` at `depth + 1`.
///
/// Leaving the marks to `step` is what lets one loop serve walks whose marks
/// are a `Vec<bool>`, an [`EpochMarks`](crate::EpochMarks), or a distance
/// table that doubles as one. The core reads `steps` in place, without
/// copying a node's adjacency.
#[inline]
pub(crate) fn breadth_first_core<G, S, B>(
    steps: S,
    graph: &G,
    queue: &mut Vec<S::Node>,
    mut visit: impl FnMut(S::Node, usize) -> ControlFlow<B, Visit>,
    mut step: impl FnMut(S::Node, S::Step, S::Node, usize) -> bool,
) -> Option<B>
where
    G: GraphView,
    S: Steps<G>,
{
    let mut head = 0;
    let mut depth = 0;
    let mut level_end = queue.len();
    while head < queue.len() {
        if head == level_end {
            depth += 1;
            level_end = queue.len();
        }
        let node = queue[head];
        head += 1;
        match visit(node, depth) {
            ControlFlow::Break(value) => return Some(value),
            ControlFlow::Continue(Visit::Skip) => continue,
            ControlFlow::Continue(Visit::Descend) => {}
        }
        for out in steps.steps(graph, node) {
            let to = steps.target(graph, out);
            if step(node, out, to, depth) {
                queue.push(to);
            }
        }
    }
    None
}
