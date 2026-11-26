//! Breadth-first event walk over an open node space.

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use crate::graph::search::{Visit, may_expand};

use super::super::marks::Marks;

/// An event of an [`open_breadth_first_events`] walk.
///
/// Nodes are reported by reference so the walk retains ownership of its
/// frontier. There is no `Finish` event: breadth-first traversal does not
/// unwind a nested subtree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenBfsEvent<'a, N> {
    /// The node was accepted for the first time, at this depth.
    ///
    /// This is the only event whose [`Visit`] verdict is read.
    Discover(&'a N, usize),
    /// The walk reached a node already discovered through another route.
    ///
    /// The returned [`Visit`] verdict is ignored because the node is already
    /// represented in the frontier or has already been expanded.
    Refused(&'a N, usize),
}

/// Walk a lazily discovered node space breadth-first from `seeds`, reporting
/// every discovery and every route refused by global marks, and return the
/// value the callback broke with (or `None` when the walk completed).
///
/// `successors` receives a node and a scratch buffer that is **cleared before
/// every call**. It pushes successors in their desired discovery order. Nodes
/// are compared and marked by `Ord`.
///
/// Breadth-first traversal always marks nodes globally: unlike a depth-first
/// stack, its frontier has no unwind point at which a path mark could be
/// released, so the depth bound is the walk's only setting. Once-per-route
/// semantics breadth-first is a different algorithm, whose frontier entries
/// each carry their own route:
/// [`open_breadth_first_paths`](super::super::open_breadth_first_paths).
///
/// # Semantics
///
/// - Seeds are discovered at depth zero in the order supplied. A duplicate
///   seed produces [`Refused`](OpenBfsEvent::Refused).
/// - Nodes are expanded level by level in discovery order. Successors are
///   discovered in the order the closure pushes them.
/// - [`Visit::Skip`] prevents expansion without removing the node's mark.
/// - `max_depth` bounds expansion, not discovery: `None` is unbounded, and a
///   node at the bound still produces [`Discover`](OpenBfsEvent::Discover),
///   so `Some(0)` discovers the seeds alone.
/// - A route to any previously discovered node produces `Refused`, including
///   routes to a node still waiting in the frontier.
/// - [`ControlFlow::Break`] stops immediately and is returned as `Some(value)`.
///
/// # Examples
///
/// ```
/// use core::ops::ControlFlow;
///
/// use cfglib::{OpenBfsEvent, Visit, open_breadth_first_events};
///
/// let successors = |node: &&'static str, out: &mut Vec<&'static str>| match *node {
///     "root" => out.extend(["left", "right"]),
///     "left" | "right" => out.push("merge"),
///     _ => {}
/// };
///
/// let mut discovered = Vec::new();
/// let mut refused = Vec::new();
/// let outcome = open_breadth_first_events::<_, ()>(
///     ["root"],
///     None,
///     successors,
///     |event| {
///         match event {
///             OpenBfsEvent::Discover(&node, depth) => discovered.push((node, depth)),
///             OpenBfsEvent::Refused(&node, _) => refused.push(node),
///         }
///         ControlFlow::Continue(Visit::Descend)
///     },
/// );
///
/// assert_eq!(outcome, None);
/// assert_eq!(discovered, vec![("root", 0), ("left", 1), ("right", 1), ("merge", 2)]);
/// assert_eq!(refused, vec!["merge"]);
/// ```
#[must_use]
pub fn open_breadth_first_events<N: Clone + Ord, B>(
    seeds: impl IntoIterator<Item = N>,
    max_depth: Option<usize>,
    mut successors: impl FnMut(&N, &mut Vec<N>),
    mut on_event: impl FnMut(OpenBfsEvent<'_, N>) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let mut marks = Marks::new();
    let mut queue = VecDeque::new();

    for seed in seeds {
        if let ControlFlow::Break(value) =
            discover(seed, 0, max_depth, &mut marks, &mut queue, &mut on_event)
        {
            return Some(value);
        }
    }

    let mut discovered = Vec::new();
    while let Some((node, depth)) = queue.pop_front() {
        discovered.clear();
        successors(&node, &mut discovered);
        for next in discovered.drain(..) {
            if let ControlFlow::Break(value) = discover(
                next,
                depth + 1,
                max_depth,
                &mut marks,
                &mut queue,
                &mut on_event,
            ) {
                return Some(value);
            }
        }
    }

    None
}

fn discover<N: Clone + Ord, B>(
    node: N,
    depth: usize,
    max_depth: Option<usize>,
    marks: &mut Marks<N>,
    queue: &mut VecDeque<(N, usize)>,
    on_event: &mut impl FnMut(OpenBfsEvent<'_, N>) -> ControlFlow<B, Visit>,
) -> ControlFlow<B> {
    if !marks.insert(node.clone()) {
        return match on_event(OpenBfsEvent::Refused(&node, depth)) {
            ControlFlow::Break(value) => ControlFlow::Break(value),
            ControlFlow::Continue(_) => ControlFlow::Continue(()),
        };
    }

    match on_event(OpenBfsEvent::Discover(&node, depth)) {
        ControlFlow::Break(value) => ControlFlow::Break(value),
        ControlFlow::Continue(Visit::Descend) if may_expand(max_depth, depth) => {
            queue.push_back((node, depth));
            ControlFlow::Continue(())
        }
        ControlFlow::Continue(Visit::Descend | Visit::Skip) => ControlFlow::Continue(()),
    }
}
