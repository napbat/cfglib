//! Breadth-first event walk over an open node space.

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use crate::graph::search::Visit;

use super::super::marks::Marks;
use super::super::may_expand;

/// Configuration for an [`open_breadth_first_events`] walk.
///
/// Breadth-first traversal always marks nodes globally: unlike a depth-first
/// stack, its frontier has no unwind point at which a path mark could be
/// released. The type therefore exposes only the meaningful depth bound.
///
/// Once-per-route semantics *breadth-first* is not impossible — it is a
/// different algorithm. A frontier whose entries each carry their own route
/// enumerates simple paths shortest-first, trading this walk's `O(nodes)`
/// marks for a route tree that is exponential in the worst case, and its
/// events carry the route to mean anything. That walk is
/// [`open_breadth_first_paths`](super::super::open_breadth_first_paths), a
/// separate entry point rather than a policy field here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OpenBfsConfig {
    /// Maximum depth to expand from, in hops from a seed. `None` is
    /// unbounded; `Some(0)` discovers the seeds alone.
    pub max_depth: Option<usize>,
}

impl OpenBfsConfig {
    /// An unbounded breadth-first event walk.
    #[must_use]
    pub const fn new() -> Self {
        Self { max_depth: None }
    }

    /// Return this configuration bounded to `max_depth` hops from a seed.
    #[must_use]
    pub const fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }
}

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
/// # Semantics
///
/// - Seeds are discovered at depth zero in the order supplied. A duplicate
///   seed produces [`Refused`](OpenBfsEvent::Refused).
/// - Nodes are expanded level by level in discovery order. Successors are
///   discovered in the order the closure pushes them.
/// - [`Visit::Skip`] prevents expansion without removing the node's mark.
/// - `max_depth` bounds expansion, not discovery. A node at the bound still
///   produces [`Discover`](OpenBfsEvent::Discover).
/// - A route to any previously discovered node produces `Refused`, including
///   routes to a node still waiting in the frontier.
/// - [`ControlFlow::Break`] stops immediately and is returned as `Some(value)`.
///
/// # Examples
///
/// ```
/// use core::ops::ControlFlow;
///
/// use cfglib::{OpenBfsConfig, OpenBfsEvent, Visit, open_breadth_first_events};
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
///     OpenBfsConfig::new(),
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
    config: OpenBfsConfig,
    mut successors: impl FnMut(&N, &mut Vec<N>),
    mut on_event: impl FnMut(OpenBfsEvent<'_, N>) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let mut marks = Marks::new();
    let mut queue = VecDeque::new();

    for seed in seeds {
        if let ControlFlow::Break(value) =
            discover(seed, 0, config, &mut marks, &mut queue, &mut on_event)
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
                config,
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
    config: OpenBfsConfig,
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
        ControlFlow::Continue(Visit::Descend) if may_expand(config.max_depth, depth) => {
            queue.push_back((node, depth));
            ControlFlow::Continue(())
        }
        ControlFlow::Continue(Visit::Descend | Visit::Skip) => ControlFlow::Continue(()),
    }
}
