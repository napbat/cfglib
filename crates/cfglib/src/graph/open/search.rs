//! Configurable search over an open node space.

extern crate alloc;

use alloc::vec::Vec;
use core::mem;
use core::ops::ControlFlow;
use smallvec::SmallVec;

use crate::graph::search::{SearchOrder, Visit, VisitedPolicy, may_expand};

use super::marks::{INLINE_NODES, Marks};

/// The discipline an [`open_search`] runs under.
///
/// This is [`SearchConfig`](crate::graph::search::SearchConfig) minus its
/// `direction`: in an open space the successor closure *is* the edge
/// relation, so a walk runs backwards by being given a closure that reports
/// predecessors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenSearchConfig {
    /// Frontier discipline: depth-first or breadth-first.
    pub order: SearchOrder,
    /// Whether visited marks are global or per path.
    pub visited: VisitedPolicy,
    /// Maximum depth to expand from, in hops from a seed. `None` is
    /// unbounded; `Some(0)` visits the seeds alone.
    pub max_depth: Option<usize>,
}

impl OpenSearchConfig {
    /// A globally marked, unbounded search in `order`.
    #[must_use]
    pub const fn new(order: SearchOrder) -> Self {
        Self {
            order,
            visited: VisitedPolicy::Global,
            max_depth: None,
        }
    }

    /// Return this configuration with a different [`VisitedPolicy`].
    #[must_use]
    pub const fn with_visited(mut self, visited: VisitedPolicy) -> Self {
        self.visited = visited;
        self
    }

    /// Return this configuration bounded to `max_depth` hops from a seed.
    #[must_use]
    pub const fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }
}

/// Search a lazily discovered node space from `seeds`, returning the value the
/// visitor broke with, or `None` when the walk ran to completion.
///
/// `successors` is handed a node and a scratch buffer, **cleared before every
/// call**, and pushes that node's successors *in the order they should be
/// explored*. Nodes are compared and marked by `Ord`, so identity is whatever
/// the consumer's node type says it is — a `(file, name)` pair, a resolved
/// module handle, an interned id.
///
/// # Semantics
///
/// Identical to [`search`](crate::graph::search::search), with visited marks held
/// in an ordered set (a short inline list while the walk is small) instead of a
/// dense table:
///
/// - **Seeds** are visited at depth 0 in the order given; deduplicated under
///   [`VisitedPolicy::Global`], each starting a fresh path context under
///   [`VisitedPolicy::Path`].
/// - **Depth-first** explores the first pushed successor first.
///   **Breadth-first** visits in discovery order.
/// - **`max_depth` bounds expansion, not visiting.**
/// - **[`Visit::Skip`]** prunes a node's successors; the closure is not even
///   called for it, which matters when discovering successors costs a file
///   read.
/// - **[`ControlFlow::Break`]** returns immediately with `Some(value)`.
///
/// # Examples
///
/// Chasing a name through re-export barrels, named re-exports before stars,
/// stopping at the first module that defines it:
///
/// ```
/// use core::ops::ControlFlow;
///
/// use cfglib::{OpenSearchConfig, SearchOrder, Visit, open_search};
///
/// // Module 0 re-exports `Widget` from module 1 by name and has a star
/// // re-export of module 2; module 2 stars module 3. Both 1 and 3 define
/// // `Widget`, so the push order decides which one wins.
/// let successors = |node: &(u32, &'static str), out: &mut Vec<(u32, &'static str)>| {
///     let (module, name) = *node;
///     match module {
///         0 => {
///             out.push((1, name)); // named re-export first
///             out.push((2, name)); // then the star
///         }
///         2 => out.push((3, name)),
///         _ => {}
///     }
/// };
///
/// let defines = |&(module, _): &(u32, &str)| module == 1 || module == 3;
/// let found = open_search(
///     [(0, "Widget")],
///     OpenSearchConfig::new(SearchOrder::DepthFirst),
///     successors,
///     |node, _depth| {
///         if defines(node) {
///             return ControlFlow::Break(*node);
///         }
///         ControlFlow::Continue(Visit::Descend)
///     },
/// );
///
/// assert_eq!(found, Some((1, "Widget")));
/// ```
///
/// # Panics
///
/// Panics when `config` pairs [`SearchOrder::BreadthFirst`] with
/// [`VisitedPolicy::Path`] — a breadth-first frontier has no unwind on which
/// to un-mark.
#[must_use]
pub fn open_search<N: Clone + Ord, B>(
    seeds: impl IntoIterator<Item = N>,
    config: OpenSearchConfig,
    successors: impl FnMut(&N, &mut Vec<N>),
    visitor: impl FnMut(&N, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    match (config.order, config.visited) {
        (SearchOrder::DepthFirst, VisitedPolicy::Global) => {
            open_depth_first_global(seeds, config, successors, visitor)
        }
        (SearchOrder::DepthFirst, VisitedPolicy::Path) => {
            open_depth_first_path(seeds, config, successors, visitor)
        }
        (SearchOrder::BreadthFirst, VisitedPolicy::Global) => {
            open_breadth_first_global(seeds, config, successors, visitor)
        }
        (SearchOrder::BreadthFirst, VisitedPolicy::Path) => {
            panic!(
                "VisitedPolicy::Path requires SearchOrder::DepthFirst: a breadth-first frontier never unwinds, so a path mark could never be removed"
            )
        }
    }
}

fn open_depth_first_global<N: Clone + Ord, B>(
    seeds: impl IntoIterator<Item = N>,
    config: OpenSearchConfig,
    mut successors: impl FnMut(&N, &mut Vec<N>),
    mut visitor: impl FnMut(&N, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let mut visited = Marks::new();
    // Reversed so the first seed pops first; the same convention applies to
    // successors below, so push order is expansion order.
    let mut stack: SmallVec<[(N, usize); INLINE_NODES]> =
        seeds.into_iter().map(|seed| (seed, 0)).collect();
    stack.reverse();
    let mut discovered = Vec::new();

    while let Some((node, depth)) = stack.pop() {
        if !visited.insert(node.clone()) {
            continue;
        }
        match visitor(&node, depth) {
            ControlFlow::Break(value) => return Some(value),
            ControlFlow::Continue(Visit::Skip) => continue,
            ControlFlow::Continue(Visit::Descend) => {}
        }
        if !may_expand(config.max_depth, depth) {
            continue;
        }
        discovered.clear();
        successors(&node, &mut discovered);
        for next in discovered.drain(..).rev() {
            if !visited.contains(&next) {
                stack.push((next, depth + 1));
            }
        }
    }

    None
}

/// One step of a path-marked open depth-first walk.
enum PathStep<N> {
    /// Enter `node` at this depth, marking it on the current path.
    Enter(N, usize),
    /// Leave `node`: the unwind point at which its path mark is removed.
    Leave(N),
}

fn open_depth_first_path<N: Clone + Ord, B>(
    seeds: impl IntoIterator<Item = N>,
    config: OpenSearchConfig,
    mut successors: impl FnMut(&N, &mut Vec<N>),
    mut visitor: impl FnMut(&N, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let mut on_path = Marks::new();
    let mut stack: SmallVec<[PathStep<N>; INLINE_NODES]> = seeds
        .into_iter()
        .map(|seed| PathStep::Enter(seed, 0))
        .collect();
    stack.reverse();
    let mut discovered = Vec::new();

    while let Some(step) = stack.pop() {
        let (node, depth) = match step {
            // Every `Enter` pushes its own `Leave` before any successor, so
            // this pops exactly when the node's subtree is exhausted — and
            // between two seeds the path is empty again.
            PathStep::Leave(node) => {
                on_path.remove(&node);
                continue;
            }
            PathStep::Enter(node, depth) => (node, depth),
        };
        if !on_path.insert(node.clone()) {
            continue;
        }
        stack.push(PathStep::Leave(node.clone()));
        match visitor(&node, depth) {
            ControlFlow::Break(value) => return Some(value),
            ControlFlow::Continue(Visit::Skip) => continue,
            ControlFlow::Continue(Visit::Descend) => {}
        }
        if !may_expand(config.max_depth, depth) {
            continue;
        }
        discovered.clear();
        successors(&node, &mut discovered);
        for next in discovered.drain(..).rev() {
            stack.push(PathStep::Enter(next, depth + 1));
        }
    }

    None
}

fn open_breadth_first_global<N: Clone + Ord, B>(
    seeds: impl IntoIterator<Item = N>,
    config: OpenSearchConfig,
    mut successors: impl FnMut(&N, &mut Vec<N>),
    mut visitor: impl FnMut(&N, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let mut visited = Marks::new();
    // The frontier a first-in first-out queue would hold, one depth at a time:
    // every node of one depth precedes every node of the next in discovery
    // order, so draining a level and filling the following one visits nodes in
    // the queue's order without storing a depth beside each node.
    let mut level: SmallVec<[N; INLINE_NODES]> = SmallVec::new();
    for seed in seeds {
        if visited.insert(seed.clone()) {
            level.push(seed);
        }
    }
    let mut following: SmallVec<[N; INLINE_NODES]> = SmallVec::new();
    let mut discovered = Vec::new();
    let mut depth = 0;

    while !level.is_empty() {
        for node in level.drain(..) {
            match visitor(&node, depth) {
                ControlFlow::Break(value) => return Some(value),
                ControlFlow::Continue(Visit::Skip) => continue,
                ControlFlow::Continue(Visit::Descend) => {}
            }
            if !may_expand(config.max_depth, depth) {
                continue;
            }
            discovered.clear();
            successors(&node, &mut discovered);
            for next in discovered.drain(..) {
                if visited.insert(next.clone()) {
                    following.push(next);
                }
            }
        }
        mem::swap(&mut level, &mut following);
        depth += 1;
    }

    None
}
