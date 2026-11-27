//! Discipline-configurable search over any [`GraphView`].
//!
//! The traversals in [`traverse`](super::traverse) each answer one fixed
//! question — "every node in preorder", "every node reachable from a seed
//! set" — with one fixed discipline. Consumer walks in code intelligence
//! usually carry a *different* discipline, and that discipline is normally
//! the entire reason the walk was hand-rolled: stop at the first acceptable
//! answer, prune a subtree that cannot contain one, bound how far a chase
//! may go, or deliberately reach one node along two paths because both paths
//! are the answer (an ambiguous C++ base, two import routes to one symbol).
//!
//! [`search`] turns those into configuration: [`SearchOrder`] chooses the
//! frontier, [`VisitedPolicy`] chooses whether marks are global or per path,
//! [`SearchConfig::max_depth`] bounds expansion, the visitor's [`Visit`]
//! verdict prunes, and its [`ControlFlow::Break`] ends the walk with an
//! answer. [`depth_first_events`] serves the other family: consumers whose
//! output order *is* the traversal (cycle diagnostics, tri-color edge
//! classification).
//!
//! [`open_search`](super::open::open_search) applies the same
//! [`SearchConfig`] to a lazily discovered node space that has no dense
//! identities, and
//! [`open_depth_first_events`](super::open::open_depth_first_events) is this
//! module's event walk over that space — discover/finish pairs for folds,
//! with per-path marks that re-fold a shared node once per route.
//!
//! [`search_in`] is [`search`] over a caller-owned [`SearchScratch`], for
//! passes that search once per root over one node space: the visited marks
//! and every buffer of the walk then live across the searches of the pass.

extern crate alloc;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use crate::graph::epoch::EpochMarks;
use crate::graph::traverse::{
    Adjacency, DepthFirstStack, Indexed, PathMarks, breadth_first_core, by_axis, depth_first_core,
};
use crate::graph::view::{DenseId, Direction, GraphView, assert_in_bound};

/// The frontier discipline of a [`search`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchOrder {
    /// Expand the most recently discovered node first.
    ///
    /// Successors are expanded in adjacency order — the first successor of a
    /// node is expanded before the second — matching
    /// [`depth_first_preorder`](super::traverse::depth_first_preorder).
    #[default]
    DepthFirst,
    /// Expand nodes in discovery order, level by level.
    BreadthFirst,
}

/// Whether a search marks a node for the whole walk or only for the path it
/// is currently on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisitedPolicy {
    /// Mark a node the first time it is visited; never visit it again.
    ///
    /// This is what every traversal in [`traverse`](super::traverse) does,
    /// and the right discipline whenever the question is about nodes.
    #[default]
    Global,
    /// Mark a node on entry and **un-mark it on unwind**, so a node is
    /// visited once per distinct path that reaches it.
    ///
    /// This is the backtracking discipline: when the question is about
    /// *paths* — every derivation that reaches a symbol, every base
    /// subobject that makes a name ambiguous — a globally marked node
    /// silently hides the second answer. Termination comes from the
    /// path-cycle guard (a node already on the current path is not
    /// re-entered) plus [`SearchConfig::max_depth`]; the number of simple
    /// paths can be exponential, so bound the depth on dense graphs.
    ///
    /// Requires [`SearchOrder::DepthFirst`]: a breadth-first frontier has no
    /// unwind point at which to un-mark, so the combination is rejected.
    Path,
}

/// A visitor's verdict on the node it was handed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visit {
    /// Expand this node: discover its successors (subject to the depth
    /// bound).
    Descend,
    /// Prune here: the node counts as visited, but its successors are not
    /// discovered through it.
    Skip,
}

/// The discipline a [`search`] runs under, dense or
/// [open](super::open::open_search).
///
/// A plain struct: write the fields that differ from the default (a globally
/// marked, unbounded depth-first search) and take the rest with
/// `..SearchConfig::default()`. The direction of a dense search is its own
/// argument, because an open search has none: its successor closure *is* the
/// edge relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SearchConfig {
    /// Frontier discipline: depth-first or breadth-first.
    pub order: SearchOrder,
    /// Whether visited marks are global or per path.
    pub visited: VisitedPolicy,
    /// Maximum depth to expand from, in hops from a seed. `None` is
    /// unbounded; `Some(0)` visits the seeds alone.
    pub max_depth: Option<usize>,
}

/// Everything a repeated search allocates, owned by the caller: the visited
/// marks plus the seed, frontier, and stack buffers of the walk.
///
/// A pass that runs many searches over one large node space — a nulling
/// closure per grammar nonterminal, an alias chase per binding — where each
/// search visits a handful of nodes pays mostly for the call: a mark buffer
/// sized to the whole graph, a seed vector, a frontier, each a malloc and a
/// free for a four-node answer. Hand one scratch to every [`search_in`] of the
/// pass: the marks reset by an epoch bump and the buffers are emptied, both in
/// O(1), so the pass allocates only while the scratch grows to its largest
/// search.
///
/// The scratch starts empty and grows on entry to fit the graph, so one
/// scratch serves a sequence of graphs. Nodes are stored as dense indices, so
/// it carries no node-id type.
///
/// # Cost
///
/// Measured on 16,384 nodes whose closures are four nodes each, one search per
/// node: 8,143us with a fresh [`search`] per root and 235us over one reused
/// scratch. The win is per **call**, so it shrinks as searches grow: a walk
/// that visits a large fraction of the graph amortises its own buffers.
///
/// # Examples
///
/// ```
/// use core::ops::ControlFlow;
///
/// use cfglib::{Direction, Graph, GraphView, SearchConfig, SearchScratch, Visit, search_in};
///
/// //     a -> b -> c,  d -> c
/// let mut graph = Graph::<&str, ()>::new();
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let c = graph.add_node("c");
/// let d = graph.add_node("d");
/// graph.add_edge(a, b, ());
/// graph.add_edge(b, c, ());
/// graph.add_edge(d, c, ());
///
/// let mut scratch = SearchScratch::default();
/// let mut closures = Vec::new();
/// for root in graph.node_ids() {
///     let mut reached = Vec::new();
///     let config = SearchConfig::default();
///     let outcome = search_in(&mut scratch, &graph, [root], Direction::Forward, config, |node, _| {
///         reached.push(graph[node]);
///         ControlFlow::<(), _>::Continue(Visit::Descend)
///     });
///     assert_eq!(outcome, None);
///     closures.push(reached);
/// }
///
/// // Each root's closure is its own: nothing of the previous search survives.
/// assert_eq!(closures, vec![vec!["a", "b", "c"], vec!["b", "c"], vec!["c"], vec!["d", "c"]]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SearchScratch {
    /// The visited marks, reset by an epoch bump per search.
    marks: EpochMarks,
    /// The seeds, materialized so they can be validated before the walk.
    seeds: Vec<usize>,
    /// The breadth-first queue.
    queue: Vec<usize>,
    /// The depth-first frames and their steps.
    stack: DepthFirstStack<usize, usize>,
}

/// Search `graph` from `seeds` along `direction` under a configurable
/// discipline, returning the value the visitor broke with, or `None` when the
/// walk ran to completion.
///
/// # Semantics
///
/// - **Seeds** are visited at depth 0 in the order given. Under
///   [`VisitedPolicy::Global`] a seed already reached from an earlier seed is
///   skipped (so duplicate seeds are visited once); under
///   [`VisitedPolicy::Path`] each seed starts a fresh path context, so a
///   repeated seed is searched again.
/// - **Depth-first** expands the first successor of a node before the second
///   (adjacency order). **Breadth-first** visits in discovery order.
/// - **Depth** counts hops from the seed the search arrived through. Under
///   breadth-first that is the hop count from the nearest seed; under
///   depth-first it is the length of the path this walk actually took, which
///   need not be the shortest one.
/// - **`max_depth` bounds expansion, not visiting**: a node at the bound is
///   visited, but its successors are not discovered through it.
/// - **[`Visit::Skip`]** prunes the same way at any depth.
/// - **[`ControlFlow::Break`]** returns immediately with `Some(value)`; no
///   further node is visited.
///
/// Each call allocates its buffers; a pass that searches once per root over
/// one graph hands a [`SearchScratch`] to [`search_in`] instead.
///
/// # Examples
///
/// The first match depends on the discipline, which is the point:
///
/// ```
/// use core::ops::ControlFlow;
///
/// use cfglib::{Direction, Graph, SearchConfig, SearchOrder, Visit, search};
///
/// //     a -> b -> d
/// //     a -> c
/// let mut graph = Graph::<&str, ()>::new();
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let c = graph.add_node("c");
/// let d = graph.add_node("d");
/// graph.add_edge(a, b, ());
/// graph.add_edge(a, c, ());
/// graph.add_edge(b, d, ());
///
/// let first_leaf = |order| {
///     let config = SearchConfig { order, ..SearchConfig::default() };
///     search(&graph, [a], Direction::Forward, config, |node, _depth| {
///         if graph.successors(node).count() == 0 {
///             return ControlFlow::Break(graph[node]);
///         }
///         ControlFlow::Continue(Visit::Descend)
///     })
/// };
///
/// assert_eq!(first_leaf(SearchOrder::DepthFirst), Some("d"));
/// assert_eq!(first_leaf(SearchOrder::BreadthFirst), Some("c"));
/// ```
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`, or when `config` pairs
/// [`SearchOrder::BreadthFirst`] with [`VisitedPolicy::Path`] — a
/// breadth-first frontier has no unwind on which to un-mark.
#[must_use]
pub fn search<G: GraphView, B>(
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    direction: Direction,
    config: SearchConfig,
    visitor: impl FnMut(G::NodeId, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    search_in(
        &mut SearchScratch::default(),
        graph,
        seeds,
        direction,
        config,
        visitor,
    )
}

/// [`search`] over a caller-owned [`SearchScratch`].
///
/// Every semantic of [`search`] is unchanged — order, depth, pruning, early
/// exit, and both [`VisitedPolicy`] disciplines.
///
/// # Marks and buffers reset on entry
///
/// The scratch grows to the graph, its epoch is bumped and every buffer is
/// emptied before the walk starts, so a search never inherits marks *or* a
/// frontier — including from one that ended in [`ControlFlow::Break`] with
/// its frontier still loaded. There is no way to switch that off; a pass
/// wanting one mark set *shared* across several roots spells that by handing
/// all of those roots to a single call as seeds.
///
/// # Panics
///
/// Panics on the inputs [`search`] rejects: a seed that is not a node in
/// `graph`, and [`SearchOrder::BreadthFirst`] paired with
/// [`VisitedPolicy::Path`].
#[must_use]
pub fn search_in<G: GraphView, B>(
    scratch: &mut SearchScratch,
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    direction: Direction,
    config: SearchConfig,
    visitor: impl FnMut(G::NodeId, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    // `reserve_exact` before the fill: it is a no-op on a scratch that already
    // has the room, and on the *fresh* scratch `search` hands over it is what
    // keeps a one-seed search cheap. A bare `extend` from an empty buffer
    // reserves a four-element minimum, four times the bytes a one-seed search
    // needs — and at these sizes the allocator charges by the byte.
    scratch.seeds.clear();
    let staged = seeds.into_iter().map(|seed| {
        assert_in_bound(graph, seed, "seed node");
        seed.index()
    });
    scratch.seeds.reserve_exact(staged.size_hint().0);
    scratch.seeds.extend(staged);
    scratch.marks.grow_to(graph.node_bound());
    scratch.marks.reset();

    // The direction becomes a type here and nowhere else: each core is generic
    // over its adjacency axis, so it reads the graph's own adjacency in place
    // instead of branching per expanded node.
    match (config.order, config.visited) {
        (SearchOrder::DepthFirst, VisitedPolicy::Global) => by_axis!(
            direction,
            depth_first_global(graph, config, scratch, visitor)
        ),
        (SearchOrder::DepthFirst, VisitedPolicy::Path) => {
            by_axis!(direction, depth_first_path(graph, config, scratch, visitor))
        }
        (SearchOrder::BreadthFirst, VisitedPolicy::Global) => by_axis!(
            direction,
            breadth_first_global(graph, config, scratch, visitor)
        ),
        (SearchOrder::BreadthFirst, VisitedPolicy::Path) => {
            panic!(
                "VisitedPolicy::Path requires SearchOrder::DepthFirst: a breadth-first frontier never unwinds, so a path mark could never be removed"
            )
        }
    }
}

/// Whether a node at `depth` may expand under a `max_depth` bound.
///
/// The one statement of the rule every search discipline shares, dense and
/// open: the bound limits expansion, never visiting.
pub(crate) fn may_expand(max_depth: Option<usize>, depth: usize) -> bool {
    max_depth.is_none_or(|limit| depth < limit)
}

/// The visitor's verdict on a node, with a node at the depth bound pruned
/// like one the visitor skipped.
#[inline]
fn bounded<N: DenseId, B>(
    visitor: &mut impl FnMut(N, usize) -> ControlFlow<B, Visit>,
    max_depth: Option<usize>,
    node: usize,
    depth: usize,
) -> ControlFlow<B, Visit> {
    match visitor(N::from_index(node), depth) {
        ControlFlow::Continue(Visit::Descend) if !may_expand(max_depth, depth) => {
            ControlFlow::Continue(Visit::Skip)
        }
        verdict => verdict,
    }
}

fn depth_first_global<G: GraphView, A: Adjacency, B>(
    axis: A,
    graph: &G,
    config: SearchConfig,
    scratch: &mut SearchScratch,
    mut visitor: impl FnMut(G::NodeId, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let SearchScratch {
        marks,
        seeds,
        stack,
        ..
    } = scratch;
    depth_first_core(
        Indexed(axis),
        graph,
        seeds.iter().copied(),
        marks,
        stack,
        &mut |node, depth| bounded(&mut visitor, config.max_depth, node, depth),
    )
}

fn depth_first_path<G: GraphView, A: Adjacency, B>(
    axis: A,
    graph: &G,
    config: SearchConfig,
    scratch: &mut SearchScratch,
    mut visitor: impl FnMut(G::NodeId, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let SearchScratch {
        marks,
        seeds,
        stack,
        ..
    } = scratch;
    depth_first_core(
        Indexed(axis),
        graph,
        seeds.iter().copied(),
        &mut PathMarks(marks),
        stack,
        &mut |node, depth| bounded(&mut visitor, config.max_depth, node, depth),
    )
}

fn breadth_first_global<G: GraphView, A: Adjacency, B>(
    axis: A,
    graph: &G,
    config: SearchConfig,
    scratch: &mut SearchScratch,
    mut visitor: impl FnMut(G::NodeId, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let SearchScratch {
        marks,
        seeds,
        queue,
        ..
    } = scratch;
    queue.clear();
    for &seed in &*seeds {
        if !marks.is_marked(seed) {
            marks.mark(seed);
            queue.push(seed);
        }
    }
    breadth_first_core(
        Indexed(axis),
        graph,
        queue,
        |node, depth| bounded(&mut visitor, config.max_depth, node, depth),
        |_, _, to, _| {
            if marks.is_marked(to) {
                return false;
            }
            marks.mark(to);
            true
        },
    )
}

mod events;
pub use events::{DfsEvent, depth_first_events};

mod path_budget;
mod path_frontier;
mod path_results;
pub(crate) use path_budget::PathBudget;
pub use path_budget::{PathSearchConfig, PathSearchStats};
pub(crate) use path_frontier::PathFrontier;
pub(crate) use path_results::{retain_unshadowed, sources_by_target};

#[cfg(test)]
mod tests;
