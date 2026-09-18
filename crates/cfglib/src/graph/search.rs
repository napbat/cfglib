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
//! [`open_search`](super::open::open_search) applies the same discipline to a
//! lazily discovered node space that has no dense identities, and
//! [`open_depth_first_events`](super::open::open_depth_first_events) is this
//! module's event walk over that space — discover/finish pairs for folds,
//! with per-path marks that re-fold a shared node once per route.
//!
//! [`search_with_marks`] is [`search`] with its visited marks moved into a
//! caller-owned [`EpochMarks`], for passes that search once per root over one
//! node space and cannot pay an O(node count) mark buffer per root;
//! [`search_with_scratch`] moves the rest of the call's buffers out too, for
//! passes whose searches are so small that the *call* is the cost.

extern crate alloc;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use crate::graph::epoch::EpochMarks;
use crate::graph::traverse::{Adjacency, TraversalDirection, by_axis};
use crate::graph::view::{DenseId, GraphView};

/// The frontier discipline of a [`search`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchOrder {
    /// Expand the most recently discovered node first.
    ///
    /// Successors are expanded in adjacency order — the first successor of a
    /// node is expanded before the second — matching the rev-push convention
    /// of [`depth_first_preorder`](super::traverse::depth_first_preorder).
    DepthFirst,
    /// Expand nodes in discovery order, level by level.
    BreadthFirst,
}

/// Whether a search marks a node for the whole walk or only for the path it
/// is currently on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisitedPolicy {
    /// Mark a node the first time it is visited; never visit it again.
    ///
    /// This is what every traversal in [`traverse`](super::traverse) does,
    /// and the right discipline whenever the question is about nodes.
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

/// The discipline a [`search`] runs under.
///
/// The fields are public and the struct is `Copy`, so a consumer that stores
/// a discipline as data can write it as a literal;
/// [`new`](SearchConfig::new) plus the two setters are the ergonomic form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchConfig {
    /// Frontier discipline: depth-first or breadth-first.
    pub order: SearchOrder,
    /// Whether visited marks are global or per path.
    pub visited: VisitedPolicy,
    /// Whether the walk follows edges forwards or backwards.
    pub direction: TraversalDirection,
    /// Maximum depth to expand from, in hops from a seed. `None` is
    /// unbounded; `Some(0)` visits the seeds alone.
    pub max_depth: Option<usize>,
}

impl SearchConfig {
    /// A globally marked, unbounded search in `order` along `direction`.
    #[must_use]
    pub const fn new(order: SearchOrder, direction: TraversalDirection) -> Self {
        Self {
            order,
            visited: VisitedPolicy::Global,
            direction,
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

/// One step of a path-marked depth-first walk, as a dense node index.
#[derive(Debug, Clone, Copy)]
enum PathStep {
    /// Enter this node at this depth, marking it on the current path.
    Enter(usize, usize),
    /// Leave this node: the unwind point at which its path mark is removed.
    Leave(usize),
}

/// Every buffer a search fills whose size is O(the walk) rather than O(the
/// graph) — the ones [`EpochMarks`] deliberately left in the call.
///
/// Nodes are stored as the dense indices [`DenseId`] guarantees they are,
/// and the cores convert at the boundary. That is what keeps the buffers — and
/// so [`SearchScratch`] — free of a node-id type parameter: one scratch type
/// serves every [`GraphView`], and one scratch instance can be reused
/// across graphs whose ids differ.
#[derive(Debug, Clone, Default)]
struct SearchBuffers {
    /// The seeds, materialized so they can be validated once and pushed in
    /// reverse.
    seeds: Vec<usize>,
    /// The frontier of the two globally marked cores: popped from the back as
    /// a stack by the depth-first core, read through a head cursor as a queue
    /// by the breadth-first one. The queue never reuses the space in front of
    /// its head, which costs nothing here — a globally marked walk enqueues
    /// each node at most once, so the buffer is bounded by the node count
    /// either way, and one buffer then serves both disciplines.
    frontier: Vec<(usize, usize)>,
    /// The frontier of the path-marked core, whose unwind markers the other
    /// two have no use for.
    path: Vec<PathStep>,
    /// One expanded node's adjacency, refilled per expansion — the buffer only
    /// the depth-first cores still need, to read a node's successors in
    /// reverse.
    adjacent: Vec<usize>,
}

impl SearchBuffers {
    /// Empty every buffer, so a search never reads a previous one's leftovers.
    ///
    /// This is a correctness step, not only hygiene: a walk that ended in
    /// [`ControlFlow::Break`] returns with its frontier still loaded.
    fn reset(&mut self) {
        self.seeds.clear();
        self.frontier.clear();
        self.path.clear();
        self.adjacent.clear();
    }
}

/// Everything a repeated search allocates, owned by the caller: the visited
/// marks of [`EpochMarks`] plus the seed, frontier, and adjacency buffers of
/// the call itself.
///
/// [`EpochMarks`] removed the term that scales with the **graph**. This removes
/// the term that scales with the **call**, which is what is left when a pass
/// runs many *tiny* searches over one large node space — a nulling closure per
/// grammar nonterminal, an alias chase per binding — and each search visits a
/// handful of nodes. There the walk itself is nearly free and the cost is the
/// call: a seed vector, a frontier, and (depth-first only) an adjacency
/// buffer, each a malloc and a free for a four-node answer.
///
/// Hand one scratch to every [`search_with_scratch`] of the pass. The buffers
/// grow to the largest search the pass performs and are then reused by every
/// search after it, so the whole pass allocates a bounded number of times
/// instead of a few times per root.
///
/// # Cost
///
/// Measured on 16,384 nodes whose closures are four nodes each, one search per
/// node: 8,143us with a fresh [`search`] per root, 1,413us over a reused
/// [`EpochMarks`] alone, and 235us over a reused `SearchScratch` — another 6.0x
/// on top of the 5.8x the marks already gave. The win is per **call**, so it
/// shrinks as searches grow: a walk that visits a large fraction of the graph
/// amortises its own buffers, and the same measurement over a 1,024-node chain
/// walked from every root moves by a few percent.
///
/// # Marks and buffers both reset on entry
///
/// A search never inherits marks (the epoch bump of [`EpochMarks`]) and never
/// inherits a frontier, including from a search that ended in
/// [`ControlFlow::Break`] with its frontier still loaded. Both resets are O(1)
/// and neither can be switched off, so the whole public surface stays [`new`]
/// plus [`capacity`].
///
/// [`new`]: SearchScratch::new
/// [`capacity`]: SearchScratch::capacity
///
/// # Sizing
///
/// [`capacity`](SearchScratch::capacity) is the node space the *marks* cover,
/// and it is fixed at construction on the terms [`EpochMarks`] sets: a scratch
/// smaller than the graph is a panic rather than a resize, and one larger than
/// the graph is fine, which is how one scratch covers a set of graphs. The
/// other buffers carry no such promise, because their size is O(the walk) and
/// cannot be known from a node count — they grow on the searches that need
/// them and are reused by the ones that follow.
///
/// # Examples
///
/// ```
/// use cfglib::SearchScratch;
///
/// let scratch = SearchScratch::new(64);
/// assert_eq!(scratch.capacity(), 64);
/// ```
#[derive(Debug, Clone)]
pub struct SearchScratch {
    /// The visited marks, reset by an epoch bump per search.
    marks: EpochMarks,
    /// The per-call buffers, emptied per search.
    buffers: SearchBuffers,
}

impl SearchScratch {
    /// Scratch covering `node_count` nodes, with nothing marked.
    ///
    /// Only the marks are sized here; the other buffers start empty and grow
    /// to what the pass actually walks.
    #[must_use]
    pub fn new(node_count: usize) -> Self {
        Self {
            marks: EpochMarks::new(node_count),
            buffers: SearchBuffers::default(),
        }
    }

    /// Return how many nodes this scratch's marks cover.
    ///
    /// A [`search_with_scratch`] over a graph with more nodes than this panics;
    /// a consumer whose node space grew builds a new scratch.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.marks.capacity()
    }
}

/// Search `graph` from `seeds` under a configurable discipline, returning the
/// value the visitor broke with, or `None` when the walk ran to completion.
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
/// # Examples
///
/// The first match depends on the discipline, which is the point:
///
/// ```
/// use core::ops::ControlFlow;
///
/// use cfglib::{Graph, SearchConfig, SearchOrder, TraversalDirection, Visit, search};
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
///     search(
///         &graph,
///         [a],
///         SearchConfig::new(order, TraversalDirection::Outgoing),
///         |node, _depth| {
///             if graph.successors(node).count() == 0 {
///                 return ControlFlow::Break(graph[node]);
///             }
///             ControlFlow::Continue(Visit::Descend)
///         },
///     )
/// };
///
/// assert_eq!(first_leaf(SearchOrder::DepthFirst), Some("d"));
/// assert_eq!(first_leaf(SearchOrder::BreadthFirst), Some("c"));
/// ```
///
/// A pass that searches once per root over one graph should hand its marks to
/// [`search_with_marks`] instead, or its whole scratch to
/// [`search_with_scratch`]; both are this function with buffers lifted out of
/// the call.
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
    config: SearchConfig,
    visitor: impl FnMut(G::NodeId, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let mut scratch = SearchScratch::new(graph.node_bound());
    search_with_scratch(graph, seeds, config, &mut scratch, visitor)
}

/// [`search`], with the visited marks living in a caller-owned [`EpochMarks`]
/// instead of being allocated per call.
///
/// Every semantic of [`search`] is unchanged — order, depth, pruning, early
/// exit, and both [`VisitedPolicy`] disciplines all read and write `marks`
/// exactly where `search` reads and writes its own buffer. This is
/// [`search_with_scratch`] over a fresh frontier, and `search` is it over a
/// fresh everything.
///
/// A pass whose searches are *small* wants [`search_with_scratch`] instead:
/// the marks are the buffer that scales with the graph, but the seed vector,
/// the frontier, and the depth-first cores' adjacency buffer are still
/// allocated per call, and on a four-node closure those are the cost.
///
/// # Marks are reset on entry
///
/// A search never inherits marks: it bumps the epoch first, so no node is
/// marked when the walk starts and no consumer can search through another
/// search's leftovers. That is O(1) and there is no way to switch it off — a
/// pass wanting one mark set *shared* across several roots already has one, by
/// handing all of those roots to a single call as seeds, which is exactly what
/// sharing marks would mean.
///
/// # Examples
///
/// One closure per root over one buffer — the shape this exists for. The
/// marks are allocated once for the whole pass; only the answers are per
/// root:
///
/// ```
/// use core::ops::ControlFlow;
///
/// use cfglib::{
///     Graph, GraphView, EpochMarks, SearchConfig, SearchOrder,
///     TraversalDirection, Visit, search_with_marks,
/// };
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
/// let config = SearchConfig::new(SearchOrder::BreadthFirst, TraversalDirection::Outgoing);
/// let mut marks = EpochMarks::new(graph.node_bound());
/// let mut closures = Vec::new();
///
/// for root in graph.node_ids() {
///     let mut reached = Vec::new();
///     let outcome = search_with_marks(&graph, [root], config, &mut marks, |node, _depth| {
///         reached.push(graph[node]);
///         ControlFlow::<(), _>::Continue(Visit::Descend)
///     });
///     assert_eq!(outcome, None);
///     closures.push(reached);
/// }
///
/// // Each root's closure is its own: the previous root's marks are gone.
/// assert_eq!(closures, vec![vec!["a", "b", "c"], vec!["b", "c"], vec!["c"], vec!["d", "c"]]);
/// ```
///
/// # Panics
///
/// Panics when `marks` cover fewer nodes than `graph` — a marks buffer is
/// sized at construction and never resized, so that the pass it serves cannot
/// allocate behind the consumer's back. Also panics on the two inputs
/// [`search`] rejects: a seed that is not a node in `graph`, and
/// [`SearchOrder::BreadthFirst`] paired with [`VisitedPolicy::Path`].
#[must_use]
pub fn search_with_marks<G: GraphView, B>(
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    config: SearchConfig,
    marks: &mut EpochMarks,
    visitor: impl FnMut(G::NodeId, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let mut buffers = SearchBuffers::default();
    search_in(graph, seeds, config, marks, &mut buffers, visitor)
}

/// [`search`], with **every** buffer it would allocate living in a
/// caller-owned [`SearchScratch`].
///
/// This is the one implementation: [`search_with_marks`] is this function over
/// a fresh frontier, and [`search`] is it over a fresh scratch. Every semantic
/// of [`search`] is therefore unchanged — order, depth, pruning, early exit,
/// and both [`VisitedPolicy`] disciplines.
///
/// # Marks and buffers reset on entry
///
/// The epoch is bumped and every buffer emptied before the walk starts, so a
/// search never inherits marks *or* a frontier — including from one that ended
/// in [`ControlFlow::Break`] with its frontier still loaded. Both are O(1) and
/// there is no way to switch them off; a pass wanting one mark set *shared*
/// across several roots spells that by handing all of those roots to a single
/// call as seeds.
///
/// # Examples
///
/// The shape this exists for: many small searches over one large node space,
/// where the pass allocates for the first search and nothing after it.
///
/// ```
/// use core::ops::ControlFlow;
///
/// use cfglib::{
///     Graph, GraphView, SearchConfig, SearchOrder, SearchScratch,
///     TraversalDirection, Visit, search_with_scratch,
/// };
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
/// let config = SearchConfig::new(SearchOrder::DepthFirst, TraversalDirection::Outgoing);
/// let mut scratch = SearchScratch::new(graph.node_bound());
/// let mut closures = Vec::new();
///
/// for root in graph.node_ids() {
///     let mut reached = Vec::new();
///     let outcome = search_with_scratch(&graph, [root], config, &mut scratch, |node, _depth| {
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
///
/// # Panics
///
/// Panics when `scratch` covers fewer nodes than `graph` — its marks are sized
/// at construction and never resized, so that the pass it serves cannot
/// allocate behind the consumer's back. Also panics on the two inputs
/// [`search`] rejects: a seed that is not a node in `graph`, and
/// [`SearchOrder::BreadthFirst`] paired with [`VisitedPolicy::Path`].
#[must_use]
pub fn search_with_scratch<G: GraphView, B>(
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    config: SearchConfig,
    scratch: &mut SearchScratch,
    visitor: impl FnMut(G::NodeId, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let SearchScratch { marks, buffers } = scratch;
    search_in(graph, seeds, config, marks, buffers, visitor)
}

/// The one search: validate, reset, and dispatch to the discipline's core.
fn search_in<G: GraphView, B>(
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    config: SearchConfig,
    marks: &mut EpochMarks,
    buffers: &mut SearchBuffers,
    visitor: impl FnMut(G::NodeId, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    buffers.reset();
    // `reserve_exact` before the fill, here and in the cores below: it is a
    // no-op on a scratch that already has the room, and on the *fresh* buffers
    // `search` and `search_with_marks` hand over it is what keeps them as
    // cheap as they were. A bare `extend` from an empty buffer reserves a
    // four-element minimum, which for a one-seed search is four times the
    // bytes a `collect` of the same iterator used to ask the allocator for —
    // and at these sizes the allocator charges by the byte.
    let staged = seeds.into_iter().map(DenseId::index);
    buffers.seeds.reserve_exact(staged.size_hint().0);
    buffers.seeds.extend(staged);
    for &seed in &buffers.seeds {
        assert!(seed < graph.node_bound(), "seed node is out of range");
    }
    assert!(
        marks.capacity() >= graph.node_bound(),
        "visited marks cover {} nodes but the graph has {}: size EpochMarks by the node space it is reused over",
        marks.capacity(),
        graph.node_bound()
    );
    marks.reset();

    // The direction becomes a type here and nowhere else: each core is generic
    // over its adjacency axis, so it reads the graph's own adjacency in place
    // instead of branching per expanded node and copying into a buffer.
    match (config.order, config.visited) {
        (SearchOrder::DepthFirst, VisitedPolicy::Global) => by_axis!(
            config.direction,
            depth_first_global(graph, config, marks, buffers, visitor)
        ),
        (SearchOrder::DepthFirst, VisitedPolicy::Path) => by_axis!(
            config.direction,
            depth_first_path(graph, config, marks, buffers, visitor)
        ),
        (SearchOrder::BreadthFirst, VisitedPolicy::Global) => by_axis!(
            config.direction,
            breadth_first_global(graph, config, marks, buffers, visitor)
        ),
        (SearchOrder::BreadthFirst, VisitedPolicy::Path) => {
            panic!(
                "VisitedPolicy::Path requires SearchOrder::DepthFirst: a breadth-first frontier never unwinds, so a path mark could never be removed"
            )
        }
    }
}

/// Whether a node at `depth` may expand under `config`.
fn may_expand(config: SearchConfig, depth: usize) -> bool {
    config.max_depth.is_none_or(|limit| depth < limit)
}

fn depth_first_global<G: GraphView, A: Adjacency, B>(
    axis: A,
    graph: &G,
    config: SearchConfig,
    visited: &mut EpochMarks,
    buffers: &mut SearchBuffers,
    mut visitor: impl FnMut(G::NodeId, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let SearchBuffers {
        seeds,
        frontier: stack,
        adjacent,
        ..
    } = buffers;
    // Reversed so the first seed pops first; the same convention applies to
    // successors below, so adjacency order is expansion order.
    stack.reserve_exact(seeds.len());
    stack.extend(seeds.iter().rev().map(|&seed| (seed, 0)));

    while let Some((node, depth)) = stack.pop() {
        if visited.is_marked(node) {
            continue;
        }
        visited.mark(node);
        let id = G::NodeId::from_index(node);
        match visitor(id, depth) {
            ControlFlow::Break(value) => return Some(value),
            ControlFlow::Continue(Visit::Skip) => continue,
            ControlFlow::Continue(Visit::Descend) => {}
        }
        if !may_expand(config, depth) {
            continue;
        }
        adjacent.clear();
        adjacent.extend(axis.neighbors(graph, id).map(DenseId::index));
        for &successor in adjacent.iter().rev() {
            if !visited.is_marked(successor) {
                stack.push((successor, depth + 1));
            }
        }
    }

    None
}

fn depth_first_path<G: GraphView, A: Adjacency, B>(
    axis: A,
    graph: &G,
    config: SearchConfig,
    on_path: &mut EpochMarks,
    buffers: &mut SearchBuffers,
    mut visitor: impl FnMut(G::NodeId, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let SearchBuffers {
        seeds,
        path: stack,
        adjacent,
        ..
    } = buffers;
    stack.reserve_exact(seeds.len());
    stack.extend(seeds.iter().rev().map(|&seed| PathStep::Enter(seed, 0)));

    while let Some(step) = stack.pop() {
        let (node, depth) = match step {
            // Every `Enter` pushes its own `Leave` before any successor, so
            // this pops exactly when the node's subtree is exhausted — and
            // between two seeds the path is empty again.
            PathStep::Leave(node) => {
                on_path.unmark(node);
                continue;
            }
            PathStep::Enter(node, depth) => (node, depth),
        };
        if on_path.is_marked(node) {
            continue;
        }
        on_path.mark(node);
        stack.push(PathStep::Leave(node));
        let id = G::NodeId::from_index(node);
        match visitor(id, depth) {
            ControlFlow::Break(value) => return Some(value),
            ControlFlow::Continue(Visit::Skip) => continue,
            ControlFlow::Continue(Visit::Descend) => {}
        }
        if !may_expand(config, depth) {
            continue;
        }
        adjacent.clear();
        adjacent.extend(axis.neighbors(graph, id).map(DenseId::index));
        for &successor in adjacent.iter().rev() {
            stack.push(PathStep::Enter(successor, depth + 1));
        }
    }

    None
}

fn breadth_first_global<G: GraphView, A: Adjacency, B>(
    axis: A,
    graph: &G,
    config: SearchConfig,
    visited: &mut EpochMarks,
    buffers: &mut SearchBuffers,
    mut visitor: impl FnMut(G::NodeId, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let SearchBuffers {
        seeds,
        frontier: queue,
        ..
    } = buffers;
    for &seed in &*seeds {
        if !visited.is_marked(seed) {
            visited.mark(seed);
            queue.push((seed, 0));
        }
    }

    // The queue is read through a head cursor rather than drained from the
    // front: a push is `push_back` and reading at `head` is `pop_front`, in
    // the same order. A globally marked walk enqueues each node at most once,
    // so the space left behind the head is bounded by the node count, and one
    // buffer serves this discipline and the depth-first stack alike.
    let mut head = 0;
    while head < queue.len() {
        let (node, depth) = queue[head];
        head += 1;
        let id = G::NodeId::from_index(node);
        match visitor(id, depth) {
            ControlFlow::Break(value) => return Some(value),
            ControlFlow::Continue(Visit::Skip) => continue,
            ControlFlow::Continue(Visit::Descend) => {}
        }
        if !may_expand(config, depth) {
            continue;
        }
        for next in axis.neighbors(graph, id).map(DenseId::index) {
            if !visited.is_marked(next) {
                visited.mark(next);
                queue.push((next, depth + 1));
            }
        }
    }

    None
}

mod events;
pub use events::{BfsEvent, DfsEvent, breadth_first_events, depth_first_events};

#[cfg(test)]
mod tests;
