//! Discipline-configurable search over any [`DirectedGraphView`].
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
use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use crate::graph::traverse::{Adjacency, TraversalDirection, by_axis};
use crate::graph::view::{DenseNodeId, DirectedGraphView};

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

/// Reusable visited marks for repeated searches over one dense node space.
///
/// [`search`] owns its marks, so every call allocates and zeroes a buffer
/// sized to the whole graph. A pass that searches **once per root** over one
/// node space — a nulling closure per grammar nonterminal, a reachable set per
/// definition — pays that O(node count) buffer per root, so its cost scales
/// with the graph even when each search touches a handful of nodes. That is
/// the shape consumers hand-roll an epoch stamp for, and why they decline a
/// substrate that owns its marks.
///
/// `EpochMarks` is that stamp, owned by the caller: each node holds the epoch
/// it was last marked in, so clearing the marks is a bump of the current epoch
/// rather than a walk over the buffer. Allocate one per node space, hand it to
/// every [`search_with_marks`] of the pass, and marking costs O(1) amortized
/// per root instead of O(node count).
///
/// # Cost
///
/// The win is exactly the buffer, so it is largest when each search is small
/// against the graph. Measured on 16,384 nodes whose closures are four nodes
/// each, one search per node: 8.3ms with a fresh buffer per search, 1.5ms over
/// one reused buffer (5.4x). It narrows as searches grow — a search that
/// visits a large fraction of the graph is dominated by the walk, and reuse
/// lands in the noise.
///
/// # Allocation
///
/// The buffer holds one `u32` stamp per node and is the only allocation in a
/// search whose size is O(node count); what remains per call is O(seeds) and
/// O(nodes visited) — the seed vector, the frontier, and (for the depth-first
/// cores alone, which read a node's successors in reverse) one adjacency
/// buffer refilled per expansion. Those are what [`SearchScratch`] owns, for
/// the pass whose searches are so small that the call itself is the cost.
/// Sizing is fixed at
/// construction: a buffer smaller than the graph is a panic, not a resize, so
/// that a marks buffer never silently reallocates in the middle of the pass it
/// exists to keep allocation-free. A buffer **larger** than the graph is fine,
/// which is how one buffer covers a set of graphs — size it by the largest.
///
/// # Examples
///
/// ```
/// use cfglib::EpochMarks;
///
/// let marks = EpochMarks::new(64);
/// assert_eq!(marks.capacity(), 64);
/// ```
#[derive(Debug, Clone)]
pub struct EpochMarks {
    /// Per node, the epoch it was last marked in; marked when it equals
    /// `epoch`.
    stamps: Vec<u32>,
    /// The current epoch. Never zero, so a zero stamp is always unmarked —
    /// which is both the initial state and the un-mark of
    /// [`VisitedPolicy::Path`].
    epoch: u32,
}

impl EpochMarks {
    /// Marks covering `node_count` nodes, with nothing marked.
    #[must_use]
    pub fn new(node_count: usize) -> Self {
        Self {
            stamps: vec![0; node_count],
            epoch: 1,
        }
    }

    /// Return how many nodes these marks cover.
    ///
    /// A [`search_with_marks`] over a graph with more nodes than this panics;
    /// a consumer whose node space grew builds a new buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.stamps.len()
    }

    /// Clear every mark in O(1) by moving to a fresh epoch.
    ///
    /// The buffer is only walked when the epoch would wrap, which needs
    /// `u32::MAX` searches over one buffer.
    pub(crate) fn reset(&mut self) {
        if self.epoch == u32::MAX {
            self.stamps.fill(0);
            self.epoch = 1;
        } else {
            self.epoch += 1;
        }
    }

    /// Whether `index` is marked in the current epoch.
    pub(crate) fn is_marked(&self, index: usize) -> bool {
        self.stamps[index] == self.epoch
    }

    /// Mark `index` for the current epoch.
    pub(crate) fn mark(&mut self, index: usize) {
        self.stamps[index] = self.epoch;
    }

    /// Un-mark `index`, the unwind of [`VisitedPolicy::Path`].
    fn unmark(&mut self, index: usize) {
        self.stamps[index] = 0;
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
/// Nodes are stored as the dense indices [`DenseNodeId`] guarantees they are,
/// and the cores convert at the boundary. That is what keeps the buffers — and
/// so [`SearchScratch`] — free of a node-id type parameter: one scratch type
/// serves every [`DirectedGraphView`], and one scratch instance can be reused
/// across graphs whose ids differ.
#[derive(Debug, Clone, Default)]
struct SearchBuffers {
    /// The seeds, materialised so they can be validated once and pushed in
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
/// use cfglib::{DirectedGraph, SearchConfig, SearchOrder, TraversalDirection, Visit, search};
///
/// //     a -> b -> d
/// //     a -> c
/// let mut graph = DirectedGraph::<&str, ()>::new();
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
pub fn search<G: DirectedGraphView, B>(
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    config: SearchConfig,
    visitor: impl FnMut(G::NodeId, usize) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let mut scratch = SearchScratch::new(graph.node_count());
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
///     DirectedGraph, DirectedGraphView, EpochMarks, SearchConfig, SearchOrder,
///     TraversalDirection, Visit, search_with_marks,
/// };
///
/// //     a -> b -> c,  d -> c
/// let mut graph = DirectedGraph::<&str, ()>::new();
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let c = graph.add_node("c");
/// let d = graph.add_node("d");
/// graph.add_edge(a, b, ());
/// graph.add_edge(b, c, ());
/// graph.add_edge(d, c, ());
///
/// let config = SearchConfig::new(SearchOrder::BreadthFirst, TraversalDirection::Outgoing);
/// let mut marks = EpochMarks::new(graph.node_count());
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
pub fn search_with_marks<G: DirectedGraphView, B>(
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
///     DirectedGraph, DirectedGraphView, SearchConfig, SearchOrder, SearchScratch,
///     TraversalDirection, Visit, search_with_scratch,
/// };
///
/// //     a -> b -> c,  d -> c
/// let mut graph = DirectedGraph::<&str, ()>::new();
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let c = graph.add_node("c");
/// let d = graph.add_node("d");
/// graph.add_edge(a, b, ());
/// graph.add_edge(b, c, ());
/// graph.add_edge(d, c, ());
///
/// let config = SearchConfig::new(SearchOrder::DepthFirst, TraversalDirection::Outgoing);
/// let mut scratch = SearchScratch::new(graph.node_count());
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
pub fn search_with_scratch<G: DirectedGraphView, B>(
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
fn search_in<G: DirectedGraphView, B>(
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
    let staged = seeds.into_iter().map(DenseNodeId::index);
    buffers.seeds.reserve_exact(staged.size_hint().0);
    buffers.seeds.extend(staged);
    for &seed in &buffers.seeds {
        assert!(seed < graph.node_count(), "seed node is out of range");
    }
    assert!(
        marks.capacity() >= graph.node_count(),
        "visited marks cover {} nodes but the graph has {}: size EpochMarks by the node space it is reused over",
        marks.capacity(),
        graph.node_count()
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

fn depth_first_global<G: DirectedGraphView, A: Adjacency, B>(
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
        adjacent.extend(axis.neighbors(graph, id).map(DenseNodeId::index));
        for &successor in adjacent.iter().rev() {
            if !visited.is_marked(successor) {
                stack.push((successor, depth + 1));
            }
        }
    }

    None
}

fn depth_first_path<G: DirectedGraphView, A: Adjacency, B>(
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
        adjacent.extend(axis.neighbors(graph, id).map(DenseNodeId::index));
        for &successor in adjacent.iter().rev() {
            stack.push(PathStep::Enter(successor, depth + 1));
        }
    }

    None
}

fn breadth_first_global<G: DirectedGraphView, A: Adjacency, B>(
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
        for next in axis.neighbors(graph, id).map(DenseNodeId::index) {
            if !visited.is_marked(next) {
                visited.mark(next);
                queue.push((next, depth + 1));
            }
        }
    }

    None
}

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
/// [`search`] answers questions about nodes; this answers questions whose
/// output order *is* the traversal — reporting a cycle at the edge that
/// closes it, emitting a nesting structure on discover/finish pairs,
/// classifying edges. Doing that on top of a node-yielding traversal is what
/// forces a consumer to hand-roll a walk.
///
/// # Order
///
/// The event order is deterministic and part of the API. Successors are
/// examined in adjacency order (equivalently, the rev-push convention of
/// [`depth_first_preorder`](super::traverse::depth_first_preorder)). For a
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
/// use cfglib::{DfsEvent, DirectedGraph, TraversalDirection, depth_first_events};
///
/// // a -> b -> c -> a, plus a chord a -> c.
/// let mut graph = DirectedGraph::<&str, ()>::new();
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
pub fn depth_first_events<G: DirectedGraphView, B>(
    graph: &G,
    start: G::NodeId,
    direction: TraversalDirection,
    on_event: impl FnMut(DfsEvent<G::NodeId>) -> ControlFlow<B>,
) -> Option<B> {
    by_axis!(direction, events_from(graph, start, on_event))
}

fn events_from<G: DirectedGraphView, A: Adjacency, B>(
    axis: A,
    graph: &G,
    start: G::NodeId,
    mut on_event: impl FnMut(DfsEvent<G::NodeId>) -> ControlFlow<B>,
) -> Option<B> {
    assert!(
        start.index() < graph.node_count(),
        "start node is out of range"
    );

    macro_rules! emit {
        ($event:expr) => {
            if let ControlFlow::Break(value) = on_event($event) {
                return Some(value);
            }
        };
    }

    let mut color = vec![Color::White; graph.node_count()];
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
    emit!(DfsEvent::Discover(start, 0));
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
                    emit!(DfsEvent::TreeEdge(node, successor));
                    color[successor.index()] = Color::Gray;
                    emit!(DfsEvent::Discover(successor, depth + 1));
                    descend!(successor, depth + 1);
                }
                Color::Gray => emit!(DfsEvent::BackEdge(node, successor)),
                Color::Black => emit!(DfsEvent::ForwardOrCross(node, successor)),
            }
            continue;
        }

        color[node.index()] = Color::Black;
        emit!(DfsEvent::Finish(node));
        let Some(finished) = stack.pop() else { break };
        arena.truncate(finished.start);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::directed::{DirectedGraph, NodeId};
    use alloc::vec;

    /// `a -> b -> d`, `a -> c`: depth-first reaches `d` before `c`,
    /// breadth-first the other way round.
    fn fork() -> (DirectedGraph<(), ()>, [NodeId; 4]) {
        let mut graph = DirectedGraph::<(), ()>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(a, c, ());
        graph.add_edge(b, d, ());
        (graph, [a, b, c, d])
    }

    /// `a -> b`, `a -> c`, `b -> d`, `c -> d`: two paths reach `d`.
    fn diamond() -> (DirectedGraph<(), ()>, [NodeId; 4]) {
        let mut graph = DirectedGraph::<(), ()>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(a, c, ());
        graph.add_edge(b, d, ());
        graph.add_edge(c, d, ());
        (graph, [a, b, c, d])
    }

    fn config(order: SearchOrder) -> SearchConfig {
        SearchConfig::new(order, TraversalDirection::Outgoing)
    }

    /// Visit order under `config`, seeded at `seeds`, descending everywhere.
    fn visit_order<G: DirectedGraphView>(
        graph: &G,
        seeds: impl IntoIterator<Item = G::NodeId>,
        config: SearchConfig,
    ) -> Vec<(G::NodeId, usize)> {
        let mut order = Vec::new();
        let outcome = search(graph, seeds, config, |node, depth| {
            order.push((node, depth));
            ControlFlow::<(), _>::Continue(Visit::Descend)
        });
        assert_eq!(outcome, None, "a descending search never breaks");
        order
    }

    fn nodes<N: Copy>(order: &[(N, usize)]) -> Vec<N> {
        order.iter().map(|&(node, _)| node).collect()
    }

    /// The same, over marks the caller owns.
    fn marked_order<G: DirectedGraphView>(
        graph: &G,
        seeds: impl IntoIterator<Item = G::NodeId>,
        config: SearchConfig,
        marks: &mut EpochMarks,
    ) -> Vec<(G::NodeId, usize)> {
        let mut order = Vec::new();
        let outcome = search_with_marks(graph, seeds, config, marks, |node, depth| {
            order.push((node, depth));
            ControlFlow::<(), _>::Continue(Visit::Descend)
        });
        assert_eq!(outcome, None, "a descending search never breaks");
        order
    }

    /// The same, over a scratch the caller owns.
    fn scratch_order<G: DirectedGraphView>(
        graph: &G,
        seeds: impl IntoIterator<Item = G::NodeId>,
        config: SearchConfig,
        scratch: &mut SearchScratch,
    ) -> Vec<(G::NodeId, usize)> {
        let mut order = Vec::new();
        let outcome = search_with_scratch(graph, seeds, config, scratch, |node, depth| {
            order.push((node, depth));
            ControlFlow::<(), _>::Continue(Visit::Descend)
        });
        assert_eq!(outcome, None, "a descending search never breaks");
        order
    }

    /// Every discipline `search` accepts.
    fn disciplines() -> [SearchConfig; 3] {
        [
            config(SearchOrder::DepthFirst),
            config(SearchOrder::BreadthFirst),
            config(SearchOrder::DepthFirst).with_visited(VisitedPolicy::Path),
        ]
    }

    #[test]
    fn first_match_depends_on_the_search_order() {
        let (graph, [a, b, c, d]) = fork();
        let first_leaf = |order| {
            search(&graph, [a], config(order), |node, _| {
                if graph.successors(node).count() == 0 {
                    return ControlFlow::Break(node);
                }
                ControlFlow::Continue(Visit::Descend)
            })
        };

        // The same graph, the same seed, two disciplines, two answers.
        assert_eq!(first_leaf(SearchOrder::DepthFirst), Some(d));
        assert_eq!(first_leaf(SearchOrder::BreadthFirst), Some(c));

        assert_eq!(
            nodes(&visit_order(&graph, [a], config(SearchOrder::DepthFirst))),
            vec![a, b, d, c]
        );
        assert_eq!(
            nodes(&visit_order(&graph, [a], config(SearchOrder::BreadthFirst))),
            vec![a, b, c, d]
        );
    }

    #[test]
    fn path_policy_reports_every_path_to_a_shared_node() {
        let (graph, [a, b, c, d]) = diamond();
        // Globally marked, `d` is one node reached once.
        assert_eq!(
            nodes(&visit_order(&graph, [a], config(SearchOrder::DepthFirst))),
            vec![a, b, d, c]
        );
        // Path-marked, `d` is reported once per route into it — the shape of
        // an ambiguous base or a symbol reachable through two imports.
        let paths = visit_order(
            &graph,
            [a],
            config(SearchOrder::DepthFirst).with_visited(VisitedPolicy::Path),
        );
        assert_eq!(nodes(&paths), vec![a, b, d, c, d]);
        assert_eq!(paths[2], (d, 2));
        assert_eq!(paths[4], (d, 2));
    }

    #[test]
    fn skip_prunes_the_subtree_under_both_policies() {
        let (graph, [a, b, c, d]) = fork();
        for visited in [VisitedPolicy::Global, VisitedPolicy::Path] {
            let mut order = Vec::new();
            let outcome = search(
                &graph,
                [a],
                config(SearchOrder::DepthFirst).with_visited(visited),
                |node, _| {
                    order.push(node);
                    if node == b {
                        return ControlFlow::<(), _>::Continue(Visit::Skip);
                    }
                    ControlFlow::Continue(Visit::Descend)
                },
            );
            // `b` is visited, `d` (only reachable through it) is not.
            assert_eq!(outcome, None);
            assert_eq!(order, vec![a, b, c], "{visited:?}");
        }
        assert_eq!(
            nodes(&visit_order(&graph, [a], config(SearchOrder::DepthFirst))),
            vec![a, b, d, c],
            "without the skip, `d` is reached"
        );
    }

    #[test]
    fn seeds_are_searched_in_the_order_given() {
        let (graph, [a, b, c, d]) = fork();
        // Seeding `c` first puts it before the whole `a` subtree.
        assert_eq!(
            nodes(&visit_order(
                &graph,
                [c, a],
                config(SearchOrder::DepthFirst)
            )),
            vec![c, a, b, d]
        );
        assert_eq!(
            nodes(&visit_order(
                &graph,
                [a, c],
                config(SearchOrder::DepthFirst)
            )),
            vec![a, b, d, c]
        );
        // Breadth-first interleaves the seeds' levels, still in seed order.
        assert_eq!(
            nodes(&visit_order(
                &graph,
                [b, a],
                config(SearchOrder::BreadthFirst)
            )),
            vec![b, a, d, c]
        );
        // Seeds are at depth 0 even when another seed reaches them deeper.
        assert_eq!(
            visit_order(&graph, [d, a], config(SearchOrder::DepthFirst)),
            vec![(d, 0), (a, 0), (b, 1), (c, 1)]
        );
    }

    #[test]
    fn duplicate_seeds_dedup_globally_and_repeat_on_paths() {
        let (graph, [a, b, c, d]) = fork();
        assert_eq!(
            nodes(&visit_order(
                &graph,
                [a, a],
                config(SearchOrder::DepthFirst)
            )),
            vec![a, b, d, c]
        );
        // Each seed starts a fresh path context, so the second one searches
        // again from an empty path.
        assert_eq!(
            nodes(&visit_order(
                &graph,
                [a, a],
                config(SearchOrder::DepthFirst).with_visited(VisitedPolicy::Path)
            )),
            vec![a, b, d, c, a, b, d, c]
        );
    }

    #[test]
    fn cycles_terminate_under_both_policies() {
        // a -> b -> c -> a, with a self-edge on c.
        let mut graph = DirectedGraph::<(), ()>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, a, ());
        graph.add_edge(c, c, ());

        assert_eq!(
            nodes(&visit_order(&graph, [a], config(SearchOrder::DepthFirst))),
            vec![a, b, c]
        );
        assert_eq!(
            nodes(&visit_order(&graph, [a], config(SearchOrder::BreadthFirst))),
            vec![a, b, c]
        );
        // The path guard refuses to re-enter a node already on the path, so
        // the walk terminates without a global mark.
        assert_eq!(
            nodes(&visit_order(
                &graph,
                [a],
                config(SearchOrder::DepthFirst).with_visited(VisitedPolicy::Path)
            )),
            vec![a, b, c]
        );
    }

    #[test]
    fn max_depth_bounds_expansion_not_visiting() {
        // A chain a -> b -> c -> d.
        let mut graph = DirectedGraph::<(), ()>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, d, ());

        for order in [SearchOrder::DepthFirst, SearchOrder::BreadthFirst] {
            assert_eq!(
                visit_order(&graph, [a], config(order).with_max_depth(0)),
                vec![(a, 0)],
                "{order:?}"
            );
            // The node at the bound is visited; its successor is not
            // discovered through it.
            assert_eq!(
                visit_order(&graph, [a], config(order).with_max_depth(2)),
                vec![(a, 0), (b, 1), (c, 2)],
                "{order:?}"
            );
            assert_eq!(
                visit_order(&graph, [a], config(order).with_max_depth(9)),
                vec![(a, 0), (b, 1), (c, 2), (d, 3)],
                "{order:?}"
            );
        }
    }

    #[test]
    fn break_returns_immediately() {
        let (graph, [a, b, _c, _d]) = fork();
        let mut seen = Vec::new();
        let found = search(
            &graph,
            [a],
            config(SearchOrder::DepthFirst),
            |node, depth| {
                seen.push(node);
                if node == b {
                    return ControlFlow::Break(depth);
                }
                ControlFlow::Continue(Visit::Descend)
            },
        );

        assert_eq!(found, Some(1));
        assert_eq!(seen, vec![a, b], "nothing after the break is visited");
    }

    #[test]
    fn the_incoming_direction_searches_predecessors() {
        let (graph, [a, b, _c, d]) = fork();
        assert_eq!(
            nodes(&visit_order(
                &graph,
                [d],
                SearchConfig::new(SearchOrder::DepthFirst, TraversalDirection::Incoming)
            )),
            vec![d, b, a]
        );
    }

    #[test]
    fn every_discipline_walks_the_incoming_axis() {
        // The direction is resolved to a type once per call, so each
        // discipline reaches its reverse axis through its own dispatch arm:
        // a mixed-up arm would silently walk the graph the other way round.
        let (graph, [a, b, c, d]) = diamond();
        let reverse = |order| SearchConfig::new(order, TraversalDirection::Incoming);

        assert_eq!(
            nodes(&visit_order(
                &graph,
                [d],
                reverse(SearchOrder::BreadthFirst)
            )),
            vec![d, b, c, a]
        );
        // Path marks un-mark on unwind, so `a` is reported once per route.
        assert_eq!(
            nodes(&visit_order(
                &graph,
                [d],
                reverse(SearchOrder::DepthFirst).with_visited(VisitedPolicy::Path)
            )),
            vec![d, b, a, c, a]
        );
        // And the forward axis of the same graph is a different answer.
        assert_eq!(
            nodes(&visit_order(&graph, [a], config(SearchOrder::BreadthFirst))),
            vec![a, b, c, d]
        );
    }

    #[test]
    #[should_panic(expected = "VisitedPolicy::Path requires SearchOrder::DepthFirst")]
    fn breadth_first_with_path_marks_is_rejected() {
        let (graph, [a, _, _, _]) = fork();
        let _ = search(
            &graph,
            [a],
            config(SearchOrder::BreadthFirst).with_visited(VisitedPolicy::Path),
            |_, _| ControlFlow::<(), _>::Continue(Visit::Descend),
        );
    }

    #[test]
    #[should_panic(expected = "seed node is out of range")]
    fn an_out_of_range_seed_panics() {
        let (graph, _) = fork();
        let _ = search(
            &graph,
            [NodeId::from_index(9)],
            config(SearchOrder::DepthFirst),
            |_, _| ControlFlow::<(), _>::Continue(Visit::Descend),
        );
    }

    #[test]
    fn reused_marks_equal_one_fresh_search_per_root() {
        let (graph, roots) = diamond();
        for config in disciplines() {
            let mut marks = EpochMarks::new(graph.node_count());
            // Twice around, so every root is also searched over marks a full
            // pass has already written.
            for &root in roots.iter().chain(roots.iter()) {
                assert_eq!(
                    marked_order(&graph, [root], config, &mut marks),
                    visit_order(&graph, [root], config),
                    "{config:?} from {root:?}"
                );
            }
            // Seed handling reads the same marks, so multi-seed searches (and
            // the duplicate-seed rules) have to survive the reuse too.
            for seeds in [[roots[3], roots[0]], [roots[0], roots[0]]] {
                assert_eq!(
                    marked_order(&graph, seeds, config, &mut marks),
                    visit_order(&graph, seeds, config),
                    "{config:?} from {seeds:?}"
                );
            }
        }
    }

    #[test]
    fn a_reused_search_never_inherits_the_previous_marks() {
        let (graph, [a, b, c, d]) = diamond();
        let dfs = config(SearchOrder::DepthFirst);
        let path = dfs.with_visited(VisitedPolicy::Path);
        let mut marks = EpochMarks::new(graph.node_count());

        // A pass over the whole graph, then a search of a part of it: the
        // second search sees none of the first one's marks.
        assert_eq!(
            nodes(&marked_order(&graph, [a], dfs, &mut marks)),
            vec![a, b, d, c]
        );
        assert_eq!(
            nodes(&marked_order(&graph, [b], dfs, &mut marks)),
            vec![b, d]
        );
        assert_eq!(
            nodes(&marked_order(&graph, [c], dfs, &mut marks)),
            vec![c, d]
        );

        // A search that broke mid-walk leaves marks set under both policies —
        // and the walk after it still starts from a clean set.
        for config in [dfs, path] {
            let found = search_with_marks(&graph, [a], config, &mut marks, |node, _| {
                if node == b {
                    return ControlFlow::Break(node);
                }
                ControlFlow::Continue(Visit::Descend)
            });
            assert_eq!(found, Some(b), "{config:?}");
            assert_eq!(
                nodes(&marked_order(&graph, [a], dfs, &mut marks)),
                vec![a, b, d, c],
                "{config:?}"
            );
        }
    }

    #[test]
    fn marks_stay_correct_across_an_epoch_wrap() {
        let (graph, [a, b, c, d]) = diamond();
        let dfs = config(SearchOrder::DepthFirst);
        let mut marks = EpochMarks::new(graph.node_count());
        // The state one search short of the wrap, carrying a stamp from the
        // last time the epoch was 1 — the one value bumping alone would not
        // invalidate, so only clearing the buffer keeps `c` visitable.
        marks.epoch = u32::MAX - 1;
        marks.stamps[c.index()] = 1;

        // This search takes the epoch to its last value and never touches `c`.
        assert_eq!(
            nodes(&marked_order(&graph, [b], dfs, &mut marks)),
            vec![b, d]
        );
        assert_eq!(marks.epoch, u32::MAX);
        // This one wraps.
        assert_eq!(
            nodes(&marked_order(&graph, [a], dfs, &mut marks)),
            vec![a, b, d, c]
        );
        assert_eq!(marks.epoch, 1, "the epoch wrapped to its first value");
        // And the buffer keeps working on the far side of the wrap.
        assert_eq!(
            nodes(&marked_order(&graph, [a], dfs, &mut marks)),
            vec![a, b, d, c]
        );
        assert_eq!(marks.epoch, 2);
    }

    #[test]
    fn marks_larger_than_the_graph_are_accepted() {
        // One buffer sized by the largest node space serves the smaller ones.
        let (graph, [a, b, c, d]) = diamond();
        let mut marks = EpochMarks::new(graph.node_count() + 8);
        assert_eq!(
            nodes(&marked_order(
                &graph,
                [a],
                config(SearchOrder::DepthFirst),
                &mut marks
            )),
            vec![a, b, d, c]
        );
    }

    #[test]
    #[should_panic(expected = "visited marks cover 2 nodes but the graph has 4")]
    fn marks_smaller_than_the_graph_panic() {
        let (graph, [a, _, _, _]) = diamond();
        let mut marks = EpochMarks::new(2);
        let _ = search_with_marks(
            &graph,
            [a],
            config(SearchOrder::DepthFirst),
            &mut marks,
            |_, _| ControlFlow::<(), _>::Continue(Visit::Descend),
        );
    }

    #[test]
    fn a_reused_scratch_equals_one_fresh_search_per_root() {
        let (graph, roots) = diamond();
        for config in disciplines() {
            let mut scratch = SearchScratch::new(graph.node_count());
            // Twice around, so every root is also searched over buffers a full
            // pass has already written.
            for &root in roots.iter().chain(roots.iter()) {
                assert_eq!(
                    scratch_order(&graph, [root], config, &mut scratch),
                    visit_order(&graph, [root], config),
                    "{config:?} from {root:?}"
                );
            }
            // Seeds are collected into the scratch too, so multi-seed searches
            // (and the duplicate-seed rules) have to survive the reuse.
            for seeds in [[roots[3], roots[0]], [roots[0], roots[0]]] {
                assert_eq!(
                    scratch_order(&graph, seeds, config, &mut scratch),
                    visit_order(&graph, seeds, config),
                    "{config:?} from {seeds:?}"
                );
            }
        }
    }

    #[test]
    fn one_scratch_serves_every_discipline_in_turn() {
        // The two globally marked cores share one frontier buffer — a stack
        // for the depth-first walk, a queue for the breadth-first one — so a
        // pass that changes discipline mid-way is the case that would catch a
        // buffer left in the other core's shape.
        let (graph, [a, b, c, d]) = diamond();
        let mut scratch = SearchScratch::new(graph.node_count());
        for _ in 0..2 {
            assert_eq!(
                nodes(&scratch_order(
                    &graph,
                    [a],
                    config(SearchOrder::DepthFirst),
                    &mut scratch
                )),
                vec![a, b, d, c]
            );
            assert_eq!(
                nodes(&scratch_order(
                    &graph,
                    [a],
                    config(SearchOrder::BreadthFirst),
                    &mut scratch
                )),
                vec![a, b, c, d]
            );
            assert_eq!(
                nodes(&scratch_order(
                    &graph,
                    [a],
                    config(SearchOrder::DepthFirst).with_visited(VisitedPolicy::Path),
                    &mut scratch
                )),
                vec![a, b, d, c, d]
            );
        }
    }

    #[test]
    fn a_reused_scratch_never_inherits_the_previous_frontier() {
        let (graph, [a, b, c, d]) = diamond();
        let dfs = config(SearchOrder::DepthFirst);
        let mut scratch = SearchScratch::new(graph.node_count());

        // A search that broke mid-walk returns with its frontier still loaded
        // — under every discipline — and the walk after it still starts from
        // an empty one.
        for config in disciplines() {
            let found = search_with_scratch(&graph, [a], config, &mut scratch, |node, _| {
                if node == b {
                    return ControlFlow::Break(node);
                }
                ControlFlow::Continue(Visit::Descend)
            });
            assert_eq!(found, Some(b), "{config:?}");
            assert_eq!(
                nodes(&scratch_order(&graph, [a], dfs, &mut scratch)),
                vec![a, b, d, c],
                "{config:?}"
            );
        }

        // A whole-graph pass followed by a search of a part of it: neither the
        // marks nor the buffers of the first survive into the second.
        assert_eq!(
            nodes(&scratch_order(&graph, [b], dfs, &mut scratch)),
            vec![b, d]
        );
        assert_eq!(
            nodes(&scratch_order(&graph, [c], dfs, &mut scratch)),
            vec![c, d]
        );
    }

    #[test]
    fn a_scratch_larger_than_the_graph_is_accepted() {
        // One scratch sized by the largest node space serves the smaller ones.
        let (graph, [a, b, c, d]) = diamond();
        let mut scratch = SearchScratch::new(graph.node_count() + 8);
        assert_eq!(scratch.capacity(), graph.node_count() + 8);
        assert_eq!(
            nodes(&scratch_order(
                &graph,
                [a],
                config(SearchOrder::DepthFirst),
                &mut scratch
            )),
            vec![a, b, d, c]
        );
    }

    #[test]
    #[should_panic(expected = "visited marks cover 2 nodes but the graph has 4")]
    fn a_scratch_smaller_than_the_graph_panics() {
        let (graph, [a, _, _, _]) = diamond();
        let mut scratch = SearchScratch::new(2);
        let _ = search_with_scratch(
            &graph,
            [a],
            config(SearchOrder::DepthFirst),
            &mut scratch,
            |_, _| ControlFlow::<(), _>::Continue(Visit::Descend),
        );
    }

    /// `a -> b`, `a -> d`, `a -> c`, `b -> c`, `c -> a`, `d -> c`: one graph
    /// carrying all four edge classes.
    fn classified() -> (DirectedGraph<(), ()>, [NodeId; 4]) {
        let mut graph = DirectedGraph::<(), ()>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(a, d, ());
        graph.add_edge(a, c, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, a, ());
        graph.add_edge(d, c, ());
        (graph, [a, b, c, d])
    }

    fn events(graph: &DirectedGraph<(), ()>, start: NodeId) -> Vec<DfsEvent<NodeId>> {
        let mut log = Vec::new();
        let outcome = depth_first_events(graph, start, TraversalDirection::Outgoing, |event| {
            log.push(event);
            ControlFlow::<()>::Continue(())
        });
        assert_eq!(outcome, None);
        log
    }

    #[test]
    fn depth_first_events_classify_every_edge_in_a_pinned_order() {
        let (graph, [a, b, c, d]) = classified();
        assert_eq!(
            events(&graph, a),
            vec![
                DfsEvent::Discover(a, 0),
                DfsEvent::TreeEdge(a, b),
                DfsEvent::Discover(b, 1),
                DfsEvent::TreeEdge(b, c),
                DfsEvent::Discover(c, 2),
                // c -> a closes the cycle: `a` is an ancestor on the path.
                DfsEvent::BackEdge(c, a),
                DfsEvent::Finish(c),
                DfsEvent::Finish(b),
                DfsEvent::TreeEdge(a, d),
                DfsEvent::Discover(d, 1),
                // d -> c is a cross edge into a finished sibling subtree.
                DfsEvent::ForwardOrCross(d, c),
                DfsEvent::Finish(d),
                // a -> c is a forward edge to a finished descendant.
                DfsEvent::ForwardOrCross(a, c),
                DfsEvent::Finish(a),
            ]
        );
    }

    #[test]
    fn depth_first_events_report_self_edges_as_back_edges() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let only = graph.add_node(());
        graph.add_edge(only, only, ());
        assert_eq!(
            events(&graph, only),
            vec![
                DfsEvent::Discover(only, 0),
                DfsEvent::BackEdge(only, only),
                DfsEvent::Finish(only),
            ]
        );
    }

    #[test]
    fn depth_first_events_only_cover_the_reachable_set() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let unreached = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(unreached, a, ());

        assert_eq!(
            events(&graph, a),
            vec![
                DfsEvent::Discover(a, 0),
                DfsEvent::TreeEdge(a, b),
                DfsEvent::Discover(b, 1),
                DfsEvent::Finish(b),
                DfsEvent::Finish(a),
            ],
            "a predecessor-only node produces no events"
        );
    }

    #[test]
    fn depth_first_events_break_stops_the_walk() {
        let (graph, [a, b, c, _d]) = classified();
        let mut log = Vec::new();
        let found = depth_first_events(&graph, a, TraversalDirection::Outgoing, |event| {
            log.push(event);
            match event {
                DfsEvent::BackEdge(from, to) => ControlFlow::Break((from, to)),
                _ => ControlFlow::Continue(()),
            }
        });

        assert_eq!(found, Some((c, a)));
        assert_eq!(
            log,
            vec![
                DfsEvent::Discover(a, 0),
                DfsEvent::TreeEdge(a, b),
                DfsEvent::Discover(b, 1),
                DfsEvent::TreeEdge(b, c),
                DfsEvent::Discover(c, 2),
                DfsEvent::BackEdge(c, a),
            ]
        );
    }
}
