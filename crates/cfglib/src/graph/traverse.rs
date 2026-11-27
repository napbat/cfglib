//! Generic directed-graph traversals and ordering algorithms.
//!
//! The walks here, and those of [`search`](super::search),
//! [`edge_traverse`](super::edge_traverse) and the graph store's orderings,
//! are thin wrappers over two crate-private cores: a breadth-first core that
//! expands a queue level by level, and a depth-first core that keeps one frame
//! per open node over one arena of pending steps. A wrapper only says what it
//! records (an order, a distance, a parent, an event) and which marks it keeps
//! (visited flags, epoch marks, marks per path, the three colors of an edge
//! classification). Both cores are generic over what a step is — a neighbor
//! of a node walk, an edge of an edge walk — and over the adjacency axis, so
//! one loop serves both directions and both kinds of walk.
//!
//! Four loops stay out of the cores on measured grounds, each documented
//! where it lives: [`depth_first_preorder`], which needs no frames;
//! [`reachable`], whose order is free, so a stack holds only the frontier;
//! [`topological_sort`], whose Kahn loop is not a search; and the
//! bidirectional [`shortest_path`], which advances two searches one edge at a
//! time. Elsewhere, the depth-first walks whose bookkeeping doubles as their
//! marks keep their own loops too, with their measurements: the
//! [`canonical_order`](crate::canonical_order) preorder, Tarjan's walk of the
//! strongly connected components, and Havlak's cycle walk behind
//! [`LoopForest::compute_cycles`](crate::LoopForest::compute_cycles) and the
//! weak topological order.

extern crate alloc;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use crate::graph::search::Visit;
use crate::graph::view::{DenseId, Direction, GraphView, assert_in_bound};

/// One adjacency axis of a [`GraphView`], as a **type** rather than a
/// value.
///
/// [`Direction`] is a public parameter of every walk here, and reading it
/// *inside* a walk costs more than the branch it looks like: `successors` and
/// `predecessors` are distinct opaque iterator types, so a core that decides
/// per expanded node cannot hold either of them. It has to materialize the
/// adjacency into a buffer first and then test and push out of that copy.
///
/// A zero-sized axis moves the decision to monomorphisation: a core generic
/// over `A: Adjacency` names one concrete iterator type, so it can iterate the
/// graph's own adjacency **in place**. The public functions keep their
/// [`Direction`] argument and turn it into a type exactly once, at entry, with
/// [`by_axis!`]. The breadth-first core reads the axis directly and keeps no
/// buffer at all, which measured 1.2x to 1.9x on the pinned fixtures against
/// a core that copied the adjacency. The depth-first core copies each
/// expanded node's steps once, into the arena its frames share, because a
/// frame examines its steps one at a time across the walk.
///
/// The price of the axis is one monomorphisation of every core per direction,
/// which is why it stays an implementation detail rather than becoming a
/// second public spelling of a direction consumers already pass.
pub(crate) trait Adjacency: Copy {
    /// The axis that walks the same edges the other way.
    type Against: Adjacency<Against = Self>;

    /// The opposite axis, for a core that reads one axis upstream and the
    /// other downstream and takes only one of them as its leading argument.
    fn against(self) -> Self::Against;

    /// Iterate `node`'s neighbors along this axis.
    ///
    /// Implementations are inlined into the cores, so the iterator is built
    /// in the core's own frame instead of returned through a hidden slot and
    /// copied (see `Graph::adjacent`).
    fn neighbors<G: GraphView>(
        self,
        graph: &G,
        node: G::NodeId,
    ) -> impl Iterator<Item = G::NodeId> + '_;
}

/// The forward axis: [`GraphView::successors`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Outgoing;

/// The reverse axis: [`GraphView::predecessors`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Incoming;

impl Adjacency for Outgoing {
    type Against = Incoming;

    #[inline]
    fn against(self) -> Incoming {
        Incoming
    }

    #[inline]
    fn neighbors<G: GraphView>(
        self,
        graph: &G,
        node: G::NodeId,
    ) -> impl Iterator<Item = G::NodeId> + '_ {
        graph.successors(node)
    }
}

impl Adjacency for Incoming {
    type Against = Outgoing;

    #[inline]
    fn against(self) -> Outgoing {
        Outgoing
    }

    #[inline]
    fn neighbors<G: GraphView>(
        self,
        graph: &G,
        node: G::NodeId,
    ) -> impl Iterator<Item = G::NodeId> + '_ {
        graph.predecessors(node)
    }
}

/// Call an axis-generic core, resolving a [`Direction`] **once**.
///
/// The core takes its axis as a leading argument, so the value becomes a type
/// at one place per public walk — its entry — and the two arms cannot drift
/// apart across the dozen walks that need them. Written as a macro rather than
/// a helper because the arms differ only in a *type*, which no value-taking
/// helper can carry.
macro_rules! by_axis {
    ($direction:expr, $core:ident($($argument:expr),* $(,)?)) => {
        match $direction {
            $crate::graph::view::Direction::Forward => {
                $core($crate::graph::traverse::Outgoing, $($argument),*)
            }
            $crate::graph::view::Direction::Backward => {
                $core($crate::graph::traverse::Incoming, $($argument),*)
            }
        }
    };
}

pub(crate) use by_axis;

/// What one step of a walk core is, where it leads, and how the walk names
/// its nodes.
///
/// A node walk steps to a neighbor, so a step is the neighbor itself. An edge
/// walk steps along an edge ([`Edges`](super::edge_traverse::Edges)), so a
/// step is the edge and leads to its far end along the axis. Both name nodes
/// by the view's own identity, so a walk records them without converting.
/// [`Indexed`] names nodes and steps by dense index instead, for a walk whose
/// buffers outlive one graph and so cannot carry an identity type.
pub(crate) trait Steps<G: GraphView>: Copy {
    /// How the walk names a node.
    type Node: Copy;
    /// How the walk names a step.
    type Step: Copy;

    /// The steps out of `node`, in adjacency order.
    fn steps(self, graph: &G, node: Self::Node) -> impl Iterator<Item = Self::Step> + '_;

    /// The node `step` leads to.
    fn target(self, graph: &G, step: Self::Step) -> Self::Node;

    /// The dense index of `node`, which the walk's marks are keyed by.
    fn index(node: Self::Node) -> usize;
}

impl<G: GraphView, A: Adjacency> Steps<G> for A {
    type Node = G::NodeId;
    type Step = G::NodeId;

    #[inline]
    fn steps(self, graph: &G, node: G::NodeId) -> impl Iterator<Item = G::NodeId> + '_ {
        self.neighbors(graph, node)
    }

    #[inline]
    fn target(self, _graph: &G, step: G::NodeId) -> G::NodeId {
        step
    }

    #[inline]
    fn index(node: G::NodeId) -> usize {
        node.index()
    }
}

/// The node steps of an axis by dense index, for a walk over a scratch that
/// serves graphs of every identity type.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Indexed<A>(pub(crate) A);

impl<G: GraphView, A: Adjacency> Steps<G> for Indexed<A> {
    type Node = usize;
    type Step = usize;

    #[inline]
    fn steps(self, graph: &G, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.0
            .neighbors(graph, G::NodeId::from_index(node))
            .map(DenseId::index)
    }

    #[inline]
    fn target(self, _graph: &G, step: usize) -> usize {
        step
    }

    #[inline]
    fn index(node: usize) -> usize {
        node
    }
}

/// A depth-first visitor that records the postorder.
struct Postorder<N>(Vec<N>);

impl<N: Copy> DepthFirstVisitor<N, N, ()> for Postorder<N> {
    #[inline]
    fn discover(&mut self, _node: N, _depth: usize) -> ControlFlow<(), Visit> {
        ControlFlow::Continue(Visit::Descend)
    }

    #[inline]
    fn finish(&mut self, node: N) -> ControlFlow<()> {
        self.0.push(node);
        ControlFlow::Continue(())
    }
}

/// Return nodes in depth-first preorder from `start`.
///
/// Successors are entered in adjacency order: the first successor of a node
/// and everything it reaches comes before the second.
///
/// # Panics
///
/// Panics when `start` is not a node in `graph`.
#[must_use]
pub fn depth_first_preorder<G: GraphView>(
    graph: &G,
    start: G::NodeId,
    direction: Direction,
) -> Vec<G::NodeId> {
    by_axis!(direction, preorder(graph, start))
}

/// The preorder walk, kept out of the depth-first core on purpose.
///
/// A preorder needs neither finishes nor depths, so this loop keeps no frames:
/// a node is visited when it is popped and pushes its unvisited successors in
/// reverse, which yields the order a recursive search would. Routed through
/// the core's frame stack it measured 8 to 10% slower on the 4,096-node chain
/// of the `cfg_depth_first_preorder` bench, whose search runs the length of
/// the graph; every other depth-first walk runs on the core.
fn preorder<G: GraphView, A: Adjacency>(axis: A, graph: &G, start: G::NodeId) -> Vec<G::NodeId> {
    assert_in_bound(graph, start, "start node");
    let mut visited = vec![false; graph.node_bound()];
    let mut order = Vec::with_capacity(graph.node_bound());
    let mut stack = vec![start];
    let mut adjacent = Vec::new();
    while let Some(node) = stack.pop() {
        if visited[node.index()] {
            continue;
        }
        visited[node.index()] = true;
        order.push(node);
        adjacent.clear();
        adjacent.extend(axis.neighbors(graph, node));
        for &next in adjacent.iter().rev() {
            if !visited[next.index()] {
                stack.push(next);
            }
        }
    }
    order
}

/// Return nodes in depth-first postorder from `start`.
///
/// A node comes after everything it reaches that was not reached before it,
/// so in an acyclic graph every node comes after its successors.
///
/// # Panics
///
/// Panics when `start` is not a node in `graph`.
#[must_use]
pub fn depth_first_postorder<G: GraphView>(
    graph: &G,
    start: G::NodeId,
    direction: Direction,
) -> Vec<G::NodeId> {
    assert_in_bound(graph, start, "start node");
    postorder_in(graph, [start], direction, &mut Vec::new())
}

/// Return reverse postorder from `start`.
///
/// # Panics
///
/// Panics when `start` is not a node in `graph`.
#[must_use]
pub fn reverse_postorder<G: GraphView>(
    graph: &G,
    start: G::NodeId,
    direction: Direction,
) -> Vec<G::NodeId> {
    let mut order = depth_first_postorder(graph, start, direction);
    order.reverse();
    order
}

/// The postorder of the depth-first forest grown from each of `seeds` in
/// turn, over caller-owned marks.
///
/// A seed reached from an earlier one adds nothing, so the forest visits each
/// node once: this is the postorder from a virtual root whose successors are
/// the seeds, in order, without the root. `visited` is cleared and sized to
/// the node bound, and holds the reached nodes afterwards.
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`.
pub(crate) fn postorder_in<G: GraphView>(
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    direction: Direction,
    visited: &mut Vec<bool>,
) -> Vec<G::NodeId> {
    visited.clear();
    visited.resize(graph.node_bound(), false);
    let seeds = seeds.into_iter().inspect(|&seed| {
        assert_in_bound(graph, seed, "seed node");
    });
    by_axis!(direction, postorder(graph, seeds, visited))
}

fn postorder<G: GraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    visited: &mut [bool],
) -> Vec<G::NodeId> {
    let mut order = Postorder(Vec::with_capacity(graph.node_bound()));
    depth_first_core(
        axis,
        graph,
        seeds,
        visited,
        &mut DepthFirstStack::default(),
        &mut order,
    );
    order.0
}

/// Return nodes in breadth-first order from `start`.
///
/// # Panics
///
/// Panics when `start` is not a node in `graph`.
#[must_use]
pub fn breadth_first<G: GraphView>(
    graph: &G,
    start: G::NodeId,
    direction: Direction,
) -> Vec<G::NodeId> {
    by_axis!(direction, breadth_first_order(graph, start))
}

fn breadth_first_order<G: GraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    start: G::NodeId,
) -> Vec<G::NodeId> {
    assert_in_bound(graph, start, "start node");
    let mut visited = vec![false; graph.node_bound()];
    let mut queue = Vec::with_capacity(graph.node_bound());
    visited[start.index()] = true;
    queue.push(start);
    breadth_first_core::<_, _, ()>(
        axis,
        graph,
        &mut queue,
        |_, _| ControlFlow::Continue(Visit::Descend),
        |_, _, to: G::NodeId, _| !core::mem::replace(&mut visited[to.index()], true),
    );
    queue
}

/// Mark every node reachable from `seeds` by walking `direction` edges.
///
/// Returns a dense `Vec<bool>` indexed by node id, `true` for every seed and
/// every node reachable from one. Duplicate seeds are fine. The result is a
/// set, so it is deterministic and independent of seed order — unlike the
/// order-yielding traversals above, which answer a single-start question.
///
/// Multi-source reachability is the shape most whole-program queries take:
/// live code from a set of roots, the callees of an entry-point set, the
/// nodes a set of definitions can flow to, and (with [`Direction::Backward`])
/// everything that can reach a set of sinks. Confine the walk to a region by
/// passing an [`Induced`](crate::Induced) view.
///
/// # Examples
///
/// ```
/// use cfglib::{Direction, Graph, reachable};
///
/// let mut graph = Graph::<&str, ()>::new();
/// let main = graph.add_node("main");
/// let helper = graph.add_node("helper");
/// let orphan = graph.add_node("orphan");
/// graph.add_edge(main, helper, ());
///
/// let live = reachable(&graph, [main], Direction::Forward);
/// assert_eq!(live, vec![true, true, false]);
///
/// // Seeding the orphan too covers the whole graph.
/// let live = reachable(&graph, [main, orphan], Direction::Forward);
/// assert_eq!(live, vec![true, true, true]);
/// ```
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`.
#[must_use]
pub fn reachable<G: GraphView>(
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    direction: Direction,
) -> Vec<bool> {
    let mut visited = Vec::new();
    reachable_in(graph, seeds, direction, None, &mut visited, &mut Vec::new());
    visited
}

/// [`reachable`] over caller-owned buffers, optionally confined to the nodes
/// whose `within` entry is `true`.
///
/// The seeds are marked whether or not they are within. `visited` is cleared
/// and sized to the node bound, then holds the answer. `stack` is cleared
/// first and left empty. The stack holds dense indices, not node ids, so one
/// pair of buffers serves every node-id type, which is what a scratch that
/// outlives one graph needs.
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`, or when `within` is shorter
/// than an index the walk reaches.
///
/// The confinement is a mask, not an [`Induced`](super::view::Induced) view:
/// through the view's filtered adjacency, the post-dominator tree's backward
/// walk made `cfg_post_dominators` 9% slower. Both functions of the walk are
/// `#[inline]`: left out of line, the shared instance ran the post-dominator
/// tree's forward walk 2.2 times slower than the loop it replaced, and
/// compiled into each caller it matches it.
#[inline]
pub(crate) fn reachable_in<G: GraphView>(
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    direction: Direction,
    within: Option<&[bool]>,
    visited: &mut Vec<bool>,
    stack: &mut Vec<usize>,
) {
    by_axis!(direction, reach(graph, seeds, within, visited, stack));
}

/// The reachability walk, kept out of the breadth-first core on purpose.
///
/// The answer is a set, so the order is free, and a LIFO stack keeps the
/// working set to the open frontier where the core's queue keeps every node it
/// reached: through the core, `reachable` measured 29% slower on the
/// call-graph pairs of the `api_reach_index_reachable_pairs_call_graph` bench
/// and 17% slower on the small graphs of `api_remove_unreachable_mapped`.
#[inline]
fn reach<G: GraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    within: Option<&[bool]>,
    visited: &mut Vec<bool>,
    stack: &mut Vec<usize>,
) {
    visited.clear();
    visited.resize(graph.node_bound(), false);
    stack.clear();
    for seed in seeds {
        assert_in_bound(graph, seed, "seed node");
        if !core::mem::replace(&mut visited[seed.index()], true) {
            stack.push(seed.index());
        }
    }
    while let Some(index) = stack.pop() {
        for next in axis.neighbors(graph, G::NodeId::from_index(index)) {
            let next = next.index();
            if within.is_none_or(|within| within[next])
                && !core::mem::replace(&mut visited[next], true)
            {
                stack.push(next);
            }
        }
    }
}

/// Return a topological ordering of the live nodes, or `None` when they
/// contain a cycle.
///
/// [`topological_sort_or_cycle`] returns the same order, and in place of
/// `None` one cycle as the witness.
#[must_use]
pub fn topological_sort<G: GraphView>(graph: &G) -> Option<Vec<G::NodeId>> {
    // Kahn's loop is written out here a second time on purpose: routed through
    // the copy `topological_sort_or_cycle` keeps (which also returns the
    // counts its witness needs), this function measured about 4% slower on a
    // 1,024-node chain. A change to one loop belongs in both.
    let mut incoming_counts = vec![0_usize; graph.node_bound()];
    let mut live = 0;
    for node in graph.node_ids() {
        incoming_counts[node.index()] = graph.predecessors(node).count();
        live += 1;
    }

    let mut queue: VecDeque<G::NodeId> = graph
        .node_ids()
        .filter(|node| incoming_counts[node.index()] == 0)
        .collect();
    let mut order = Vec::with_capacity(live);

    while let Some(node) = queue.pop_front() {
        order.push(node);
        for successor in graph.successors(node) {
            let count = &mut incoming_counts[successor.index()];
            *count -= 1;
            if *count == 0 {
                queue.push_back(successor);
            }
        }
    }

    (order.len() == live).then_some(order)
}

mod ancestors;
mod breadth_first;
mod cycle_witness;
mod depth_first;
mod shortest_path;

pub use ancestors::{CommonAncestor, common_ancestors, nearest_common_ancestor};
pub(crate) use breadth_first::breadth_first_core;
pub use cycle_witness::topological_sort_or_cycle;
pub(crate) use depth_first::{
    DepthFirstStack, DepthFirstVisitor, Mark, PathMarks, StepKind, depth_first_core,
};
pub(crate) use shortest_path::edge_path_in;
pub use shortest_path::{ShortestPathScratch, shortest_path, shortest_path_in};

#[cfg(test)]
mod tests;
