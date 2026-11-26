//! Generic directed-graph traversals and ordering algorithms.
//!
//! Every walk here, and every walk of [`search`](super::search),
//! [`edge_traverse`](super::edge_traverse) and the graph store's orderings,
//! is a thin wrapper over one of two crate-private cores: a breadth-first core
//! that expands a queue level by level, and a depth-first core that keeps one
//! frame per open node over one arena of steps. A wrapper only says what it
//! records (an order, a distance, a parent, an event) and which marks it keeps
//! (visited flags, epoch marks, marks per path, the three colors of an edge
//! classification). Both cores are generic over what a step is — a neighbor
//! of a node walk, an edge of an edge walk — and over the adjacency axis, so
//! one loop serves both directions and both kinds of walk.

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

/// What one step of a walk core is, and where it leads.
///
/// A node walk steps to a neighbor, so a step is the neighbor's dense index
/// and leads to itself. An edge walk steps along an edge
/// ([`Edges`](super::edge_traverse::Edges)), so a step is the edge's dense
/// index and leads to the edge's far end along the axis. The cores record
/// steps as indices, so their buffers carry no identity type and one buffer
/// serves every graph.
pub(crate) trait Steps<G: GraphView>: Copy {
    /// The dense indices of the steps out of `node`, in adjacency order.
    fn steps(self, graph: &G, node: G::NodeId) -> impl Iterator<Item = usize> + '_;

    /// The dense index of the node `step` leads to.
    fn target(self, graph: &G, step: usize) -> usize;
}

impl<G: GraphView, A: Adjacency> Steps<G> for A {
    #[inline]
    fn steps(self, graph: &G, node: G::NodeId) -> impl Iterator<Item = usize> + '_ {
        self.neighbors(graph, node).map(DenseId::index)
    }

    #[inline]
    fn target(self, _graph: &G, step: usize) -> usize {
        step
    }
}

/// The two orders a depth-first walk records, as one visitor: preorder when
/// `POST` is `false`, postorder when it is `true`.
struct Order<N, const POST: bool>(Vec<N>);

impl<N: DenseId, const POST: bool> DepthFirstVisitor<()> for Order<N, POST> {
    #[inline]
    fn discover(&mut self, node: usize, _depth: usize) -> ControlFlow<(), Visit> {
        if !POST {
            self.0.push(N::from_index(node));
        }
        ControlFlow::Continue(Visit::Descend)
    }

    #[inline]
    fn finish(&mut self, node: usize) -> ControlFlow<()> {
        if POST {
            self.0.push(N::from_index(node));
        }
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
    assert_in_bound(graph, start, "start node");
    let mut visited = vec![false; graph.node_bound()];
    by_axis!(direction, preorder(graph, [start.index()], &mut visited))
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
    let seeds = seeds.into_iter().map(|seed| {
        assert_in_bound(graph, seed, "seed node");
        seed.index()
    });
    by_axis!(direction, postorder(graph, seeds, visited))
}

fn preorder<G: GraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    seeds: impl IntoIterator<Item = usize>,
    visited: &mut [bool],
) -> Vec<G::NodeId> {
    depth_first_order::<G, A, false>(axis, graph, seeds, visited)
}

fn postorder<G: GraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    seeds: impl IntoIterator<Item = usize>,
    visited: &mut [bool],
) -> Vec<G::NodeId> {
    depth_first_order::<G, A, true>(axis, graph, seeds, visited)
}

fn depth_first_order<G: GraphView, A: Adjacency, const POST: bool>(
    axis: A,
    graph: &G,
    seeds: impl IntoIterator<Item = usize>,
    visited: &mut [bool],
) -> Vec<G::NodeId> {
    let mut order = Order::<G::NodeId, POST>(Vec::with_capacity(graph.node_bound()));
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
    queue.push(start.index());
    breadth_first_core::<_, _, ()>(
        axis,
        graph,
        &mut queue,
        |_, _| ControlFlow::Continue(Visit::Descend),
        |_, _, to, _| !core::mem::replace(&mut visited[to], true),
    );
    queue.into_iter().map(G::NodeId::from_index).collect()
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
    reachable_in(graph, seeds, direction, &mut visited, &mut Vec::new());
    visited
}

/// [`reachable`] over caller-owned buffers.
///
/// `visited` is cleared and sized to the node bound, then holds the answer.
/// `queue` is cleared first and holds the reached nodes afterwards. Both hold
/// dense indices, not node ids, so one pair of buffers serves every node-id
/// type, which is what a scratch that outlives one graph needs.
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`.
///
/// Both functions of the walk are `#[inline]`: left out of line, the shared
/// instance ran the post-dominator tree's forward walk 2.2 times slower than
/// the loop it replaced, and compiled into each caller it matches it.
#[inline]
pub(crate) fn reachable_in<G: GraphView>(
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    direction: Direction,
    visited: &mut Vec<bool>,
    queue: &mut Vec<usize>,
) {
    visited.clear();
    visited.resize(graph.node_bound(), false);
    queue.clear();
    for seed in seeds {
        assert_in_bound(graph, seed, "seed node");
        if !core::mem::replace(&mut visited[seed.index()], true) {
            queue.push(seed.index());
        }
    }
    by_axis!(direction, reach(graph, visited, queue));
}

#[inline]
fn reach<G: GraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    visited: &mut [bool],
    queue: &mut Vec<usize>,
) {
    breadth_first_core::<_, _, ()>(
        axis,
        graph,
        queue,
        |_, _| ControlFlow::Continue(Visit::Descend),
        |_, _, to, _| !core::mem::replace(&mut visited[to], true),
    );
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
