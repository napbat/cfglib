//! Generic directed-graph traversals and ordering algorithms.

extern crate alloc;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::view::{DenseNodeId, DirectedGraphView};

/// Direction in which a graph traversal follows edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalDirection {
    /// Follow edges from source to target.
    Outgoing,
    /// Follow edges from target to source.
    Incoming,
}

/// One adjacency axis of a [`DirectedGraphView`], as a **type** rather than a
/// value.
///
/// [`TraversalDirection`] is a public parameter of every walk here, and
/// reading it *inside* a walk costs more than the branch it looks like:
/// `successors` and `predecessors` are distinct opaque iterator types, so a
/// core that decides per expanded node cannot hold either of them. It has to
/// materialise the adjacency into a buffer first and then test and push out of
/// that copy. Owning one buffer per walk removed the copy's *allocation*; the
/// copy itself is what was left.
///
/// A zero-sized axis moves the decision to monomorphisation: a core generic
/// over `A: Adjacency` names one concrete iterator type, so it can iterate the
/// graph's own adjacency **in place**. The public functions keep their
/// `TraversalDirection` argument and turn it into a type exactly once, at
/// entry, with [`by_axis!`].
///
/// # Which walks the copy actually leaves
///
/// A walk that consumes adjacency **in adjacency order** — every
/// breadth-first frontier, the reachability stack, the bounded meets, the
/// breadth-first [`search`](super::search::search) core — reads the axis
/// directly and keeps no buffer at all, which measured 1.2x to 1.9x on the
/// pinned fixtures.
///
/// A **depth-first** walk does not: its rev-push convention needs the
/// successors in reverse, an axis yields a plain `Iterator`, and requiring a
/// `DoubleEndedIterator` of every consumer-owned view would be a contract
/// change for a walk-local convenience. The alternative that needs no reversed
/// read — push forward, then reverse the frontier's tail — was measured on the
/// same fixtures and is not one: it is 1.2x to 1.5x *faster* where nodes have
/// at most one successor and 1.9x *slower* at out-degree two, and a substrate
/// core cannot pick per graph shape. So the depth-first cores keep one
/// adjacency buffer per walk and take from the axis only the branch, which is
/// perfectly predicted and measures as parity.
///
/// The price of the axis is one monomorphisation of every core per direction,
/// which is why it stays an implementation detail rather than becoming a
/// second public spelling of a direction consumers already pass.
pub(crate) trait Adjacency: Copy {
    /// Iterate `node`'s neighbors along this axis.
    fn neighbors<G: DirectedGraphView>(
        self,
        graph: &G,
        node: G::NodeId,
    ) -> impl Iterator<Item = G::NodeId> + '_;
}

/// The forward axis: [`DirectedGraphView::successors`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Outgoing;

/// The reverse axis: [`DirectedGraphView::predecessors`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Incoming;

impl Adjacency for Outgoing {
    fn neighbors<G: DirectedGraphView>(
        self,
        graph: &G,
        node: G::NodeId,
    ) -> impl Iterator<Item = G::NodeId> + '_ {
        graph.successors(node)
    }
}

impl Adjacency for Incoming {
    fn neighbors<G: DirectedGraphView>(
        self,
        graph: &G,
        node: G::NodeId,
    ) -> impl Iterator<Item = G::NodeId> + '_ {
        graph.predecessors(node)
    }
}

/// Call an axis-generic core, resolving a [`TraversalDirection`] **once**.
///
/// The core takes its axis as a leading argument, so the value becomes a type
/// at one place per public walk — its entry — and the two arms cannot drift
/// apart across the dozen walks that need them. Written as a macro rather than
/// a helper because the arms differ only in a *type*, which no value-taking
/// helper can carry.
macro_rules! by_axis {
    ($direction:expr, $core:ident($($argument:expr),* $(,)?)) => {
        match $direction {
            $crate::graph::traverse::TraversalDirection::Outgoing => {
                $core($crate::graph::traverse::Outgoing, $($argument),*)
            }
            $crate::graph::traverse::TraversalDirection::Incoming => {
                $core($crate::graph::traverse::Incoming, $($argument),*)
            }
        }
    };
}

pub(crate) use by_axis;

/// Return nodes in depth-first preorder from `start`.
///
/// # Panics
///
/// Panics when `start` is not a node in `graph`.
#[must_use]
pub fn depth_first_preorder<G: DirectedGraphView>(
    graph: &G,
    start: G::NodeId,
    direction: TraversalDirection,
) -> Vec<G::NodeId> {
    by_axis!(direction, preorder_from(graph, start))
}

fn preorder_from<G: DirectedGraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    start: G::NodeId,
) -> Vec<G::NodeId> {
    assert!(
        start.index() < graph.node_count(),
        "start node is out of range"
    );
    let mut visited = vec![false; graph.node_count()];
    let mut order = Vec::with_capacity(graph.node_count());
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
        for &successor in adjacent.iter().rev() {
            if !visited[successor.index()] {
                stack.push(successor);
            }
        }
    }

    order
}

/// Return nodes in depth-first postorder from `start`.
///
/// # Panics
///
/// Panics when `start` is not a node in `graph`.
#[must_use]
pub fn depth_first_postorder<G: DirectedGraphView>(
    graph: &G,
    start: G::NodeId,
    direction: TraversalDirection,
) -> Vec<G::NodeId> {
    by_axis!(direction, postorder_from(graph, start))
}

fn postorder_from<G: DirectedGraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    start: G::NodeId,
) -> Vec<G::NodeId> {
    assert!(
        start.index() < graph.node_count(),
        "start node is out of range"
    );
    let mut visited = vec![false; graph.node_count()];
    let mut order = Vec::with_capacity(graph.node_count());
    let mut stack = vec![(start, false)];
    let mut adjacent = Vec::new();

    while let Some((node, processed)) = stack.pop() {
        if processed {
            order.push(node);
            continue;
        }
        if visited[node.index()] {
            continue;
        }

        visited[node.index()] = true;
        stack.push((node, true));
        adjacent.clear();
        adjacent.extend(axis.neighbors(graph, node));
        for &successor in adjacent.iter().rev() {
            if !visited[successor.index()] {
                stack.push((successor, false));
            }
        }
    }

    order
}

/// Return reverse postorder from `start`.
#[must_use]
pub fn reverse_postorder<G: DirectedGraphView>(
    graph: &G,
    start: G::NodeId,
    direction: TraversalDirection,
) -> Vec<G::NodeId> {
    let mut order = depth_first_postorder(graph, start, direction);
    order.reverse();
    order
}

/// Return nodes in breadth-first order from `start`.
///
/// # Panics
///
/// Panics when `start` is not a node in `graph`.
#[must_use]
pub fn breadth_first<G: DirectedGraphView>(
    graph: &G,
    start: G::NodeId,
    direction: TraversalDirection,
) -> Vec<G::NodeId> {
    by_axis!(direction, breadth_first_from(graph, start))
}

fn breadth_first_from<G: DirectedGraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    start: G::NodeId,
) -> Vec<G::NodeId> {
    assert!(
        start.index() < graph.node_count(),
        "start node is out of range"
    );
    let mut visited = vec![false; graph.node_count()];
    let mut order = Vec::with_capacity(graph.node_count());
    let mut queue = VecDeque::new();
    visited[start.index()] = true;
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        order.push(node);
        for next in axis.neighbors(graph, node) {
            if !visited[next.index()] {
                visited[next.index()] = true;
                queue.push_back(next);
            }
        }
    }

    order
}

/// Return a shortest unweighted path from `start` to `goal`.
///
/// The result includes both endpoints. Parallel edges do not affect the path.
///
/// # Panics
///
/// Panics when either endpoint is not a node in `graph`.
#[must_use]
pub fn shortest_path<G: DirectedGraphView>(
    graph: &G,
    start: G::NodeId,
    goal: G::NodeId,
    direction: TraversalDirection,
) -> Option<Vec<G::NodeId>> {
    by_axis!(direction, shortest_path_from(graph, start, goal))
}

fn shortest_path_from<G: DirectedGraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    start: G::NodeId,
    goal: G::NodeId,
) -> Option<Vec<G::NodeId>> {
    assert!(
        start.index() < graph.node_count(),
        "start node is out of range"
    );
    assert!(
        goal.index() < graph.node_count(),
        "goal node is out of range"
    );
    // `visited` guards every read, so an arbitrary valid node is a cheaper
    // uninitialised marker than `Option<NodeId>` (which is wider for dense
    // integer IDs).
    let mut previous = vec![start; graph.node_count()];
    let mut visited = vec![false; graph.node_count()];
    let mut queue = VecDeque::new();
    visited[start.index()] = true;
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        if node == goal {
            let mut path = vec![goal];
            let mut current = goal;
            while current != start {
                current = previous[current.index()];
                path.push(current);
            }
            path.reverse();
            return Some(path);
        }

        for next in axis.neighbors(graph, node) {
            if !visited[next.index()] {
                visited[next.index()] = true;
                previous[next.index()] = node;
                queue.push_back(next);
            }
        }
    }

    None
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
/// nodes a set of definitions can flow to, and (with
/// [`TraversalDirection::Incoming`]) everything that can reach a set of
/// sinks.
///
/// # Examples
///
/// ```
/// use cfglib::{DirectedGraph, TraversalDirection, reachable};
///
/// let mut graph = DirectedGraph::<&str, ()>::new();
/// let main = graph.add_node("main");
/// let helper = graph.add_node("helper");
/// let orphan = graph.add_node("orphan");
/// graph.add_edge(main, helper, ());
///
/// let live = reachable(&graph, [main], TraversalDirection::Outgoing);
/// assert_eq!(live, vec![true, true, false]);
///
/// // Seeding the orphan too covers the whole graph.
/// let live = reachable(&graph, [main, orphan], TraversalDirection::Outgoing);
/// assert_eq!(live, vec![true, true, true]);
/// ```
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`.
#[must_use]
pub fn reachable<G: DirectedGraphView>(
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    direction: TraversalDirection,
) -> Vec<bool> {
    by_axis!(direction, reachable_from(graph, seeds))
}

fn reachable_from<G: DirectedGraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
) -> Vec<bool> {
    let mut visited = vec![false; graph.node_count()];
    let mut stack = Vec::new();

    for seed in seeds {
        assert!(
            seed.index() < graph.node_count(),
            "seed node is out of range"
        );
        if !visited[seed.index()] {
            visited[seed.index()] = true;
            stack.push(seed);
        }
    }

    while let Some(node) = stack.pop() {
        for next in axis.neighbors(graph, node) {
            if !visited[next.index()] {
                visited[next.index()] = true;
                stack.push(next);
            }
        }
    }

    visited
}

/// Hop counts from `start`, using `usize::MAX` for unreachable nodes, while
/// reporting each node in breadth-first discovery order to `discovered`.
///
/// `max_depth` bounds the walk: nodes farther than that many hops are neither
/// discovered nor measured. `None` walks the whole reachable set. `start`
/// itself is reported first, at distance 0.
fn breadth_first_bounded<G: DirectedGraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    start: G::NodeId,
    max_depth: Option<usize>,
) -> Vec<usize> {
    breadth_first_bounded_with(axis, graph, start, max_depth, |_| {})
}

fn breadth_first_bounded_with<G: DirectedGraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    start: G::NodeId,
    max_depth: Option<usize>,
    mut discovered: impl FnMut(G::NodeId),
) -> Vec<usize> {
    let mut distances = vec![usize::MAX; graph.node_count()];
    let mut queue = VecDeque::new();
    distances[start.index()] = 0;
    discovered(start);
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        let depth = distances[node.index()];
        if max_depth.is_some_and(|limit| depth >= limit) {
            continue;
        }
        for next in axis.neighbors(graph, node) {
            if distances[next.index()] == usize::MAX {
                distances[next.index()] = depth + 1;
                discovered(next);
                queue.push_back(next);
            }
        }
    }

    distances
}

/// Hop counts from `start` to every node reachable along `axis`, using
/// `usize::MAX` for unreachable nodes. `start` itself is at distance 0.
fn breadth_first_distances<G: DirectedGraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    start: G::NodeId,
) -> Vec<usize> {
    breadth_first_bounded(axis, graph, start, None)
}

/// Return the nearest node reachable from both `a` and `b` by walking
/// `direction` edges — the meeting point of two outward searches.
///
/// With [`TraversalDirection::Incoming`] this is the classic *nearest common
/// ancestor*: the closest node from which both `a` and `b` can be reached
/// (a shared dominator-like join in a CFG, a shared caller in a call graph, a
/// shared origin in a value-flow graph). With
/// [`TraversalDirection::Outgoing`] it is the mirror question — the closest
/// node both can reach, such as a shared sink or merge point. The graph need
/// not be a tree or a DAG; cycles terminate the searches like every other
/// traversal here.
///
/// Both endpoints are at distance 0 from themselves, so `b` is the answer
/// whenever it is reachable from `a` at all, and `a == b` answers `a`.
/// Returns `None` when no node is reachable from both.
///
/// # Determinism
///
/// Candidates are ranked by **smallest combined distance first; ties broken
/// by smallest node id.** The combined distance of a candidate is its hop
/// count from `a` plus its hop count from `b`. The answer therefore never
/// depends on adjacency order or on which endpoint was passed first.
///
/// That rank is a `min` over `(combined distance, node id)` and nothing else.
/// A consumer whose language fixes another one — a linearization order, a
/// declaration order, a tie-break on the distance from a single endpoint —
/// asks [`common_ancestors`] for every shared node with both distances and
/// applies its own rank; this function is that generalization ranked the one
/// way a tuple can express.
///
/// # Examples
///
/// ```
/// use cfglib::{DirectedGraph, TraversalDirection, nearest_common_ancestor};
///
/// //   root         `left` and `right` share one predecessor
/// //   /  \
/// // left right
/// let mut graph = DirectedGraph::<&str, ()>::new();
/// let root = graph.add_node("root");
/// let left = graph.add_node("left");
/// let right = graph.add_node("right");
/// graph.add_edge(root, left, ());
/// graph.add_edge(root, right, ());
///
/// // Walking predecessors from both leaves meets at the root.
/// assert_eq!(
///     nearest_common_ancestor(&graph, left, right, TraversalDirection::Incoming),
///     Some(root)
/// );
///
/// // Forward, `left` is reachable from `root` and from itself, so the meet
/// // is `left` at a combined distance of 1.
/// assert_eq!(
///     nearest_common_ancestor(&graph, root, left, TraversalDirection::Outgoing),
///     Some(left)
/// );
///
/// // Two leaves have no common successor.
/// assert_eq!(
///     nearest_common_ancestor(&graph, left, right, TraversalDirection::Outgoing),
///     None
/// );
/// ```
///
/// # Panics
///
/// Panics when either endpoint is not a node in `graph`.
#[must_use]
pub fn nearest_common_ancestor<G: DirectedGraphView>(
    graph: &G,
    a: G::NodeId,
    b: G::NodeId,
    direction: TraversalDirection,
) -> Option<G::NodeId> {
    by_axis!(direction, nearest_meet(graph, a, b))
}

fn nearest_meet<G: DirectedGraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    a: G::NodeId,
    b: G::NodeId,
) -> Option<G::NodeId> {
    assert!(a.index() < graph.node_count(), "node `a` is out of range");
    assert!(b.index() < graph.node_count(), "node `b` is out of range");
    let from_a = breadth_first_distances(axis, graph, a);
    let from_b = breadth_first_distances(axis, graph, b);

    // `min` over `(combined distance, node id)` *is* the documented
    // tie-break: node ids are dense and ordered, so the tuple ordering
    // ranks by distance and settles ties on the smaller id.
    graph
        .node_ids()
        .filter_map(|node| {
            let reached_from_a = from_a[node.index()];
            let reached_from_b = from_b[node.index()];
            (reached_from_a != usize::MAX && reached_from_b != usize::MAX)
                .then(|| (reached_from_a + reached_from_b, node))
        })
        .min()
        .map(|(_, node)| node)
}

/// A node reachable from both endpoints of a [`common_ancestors`] query,
/// carrying the hop count from each so the consumer can rank it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CommonAncestor<N> {
    /// The shared node.
    pub node: N,
    /// Hop count from the first endpoint (`a`).
    pub from_a: usize,
    /// Hop count from the second endpoint (`b`).
    pub from_b: usize,
}

impl<N> CommonAncestor<N> {
    /// The combined distance — the hop count from `a` plus the hop count
    /// from `b`, the primary rank [`nearest_common_ancestor`] uses.
    #[inline]
    #[must_use]
    pub fn combined(&self) -> usize {
        self.from_a + self.from_b
    }
}

/// Return every node reachable from both `a` and `b` by walking `direction`
/// edges, each with its hop count from either endpoint.
///
/// This is the consumer-rankable generalization of
/// [`nearest_common_ancestor`], which collapses the same candidate set with
/// one fixed rank — smallest combined distance, ties by smallest node id.
/// Language semantics routinely fix a different one (a C3 linearization
/// prefers the base nearest the *second* operand, an overload resolution
/// prefers the declaration seen first), and a rank that is not a function of
/// `(distance, node id)` cannot be expressed by returning a single node.
///
/// `max_depth` bounds **each side independently**: with `Some(limit)` a node
/// qualifies only when it is within `limit` hops of `a` *and* within `limit`
/// hops of `b`. `None` searches the whole reachable set. Both endpoints are
/// at distance 0 from themselves, so `a == b` yields everything reachable
/// from `a` within the bound, each entry with equal distances.
///
/// # Order
///
/// The result is in **`b`'s breadth-first discovery order**: the order a
/// breadth-first walk from `b` along `direction` edges first reaches each
/// node, with ties inside a level following adjacency order. That guarantee
/// is the point of the function — a consumer scanning the list for its own
/// first best match reproduces a scan-order tie-break (`min_by_key` and
/// `max_by_key` both keep the first of equal elements) instead of having to
/// re-derive one from node ids. Distances are exact regardless of the order.
///
/// # Examples
///
/// ```
/// use cfglib::{DirectedGraph, TraversalDirection, common_ancestors};
///
/// // An inheritance graph with edges base -> derived, so predecessors are
/// // base classes:  object -> mixin, mixin -> a, mixin -> b.
/// let mut graph = DirectedGraph::<&str, ()>::new();
/// let object = graph.add_node("object");
/// let mixin = graph.add_node("mixin");
/// let a = graph.add_node("A");
/// let b = graph.add_node("B");
/// graph.add_edge(object, mixin, ());
/// graph.add_edge(mixin, a, ());
/// graph.add_edge(mixin, b, ());
///
/// // Both shared bases, in the order a walk from `b` discovers them.
/// let shared = common_ancestors(&graph, a, b, TraversalDirection::Incoming, None);
/// assert_eq!(
///     shared.iter().map(|found| found.node).collect::<Vec<_>>(),
///     vec![mixin, object]
/// );
/// assert_eq!((shared[0].from_a, shared[0].from_b), (1, 1));
///
/// // A consumer applies its own rank over that order — here "nearest to
/// // both, ties to the one nearest `b`, ties to whichever came first".
/// let chosen = shared
///     .iter()
///     .min_by_key(|found| (found.combined(), found.from_b))
///     .map(|found| found.node);
/// assert_eq!(chosen, Some(mixin));
///
/// // Bounding the search to one hop from each endpoint drops `object`.
/// let near = common_ancestors(&graph, a, b, TraversalDirection::Incoming, Some(1));
/// assert_eq!(
///     near.iter().map(|found| found.node).collect::<Vec<_>>(),
///     vec![mixin]
/// );
/// ```
///
/// # Panics
///
/// Panics when either endpoint is not a node in `graph`.
#[must_use]
pub fn common_ancestors<G: DirectedGraphView>(
    graph: &G,
    a: G::NodeId,
    b: G::NodeId,
    direction: TraversalDirection,
    max_depth: Option<usize>,
) -> Vec<CommonAncestor<G::NodeId>> {
    by_axis!(direction, all_meets(graph, a, b, max_depth))
}

fn all_meets<G: DirectedGraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    a: G::NodeId,
    b: G::NodeId,
    max_depth: Option<usize>,
) -> Vec<CommonAncestor<G::NodeId>> {
    assert!(a.index() < graph.node_count(), "node `a` is out of range");
    assert!(b.index() < graph.node_count(), "node `b` is out of range");
    let from_a = breadth_first_bounded(axis, graph, a, max_depth);
    let mut order_b = Vec::new();
    let from_b = breadth_first_bounded_with(axis, graph, b, max_depth, |node| {
        order_b.push(node);
    });

    // Walking `b`'s discovery order — rather than the node ids — is what
    // makes the result's order the documented one; the bound is already
    // applied by both walks, so a node present in either table is in range.
    order_b
        .into_iter()
        .filter_map(|node| {
            let from_a = from_a[node.index()];
            let from_b = from_b[node.index()];
            (from_a != usize::MAX && from_b != usize::MAX).then_some(CommonAncestor {
                node,
                from_a,
                from_b,
            })
        })
        .collect()
}

/// Return a topological ordering, or `None` when the graph contains a cycle.
#[must_use]
pub fn topological_sort<G: DirectedGraphView>(graph: &G) -> Option<Vec<G::NodeId>> {
    let mut incoming_counts = vec![0_usize; graph.node_count()];
    for node in graph.node_ids() {
        incoming_counts[node.index()] = graph.predecessors(node).count();
    }

    let mut queue: VecDeque<G::NodeId> = graph
        .node_ids()
        .filter(|node| incoming_counts[node.index()] == 0)
        .collect();
    let mut order = Vec::with_capacity(graph.node_count());

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

    (order.len() == graph.node_count()).then_some(order)
}

impl<I> Cfg<I> {
    /// Depth-first preorder traversal starting from the entry block.
    #[must_use]
    pub fn dfs_preorder(&self) -> Vec<BlockId> {
        depth_first_preorder(self, self.entry(), TraversalDirection::Outgoing)
    }

    /// Depth-first postorder traversal starting from the entry block.
    #[must_use]
    pub fn dfs_postorder(&self) -> Vec<BlockId> {
        depth_first_postorder(self, self.entry(), TraversalDirection::Outgoing)
    }

    /// Reverse postorder starting from the entry block.
    #[must_use]
    pub fn reverse_postorder(&self) -> Vec<BlockId> {
        reverse_postorder(self, self.entry(), TraversalDirection::Outgoing)
    }

    /// Breadth-first traversal starting from the entry block.
    #[must_use]
    pub fn bfs(&self) -> Vec<BlockId> {
        breadth_first(self, self.entry(), TraversalDirection::Outgoing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeKind;
    use crate::graph::directed::{DirectedGraph, NodeId};
    use crate::test_util::ff;
    use alloc::vec;

    #[test]
    fn cfg_traversal_methods_delegate_to_generic_algorithms() {
        let mut cfg = Cfg::new();
        let middle = cfg.new_block();
        let last = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("entry"));
        cfg.block_mut(middle).push(ff("middle"));
        cfg.block_mut(last).push(ff("last"));
        cfg.add_edge(cfg.entry(), middle, EdgeKind::Fallthrough);
        cfg.add_edge(middle, last, EdgeKind::Fallthrough);

        assert_eq!(cfg.dfs_preorder(), vec![cfg.entry(), middle, last]);
        assert_eq!(cfg.dfs_postorder(), vec![last, middle, cfg.entry()]);
        assert_eq!(cfg.reverse_postorder(), vec![cfg.entry(), middle, last]);
        assert_eq!(cfg.bfs(), vec![cfg.entry(), middle, last]);
    }

    #[test]
    fn directed_graph_can_be_walked_in_both_directions() {
        let mut graph = DirectedGraph::<&str, ()>::new();
        let first = graph.add_node("first");
        let second = graph.add_node("second");
        let third = graph.add_node("third");
        graph.add_edge(first, second, ());
        graph.add_edge(second, third, ());

        assert_eq!(
            breadth_first(&graph, first, TraversalDirection::Outgoing),
            vec![first, second, third]
        );
        assert_eq!(
            breadth_first(&graph, third, TraversalDirection::Incoming),
            vec![third, second, first]
        );
        assert_eq!(
            shortest_path(&graph, first, third, TraversalDirection::Outgoing),
            Some(vec![first, second, third])
        );
    }

    /// `a -> b -> c` with a `c -> b` back edge, plus a disconnected `d`.
    fn reach_fixture() -> (DirectedGraph<(), ()>, [NodeId; 4]) {
        let mut graph = DirectedGraph::<(), ()>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, b, ());
        (graph, [a, b, c, d])
    }

    #[test]
    fn reachable_from_no_seeds_marks_nothing() {
        let (graph, _) = reach_fixture();
        assert_eq!(
            reachable(&graph, [], TraversalDirection::Outgoing),
            vec![false; 4]
        );

        // An empty graph yields an empty table rather than panicking.
        let empty = DirectedGraph::<(), ()>::new();
        assert_eq!(reachable(&empty, [], TraversalDirection::Outgoing).len(), 0);
    }

    #[test]
    fn reachable_unions_multiple_sources_and_terminates_on_cycles() {
        let (graph, [a, _, c, d]) = reach_fixture();
        // The b <-> c cycle terminates; d stays unreached.
        assert_eq!(
            reachable(&graph, [a], TraversalDirection::Outgoing),
            vec![true, true, true, false]
        );
        // A second seed unions in, and duplicate seeds change nothing.
        assert_eq!(
            reachable(&graph, [a, d, a, d], TraversalDirection::Outgoing),
            vec![true; 4]
        );
        // Order-insensitive: the answer is a set.
        assert_eq!(
            reachable(&graph, [d, c], TraversalDirection::Outgoing),
            reachable(&graph, [c, d], TraversalDirection::Outgoing)
        );
        assert_eq!(
            reachable(&graph, [c], TraversalDirection::Outgoing),
            vec![false, true, true, false]
        );
    }

    #[test]
    fn reachable_walks_predecessors_in_the_incoming_direction() {
        let (graph, [a, _, c, _]) = reach_fixture();
        assert_eq!(
            reachable(&graph, [c], TraversalDirection::Incoming),
            vec![true, true, true, false]
        );
        assert_eq!(
            reachable(&graph, [a], TraversalDirection::Incoming),
            vec![true, false, false, false]
        );
    }

    #[test]
    fn reachable_handles_self_loops() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let only = graph.add_node(());
        let other = graph.add_node(());
        graph.add_edge(only, only, ());
        assert_eq!(
            reachable(&graph, [only], TraversalDirection::Outgoing),
            vec![true, false]
        );
        // A self-loop is not a reason to be reachable from elsewhere.
        assert_eq!(
            reachable(&graph, [other], TraversalDirection::Outgoing),
            vec![false, true]
        );
    }

    /// `root -> mid`, `mid -> left`, `mid -> right`, both legs into `bottom`.
    /// `root` has the smallest id but is the *farther* common ancestor.
    fn diamond() -> (DirectedGraph<(), ()>, [NodeId; 5]) {
        let mut graph = DirectedGraph::<(), ()>::new();
        let root = graph.add_node(());
        let mid = graph.add_node(());
        let left = graph.add_node(());
        let right = graph.add_node(());
        let bottom = graph.add_node(());
        graph.add_edge(root, mid, ());
        graph.add_edge(mid, left, ());
        graph.add_edge(mid, right, ());
        graph.add_edge(left, bottom, ());
        graph.add_edge(right, bottom, ());
        (graph, [root, mid, left, right, bottom])
    }

    #[test]
    fn nearest_common_ancestor_meets_at_the_closest_shared_node() {
        let (graph, [_root, mid, left, right, bottom]) = diamond();
        // `mid` (combined 2) beats `root` (combined 4) even though `root`
        // has the smaller id: distance ranks first.
        assert_eq!(
            nearest_common_ancestor(&graph, left, right, TraversalDirection::Incoming),
            Some(mid)
        );
        // Forward, the same two legs merge at the bottom.
        assert_eq!(
            nearest_common_ancestor(&graph, left, right, TraversalDirection::Outgoing),
            Some(bottom)
        );
        // The answer does not depend on which endpoint is passed first.
        assert_eq!(
            nearest_common_ancestor(&graph, right, left, TraversalDirection::Incoming),
            Some(mid)
        );
        assert_eq!(
            nearest_common_ancestor(&graph, bottom, mid, TraversalDirection::Incoming),
            Some(mid)
        );
    }

    #[test]
    fn nearest_common_ancestor_treats_endpoints_as_distance_zero() {
        let (graph, [root, _, left, _, bottom]) = diamond();
        // A node is its own meet, in either direction.
        assert_eq!(
            nearest_common_ancestor(&graph, root, root, TraversalDirection::Outgoing),
            Some(root)
        );
        assert_eq!(
            nearest_common_ancestor(&graph, bottom, bottom, TraversalDirection::Incoming),
            Some(bottom)
        );
        // `left` is reachable from `root`, so the meet is `left` itself.
        assert_eq!(
            nearest_common_ancestor(&graph, root, left, TraversalDirection::Outgoing),
            Some(left)
        );
        assert_eq!(
            nearest_common_ancestor(&graph, left, root, TraversalDirection::Outgoing),
            Some(left)
        );
    }

    #[test]
    fn nearest_common_ancestor_breaks_ties_by_smallest_node_id() {
        // Adjacency deliberately offers the higher-id sink first, so a
        // discovery-order answer would pick `second_sink`.
        let (graph, [first_sink, second_sink, left, right]) = twin_sinks();

        // Both sinks sit at combined distance 2; the smaller id wins.
        assert_eq!(
            nearest_common_ancestor(&graph, left, right, TraversalDirection::Outgoing),
            Some(first_sink)
        );
        assert_eq!(
            nearest_common_ancestor(&graph, right, left, TraversalDirection::Outgoing),
            Some(first_sink)
        );
        assert_eq!(
            nearest_common_ancestor(
                &graph,
                first_sink,
                second_sink,
                TraversalDirection::Incoming
            ),
            Some(left)
        );
    }

    #[test]
    fn nearest_common_ancestor_without_a_shared_node_is_none() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let start = graph.add_node(());
        let lonely = graph.add_node(());
        let end = graph.add_node(());
        graph.add_edge(start, end, ());

        assert_eq!(
            nearest_common_ancestor(&graph, start, lonely, TraversalDirection::Outgoing),
            None
        );
        assert_eq!(
            nearest_common_ancestor(&graph, start, lonely, TraversalDirection::Incoming),
            None
        );
        // Two nodes with no shared successor, both inside the connected part.
        assert_eq!(
            nearest_common_ancestor(&graph, end, lonely, TraversalDirection::Incoming),
            None
        );
    }

    #[test]
    fn nearest_common_ancestor_terminates_on_cycles() {
        let (mut graph, [_, mid, left, right, bottom]) = diamond();
        graph.add_edge(bottom, mid, ());

        // Every node is now reachable from both legs; `bottom` (1 + 1) is
        // the closest forward meet and `mid` (1 + 1) the closest backward one.
        assert_eq!(
            nearest_common_ancestor(&graph, left, right, TraversalDirection::Outgoing),
            Some(bottom)
        );
        assert_eq!(
            nearest_common_ancestor(&graph, left, right, TraversalDirection::Incoming),
            Some(mid)
        );
    }

    /// Two shared sinks at the same distance, offered to the traversal in
    /// descending id order — discovery order and id order disagree.
    fn twin_sinks() -> (DirectedGraph<(), ()>, [NodeId; 4]) {
        let mut graph = DirectedGraph::<(), ()>::new();
        let first_sink = graph.add_node(());
        let second_sink = graph.add_node(());
        let left = graph.add_node(());
        let right = graph.add_node(());
        graph.add_edge(left, second_sink, ());
        graph.add_edge(left, first_sink, ());
        graph.add_edge(right, second_sink, ());
        graph.add_edge(right, first_sink, ());
        (graph, [first_sink, second_sink, left, right])
    }

    fn ancestor_nodes(found: &[CommonAncestor<NodeId>]) -> Vec<NodeId> {
        found.iter().map(|entry| entry.node).collect()
    }

    #[test]
    fn common_ancestors_returns_every_shared_node_with_both_distances() {
        let (graph, [root, mid, left, right, bottom]) = diamond();
        let found = common_ancestors(&graph, left, right, TraversalDirection::Incoming, None);
        assert_eq!(ancestor_nodes(&found), vec![mid, root]);
        assert_eq!((found[0].from_a, found[0].from_b), (1, 1));
        assert_eq!((found[1].from_a, found[1].from_b), (2, 2));
        assert_eq!(found[1].combined(), 4);

        // Forward, the same two legs share only the merge point.
        let merged = common_ancestors(&graph, left, right, TraversalDirection::Outgoing, None);
        assert_eq!(ancestor_nodes(&merged), vec![bottom]);
    }

    #[test]
    fn common_ancestors_are_in_b_discovery_order_not_id_order() {
        let (graph, [first_sink, second_sink, left, right]) = twin_sinks();
        // A walk from `right` reaches `second_sink` first because adjacency
        // offers it first, even though its id is larger.
        let found = common_ancestors(&graph, left, right, TraversalDirection::Outgoing, None);
        assert_eq!(ancestor_nodes(&found), vec![second_sink, first_sink]);
        // Both sit at the same combined distance, so a consumer's first-match
        // scan takes the discovery-order winner while the fixed rank of
        // `nearest_common_ancestor` takes the smaller id.
        let scanned = found.iter().min_by_key(|entry| entry.combined());
        assert_eq!(scanned.map(|entry| entry.node), Some(second_sink));
        assert_eq!(
            nearest_common_ancestor(&graph, left, right, TraversalDirection::Outgoing),
            Some(first_sink)
        );

        // Swapping the endpoints swaps the order, since it is `b`'s.
        let swapped = common_ancestors(&graph, right, left, TraversalDirection::Outgoing, None);
        assert_eq!(ancestor_nodes(&swapped), vec![second_sink, first_sink]);
    }

    #[test]
    fn common_ancestors_bound_each_side_by_max_depth() {
        let (graph, [_root, mid, left, right, _bottom]) = diamond();
        // `root` is two hops from either leg, `mid` one.
        assert_eq!(
            ancestor_nodes(&common_ancestors(
                &graph,
                left,
                right,
                TraversalDirection::Incoming,
                Some(1)
            )),
            vec![mid]
        );
        assert_eq!(
            common_ancestors(&graph, left, right, TraversalDirection::Incoming, Some(0)).len(),
            0
        );
        // A bound at or beyond the eccentricity is the unbounded answer.
        assert_eq!(
            common_ancestors(&graph, left, right, TraversalDirection::Incoming, Some(2)),
            common_ancestors(&graph, left, right, TraversalDirection::Incoming, None)
        );
    }

    #[test]
    fn common_ancestors_of_a_node_with_itself_is_its_reachable_set() {
        let (graph, [_root, _mid, left, _right, bottom]) = diamond();
        let found = common_ancestors(&graph, left, left, TraversalDirection::Outgoing, None);
        assert_eq!(ancestor_nodes(&found), vec![left, bottom]);
        assert!(
            found
                .iter()
                .all(|entry| entry.from_a == entry.from_b && entry.combined() % 2 == 0)
        );
        // Bounded to zero hops, only the node itself.
        assert_eq!(
            ancestor_nodes(&common_ancestors(
                &graph,
                left,
                left,
                TraversalDirection::Outgoing,
                Some(0)
            )),
            vec![left]
        );
    }

    #[test]
    fn common_ancestors_without_a_shared_node_is_empty() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let start = graph.add_node(());
        let lonely = graph.add_node(());
        let end = graph.add_node(());
        graph.add_edge(start, end, ());
        assert_eq!(
            common_ancestors(&graph, start, lonely, TraversalDirection::Outgoing, None).len(),
            0
        );
        assert_eq!(
            common_ancestors(&graph, end, lonely, TraversalDirection::Incoming, None).len(),
            0
        );
    }

    #[test]
    fn common_ancestors_terminates_on_cycles() {
        let (mut graph, [root, mid, left, right, bottom]) = diamond();
        graph.add_edge(bottom, mid, ());

        // Every node now reaches both legs; the walk still terminates and
        // reports each node exactly once, in `right`'s discovery order.
        let found = common_ancestors(&graph, left, right, TraversalDirection::Incoming, None);
        assert_eq!(ancestor_nodes(&found), vec![right, mid, root, bottom, left]);
        assert_eq!(
            found
                .iter()
                .min_by_key(|entry| entry.combined())
                .map(|entry| entry.node),
            Some(mid)
        );
    }

    #[test]
    fn common_ancestors_generalizes_nearest_common_ancestor() {
        let (plain, _) = diamond();
        let (twins, _) = twin_sinks();
        let (mut cyclic, [_, mid, _, _, bottom]) = diamond();
        cyclic.add_edge(bottom, mid, ());
        for graph in [&plain, &twins, &cyclic] {
            for a in graph.node_ids() {
                for b in graph.node_ids() {
                    for direction in [TraversalDirection::Incoming, TraversalDirection::Outgoing] {
                        // The fixed rank is a `min` over `(combined, id)` of
                        // exactly this candidate set.
                        let ranked = common_ancestors(graph, a, b, direction, None)
                            .into_iter()
                            .min_by_key(|entry| (entry.combined(), entry.node))
                            .map(|entry| entry.node);
                        assert_eq!(ranked, nearest_common_ancestor(graph, a, b, direction));
                    }
                }
            }
        }
    }

    #[test]
    fn topological_sort_rejects_cycles() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let left = graph.add_node(());
        let right = graph.add_node(());
        graph.add_edge(left, right, ());
        assert_eq!(topological_sort(&graph), Some(vec![left, right]));
        graph.add_edge(right, left, ());
        assert!(topological_sort(&graph).is_none());
    }
}
