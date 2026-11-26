//! Shared ancestors: the meeting points of two breadth-first walks.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use super::{Adjacency, breadth_first_core};
use crate::graph::search::{Visit, may_expand};
use crate::graph::view::{DenseId, Direction, GraphView, assert_in_bound};

/// Return the nearest node reachable from both `a` and `b` by walking
/// `direction` edges — the meeting point of two outward searches.
///
/// With [`Direction::Backward`] this is the classic *nearest common
/// ancestor*: the closest node from which both `a` and `b` can be reached
/// (a shared dominator-like join in a CFG, a shared caller in a call graph, a
/// shared origin in a value-flow graph). With [`Direction::Forward`] it is
/// the mirror question — the closest node both can reach, such as a shared
/// sink or merge point. The graph need not be a tree or a DAG; cycles
/// terminate the searches like every other traversal here.
///
/// Both endpoints are at distance 0 from themselves, so each is itself a
/// candidate: when `b` is reachable from `a`, `b` competes at a combined
/// distance of the hops from `a` to `b`, and another shared node that is
/// closer overall beats it. `a == b` answers `a`. Returns `None` when no node
/// is reachable from both.
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
/// # Cost
///
/// The two breadth-first searches advance one level at a time, the shallower
/// one first, and stop as soon as both have reached the best combined
/// distance found: no node they have not seen could beat or tie it. A meet
/// at combined distance `d` therefore costs the nodes within `d` hops of
/// either endpoint, not everything both can reach. Each call still allocates
/// two distance tables of `node_bound` entries.
///
/// # Examples
///
/// ```
/// use cfglib::{Direction, Graph, nearest_common_ancestor};
///
/// //   root         `left` and `right` share one predecessor
/// //   /  \
/// // left right
/// let mut graph = Graph::<&str, ()>::new();
/// let root = graph.add_node("root");
/// let left = graph.add_node("left");
/// let right = graph.add_node("right");
/// graph.add_edge(root, left, ());
/// graph.add_edge(root, right, ());
///
/// // Walking predecessors from both leaves meets at the root.
/// assert_eq!(
///     nearest_common_ancestor(&graph, left, right, Direction::Backward),
///     Some(root)
/// );
///
/// // Forward, `left` is reachable from `root` and from itself, so the meet
/// // is `left` at a combined distance of 1.
/// assert_eq!(
///     nearest_common_ancestor(&graph, root, left, Direction::Forward),
///     Some(left)
/// );
///
/// // Two leaves have no common successor.
/// assert_eq!(
///     nearest_common_ancestor(&graph, left, right, Direction::Forward),
///     None
/// );
/// ```
///
/// # Panics
///
/// Panics when either endpoint is not a node in `graph`.
#[must_use]
pub fn nearest_common_ancestor<G: GraphView>(
    graph: &G,
    a: G::NodeId,
    b: G::NodeId,
    direction: Direction,
) -> Option<G::NodeId> {
    by_axis!(direction, nearest_meet(graph, a, b))
}

fn nearest_meet<G: GraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    a: G::NodeId,
    b: G::NodeId,
) -> Option<G::NodeId> {
    assert_in_bound(graph, a, "node `a`");
    assert_in_bound(graph, b, "node `b`");
    let mut sides = [MeetSide::new(graph, a), MeetSide::new(graph, b)];
    // `min` over `(combined distance, node id)` *is* the documented
    // tie-break: node ids are dense and ordered, so the tuple ordering ranks
    // by distance and settles ties on the smaller id.
    let mut best = (a == b).then_some((0, a));

    // Each step finishes one more breadth-first level of the side whose
    // radius is smaller. A node joins the candidates when the second search
    // reaches it, with both of its distances exact. Every node whose combined
    // distance is at most `d` lies within `d` of both endpoints, so once both
    // radii reach the best combined distance found, every node that could
    // beat it or tie it has been seen, and the answer is final.
    loop {
        if best.is_some_and(|(distance, _)| sides[0].radius.min(sides[1].radius) >= distance) {
            break;
        }
        let side = match (sides[0].done(), sides[1].done()) {
            (true, true) => break,
            (false, true) => 0,
            (true, false) => 1,
            (false, false) => usize::from(sides[1].radius < sides[0].radius),
        };
        let [first, second] = &mut sides;
        let (this, other) = if side == 0 {
            (first, &*second)
        } else {
            (second, &*first)
        };
        this.expand(axis, graph, |node, distance| {
            let from_other = other.distances[node.index()];
            if from_other != usize::MAX {
                let candidate = (distance + from_other, node);
                best = Some(best.map_or(candidate, |known| known.min(candidate)));
            }
        });
    }
    best.map(|(_, node)| node)
}

/// One endpoint's breadth-first search in [`nearest_meet`], advanced a whole
/// level at a time.
struct MeetSide<N> {
    /// Hop count from the endpoint per dense index, `usize::MAX` for a node
    /// not reached yet.
    distances: Vec<usize>,
    /// The nodes at distance `radius`.
    frontier: Vec<N>,
    /// Every node within this many hops is in `distances`; `usize::MAX` once
    /// the search has reached everything it can.
    radius: usize,
}

impl<N: DenseId> MeetSide<N> {
    fn new<G: GraphView<NodeId = N>>(graph: &G, start: N) -> Self {
        let mut distances = vec![usize::MAX; graph.node_bound()];
        distances[start.index()] = 0;
        Self {
            distances,
            frontier: vec![start],
            radius: 0,
        }
    }

    fn done(&self) -> bool {
        self.radius == usize::MAX
    }

    /// Discovers the next level, reporting each new node with its distance.
    ///
    /// The frontier is the walk's depth-0 level and every node it discovers
    /// is skipped rather than expanded, so the core expands exactly one level
    /// and leaves the next one behind the frontier in its queue.
    fn expand<G: GraphView<NodeId = N>, A: Adjacency>(
        &mut self,
        axis: A,
        graph: &G,
        mut discovered: impl FnMut(N, usize),
    ) {
        let distance = self.radius + 1;
        let level = self.frontier.len();
        let distances = &mut self.distances;
        breadth_first_core::<_, _, ()>(
            axis,
            graph,
            &mut self.frontier,
            |_, depth| {
                ControlFlow::Continue(if depth == 0 {
                    Visit::Descend
                } else {
                    Visit::Skip
                })
            },
            |_, _, to: N, _| {
                if distances[to.index()] != usize::MAX {
                    return false;
                }
                distances[to.index()] = distance;
                discovered(to, distance);
                true
            },
        );
        self.frontier.drain(..level);
        self.radius = if self.frontier.is_empty() {
            usize::MAX
        } else {
            distance
        };
    }
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
/// use cfglib::{Direction, Graph, common_ancestors};
///
/// // An inheritance graph with edges base -> derived, so predecessors are
/// // base classes:  object -> mixin, mixin -> a, mixin -> b.
/// let mut graph = Graph::<&str, ()>::new();
/// let object = graph.add_node("object");
/// let mixin = graph.add_node("mixin");
/// let a = graph.add_node("A");
/// let b = graph.add_node("B");
/// graph.add_edge(object, mixin, ());
/// graph.add_edge(mixin, a, ());
/// graph.add_edge(mixin, b, ());
///
/// // Both shared bases, in the order a walk from `b` discovers them.
/// let shared = common_ancestors(&graph, a, b, Direction::Backward, None);
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
/// let near = common_ancestors(&graph, a, b, Direction::Backward, Some(1));
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
pub fn common_ancestors<G: GraphView>(
    graph: &G,
    a: G::NodeId,
    b: G::NodeId,
    direction: Direction,
    max_depth: Option<usize>,
) -> Vec<CommonAncestor<G::NodeId>> {
    by_axis!(direction, all_meets(graph, a, b, max_depth))
}

fn all_meets<G: GraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    a: G::NodeId,
    b: G::NodeId,
    max_depth: Option<usize>,
) -> Vec<CommonAncestor<G::NodeId>> {
    assert_in_bound(graph, a, "node `a`");
    assert_in_bound(graph, b, "node `b`");
    let (from_a, _) = distances(axis, graph, a, max_depth);
    let (from_b, order_b) = distances(axis, graph, b, max_depth);

    // Walking `b`'s discovery order — rather than the node ids — is what
    // makes the result's order the documented one; the bound is already
    // applied by both walks, so a node present in either table is in range.
    order_b
        .into_iter()
        .filter_map(|node| {
            let from_a = from_a[node.index()];
            (from_a != usize::MAX).then(|| CommonAncestor {
                node,
                from_a,
                from_b: from_b[node.index()],
            })
        })
        .collect()
}

/// Hop counts from `start`, `usize::MAX` for a node the walk did not reach,
/// and the nodes it reached in breadth-first discovery order.
///
/// `max_depth` bounds the walk: a node farther than that many hops is neither
/// reached nor measured. `start` itself comes first, at distance 0.
fn distances<G: GraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    start: G::NodeId,
    max_depth: Option<usize>,
) -> (Vec<usize>, Vec<G::NodeId>) {
    let mut distances = vec![usize::MAX; graph.node_bound()];
    distances[start.index()] = 0;
    let mut queue = vec![start];
    breadth_first_core::<_, _, ()>(
        axis,
        graph,
        &mut queue,
        |_, depth| {
            ControlFlow::Continue(if may_expand(max_depth, depth) {
                Visit::Descend
            } else {
                Visit::Skip
            })
        },
        |_, _, to: G::NodeId, depth| {
            if distances[to.index()] != usize::MAX {
                return false;
            }
            distances[to.index()] = depth + 1;
            true
        },
    );
    (distances, queue)
}
