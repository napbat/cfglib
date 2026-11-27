//! Shortest unweighted paths, searched from both ends at once.
//!
//! A point-to-point question ("is there a call path from A to B, and which")
//! asked of a search from one end explores the whole ball around the start
//! until it reaches the goal, and a pass that asks many of them over one graph
//! also pays for tables sized to the whole node space per query. The search
//! here runs from both ends and keeps its tables in a [`ShortestPathScratch`]
//! the caller can reuse. It alternates one edge examination forward from the
//! start with one backward from the goal, and stops once the best path found
//! is no longer than the sum of the two searches' current depths, the stopping
//! rule of Haeupler, Hladík, Rozhoň, Tarjan and Tětek (*Bidirectional
//! Dijkstra's Algorithm is Instance-Optimal*, SOSA 2025) applied to unweighted
//! edges. Stopping at the first node both searches have seen would be wrong:
//! the first meeting need not lie on a shortest path.
//!
//! The search advances one edge at a time on alternating sides, which is a
//! different stepping from the level-by-level breadth-first core, so it keeps
//! its own loop. [`shortest_path_edges`](crate::shortest_path_edges) is the
//! edge-identity form over the same search.

extern crate alloc;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

use super::Adjacency;
use crate::graph::edge_traverse::EdgeAdjacency;
use crate::graph::edge_view::EdgeView;
use crate::graph::epoch::EpochMarks;
use crate::graph::view::{DenseId, Direction, GraphView, assert_in_bound};
use crate::identity::{check_bound, slot, word};

/// Caller-owned buffers for repeated point-to-point queries, read by
/// [`shortest_path_in`] and
/// [`shortest_path_edges_in`](crate::shortest_path_edges_in).
///
/// It holds two searches' worth of tables: per node, an [`EpochMarks`] stamp,
/// a hop count and a link back toward the search's endpoint, each a `u32`, and
/// a queue per search. A query resets the marks by an epoch bump, O(1), and
/// reads a hop count or a link only for a node marked in the current query, so
/// the tables are never cleared. Nothing in a query costs O(node count) once
/// the scratch covers the graph.
///
/// The scratch starts empty and a query grows its tables on entry to the
/// graph's node bound, so one scratch serves a sequence of graphs and holds
/// the largest. It carries no node-id type: one instance serves every view.
///
/// # Examples
///
/// ```
/// use cfglib::{Direction, Graph, ShortestPathScratch, shortest_path_in};
///
/// let mut graph = Graph::<(), ()>::new();
/// let [a, b, c] = [(); 3].map(|()| graph.add_node(()));
/// graph.add_edge(a, b, ());
/// graph.add_edge(b, c, ());
///
/// let mut scratch = ShortestPathScratch::default();
/// for (start, goal) in [(a, c), (b, c), (c, a)] {
///     let path = shortest_path_in(&mut scratch, &graph, start, goal, Direction::Forward);
///     assert_eq!(path.is_some(), start != c);
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct ShortestPathScratch {
    /// The search from the start, then the search from the goal.
    sides: [SideBuffers; 2],
}

/// The tables of one of a query's two searches.
#[derive(Debug, Clone, Default)]
struct SideBuffers {
    /// The nodes this search has reached in the current query.
    reached: EpochMarks,
    /// A reached node's hop count from this search's endpoint.
    depth: Vec<u32>,
    /// A reached node's link back toward this search's endpoint: the node it
    /// was reached from in a node query, the edge it was reached through in an
    /// edge query.
    link: Vec<u32>,
    /// Reached nodes whose neighbors this search has yet to examine, in the
    /// order it reached them.
    queue: VecDeque<usize>,
}

impl SideBuffers {
    /// Start a search from `endpoint` over a node space of `node_bound`.
    fn begin(&mut self, node_bound: usize, endpoint: usize) {
        self.reached.grow_to(node_bound);
        if self.depth.len() < node_bound {
            self.depth.resize(node_bound, 0);
            self.link.resize(node_bound, 0);
        }
        self.reached.reset();
        self.queue.clear();
        self.reached.mark(endpoint);
        self.depth[endpoint] = 0;
    }

    /// The hop count of `node` from this search's endpoint, if reached.
    fn depth_of(&self, node: usize) -> Option<u32> {
        self.reached.is_marked(node).then(|| self.depth[node])
    }
}

/// The node a search is examining the neighbors of, with the ones it has not
/// examined yet.
struct Front<I> {
    node: usize,
    depth: u32,
    neighbors: I,
}

/// The best join of the two searches found so far: an edge from `ahead`,
/// reached by the forward search, to `behind`, reached by the backward search,
/// and the link that names it. The length is a `usize`: two depths below a
/// `u32` bound can sum past it.
#[derive(Debug, Clone, Copy)]
struct Meet {
    length: usize,
    ahead: usize,
    behind: usize,
    link: u32,
}

/// Run the two searches of one query to the shortest join of `start` and
/// `goal`, or `None` when no path joins them.
///
/// `forward` yields a node's neighbors along the query's direction and
/// `backward` against it, each paired with the link that reaches it.
fn meet<FI, BI>(
    sides: &mut [SideBuffers; 2],
    node_bound: usize,
    (start, goal): (usize, usize),
    forward: impl Fn(usize) -> FI,
    backward: impl Fn(usize) -> BI,
) -> Option<Meet>
where
    FI: Iterator<Item = (usize, u32)>,
    BI: Iterator<Item = (usize, u32)>,
{
    let [ahead, behind] = sides;
    ahead.begin(node_bound, start);
    behind.begin(node_bound, goal);
    let mut ahead_front = Front {
        node: start,
        depth: 0,
        neighbors: forward(start),
    };
    let mut behind_front = Front {
        node: goal,
        depth: 0,
        neighbors: backward(goal),
    };
    let mut best: Option<Meet> = None;
    let mut forward_turn = true;
    // Every node shallower than a search's front has had all its edges
    // examined. A shortest path of length `L` shorter than the best join has
    // an edge whose near end lies above the forward front and whose far end
    // lies above the backward front whenever `L` is below the sum of the two
    // front depths, and both searches examining that edge would have joined at
    // it. So once the best join is no longer than that sum it is a shortest
    // one. A search that runs out has examined every edge it can reach, which
    // includes the edge into the other endpoint on every path.
    loop {
        if best
            .is_some_and(|meet| meet.length <= slot(ahead_front.depth) + slot(behind_front.depth))
        {
            break;
        }
        let examined = if forward_turn {
            examine(ahead, &mut ahead_front, behind, &forward, &mut best, true)
        } else {
            examine(
                behind,
                &mut behind_front,
                ahead,
                &backward,
                &mut best,
                false,
            )
        };
        if !examined {
            break;
        }
        forward_turn = !forward_turn;
    }
    best
}

/// Examine one more edge of one search, moving its front on to the next
/// queued node as needed; `false` when the search has nothing left.
///
/// An edge to a node the other search reached is a join; only a strictly
/// shorter join replaces the best one, so ties go to the first found.
fn examine<I: Iterator<Item = (usize, u32)>>(
    this: &mut SideBuffers,
    front: &mut Front<I>,
    other: &SideBuffers,
    neighbors: impl Fn(usize) -> I,
    best: &mut Option<Meet>,
    forward: bool,
) -> bool {
    loop {
        if let Some((next, link)) = front.neighbors.next() {
            if let Some(remaining) = other.depth_of(next) {
                let length = slot(front.depth) + 1 + slot(remaining);
                if best.is_none_or(|meet| length < meet.length) {
                    let (ahead, behind) = if forward {
                        (front.node, next)
                    } else {
                        (next, front.node)
                    };
                    *best = Some(Meet {
                        length,
                        ahead,
                        behind,
                        link,
                    });
                }
            }
            if !this.reached.is_marked(next) {
                this.reached.mark(next);
                this.depth[next] = front.depth + 1;
                this.link[next] = link;
                this.queue.push_back(next);
            }
            return true;
        }
        let Some(node) = this.queue.pop_front() else {
            return false;
        };
        *front = Front {
            node,
            depth: this.depth[node],
            neighbors: neighbors(node),
        };
    }
}

/// Return a shortest unweighted path from `start` to `goal` along `direction`
/// edges.
///
/// The path includes both endpoints, `start == goal` is the one-node path,
/// and `None` means `goal` is unreachable. Parallel edges do not affect the
/// path. Each call allocates its own [`ShortestPathScratch`]; a pass that
/// asks many questions over one graph reuses one through
/// [`shortest_path_in`].
///
/// # Search
///
/// One search runs from `start` along `direction` and one from `goal` against
/// it, alternating one edge examination each, forward first. An edge from a
/// node one search reached to a node the other reached joins them into a
/// candidate path. The query stops once the best candidate is no longer than
/// the sum of the two searches' current depths, or once either search runs
/// out.
///
/// # Which path
///
/// Among the shortest paths, the witness is the first join of that length the
/// alternation finds, with neighbors in adjacency order. It depends on nothing
/// else, so the same graph gives the same path.
///
/// # Cost
///
/// O(edges examined), plus the path's length to build it, with no term in
/// the node count once the scratch covers the graph. Each side examines about
/// as many edges as the other, so a goal that one side reaches through a
/// short chain is found from that side cheaply even when the other side fans
/// out.
///
/// The backward search reads [`predecessors`](GraphView::predecessors) (or
/// `successors` for [`Direction::Backward`]). A view that stores only forward
/// adjacency serves that by [`scan_predecessors`](crate::scan_predecessors),
/// O(nodes × edges) per call; give such a view real reverse adjacency before
/// asking it for paths.
///
/// # Panics
///
/// Panics when either endpoint is not a node in `graph`, or when the node
/// bound exceeds `u32::MAX`.
///
/// # Examples
///
/// ```
/// use cfglib::{Direction, Graph, shortest_path};
///
/// // A call graph: `main` calls `parse` and `run`, and both call `log`.
/// let mut graph = Graph::<&str, ()>::new();
/// let main = graph.add_node("main");
/// let parse = graph.add_node("parse");
/// let run = graph.add_node("run");
/// let log = graph.add_node("log");
/// graph.add_edge(main, parse, ());
/// graph.add_edge(main, run, ());
/// graph.add_edge(parse, log, ());
/// graph.add_edge(run, log, ());
///
/// assert_eq!(
///     shortest_path(&graph, main, log, Direction::Forward),
///     Some(vec![main, parse, log]),
/// );
/// assert_eq!(shortest_path(&graph, log, main, Direction::Forward), None);
/// // Against the edges, from a callee up to a caller.
/// assert_eq!(
///     shortest_path(&graph, log, main, Direction::Backward).map(|path| path.len()),
///     Some(3),
/// );
/// ```
#[must_use]
pub fn shortest_path<G: GraphView>(
    graph: &G,
    start: G::NodeId,
    goal: G::NodeId,
    direction: Direction,
) -> Option<Vec<G::NodeId>> {
    shortest_path_in(
        &mut ShortestPathScratch::default(),
        graph,
        start,
        goal,
        direction,
    )
}

/// [`shortest_path`] over a caller-owned [`ShortestPathScratch`], for a pass
/// that asks many point-to-point questions.
///
/// The answer is the one [`shortest_path`] gives; only the tables live in
/// `scratch`, which grows to the graph on entry.
///
/// # Panics
///
/// Panics when either endpoint is not a node in `graph`, or when the node
/// bound exceeds `u32::MAX`.
///
/// # Examples
///
/// ```
/// use cfglib::{Direction, Graph, ShortestPathScratch, shortest_path_in};
///
/// let mut graph = Graph::<(), ()>::new();
/// let [a, b, c] = [(); 3].map(|()| graph.add_node(()));
/// graph.add_edge(a, b, ());
/// graph.add_edge(b, c, ());
///
/// // One scratch serves every query over the graph.
/// let mut scratch = ShortestPathScratch::default();
/// assert_eq!(
///     shortest_path_in(&mut scratch, &graph, a, c, Direction::Forward),
///     Some(vec![a, b, c]),
/// );
/// assert_eq!(
///     shortest_path_in(&mut scratch, &graph, c, a, Direction::Backward),
///     Some(vec![c, b, a]),
/// );
/// ```
#[must_use]
pub fn shortest_path_in<G: GraphView>(
    scratch: &mut ShortestPathScratch,
    graph: &G,
    start: G::NodeId,
    goal: G::NodeId,
    direction: Direction,
) -> Option<Vec<G::NodeId>> {
    by_axis!(direction, node_path(graph, start, goal, scratch))
}

fn node_path<G: GraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    start: G::NodeId,
    goal: G::NodeId,
    scratch: &mut ShortestPathScratch,
) -> Option<Vec<G::NodeId>> {
    assert_in_bound(graph, start, "start node");
    assert_in_bound(graph, goal, "goal node");
    if start == goal {
        return Some(alloc::vec![start]);
    }
    check_bound(graph.node_bound(), "node bound");
    let along = |node: usize| {
        let link = word(node);
        axis.neighbors(graph, G::NodeId::from_index(node))
            .map(move |next| (next.index(), link))
    };
    let against = |node: usize| {
        let link = word(node);
        axis.against()
            .neighbors(graph, G::NodeId::from_index(node))
            .map(move |next| (next.index(), link))
    };
    let (start, goal) = (start.index(), goal.index());
    let meet = meet(
        &mut scratch.sides,
        graph.node_bound(),
        (start, goal),
        along,
        against,
    )?;
    let [ahead, behind] = &scratch.sides;
    let mut path = Vec::with_capacity(meet.length + 1);
    let mut node = meet.ahead;
    path.push(node);
    while node != start {
        node = slot(ahead.link[node]);
        path.push(node);
    }
    path.reverse();
    let mut node = meet.behind;
    path.push(node);
    while node != goal {
        node = slot(behind.link[node]);
        path.push(node);
    }
    Some(path.into_iter().map(G::NodeId::from_index).collect())
}

/// The edges of a shortest path from `from` to `to` along `direction` edges,
/// over a caller-owned scratch: the search of [`shortest_path`] over
/// [`outgoing`](EdgeView::outgoing) and [`incoming`](EdgeView::incoming).
///
/// `from == to` is the empty path, and parallel edges stay distinct, so the
/// witness names the edge it took.
pub(crate) fn edge_path_in<G: EdgeView>(
    scratch: &mut ShortestPathScratch,
    graph: &G,
    from: G::NodeId,
    to: G::NodeId,
    direction: Direction,
) -> Option<Vec<G::EdgeId>> {
    by_axis!(direction, edge_path(graph, from, to, scratch))
}

fn edge_path<G: EdgeView, A: EdgeAdjacency>(
    axis: A,
    graph: &G,
    from: G::NodeId,
    to: G::NodeId,
    scratch: &mut ShortestPathScratch,
) -> Option<Vec<G::EdgeId>>
where
    A::Against: EdgeAdjacency,
{
    assert_in_bound(graph, from, "source node");
    assert_in_bound(graph, to, "target node");
    if from == to {
        return Some(Vec::new());
    }
    check_bound(graph.node_bound(), "node bound");
    check_bound(graph.edge_bound(), "edge bound");
    let along = |node: usize| {
        axis.edges(graph, G::NodeId::from_index(node))
            .map(move |edge| (axis.next(graph.edge(edge)).index(), word(edge.index())))
    };
    let against = |node: usize| {
        let axis = axis.against();
        axis.edges(graph, G::NodeId::from_index(node))
            .map(move |edge| (axis.next(graph.edge(edge)).index(), word(edge.index())))
    };
    let (from, to) = (from.index(), to.index());
    let meet = meet(
        &mut scratch.sides,
        graph.node_bound(),
        (from, to),
        along,
        against,
    )?;
    let [ahead, behind] = &scratch.sides;
    let edge = |link: u32| G::EdgeId::from_index(slot(link));
    let mut path = Vec::with_capacity(meet.length);
    let mut node = meet.ahead;
    while node != from {
        let link = ahead.link[node];
        path.push(edge(link));
        node = axis.previous(graph.edge(edge(link))).index();
    }
    path.reverse();
    path.push(edge(meet.link));
    let mut node = meet.behind;
    while node != to {
        let link = behind.link[node];
        path.push(edge(link));
        node = axis.next(graph.edge(edge(link))).index();
    }
    Some(path)
}

#[cfg(test)]
mod tests;
