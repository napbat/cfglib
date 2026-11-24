use super::*;
use crate::graph::edge_traverse::shortest_path_edges;
use crate::graph::store::{EdgeId, Graph, NodeId};
use crate::graph::traverse::shortest_path;
use crate::test_util::rng::Rng;
use alloc::vec;
use core::cell::Cell;

const DIRECTIONS: [TraversalDirection; 2] =
    [TraversalDirection::Outgoing, TraversalDirection::Incoming];

/// A random multigraph over `node_count` slots with `edge_count` edges:
/// cycles, self-loops and parallel edges occur, and about one slot in six is
/// removed again, so the bound exceeds the live nodes.
fn random_graph(rng: &mut Rng, node_count: usize, edge_count: usize) -> Graph<(), ()> {
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..node_count).map(|_| graph.add_node(())).collect();
    for _ in 0..edge_count {
        graph.add_edge(
            nodes[rng.below(node_count)],
            nodes[rng.below(node_count)],
            (),
        );
    }
    for &node in &nodes {
        if rng.one_in(6) {
            graph.remove_node(node);
        }
    }
    graph
}

/// Asserts that `path` walks from `start` to `goal` along `direction`, one
/// edge per step.
fn assert_node_path(
    graph: &Graph<(), ()>,
    path: &[NodeId],
    (start, goal): (NodeId, NodeId),
    direction: TraversalDirection,
) {
    assert_eq!(path.first(), Some(&start));
    assert_eq!(path.last(), Some(&goal));
    for pair in path.windows(2) {
        let joined = match direction {
            TraversalDirection::Outgoing => {
                GraphView::successors(graph, pair[0]).any(|n| n == pair[1])
            }
            TraversalDirection::Incoming => {
                GraphView::predecessors(graph, pair[0]).any(|n| n == pair[1])
            }
        };
        assert!(
            joined,
            "{path:?}: {:?} -> {:?} is no edge",
            pair[0], pair[1]
        );
    }
}

/// Asserts that the edges of `path` chain from `from` to `to` along
/// `direction`.
fn assert_edge_path(
    graph: &Graph<(), ()>,
    path: &[EdgeId],
    (from, to): (NodeId, NodeId),
    direction: TraversalDirection,
) {
    let mut at = from;
    for &edge in path {
        let edge = EdgeView::edge(graph, edge);
        let (near, far) = match direction {
            TraversalDirection::Outgoing => (edge.source(), edge.target()),
            TraversalDirection::Incoming => (edge.target(), edge.source()),
        };
        assert_eq!(near, at, "{path:?} breaks at {at:?}");
        at = far;
    }
    assert_eq!(at, to);
}

/// Every ordered pair of live nodes of random multigraphs, both directions,
/// both variants: the bidirectional answer is a valid path exactly when the
/// one-sided one is, and as short. One scratch serves the whole sequence and
/// grows with it.
#[test]
fn bidirectional_paths_are_as_short_as_the_one_sided_ones() {
    let mut scratch = ShortestPathScratch::default();
    let mut rng = Rng::new(41);
    let (mut joined, mut apart) = (0, 0);
    for round in 0..120 {
        let node_count = 1 + rng.below(4 + round / 4);
        let edge_count = rng.below(3 * node_count + 1);
        let graph = random_graph(&mut rng, node_count, edge_count);
        let live: Vec<NodeId> = graph.node_ids().collect();
        for direction in DIRECTIONS {
            for &start in &live {
                for &goal in &live {
                    let ends = (start, goal);
                    let expected = shortest_path(&graph, start, goal, direction);
                    let path =
                        shortest_path_bidirectional(&graph, start, goal, direction, &mut scratch);
                    assert_eq!(
                        path.as_ref().map(Vec::len),
                        expected.as_ref().map(Vec::len),
                        "{ends:?} {direction:?}"
                    );
                    match &path {
                        Some(path) => {
                            joined += 1;
                            assert_node_path(&graph, path, ends, direction);
                        }
                        None => apart += 1,
                    }

                    let expected = shortest_path_edges(&graph, start, goal, direction);
                    let edges = shortest_path_edges_bidirectional(
                        &graph,
                        start,
                        goal,
                        direction,
                        &mut scratch,
                    );
                    assert_eq!(
                        edges.as_ref().map(Vec::len),
                        expected.as_ref().map(Vec::len),
                        "{ends:?} {direction:?}"
                    );
                    if let Some(edges) = &edges {
                        assert_edge_path(&graph, edges, ends, direction);
                    }
                }
            }
        }
    }
    assert!(
        joined > 5_000 && apart > 5_000,
        "{joined} joined, {apart} apart"
    );
}

/// A scratch carried across graphs that grow and shrink gives the witness a
/// fresh scratch gives, so nothing of one query leaks into the next.
#[test]
fn a_reused_scratch_answers_like_a_fresh_one() {
    let mut reused = ShortestPathScratch::new(2);
    let mut rng = Rng::new(7);
    for node_count in [3, 10, 40, 5, 80, 1, 25] {
        let graph = random_graph(&mut rng, node_count, 2 * node_count);
        let live: Vec<NodeId> = graph.node_ids().collect();
        if live.is_empty() {
            continue;
        }
        for _ in 0..200 {
            let start = live[rng.below(live.len())];
            let goal = live[rng.below(live.len())];
            let direction = DIRECTIONS[rng.below(2)];
            assert_eq!(
                shortest_path_bidirectional(&graph, start, goal, direction, &mut reused),
                shortest_path_bidirectional(
                    &graph,
                    start,
                    goal,
                    direction,
                    &mut ShortestPathScratch::default()
                ),
            );
            assert_eq!(
                shortest_path_edges_bidirectional(&graph, start, goal, direction, &mut reused),
                shortest_path_edges_bidirectional(
                    &graph,
                    start,
                    goal,
                    direction,
                    &mut ShortestPathScratch::new(node_count)
                ),
            );
        }
    }
}

/// A view that counts the adjacency lists it is asked for.
struct Counting<'g> {
    graph: &'g Graph<(), ()>,
    calls: Cell<usize>,
}

impl GraphView for Counting<'_> {
    type NodeId = NodeId;

    fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.graph.node_ids()
    }

    fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.calls.set(self.calls.get() + 1);
        GraphView::successors(self.graph, node)
    }

    fn predecessors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.calls.set(self.calls.get() + 1);
        GraphView::predecessors(self.graph, node)
    }
}

/// The issue's examined-edge case: a complete binary out-tree of depth
/// `2h = 16`, from the root to the last leaf. The backward search climbs the
/// leaf's single parent chain, one adjacency list per level, and the
/// alternation holds the forward search to as many edges as that, so neither
/// side asks for more than `DEPTH + 1` lists (22 in all, as it runs). The
/// one-sided search asks for one per node it dequeues, which is every node
/// before the last leaf.
#[test]
fn the_backward_search_climbs_a_chain_instead_of_flooding_the_tree() {
    const DEPTH: usize = 16;
    let mut graph = Graph::<(), ()>::new();
    let nodes: Vec<NodeId> = (0..(1 << (DEPTH + 1)) - 1)
        .map(|_| graph.add_node(()))
        .collect();
    for (index, &node) in nodes.iter().enumerate().skip(1) {
        graph.add_edge(nodes[(index - 1) / 2], node, ());
    }
    let (root, leaf) = (nodes[0], nodes[nodes.len() - 1]);
    let counting = Counting {
        graph: &graph,
        calls: Cell::new(0),
    };

    let mut scratch = ShortestPathScratch::new(nodes.len());
    let outgoing = TraversalDirection::Outgoing;
    let path = shortest_path_bidirectional(&counting, root, leaf, outgoing, &mut scratch)
        .expect("the root reaches every leaf");
    assert_eq!(path.len(), DEPTH + 1);
    let bidirectional = counting.calls.replace(0);
    assert!(bidirectional <= 2 * (DEPTH + 1), "{bidirectional} calls");

    let one_sided = shortest_path(&counting, root, leaf, outgoing).expect("reachable");
    assert_eq!(one_sided.len(), DEPTH + 1);
    assert!(
        counting.calls.get() >= 1 << DEPTH,
        "{} calls",
        counting.calls.get()
    );
}

#[test]
fn trivial_direct_and_missing_paths() {
    let mut graph = Graph::<(), ()>::new();
    let [a, removed, b, c] = [(); 4].map(|()| graph.add_node(()));
    graph.remove_node(removed);
    graph.add_edge(a, a, ());
    let direct = graph.add_edge(a, b, ());
    let mut scratch = ShortestPathScratch::default();
    let outgoing = TraversalDirection::Outgoing;
    let incoming = TraversalDirection::Incoming;

    // A node is its own path, self-loop or not, as in the one-sided search.
    assert_eq!(
        shortest_path_bidirectional(&graph, a, a, outgoing, &mut scratch),
        Some(vec![a])
    );
    assert_eq!(
        shortest_path_edges_bidirectional(&graph, a, a, outgoing, &mut scratch),
        Some(Vec::new())
    );
    assert_eq!(
        shortest_path_bidirectional(&graph, a, b, outgoing, &mut scratch),
        Some(vec![a, b])
    );
    assert_eq!(
        shortest_path_edges_bidirectional(&graph, b, a, incoming, &mut scratch),
        Some(vec![direct])
    );
    assert_eq!(
        shortest_path_bidirectional(&graph, b, a, outgoing, &mut scratch),
        None
    );
    assert_eq!(
        shortest_path_edges_bidirectional(&graph, a, c, outgoing, &mut scratch),
        None
    );
}

#[test]
#[should_panic(expected = "goal node is out of range")]
fn a_goal_beyond_the_bound_panics() {
    let mut graph = Graph::<(), ()>::new();
    let a = graph.add_node(());
    let _ = shortest_path_bidirectional(
        &graph,
        a,
        NodeId::from_index(3),
        TraversalDirection::Outgoing,
        &mut ShortestPathScratch::default(),
    );
}
