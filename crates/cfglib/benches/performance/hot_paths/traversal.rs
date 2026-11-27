use cfglib::{
    BlockId, Cfg, CommonAncestor, Direction, EdgeId, Graph, GraphView, NodeId, ShortestPathScratch,
    breadth_first, breadth_first_edges, common_ancestors, depth_first_preorder,
    nearest_common_ancestor, shortest_path, shortest_path_edges, shortest_path_edges_in,
    shortest_path_in,
};

use super::NODE_COUNT;
use crate::analysis_oracles::{
    assert_common_ancestor_results, assert_edge_path, assert_edge_traversal, assert_node_path,
    directed_distances, reference_cfg_breadth_first, reference_cfg_preorder,
};
use crate::fixtures::branchy_graph;
use crate::harness::{BenchmarkSuite, benchmark_case};
use crate::structural_oracles::assert_dense_permutation;

/// Nodes of the branchy graph behind the large meet cases.
const LARGE_NODE_COUNT: usize = 1 << 16;
/// Nodes of the hub-heavy graph behind the repeated point queries.
const HUB_GRAPH_NODES: usize = 1 << 17;
/// Hubs of that graph, each called by and calling `HUB_FAN` other nodes.
const HUBS: usize = 64;
const HUB_FAN: usize = 1_024;
/// Point queries one measured operation of the hub cases answers.
const POINT_QUERIES: usize = 16;

fn register_cfg_walks(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>) {
    benchmark_case!(
        suite,
        "cfg_depth_first_preorder",
        || depth_first_preorder(cfg, cfg.entry(), Direction::Forward),
        |result: &Vec<BlockId>| {
            assert_dense_permutation(result, NODE_COUNT);
            assert_eq!(*result, reference_cfg_preorder(cfg));
        }
    );
    benchmark_case!(
        suite,
        "cfg_breadth_first",
        || breadth_first(cfg, cfg.entry(), Direction::Forward),
        |result: &Vec<BlockId>| {
            assert_dense_permutation(result, NODE_COUNT);
            assert_eq!(*result, reference_cfg_breadth_first(cfg));
        }
    );
}

fn register_graph_walks(suite: &mut BenchmarkSuite<'_>, graph: &Graph<(), ()>) {
    benchmark_case!(
        suite,
        "directed_breadth_first_edges",
        || breadth_first_edges(graph, NodeId::from_raw(0), Direction::Forward),
        |result: &Vec<EdgeId>| assert_edge_traversal(result, graph)
    );
    benchmark_case!(
        suite,
        "directed_shortest_path",
        || {
            shortest_path(
                graph,
                NodeId::from_raw(0),
                NodeId::from_index(NODE_COUNT - 1),
                Direction::Forward,
            )
            .expect("fixture target is reachable")
        },
        |result: &Vec<NodeId>| assert_node_path(
            result,
            graph,
            NodeId::from_raw(0),
            NodeId::from_index(NODE_COUNT - 1),
        )
    );
    benchmark_case!(
        suite,
        "directed_shortest_path_edges",
        || {
            shortest_path_edges(
                graph,
                NodeId::from_raw(0),
                NodeId::from_index(NODE_COUNT - 1),
                Direction::Forward,
            )
            .expect("fixture target is reachable")
        },
        |result: &Vec<EdgeId>| assert_edge_path(
            result,
            graph,
            NodeId::from_raw(0),
            NodeId::from_index(NODE_COUNT - 1),
        )
    );
    // One scratch serves both scratch-reusing cases and every iteration of
    // each, the way a pass of point queries holds it.
    let mut scratch = ShortestPathScratch::default();
    benchmark_case!(
        suite,
        "directed_shortest_path_in",
        || {
            shortest_path_in(
                &mut scratch,
                graph,
                NodeId::from_raw(0),
                NodeId::from_index(NODE_COUNT - 1),
                Direction::Forward,
            )
            .expect("fixture target is reachable")
        },
        |result: &Vec<NodeId>| assert_node_path(
            result,
            graph,
            NodeId::from_raw(0),
            NodeId::from_index(NODE_COUNT - 1),
        )
    );
    benchmark_case!(
        suite,
        "directed_shortest_path_edges_in",
        || {
            shortest_path_edges_in(
                &mut scratch,
                graph,
                NodeId::from_raw(0),
                NodeId::from_index(NODE_COUNT - 1),
                Direction::Forward,
            )
            .expect("fixture target is reachable")
        },
        |result: &Vec<EdgeId>| assert_edge_path(
            result,
            graph,
            NodeId::from_raw(0),
            NodeId::from_index(NODE_COUNT - 1),
        )
    );
}

/// A deterministic spread of `value` over `0..HUB_GRAPH_NODES`.
fn hashed(value: usize) -> usize {
    (value.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 7) % HUB_GRAPH_NODES
}

/// A whole-program call graph in miniature: every function calls two others
/// at random, and the first `HUBS` functions are each called by, and call,
/// `HUB_FAN` more. A one-sided search reaches a hub within a few hops and
/// fans out over most of the graph before it meets a distant goal.
fn hub_graph() -> Graph<(), ()> {
    let mut graph = Graph::with_capacity(HUB_GRAPH_NODES, 2 * (HUB_GRAPH_NODES + HUBS * HUB_FAN));
    let nodes: Vec<NodeId> = (0..HUB_GRAPH_NODES).map(|_| graph.add_node(())).collect();
    for index in 0..HUB_GRAPH_NODES {
        graph.add_edge(nodes[index], nodes[hashed(index)], ());
        graph.add_edge(nodes[index], nodes[hashed(index + HUB_GRAPH_NODES)], ());
    }
    for hub in 0..HUBS {
        for fan in 0..HUB_FAN {
            let seed = 2 * HUB_GRAPH_NODES + 2 * (hub * HUB_FAN + fan);
            graph.add_edge(nodes[hashed(seed)], nodes[hub], ());
            graph.add_edge(nodes[hub], nodes[hashed(seed + 1)], ());
        }
    }
    graph
}

/// Repeated point queries over one large graph: a fresh scratch per query,
/// against one scratch reused by every query, on the same pairs.
fn register_point_queries(suite: &mut BenchmarkSuite<'_>) {
    let graph = hub_graph();
    let pairs: Vec<(NodeId, NodeId)> = (0..POINT_QUERIES)
        .map(|query| {
            let seed = 4 * HUB_GRAPH_NODES + 2 * query;
            (
                NodeId::from_index(hashed(seed)),
                NodeId::from_index(hashed(seed + 1)),
            )
        })
        .collect();
    let outgoing = Direction::Forward;
    let lengths: Vec<Option<usize>> = pairs
        .iter()
        .map(|&(start, goal)| {
            let (distances, _) = directed_distances(&graph, start, outgoing);
            (distances[goal.index()] != usize::MAX).then(|| distances[goal.index()] + 1)
        })
        .collect();
    let check = |paths: &Vec<Option<Vec<NodeId>>>| {
        for ((path, &(start, goal)), length) in paths.iter().zip(&pairs).zip(&lengths) {
            assert_eq!(path.as_ref().map(Vec::len), *length);
            if let Some(path) = path {
                assert_eq!((path[0], path[path.len() - 1]), (start, goal));
                assert!(path.windows(2).all(|pair| {
                    GraphView::successors(&graph, pair[0]).any(|next| next == pair[1])
                }));
            }
        }
    };
    benchmark_case!(
        suite,
        "directed_shortest_path_hub_queries",
        || {
            pairs
                .iter()
                .map(|&(start, goal)| shortest_path(&graph, start, goal, outgoing))
                .collect::<Vec<_>>()
        },
        check
    );
    let mut scratch = ShortestPathScratch::default();
    benchmark_case!(
        suite,
        "directed_shortest_path_in_hub_queries",
        || {
            pairs
                .iter()
                .map(|&(start, goal)| shortest_path_in(&mut scratch, &graph, start, goal, outgoing))
                .collect::<Vec<_>>()
        },
        check
    );
}

/// The meet `nearest_common_ancestor` must return, from two full searches.
fn expected_meet(graph: &Graph<(), ()>, a: NodeId, b: NodeId) -> Option<NodeId> {
    let (from_a, _) = directed_distances(graph, a, Direction::Backward);
    let (from_b, _) = directed_distances(graph, b, Direction::Backward);
    graph
        .node_ids()
        .filter(|node| from_a[node.index()] != usize::MAX && from_b[node.index()] != usize::MAX)
        .min_by_key(|node| (from_a[node.index()] + from_b[node.index()], node.index()))
}

fn register_ancestors(suite: &mut BenchmarkSuite<'_>, graph: &Graph<(), ()>) {
    benchmark_case!(
        suite,
        "directed_nearest_common_ancestor",
        || {
            nearest_common_ancestor(
                graph,
                NodeId::from_index(NODE_COUNT - 1),
                NodeId::from_index(NODE_COUNT - 2),
                Direction::Backward,
            )
        },
        |result: &Option<NodeId>| assert_eq!(
            *result,
            expected_meet(
                graph,
                NodeId::from_index(NODE_COUNT - 1),
                NodeId::from_index(NODE_COUNT - 2)
            )
        )
    );
    // Two endpoints whose meet is near, and two whose meet is about a hundred
    // hops away, in a graph in which every node reaches both.
    let large = branchy_graph(LARGE_NODE_COUNT);
    let a = NodeId::from_index(LARGE_NODE_COUNT - 1);
    let (near, far) = (
        NodeId::from_index(LARGE_NODE_COUNT - 2),
        NodeId::from_index(LARGE_NODE_COUNT - 201),
    );
    let (near_meet, far_meet) = (
        expected_meet(&large, a, near),
        expected_meet(&large, a, far),
    );
    benchmark_case!(
        suite,
        "directed_nearest_common_ancestor_large_near",
        || nearest_common_ancestor(&large, a, near, Direction::Backward),
        |result: &Option<NodeId>| assert_eq!(*result, near_meet)
    );
    benchmark_case!(
        suite,
        "directed_nearest_common_ancestor_large_far",
        || nearest_common_ancestor(&large, a, far, Direction::Backward),
        |result: &Option<NodeId>| assert_eq!(*result, far_meet)
    );
    benchmark_case!(
        suite,
        "directed_common_ancestors",
        || {
            common_ancestors(
                graph,
                NodeId::from_index(NODE_COUNT - 1),
                NodeId::from_index(NODE_COUNT - 2),
                Direction::Backward,
                None,
            )
        },
        |result: &Vec<CommonAncestor<NodeId>>| assert_common_ancestor_results(
            result,
            graph,
            NodeId::from_index(NODE_COUNT - 1),
            NodeId::from_index(NODE_COUNT - 2),
        )
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>, graph: &Graph<(), ()>) {
    register_cfg_walks(suite, cfg);
    register_graph_walks(suite, graph);
    register_point_queries(suite);
    register_ancestors(suite, graph);
}
