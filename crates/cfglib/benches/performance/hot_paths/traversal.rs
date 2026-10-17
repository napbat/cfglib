use cfglib::{
    BlockId, Cfg, CommonAncestor, EdgeId, EdgeStep, Graph, NodeId, TraversalDirection,
    breadth_first, breadth_first_edges, common_ancestors, depth_first_preorder,
    nearest_common_ancestor, shortest_path, shortest_path_edges,
};

use super::NODE_COUNT;
use crate::analysis_oracles::{
    assert_common_ancestor_results, assert_edge_path, assert_edge_traversal, assert_node_path,
    directed_distances, reference_cfg_breadth_first, reference_cfg_preorder,
};
use crate::harness::{BenchmarkSuite, benchmark_case};
use crate::structural_oracles::assert_dense_permutation;

fn register_cfg_walks(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>) {
    benchmark_case!(
        suite,
        "cfg_depth_first_preorder",
        || depth_first_preorder(cfg, cfg.entry(), TraversalDirection::Outgoing),
        |result: &Vec<BlockId>| {
            assert_dense_permutation(result, NODE_COUNT);
            assert_eq!(*result, reference_cfg_preorder(cfg));
        }
    );
    benchmark_case!(
        suite,
        "cfg_breadth_first",
        || breadth_first(cfg, cfg.entry(), TraversalDirection::Outgoing),
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
        || breadth_first_edges(graph, NodeId::from_raw(0), TraversalDirection::Outgoing),
        |result: &Vec<EdgeStep>| assert_edge_traversal(result, graph)
    );
    benchmark_case!(
        suite,
        "directed_shortest_path",
        || {
            shortest_path(
                graph,
                NodeId::from_raw(0),
                NodeId::from_index(NODE_COUNT - 1),
                TraversalDirection::Outgoing,
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
                TraversalDirection::Outgoing,
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

fn register_ancestors(suite: &mut BenchmarkSuite<'_>, graph: &Graph<(), ()>) {
    benchmark_case!(
        suite,
        "directed_nearest_common_ancestor",
        || {
            nearest_common_ancestor(
                graph,
                NodeId::from_index(NODE_COUNT - 1),
                NodeId::from_index(NODE_COUNT - 2),
                TraversalDirection::Incoming,
            )
        },
        |result: &Option<NodeId>| {
            let a = NodeId::from_index(NODE_COUNT - 1);
            let b = NodeId::from_index(NODE_COUNT - 2);
            let (from_a, _) = directed_distances(graph, a, TraversalDirection::Incoming);
            let (from_b, _) = directed_distances(graph, b, TraversalDirection::Incoming);
            let expected = graph
                .node_ids()
                .filter(|node| {
                    from_a[node.index()] != usize::MAX && from_b[node.index()] != usize::MAX
                })
                .min_by_key(|node| (from_a[node.index()] + from_b[node.index()], node.index()));
            assert_eq!(*result, expected);
        }
    );
    benchmark_case!(
        suite,
        "directed_common_ancestors",
        || {
            common_ancestors(
                graph,
                NodeId::from_index(NODE_COUNT - 1),
                NodeId::from_index(NODE_COUNT - 2),
                TraversalDirection::Incoming,
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
    register_ancestors(suite, graph);
}
