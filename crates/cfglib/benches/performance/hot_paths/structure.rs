use cfglib::{
    Cfg, DominatorTree, Graph, IntervalAnalysis, NaturalLoop, NodeId, Rooted, SccDecomposition,
    detect_loops, kosaraju_scc, tarjan_scc,
};

use super::NODE_COUNT;
use crate::analysis_oracles::{assert_cfg_intervals, assert_reverse_chain_intervals};
use crate::fixtures::{multi_latch_graph, reverse_id_chain_graph};
use crate::harness::{BenchmarkSuite, benchmark_case};

/// Nodes of the one-path graphs, whose walk is as deep as the graph.
const DEEP_PATH_NODES: usize = 1 << 18;

/// `0 -> 1 -> ... -> nodes - 1`, closed back to `0` when `closed`.
fn deep_path(nodes: usize, closed: bool) -> Graph<(), ()> {
    let mut graph = Graph::with_capacity(nodes, nodes);
    let ids: Vec<NodeId> = (0..nodes).map(|_| graph.add_node(())).collect();
    for pair in ids.windows(2) {
        graph.add_edge(pair[0], pair[1], ());
    }
    if closed {
        graph.add_edge(ids[nodes - 1], ids[0], ());
    }
    graph
}

fn register_components_and_loops(suite: &mut BenchmarkSuite<'_>, graph: &Graph<(), ()>) {
    let (multi_latch, multi_latch_root) = multi_latch_graph(2_048, 256);
    let multi_latch_dominators =
        DominatorTree::compute(&Rooted::new(&multi_latch, multi_latch_root));
    benchmark_case!(
        suite,
        "directed_tarjan_scc",
        || tarjan_scc(graph),
        |result: &SccDecomposition<NodeId>| {
            let cycle_count = (NODE_COUNT - 1) / 32;
            assert_eq!(result.len(), NODE_COUNT - 16 * cycle_count);
            let mut seen = vec![false; NODE_COUNT];
            for (component_index, component) in result.components().enumerate() {
                assert_ne!(component, []);
                assert!(component.windows(2).all(|pair| pair[0] < pair[1]));
                for &node in component {
                    assert!(!seen[node.index()]);
                    seen[node.index()] = true;
                    assert_eq!(result.component_index(node), component_index);
                }
            }
            assert!(seen.into_iter().all(core::convert::identity));
            for cycle in 1..=cycle_count {
                let end = cycle * 32;
                let component = result.component(NodeId::from_index(end));
                assert_eq!(component.nodes().len(), 17);
                assert!(
                    (end - 16..=end).all(|index| component.contains(NodeId::from_index(index)))
                );
            }
            assert_eq!(
                result
                    .components()
                    .filter(|component| component.len() == 17)
                    .count(),
                cycle_count
            );
            assert!(
                result
                    .components()
                    .all(|component| component.len() == 1 || component.len() == 17)
            );
        }
    );
    // The deepest walk a graph of this size allows: one path, which is one
    // component per node, and the same path closed into one cycle.
    let path = deep_path(DEEP_PATH_NODES, false);
    benchmark_case!(
        suite,
        "directed_tarjan_scc_deep_path",
        || tarjan_scc(&path),
        |result: &SccDecomposition<NodeId>| {
            assert_eq!(result.len(), DEEP_PATH_NODES);
            assert_eq!(
                result.component_index(NodeId::from_index(0)),
                DEEP_PATH_NODES - 1
            );
        }
    );
    let cycle = deep_path(DEEP_PATH_NODES, true);
    benchmark_case!(
        suite,
        "directed_tarjan_scc_deep_cycle",
        || tarjan_scc(&cycle),
        |result: &SccDecomposition<NodeId>| assert_eq!(result.len(), 1)
    );
    benchmark_case!(
        suite,
        "directed_kosaraju_scc_deep_path",
        || kosaraju_scc(&path),
        |result: &SccDecomposition<NodeId>| {
            assert_eq!(result.len(), DEEP_PATH_NODES);
            assert_eq!(result.component_index(NodeId::from_index(0)), 0);
        }
    );
    benchmark_case!(
        suite,
        "directed_detect_loops_multilatch",
        || detect_loops(&multi_latch, &multi_latch_dominators),
        |result: &Vec<NaturalLoop<NodeId>>| {
            assert_eq!(result.len(), 1);
            let natural_loop = &result[0];
            assert_eq!(natural_loop.header, multi_latch_root);
            assert_eq!(natural_loop.depth, 0);
            assert_eq!(natural_loop.body.len(), 2_048 + 256 + 1);
            assert!(
                (0..=2_048 + 256)
                    .all(|index| natural_loop.body.contains(&NodeId::from_index(index)))
            );
            assert_eq!(natural_loop.latches.len(), 256);
            assert!(
                (2_049..=2_304)
                    .all(|index| natural_loop.latches.contains(&NodeId::from_index(index)))
            );
        }
    );
}

fn register_intervals(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>) {
    let (reverse_id_chain, reverse_id_root) = reverse_id_chain_graph(NODE_COUNT);
    benchmark_case!(
        suite,
        "cfg_interval_analysis",
        || IntervalAnalysis::compute(cfg),
        |result: &IntervalAnalysis| assert_cfg_intervals(result, cfg)
    );
    benchmark_case!(
        suite,
        "directed_interval_reverse_id_chain",
        || IntervalAnalysis::compute(&Rooted::new(&reverse_id_chain, reverse_id_root)),
        |result: &IntervalAnalysis<NodeId>| assert_reverse_chain_intervals(
            result,
            NODE_COUNT,
            reverse_id_root,
        )
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>, graph: &Graph<(), ()>) {
    register_components_and_loops(suite, graph);
    register_intervals(suite, cfg);
}
