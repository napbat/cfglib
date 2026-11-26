use cfglib::{
    BlockId, Cfg, DominatorTree, EdgeKind, Graph, LoopForest, NodeId, Rooted, SccDecomposition,
    SccNumbering,
};

use super::NODE_COUNT;
use crate::fixtures::multi_latch_graph;
use crate::harness::{BenchmarkSuite, benchmark_case};
use crate::rng::Rng;

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

/// Depth of the loop nests the cycle forest reads.
const NEST_DEPTH: usize = 100_000;

/// Blocks of the random CFG the cycle forest reads.
const RANDOM_CFG_BLOCKS: usize = 1_000_000;

/// Loops nested `depth` deep, with no tagged edge. Header `i` enters header
/// `i + 1` and its latch `i`, which returns to it, and latch `i + 1` returns
/// to latch `i`. With `irreducible`, header `i` also enters latch `i + 1`, a
/// second entry of every loop but the outermost.
fn deep_nest(depth: usize, irreducible: bool) -> Cfg<u32> {
    let mut cfg = Cfg::new();
    let headers: Vec<BlockId> = std::iter::once(cfg.entry())
        .chain((1..depth).map(|_| cfg.new_block()))
        .collect();
    let latches: Vec<BlockId> = (0..depth).map(|_| cfg.new_block()).collect();
    let exit = cfg.new_block();
    for index in 0..depth {
        if index + 1 < depth {
            cfg.add_edge(
                headers[index],
                headers[index + 1],
                EdgeKind::ConditionalTrue,
            );
            if irreducible {
                cfg.add_edge(headers[index], latches[index + 1], EdgeKind::Jump);
            }
            cfg.add_edge(latches[index + 1], latches[index], EdgeKind::Fallthrough);
        }
        cfg.add_edge(headers[index], latches[index], EdgeKind::ConditionalFalse);
        cfg.add_edge(latches[index], headers[index], EdgeKind::Jump);
    }
    cfg.add_edge(latches[0], exit, EdgeKind::ConditionalFalse);
    cfg
}

/// A chain of `blocks` blocks, each also jumping to a block drawn from a
/// fixed xorshift sequence: one within 64 blocks either way, like the ifs
/// and loops of a function, and one in 64 anywhere. No edge is tagged.
fn random_cfg(blocks: usize) -> Cfg<u32> {
    let mut cfg = Cfg::new();
    let ids: Vec<BlockId> = std::iter::once(cfg.entry())
        .chain((1..blocks).map(|_| cfg.new_block()))
        .collect();
    for pair in ids.windows(2) {
        cfg.add_edge(pair[0], pair[1], EdgeKind::Fallthrough);
    }
    let mut rng = Rng::xorshift(0x9E37_79B9_7F4A_7C15);
    for (index, &block) in ids.iter().enumerate() {
        let target = if rng.below(64) == 0 {
            rng.below(blocks)
        } else {
            (index + rng.below(128)).saturating_sub(64).min(blocks - 1)
        };
        cfg.add_edge(block, ids[target], EdgeKind::Jump);
    }
    cfg
}

/// The cycle forest of the deepest nests a graph of this size allows, and of
/// a random CFG whose outermost cycles are its strongly connected components.
fn register_cycles(suite: &mut BenchmarkSuite<'_>) {
    for (name, irreducible) in [
        ("cfg_cycle_forest_reducible_nest", false),
        ("cfg_cycle_forest_irreducible_nest", true),
    ] {
        let nest = deep_nest(NEST_DEPTH, irreducible);
        suite.case(
            name,
            &[],
            move || LoopForest::compute_cycles(&nest),
            move |forest: &LoopForest| {
                assert_eq!(forest.loop_count(), NEST_DEPTH);
                assert_eq!(forest.max_nesting_depth(), NEST_DEPTH);
                let irreducible_loops = forest
                    .loops()
                    .filter(|&lp| forest.is_irreducible(lp))
                    .count();
                assert_eq!(
                    irreducible_loops,
                    if irreducible { NEST_DEPTH - 1 } else { 0 }
                );
            },
        );
    }
    let random = random_cfg(RANDOM_CFG_BLOCKS);
    let components = SccDecomposition::compute(&random, SccNumbering::LeavesFirst);
    benchmark_case!(
        suite,
        "cfg_cycle_forest_random_1m",
        || LoopForest::compute_cycles(&random),
        |forest: &LoopForest| {
            let cyclic: Vec<usize> = components
                .components()
                .filter(|nodes| {
                    nodes.len() > 1 || random.successors(nodes[0]).any(|next| next == nodes[0])
                })
                .map(<[BlockId]>::len)
                .collect();
            let mut outermost: Vec<usize> = forest
                .loops()
                .filter(|&lp| forest.parent(lp).is_none())
                .map(|lp| forest.body_count(lp))
                .collect();
            let mut expected = cyclic;
            expected.sort_unstable();
            outermost.sort_unstable();
            assert_eq!(outermost, expected);
            assert!(forest.max_nesting_depth() > 2);
        }
    );
}

fn register_components_and_loops(suite: &mut BenchmarkSuite<'_>, graph: &Graph<(), ()>) {
    let (multi_latch, multi_latch_root) = multi_latch_graph(2_048, 256);
    let multi_latch_dominators =
        DominatorTree::compute(&Rooted::new(&multi_latch, multi_latch_root));
    benchmark_case!(
        suite,
        "directed_scc_leaves_first",
        || SccDecomposition::compute(graph, SccNumbering::LeavesFirst),
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
        "directed_scc_leaves_first_deep_path",
        || SccDecomposition::compute(&path, SccNumbering::LeavesFirst),
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
        "directed_scc_leaves_first_deep_cycle",
        || SccDecomposition::compute(&cycle, SccNumbering::LeavesFirst),
        |result: &SccDecomposition<NodeId>| assert_eq!(result.len(), 1)
    );
    benchmark_case!(
        suite,
        "directed_scc_sources_first_deep_path",
        || SccDecomposition::compute(&path, SccNumbering::SourcesFirst),
        |result: &SccDecomposition<NodeId>| {
            assert_eq!(result.len(), DEEP_PATH_NODES);
            assert_eq!(result.component_index(NodeId::from_index(0)), 0);
        }
    );
    benchmark_case!(
        suite,
        "directed_loop_forest_multilatch",
        || LoopForest::compute(&multi_latch, &multi_latch_dominators),
        |forest: &LoopForest<NodeId>| {
            assert_eq!(forest.loop_count(), 1);
            let lp = forest
                .loop_of_header(multi_latch_root)
                .expect("the root heads the loop");
            assert_eq!(forest.depth(lp), 0);
            assert_eq!(forest.body_count(lp), 2_048 + 256 + 1);
            assert!((0..=2_048 + 256).all(|index| forest.contains(lp, NodeId::from_index(index))));
            assert_eq!(
                forest.latches(lp),
                (2_049..=2_304).map(NodeId::from_index).collect::<Vec<_>>()
            );
        }
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>, graph: &Graph<(), ()>) {
    register_components_and_loops(suite, graph);
    register_cycles(suite);
}
