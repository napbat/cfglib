//! `ReachIndex` against one `reachable` walk per pair.
//!
//! Each fixture has three rows: computing the index, answering
//! `PAIR_COUNT` pseudo-random pairs through one reused scratch, and answering
//! the same pairs with one `reachable` walk each, which is what a consumer
//! pays per pair without the index. `branchy_graph` is the CFG shape of
//! `graph.rs`; the call graph is wide and shallow, with leaf functions and
//! small recursive cycles, which leaves the index more pairs to search.

use cfglib::{Direction, Graph, NodeId, ReachIndex, ReachIndexConfig, ReachScratch, reachable};

use super::BenchmarkSuite;
use crate::fixtures::branchy_graph;
use crate::harness::benchmark_case;

/// Nodes of the branchy CFG fixture.
const BRANCHY_NODES: usize = 1_024;
/// Functions of the call-graph fixture.
const CALL_GRAPH_FUNCTIONS: usize = 2_048;
/// Pairs answered by one measured query operation.
const PAIR_COUNT: usize = 10_000;

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let graph = branchy_graph(BRANCHY_NODES);
    let pairs = hashed_pairs(BRANCHY_NODES, PAIR_COUNT);
    let index = ReachIndex::compute(&graph, ReachIndexConfig::default());
    let mut scratch = ReachScratch::new();
    benchmark_case!(
        suite,
        "api_reach_index_build",
        || ReachIndex::compute(&graph, ReachIndexConfig::default()),
        |built: &ReachIndex<NodeId>| assert_eq!(built, &index)
    );
    benchmark_case!(
        suite,
        "api_reach_index_query",
        || answer_by_index(&index, &pairs, &mut scratch),
        |reached: &usize| assert_eq!(*reached, answer_by_walks(&graph, &pairs))
    );
    benchmark_case!(
        suite,
        "api_reach_index_reachable_pairs",
        || answer_by_walks(&graph, &pairs),
        |reached: &usize| assert_eq!(
            *reached,
            answer_by_index(&index, &pairs, &mut ReachScratch::new())
        )
    );

    let graph = call_graph(CALL_GRAPH_FUNCTIONS);
    let pairs = hashed_pairs(CALL_GRAPH_FUNCTIONS, PAIR_COUNT);
    let index = ReachIndex::compute(&graph, ReachIndexConfig::default());
    let mut scratch = ReachScratch::new();
    benchmark_case!(
        suite,
        "api_reach_index_build_call_graph",
        || ReachIndex::compute(&graph, ReachIndexConfig::default()),
        |built: &ReachIndex<NodeId>| assert_eq!(built, &index)
    );
    benchmark_case!(
        suite,
        "api_reach_index_query_call_graph",
        || answer_by_index(&index, &pairs, &mut scratch),
        |reached: &usize| assert_eq!(*reached, answer_by_walks(&graph, &pairs))
    );
    benchmark_case!(
        suite,
        "api_reach_index_reachable_pairs_call_graph",
        || answer_by_walks(&graph, &pairs),
        |reached: &usize| assert_eq!(
            *reached,
            answer_by_index(&index, &pairs, &mut ReachScratch::new())
        )
    );
}

/// How many of `pairs` the index says are reachable.
fn answer_by_index(
    index: &ReachIndex<NodeId>,
    pairs: &[(NodeId, NodeId)],
    scratch: &mut ReachScratch,
) -> usize {
    pairs
        .iter()
        .filter(|&&(source, target)| index.reaches(source, target, scratch))
        .count()
}

/// How many of `pairs` are reachable, by one `reachable` walk per pair.
fn answer_by_walks(graph: &Graph<(), ()>, pairs: &[(NodeId, NodeId)]) -> usize {
    pairs
        .iter()
        .filter(|&&(source, target)| reachable(graph, [source], Direction::Forward)[target.index()])
        .count()
}

/// `count` pairs of node indices below `node_count`, spread by a
/// multiplicative hash.
fn hashed_pairs(node_count: usize, count: usize) -> Vec<(NodeId, NodeId)> {
    (0..count)
        .map(|pair| {
            (
                NodeId::from_index(spread(2 * pair) % node_count),
                NodeId::from_index(spread(2 * pair + 1) % node_count),
            )
        })
        .collect()
}

/// `functions` functions numbered callers first: a quarter are leaves, the
/// others call two functions at most 64 ahead, and every 32nd also calls one
/// at most 8 back, closing a small recursive cycle. The reach index unit
/// tests print their coverage report over this same fixture.
fn call_graph(functions: usize) -> Graph<(), ()> {
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..functions).map(|_| graph.add_node(())).collect();
    for caller in 0..functions {
        if spread(4 * caller).is_multiple_of(4) {
            continue;
        }
        for call in 1..=2 {
            let callee = caller + 1 + spread(4 * caller + call) % 64;
            if callee < functions {
                graph.add_edge(nodes[caller], nodes[callee], ());
            }
        }
        if caller % 32 == 31 {
            let callee = caller - 1 - spread(4 * caller + 3) % 8;
            graph.add_edge(nodes[caller], nodes[callee], ());
        }
    }
    graph
}

/// A multiplicative hash of `value`, its top 24 bits.
fn spread(value: usize) -> usize {
    let value = u64::try_from(value).expect("a fixture index fits in u64");
    usize::try_from(value.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40)
        .expect("a 24-bit hash fits in usize")
}
