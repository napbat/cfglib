extern crate alloc;

use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;
use std::println;

use super::{ReachIndex, ReachIndexConfig, ReachScratch};
use crate::graph::store::{Graph, NodeId};
use crate::test_util::rng::Rng;

/// A configuration with these fields.
const fn config(orders: usize, supportive: usize, exact: Option<usize>) -> ReachIndexConfig {
    ReachIndexConfig {
        orders,
        supportive,
        exact_max_chains: exact,
    }
}

/// Both modes, with the order and supportive counts at zero, at one, past
/// the four fixed alternatives, and at the limit of 64.
const CONFIGS: [ReachIndexConfig; 10] = [
    ReachIndexConfig::new(),
    config(0, 0, None),
    config(0, 64, None),
    config(1, 0, None),
    config(6, 1, None),
    config(9, 7, None),
    config(2, 32, Some(usize::MAX)),
    config(0, 0, Some(usize::MAX)),
    config(2, 32, Some(4)),
    config(2, 32, Some(0)),
];

/// Per node slot, whether a breadth-first walk from `source` reaches it.
fn reached_from(graph: &Graph<(), ()>, source: NodeId) -> Vec<bool> {
    let mut reached = vec![false; graph.node_bound()];
    reached[source.index()] = true;
    let mut queue = VecDeque::from([source]);
    while let Some(node) = queue.pop_front() {
        for next in graph.successors(node) {
            if !reached[next.index()] {
                reached[next.index()] = true;
                queue.push_back(next);
            }
        }
    }
    reached
}

/// Uniformly random endpoints, so self-loops, parallel edges and large
/// strongly connected components occur.
fn uniform(rng: &mut Rng, nodes: usize, edges: usize) -> Graph<(), ()> {
    let mut graph = Graph::new();
    let ids: Vec<NodeId> = (0..nodes).map(|_| graph.add_node(())).collect();
    for _ in 0..edges {
        graph.add_edge(ids[rng.below(nodes)], ids[rng.below(nodes)], ());
    }
    graph
}

/// Mostly short forward edges with a few back edges, so the condensation is
/// a long, wide DAG of small components, as in a call graph.
fn forward_biased(rng: &mut Rng, nodes: usize, edges: usize) -> Graph<(), ()> {
    let mut graph = Graph::new();
    let ids: Vec<NodeId> = (0..nodes).map(|_| graph.add_node(())).collect();
    for _ in 0..edges {
        let source = rng.below(nodes);
        let reach = 1 + rng.below(40);
        let target = if rng.one_in(25) {
            source.saturating_sub(reach)
        } else {
            (source + reach).min(nodes - 1)
        };
        graph.add_edge(ids[source], ids[target], ());
    }
    graph
}

/// A structured function body of about `nodes` blocks, numbered in the order
/// a frontend creates them: sequences, if/else diamonds, loops with a back
/// edge, and early returns into the one exit. Every block is reachable from
/// the entry, and the condensation is narrow, as in one function's CFG.
fn structured(rng: &mut Rng, nodes: usize) -> Graph<(), ()> {
    let mut graph = Graph::new();
    let entry = graph.add_node(());
    let exit = graph.add_node(());
    let last = region(rng, &mut graph, entry, exit, nodes.saturating_sub(2));
    graph.add_edge(last, exit, ());
    graph
}

/// Lower a region of about `budget` blocks that control enters from
/// `entry`, returning the block it leaves from.
fn region(
    rng: &mut Rng,
    graph: &mut Graph<(), ()>,
    entry: NodeId,
    exit: NodeId,
    budget: usize,
) -> NodeId {
    if budget < 4 {
        let mut tail = entry;
        for _ in 0..budget {
            let next = graph.add_node(());
            graph.add_edge(tail, next, ());
            tail = next;
        }
        return tail;
    }
    match rng.below(20) {
        // A sequence, split near the middle so the recursion stays shallow.
        0..7 => {
            let first = budget / 4 + rng.below(budget / 2);
            let middle = region(rng, graph, entry, exit, first);
            region(rng, graph, middle, exit, budget - first)
        }
        7..13 => {
            let then_entry = graph.add_node(());
            let else_entry = graph.add_node(());
            graph.add_edge(entry, then_entry, ());
            graph.add_edge(entry, else_entry, ());
            let arms = budget - 3;
            let then_budget = rng.below(arms + 1);
            let then_exit = region(rng, graph, then_entry, exit, then_budget);
            let else_exit = region(rng, graph, else_entry, exit, arms - then_budget);
            let join = graph.add_node(());
            graph.add_edge(then_exit, join, ());
            graph.add_edge(else_exit, join, ());
            join
        }
        13..18 => {
            let header = graph.add_node(());
            graph.add_edge(entry, header, ());
            let latch = region(rng, graph, header, exit, budget - 2);
            graph.add_edge(latch, header, ());
            let after = graph.add_node(());
            graph.add_edge(header, after, ());
            after
        }
        _ => {
            let early = graph.add_node(());
            graph.add_edge(entry, early, ());
            graph.add_edge(early, exit, ());
            let rest = graph.add_node(());
            graph.add_edge(entry, rest, ());
            region(rng, graph, rest, exit, budget - 2)
        }
    }
}

/// The `branchy_graph` fixture of the API benchmarks, the CFG fixture of the
/// reach index benchmarks.
fn branchy(nodes: usize) -> Graph<(), ()> {
    let mut graph = Graph::new();
    let ids: Vec<NodeId> = (0..nodes).map(|_| graph.add_node(())).collect();
    for index in 0..nodes - 1 {
        graph.add_edge(ids[index], ids[index + 1], ());
    }
    for index in (0..nodes.saturating_sub(2)).step_by(2) {
        graph.add_edge(ids[index], ids[index + 2], ());
    }
    for index in (32..nodes).step_by(32) {
        graph.add_edge(ids[index], ids[index - 16], ());
    }
    graph
}

/// The call-graph fixture of the reach index benchmarks: a quarter of the
/// functions are leaves, the others call two functions at most 64 ahead, and
/// every 32nd also calls one at most 8 back.
fn bench_call_graph(functions: usize) -> Graph<(), ()> {
    let mut graph = Graph::new();
    let ids: Vec<NodeId> = (0..functions).map(|_| graph.add_node(())).collect();
    for caller in 0..functions {
        if spread(4 * caller).is_multiple_of(4) {
            continue;
        }
        for call in 1..=2 {
            let callee = caller + 1 + spread(4 * caller + call) % 64;
            if callee < functions {
                graph.add_edge(ids[caller], ids[callee], ());
            }
        }
        if caller % 32 == 31 {
            let callee = caller - 1 - spread(4 * caller + 3) % 8;
            graph.add_edge(ids[caller], ids[callee], ());
        }
    }
    graph
}

/// The benchmarks' multiplicative hash of `value`, its top 24 bits.
fn spread(value: usize) -> usize {
    let value = u64::try_from(value).expect("a fixture index fits in u64");
    usize::try_from(value.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40)
        .expect("a 24-bit hash fits in usize")
}

/// Remove about `percent` of the nodes, leaving their slots empty.
fn remove_some(rng: &mut Rng, graph: &mut Graph<(), ()>, percent: usize) {
    for index in 0..graph.node_bound() {
        if rng.chance(percent) {
            graph.remove_node(NodeId::from_index(index));
        }
    }
}

/// A random graph of up to `max_nodes` nodes from one of the generators,
/// sometimes with removed slots.
fn random_graph(rng: &mut Rng, max_nodes: usize) -> Graph<(), ()> {
    let nodes = 1 + rng.below(max_nodes);
    let mut graph = match rng.below(3) {
        0 => {
            let edges = rng.below(2 * nodes + 1);
            uniform(rng, nodes, edges)
        }
        1 => {
            let edges = nodes + rng.below(2 * nodes);
            forward_biased(rng, nodes, edges)
        }
        _ => structured(rng, nodes),
    };
    if rng.one_in(3) {
        remove_some(rng, &mut graph, 20);
    }
    graph
}

/// `count` random pairs of live nodes, drawn as `count.isqrt()` sources with
/// that many targets each, so the oracle walks once per source.
fn random_pairs(rng: &mut Rng, graph: &Graph<(), ()>, count: usize) -> Vec<(NodeId, Vec<NodeId>)> {
    let live: Vec<NodeId> = graph.node_ids().collect();
    let side = count.isqrt();
    (0..side)
        .map(|_| {
            let source = live[rng.below(live.len())];
            let targets = (0..side).map(|_| live[rng.below(live.len())]).collect();
            (source, targets)
        })
        .collect()
}

#[test]
fn all_pairs_match_breadth_first_search_on_small_graphs() {
    let mut rng = Rng::new(67);
    // One scratch for every graph and configuration, so it grows and shrinks
    // between indexes of different sizes.
    let mut scratch = ReachScratch::default();
    for _ in 0..48 {
        let graph = random_graph(&mut rng, 64);
        let live: Vec<NodeId> = graph.node_ids().collect();
        let oracle: Vec<Vec<bool>> = live
            .iter()
            .map(|&source| reached_from(&graph, source))
            .collect();
        for config in CONFIGS {
            let index = ReachIndex::compute(&graph, config);
            for (row, &source) in oracle.iter().zip(&live) {
                for &target in &live {
                    assert_eq!(
                        index.reaches_in(&mut scratch, source, target),
                        row[target.index()],
                        "{source:?} -> {target:?} under {config:?} in {graph:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn random_pairs_match_breadth_first_search_on_large_graphs() {
    const NODES: usize = 10_000;
    let mut rng = Rng::new(10_000);
    let mut scratch = ReachScratch::default();
    let mut wide = uniform(&mut rng, NODES, NODES);
    remove_some(&mut rng, &mut wide, 5);
    let mut biased = forward_biased(&mut rng, NODES, 2 * NODES);
    remove_some(&mut rng, &mut biased, 5);
    // The narrow graph keeps every slot: removing a block strands the blocks
    // that led only to it as sinks, and every sink ends a chain of the exact
    // mode.
    let narrow = structured(&mut rng, NODES);
    let graphs = [wide, biased, narrow];
    let configs = [
        ReachIndexConfig::new(),
        config(0, 0, None),
        config(4, 64, None),
        config(2, 32, Some(1_024)),
    ];
    for (kind, graph) in graphs.iter().enumerate() {
        let pairs = random_pairs(&mut rng, graph, 10_000);
        let oracle: Vec<Vec<bool>> = pairs
            .iter()
            .map(|&(source, _)| reached_from(graph, source))
            .collect();
        for config in configs {
            let index = ReachIndex::compute(graph, config);
            if kind == 2 && config.exact_max_chains.is_some() {
                assert!(
                    index.chain_count().is_some(),
                    "a narrow condensation fits the exact chain mode"
                );
            }
            for ((source, targets), reached) in pairs.iter().zip(&oracle) {
                for &target in targets {
                    assert_eq!(
                        index.reaches_in(&mut scratch, *source, target),
                        reached[target.index()],
                        "{source:?} -> {target:?} under {config:?} in graph {kind}"
                    );
                }
            }
        }
    }
}

#[test]
fn two_computations_compare_equal_and_answer_alike_with_or_without_a_scratch() {
    let mut rng = Rng::new(2);
    let mut scratch = ReachScratch::default();
    for _ in 0..12 {
        let graph = random_graph(&mut rng, 200);
        for config in CONFIGS {
            let first = ReachIndex::compute(&graph, config);
            let second = ReachIndex::compute(&graph, config);
            assert_eq!(first, second);
            for (source, targets) in random_pairs(&mut rng, &graph, 400) {
                for target in targets {
                    assert_eq!(
                        first.reaches_in(&mut scratch, source, target),
                        second.reaches(source, target)
                    );
                }
            }
        }
    }
}

#[test]
fn every_node_reaches_itself_and_a_cycle_reaches_around() {
    let mut graph = Graph::<(), ()>::new();
    let looped = graph.add_node(());
    let alone = graph.add_node(());
    let ring: Vec<NodeId> = (0..3).map(|_| graph.add_node(())).collect();
    graph.add_edge(looped, looped, ());
    for (index, &node) in ring.iter().enumerate() {
        graph.add_edge(node, ring[(index + 1) % ring.len()], ());
    }
    graph.add_edge(looped, ring[0], ());
    let mut scratch = ReachScratch::default();
    for config in CONFIGS {
        let index = ReachIndex::compute(&graph, config);
        assert!(index.reaches_in(&mut scratch, looped, looped));
        assert!(index.reaches_in(&mut scratch, alone, alone));
        assert!(index.reaches_in(&mut scratch, ring[2], ring[1]));
        assert!(index.reaches(looped, ring[2]));
        assert!(!index.reaches(ring[0], looped));
        assert!(!index.reaches_in(&mut scratch, alone, looped));
        assert!(!index.reaches_in(&mut scratch, looped, alone));
    }
}

#[test]
fn the_empty_graph_has_an_empty_index() {
    let graph = Graph::<(), ()>::new();
    let observed = ReachIndex::compute(&graph, ReachIndexConfig::new());
    assert_eq!(observed.chain_count(), None);
    let exact = ReachIndex::compute(&graph, config(2, 32, Some(0)));
    assert_eq!(exact.chain_count(), Some(0));
}

#[test]
fn a_diamond_chain_is_covered_by_two_chains() {
    let mut graph = Graph::<(), ()>::new();
    let mut spine = graph.add_node(());
    for _ in 0..20 {
        let left = graph.add_node(());
        let right = graph.add_node(());
        let join = graph.add_node(());
        graph.add_edge(spine, left, ());
        graph.add_edge(spine, right, ());
        graph.add_edge(left, join, ());
        graph.add_edge(right, join, ());
        spine = join;
    }
    let exact = config(2, 32, Some(2));
    assert_eq!(ReachIndex::compute(&graph, exact).chain_count(), Some(2));
    let too_few = config(2, 32, Some(1));
    assert_eq!(ReachIndex::compute(&graph, too_few).chain_count(), None);
}

#[test]
fn the_exact_mode_is_off_unless_requested() {
    let graph = branchy(64);
    assert_eq!(
        ReachIndex::compute(&graph, ReachIndexConfig::default()).chain_count(),
        None
    );
}

#[test]
fn the_default_configuration_keeps_two_orders_and_32_supportive_components() {
    let config = ReachIndexConfig::default();
    assert_eq!(config, ReachIndexConfig::new());
    assert_eq!(
        (config.orders, config.supportive, config.exact_max_chains),
        (2, 32, None)
    );
}

#[test]
#[should_panic(expected = "is not a node of the decomposed graph")]
fn a_removed_source_panics() {
    let mut graph = Graph::<(), ()>::new();
    let kept = graph.add_node(());
    let removed = graph.add_node(());
    graph.remove_node(removed);
    let index = ReachIndex::compute(&graph, ReachIndexConfig::new());
    let _ = index.reaches(removed, kept);
}

#[test]
#[should_panic(expected = "is not a node of the decomposed graph")]
fn a_removed_target_panics_in_exact_mode() {
    let mut graph = Graph::<(), ()>::new();
    let kept = graph.add_node(());
    let removed = graph.add_node(());
    graph.remove_node(removed);
    let index = ReachIndex::compute(&graph, config(2, 32, Some(8)));
    let _ = index.reaches_in(&mut ReachScratch::default(), kept, removed);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn a_node_beyond_the_bound_panics() {
    let mut graph = Graph::<(), ()>::new();
    let node = graph.add_node(());
    let index = ReachIndex::compute(&graph, ReachIndexConfig::new());
    let _ = index.reaches(node, NodeId::from_index(1));
}

#[test]
#[should_panic(expected = "at most 64 supportive components")]
fn more_than_64_supportive_components_panic() {
    let graph = branchy(8);
    let _ = ReachIndex::compute(&graph, config(2, 65, None));
}

/// Prints, per fixture, the share of queries answered without the fallback
/// search and the chain count of the exact mode (`None` past 1,024 chains).
/// A report, not a contract: nothing here is asserted.
#[test]
fn report_how_many_queries_skip_the_search() {
    const NODES: usize = 10_000;
    let mut rng = Rng::new(1);
    let fixtures = [
        ("bench branchy_graph(1_024)", branchy(1_024)),
        ("bench call graph(2_048)", bench_call_graph(2_048)),
        ("structured(10_000)", structured(&mut rng, NODES)),
        (
            "forward_biased(10_000, 20_000)",
            forward_biased(&mut rng, NODES, 2 * NODES),
        ),
        ("uniform(10_000, 10_000)", uniform(&mut rng, NODES, NODES)),
    ];
    for (name, graph) in &fixtures {
        let live: Vec<NodeId> = graph.node_ids().collect();
        let pairs: Vec<(NodeId, NodeId)> = (0..10_000)
            .map(|_| (live[rng.below(live.len())], live[rng.below(live.len())]))
            .collect();
        for config in [
            ReachIndexConfig::new(),
            config(0, 0, None),
            config(4, 64, None),
        ] {
            let index = ReachIndex::compute(graph, config);
            let mut scratch = ReachScratch::default();
            for &(source, target) in &pairs {
                let _ = index.reaches_in(&mut scratch, source, target);
            }
            let coverage = scratch.coverage;
            println!(
                "{name}: orders {}, supportive {}: {} of {} queries ({:.1}%) without the search",
                config.orders,
                config.supportive,
                coverage.queries - coverage.searched,
                coverage.queries,
                crate::usize_to_f64(coverage.queries - coverage.searched) * 100.0
                    / crate::usize_to_f64(coverage.queries),
            );
        }
        let exact = ReachIndex::compute(graph, config(2, 32, Some(1_024)));
        println!("{name}: exact mode chain count {:?}", exact.chain_count());
    }
}
