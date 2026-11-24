//! `propagate_summaries` and its widened form over one large recursive
//! cluster, where the order inside the component decides how often a
//! summary is recomputed.
//!
//! The ring of `graph.rs` has one cycle and one order that suits it. Here
//! every function calls three others drawn at random, so the whole graph is
//! one strongly connected component full of crossing cycles. Each case
//! returns its `compute` count beside its summaries, and the oracle prints
//! it, so a change of schedule shows in work as well as in time.

use std::collections::BTreeSet;

use cfglib::{
    Graph, NodeId, propagate_summaries, propagate_summaries_narrowed, propagate_summaries_widened,
};

use super::BenchmarkSuite;
use crate::harness::benchmark_case;

/// Functions in the cluster, and the calls each one makes.
const FUNCTIONS: usize = 2_048;
const CALLS: usize = 3;

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let cluster = cluster();
    let expected = kleene_masks(&cluster);
    benchmark_case!(
        suite,
        "cluster_propagate_summaries",
        || {
            let mut calls = 0_usize;
            let masks = propagate_summaries(&cluster, &0_u64, |graph, node, masks| {
                calls += 1;
                reached_mask(graph, node, masks)
            });
            (masks, calls)
        },
        |(masks, calls): &(Vec<u64>, usize)| {
            assert_eq!(*masks, expected);
            println!("cluster_propagate_summaries: {calls} computes");
        }
    );
    // The same cluster with a summary that costs what an effect set costs:
    // the sorted set of everything reached, by id modulo 256.
    let expected_sets: Vec<BTreeSet<u16>> = expected_sets(&cluster);
    benchmark_case!(
        suite,
        "cluster_propagate_summaries_sets",
        || {
            let mut calls = 0_usize;
            let sets = propagate_summaries(&cluster, &BTreeSet::new(), |graph, node, sets| {
                calls += 1;
                reached_set(graph, node, sets)
            });
            (sets, calls)
        },
        |(sets, calls): &(Vec<BTreeSet<u16>>, usize)| {
            assert_eq!(*sets, expected_sets);
            println!("cluster_propagate_summaries_sets: {calls} computes");
        }
    );
    benchmark_case!(
        suite,
        "cluster_propagate_summaries_widened",
        || {
            let (mut calls, mut widens) = (0_usize, 0_usize);
            let depths = propagate_summaries_widened(
                &cluster,
                &None,
                |graph, node, depths: &[Option<u32>]| {
                    calls += 1;
                    let deepest = graph
                        .successors(node)
                        .filter_map(|callee| depths[callee.index()])
                        .max();
                    Some(deepest.unwrap_or(0).saturating_add(1))
                },
                |_, previous, computed| {
                    widens += 1;
                    match (*previous, *computed) {
                        (Some(previous), Some(computed)) if computed > previous => Some(u32::MAX),
                        (previous, computed) => previous.max(computed),
                    }
                },
            );
            (depths, calls, widens)
        },
        |(depths, calls, widens): &(Vec<Option<u32>>, usize, usize)| {
            assert!(depths.iter().all(|depth| *depth == Some(u32::MAX)));
            println!("cluster_propagate_summaries_widened: {calls} computes, {widens} widens");
        }
    );
    register_narrowed(suite, &cluster);
}

/// A depth that every fourth function saturates at 1,000: widening sends the
/// cluster to the top, and narrowing brings back down every depth whose
/// cycles all pass a saturating function.
fn register_narrowed(suite: &mut BenchmarkSuite<'_>, cluster: &Graph<(), ()>) {
    benchmark_case!(
        suite,
        "api_propagate_summaries_narrowed",
        covers[propagate_summaries_narrowed],
        || {
            let mut calls = 0_usize;
            let depths = propagate_summaries_narrowed(
                cluster,
                &0_u32,
                |graph, node, depths| {
                    calls += 1;
                    let deepest = graph
                        .successors(node)
                        .map(|callee| depths[callee.index()])
                        .max();
                    let depth = deepest.unwrap_or(0).saturating_add(1);
                    if node.index().is_multiple_of(4) {
                        depth.min(1_000)
                    } else {
                        depth
                    }
                },
                |_, previous, computed| {
                    if computed > previous {
                        u32::MAX
                    } else {
                        *previous
                    }
                },
                |_, current, computed| {
                    if *current == u32::MAX {
                        *computed
                    } else {
                        *current
                    }
                },
            );
            (depths, calls)
        },
        |(depths, calls): &(Vec<u32>, usize)| {
            let bounded = depths.iter().filter(|depth| **depth < u32::MAX).count();
            assert!(bounded > 0, "narrowing brings some depth back down");
            println!("api_propagate_summaries_narrowed: {calls} computes, {bounded} bounded");
        }
    );
}

/// Every function calls the next one, so the cluster is strongly connected,
/// and `CALLS - 1` more drawn by a seeded xorshift.
fn cluster() -> Graph<(), ()> {
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..FUNCTIONS).map(|_| graph.add_node(())).collect();
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    for (index, &caller) in nodes.iter().enumerate() {
        graph.add_edge(caller, nodes[(index + 1) % FUNCTIONS], ());
        for _ in 1..CALLS {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let callee = usize::try_from(state % FUNCTIONS as u64).expect("an index fits");
            graph.add_edge(caller, nodes[callee], ());
        }
    }
    graph
}

/// The ids, modulo 64, of everything `node` reaches through its callees.
fn reached_mask(graph: &Graph<(), ()>, node: NodeId, masks: &[u64]) -> u64 {
    graph
        .successors(node)
        .fold(1_u64 << (node.index() % 64), |mask, callee| {
            mask | masks[callee.index()]
        })
}

/// The least fixpoint of [`reached_mask`] by plain Kleene iteration: the
/// oracle, independent of any schedule.
fn kleene_masks(graph: &Graph<(), ()>) -> Vec<u64> {
    let mut masks = vec![0_u64; graph.node_bound()];
    loop {
        let mut changed = false;
        for node in graph.node_ids() {
            let mask = reached_mask(graph, node, &masks);
            if mask != masks[node.index()] {
                masks[node.index()] = mask;
                changed = true;
            }
        }
        if !changed {
            return masks;
        }
    }
}

/// The ids, modulo 256, of everything `node` reaches through its callees.
fn reached_set(graph: &Graph<(), ()>, node: NodeId, sets: &[BTreeSet<u16>]) -> BTreeSet<u16> {
    let mut reached =
        BTreeSet::from([u16::try_from(node.index() % 256).expect("a residue fits in u16")]);
    for callee in graph.successors(node) {
        reached.extend(sets[callee.index()].iter().copied());
    }
    reached
}

/// The least fixpoint of [`reached_set`] by plain Kleene iteration.
fn expected_sets(graph: &Graph<(), ()>) -> Vec<BTreeSet<u16>> {
    let mut sets = vec![BTreeSet::new(); graph.node_bound()];
    loop {
        let mut changed = false;
        for node in graph.node_ids() {
            let set = reached_set(graph, node, &sets);
            if set != sets[node.index()] {
                sets[node.index()] = set;
                changed = true;
            }
        }
        if !changed {
            return sets;
        }
    }
}
