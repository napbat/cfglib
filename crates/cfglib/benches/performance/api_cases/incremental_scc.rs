//! `IncrementalScc` against decomposing again after every batch of edges.
//!
//! Two cases per graph start from it, add the same batches of random edges,
//! each one percent of the graph's edge count, and after every batch either
//! report it with `IncrementalScc::insert_edges` or decompose sources first from
//! scratch. Both clone the graph inside the measured operation and add the
//! same edges to it, so what differs is the update; a third case per graph
//! does only that shared part.
//!
//! - `call_graph`: the call tree that `graph.rs` gives the recursion cases,
//!   at 1,024 functions (every 64th function calls its caller back), with
//!   calls between random functions, as resolving indirect calls adds them.
//! - `nearly_acyclic`: 100,000 nodes in a chain, a short forward shortcut out
//!   of every node and a short back edge every thousand nodes. Ninety-nine
//!   new edges in a hundred point forward by up to 1,024 nodes, as citations
//!   point at older papers, and the rest back by up to 16.

use std::collections::BTreeSet;

use cfglib::{
    CallMetadata, Graph, IncrementalScc, NodeId, SccDecomposition, SccNumbering, call_graph,
};

use super::BenchmarkSuite;
use super::graph::call_tree_cfgs;
use crate::fixtures::fixture_u32;
use crate::harness::benchmark_case;

/// Functions of the call graph.
const CALL_FUNCTIONS: usize = 1_024;
/// Nodes of the nearly acyclic graph.
const ACYCLIC_NODES: usize = 100_000;
/// Batches added in one measured operation.
const BATCHES: usize = 8;

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let cfgs = call_tree_cfgs(CALL_FUNCTIONS);
    let functions: Vec<_> = cfgs
        .iter()
        .enumerate()
        .map(|(index, cfg)| (fixture_u32(index), cfg))
        .collect();
    let calls = call_graph(&functions);
    let mut draws = Draws(1);
    let call_batches = batches(calls.edge_count(), || {
        let source = draws.below(CALL_FUNCTIONS);
        let target = draws.below(CALL_FUNCTIONS);
        (NodeId::from_index(source), NodeId::from_index(target))
    });
    let call = CallMetadata {
        is_tail_call: false,
    };
    let seeded = IncrementalScc::compute(&calls);
    let added = BATCHES * call_batches[0].len();
    let expected = partition(&SccDecomposition::compute(
        &with_batches(&calls, &call_batches, &call),
        SccNumbering::SourcesFirst,
    ));
    benchmark_case!(
        suite,
        "incremental_scc_call_graph_edges_only",
        || with_batches(&calls, &call_batches, &call),
        |graph: &Graph<_, CallMetadata>| assert_eq!(graph.edge_count(), calls.edge_count() + added)
    );
    benchmark_case!(
        suite,
        "incremental_scc_call_graph_insert_edges",
        || insert_batches(&calls, &seeded, &call_batches, &call),
        |components: &IncrementalScc<NodeId>| {
            assert_eq!(partition(&components.decomposition()), expected);
        }
    );
    benchmark_case!(
        suite,
        "incremental_scc_call_graph_sources_first_per_batch",
        || decompose_per_batch(&calls, &call_batches, &call),
        |components: &SccDecomposition<NodeId>| {
            assert_eq!(partition(components), expected);
        }
    );

    let acyclic = nearly_acyclic(ACYCLIC_NODES);
    let mut draws = Draws(2);
    let acyclic_batches = batches(acyclic.edge_count(), || {
        let source = draws.below(ACYCLIC_NODES);
        let target = if draws.below(100) == 0 {
            source.saturating_sub(1 + draws.below(16))
        } else {
            (source + 1 + draws.below(1_024)).min(ACYCLIC_NODES - 1)
        };
        (NodeId::from_index(source), NodeId::from_index(target))
    });
    let added = BATCHES * acyclic_batches[0].len();
    let seeded = IncrementalScc::compute(&acyclic);
    let expected = partition(&SccDecomposition::compute(
        &with_batches(&acyclic, &acyclic_batches, &()),
        SccNumbering::SourcesFirst,
    ));
    benchmark_case!(
        suite,
        "incremental_scc_nearly_acyclic_edges_only",
        || with_batches(&acyclic, &acyclic_batches, &()),
        |graph: &Graph<(), ()>| assert_eq!(graph.edge_count(), acyclic.edge_count() + added)
    );
    benchmark_case!(
        suite,
        "incremental_scc_nearly_acyclic_insert_edges",
        || insert_batches(&acyclic, &seeded, &acyclic_batches, &()),
        |components: &IncrementalScc<NodeId>| {
            assert_eq!(partition(&components.decomposition()), expected);
        }
    );
    benchmark_case!(
        suite,
        "incremental_scc_nearly_acyclic_sources_first_per_batch",
        || decompose_per_batch(&acyclic, &acyclic_batches, &()),
        |components: &SccDecomposition<NodeId>| {
            assert_eq!(partition(components), expected);
        }
    );
}

/// [`BATCHES`] batches of `edge` draws, each one percent of `edge_count`.
fn batches(
    edge_count: usize,
    mut edge: impl FnMut() -> (NodeId, NodeId),
) -> Vec<Vec<(NodeId, NodeId)>> {
    let size = edge_count.div_ceil(100);
    (0..BATCHES)
        .map(|_| (0..size).map(|_| edge()).collect())
        .collect()
}

/// A copy of `graph` with every batch added.
fn with_batches<N: Clone, E: Clone>(
    graph: &Graph<N, E>,
    batches: &[Vec<(NodeId, NodeId)>],
    payload: &E,
) -> Graph<N, E> {
    let mut graph = graph.clone();
    for &(source, target) in batches.iter().flatten() {
        graph.add_edge(source, target, payload.clone());
    }
    graph
}

/// Add the batches to a copy of `graph` and report each to a copy of
/// `seeded`.
fn insert_batches<N: Clone, E: Clone>(
    graph: &Graph<N, E>,
    seeded: &IncrementalScc<NodeId>,
    batches: &[Vec<(NodeId, NodeId)>],
    payload: &E,
) -> IncrementalScc<NodeId> {
    let mut graph = graph.clone();
    let mut components = seeded.clone();
    for batch in batches {
        for &(source, target) in batch {
            graph.add_edge(source, target, payload.clone());
        }
        components.insert_edges(&graph, batch);
    }
    components
}

/// Add the batches to a copy of `graph`, decomposing it after each.
fn decompose_per_batch<N: Clone, E: Clone>(
    graph: &Graph<N, E>,
    batches: &[Vec<(NodeId, NodeId)>],
    payload: &E,
) -> SccDecomposition<NodeId> {
    let mut graph = graph.clone();
    let mut components = SccDecomposition::compute(&graph, SccNumbering::SourcesFirst);
    for batch in batches {
        for &(source, target) in batch {
            graph.add_edge(source, target, payload.clone());
        }
        components = SccDecomposition::compute(&graph, SccNumbering::SourcesFirst);
    }
    components
}

/// The components as sets of nodes, whatever their numbering.
fn partition(components: &SccDecomposition<NodeId>) -> BTreeSet<Vec<NodeId>> {
    components.components().map(<[NodeId]>::to_vec).collect()
}

/// A chain with a forward shortcut of up to 65 nodes out of every node, and
/// every thousandth node pointing 8 nodes back.
fn nearly_acyclic(node_count: usize) -> Graph<(), ()> {
    let mut graph = Graph::with_capacity(node_count, 2 * node_count);
    let nodes: Vec<NodeId> = (0..node_count).map(|_| graph.add_node(())).collect();
    let mut draws = Draws(3);
    for index in 0..node_count - 1 {
        graph.add_edge(nodes[index], nodes[index + 1], ());
        let shortcut = index + 2 + draws.below(64);
        if shortcut < node_count {
            graph.add_edge(nodes[index], nodes[shortcut], ());
        }
    }
    for index in (1_000..node_count).step_by(1_000) {
        graph.add_edge(nodes[index], nodes[index - 8], ());
    }
    graph
}

/// A splitmix64 sequence, so every run draws the same edges.
struct Draws(u64);

impl Draws {
    /// A value in `0..bound`, which must be nonzero.
    fn below(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^= mixed >> 31;
        let bound = u64::try_from(bound).expect("a bound fits in u64");
        usize::try_from(mixed % bound).expect("a value below a usize bound fits in usize")
    }
}
