//! `min_label_relaxation` over weighted control-flow shapes.
//!
//! `graph.rs` times the unit-weight walk of a branchy chain. These cases give
//! the edges different costs, as a minimum-cost labeling over code does: a
//! chain of if/else diamonds whose arms cost different amounts, and the
//! branchy chain with scattered weights.

use cfglib::{Graph, NodeId, TraversalDirection, min_label_relaxation};

use super::BenchmarkSuite;
use crate::harness::benchmark_case;

/// Diamonds in the chain. Each arm pair differs by a power of two, so the
/// cheaper arm, added first, is the one every later label depends on.
const DIAMONDS: u32 = 16;
/// Nodes of the branchy weighted graph.
const BRANCHY_NODES: usize = 4_096;

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let (diamonds, start, end) = diamond_chain(DIAMONDS);
    benchmark_case!(
        suite,
        "relax_weighted_diamond_chain",
        || min_label_relaxation(
            &diamonds,
            [(start, 0_u64)],
            TraversalDirection::Outgoing,
            |edge, label| label.checked_add(*edge.payload()),
        ),
        |labels: &Vec<Option<u64>>| assert_eq!(labels[end.index()], Some(0))
    );
    let branchy = weighted_branchy(BRANCHY_NODES);
    benchmark_case!(
        suite,
        "relax_weighted_branchy",
        || min_label_relaxation(
            &branchy,
            [(NodeId::from_index(0), 0_u64)],
            TraversalDirection::Outgoing,
            |edge, label| label.checked_add(*edge.payload()),
        ),
        |labels: &Vec<Option<u64>>| assert!(labels.iter().all(Option::is_some))
    );
}

/// `k` if/else diamonds in a row: a free arm added first and a costly arm
/// added second. Returns the graph, its start, and its end.
fn diamond_chain(k: u32) -> (Graph<(), u64>, NodeId, NodeId) {
    let mut graph = Graph::new();
    let start = graph.add_node(());
    let mut spine = start;
    for i in 0..k {
        let cheap = graph.add_node(());
        let costly = graph.add_node(());
        let next = graph.add_node(());
        graph.add_edge(spine, cheap, 0);
        graph.add_edge(spine, costly, 1_u64 << (k - i));
        graph.add_edge(cheap, next, 0);
        graph.add_edge(costly, next, 0);
        spine = next;
    }
    (graph, start, spine)
}

/// A chain with skip and back edges, each with a weight in `1..=64` spread
/// by a multiplicative hash of its insertion index.
fn weighted_branchy(node_count: usize) -> Graph<(), u64> {
    let mut graph = Graph::new();
    let nodes: Vec<_> = (0..node_count).map(|_| graph.add_node(())).collect();
    let mut weight = {
        let mut index = 0_u64;
        move || {
            index += 1;
            (index.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) + 1
        }
    };
    for index in 0..node_count - 1 {
        graph.add_edge(nodes[index], nodes[index + 1], weight());
    }
    for index in (0..node_count - 2).step_by(2) {
        graph.add_edge(nodes[index], nodes[index + 2], weight());
    }
    for index in (32..node_count).step_by(32) {
        graph.add_edge(nodes[index], nodes[index - 16], weight());
    }
    graph
}
