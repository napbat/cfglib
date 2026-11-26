//! `linear_arrangement` and `feedback_arc_set` over cyclic graphs.
//!
//! The timed cases run on the branchy chain behind `api_scc_sources_first`, whose
//! latches close a cycle every 32 nodes. Before the plain set is timed, its
//! oracle prints how the set compares with the depth-first back edges over
//! that chain and over a denser tangle. The sizes are reported, not asserted:
//! neither heuristic promises the smaller set.

use std::ops::ControlFlow;

use cfglib::{
    DfsEvent, Direction, EdgeId, FeedbackConfig, FilteredEdges, Graph, NodeId, depth_first_events,
    feedback_arc_set, linear_arrangement, topological_sort,
};

use super::BenchmarkSuite;
use crate::fixtures::branchy_graph;
use crate::harness::benchmark_case;

const NODE_COUNT: usize = 1_024;

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let graph = branchy_graph(NODE_COUNT);
    benchmark_case!(
        suite,
        "api_linear_arrangement",
        covers[linear_arrangement],
        || linear_arrangement(&graph),
        |order: &Vec<NodeId>| {
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert!(sorted.into_iter().eq(graph.node_ids()));
        }
    );
    benchmark_case!(
        suite,
        "api_feedback_arc_set",
        covers[feedback_arc_set],
        || feedback_arc_set(&graph, FeedbackConfig::new()),
        |set: &Vec<EdgeId>| {
            assert!(acyclic_without(&graph, set));
            report_quality();
        }
    );
    benchmark_case!(
        suite,
        "api_feedback_arc_set_minimal",
        || feedback_arc_set(&graph, FeedbackConfig::new().with_minimal(true)),
        |set: &Vec<EdgeId>| assert!(acyclic_without(&graph, set))
    );
}

/// Whether `graph` is acyclic without the edges of `set`, which is ascending.
fn acyclic_without(graph: &Graph<(), ()>, set: &[EdgeId]) -> bool {
    let rest = FilteredEdges::new(graph, |edge: EdgeId, (): &()| {
        set.binary_search(&edge).is_err()
    });
    topological_sort(&rest).is_some()
}

/// Print each fixture's set sizes beside its depth-first back edges.
fn report_quality() {
    println!("feedback arc set size (reported, not asserted):");
    println!(
        "  {:<8} {:>6} {:>6} {:>8} {:>15}",
        "fixture", "edges", "set", "minimal", "dfs back edges"
    );
    for (name, graph) in [
        ("branchy", branchy_graph(NODE_COUNT)),
        ("tangle", tangle_graph(NODE_COUNT)),
    ] {
        let set = feedback_arc_set(&graph, FeedbackConfig::new()).len();
        let minimal = feedback_arc_set(&graph, FeedbackConfig::new().with_minimal(true)).len();
        println!(
            "  {name:<8} {:>6} {set:>6} {minimal:>8} {:>15}",
            graph.edge_count(),
            depth_first_back_edges(&graph)
        );
    }
}

/// The back edges of one depth-first walk from node 0, which reaches every
/// node of both fixtures: the feedback set a consumer assembles by hand.
fn depth_first_back_edges(graph: &Graph<(), ()>) -> usize {
    let mut discovered = 0;
    let mut back_edges = 0;
    let outcome =
        depth_first_events::<_, ()>(graph, NodeId::from_raw(0), Direction::Forward, |event| {
            match event {
                DfsEvent::Discover(..) => discovered += 1,
                DfsEvent::BackEdge(..) => back_edges += 1,
                _ => {}
            }
            ControlFlow::Continue(())
        });
    assert_eq!(outcome, None);
    assert_eq!(discovered, graph.node_count(), "node 0 reaches every node");
    back_edges
}

/// A ring of `node_count` nodes, each with two more edges to nodes picked by
/// multiplicative hashing: one component of many overlapping cycles, the
/// shape of a cyclic dependency graph rather than of structured control flow.
fn tangle_graph(node_count: usize) -> Graph<(), ()> {
    let mut graph = Graph::with_capacity(node_count, 3 * node_count);
    let nodes: Vec<NodeId> = (0..node_count).map(|_| graph.add_node(())).collect();
    for index in 0..node_count {
        graph.add_edge(nodes[index], nodes[(index + 1) % node_count], ());
        graph.add_edge(nodes[index], nodes[(index * 7 + 3) % node_count], ());
        graph.add_edge(nodes[index], nodes[(index * 13 + 5) % node_count], ());
    }
    graph
}
