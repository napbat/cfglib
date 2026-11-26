//! Property and edge-case tests for the arrangement and the feedback set.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{FeedbackConfig, feedback_arc_set, linear_arrangement};
use crate::graph::edge_view::FilteredEdges;
use crate::graph::scc::{SccDecomposition, SccNumbering};
use crate::graph::store::{EdgeId, Graph, NodeId};
use crate::graph::traverse::topological_sort;
use crate::test_util::rng::Rng;

const PLAIN: FeedbackConfig = FeedbackConfig::new();
const MINIMAL: FeedbackConfig = FeedbackConfig { minimal: true };

/// A random multigraph over up to `max_slots` node slots and about
/// `density` edges per slot: self-loops and parallel edges occur, and some
/// nodes and edges are removed.
fn random_multigraph(rng: &mut Rng, max_slots: usize, density: usize) -> Graph<(), ()> {
    let slots = rng.below(max_slots + 1);
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..slots).map(|_| graph.add_node(())).collect();
    if slots == 0 {
        return graph;
    }
    for _ in 0..rng.below(density * slots + 1) {
        let source = nodes[rng.below(slots)];
        let target = if rng.one_in(8) {
            source
        } else {
            nodes[rng.below(slots)]
        };
        let edge = graph.add_edge(source, target, ());
        if rng.one_in(6) {
            graph.add_edge(source, target, ());
        }
        if rng.one_in(10) {
            assert!(graph.remove_edge(edge));
        }
    }
    for &node in &nodes {
        if rng.one_in(6) {
            assert!(graph.remove_node(node));
        }
    }
    graph
}

/// A random acyclic multigraph: every edge runs forward in a shuffled order
/// of the slots, so node ids say nothing about direction, and some nodes are
/// removed.
fn random_dag(rng: &mut Rng, max_slots: usize) -> Graph<(), ()> {
    let slots = rng.below(max_slots + 1);
    let mut graph = Graph::new();
    let mut nodes: Vec<NodeId> = (0..slots).map(|_| graph.add_node(())).collect();
    rng.shuffle(&mut nodes);
    for _ in 0..rng.below(3 * slots + 1) {
        let (first, second) = (rng.below(slots), rng.below(slots));
        if first != second {
            let (source, target) = (nodes[first.min(second)], nodes[first.max(second)]);
            graph.add_edge(source, target, ());
            if rng.one_in(6) {
                graph.add_edge(source, target, ());
            }
        }
    }
    for &node in &nodes {
        if rng.one_in(6) {
            assert!(graph.remove_node(node));
        }
    }
    graph
}

/// Whether `graph` is acyclic once the edges of `set` are left out.
fn acyclic_without(graph: &Graph<(), ()>, set: &[EdgeId]) -> bool {
    let rest = FilteredEdges::new(graph, |edge: EdgeId, (): &()| !set.contains(&edge));
    topological_sort(&rest).is_some()
}

/// Each live node's rank in `order`.
fn ranks(graph: &Graph<(), ()>, order: &[NodeId]) -> Vec<usize> {
    let mut rank = vec![usize::MAX; graph.node_bound()];
    for (position, node) in order.iter().enumerate() {
        rank[node.index()] = position;
    }
    rank
}

/// Check every documented property of both functions on `graph`.
fn check(graph: &Graph<(), ()>) {
    let order = linear_arrangement(graph);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        graph.node_ids().collect::<Vec<_>>(),
        "a permutation of the live nodes"
    );
    let components = SccDecomposition::compute(graph, SccNumbering::SourcesFirst);
    assert!(
        order.windows(2).all(|pair| {
            components.component_index(pair[0]) <= components.component_index(pair[1])
        }),
        "components in sources-first order"
    );

    let rank = ranks(graph, &order);
    let set = feedback_arc_set(graph, PLAIN);
    let backward: Vec<EdgeId> = graph
        .edge_ids()
        .filter(|&edge| {
            let record = graph.edge(edge);
            rank[record.target().index()] <= rank[record.source().index()]
        })
        .collect();
    assert_eq!(set, backward, "exactly the backward edges, in id order");
    assert!(acyclic_without(graph, &set));
    assert_eq!(set.is_empty(), topological_sort(graph).is_some());

    // Each component's backward edges, self-loops aside, are at most half of
    // its edges, self-loops aside.
    let mut inside = vec![0_usize; components.len()];
    let mut cut = vec![0_usize; components.len()];
    for edge in graph.edge_ids() {
        let record = graph.edge(edge);
        let (source, target) = (record.source(), record.target());
        let component = components.component_index(source);
        if source != target && component == components.component_index(target) {
            inside[component] += 1;
            if set.contains(&edge) {
                cut[component] += 1;
            }
        }
    }
    assert!(
        inside
            .iter()
            .zip(&cut)
            .all(|(inside, cut)| 2 * cut <= *inside)
    );

    let minimal = feedback_arc_set(graph, MINIMAL);
    assert_eq!(
        minimal,
        set.iter()
            .copied()
            .filter(|edge| minimal.contains(edge))
            .collect::<Vec<_>>(),
        "a subsequence of the plain set"
    );
    assert!(acyclic_without(graph, &minimal));
    for (index, &edge) in minimal.iter().enumerate() {
        let mut returned = minimal.clone();
        returned.remove(index);
        assert!(
            !acyclic_without(graph, &returned),
            "returning {edge:?} closes no cycle"
        );
    }

    assert_eq!(linear_arrangement(graph), order);
    assert_eq!(feedback_arc_set(graph, PLAIN), set);
    assert_eq!(feedback_arc_set(graph, MINIMAL), minimal);
}

#[test]
fn the_config_defaults_to_the_plain_set() {
    assert_eq!(FeedbackConfig::default(), FeedbackConfig { minimal: false });
    assert_eq!(FeedbackConfig::new(), FeedbackConfig::default());
    assert_ne!(MINIMAL, PLAIN);
}

#[test]
fn the_empty_graph_has_no_order_and_no_set() {
    let graph = Graph::<(), ()>::new();
    assert_eq!(linear_arrangement(&graph), Vec::new());
    assert_eq!(feedback_arc_set(&graph, PLAIN), Vec::new());
    assert_eq!(feedback_arc_set(&graph, MINIMAL), Vec::new());
}

#[test]
fn a_self_loop_is_the_whole_set_in_both_modes() {
    let mut graph = Graph::<(), ()>::new();
    let node = graph.add_node(());
    let spin = graph.add_edge(node, node, ());
    assert_eq!(linear_arrangement(&graph), vec![node]);
    assert_eq!(feedback_arc_set(&graph, PLAIN), vec![spin]);
    assert_eq!(feedback_arc_set(&graph, MINIMAL), vec![spin]);
}

#[test]
fn a_two_cycle_reports_every_parallel_backward_edge() {
    let mut graph = Graph::<(), ()>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let back = [graph.add_edge(b, a, ()), graph.add_edge(b, a, ())];
    graph.add_edge(a, b, ());
    graph.add_edge(a, b, ());

    assert_eq!(linear_arrangement(&graph), vec![a, b]);
    assert_eq!(feedback_arc_set(&graph, PLAIN), back);
    // Returning either edge closes a cycle with a forward edge, so both stay.
    assert_eq!(feedback_arc_set(&graph, MINIMAL), back);
    check(&graph);
}

#[test]
fn ties_go_to_the_smaller_node_id() {
    // A ring whose edges arrive out of order: every node has one edge in and
    // one out, so the first pick is the smallest id, and peeling then runs
    // the ring backward from its predecessor.
    let mut graph = Graph::<(), ()>::new();
    let nodes: Vec<NodeId> = (0..5).map(|_| graph.add_node(())).collect();
    for index in [2, 3, 4, 0, 1] {
        graph.add_edge(nodes[index], nodes[(index + 1) % 5], ());
    }
    let closing = graph
        .edge_ids()
        .find(|&edge| graph.edge(edge).source() == nodes[4])
        .expect("the ring has an edge out of its last node");

    assert_eq!(linear_arrangement(&graph), nodes);
    assert_eq!(feedback_arc_set(&graph, PLAIN), vec![closing]);
    check(&graph);
}

#[test]
fn the_largest_degree_difference_goes_first() {
    // `hub` has the highest id but one more edge out than in; taking it first
    // cuts the two edges into it rather than the three out of it.
    let mut graph = Graph::<(), ()>::new();
    let x = graph.add_node(());
    let y = graph.add_node(());
    let hub = graph.add_node(());
    let into_hub = [graph.add_edge(x, hub, ()), graph.add_edge(y, hub, ())];
    graph.add_edge(hub, x, ());
    graph.add_edge(hub, y, ());
    graph.add_edge(hub, x, ());

    // Peeling `hub` leaves `x` and `y` sinks; the first peeled ends last.
    assert_eq!(linear_arrangement(&graph), vec![hub, y, x]);
    assert_eq!(feedback_arc_set(&graph, PLAIN), into_hub);
    check(&graph);
}

#[test]
fn edges_between_components_point_forward_whatever_their_ids() {
    // Two rings joined by edges from the later-numbered ring to the earlier.
    let mut graph = Graph::<(), ()>::new();
    let nodes: Vec<NodeId> = (0..6).map(|_| graph.add_node(())).collect();
    for ring in [&nodes[..3], &nodes[3..]] {
        for index in 0..3 {
            graph.add_edge(ring[index], ring[(index + 1) % 3], ());
        }
    }
    graph.add_edge(nodes[4], nodes[1], ());
    graph.add_edge(nodes[5], nodes[0], ());

    let order = linear_arrangement(&graph);
    assert_eq!(&order[..3], &nodes[3..]);
    assert_eq!(feedback_arc_set(&graph, PLAIN).len(), 2, "one per ring");
    check(&graph);
}

#[test]
fn random_multigraphs_meet_every_property() {
    for seed in 0..500 {
        let mut rng = Rng::new(seed);
        check(&random_multigraph(&mut rng, 12, 3));
    }
    for seed in 0..40 {
        let mut rng = Rng::new(10_000 + seed);
        check(&random_multigraph(&mut rng, 48, 4));
    }
}

#[test]
fn random_dags_have_no_set_and_a_topological_arrangement() {
    for seed in 0..300 {
        let mut rng = Rng::new(seed);
        let graph = random_dag(&mut rng, 16);
        let order = linear_arrangement(&graph);
        let rank = ranks(&graph, &order);
        assert!(graph.edge_ids().all(|edge| {
            let record = graph.edge(edge);
            rank[record.source().index()] < rank[record.target().index()]
        }));
        assert_eq!(feedback_arc_set(&graph, PLAIN), Vec::new());
        assert_eq!(feedback_arc_set(&graph, MINIMAL), Vec::new());
        check(&graph);
    }
}

#[test]
fn the_minimality_pass_shrinks_some_random_sets() {
    let shrunk = (0..200)
        .filter(|&seed| {
            let graph = random_multigraph(&mut Rng::new(seed), 24, 4);
            feedback_arc_set(&graph, MINIMAL).len() < feedback_arc_set(&graph, PLAIN).len()
        })
        .count();
    assert!(shrunk > 0, "the pass re-admitted no edge on any graph");
}
