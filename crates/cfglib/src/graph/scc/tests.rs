use super::*;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::store::Graph;
use crate::test_util::ff;

#[test]
fn cfg_uses_generic_scc_algorithm() {
    let mut cfg = Cfg::new();
    let next = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(next).push(ff("next"));
    cfg.add_edge(cfg.entry(), next, EdgeKind::Fallthrough);

    let result = tarjan_scc(&cfg);
    assert_eq!(result.len(), 2);
    assert!(result.is_dag(&cfg));
}

#[test]
fn condensation_collapses_cycles_to_a_dag() {
    // entry -> (a <-> b) -> exit
    let mut graph = Graph::<&str, ()>::new();
    let entry = graph.add_node("entry");
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let exit = graph.add_node("exit");
    graph.add_edge(entry, a, ());
    graph.add_edge(a, b, ());
    graph.add_edge(b, a, ());
    graph.add_edge(b, exit, ());

    let condensed = condensation(&graph);
    assert_eq!(condensed.node_count(), 3);
    assert_eq!(condensed.edge_count(), 2, "cycle-internal edges collapse");
    assert!(tarjan_scc(&condensed).is_dag(&condensed));
    let cycle_component = condensed
        .node_ids()
        .find(|&node| condensed.node(node).nodes.len() == 2)
        .expect("the a/b component");
    assert!(condensed.node(cycle_component).contains(a));
    assert!(condensed.node(cycle_component).contains(b));
}

#[test]
fn directed_graph_cycle_forms_one_component() {
    let mut graph = Graph::<&str, ()>::new();
    let left = graph.add_node("left");
    let right = graph.add_node("right");
    graph.add_edge(left, right, ());
    graph.add_edge(right, left, ());

    let result = tarjan_scc(&graph);
    assert_eq!(result.len(), 1);
    assert!(result.component(left).contains(right));
    assert_eq!(result.component_index(left), result.component_index(right));
    assert!(!result.is_dag(&graph));
}

/// Shapes both algorithms must agree on: a chain, a diamond, a cycle with
/// a tail, two cycles in series, a self-loop beside a plain node, a
/// disconnected pair, and the empty graph.
fn fixtures() -> Vec<(&'static str, Graph<(), ()>)> {
    fn build(node_count: usize, edges: &[(usize, usize)]) -> Graph<(), ()> {
        let mut graph = Graph::<(), ()>::new();
        for _ in 0..node_count {
            graph.add_node(());
        }
        for &(from, to) in edges {
            graph.add_edge(NodeId::from_index(from), NodeId::from_index(to), ());
        }
        graph
    }

    vec![
        ("chain", build(3, &[(0, 1), (1, 2)])),
        ("diamond", build(4, &[(0, 1), (0, 2), (1, 3), (2, 3)])),
        (
            "cycle with a tail",
            build(4, &[(0, 1), (1, 2), (2, 1), (2, 3)]),
        ),
        (
            "two cycles in series",
            build(5, &[(0, 1), (1, 2), (2, 0), (1, 3), (3, 4), (4, 3)]),
        ),
        ("self loop", build(2, &[(0, 0)])),
        ("disconnected", build(4, &[(0, 1), (2, 3)])),
        ("empty", build(0, &[])),
    ]
}

fn partition(result: &SccDecomposition<NodeId>) -> BTreeSet<BTreeSet<NodeId>> {
    result
        .components
        .iter()
        .map(|component| component.nodes.clone())
        .collect()
}

#[test]
fn kosaraju_and_tarjan_agree_on_the_partition() {
    for (name, graph) in fixtures() {
        let kosaraju = kosaraju_scc(&graph);
        let tarjan = tarjan_scc(&graph);
        assert_eq!(partition(&kosaraju), partition(&tarjan), "{name}");
        assert_eq!(kosaraju.len(), tarjan.len(), "{name}");
        assert_eq!(kosaraju.is_dag(&graph), tarjan.is_dag(&graph), "{name}");
        for node in graph.node_ids() {
            assert_eq!(
                kosaraju.component(node).nodes,
                tarjan.component(node).nodes,
                "{name}"
            );
        }
    }
}

#[test]
fn kosaraju_numbers_components_topologically() {
    for (name, graph) in fixtures() {
        let kosaraju = kosaraju_scc(&graph);
        let tarjan = tarjan_scc(&graph);
        for edge in graph.edges() {
            let (from, to) = (edge.source(), edge.target());
            let (source, target) = (kosaraju.component_index(from), kosaraju.component_index(to));
            if kosaraju.component(from).contains(to) {
                assert_eq!(source, target, "{name}: an intra-component edge");
                continue;
            }
            // Sources first: an edge always runs from a lower component
            // index to a higher one.
            assert!(source < target, "{name}: {source} !< {target}");
            // Tarjan numbers the very same edge the other way round.
            assert!(
                tarjan.component_index(from) > tarjan.component_index(to),
                "{name}"
            );
        }
    }
}

/// `a <-> b <-> c` is a cycle, `b` enters the `d <-> e` cycle, `f` enters
/// `a`, and `g` is isolated. A branching frame (`b` has two successors)
/// makes this the shape that catches a depth-first walk which loses a
/// frame's remaining successors when a sibling subtree finishes.
fn nested() -> (Graph<&'static str, ()>, [NodeId; 7]) {
    let mut graph = Graph::<&str, ()>::new();
    let outer_a = graph.add_node("a");
    let outer_b = graph.add_node("b");
    let outer_c = graph.add_node("c");
    let inner_d = graph.add_node("d");
    let inner_e = graph.add_node("e");
    let entry_f = graph.add_node("f");
    let lone_g = graph.add_node("g");
    graph.add_edge(outer_a, outer_b, ());
    graph.add_edge(outer_b, outer_c, ());
    graph.add_edge(outer_c, outer_a, ());
    graph.add_edge(outer_b, inner_d, ());
    graph.add_edge(inner_d, inner_e, ());
    graph.add_edge(inner_e, inner_d, ());
    graph.add_edge(entry_f, outer_a, ());
    (
        graph,
        [outer_a, outer_b, outer_c, inner_d, inner_e, entry_f, lone_g],
    )
}

/// The component sequence, as payload names.
fn sequence(
    result: &SccDecomposition<NodeId>,
    graph: &Graph<&'static str, ()>,
) -> Vec<Vec<&'static str>> {
    result
        .components
        .iter()
        .map(|component| {
            component
                .nodes
                .iter()
                .map(|&node| *graph.node(node))
                .collect()
        })
        .collect()
}

#[test]
fn kosaraju_numbering_matches_the_hand_computed_two_pass_order() {
    let (graph, [_, outer_b, _, _, inner_e, entry_f, lone_g]) = nested();

    // Pass 1 finishes [c, e, d, b, a, f, g], so pass 2 walks that in
    // reverse and mints components from the roots g, f, a, d — the
    // isolated node first because it finished last.
    let result = kosaraju_scc(&graph);
    assert_eq!(
        sequence(&result, &graph),
        vec![vec!["g"], vec!["f"], vec!["a", "b", "c"], vec!["d", "e"]]
    );
    assert_eq!(result.component_index(lone_g), 0);
    assert_eq!(result.component_index(entry_f), 1);
    assert_eq!(result.component_index(outer_b), 2);
    assert_eq!(result.component_index(inner_e), 3);
}

#[test]
fn tarjan_numbering_matches_the_hand_computed_leaves_first_order() {
    let (graph, [outer_a, _, _, _, inner_e, entry_f, lone_g]) = nested();

    // One pass from `a`: `c` closes the cycle back to `a` without minting,
    // then `b`'s second successor `d` mints the inner cycle first, then
    // `a` mints its own. `f` and `g` follow as later roots, in node order.
    let result = tarjan_scc(&graph);
    assert_eq!(
        sequence(&result, &graph),
        vec![vec!["d", "e"], vec!["a", "b", "c"], vec!["f"], vec!["g"]]
    );
    assert_eq!(result.component_index(inner_e), 0);
    assert_eq!(result.component_index(outer_a), 1);
    assert_eq!(result.component_index(entry_f), 2);
    assert_eq!(result.component_index(lone_g), 3);
}

/// The component pairs of a condensation, as a set.
fn condensed_pairs(dag: &Graph<(), ()>) -> BTreeSet<(usize, usize)> {
    dag.edges()
        .map(|edge| (edge.source().index(), edge.target().index()))
        .collect()
}

#[test]
fn condensation_of_numbers_its_nodes_by_the_decomposition_it_is_given() {
    for (name, graph) in fixtures() {
        for (algorithm, components) in [
            ("tarjan", tarjan_scc(&graph)),
            ("kosaraju", kosaraju_scc(&graph)),
        ] {
            let dag = condensation_of(&graph, &components);
            assert_eq!(dag.node_count(), components.len(), "{name}/{algorithm}");

            // Node index IS component index: every graph edge crossing two
            // components appears as that pair, and nothing else does.
            let expected: BTreeSet<(usize, usize)> = graph
                .edges()
                .map(|edge| {
                    (
                        components.component_index(edge.source()),
                        components.component_index(edge.target()),
                    )
                })
                .filter(|&(from, to)| from != to)
                .collect();
            assert_eq!(condensed_pairs(&dag), expected, "{name}/{algorithm}");
            assert!(tarjan_scc(&dag).is_dag(&dag), "{name}/{algorithm}");
        }

        // And under sources-first numbering the DAG's own edges only ever
        // run upwards — the property a single ascending pass rides on.
        let sources_first = kosaraju_scc(&graph);
        let dag = condensation_of(&graph, &sources_first);
        for (from, to) in condensed_pairs(&dag) {
            assert!(from < to, "{name}: {from} !< {to}");
        }
    }
}

#[test]
fn condensation_of_deduplicates_parallel_and_repeated_component_edges() {
    // Two nodes of the same component both point into the other, one of
    // them twice: three graph edges, one component edge.
    let mut graph = Graph::<(), ()>::new();
    let left = graph.add_node(());
    let right = graph.add_node(());
    let sink = graph.add_node(());
    graph.add_edge(left, right, ());
    graph.add_edge(right, left, ());
    graph.add_edge(left, sink, ());
    graph.add_edge(left, sink, ());
    graph.add_edge(right, sink, ());

    let components = kosaraju_scc(&graph);
    let dag = condensation_of(&graph, &components);
    assert_eq!(dag.node_count(), 2);
    assert_eq!(dag.edge_count(), 1);
    assert_eq!(
        condensed_pairs(&dag),
        [(
            components.component_index(left),
            components.component_index(sink)
        )]
        .into_iter()
        .collect()
    );
}

#[test]
fn condensation_of_tarjan_is_the_condensation_without_its_payloads() {
    for (name, graph) in fixtures() {
        let payloaded = condensation(&graph);
        let plain = condensation_of(&graph, &tarjan_scc(&graph));
        assert_eq!(plain.node_count(), payloaded.node_count(), "{name}");
        assert_eq!(
            condensed_pairs(&plain),
            payloaded
                .edges()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect(),
            "{name}"
        );
    }
}

#[test]
#[should_panic(expected = "the decomposition covers 4 nodes but the graph has 2")]
fn condensation_of_another_graphs_decomposition_panics() {
    let (four, _) = nested_pair();
    let mut two = Graph::<&str, ()>::new();
    let first = two.add_node("a");
    let second = two.add_node("b");
    two.add_edge(first, second, ());
    let _ = condensation_of(&two, &kosaraju_scc(&four));
}

/// A four-node graph for the mismatch test: `a <-> b`, `c -> d`.
fn nested_pair() -> (Graph<&'static str, ()>, [NodeId; 4]) {
    let mut graph = Graph::<&str, ()>::new();
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let c = graph.add_node("c");
    let d = graph.add_node("d");
    graph.add_edge(a, b, ());
    graph.add_edge(b, a, ());
    graph.add_edge(c, d, ());
    (graph, [a, b, c, d])
}

#[test]
fn self_edge_is_not_a_dag() {
    let mut graph = Graph::<(), ()>::new();
    let node = graph.add_node(());
    graph.add_edge(node, node, ());
    let result = tarjan_scc(&graph);
    assert!(result.component(node).is_singleton());
    assert!(!result.is_dag(&graph));
}
