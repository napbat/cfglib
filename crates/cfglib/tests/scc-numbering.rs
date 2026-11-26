//! Integration tests: strongly connected components, numbering contracts and
//! stores with removed slots.

use std::collections::BTreeSet;

use cfglib::{Cfg, DenseId, EdgeKind, Graph, NodeId, NodeOrder, SccDecomposition, SccNumbering};

/// `entry -> (a <-> b) -> exit`, with a second path `entry -> exit`.
fn cycle_between_entry_and_exit() -> (Graph<&'static str, ()>, [NodeId; 4]) {
    let mut graph = Graph::<&str, ()>::new();
    let entry = graph.add_node("entry");
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let exit = graph.add_node("exit");
    graph.add_edge(entry, a, ());
    graph.add_edge(a, b, ());
    graph.add_edge(b, a, ());
    graph.add_edge(b, exit, ());
    graph.add_edge(entry, exit, ());
    (graph, [entry, a, b, exit])
}

fn partition<N: DenseId>(result: &SccDecomposition<N>) -> BTreeSet<BTreeSet<N>> {
    result
        .components()
        .map(|nodes| nodes.iter().copied().collect())
        .collect()
}

#[test]
fn sources_first_runs_forward_along_every_edge_and_leaves_first_backward() {
    let (graph, [entry, a, b, exit]) = cycle_between_entry_and_exit();
    let sources_first = SccDecomposition::compute(&graph, SccNumbering::SourcesFirst);
    let leaves_first = SccDecomposition::compute(&graph, SccNumbering::LeavesFirst);

    assert_eq!(partition(&sources_first), partition(&leaves_first));
    assert_eq!(
        sources_first.component_index(a),
        sources_first.component_index(b)
    );
    for edge in graph.edges() {
        let (from, to) = (edge.source(), edge.target());
        assert!(sources_first.component_index(from) <= sources_first.component_index(to));
        assert!(leaves_first.component_index(from) >= leaves_first.component_index(to));
    }
    assert_eq!(sources_first.component_index(entry), 0);
    assert_eq!(leaves_first.component_index(exit), 0);
}

#[test]
fn condensation_keeps_the_numbering_of_the_decomposition() {
    let (graph, [entry, a, _, exit]) = cycle_between_entry_and_exit();
    let components = SccDecomposition::compute(&graph, SccNumbering::SourcesFirst);
    let dag = components.condensation(&graph);

    assert_eq!(dag.node_count(), components.component_count());
    let component = |node: NodeId| NodeId::from_index(components.component_index(node));
    assert!(
        dag.successors(component(entry))
            .any(|next| next == component(a))
    );
    assert!(
        dag.successors(component(entry))
            .any(|next| next == component(exit))
    );
    assert!(
        dag.edges()
            .all(|edge| edge.source().index() < edge.target().index()),
        "sources-first numbering makes every component edge point forward"
    );

    let leaves_first = SccDecomposition::compute(&graph, SccNumbering::LeavesFirst);
    let reversed = leaves_first.condensation(&graph);
    assert_eq!(reversed.node_count(), dag.node_count());
    assert_eq!(reversed.edge_count(), dag.edge_count());
    assert!(
        reversed
            .edges()
            .all(|edge| edge.source().index() > edge.target().index()),
        "leaves-first numbering makes every component edge point backward"
    );
}

/// `a <-> b`, a removed slot `c`, and `d -> a`.
fn with_removed_slot() -> (Graph<(), ()>, [NodeId; 4]) {
    let mut graph = Graph::<(), ()>::new();
    let [a, b, c, d] = [(); 4].map(|()| graph.add_node(()));
    graph.add_edge(a, b, ());
    graph.add_edge(b, a, ());
    graph.add_edge(d, a, ());
    assert!(graph.remove_node(c));
    (graph, [a, b, c, d])
}

#[test]
fn both_numberings_agree_on_a_graph_with_a_removed_slot() {
    let (graph, [a, b, _, d]) = with_removed_slot();
    let expected = BTreeSet::from([BTreeSet::from([a, b]), BTreeSet::from([d])]);

    let leaves_first = SccDecomposition::compute(&graph, SccNumbering::LeavesFirst);
    assert_eq!(partition(&leaves_first), expected);
    let sources_first = SccDecomposition::compute(&graph, SccNumbering::SourcesFirst);
    assert_eq!(partition(&sources_first), expected);
    assert_eq!(leaves_first.condensation(&graph).node_count(), 2);
}

#[test]
#[should_panic(expected = "is not a node of the decomposed graph")]
fn leaves_first_refuses_to_place_a_removed_slot_in_a_component() {
    let (graph, [_, _, removed, _]) = with_removed_slot();
    let _ = SccDecomposition::compute(&graph, SccNumbering::LeavesFirst).component_index(removed);
}

#[test]
#[should_panic(expected = "is not a node of the decomposed graph")]
fn sources_first_refuses_to_place_a_removed_slot_in_a_component() {
    let (graph, [_, _, removed, _]) = with_removed_slot();
    let _ = SccDecomposition::compute(&graph, SccNumbering::SourcesFirst).component(removed);
}

#[test]
fn a_cfg_with_a_removed_block_decomposes_like_its_compacted_form() {
    let mut cfg = Cfg::<u32>::new();
    let entry = cfg.entry();
    let head = cfg.new_block();
    let body = cfg.new_block();
    let dead = cfg.new_block();
    let exit = cfg.new_block();
    cfg.add_edge(entry, head, EdgeKind::Fallthrough);
    cfg.add_edge(head, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, head, EdgeKind::Back);
    cfg.add_edge(head, exit, EdgeKind::ConditionalFalse);
    assert!(cfg.remove_block(dead));

    for result in [
        SccDecomposition::compute(&cfg, SccNumbering::LeavesFirst),
        SccDecomposition::compute(&cfg, SccNumbering::SourcesFirst),
    ] {
        assert_eq!(result.component_count(), 3);
        assert_eq!(result.component(head).nodes(), [head, body]);
        assert!(!result.is_dag(&cfg));
    }

    let renumbering = cfg.compact(NodeOrder::Insertion);
    let compacted = SccDecomposition::compute(&cfg, SccNumbering::LeavesFirst);
    assert_eq!(compacted.component_count(), 3);
    let head = renumbering.node(head).expect("head survives");
    let body = renumbering.node(body).expect("body survives");
    assert_eq!(compacted.component(head).nodes(), [head, body]);
}

#[test]
fn a_component_reports_whether_it_holds_a_cycle() {
    // entry -> (a <-> b), and a block that branches back to itself.
    let mut graph = Graph::<&str, ()>::new();
    let entry = graph.add_node("entry");
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let spin = graph.add_node("spin");
    graph.add_edge(entry, a, ());
    graph.add_edge(a, b, ());
    graph.add_edge(b, a, ());
    graph.add_edge(entry, spin, ());
    graph.add_edge(spin, spin, ());

    let components = SccDecomposition::compute(&graph, SccNumbering::LeavesFirst);

    assert!(!components.component(entry).is_cyclic(&graph));
    assert!(components.component(a).is_cyclic(&graph));
    assert!(components.component(spin).is_cyclic(&graph));
    assert!(!components.is_dag(&graph));
}
