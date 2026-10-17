//! `compact`, `Renumbering` and appending after a compaction.

use cfglib::{EdgeId, Graph, NodeId};

use crate::{Relation, Symbol, relation, symbol};

#[test]
fn compaction_renumbers_every_survivor_densely_and_reports_the_removed_as_none() {
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..6)
        .map(|index| graph.add_node(symbol(&format!("n{index}"))))
        .collect();
    let edge_endpoints = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (2, 2),
        (3, 5),
    ];
    let edges: Vec<EdgeId> = edge_endpoints
        .iter()
        .map(|&(source, target)| {
            graph.add_edge(
                nodes[source],
                nodes[target],
                relation(&format!("e{source}{target}")),
            )
        })
        .collect();
    graph.remove_node(nodes[1]);
    graph.remove_node(nodes[4]);
    graph.remove_edge(edges[7]);

    let live_nodes = [0_usize, 2, 3, 5];
    let live_edges = [2_usize, 5, 6];
    let renumbering = graph.compact();

    assert_eq!(renumbering.node_bound(), 6);
    assert_eq!(renumbering.edge_bound(), 8);
    for (old, &node) in nodes.iter().enumerate() {
        let position = live_nodes.iter().position(|&live| live == old);
        let expected = position.map(NodeId::from_index);
        assert_eq!(renumbering.node(node), expected, "node n{old}");
        if let Some(new) = expected {
            assert_eq!(graph.node(new).name, format!("n{old}"));
        }
    }
    for (old, &edge) in edges.iter().enumerate() {
        let position = live_edges.iter().position(|&live| live == old);
        let expected = position.map(EdgeId::from_index);
        assert_eq!(renumbering.edge(edge), expected, "edge slot {old}");
        if let Some(new) = expected {
            let (source, target) = edge_endpoints[old];
            let record = graph.edge(new);
            assert_eq!(record.payload().label, format!("e{source}{target}"));
            assert_eq!(Some(record.source()), renumbering.node(nodes[source]));
            assert_eq!(Some(record.target()), renumbering.node(nodes[target]));
        }
    }
    assert_eq!((graph.node_count(), graph.node_bound()), (4, 4));
    assert_eq!((graph.edge_count(), graph.edge_bound()), (3, 3));
    assert_eq!(
        renumbering.node(NodeId::from_index(6)),
        None,
        "minted after the compaction"
    );
    assert_eq!(renumbering.edge(EdgeId::from_index(8)), None);
}

#[test]
fn a_compaction_with_nothing_to_drop_is_the_identity_and_a_removal_ahead_of_a_survivor_is_not() {
    let mut untouched = Graph::<Symbol, Relation>::new();
    let a = untouched.add_node(symbol("a"));
    let b = untouched.add_node(symbol("b"));
    untouched.add_edge(a, b, relation("ab"));
    let renumbering = untouched.compact();
    assert!(renumbering.is_identity(), "nothing to drop");
    assert_eq!(
        (renumbering.node(a), renumbering.node(b)),
        (Some(a), Some(b))
    );

    let mut leading = Graph::<Symbol, Relation>::new();
    let a = leading.add_node(symbol("a"));
    let b = leading.add_node(symbol("b"));
    leading.remove_node(a);
    let renumbering = leading.compact();
    assert!(!renumbering.is_identity());
    assert_eq!(renumbering.node(b), Some(NodeId::from_index(0)));
}

#[test]
fn a_trailing_removal_leaves_every_survivor_where_it_was() {
    let mut graph = Graph::<Symbol, Relation>::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    graph.add_edge(a, b, relation("ab"));
    graph.remove_node(b);

    let renumbering = graph.compact();
    assert_eq!(renumbering.node(a), Some(a), "the survivor kept its number");
    assert_eq!(
        renumbering.node(b),
        None,
        "the removed node is reported as removed"
    );
    assert_eq!(renumbering.edge(EdgeId::from_index(0)), None);
    assert_eq!(graph.node(a).name, "a");
}

#[test]
fn appending_after_a_compaction_extends_the_dense_numbering() {
    let mut graph = Graph::new();
    let definition = graph.add_node(symbol("definition"));
    let call = graph.add_node(symbol("call"));
    let returns = graph.add_edge(definition, call, relation("returns"));
    graph.compact();
    assert!(graph.is_compact());

    let other = graph.add_node(symbol("other"));
    let escapes = graph.add_edge(definition, other, relation("escapes"));
    assert_eq!(
        other.index(),
        2,
        "numbering continues after the compacted nodes"
    );
    assert_eq!(escapes.index(), 1);
    assert!(!graph.is_compact(), "the delta is not a compact base");
    assert_eq!(
        graph.successors(definition).collect::<Vec<_>>(),
        [call, other]
    );
    assert_eq!(
        graph.outgoing(definition).collect::<Vec<_>>(),
        [returns, escapes]
    );
    assert_eq!(
        (graph.out_degree(definition), graph.in_degree(other)),
        (2, 1)
    );

    let renumbering = graph.compact();
    assert!(renumbering.is_identity(), "appending alone moves nothing");
    assert_eq!((renumbering.node_bound(), renumbering.edge_bound()), (3, 2));
    assert!(graph.is_compact());
    assert_eq!(
        graph.outgoing(definition).collect::<Vec<_>>(),
        [returns, escapes]
    );
    assert_eq!(graph.node(other).name, "other");
}

#[test]
fn an_appended_edge_may_attach_to_a_node_of_the_compacted_base() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    let base = graph.add_edge(a, b, relation("base"));
    graph.compact();

    let delta = graph.add_edge(b, a, relation("delta"));
    let repeat = graph.add_edge(a, b, relation("repeat"));

    assert_eq!(graph.outgoing(a).collect::<Vec<_>>(), [base, repeat]);
    assert_eq!(graph.incoming(a).collect::<Vec<_>>(), [delta]);
    assert_eq!(graph.incoming(b).collect::<Vec<_>>(), [base, repeat]);
    graph.remove_edge(base);
    assert_eq!(graph.outgoing(a).collect::<Vec<_>>(), [repeat]);
    assert_eq!((graph.out_degree(a), graph.in_degree(b)), (1, 1));
}

#[test]
fn renumberings_compose_across_a_skipped_compaction() {
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..6)
        .map(|index| graph.add_node(symbol(&format!("n{index}"))))
        .collect();
    let edges: Vec<EdgeId> = (0..5)
        .map(|index| {
            graph.add_edge(
                nodes[index],
                nodes[index + 1],
                relation(&format!("e{index}")),
            )
        })
        .collect();

    graph.remove_node(nodes[0]);
    let first = graph.compact();
    let late = graph.add_node(symbol("late"));
    let middle = first
        .node(nodes[2])
        .expect("n2 survived the first compaction");
    graph.remove_node(middle);
    let second = graph.compact();
    let composed = first.compose(&second);

    assert_eq!(composed.node_bound(), first.node_bound());
    assert_eq!(composed.edge_bound(), first.edge_bound());
    for (old, &node) in nodes.iter().enumerate() {
        let stepwise = first
            .node(node)
            .and_then(|intermediate| second.node(intermediate));
        assert_eq!(composed.node(node), stepwise, "node n{old}");
    }
    for (old, &edge) in edges.iter().enumerate() {
        let stepwise = first
            .edge(edge)
            .and_then(|intermediate| second.edge(intermediate));
        assert_eq!(composed.edge(edge), stepwise, "edge e{old}");
    }
    let survivors: Vec<_> = nodes
        .iter()
        .filter_map(|&node| composed.node(node))
        .map(|node| graph.node(node).name.as_str())
        .collect();
    assert_eq!(survivors, ["n1", "n3", "n4", "n5"]);
    assert_eq!(composed.node(nodes[0]), None);
    assert_eq!(composed.node(nodes[2]), None);
    let late = second
        .node(late)
        .expect("late survived the second compaction");
    assert_eq!(graph.node(late).name, "late");
}

#[test]
fn compacted_consumes_the_store_and_hands_back_the_numbering() {
    let mut graph = Graph::new();
    let gone = graph.add_node(symbol("gone"));
    let kept = graph.add_node(symbol("kept"));
    graph.add_edge(gone, kept, relation("dropped"));
    graph.remove_node(gone);

    let (compacted, renumbering) = graph.compacted();
    assert!(compacted.is_compact());
    assert_eq!(compacted.edge_count(), 0);
    let kept = renumbering.node(kept).expect("kept survived");
    assert_eq!(compacted.node(kept).name, "kept");
}
