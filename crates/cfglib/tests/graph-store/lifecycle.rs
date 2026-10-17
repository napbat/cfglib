//! Add, remove, redirect, compact and append: the life of one identity in a `Graph`.

use cfglib::{EdgeId, EdgeView, Graph, GraphView, Id, IdTag, NodeId, Renumbering};

use crate::{Relation, Symbol, relation, symbol};

fn names(graph: &Graph<Symbol, Relation>, nodes: impl IntoIterator<Item = NodeId>) -> Vec<String> {
    nodes
        .into_iter()
        .map(|node| graph.node(node).name.clone())
        .collect()
}

#[test]
fn identities_are_dense_insertion_slots_with_a_readable_prefix() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    let ab: EdgeId = graph.add_edge(a, b, relation("ab"));

    assert_eq!((a.index(), b.index(), ab.index()), (0, 1, 0));
    assert_eq!((a.raw(), b.raw()), (0, 1));
    assert_eq!(NodeId::from_index(1), b);
    assert_eq!(NodeId::from_raw(0), a);
    assert!(a < b, "identities order by slot");
    assert_eq!(format!("{a} {b} {ab}"), "n0 n1 e0");
    assert_eq!(format!("{b:?}"), "n1");
}

#[test]
fn payloads_are_borrowed_and_edited_in_place() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    let ab = graph.add_edge(a, b, relation("ab"));

    assert_eq!(graph[a], symbol("a"));
    graph.node_mut(a).file = 9;
    graph.edge_mut(ab).payload_mut().sites.push(4);

    assert_eq!(graph.node(a).file, 9);
    let record = graph.edge(ab);
    assert_eq!((record.source(), record.target()), (a, b));
    assert_eq!(record.payload().sites, [1, 2, 3, 4]);

    let owned = graph.edge(ab).clone().into_payload();
    assert_eq!(owned.label, "ab");
    assert_eq!(
        graph.edge(ab).payload().label,
        "ab",
        "the store keeps its own"
    );
}

#[test]
fn nodes_and_edges_iterate_live_payloads_in_identity_order() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    let c = graph.add_node(symbol("c"));
    graph.add_edge(a, b, relation("ab"));
    let bc = graph.add_edge(b, c, relation("bc"));
    graph.add_edge(c, a, relation("ca"));
    graph.remove_edge(bc);
    graph.remove_node(b);

    let nodes: Vec<_> = graph.nodes().map(|node| node.name.as_str()).collect();
    assert_eq!(nodes, ["a", "c"]);
    let edges: Vec<_> = graph
        .edges()
        .map(|edge| edge.payload().label.as_str())
        .collect();
    assert_eq!(edges, ["ca"]);
    assert_eq!(
        graph.edge_ids().collect::<Vec<_>>(),
        [EdgeId::from_index(2)]
    );
}

#[test]
fn an_empty_store_is_compact_and_compacts_to_the_identity() {
    let mut graph = Graph::<Symbol, Relation>::default();
    assert!(graph.is_empty());
    assert!(graph.is_compact());
    assert_eq!((graph.node_count(), graph.node_bound()), (0, 0));
    assert_eq!((graph.edge_count(), graph.edge_bound()), (0, 0));
    assert_eq!(graph.node_ids().count(), 0);

    let renumbering = graph.compact();
    assert!(renumbering.is_identity());
    assert_eq!((renumbering.node_bound(), renumbering.edge_bound()), (0, 0));
    assert_eq!(renumbering.node(NodeId::from_index(0)), None);
}

#[test]
fn removing_a_node_removes_its_edges_and_takes_it_out_of_the_view() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    let c = graph.add_node(symbol("c"));
    let ab = graph.add_edge(a, b, relation("ab"));
    let bc = graph.add_edge(b, c, relation("bc"));
    let ca = graph.add_edge(c, a, relation("ca"));

    assert!(graph.remove_node(b));
    assert!(!graph.remove_node(b), "a second removal has nothing to do");

    assert!(!graph.contains_node(b));
    assert!(!graph.contains_edge(ab) && !graph.contains_edge(bc));
    assert!(graph.contains_edge(ca));
    assert_eq!((graph.node_count(), graph.node_bound()), (2, 3));
    assert_eq!((graph.edge_count(), graph.edge_bound()), (1, 3));
    assert_eq!(graph.node_ids().collect::<Vec<_>>(), [a, c]);
    assert_eq!(GraphView::node_ids(&graph).count(), 2);

    assert_eq!(graph.node(b).name, "b", "a removed payload stays readable");
    assert_eq!(graph.edge(ab).payload().label, "ab");
    assert_eq!(graph.successors(a).count(), 0, "the edge into b is gone");
    assert_eq!(
        graph.predecessors(c).count(),
        0,
        "the edge out of b is gone"
    );
    assert_eq!(graph.successors(c).collect::<Vec<_>>(), [a]);
    assert_eq!(
        graph.successors(b).count(),
        0,
        "a removed node has no edges"
    );
    assert_eq!(graph.predecessors(b).count(), 0);
}

#[test]
fn removing_every_node_leaves_an_empty_store_that_still_has_slots() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    graph.add_edge(a, b, relation("ab"));
    graph.add_edge(b, b, relation("loop"));
    graph.compact();
    assert!(graph.is_compact());
    graph.remove_node(a);
    graph.remove_node(b);

    assert!(graph.is_empty());
    assert_eq!((graph.node_count(), graph.edge_count()), (0, 0));
    assert_eq!((graph.node_bound(), graph.edge_bound()), (2, 2));
    assert!(!graph.is_compact(), "a compact store has no tombstones");

    let renumbering = graph.compact();
    assert!(graph.is_empty() && graph.is_compact());
    assert_eq!((graph.node_bound(), graph.edge_bound()), (0, 0));
    assert_eq!(renumbering.node(a), None);
    assert_eq!(renumbering.node(b), None);
}

#[test]
fn removed_slots_are_not_reused_before_a_compaction() {
    let mut graph = Graph::<Symbol, Relation>::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    graph.remove_node(b);
    let late = graph.add_node(symbol("late"));

    assert_eq!(late.index(), 2, "the slot of `b` stays reserved");
    assert_ne!(late, b);
    assert!(graph.contains_node(a) && graph.contains_node(late) && !graph.contains_node(b));
    assert_eq!(graph.node_ids().collect::<Vec<_>>(), [a, late]);
}

#[test]
fn removing_an_edge_keeps_the_other_parallel_edges_in_order() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    let first = graph.add_edge(a, b, relation("first"));
    let middle = graph.add_edge(a, b, relation("middle"));
    let last = graph.add_edge(a, b, relation("last"));

    assert!(graph.remove_edge(middle));
    assert!(
        !graph.remove_edge(middle),
        "a second removal has nothing to do"
    );

    assert_eq!(graph.outgoing(a).collect::<Vec<_>>(), [first, last]);
    assert_eq!(graph.incoming(b).collect::<Vec<_>>(), [first, last]);
    assert_eq!(graph.successors(a).collect::<Vec<_>>(), [b, b]);
    assert_eq!((graph.out_degree(a), graph.in_degree(b)), (2, 2));
    assert_eq!((graph.edge_count(), graph.edge_bound()), (2, 3));
    assert!(!graph.contains_edge(middle));
    assert_eq!(graph.edge(middle).payload().label, "middle");
    assert_eq!(
        graph.node_count(),
        2,
        "an edge removal keeps both endpoints"
    );
}

#[test]
fn parallel_edges_and_self_loops_keep_their_own_identities() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    let own = graph.add_edge(a, a, relation("own"));
    let first = graph.add_edge(a, b, relation("first"));
    let second = graph.add_edge(a, b, relation("second"));

    assert_eq!(graph.outgoing(a).collect::<Vec<_>>(), [own, first, second]);
    assert_eq!(graph.incoming(a).collect::<Vec<_>>(), [own]);
    assert_eq!(graph.incoming(b).collect::<Vec<_>>(), [first, second]);
    assert_eq!(graph.successors(a).collect::<Vec<_>>(), [a, b, b]);
    assert_eq!(graph.predecessors(a).collect::<Vec<_>>(), [a]);
    assert_eq!((graph.out_degree(a), graph.in_degree(a)), (3, 1));
    assert_eq!((graph.out_degree(b), graph.in_degree(b)), (0, 2));

    let renumbering = graph.compact();
    assert!(renumbering.is_identity());
    assert_eq!(graph.outgoing(a).collect::<Vec<_>>(), [own, first, second]);
    assert_eq!(graph.predecessors(b).collect::<Vec<_>>(), [a, a]);
    assert_eq!((graph.out_degree(a), graph.in_degree(a)), (3, 1));
}

#[test]
fn a_removed_self_loop_leaves_both_of_its_adjacency_lists() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let own = graph.add_edge(a, a, relation("own"));
    graph.remove_edge(own);

    assert_eq!((graph.out_degree(a), graph.in_degree(a)), (0, 0));
    assert!(graph.successors(a).next().is_none());
    assert!(graph.predecessors(a).next().is_none());
}

#[test]
fn redirecting_a_target_keeps_the_edge_identity_payload_and_source() {
    let mut graph = Graph::new();
    let source = graph.add_node(symbol("source"));
    let first = graph.add_node(symbol("first"));
    let second = graph.add_node(symbol("second"));
    let call = graph.add_edge(source, first, relation("call"));

    assert_eq!(graph.redirect_edge_target(call, second), first);

    assert_eq!(graph.successors(source).collect::<Vec<_>>(), [second]);
    assert!(graph.incoming(first).next().is_none());
    assert_eq!(graph.incoming(second).collect::<Vec<_>>(), [call]);
    let record = graph.edge(call);
    assert_eq!((record.source(), record.target()), (source, second));
    assert_eq!(
        record.payload(),
        &relation("call"),
        "the edge is the same edge"
    );
    assert_eq!((graph.edge_count(), graph.edge_bound()), (1, 1));
}

#[test]
fn redirecting_to_the_current_endpoint_changes_nothing() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    let first = graph.add_edge(a, b, relation("first"));
    let second = graph.add_edge(a, b, relation("second"));

    assert_eq!(graph.redirect_edge_target(first, b), b);
    assert_eq!(graph.redirect_edge_source(second, a), a);

    assert_eq!(graph.outgoing(a).collect::<Vec<_>>(), [first, second]);
    assert_eq!(graph.incoming(b).collect::<Vec<_>>(), [first, second]);
}

#[test]
fn redirecting_a_source_appends_the_edge_after_the_ones_already_at_the_new_source() {
    let mut graph = Graph::new();
    let p = graph.add_node(symbol("p"));
    let q = graph.add_node(symbol("q"));
    let r = graph.add_node(symbol("r"));
    let moved = graph.add_edge(p, r, relation("moved"));
    let first = graph.add_edge(q, r, relation("first"));
    let second = graph.add_edge(q, p, relation("second"));

    assert_eq!(graph.redirect_edge_source(moved, q), p);

    assert_eq!(
        graph.outgoing(q).collect::<Vec<_>>(),
        [first, second, moved]
    );
    assert_eq!(graph.successors(q).collect::<Vec<_>>(), [r, p, r]);
    assert!(graph.outgoing(p).next().is_none());
    assert_eq!(graph.incoming(r).collect::<Vec<_>>(), [moved, first]);
    assert_eq!(graph.incoming(p).collect::<Vec<_>>(), [second]);
    assert_eq!(graph.edge(moved).source(), q);
    assert_eq!(graph.edge(moved).payload().label, "moved");

    let after = graph.add_edge(q, p, relation("after"));
    assert_eq!(
        graph.outgoing(q).collect::<Vec<_>>(),
        [first, second, moved, after],
        "an edge appended after the move follows it"
    );
}

#[test]
fn removal_then_redirect_then_compact_leaves_every_survivor_addressable() {
    let mut graph = Graph::new();
    let hub = graph.add_node(symbol("hub"));
    let gone = graph.add_node(symbol("gone"));
    let sink = graph.add_node(symbol("sink"));
    let other = graph.add_node(symbol("other"));
    let to_gone = graph.add_edge(hub, gone, relation("to_gone"));
    let to_sink = graph.add_edge(hub, sink, relation("to_sink"));
    let from_other = graph.add_edge(other, sink, relation("from_other"));

    assert!(graph.remove_node(gone));
    assert!(!graph.contains_edge(to_gone));
    assert_eq!(graph.redirect_edge_target(to_sink, other), sink);
    assert_eq!(graph.predecessors(sink).collect::<Vec<_>>(), [other]);

    let renumbering = graph.compact();
    assert!(graph.is_compact());
    assert_eq!((graph.node_count(), graph.edge_count()), (3, 2));

    assert_eq!(renumbering.node(gone), None);
    assert_eq!(renumbering.edge(to_gone), None);
    let hub = renumbering.node(hub).expect("hub survived");
    let sink = renumbering.node(sink).expect("sink survived");
    let other = renumbering.node(other).expect("other survived");
    assert_eq!(names(&graph, [hub, sink, other]), ["hub", "sink", "other"]);
    assert_eq!((hub.index(), sink.index(), other.index()), (0, 1, 2));

    let to_sink = renumbering
        .edge(to_sink)
        .expect("the redirected edge survived");
    let from_other = renumbering.edge(from_other).expect("from_other survived");
    assert_eq!(graph.edge(to_sink).payload().label, "to_sink");
    assert_eq!(
        (graph.edge(to_sink).source(), graph.edge(to_sink).target()),
        (hub, other)
    );
    assert_eq!(graph.edge(from_other).payload().label, "from_other");
    assert_eq!(
        (
            graph.edge(from_other).source(),
            graph.edge(from_other).target()
        ),
        (other, sink)
    );
    assert_eq!(graph.successors(hub).collect::<Vec<_>>(), [other]);
    assert_eq!(graph.predecessors(sink).collect::<Vec<_>>(), [other]);
}

#[test]
#[should_panic(expected = "has been removed")]
fn redirecting_onto_a_removed_node_panics() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    let gone = graph.add_node(symbol("gone"));
    let edge = graph.add_edge(a, b, relation("ab"));
    graph.remove_node(gone);
    graph.redirect_edge_target(edge, gone);
}

#[test]
#[should_panic(expected = "has been removed")]
fn redirecting_a_removed_edge_panics() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    let edge = graph.add_edge(a, b, relation("ab"));
    graph.remove_edge(edge);
    graph.redirect_edge_source(edge, b);
}

#[test]
#[should_panic(expected = "has been removed")]
fn adding_an_edge_to_a_removed_node_panics() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let gone = graph.add_node(symbol("gone"));
    graph.remove_node(gone);
    graph.add_edge(a, gone, relation("late"));
}

#[test]
#[should_panic(expected = "out of range")]
fn adding_an_edge_to_a_node_the_store_never_minted_panics() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    graph.add_edge(a, NodeId::from_index(5), relation("stray"));
}

#[test]
fn a_clone_is_independent_of_the_store_it_came_from() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    graph.add_edge(a, b, relation("ab"));
    let snapshot = graph.clone();
    assert_eq!(snapshot, graph);

    graph.remove_node(b);
    graph.node_mut(a).file = 99;

    assert_eq!(snapshot.node_count(), 2);
    assert_eq!(snapshot.node(a).file, 7);
    assert_eq!(snapshot.successors(a).collect::<Vec<_>>(), [b]);
    assert_ne!(snapshot, graph);
}

#[test]
fn adjacent_edges_can_be_cloned_mid_walk_and_resumed_independently() {
    let mut graph = Graph::new();
    let hub = graph.add_node(symbol("hub"));
    let spokes: Vec<EdgeId> = (0..4)
        .map(|index| {
            let node = graph.add_node(symbol(&format!("s{index}")));
            graph.add_edge(hub, node, relation(&format!("s{index}")))
        })
        .collect();

    let mut walk = graph.outgoing(hub);
    assert_eq!(walk.next(), Some(spokes[0]));
    let resumed = walk.clone();
    assert_eq!(walk.collect::<Vec<_>>(), spokes[1..]);
    assert_eq!(resumed.collect::<Vec<_>>(), spokes[1..]);
}

fn endpoint_pairs<G: EdgeView>(view: &G, node: G::NodeId) -> Vec<(G::NodeId, G::NodeId)> {
    view.outgoing(node)
        .map(|edge| {
            let edge = view.edge(edge);
            (edge.source(), edge.target())
        })
        .collect()
}

#[test]
fn the_store_serves_the_generic_view_traits() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    let c = graph.add_node(symbol("c"));
    graph.add_edge(a, b, relation("ab"));
    graph.add_edge(a, c, relation("ac"));
    let gone = graph.add_edge(b, c, relation("bc"));
    graph.remove_edge(gone);

    assert_eq!(endpoint_pairs(&graph, a), [(a, b), (a, c)]);
    assert_eq!(endpoint_pairs(&graph, b).len(), 0);
    assert_eq!(EdgeView::edge_bound(&graph), 3);
    assert_eq!(EdgeView::edge_ids(&graph).count(), 2);
    assert_eq!(GraphView::node_bound(&graph), 3);
}

#[test]
#[should_panic(expected = "has been removed")]
fn the_edge_view_refuses_a_removed_edge_that_the_inherent_accessor_still_reads() {
    let mut graph = Graph::new();
    let a = graph.add_node(symbol("a"));
    let b = graph.add_node(symbol("b"));
    let edge = graph.add_edge(a, b, relation("ab"));
    graph.remove_edge(edge);
    assert_eq!(graph.edge(edge).payload().label, "ab");
    let _ = EdgeView::edge(&graph, edge);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct SymbolTag;

impl IdTag for SymbolTag {
    const PREFIX: &'static str = "sym";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ReferenceTag;

impl IdTag for ReferenceTag {
    const PREFIX: &'static str = "ref";
}

#[test]
fn domain_tagged_identities_stay_distinct_through_compaction() {
    type IndexGraph = Graph<Symbol, Relation, SymbolTag, ReferenceTag>;
    let mut graph = IndexGraph::tagged_with_capacity(4, 4);
    let first: Id<SymbolTag> = graph.add_node(symbol("first"));
    let second = graph.add_node(symbol("second"));
    let third = graph.add_node(symbol("third"));
    let used: Id<ReferenceTag> = graph.add_edge(second, third, relation("used"));
    graph.add_edge(first, second, relation("dropped"));
    graph.remove_node(first);

    assert_eq!(format!("{first} {used}"), "sym0 ref0");
    let renumbering: Renumbering<SymbolTag, ReferenceTag> = graph.compact();
    assert_eq!(renumbering.node(first), None);
    let second = renumbering.node(second).expect("second survived");
    let third = renumbering.node(third).expect("third survived");
    assert_eq!(format!("{second} {third}"), "sym0 sym1");
    assert_eq!(
        renumbering.edge(used).map(|edge| edge.to_string()),
        Some("ref0".to_owned())
    );
    assert_eq!(graph.successors(second).collect::<Vec<_>>(), [third]);
    assert_eq!(graph.node(third).name, "third");

    let sized = Graph::<Symbol, Relation>::with_capacity(8, 8);
    assert!(sized.is_empty() && sized.is_compact());
}
