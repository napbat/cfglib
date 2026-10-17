//! Behavior tests for the incrementally compacted store.

extern crate alloc;

use alloc::vec::Vec;

use crate::graph::dominator::DominatorTree;
use crate::graph::edge_view::EdgeView;
use crate::graph::scc::tarjan_scc;
use crate::graph::store::Graph;
use crate::graph::traverse::{TraversalDirection, depth_first_preorder};
use crate::graph::view::{GraphView, Rooted};

use super::{Id, NodeId};

mod model;
#[cfg(feature = "serde")]
mod serde_format;

/// Four nodes in a diamond, built and left uncompacted.
fn diamond() -> (Graph<&'static str, &'static str>, Vec<NodeId>) {
    let mut graph = Graph::new();
    let nodes: Vec<_> = ["entry", "left", "right", "exit"]
        .into_iter()
        .map(|payload| graph.add_node(payload))
        .collect();
    graph.add_edge(nodes[0], nodes[1], "a");
    graph.add_edge(nodes[0], nodes[2], "b");
    graph.add_edge(nodes[1], nodes[3], "c");
    graph.add_edge(nodes[2], nodes[3], "d");
    (graph, nodes)
}

/// The same shape in the arena store, for view comparisons.
fn arena_diamond() -> (Graph<&'static str, &'static str>, Vec<crate::NodeId>) {
    let mut graph = Graph::new();
    let nodes: Vec<_> = ["entry", "left", "right", "exit"]
        .into_iter()
        .map(|payload| graph.add_node(payload))
        .collect();
    graph.add_edge(nodes[0], nodes[1], "a");
    graph.add_edge(nodes[0], nodes[2], "b");
    graph.add_edge(nodes[1], nodes[3], "c");
    graph.add_edge(nodes[2], nodes[3], "d");
    (graph, nodes)
}

fn indexes<T: Copy, I: IntoIterator<Item = T>>(
    values: I,
    index: impl Fn(T) -> usize,
) -> Vec<usize> {
    values.into_iter().map(index).collect()
}

#[test]
fn adjacency_is_maintained_in_both_directions() {
    let (graph, nodes) = diamond();
    assert_eq!(graph.node_count(), 4);
    assert_eq!(graph.edge_count(), 4);
    assert_eq!(
        graph.successors(nodes[0]).collect::<Vec<_>>(),
        [nodes[1], nodes[2]]
    );
    assert_eq!(
        graph.predecessors(nodes[3]).collect::<Vec<_>>(),
        [nodes[1], nodes[2]]
    );
    assert!(graph.predecessors(nodes[0]).next().is_none());
}

#[test]
fn an_empty_store_has_nothing_and_is_already_compact() {
    let graph = Graph::<(), ()>::new();
    assert!(graph.is_empty());
    assert!(graph.is_compact());
    assert_eq!(graph.node_count(), 0);
    assert_eq!(graph.edge_count(), 0);
    assert_eq!(graph.node_bound(), 0);
    assert_eq!(graph.node_ids().count(), 0);
    assert_eq!(graph.edge_ids().count(), 0);

    let reserved = Graph::<u8, u8>::with_capacity(16, 64);
    assert!(reserved.is_empty());
    assert_eq!(reserved, Graph::default());
}

#[test]
fn insertion_order_survives_the_base_delta_boundary() {
    let (mut graph, nodes) = diamond();
    graph.compact();
    assert!(graph.is_compact());

    // Both endpoints are in the compressed base, so this edge can only be
    // reached through the delta chain hanging off the base run.
    let appended = graph.add_node("appended");
    graph.add_edge(nodes[0], nodes[3], "e");
    graph.add_edge(nodes[0], appended, "f");
    assert!(!graph.is_compact());

    let expected = ["a", "b", "e", "f"];
    let observed: Vec<_> = graph
        .outgoing(nodes[0])
        .map(|edge| *graph.edge(edge).payload())
        .collect();
    assert_eq!(observed, expected);

    // Compaction rebuilds the base from that same order.
    graph.compact();
    let after: Vec<_> = graph
        .outgoing(nodes[0])
        .map(|edge| *graph.edge(edge).payload())
        .collect();
    assert_eq!(after, expected);
    assert_eq!(
        graph
            .incoming(nodes[3])
            .map(|edge| *graph.edge(edge).payload())
            .collect::<Vec<_>>(),
        ["c", "d", "e"]
    );
}

/// The delta charges nothing per node of a store that is entirely base: a
/// compacted graph, and a graph with nodes but no edges, hold no chain slots
/// at all.
#[test]
fn the_delta_charges_nothing_per_node_of_a_compacted_store() {
    let mut isolated = Graph::<&'static str, &'static str>::new();
    isolated.add_node("no edges");
    assert_eq!(isolated.chains.node_slots(), 0);

    let (mut graph, nodes) = diamond();
    assert!(graph.chains.node_slots() > 0);

    graph.compact();
    assert_eq!(graph.chains.node_slots(), 0);
    assert_eq!(
        graph.successors(nodes[0]).collect::<Vec<_>>(),
        [nodes[1], nodes[2]],
        "the base still answers for every node the delta released"
    );

    // One append re-materializes the slots, and the store is back to
    // charging for them until the next compaction.
    graph.add_edge(nodes[1], nodes[2], "e");
    assert_eq!(graph.chains.node_slots(), graph.node_bound());
    assert_eq!(
        graph.successors(nodes[1]).collect::<Vec<_>>(),
        [nodes[3], nodes[2]]
    );
    assert_eq!(
        graph.predecessors(nodes[2]).collect::<Vec<_>>(),
        [nodes[0], nodes[1]]
    );

    graph.compact();
    assert_eq!(graph.chains.node_slots(), 0);
}

#[test]
fn parallel_and_self_edges_are_retained_in_order() {
    let mut graph = Graph::new();
    let node = graph.add_node("only");
    let other = graph.add_node("other");
    let first = graph.add_edge(node, other, 1);
    let loop_edge = graph.add_edge(node, node, 2);
    let parallel = graph.add_edge(node, other, 3);

    assert_eq!(
        graph.outgoing(node).collect::<Vec<_>>(),
        [first, loop_edge, parallel]
    );
    assert_eq!(graph.incoming(node).collect::<Vec<_>>(), [loop_edge]);
    assert_eq!(
        graph.successors(node).collect::<Vec<_>>(),
        [other, node, other]
    );

    graph.compact();
    assert_eq!(
        graph.outgoing(node).collect::<Vec<_>>(),
        [first, loop_edge, parallel]
    );
    assert_eq!(graph.incoming(node).collect::<Vec<_>>(), [loop_edge]);
}

#[test]
fn removing_a_node_removes_its_edges_in_both_directions() {
    let (mut graph, nodes) = diamond();
    assert!(graph.remove_node(nodes[1]));
    assert!(!graph.remove_node(nodes[1]));

    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.edge_count(), 2);
    assert_eq!(graph.successors(nodes[0]).collect::<Vec<_>>(), [nodes[2]]);
    assert_eq!(graph.predecessors(nodes[3]).collect::<Vec<_>>(), [nodes[2]]);
    assert!(graph.outgoing(nodes[1]).next().is_none());
    assert!(graph.incoming(nodes[1]).next().is_none());
    assert!(!graph.contains_node(nodes[1]));
    assert!(graph.edge_ids().all(|edge| graph.contains_edge(edge)));
}

#[test]
fn removing_an_edge_leaves_every_other_identity_alone() {
    let (mut graph, nodes) = diamond();
    let edges: Vec<_> = graph.edge_ids().collect();
    assert!(graph.remove_edge(edges[0]));
    assert!(!graph.remove_edge(edges[0]));
    assert!(!graph.remove_edge(Id::from_index(99)));

    assert_eq!(graph.edge_count(), 3);
    assert_eq!(graph.edge_bound(), 4);
    assert_eq!(graph.successors(nodes[0]).collect::<Vec<_>>(), [nodes[2]]);
    assert_eq!(graph.edge_ids().collect::<Vec<_>>(), edges[1..]);
    assert_eq!(graph.edge(edges[1]).payload(), &"b");
}

#[test]
fn degrees_agree_with_the_walk_in_every_store_state() {
    let (mut graph, nodes) = diamond();
    let degrees = |graph: &Graph<&'static str, &'static str>| {
        nodes
            .iter()
            .map(|&node| (graph.out_degree(node), graph.in_degree(node)))
            .collect::<Vec<_>>()
    };
    let walked = |graph: &Graph<&'static str, &'static str>| {
        nodes
            .iter()
            .map(|&node| (graph.outgoing(node).count(), graph.incoming(node).count()))
            .collect::<Vec<_>>()
    };

    // Entirely in the delta, where every edge is one chain step.
    assert_eq!(degrees(&graph), walked(&graph));
    assert_eq!(degrees(&graph), [(2, 0), (1, 1), (1, 1), (0, 2)]);

    // Entirely in the compressed base, which is the constant-time answer.
    graph.compact();
    assert_eq!(degrees(&graph), walked(&graph));

    // A base plus a delta, and then a tombstone, which is the counted walk.
    graph.add_edge(nodes[3], nodes[0], "back");
    assert_eq!(degrees(&graph), walked(&graph));
    assert_eq!(graph.out_degree(nodes[3]), 1);
    let removed = graph.edge_ids().next().expect("the diamond has edges");
    assert!(graph.remove_edge(removed));
    assert_eq!(degrees(&graph), walked(&graph));
    assert_eq!(graph.out_degree(nodes[0]), 1);
}

#[test]
fn a_redirected_endpoint_is_counted_at_its_new_node() {
    let (mut graph, nodes) = diamond();
    graph.compact();
    let edge = graph.edge_ids().next().expect("the diamond has edges");
    graph.redirect_edge_target(edge, nodes[2]);

    for &node in &nodes {
        assert_eq!(
            (graph.out_degree(node), graph.in_degree(node)),
            (graph.outgoing(node).count(), graph.incoming(node).count()),
        );
    }
    assert_eq!(graph.in_degree(nodes[1]), 0);
    assert_eq!(graph.in_degree(nodes[2]), 2);
}

/// Every live edge is listed once by its source and once by its target, and
/// every listing agrees with the edge's own endpoints.
fn assert_mirrored<N, E>(graph: &Graph<N, E>) {
    let mut listed = Vec::new();
    for node in graph.node_ids() {
        for edge in graph.outgoing(node) {
            assert_eq!(graph.edge(edge).source(), node, "outgoing listing");
            listed.push((edge, TraversalDirection::Outgoing));
        }
        for edge in graph.incoming(node) {
            assert_eq!(graph.edge(edge).target(), node, "incoming listing");
            listed.push((edge, TraversalDirection::Incoming));
        }
    }
    for edge in graph.edge_ids() {
        for direction in [TraversalDirection::Outgoing, TraversalDirection::Incoming] {
            let count = listed
                .iter()
                .filter(|&&(candidate, side)| candidate == edge && side == direction)
                .count();
            assert_eq!(
                count, 1,
                "edge {edge:?} listed {count} time(s) {direction:?}"
            );
        }
    }
}

#[test]
fn an_edge_appended_after_a_relocation_survives_the_next_one() {
    // A relocation gives `c` an overlay run, an append then lands in its
    // delta chain, and a second relocation onto `c` must carry both.
    let mut graph = Graph::<&'static str, &'static str>::new();
    let [a, b, c, d] = ["a", "b", "c", "d"].map(|payload| graph.add_node(payload));
    let moved = graph.add_edge(a, b, "moved");
    let resident = graph.add_edge(c, d, "resident");
    graph.redirect_edge_source(moved, c);
    let appended = graph.add_edge(c, b, "appended");
    let arriving = graph.add_edge(d, a, "arriving");
    assert_eq!(
        graph.outgoing(c).collect::<Vec<_>>(),
        [resident, moved, appended]
    );

    graph.redirect_edge_source(arriving, c);

    assert_eq!(
        graph.outgoing(c).collect::<Vec<_>>(),
        [resident, moved, appended, arriving]
    );
    assert_eq!(graph.incoming(b).collect::<Vec<_>>(), [moved, appended]);
    assert_eq!(graph.out_degree(c), 4);
    assert_mirrored(&graph);
}

#[test]
fn a_chain_edge_leaves_a_node_that_already_has_an_overlay() {
    // The same shape, relocating *away* from the node whose delta chain
    // holds the edge: the survivors must keep the rest of the chain.
    let mut graph = Graph::<&'static str, &'static str>::new();
    let [a, b, c, d] = ["a", "b", "c", "d"].map(|payload| graph.add_node(payload));
    let moved = graph.add_edge(a, b, "moved");
    let resident = graph.add_edge(c, d, "resident");
    graph.redirect_edge_source(moved, c);
    let first = graph.add_edge(c, b, "first");
    let second = graph.add_edge(c, a, "second");

    graph.redirect_edge_source(first, d);

    assert_eq!(
        graph.outgoing(c).collect::<Vec<_>>(),
        [resident, moved, second]
    );
    assert_eq!(graph.outgoing(d).collect::<Vec<_>>(), [first]);
    assert_mirrored(&graph);
}

#[test]
fn a_store_carries_an_overlay_only_once_it_relocates() {
    let (mut graph, nodes) = diamond();
    graph.compact();
    assert!(!graph.has_relocations());
    let edge = graph.edge_ids().next().expect("the diamond has edges");

    graph.redirect_edge_target(edge, nodes[3]);
    assert!(graph.has_relocations());
    assert!(!graph.is_compact());

    graph.compact();
    assert!(!graph.has_relocations());
    assert!(graph.is_compact());
}

#[test]
fn nodes_added_after_the_first_relocation_relocate_too() {
    let mut graph = Graph::<&'static str, &'static str>::new();
    let [a, b] = ["a", "b"].map(|payload| graph.add_node(payload));
    let first = graph.add_edge(a, b, "first");
    let second = graph.add_edge(b, a, "second");
    graph.redirect_edge_source(first, b);

    // The overlay's index was sized for two nodes; these are past its end.
    let later: Vec<_> = (0..40).map(|_| graph.add_node("later")).collect();
    for pair in later.windows(2) {
        graph.add_edge(pair[0], pair[1], "chain");
    }
    let (head, tail) = (later[0], later[39]);
    assert!(graph.outgoing(tail).next().is_none());
    assert_mirrored(&graph);

    graph.redirect_edge_source(second, tail);
    graph.redirect_edge_target(first, tail);
    graph.move_adjacency(head, tail, TraversalDirection::Outgoing);

    assert_eq!(graph.outgoing(tail).count(), 2);
    assert_eq!(graph.incoming(tail).count(), 2);
    assert!(graph.outgoing(head).next().is_none());
    assert_eq!(graph.edge(second).source(), tail);
    assert_mirrored(&graph);
}

#[test]
fn stores_that_relocated_the_same_edges_in_either_order_are_equal() {
    let relocate = |first_moves_first: bool| {
        let mut graph = Graph::<&'static str, &'static str>::new();
        let [a, b, c, d] = ["a", "b", "c", "d"].map(|payload| graph.add_node(payload));
        let first = graph.add_edge(a, b, "first");
        let second = graph.add_edge(c, d, "second");
        let moves = [(first, d), (second, b)];
        let order = if first_moves_first {
            [moves[0], moves[1]]
        } else {
            [moves[1], moves[0]]
        };
        for (edge, source) in order {
            graph.redirect_edge_source(edge, source);
        }
        graph
    };

    let forward = relocate(true);
    let backward = relocate(false);
    assert!(forward.has_relocations());
    assert_eq!(forward, backward);
}

#[test]
fn a_bulk_move_carries_the_overlay_and_the_delta_chain_in_order() {
    let mut graph = Graph::<&'static str, &'static str>::new();
    let [a, b, c, d] = ["a", "b", "c", "d"].map(|payload| graph.add_node(payload));
    let moved = graph.add_edge(a, b, "moved");
    let resident = graph.add_edge(c, d, "resident");
    graph.redirect_edge_source(moved, c);
    let appended = graph.add_edge(c, b, "appended");
    let own = graph.add_edge(a, d, "own");

    graph.move_adjacency(c, a, TraversalDirection::Outgoing);

    assert_eq!(
        graph.outgoing(a).collect::<Vec<_>>(),
        [own, resident, moved, appended]
    );
    assert!(graph.outgoing(c).next().is_none());
    for edge in [resident, moved, appended] {
        assert_eq!(graph.edge(edge).source(), a);
    }
    assert_eq!(graph.incoming(b).collect::<Vec<_>>(), [moved, appended]);
    assert_mirrored(&graph);

    // The emptied node is an ordinary node again.
    let later = graph.add_edge(c, d, "later");
    assert_eq!(graph.outgoing(c).collect::<Vec<_>>(), [later]);
    assert_mirrored(&graph);
}

#[test]
fn a_bulk_move_leaves_removed_edges_as_they_were() {
    let mut graph = Graph::<&'static str, &'static str>::new();
    let [a, b, c] = ["a", "b", "c"].map(|payload| graph.add_node(payload));
    let dead = graph.add_edge(a, b, "dead");
    let live = graph.add_edge(a, c, "live");
    graph.compact();
    assert!(graph.remove_edge(dead));

    graph.move_adjacency(a, b, TraversalDirection::Outgoing);

    assert_eq!(graph.edge(dead).source(), a, "a removed edge is not moved");
    assert_eq!(graph.edge(live).source(), b);
    assert_eq!(graph.outgoing(b).collect::<Vec<_>>(), [live]);
    assert_mirrored(&graph);
}

#[test]
fn a_bulk_move_of_nothing_names_no_replacement() {
    let mut graph = Graph::<&'static str, &'static str>::new();
    let [empty, gone] = ["empty", "gone"].map(|payload| graph.add_node(payload));
    graph.remove_node(gone);

    graph.move_adjacency(empty, gone, TraversalDirection::Incoming);

    assert!(graph.incoming(empty).next().is_none());
    assert!(graph.incoming(gone).next().is_none());
}

#[test]
fn verify_reports_an_edge_its_endpoint_stopped_listing() {
    // The shape of a dropped listing: the edge is live and its other
    // endpoint still lists it, but one endpoint's chain has forgotten it.
    for (direction, (outgoing, incoming)) in [
        (TraversalDirection::Outgoing, (0, 1)),
        (TraversalDirection::Incoming, (1, 0)),
    ] {
        let mut cfg = crate::Cfg::<u32>::new();
        let source = cfg.entry();
        let target = cfg.new_block();
        let edge = cfg.add_edge(source, target, crate::EdgeKind::Unconditional);
        assert!(crate::verify(&cfg).is_ok());

        let forgetful = match direction {
            TraversalDirection::Outgoing => source,
            TraversalDirection::Incoming => target,
        };
        cfg.graph.chains.detach_node(forgetful.index(), direction);

        let report = crate::verify(&cfg);
        let expected = alloc::format!(
            "edge index {} appears {outgoing} time(s) in outgoing and {incoming} time(s) in incoming adjacency",
            edge.index()
        );
        assert!(
            report.errors.iter().any(|error| error.message == expected),
            "a dropped {direction:?} listing must be reported: {:?}",
            report.errors
        );
    }
}

#[test]
fn removed_payloads_stay_readable_until_compaction() {
    let (mut graph, nodes) = diamond();
    let edge = graph.edge_ids().next().expect("the diamond has edges");
    graph.remove_node(nodes[1]);

    assert!(!graph.contains_node(nodes[1]));
    assert_eq!(graph.node(nodes[1]), &"left");
    assert!(!graph.contains_edge(edge));
    assert_eq!(graph.edge(edge).payload(), &"a");

    let renumbering = graph.compact();
    assert_eq!(renumbering.node(nodes[1]), None);
    assert_eq!(renumbering.edge(edge), None);
    assert_eq!(graph.node_bound(), 3);
    assert_eq!(graph.edge_bound(), 2);
}

#[test]
fn node_and_edge_payloads_are_mutable_in_place() {
    let (mut graph, nodes) = diamond();
    graph.compact();
    *graph.node_mut(nodes[0]) = "renamed";
    let edge = graph.edge_ids().next().expect("the diamond has edges");
    *graph.edge_mut(edge).payload_mut() = "retagged";

    assert_eq!(graph.node(nodes[0]), &"renamed");
    assert_eq!(graph.edge(edge).payload(), &"retagged");
    assert_eq!(graph.edge(edge).source(), nodes[0]);
}

#[test]
fn compacting_an_untouched_store_is_the_identity() {
    let (mut graph, _) = diamond();
    let first = graph.compact();
    assert!(first.is_identity());
    assert_eq!(first.node_bound(), 4);
    assert_eq!(first.edge_bound(), 4);

    let second = graph.compact();
    assert!(second.is_identity());
    assert!(first.compose(&second).is_identity());
}

#[test]
fn composed_renumberings_match_the_two_compactions_they_describe() {
    let (mut graph, nodes) = diamond();
    graph.remove_node(nodes[0]);
    let first = graph.compact();

    let added = graph.add_node("added");
    let surviving = first.node(nodes[2]).expect("right survived the compaction");
    graph.add_edge(surviving, added, "g");
    graph.remove_node(first.node(nodes[1]).expect("left survived the compaction"));
    let second = graph.compact();

    let composed = first.compose(&second);
    assert_eq!(composed.node(nodes[0]), None, "removed before the first");
    assert_eq!(composed.node(nodes[1]), None, "removed before the second");
    for original in [nodes[2], nodes[3]] {
        let direct = first
            .node(original)
            .and_then(|intermediate| second.node(intermediate));
        assert_eq!(composed.node(original), direct);
    }

    // Composition is only useful if the identity it reports still names the
    // entity the original identity named.
    assert_eq!(
        graph.node(composed.node(nodes[2]).expect("right is still present")),
        &"right"
    );
    assert_eq!(
        graph.node(composed.node(nodes[3]).expect("exit is still present")),
        &"exit"
    );
    assert_eq!(composed.node_bound(), 4);
}

#[test]
fn the_view_reports_slots_so_dense_analyses_stay_in_bounds() {
    let (mut graph, nodes) = diamond();
    graph.remove_node(nodes[1]);

    assert_eq!(GraphView::node_bound(&graph), 4);
    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.node_ids().count(), 3);

    // A removed node is not a node of the view at all, so a whole-graph
    // partition reports no phantom singleton for it.
    let uncompacted = tarjan_scc(&graph);
    assert_eq!(uncompacted.len(), 3);

    graph.compact();
    assert_eq!(GraphView::node_bound(&graph), 3);
    assert_eq!(tarjan_scc(&graph).len(), 3);
}

#[test]
fn liveness_answers_survive_the_tombstone_boundary() {
    let (mut graph, nodes) = diamond();
    let edges: Vec<_> = graph.edge_ids().collect();

    // Nothing has been removed, so containment is answered from the counts.
    assert!(graph.contains_node(nodes[1]));
    assert!(!graph.contains_node(NodeId::from_raw(4)));
    assert!(graph.contains_edge(edges[0]));

    graph.remove_node(nodes[1]);
    assert!(!graph.contains_node(nodes[1]));
    assert!(graph.contains_node(nodes[2]));
    assert!(!graph.contains_edge(edges[0]), "its edges went with it");
    assert!(graph.outgoing(nodes[1]).next().is_none());
    assert_eq!(graph.incoming(nodes[3]).count(), 1);

    // Compaction retires every tombstone, so the counts answer again.
    graph.compact();
    assert_eq!(graph.node_ids().count(), 3);
    assert_eq!(graph.edge_ids().count(), 2);
    assert!(graph.contains_node(NodeId::from_raw(2)));
    assert!(!graph.contains_node(NodeId::from_raw(3)));
}

#[test]
fn the_edge_view_exposes_live_edges_and_their_endpoints() {
    let (mut graph, nodes) = diamond();
    let edges: Vec<_> = graph.edge_ids().collect();
    graph.remove_edge(edges[0]);

    assert_eq!(EdgeView::edge_bound(&graph), 4);
    assert_eq!(graph.edge_ids().count(), 3);
    assert_eq!(
        EdgeView::outgoing(&graph, nodes[0]).collect::<Vec<_>>(),
        [edges[1]]
    );
    let reference = graph.edge(edges[1]);
    assert_eq!(reference.source(), nodes[0]);
    assert_eq!(reference.target(), nodes[2]);
    assert_eq!(reference.payload(), &"b");
    assert_eq!(graph.node(nodes[0]), &"entry");
}

#[test]
#[should_panic(expected = "edge has been removed")]
fn the_edge_view_refuses_a_removed_edge() {
    let (mut graph, _) = diamond();
    let edge = graph.edge_ids().next().expect("the diamond has edges");
    graph.remove_edge(edge);
    // The inherent accessor keeps the record readable; the view does not.
    assert_eq!(graph.edge(edge).payload(), &"a");
    let _ = EdgeView::edge(&graph, edge);
}

#[test]
#[should_panic(expected = "source node has been removed")]
fn an_edge_cannot_attach_to_a_removed_node() {
    let (mut graph, nodes) = diamond();
    graph.remove_node(nodes[0]);
    graph.add_edge(nodes[0], nodes[3], "impossible");
}

#[test]
fn the_algorithms_agree_with_the_arena_store() {
    let (mut store, store_nodes) = diamond();
    // A back edge makes the comparison cover a cycle as well as a diamond.
    store.add_edge(store_nodes[3], store_nodes[0], "back");
    store.compact();
    let (mut arena, arena_nodes) = arena_diamond();
    arena.add_edge(arena_nodes[3], arena_nodes[0], "back");

    let store_rooted = Rooted::new(&store, store_nodes[0]);
    let arena_rooted = Rooted::new(&arena, arena_nodes[0]);
    let store_dominators = DominatorTree::compute(&store_rooted);
    let arena_dominators = DominatorTree::compute(&arena_rooted);
    for (store_node, arena_node) in store_nodes.iter().zip(&arena_nodes) {
        assert_eq!(
            store_dominators.idom(*store_node).map(Id::index),
            arena_dominators.idom(*arena_node).map(crate::NodeId::index),
            "immediate dominator of {store_node}"
        );
    }

    let store_components = tarjan_scc(&store);
    let arena_components = tarjan_scc(&arena);
    assert_eq!(
        store_components
            .components()
            .map(|nodes| indexes(nodes.iter().copied(), Id::index))
            .collect::<Vec<_>>(),
        arena_components
            .components()
            .map(|nodes| indexes(nodes.iter().copied(), crate::NodeId::index))
            .collect::<Vec<_>>()
    );

    assert_eq!(
        indexes(
            depth_first_preorder(&store, store_nodes[0], TraversalDirection::Outgoing),
            Id::index
        ),
        indexes(
            depth_first_preorder(&arena, arena_nodes[0], TraversalDirection::Outgoing),
            crate::NodeId::index
        )
    );
}

#[test]
fn a_compacted_store_hands_back_its_renumbering() {
    let (mut graph, nodes) = diamond();
    graph.remove_node(nodes[0]);
    let (compacted, renumbering) = graph.compacted();

    assert!(compacted.is_compact());
    assert_eq!(compacted.node_count(), 3);
    assert_eq!(compacted.edge_count(), 2);
    assert_eq!(renumbering.node(nodes[0]), None);
    assert_eq!(
        compacted.node(renumbering.node(nodes[3]).expect("exit survived")),
        &"exit"
    );
}

#[cfg(feature = "serde")]
#[test]
fn a_compacted_store_round_trips_through_serde() {
    use alloc::string::ToString;

    let mut graph = Graph::<alloc::string::String, u32>::new();
    let nodes: Vec<_> = ["entry", "middle", "exit"]
        .into_iter()
        .map(|payload| graph.add_node(payload.to_string()))
        .collect();
    graph.add_edge(nodes[0], nodes[1], 1);
    graph.add_edge(nodes[1], nodes[2], 2);
    graph.remove_node(nodes[1]);
    graph.compact();
    graph.add_edge(nodes[0], nodes[0], 3);

    let encoded = serde_json::to_string(&graph).expect("the store serializes");
    let decoded: Graph<alloc::string::String, u32> =
        serde_json::from_str(&encoded).expect("the encoding round trips");
    assert_eq!(decoded, graph);
    assert_eq!(
        decoded.successors(nodes[0]).collect::<Vec<_>>(),
        graph.successors(nodes[0]).collect::<Vec<_>>()
    );
    assert_eq!(decoded.node_count(), graph.node_count());
}
