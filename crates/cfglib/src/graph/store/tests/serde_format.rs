//! The serialized form of a store whose edges have moved.
//!
//! The encoding is a compatibility contract: data written by an earlier
//! release must still load, and the same store must still encode to the same
//! bytes. The golden file is the encoding the relocation overlay produced
//! when it was an ordered map of vectors, so a change of its in-memory shape
//! cannot silently change what is written.

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::graph::store::{EdgeId, Graph, NodeId};
use crate::graph::traverse::TraversalDirection;
use crate::test_util::golden::assert_golden;

type Store = Graph<String, u32>;

const GOLDEN: &str = "serde/relocated-store.json";

/// Seven nodes and nine edges that leave overlays in both directions: out of
/// order, one left empty, one carrying a removed edge, and one node added
/// after the first relocation.
fn relocated_store() -> (Store, Vec<NodeId>, Vec<EdgeId>) {
    let mut graph = Store::new();
    let mut nodes: Vec<_> = (0..6)
        .map(|index| graph.add_node(alloc::format!("n{index}")))
        .collect();
    let mut edges = Vec::new();
    for (payload, (source, target)) in [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (2, 5)]
        .into_iter()
        .enumerate()
    {
        let payload = u32::try_from(payload).expect("the fixture is small");
        edges.push(graph.add_edge(nodes[source], nodes[target], payload));
    }
    graph.compact();
    edges.push(graph.add_edge(nodes[5], nodes[0], 7));

    // Node 4 loses its only outgoing edge, node 2 its only incoming one, so
    // both keep an empty overlay run.
    graph.redirect_edge_source(edges[5], nodes[1]);
    graph.redirect_edge_target(edges[1], nodes[3]);
    graph.redirect_edge_source(edges[6], nodes[0]);
    // A removed edge stays in the runs that listed it until they are swept.
    assert!(graph.remove_edge(edges[2]));

    nodes.push(graph.add_node("n6".to_string()));
    edges.push(graph.add_edge(nodes[6], nodes[3], 8));
    (graph, nodes, edges)
}

/// The golden, loaded: what a store written by an earlier release becomes.
fn stored_store() -> Store {
    serde_json::from_str(include_str!(
        "../../../../tests/golden/serde/relocated-store.json"
    ))
    .expect("the stored encoding loads")
}

fn listing(graph: &Store, node: NodeId) -> (Vec<EdgeId>, Vec<EdgeId>) {
    (
        graph.outgoing(node).collect(),
        graph.incoming(node).collect(),
    )
}

fn assert_same_listings(actual: &Store, expected: &Store) {
    assert_eq!(actual, expected);
    for node in expected.node_ids() {
        assert_eq!(listing(actual, node), listing(expected, node), "{node}");
    }
}

#[test]
fn a_relocated_store_encodes_as_it_always_has() {
    let (graph, ..) = relocated_store();
    let encoded = serde_json::to_string(&graph).expect("the store serializes");
    assert_golden(GOLDEN, &encoded);
}

#[test]
fn the_stored_encoding_still_loads_and_keeps_relocating() {
    let (mut expected, nodes, edges) = relocated_store();
    let mut decoded = stored_store();
    assert_same_listings(&decoded, &expected);

    // Relocate onto nodes that have an overlay, onto one that has none, and
    // onto the node added after the overlays were built.
    for graph in [&mut expected, &mut decoded] {
        graph.redirect_edge_target(edges[8], nodes[5]);
        graph.redirect_edge_source(edges[0], nodes[6]);
        graph.move_adjacency(nodes[2], nodes[1], TraversalDirection::Outgoing);
        graph.move_adjacency(nodes[3], nodes[6], TraversalDirection::Incoming);
    }
    assert_same_listings(&decoded, &expected);

    let reloaded: Store = serde_json::from_str(
        &serde_json::to_string(&decoded).expect("the loaded store serializes"),
    )
    .expect("and loads again");
    assert_same_listings(&reloaded, &expected);
}

#[test]
fn a_store_that_never_relocated_writes_no_overlay() {
    let mut graph = Store::new();
    let first = graph.add_node("first".to_string());
    let second = graph.add_node("second".to_string());
    graph.add_edge(first, second, 1);
    graph.compact();

    let encoded = serde_json::to_string(&graph).expect("the store serializes");
    assert!(!encoded.contains("relocations"), "{encoded}");
    let decoded: Store = serde_json::from_str(&encoded).expect("the encoding loads");
    assert_same_listings(&decoded, &graph);
}

#[test]
fn compaction_folds_a_loaded_overlay_into_the_index() {
    let (expected, ..) = relocated_store();
    let mut decoded = stored_store();
    let mut compacted = expected;
    compacted.compact();
    decoded.compact();

    assert!(decoded.is_compact());
    assert_same_listings(&decoded, &compacted);
    let encoded = serde_json::to_string(&decoded).expect("the store serializes");
    assert!(!encoded.contains("relocations"), "{encoded}");
}

/// The golden with its incoming map replaced, to exercise the loader.
fn stored_with_incoming(incoming: &str) -> String {
    let stored = include_str!("../../../../tests/golden/serde/relocated-store.json");
    let original = r#""incoming":{"2":[],"3":[2,3,1]}"#;
    assert!(stored.contains(original), "the golden's overlay changed");
    stored.replace(original, &alloc::format!(r#""incoming":{incoming}"#))
}

#[test]
fn overlay_keys_may_arrive_in_any_order() {
    let (expected, ..) = relocated_store();
    let decoded: Store = serde_json::from_str(&stored_with_incoming(r#"{"3":[2,3,1],"2":[]}"#))
        .expect("the reordered encoding loads");
    assert_same_listings(&decoded, &expected);
}

#[test]
fn an_overlay_key_that_cannot_name_a_node_is_rejected() {
    let error = serde_json::from_str::<Store>(&stored_with_incoming(r#"{"4294967295":[]}"#))
        .expect_err("no node has the last identity");
    assert!(error.to_string().contains("does not hold"), "{error}");
}

/// A small store in both states: a compacted base plus one appended edge.
fn base_and_delta_store() -> serde_json::Value {
    let mut graph = Store::new();
    let [a, b, c] = ["a", "b", "c"].map(|label| graph.add_node(label.to_string()));
    graph.add_edge(a, b, 1);
    graph.add_edge(b, c, 2);
    graph.compact();
    graph.add_edge(c, a, 3);
    serde_json::to_value(&graph).expect("the store serializes")
}

#[test]
fn a_document_whose_columns_disagree_is_rejected_not_trusted() {
    let sound = base_and_delta_store();
    assert!(serde_json::from_value::<Store>(sound.clone()).is_ok());

    // Each tampering replaces one value, named by its JSON pointer.
    let tamperings = [
        (
            "a live count that lies",
            "/live_node_count",
            serde_json::json!(2),
        ),
        (
            "an edge number past the edges",
            "/out_edges/0",
            serde_json::json!(9999),
        ),
        (
            "an edge into a missing node",
            "/base_edges/0/target",
            serde_json::json!(77),
        ),
        (
            "an index that is too short",
            "/out_offsets",
            serde_json::json!([0]),
        ),
        (
            "a chain ending past the delta",
            "/chains/last/2/0",
            serde_json::json!(9),
        ),
        (
            "a ring that never returns to its end",
            "/chains/next/0",
            serde_json::json!([5, 5]),
        ),
        (
            "adjacency that lists an edge twice",
            "/in_edges",
            serde_json::json!([0, 0]),
        ),
    ];
    for (label, pointer, value) in tamperings {
        let mut document = sound.clone();
        *document
            .pointer_mut(pointer)
            .expect("the encoding has that column") = value;
        assert!(
            serde_json::from_value::<Store>(document).is_err(),
            "{label} must not load"
        );
    }
}
