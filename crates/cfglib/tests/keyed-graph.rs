//! Integration tests: keyed graphs keep their key table in step with the
//! graph underneath.

use cfglib::KeyedGraph;

fn names() -> KeyedGraph<&'static str, &'static str, ()> {
    let mut graph = KeyedGraph::new();
    let a = graph.ensure_node(&"a", || "a");
    let b = graph.ensure_node(&"b", || "b");
    let c = graph.ensure_node(&"c", || "c");
    graph.add_edge(a, b, ());
    graph.add_edge(b, c, ());
    graph
}

#[test]
fn every_key_names_its_own_node_after_a_removal_and_a_compaction() {
    let mut graph = names();
    let before = graph.node_id(&"c").expect("minted");
    assert!(graph.remove_node(&"a"));

    let renumbering = graph.compact();

    assert_eq!(graph.node_id(&"a"), None);
    assert_eq!(graph.node_id(&"c"), renumbering.node(before));
    assert_eq!(graph.keys().count(), 2);
    for (key, node) in graph.keys() {
        assert_eq!(graph.graph().node(node), key);
    }
}

#[test]
fn a_removed_key_can_be_minted_again_and_linked() {
    let mut graph = names();
    assert!(graph.remove_node(&"b"));

    let b = graph.ensure_node(&"b", || "b");
    let a = graph.node_id(&"a").expect("a is still minted");
    graph.add_edge(a, b, ());

    assert_eq!(graph.graph().edge_count(), 1);
    assert_eq!(graph.graph().successors(a).collect::<Vec<_>>(), [b]);
}

#[cfg(feature = "serde")]
mod serde_round_trip {
    use super::{KeyedGraph, names};

    #[test]
    fn a_compacted_graph_round_trips() {
        let mut graph = names();
        graph.remove_node(&"a");
        graph.compact();

        let json = serde_json::to_string(&graph).expect("serializes");
        let loaded: KeyedGraph<&str, &str, ()> = serde_json::from_str(&json).expect("loads");
        assert_eq!(loaded, graph);
    }

    #[test]
    fn a_key_table_that_disagrees_with_the_graph_is_rejected() {
        let graph = names();
        let sound = serde_json::to_value(&graph).expect("serializes");
        assert!(serde_json::from_value::<KeyedGraph<String, String, ()>>(sound.clone()).is_ok());

        // Each tampering leaves the graph alone and changes only the table:
        // a node left without a key, a key beyond the graph, and one node
        // under two keys.
        for (label, table) in [
            ("a missing key", serde_json::json!({ "a": 0, "b": 1 })),
            (
                "a dangling key",
                serde_json::json!({ "a": 0, "b": 1, "c": 7 }),
            ),
            (
                "a shared node",
                serde_json::json!({ "a": 0, "b": 1, "c": 1 }),
            ),
        ] {
            let mut stored = sound.clone();
            stored["ids"] = table;
            let loaded = serde_json::from_value::<KeyedGraph<String, String, ()>>(stored);
            assert!(loaded.is_err(), "{label} must not load");
        }
    }
}
