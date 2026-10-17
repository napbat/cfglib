//! Integration tests: traversal orders over stores with removed nodes.

use cfglib::{Graph, topological_sort};

#[test]
fn topological_sort_ignores_removed_nodes() {
    let mut graph = Graph::new();
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let removed = graph.add_node("removed");
    graph.add_edge(a, b, ());
    assert!(graph.remove_node(removed));

    assert_eq!(topological_sort(&graph), Some(vec![a, b]));
}

#[test]
fn topological_sort_still_rejects_a_cycle_next_to_a_removed_node() {
    let mut graph = Graph::new();
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let removed = graph.add_node("removed");
    graph.add_edge(a, b, ());
    graph.add_edge(b, a, ());
    assert!(graph.remove_node(removed));

    assert_eq!(topological_sort(&graph), None);
}
