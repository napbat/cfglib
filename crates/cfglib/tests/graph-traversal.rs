//! Integration tests: traversal orders over stores with removed nodes.

use cfglib::{Graph, GraphView, scan_predecessors, topological_sort};

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

/// A view that stores forward adjacency only, as the docs of
/// `scan_predecessors` describe.
struct ForwardOnly(Vec<Vec<usize>>);

impl GraphView for ForwardOnly {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.0.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.0.len()
    }

    fn successors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.0[node].iter().copied()
    }

    fn predecessors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        scan_predecessors(self, node)
    }
}

#[test]
fn a_forward_only_view_with_parallel_edges_sorts_topologically() {
    let view = ForwardOnly(vec![vec![1, 1, 2], vec![2], vec![]]);

    assert_eq!(view.predecessors(1).collect::<Vec<_>>(), [0, 0]);
    assert_eq!(topological_sort(&view), Some(vec![0, 1, 2]));
}
