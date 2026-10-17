//! Integration tests: traversal orders over stores with removed nodes.

use cfglib::{
    Graph, GraphView, NodeId, TraversalDirection, breadth_first, common_ancestors,
    depth_first_postorder, depth_first_preorder, nearest_common_ancestor, reachable,
    reverse_postorder, scan_predecessors, shortest_path, topological_sort,
};

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

/// A diamond `top -> {left, right} -> join` with `join -> tail`, where the slot
/// between `top` and `left` held a node that was removed, so every live index
/// above it is higher than the live count.
fn diamond_with_a_hole() -> (Graph<&'static str, ()>, [NodeId; 5]) {
    let mut graph = Graph::new();
    let top = graph.add_node("top");
    let removed = graph.add_node("removed");
    let [left, right, join, tail] =
        ["left", "right", "join", "tail"].map(|label| graph.add_node(label));
    for (source, target) in [
        (top, left),
        (top, right),
        (left, join),
        (right, join),
        (join, tail),
        (removed, join),
    ] {
        graph.add_edge(source, target, ());
    }
    assert!(graph.remove_node(removed));
    (graph, [top, left, right, join, tail])
}

#[test]
fn traversals_over_a_hole_equal_the_traversals_after_compaction() {
    let (graph, [top, left, right, join, tail]) = diamond_with_a_hole();
    let mut compacted = graph.clone();
    let renumbering = compacted.compact();
    let renumbered = |id: NodeId| renumbering.node(id).expect("a live node survives");
    let renumber = |ids: Vec<NodeId>| -> Vec<NodeId> { ids.into_iter().map(renumbered).collect() };

    for (direction, start) in [
        (TraversalDirection::Outgoing, top),
        (TraversalDirection::Incoming, tail),
    ] {
        let new_start = renumbered(start);
        assert_eq!(
            renumber(breadth_first(&graph, start, direction)),
            breadth_first(&compacted, new_start, direction),
            "breadth first {direction:?}"
        );
        assert_eq!(
            renumber(depth_first_preorder(&graph, start, direction)),
            depth_first_preorder(&compacted, new_start, direction),
            "preorder {direction:?}"
        );
        assert_eq!(
            renumber(depth_first_postorder(&graph, start, direction)),
            depth_first_postorder(&compacted, new_start, direction),
            "postorder {direction:?}"
        );
        assert_eq!(
            renumber(reverse_postorder(&graph, start, direction)),
            reverse_postorder(&compacted, new_start, direction),
            "reverse postorder {direction:?}"
        );
    }

    assert_eq!(
        shortest_path(&graph, top, tail, TraversalDirection::Outgoing).map(|path| path.len()),
        shortest_path(
            &compacted,
            renumbered(top),
            renumbered(tail),
            TraversalDirection::Outgoing
        )
        .map(|path| path.len())
    );
    assert_eq!(
        nearest_common_ancestor(&graph, left, right, TraversalDirection::Outgoing),
        Some(join)
    );
    assert_eq!(
        nearest_common_ancestor(
            &compacted,
            renumbered(left),
            renumbered(right),
            TraversalDirection::Outgoing
        ),
        Some(renumbered(join))
    );
    let shared = common_ancestors(&graph, left, right, TraversalDirection::Outgoing, None);
    assert_eq!(
        shared.iter().map(|found| found.node).collect::<Vec<_>>(),
        [join, tail]
    );

    // The reachability table is slot-indexed, and the removed slot is not
    // reachable even though an edge into `join` once left it.
    assert_eq!(
        reachable(&graph, [top], TraversalDirection::Outgoing),
        [true, false, true, true, true, true]
    );
    assert_eq!(
        reachable(&graph, [tail], TraversalDirection::Incoming),
        [true, false, true, true, true, true]
    );
}
