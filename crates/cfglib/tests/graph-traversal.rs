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

use cfglib::{
    BfsEvent, DfsEvent, OpenBfsConfig, OpenBfsEvent, OpenDfsConfig, OpenDfsEvent, OpenSearchConfig,
    SearchConfig, SearchOrder, Visit, VisitedPolicy, breadth_first_events, depth_first_events,
    open_breadth_first_events, open_depth_first_events, open_search, search,
};
use core::ops::ControlFlow;

/// A named graph: how many nodes it adds, which edges it adds between them in
/// order, and which nodes it removes afterwards.
struct Shape {
    name: &'static str,
    nodes: usize,
    edges: &'static [(usize, usize)],
    removed: &'static [usize],
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "diamond",
        nodes: 4,
        edges: &[(0, 1), (0, 2), (1, 3), (2, 3)],
        removed: &[],
    },
    Shape {
        name: "cycle",
        nodes: 3,
        edges: &[(0, 1), (1, 2), (2, 0)],
        removed: &[],
    },
    Shape {
        name: "parallel edges",
        nodes: 3,
        edges: &[(0, 1), (0, 2), (0, 1), (1, 2), (1, 2), (2, 0)],
        removed: &[],
    },
    Shape {
        name: "self loop",
        nodes: 3,
        edges: &[(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)],
        removed: &[],
    },
    Shape {
        name: "disconnected component",
        nodes: 6,
        edges: &[(0, 1), (0, 2), (1, 2), (3, 4), (4, 5), (5, 3)],
        removed: &[],
    },
    Shape {
        name: "removed node",
        nodes: 6,
        edges: &[
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 1),
            (2, 5),
        ],
        removed: &[2],
    },
    Shape {
        name: "removed first node",
        nodes: 4,
        edges: &[(0, 1), (1, 2), (1, 3), (2, 3), (3, 1), (3, 0)],
        removed: &[0],
    },
    Shape {
        name: "wide fan-out past the inline marks",
        nodes: 14,
        edges: &[
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (13, 12),
            (12, 11),
            (11, 10),
            (10, 9),
            (9, 8),
            (8, 7),
            (7, 6),
            (6, 5),
            (5, 4),
            (4, 3),
            (3, 2),
            (2, 1),
            (1, 0),
        ],
        removed: &[],
    },
];

fn build(shape: &Shape) -> Graph<usize, ()> {
    let mut graph = Graph::new();
    let ids: Vec<NodeId> = (0..shape.nodes)
        .map(|index| graph.add_node(index))
        .collect();
    for &(source, target) in shape.edges {
        graph.add_edge(ids[source], ids[target], ());
    }
    for &node in shape.removed {
        assert!(graph.remove_node(ids[node]), "{}", shape.name);
    }
    graph
}

/// Every `(start, direction)` the agreement is checked from: each live node,
/// along both edge directions.
fn starts(graph: &Graph<usize, ()>) -> Vec<(NodeId, TraversalDirection)> {
    let mut starts = Vec::new();
    for direction in [TraversalDirection::Outgoing, TraversalDirection::Incoming] {
        starts.extend(graph.node_ids().map(|node| (node, direction)));
    }
    starts
}

/// The successor relation an open walk is driven by, matching `direction`.
fn open_edges(
    graph: &Graph<usize, ()>,
    direction: TraversalDirection,
) -> impl FnMut(&NodeId, &mut Vec<NodeId>) + '_ {
    move |&node, out| match direction {
        TraversalDirection::Outgoing => out.extend(graph.successors(node)),
        TraversalDirection::Incoming => out.extend(graph.predecessors(node)),
    }
}

/// The nodes `search` visits, with the depth it reports for each.
fn searched(
    graph: &Graph<usize, ()>,
    start: NodeId,
    order: SearchOrder,
    visited: VisitedPolicy,
    direction: TraversalDirection,
) -> Vec<(NodeId, usize)> {
    let mut visits = Vec::new();
    let outcome = search(
        graph,
        [start],
        SearchConfig::new(order, direction).with_visited(visited),
        |node, depth| {
            visits.push((node, depth));
            ControlFlow::<(), _>::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    visits
}

/// The nodes `open_search` visits, with the depth it reports for each.
fn open_searched(
    graph: &Graph<usize, ()>,
    start: NodeId,
    order: SearchOrder,
    visited: VisitedPolicy,
    direction: TraversalDirection,
) -> Vec<(NodeId, usize)> {
    let mut visits = Vec::new();
    let outcome = open_search(
        [start],
        OpenSearchConfig::new(order).with_visited(visited),
        open_edges(graph, direction),
        |&node, depth| {
            visits.push((node, depth));
            ControlFlow::<(), _>::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    visits
}

fn nodes_of(visits: &[(NodeId, usize)]) -> Vec<NodeId> {
    visits.iter().map(|&(node, _)| node).collect()
}

/// A depth-first walk's discovery order with depths, and its finish order.
struct DepthFirst {
    discovered: Vec<(NodeId, usize)>,
    finished: Vec<NodeId>,
}

fn dense_depth_first_events(
    graph: &Graph<usize, ()>,
    start: NodeId,
    direction: TraversalDirection,
) -> DepthFirst {
    let mut walk = DepthFirst {
        discovered: Vec::new(),
        finished: Vec::new(),
    };
    let outcome = depth_first_events(graph, start, direction, |event| {
        match event {
            DfsEvent::Discover(node, depth) => walk.discovered.push((node, depth)),
            DfsEvent::Finish(node) => walk.finished.push(node),
            DfsEvent::TreeEdge(..) | DfsEvent::BackEdge(..) | DfsEvent::ForwardOrCross(..) => {}
        }
        ControlFlow::<()>::Continue(())
    });
    assert_eq!(outcome, None);
    walk
}

fn open_depth_first_walk(
    graph: &Graph<usize, ()>,
    start: NodeId,
    visited: VisitedPolicy,
    direction: TraversalDirection,
) -> DepthFirst {
    let mut walk = DepthFirst {
        discovered: Vec::new(),
        finished: Vec::new(),
    };
    let outcome = open_depth_first_events(
        [start],
        OpenDfsConfig::new(visited),
        open_edges(graph, direction),
        |event| {
            match event {
                OpenDfsEvent::Discover(&node, depth) => walk.discovered.push((node, depth)),
                OpenDfsEvent::Finish(&node) => walk.finished.push(node),
                OpenDfsEvent::Refused(..) => {}
            }
            ControlFlow::<(), _>::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    walk
}

fn dense_breadth_first_events(
    graph: &Graph<usize, ()>,
    start: NodeId,
    direction: TraversalDirection,
) -> Vec<(NodeId, usize)> {
    let mut discovered = Vec::new();
    let outcome = breadth_first_events(graph, start, direction, |event| {
        if let BfsEvent::Discover(node, depth) = event {
            discovered.push((node, depth));
        }
        ControlFlow::<()>::Continue(())
    });
    assert_eq!(outcome, None);
    discovered
}

fn open_breadth_first_walk(
    graph: &Graph<usize, ()>,
    start: NodeId,
    direction: TraversalDirection,
) -> Vec<(NodeId, usize)> {
    let mut discovered = Vec::new();
    let outcome = open_breadth_first_events(
        [start],
        OpenBfsConfig::new(),
        open_edges(graph, direction),
        |event| {
            if let OpenBfsEvent::Discover(&node, depth) = event {
                discovered.push((node, depth));
            }
            ControlFlow::<(), _>::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    discovered
}

#[test]
fn every_depth_first_preorder_visits_the_same_nodes_in_the_same_order() {
    for shape in SHAPES {
        let graph = build(shape);
        for (start, direction) in starts(&graph) {
            let context = format!("{} from {start} {direction:?}", shape.name);
            let expected = depth_first_preorder(&graph, start, direction);

            let searched = searched(
                &graph,
                start,
                SearchOrder::DepthFirst,
                VisitedPolicy::Global,
                direction,
            );
            assert_eq!(nodes_of(&searched), expected, "search: {context}");

            let events = dense_depth_first_events(&graph, start, direction);
            assert_eq!(nodes_of(&events.discovered), expected, "events: {context}");
            assert_eq!(events.discovered, searched, "event depths: {context}");

            let open = open_searched(
                &graph,
                start,
                SearchOrder::DepthFirst,
                VisitedPolicy::Global,
                direction,
            );
            assert_eq!(open, searched, "open search: {context}");

            let open_events =
                open_depth_first_walk(&graph, start, VisitedPolicy::Global, direction);
            assert_eq!(open_events.discovered, searched, "open events: {context}");
        }
    }
}

#[test]
fn every_depth_first_postorder_finishes_the_same_nodes_in_the_same_order() {
    for shape in SHAPES {
        let graph = build(shape);
        for (start, direction) in starts(&graph) {
            let context = format!("{} from {start} {direction:?}", shape.name);
            let expected = depth_first_postorder(&graph, start, direction);

            let events = dense_depth_first_events(&graph, start, direction);
            assert_eq!(events.finished, expected, "events: {context}");

            let open_events =
                open_depth_first_walk(&graph, start, VisitedPolicy::Global, direction);
            assert_eq!(open_events.finished, expected, "open events: {context}");
        }
    }
}

#[test]
fn every_breadth_first_walk_visits_the_same_nodes_in_the_same_order() {
    for shape in SHAPES {
        let graph = build(shape);
        for (start, direction) in starts(&graph) {
            let context = format!("{} from {start} {direction:?}", shape.name);
            let expected = breadth_first(&graph, start, direction);

            let searched = searched(
                &graph,
                start,
                SearchOrder::BreadthFirst,
                VisitedPolicy::Global,
                direction,
            );
            assert_eq!(nodes_of(&searched), expected, "search: {context}");

            let events = dense_breadth_first_events(&graph, start, direction);
            assert_eq!(events, searched, "events: {context}");

            let open = open_searched(
                &graph,
                start,
                SearchOrder::BreadthFirst,
                VisitedPolicy::Global,
                direction,
            );
            assert_eq!(open, searched, "open search: {context}");

            let open_events = open_breadth_first_walk(&graph, start, direction);
            assert_eq!(open_events, searched, "open events: {context}");
        }
    }
}

#[test]
fn every_path_marked_depth_first_walk_visits_each_route_in_the_same_order() {
    for shape in SHAPES {
        let graph = build(shape);
        for (start, direction) in starts(&graph) {
            let context = format!("{} from {start} {direction:?}", shape.name);

            let searched = searched(
                &graph,
                start,
                SearchOrder::DepthFirst,
                VisitedPolicy::Path,
                direction,
            );

            let open = open_searched(
                &graph,
                start,
                SearchOrder::DepthFirst,
                VisitedPolicy::Path,
                direction,
            );
            assert_eq!(open, searched, "open search: {context}");

            let open_events = open_depth_first_walk(&graph, start, VisitedPolicy::Path, direction);
            assert_eq!(open_events.discovered, searched, "open events: {context}");
            assert_eq!(
                open_events.finished.len(),
                searched.len(),
                "every route finishes: {context}"
            );
        }
    }
}
