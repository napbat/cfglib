//! Integration tests: every implementation of a traversal order agrees.
//!
//! Depth-first preorder, depth-first postorder, and breadth-first order each
//! have several independent implementations: the order-returning traversals,
//! [`search`], the dense event walks, and the open walks over a lazily
//! discovered node space. Their documentation promises the same adjacency
//! order, so on one graph they have to visit the same nodes in the same
//! sequence.

use core::ops::ControlFlow;

use cfglib::{
    BfsEvent, DfsEvent, Graph, NodeId, OpenBfsConfig, OpenBfsEvent, OpenDfsConfig, OpenDfsEvent,
    OpenSearchConfig, SearchConfig, SearchOrder, TraversalDirection, Visit, VisitedPolicy,
    breadth_first, breadth_first_events, depth_first_events, depth_first_postorder,
    depth_first_preorder, open_breadth_first_events, open_depth_first_events, open_search, search,
};

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
