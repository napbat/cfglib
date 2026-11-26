use super::*;
use crate::graph::store::{Graph, NodeId};
use crate::graph::view::Direction;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

/// `a -> b`, `a -> d`, `a -> c`, `b -> c`, `c -> a`, `d -> c`: one graph
/// carrying all four edge classes.
fn dfs_graph() -> (Graph<(), ()>, [NodeId; 4]) {
    let mut graph = Graph::<(), ()>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.add_edge(a, b, ());
    graph.add_edge(a, d, ());
    graph.add_edge(a, c, ());
    graph.add_edge(b, c, ());
    graph.add_edge(c, a, ());
    graph.add_edge(d, c, ());
    (graph, [a, b, c, d])
}

fn dfs_events(graph: &Graph<(), ()>, start: NodeId) -> Vec<DfsEvent<NodeId>> {
    let mut log = Vec::new();
    let outcome = depth_first_events(graph, start, Direction::Forward, |event| {
        log.push(event);
        ControlFlow::<()>::Continue(())
    });
    assert_eq!(outcome, None);
    log
}

#[test]
fn depth_first_events_classify_every_edge_in_a_pinned_order() {
    let (graph, [a, b, c, d]) = dfs_graph();
    assert_eq!(
        dfs_events(&graph, a),
        vec![
            DfsEvent::Discover(a, 0),
            DfsEvent::TreeEdge(a, b),
            DfsEvent::Discover(b, 1),
            DfsEvent::TreeEdge(b, c),
            DfsEvent::Discover(c, 2),
            // c -> a closes the cycle: `a` is an ancestor on the path.
            DfsEvent::BackEdge(c, a),
            DfsEvent::Finish(c),
            DfsEvent::Finish(b),
            DfsEvent::TreeEdge(a, d),
            DfsEvent::Discover(d, 1),
            // d -> c is a cross edge into a finished sibling subtree.
            DfsEvent::ForwardOrCross(d, c),
            DfsEvent::Finish(d),
            // a -> c is a forward edge to a finished descendant.
            DfsEvent::ForwardOrCross(a, c),
            DfsEvent::Finish(a),
        ]
    );
}

#[test]
fn depth_first_events_report_self_edges_as_back_edges() {
    let mut graph = Graph::<(), ()>::new();
    let only = graph.add_node(());
    graph.add_edge(only, only, ());
    assert_eq!(
        dfs_events(&graph, only),
        vec![
            DfsEvent::Discover(only, 0),
            DfsEvent::BackEdge(only, only),
            DfsEvent::Finish(only),
        ]
    );
}

#[test]
fn depth_first_events_only_cover_the_reachable_set() {
    let mut graph = Graph::<(), ()>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let unreached = graph.add_node(());
    graph.add_edge(a, b, ());
    graph.add_edge(unreached, a, ());

    assert_eq!(
        dfs_events(&graph, a),
        vec![
            DfsEvent::Discover(a, 0),
            DfsEvent::TreeEdge(a, b),
            DfsEvent::Discover(b, 1),
            DfsEvent::Finish(b),
            DfsEvent::Finish(a),
        ],
        "a predecessor-only node produces no events"
    );
}

#[test]
fn depth_first_events_break_stops_the_walk() {
    let (graph, [a, b, c, _d]) = dfs_graph();
    let mut log = Vec::new();
    let found = depth_first_events(&graph, a, Direction::Forward, |event| {
        log.push(event);
        match event {
            DfsEvent::BackEdge(from, to) => ControlFlow::Break((from, to)),
            _ => ControlFlow::Continue(()),
        }
    });

    assert_eq!(found, Some((c, a)));
    assert_eq!(
        log,
        vec![
            DfsEvent::Discover(a, 0),
            DfsEvent::TreeEdge(a, b),
            DfsEvent::Discover(b, 1),
            DfsEvent::TreeEdge(b, c),
            DfsEvent::Discover(c, 2),
            DfsEvent::BackEdge(c, a),
        ]
    );
}

#[test]
fn depth_first_events_follow_the_selected_axis() {
    let mut graph = Graph::<(), ()>::new();
    let first = graph.add_node(());
    let middle = graph.add_node(());
    let last = graph.add_node(());
    graph.add_edge(first, middle, ());
    graph.add_edge(middle, last, ());

    let mut log = Vec::new();
    let outcome = depth_first_events(&graph, last, Direction::Backward, |event| {
        log.push(event);
        ControlFlow::<()>::Continue(())
    });
    assert_eq!(outcome, None);
    assert_eq!(
        log,
        vec![
            DfsEvent::Discover(last, 0),
            DfsEvent::TreeEdge(last, middle),
            DfsEvent::Discover(middle, 1),
            DfsEvent::TreeEdge(middle, first),
            DfsEvent::Discover(first, 2),
            DfsEvent::Finish(first),
            DfsEvent::Finish(middle),
            DfsEvent::Finish(last),
        ]
    );
}
