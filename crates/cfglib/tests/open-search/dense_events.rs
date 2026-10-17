//! `depth_first_events` and `breadth_first_events`: the walks whose output order is the traversal.

use std::collections::VecDeque;
use std::ops::ControlFlow;

use cfglib::{
    BfsEvent, DfsEvent, Graph, NodeId, TraversalDirection, breadth_first, breadth_first_events,
    depth_first_events, depth_first_postorder, depth_first_preorder, reachable,
};

use crate::Lcg;
use crate::reference::{Table, chain};

/// An event of either walk, with plain indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    Discover(usize, usize),
    Tree(usize, usize),
    Back(usize, usize),
    ForwardOrCross(usize, usize),
    NonTree(usize, usize),
    Finish(usize),
}

fn from_depth_first(event: DfsEvent<NodeId>) -> Event {
    match event {
        DfsEvent::Discover(node, depth) => Event::Discover(node.index(), depth),
        DfsEvent::TreeEdge(from, to) => Event::Tree(from.index(), to.index()),
        DfsEvent::BackEdge(from, to) => Event::Back(from.index(), to.index()),
        DfsEvent::ForwardOrCross(from, to) => Event::ForwardOrCross(from.index(), to.index()),
        DfsEvent::Finish(node) => Event::Finish(node.index()),
    }
}

fn from_breadth_first(event: BfsEvent<NodeId>) -> Event {
    match event {
        BfsEvent::Discover(node, depth) => Event::Discover(node.index(), depth),
        BfsEvent::TreeEdge(from, to) => Event::Tree(from.index(), to.index()),
        BfsEvent::NonTreeEdge(from, to) => Event::NonTree(from.index(), to.index()),
    }
}

fn depth_first_stream(
    graph: &Graph<(), ()>,
    start: usize,
    direction: TraversalDirection,
) -> Vec<Event> {
    let mut events = Vec::new();
    let outcome =
        depth_first_events::<_, ()>(graph, NodeId::from_index(start), direction, |event| {
            events.push(from_depth_first(event));
            ControlFlow::Continue(())
        });
    assert_eq!(outcome, None);
    events
}

fn breadth_first_stream(
    graph: &Graph<(), ()>,
    start: usize,
    direction: TraversalDirection,
) -> Vec<Event> {
    let mut events = Vec::new();
    let outcome =
        breadth_first_events::<_, ()>(graph, NodeId::from_index(start), direction, |event| {
            events.push(from_breadth_first(event));
            ControlFlow::Continue(())
        });
    assert_eq!(outcome, None);
    events
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Color {
    White,
    Gray,
    Black,
}

/// The tri-color classification, spelled recursively from the definition.
struct DepthFirstReference<'a> {
    table: &'a Table,
    color: Vec<Color>,
    events: Vec<Event>,
}

impl DepthFirstReference<'_> {
    fn visit(&mut self, node: usize, depth: usize) {
        self.color[node] = Color::Gray;
        self.events.push(Event::Discover(node, depth));
        let table = self.table;
        for &next in &table.successors[node] {
            match self.color[next] {
                Color::White => {
                    self.events.push(Event::Tree(node, next));
                    self.visit(next, depth + 1);
                }
                Color::Gray => self.events.push(Event::Back(node, next)),
                Color::Black => self.events.push(Event::ForwardOrCross(node, next)),
            }
        }
        self.color[node] = Color::Black;
        self.events.push(Event::Finish(node));
    }
}

fn depth_first_reference(table: &Table, start: usize) -> Vec<Event> {
    let mut walk = DepthFirstReference {
        table,
        color: vec![Color::White; table.node_count()],
        events: Vec::new(),
    };
    walk.visit(start, 0);
    walk.events
}

fn breadth_first_reference(table: &Table, start: usize) -> Vec<Event> {
    let mut seen = vec![false; table.node_count()];
    let mut events = vec![Event::Discover(start, 0)];
    let mut queue = VecDeque::from([(start, 0)]);
    seen[start] = true;
    while let Some((node, depth)) = queue.pop_front() {
        for &next in &table.successors[node] {
            if seen[next] {
                events.push(Event::NonTree(node, next));
            } else {
                seen[next] = true;
                events.push(Event::Tree(node, next));
                events.push(Event::Discover(next, depth + 1));
                queue.push_back((next, depth + 1));
            }
        }
    }
    events
}

/// Whether the subgraph induced on `nodes` has a cycle, by repeatedly removing sources.
fn is_cyclic(table: &Table, nodes: &[usize]) -> bool {
    let mut indegree = vec![0_usize; table.node_count()];
    for &node in nodes {
        for &next in &table.successors[node] {
            indegree[next] += 1;
        }
    }
    let mut ready: Vec<usize> = nodes
        .iter()
        .copied()
        .filter(|&node| indegree[node] == 0)
        .collect();
    let mut removed = 0;
    while let Some(node) = ready.pop() {
        removed += 1;
        for &next in &table.successors[node] {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                ready.push(next);
            }
        }
    }
    removed < nodes.len()
}

#[test]
fn depth_first_events_match_the_recursive_tricolor_reference_in_both_directions() {
    for seed in 1..=5 {
        let table = Table::random(&mut Lcg(seed), 30, 60);
        let backward = table.reversed();
        let graph = table.graph();
        for start in [0, 7, 19] {
            assert_eq!(
                depth_first_stream(&graph, start, TraversalDirection::Outgoing),
                depth_first_reference(&table, start),
                "seed {seed}, start {start}, forward"
            );
            assert_eq!(
                depth_first_stream(&graph, start, TraversalDirection::Incoming),
                depth_first_reference(&backward, start),
                "seed {seed}, start {start}, backward"
            );
        }
    }
}

#[test]
fn breadth_first_events_match_the_queue_reference_in_both_directions() {
    for seed in 1..=5 {
        let table = Table::random(&mut Lcg(seed + 30), 30, 60);
        let backward = table.reversed();
        let graph = table.graph();
        for start in [0, 7, 19] {
            assert_eq!(
                breadth_first_stream(&graph, start, TraversalDirection::Outgoing),
                breadth_first_reference(&table, start),
                "seed {seed}, start {start}, forward"
            );
            assert_eq!(
                breadth_first_stream(&graph, start, TraversalDirection::Incoming),
                breadth_first_reference(&backward, start),
                "seed {seed}, start {start}, backward"
            );
        }
    }
}

#[test]
fn a_hand_derived_depth_first_stream_names_every_edge_class() {
    // 0 -> {1, 2}, 1 -> 2, 2 -> 0
    let table = Table {
        successors: vec![vec![1, 2], vec![2], vec![0]],
    };
    let events = depth_first_stream(&table.graph(), 0, TraversalDirection::Outgoing);
    assert_eq!(
        events,
        [
            Event::Discover(0, 0),
            Event::Tree(0, 1),
            Event::Discover(1, 1),
            Event::Tree(1, 2),
            Event::Discover(2, 2),
            Event::Back(2, 0),
            Event::Finish(2),
            Event::Finish(1),
            Event::ForwardOrCross(0, 2),
            Event::Finish(0),
        ]
    );
}

#[test]
fn a_hand_derived_breadth_first_stream_separates_tree_from_non_tree_edges() {
    let table = Table {
        successors: vec![vec![1, 2], vec![2], vec![0]],
    };
    let events = breadth_first_stream(&table.graph(), 0, TraversalDirection::Outgoing);
    assert_eq!(
        events,
        [
            Event::Discover(0, 0),
            Event::Tree(0, 1),
            Event::Discover(1, 1),
            Event::Tree(0, 2),
            Event::Discover(2, 1),
            Event::NonTree(1, 2),
            Event::NonTree(2, 0),
        ]
    );
}

#[test]
fn self_loops_and_parallel_edges_are_classified_edge_by_edge() {
    let table = Table {
        successors: vec![vec![0, 1, 1], vec![]],
    };
    let graph = table.graph();
    assert_eq!(
        depth_first_stream(&graph, 0, TraversalDirection::Outgoing),
        [
            Event::Discover(0, 0),
            Event::Back(0, 0),
            Event::Tree(0, 1),
            Event::Discover(1, 1),
            Event::Finish(1),
            Event::ForwardOrCross(0, 1),
            Event::Finish(0),
        ],
        "a self loop is a back edge; the second parallel edge reaches a finished node"
    );
    assert_eq!(
        breadth_first_stream(&graph, 0, TraversalDirection::Outgoing),
        [
            Event::Discover(0, 0),
            Event::NonTree(0, 0),
            Event::Tree(0, 1),
            Event::Discover(1, 1),
            Event::NonTree(0, 1),
        ]
    );
}

#[test]
fn depth_first_events_agree_with_the_plain_depth_first_traversals() {
    for seed in 1..=5 {
        let table = Table::random(&mut Lcg(seed + 50), 30, 60);
        let graph = table.graph();
        let events = depth_first_stream(&graph, 0, TraversalDirection::Outgoing);
        let discovered: Vec<usize> = events
            .iter()
            .filter_map(|event| match event {
                Event::Discover(node, _) => Some(*node),
                _ => None,
            })
            .collect();
        let finished: Vec<usize> = events
            .iter()
            .filter_map(|event| match event {
                Event::Finish(node) => Some(*node),
                _ => None,
            })
            .collect();
        let start = NodeId::from_index(0);
        let preorder: Vec<usize> =
            depth_first_preorder(&graph, start, TraversalDirection::Outgoing)
                .into_iter()
                .map(NodeId::index)
                .collect();
        let postorder: Vec<usize> =
            depth_first_postorder(&graph, start, TraversalDirection::Outgoing)
                .into_iter()
                .map(NodeId::index)
                .collect();
        assert_eq!(
            discovered, preorder,
            "seed {seed}: discovery is the preorder"
        );
        assert_eq!(
            finished, postorder,
            "seed {seed}: finishing is the postorder"
        );
    }
}

#[test]
fn breadth_first_events_discover_nodes_in_breadth_first_order_at_their_hop_distance() {
    for seed in 1..=5 {
        let table = Table::random(&mut Lcg(seed + 60), 30, 60);
        let graph = table.graph();
        let events = breadth_first_stream(&graph, 0, TraversalDirection::Outgoing);
        let discovered: Vec<(usize, usize)> = events
            .iter()
            .filter_map(|event| match event {
                Event::Discover(node, depth) => Some((*node, *depth)),
                _ => None,
            })
            .collect();
        let order: Vec<usize> =
            breadth_first(&graph, NodeId::from_index(0), TraversalDirection::Outgoing)
                .into_iter()
                .map(NodeId::index)
                .collect();
        assert_eq!(
            discovered.iter().map(|&(node, _)| node).collect::<Vec<_>>(),
            order
        );

        let mut distance = vec![usize::MAX; 30];
        distance[0] = 0;
        let mut queue = VecDeque::from([0]);
        while let Some(node) = queue.pop_front() {
            for &next in &table.successors[node] {
                if distance[next] == usize::MAX {
                    distance[next] = distance[node] + 1;
                    queue.push_back(next);
                }
            }
        }
        for &(node, depth) in &discovered {
            assert_eq!(depth, distance[node], "seed {seed}: depth of n{node}");
        }
        for event in &events {
            if let Event::Tree(from, to) = event {
                assert_eq!(
                    distance[*to],
                    distance[*from] + 1,
                    "a tree edge descends one level"
                );
            }
        }
    }
}

#[test]
fn every_examined_edge_is_classified_once_and_a_back_edge_exists_exactly_when_a_cycle_does() {
    let mut cyclic_cases = 0;
    let mut acyclic_cases = 0;
    for seed in 1..=12 {
        // Sparse tables are often acyclic, which the cycle check needs as much as the dense ones.
        let edges = if seed % 2 == 0 { 70 } else { 22 };
        let table = Table::random(&mut Lcg(seed + 70), 30, edges);
        let graph = table.graph();
        let events = depth_first_stream(&graph, 0, TraversalDirection::Outgoing);

        let discovered: Vec<usize> = events
            .iter()
            .filter_map(|event| match event {
                Event::Discover(node, _) => Some(*node),
                _ => None,
            })
            .collect();
        let examined: usize = discovered
            .iter()
            .map(|&node| table.successors[node].len())
            .sum();
        let classified = events
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    Event::Tree(..) | Event::Back(..) | Event::ForwardOrCross(..)
                )
            })
            .count();
        let tree = events
            .iter()
            .filter(|event| matches!(event, Event::Tree(..)))
            .count();
        let back = events
            .iter()
            .filter(|event| matches!(event, Event::Back(..)))
            .count();
        assert_eq!(
            classified, examined,
            "seed {seed}: every out-edge of a discovered node, once"
        );
        assert_eq!(
            tree,
            discovered.len() - 1,
            "seed {seed}: a tree spans the discovered nodes"
        );

        let reached = reachable(
            &graph,
            [NodeId::from_index(0)],
            TraversalDirection::Outgoing,
        );
        assert_eq!(discovered.len(), reached.iter().filter(|&&yes| yes).count());
        let cyclic = is_cyclic(&table, &discovered);
        assert_eq!(back > 0, cyclic, "seed {seed}");
        if cyclic {
            cyclic_cases += 1;
        } else {
            acyclic_cases += 1;
        }
    }
    assert!(
        cyclic_cases > 0 && acyclic_cases > 0,
        "{cyclic_cases} cyclic, {acyclic_cases} acyclic"
    );
}

#[test]
fn discover_and_finish_nest_like_a_stack_and_edge_classes_agree_with_the_node_colors() {
    for seed in 1..=5 {
        let table = Table::random(&mut Lcg(seed + 80), 30, 60);
        let graph = table.graph();
        let events = depth_first_stream(&graph, 0, TraversalDirection::Outgoing);

        let mut on_path: Vec<usize> = Vec::new();
        let mut finished = [false; 30];
        for (position, event) in events.iter().enumerate() {
            match *event {
                Event::Discover(node, depth) => {
                    assert_eq!(
                        depth,
                        on_path.len(),
                        "depth is the number of open ancestors"
                    );
                    on_path.push(node);
                }
                Event::Finish(node) => {
                    assert_eq!(
                        on_path.pop(),
                        Some(node),
                        "a finish closes the innermost discovery"
                    );
                    finished[node] = true;
                }
                Event::Tree(from, to) => {
                    assert_eq!(on_path.last(), Some(&from));
                    assert_eq!(events[position + 1], Event::Discover(to, on_path.len()));
                }
                Event::Back(from, to) => {
                    assert_eq!(on_path.last(), Some(&from));
                    assert!(
                        on_path.contains(&to),
                        "a back edge closes a cycle through the open path"
                    );
                }
                Event::ForwardOrCross(from, to) => {
                    assert_eq!(on_path.last(), Some(&from));
                    assert!(
                        finished[to] && !on_path.contains(&to),
                        "the target is finished"
                    );
                }
                Event::NonTree(..) => unreachable!("a depth-first walk emits no non-tree edges"),
            }
        }
        assert!(on_path.is_empty(), "every discovery was finished");
    }
}

#[test]
fn only_nodes_reachable_from_the_start_produce_events() {
    // 0 -> 1; 2 -> 3 is a separate component.
    let table = Table {
        successors: vec![vec![1], vec![], vec![3], vec![]],
    };
    let graph = table.graph();
    for events in [
        depth_first_stream(&graph, 0, TraversalDirection::Outgoing),
        breadth_first_stream(&graph, 0, TraversalDirection::Outgoing),
    ] {
        for event in events {
            match event {
                Event::Discover(node, _) | Event::Finish(node) => assert!(node <= 1),
                Event::Tree(from, to)
                | Event::Back(from, to)
                | Event::ForwardOrCross(from, to)
                | Event::NonTree(from, to) => assert!(from <= 1 && to <= 1),
            }
        }
    }
}

#[test]
fn a_callback_can_break_either_walk_at_any_event_and_nothing_follows() {
    let table = Table::random(&mut Lcg(99), 30, 70);
    let graph = table.graph();
    let start = NodeId::from_index(0);
    let full_depth_first = depth_first_stream(&graph, 0, TraversalDirection::Outgoing);
    let full_breadth_first = breadth_first_stream(&graph, 0, TraversalDirection::Outgoing);

    for cut in [0, 1, full_depth_first.len() / 3, full_depth_first.len() - 1] {
        let mut seen = Vec::new();
        let outcome = depth_first_events(&graph, start, TraversalDirection::Outgoing, |event| {
            seen.push(from_depth_first(event));
            if seen.len() == cut + 1 {
                ControlFlow::Break(cut)
            } else {
                ControlFlow::Continue(())
            }
        });
        assert_eq!(outcome, Some(cut));
        assert_eq!(seen, full_depth_first[..=cut]);
    }
    for cut in [
        0,
        1,
        full_breadth_first.len() / 2,
        full_breadth_first.len() - 1,
    ] {
        let mut seen = Vec::new();
        let outcome = breadth_first_events(&graph, start, TraversalDirection::Outgoing, |event| {
            seen.push(from_breadth_first(event));
            if seen.len() == cut + 1 {
                ControlFlow::Break(cut)
            } else {
                ControlFlow::Continue(())
            }
        });
        assert_eq!(outcome, Some(cut));
        assert_eq!(seen, full_breadth_first[..=cut]);
    }
}

#[test]
#[should_panic(expected = "start node is out of range")]
fn a_start_outside_the_graph_panics_in_the_depth_first_walk() {
    let graph = Table::random(&mut Lcg(1), 4, 6).graph();
    let _ = depth_first_events::<_, ()>(
        &graph,
        NodeId::from_index(4),
        TraversalDirection::Outgoing,
        |_| ControlFlow::Continue(()),
    );
}

#[test]
#[should_panic(expected = "start node is out of range")]
fn a_start_outside_the_graph_panics_in_the_breadth_first_walk() {
    let graph = Table::random(&mut Lcg(1), 4, 6).graph();
    let _ = breadth_first_events::<_, ()>(
        &graph,
        NodeId::from_index(4),
        TraversalDirection::Outgoing,
        |_| ControlFlow::Continue(()),
    );
}

#[test]
fn the_dense_event_walks_run_over_a_very_deep_graph_without_recursion() {
    let table = chain(200_000);
    let graph = table.graph();
    let depth_first = depth_first_stream(&graph, 0, TraversalDirection::Outgoing);
    assert_eq!(
        depth_first.len(),
        3 * 200_000 - 1,
        "a discovery and a finish per node, a tree edge per link"
    );
    assert_eq!(depth_first.last(), Some(&Event::Finish(0)));
    let breadth_first = breadth_first_stream(&graph, 0, TraversalDirection::Outgoing);
    assert_eq!(
        breadth_first.len(),
        2 * 200_000 - 1,
        "a discovery per node, a tree edge per link"
    );
}
