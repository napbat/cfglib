//! `open_breadth_first_events` and `open_depth_first_events` over lazily generated spaces and
//! over tables, against the dense event walks.

use std::collections::BTreeMap;
use std::ops::ControlFlow;

use cfglib::{
    BfsEvent, DfsEvent, Graph, NodeId, OpenBfsConfig, OpenBfsEvent, OpenDfsConfig, OpenDfsEvent,
    TraversalDirection, Visit, VisitedPolicy, breadth_first_events, depth_first_events,
    open_breadth_first_events, open_depth_first_events,
};

use crate::Lcg;
use crate::reference::{Table, Walk, keep_all, table_successors};
use crate::spaces::{collatz_predecessors, collatz_steps, grid_successors};

/// What an event stream says, in plain values, comparable across the dense and the open walk.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Step {
    Enter(usize, usize),
    Leave(usize),
    Refuse(usize, usize),
}

fn dense_depth_first_steps(graph: &Graph<(), ()>, start: usize) -> Vec<Step> {
    let mut depth = vec![0; graph.node_bound()];
    let mut steps = Vec::new();
    let outcome = depth_first_events::<_, ()>(
        graph,
        NodeId::from_index(start),
        TraversalDirection::Outgoing,
        |event| {
            match event {
                DfsEvent::Discover(node, at) => {
                    depth[node.index()] = at;
                    steps.push(Step::Enter(node.index(), at));
                }
                DfsEvent::Finish(node) => steps.push(Step::Leave(node.index())),
                DfsEvent::BackEdge(from, to) | DfsEvent::ForwardOrCross(from, to) => {
                    steps.push(Step::Refuse(to.index(), depth[from.index()] + 1));
                }
                DfsEvent::TreeEdge(..) => {}
            }
            ControlFlow::Continue(())
        },
    );
    assert_eq!(outcome, None);
    steps
}

fn dense_breadth_first_steps(graph: &Graph<(), ()>, start: usize) -> Vec<Step> {
    let mut depth = vec![0; graph.node_bound()];
    let mut steps = Vec::new();
    let outcome = breadth_first_events::<_, ()>(
        graph,
        NodeId::from_index(start),
        TraversalDirection::Outgoing,
        |event| {
            match event {
                BfsEvent::Discover(node, at) => {
                    depth[node.index()] = at;
                    steps.push(Step::Enter(node.index(), at));
                }
                BfsEvent::NonTreeEdge(from, to) => {
                    steps.push(Step::Refuse(to.index(), depth[from.index()] + 1));
                }
                BfsEvent::TreeEdge(..) => {}
            }
            ControlFlow::Continue(())
        },
    );
    assert_eq!(outcome, None);
    steps
}

fn open_depth_first_steps(table: &Table, config: OpenDfsConfig, seeds: &[usize]) -> Vec<Step> {
    let mut steps = Vec::new();
    let outcome = open_depth_first_events::<_, ()>(
        seeds.iter().copied(),
        config,
        table_successors(table),
        |event| {
            steps.push(match event {
                OpenDfsEvent::Discover(node, depth) => Step::Enter(*node, depth),
                OpenDfsEvent::Finish(node) => Step::Leave(*node),
                OpenDfsEvent::Refused(node, depth) => Step::Refuse(*node, depth),
            });
            ControlFlow::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    steps
}

fn open_breadth_first_steps(table: &Table, config: OpenBfsConfig, seeds: &[usize]) -> Vec<Step> {
    let mut steps = Vec::new();
    let outcome = open_breadth_first_events::<_, ()>(
        seeds.iter().copied(),
        config,
        table_successors(table),
        |event| {
            steps.push(match event {
                OpenBfsEvent::Discover(node, depth) => Step::Enter(*node, depth),
                OpenBfsEvent::Refused(node, depth) => Step::Refuse(*node, depth),
            });
            ControlFlow::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    steps
}

#[test]
fn the_open_breadth_first_events_equal_the_dense_events_with_non_tree_edges_as_refusals() {
    for seed in 1..=5 {
        let table = Table::random(&mut Lcg(seed + 100), 30, 55);
        let graph = table.graph();
        assert_eq!(
            open_breadth_first_steps(&table, OpenBfsConfig::new(), &[2]),
            dense_breadth_first_steps(&graph, 2),
            "seed {seed}"
        );
    }
}

#[test]
fn the_open_depth_first_events_equal_the_dense_events_with_back_and_cross_edges_as_refusals() {
    for seed in 1..=5 {
        let table = Table::random(&mut Lcg(seed + 110), 30, 55);
        let graph = table.graph();
        assert_eq!(
            open_depth_first_steps(&table, OpenDfsConfig::new(VisitedPolicy::Global), &[2]),
            dense_depth_first_steps(&graph, 2),
            "seed {seed}"
        );
    }
}

#[test]
fn a_breadth_first_event_walk_over_inverse_collatz_finds_exactly_the_numbers_within_the_bound() {
    let mut discovered: BTreeMap<u64, usize> = BTreeMap::new();
    let mut refused = Vec::new();
    let outcome = open_breadth_first_events::<_, ()>(
        [1_u64],
        OpenBfsConfig::new().with_max_depth(10),
        collatz_predecessors,
        |event| {
            match event {
                OpenBfsEvent::Discover(&node, depth) => {
                    assert!(
                        discovered.insert(node, depth).is_none(),
                        "{node} was discovered twice"
                    );
                }
                OpenBfsEvent::Refused(&node, depth) => refused.push((node, depth)),
            }
            ControlFlow::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);

    // A number is at depth d exactly when d Collatz steps lead from it to 1, and every number
    // within ten inverse steps of 1 is at most 2^10.
    let expected: BTreeMap<u64, usize> = (1..=1024)
        .filter(|&number| collatz_steps(number) <= 10)
        .map(|number| (number, collatz_steps(number)))
        .collect();
    assert_eq!(discovered, expected);
    assert_eq!(
        refused,
        [(1, 3)],
        "the only second route is the cycle 1, 4, 2, 1"
    );
    let mut per_depth = [0_usize; 11];
    for &depth in discovered.values() {
        per_depth[depth] += 1;
    }
    assert_eq!(
        per_depth,
        [1, 1, 1, 1, 1, 2, 2, 4, 4, 6, 6],
        "OEIS A005186: how many numbers need exactly 0, 1, ..., 10 Collatz steps"
    );
}

#[test]
fn a_depth_first_event_walk_over_inverse_collatz_finds_the_same_set_under_either_mark_policy() {
    let expected: BTreeMap<u64, usize> = (1..=1024)
        .filter(|&number| collatz_steps(number) <= 10)
        .map(|number| (number, collatz_steps(number)))
        .collect();
    for policy in [VisitedPolicy::Global, VisitedPolicy::Path] {
        let mut discovered: BTreeMap<u64, usize> = BTreeMap::new();
        let (mut entered, mut left) = (0, 0);
        let outcome = open_depth_first_events::<_, ()>(
            [1_u64],
            OpenDfsConfig::new(policy).with_max_depth(10),
            collatz_predecessors,
            |event| {
                match event {
                    OpenDfsEvent::Discover(&node, depth) => {
                        entered += 1;
                        discovered.insert(node, depth);
                    }
                    OpenDfsEvent::Finish(_) => left += 1,
                    OpenDfsEvent::Refused(&node, _) => {
                        assert_eq!(node, 1, "{policy:?}: only the cycle refuses");
                    }
                }
                ControlFlow::Continue(Visit::Descend)
            },
        );
        assert_eq!(outcome, None);
        assert_eq!(discovered, expected, "{policy:?}");
        assert_eq!(entered, left, "{policy:?}: every discovery finishes");
    }
}

#[test]
fn grid_event_counts_follow_from_its_edge_and_route_counts() {
    let grid = grid_successors;
    for policy in [VisitedPolicy::Global, VisitedPolicy::Path] {
        let (mut entered, mut left, mut refused) = (0_u64, 0_u64, 0_u64);
        let outcome = open_depth_first_events::<_, ()>(
            [(0_u32, 0_u32)],
            OpenDfsConfig::new(policy),
            grid,
            |event| {
                match event {
                    OpenDfsEvent::Discover(..) => entered += 1,
                    OpenDfsEvent::Finish(_) => left += 1,
                    OpenDfsEvent::Refused(..) => refused += 1,
                }
                ControlFlow::Continue(Visit::Descend)
            },
        );
        assert_eq!(outcome, None);
        assert_eq!(entered, left);
        match policy {
            VisitedPolicy::Global => {
                // 2 * 6 * 7 grid edges, 48 of which build the spanning tree.
                assert_eq!((entered, refused), (49, 84 - 48));
            }
            VisitedPolicy::Path => {
                assert_eq!((entered, refused), (3431, 0), "one discovery per route");
            }
        }
    }

    let (mut discovered, mut refused) = (0, 0);
    let outcome = open_breadth_first_events::<_, ()>(
        [(0_u32, 0_u32)],
        OpenBfsConfig::new(),
        grid_successors,
        |event| {
            match event {
                OpenBfsEvent::Discover(..) => discovered += 1,
                OpenBfsEvent::Refused(..) => refused += 1,
            }
            ControlFlow::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    assert_eq!((discovered, refused), (49, 36));
}

#[test]
fn path_marks_refuse_exactly_the_nodes_already_on_the_open_path_and_nest_discoveries() {
    for seed in 1..=5 {
        let table = Table::random(&mut Lcg(seed + 120), 12, 24);
        let steps = open_depth_first_steps(
            &table,
            OpenDfsConfig::new(VisitedPolicy::Path).with_max_depth(6),
            &[0, 5],
        );

        let mut path: Vec<usize> = Vec::new();
        let mut discoveries = Vec::new();
        for step in &steps {
            match *step {
                Step::Enter(node, depth) => {
                    assert_eq!(depth, path.len());
                    assert!(!path.contains(&node), "an open node is never entered again");
                    path.push(node);
                    discoveries.push((node, depth));
                }
                Step::Leave(node) => assert_eq!(path.pop(), Some(node)),
                Step::Refuse(node, depth) => {
                    assert!(
                        path.contains(&node),
                        "seed {seed}: refused n{node} is not on the path"
                    );
                    assert_eq!(depth, path.len());
                }
            }
        }
        assert_eq!(path.len(), 0);
        assert_eq!(
            discoveries,
            Walk::new(&table, Some(6), &keep_all).depth_first_routes(&[0, 5]),
            "seed {seed}"
        );
    }
}

#[test]
fn open_walk_hand_derived_diamond_streams() {
    // 0 -> {1, 2}, 1 -> 3, 2 -> 3, 3 -> 4
    let table = Table {
        successors: vec![vec![1, 2], vec![3], vec![3], vec![4], vec![]],
    };
    assert_eq!(
        open_breadth_first_steps(&table, OpenBfsConfig::new(), &[0]),
        [
            Step::Enter(0, 0),
            Step::Enter(1, 1),
            Step::Enter(2, 1),
            Step::Enter(3, 2),
            Step::Refuse(3, 2),
            Step::Enter(4, 3),
        ]
    );
    assert_eq!(
        open_depth_first_steps(&table, OpenDfsConfig::new(VisitedPolicy::Global), &[0]),
        [
            Step::Enter(0, 0),
            Step::Enter(1, 1),
            Step::Enter(3, 2),
            Step::Enter(4, 3),
            Step::Leave(4),
            Step::Leave(3),
            Step::Leave(1),
            Step::Enter(2, 1),
            Step::Refuse(3, 2),
            Step::Leave(2),
            Step::Leave(0),
        ]
    );
}

#[test]
fn a_depth_first_skip_finishes_the_pruned_node_at_once_and_never_probes_its_successors() {
    let table = Table {
        successors: vec![vec![1, 2], vec![3], vec![], vec![]],
    };
    let mut probed = Vec::new();
    let mut steps = Vec::new();
    let outcome = open_depth_first_events::<_, ()>(
        [0_usize],
        OpenDfsConfig::new(VisitedPolicy::Global),
        |node: &usize, out: &mut Vec<usize>| {
            probed.push(*node);
            out.extend_from_slice(&table.successors[*node]);
        },
        |event| match event {
            OpenDfsEvent::Discover(&node, depth) => {
                steps.push(Step::Enter(node, depth));
                ControlFlow::Continue(if node == 1 {
                    Visit::Skip
                } else {
                    Visit::Descend
                })
            }
            OpenDfsEvent::Finish(&node) => {
                steps.push(Step::Leave(node));
                ControlFlow::Continue(Visit::Descend)
            }
            OpenDfsEvent::Refused(&node, depth) => {
                steps.push(Step::Refuse(node, depth));
                ControlFlow::Continue(Visit::Descend)
            }
        },
    );
    assert_eq!(outcome, None);
    assert_eq!(
        steps,
        [
            Step::Enter(0, 0),
            Step::Enter(1, 1),
            Step::Leave(1),
            Step::Enter(2, 1),
            Step::Leave(2),
            Step::Leave(0),
        ]
    );
    assert_eq!(
        probed,
        [0, 2],
        "node 1 was pruned, so node 3 was never looked for"
    );
}

#[test]
fn a_depth_bound_discovers_and_finishes_the_node_at_the_bound_without_its_successors() {
    let chain = Table {
        successors: vec![vec![1], vec![2], vec![3], vec![]],
    };
    assert_eq!(
        open_depth_first_steps(
            &chain,
            OpenDfsConfig::new(VisitedPolicy::Global).with_max_depth(2),
            &[0]
        ),
        [
            Step::Enter(0, 0),
            Step::Enter(1, 1),
            Step::Enter(2, 2),
            Step::Leave(2),
            Step::Leave(1),
            Step::Leave(0),
        ]
    );
    assert_eq!(
        open_breadth_first_steps(&chain, OpenBfsConfig::new().with_max_depth(0), &[0]),
        [Step::Enter(0, 0)],
        "a zero bound discovers the seeds alone"
    );
}

#[test]
fn repeated_seeds_are_refused_globally_and_walked_again_under_path_marks() {
    let table = Table {
        successors: vec![vec![1], vec![]],
    };
    assert_eq!(
        open_depth_first_steps(&table, OpenDfsConfig::new(VisitedPolicy::Global), &[0, 0]),
        [
            Step::Enter(0, 0),
            Step::Enter(1, 1),
            Step::Leave(1),
            Step::Leave(0),
            Step::Refuse(0, 0),
        ]
    );
    assert_eq!(
        open_depth_first_steps(&table, OpenDfsConfig::new(VisitedPolicy::Path), &[0, 0]),
        [
            Step::Enter(0, 0),
            Step::Enter(1, 1),
            Step::Leave(1),
            Step::Leave(0),
            Step::Enter(0, 0),
            Step::Enter(1, 1),
            Step::Leave(1),
            Step::Leave(0),
        ]
    );
    assert_eq!(
        open_breadth_first_steps(&table, OpenBfsConfig::new(), &[0, 0]),
        [Step::Enter(0, 0), Step::Refuse(0, 0), Step::Enter(1, 1)],
        "a duplicate seed is refused at depth zero"
    );
}

#[test]
fn a_breadth_first_skip_keeps_the_mark_and_a_refusal_verdict_is_ignored() {
    // 0 -> {1, 2}, 1 -> 3, 2 -> 3, 3 -> 4
    let table = Table {
        successors: vec![vec![1, 2], vec![3], vec![3], vec![4], vec![]],
    };
    let mut steps = Vec::new();
    let outcome = open_breadth_first_events::<_, ()>(
        [0_usize],
        OpenBfsConfig::new(),
        table_successors(&table),
        |event| match event {
            OpenBfsEvent::Discover(&node, depth) => {
                steps.push(Step::Enter(node, depth));
                ControlFlow::Continue(if node == 1 {
                    Visit::Skip
                } else {
                    Visit::Descend
                })
            }
            OpenBfsEvent::Refused(&node, depth) => {
                steps.push(Step::Refuse(node, depth));
                ControlFlow::Continue(Visit::Skip)
            }
        },
    );
    assert_eq!(outcome, None);
    assert_eq!(
        steps,
        [
            Step::Enter(0, 0),
            Step::Enter(1, 1),
            Step::Enter(2, 1),
            Step::Enter(3, 2),
            Step::Enter(4, 3)
        ],
        "node 1 is not expanded, so 3 is first reached through 2 and never refused"
    );

    let mut refused = Vec::new();
    let outcome = open_breadth_first_events::<_, ()>(
        [0_usize],
        OpenBfsConfig::new(),
        table_successors(&table),
        |event| {
            if let OpenBfsEvent::Refused(&node, depth) = event {
                refused.push((node, depth));
                return ControlFlow::Continue(Visit::Skip);
            }
            ControlFlow::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    assert_eq!(
        refused,
        [(3, 2)],
        "a Skip verdict on a refusal changes nothing"
    );
}

#[test]
fn a_break_in_either_event_walk_returns_its_value_and_leaves_open_frames_unfinished() {
    let table = Table::random(&mut Lcg(140), 20, 40);
    let full_depth =
        open_depth_first_steps(&table, OpenDfsConfig::new(VisitedPolicy::Global), &[0]);
    let full_breadth = open_breadth_first_steps(&table, OpenBfsConfig::new(), &[0]);
    assert!(full_depth.len() > 8 && full_breadth.len() > 4);

    let cut = 4;
    let mut seen = Vec::new();
    let outcome = open_depth_first_events(
        [0_usize],
        OpenDfsConfig::new(VisitedPolicy::Global),
        table_successors(&table),
        |event| {
            seen.push(match event {
                OpenDfsEvent::Discover(node, depth) => Step::Enter(*node, depth),
                OpenDfsEvent::Finish(node) => Step::Leave(*node),
                OpenDfsEvent::Refused(node, depth) => Step::Refuse(*node, depth),
            });
            if seen.len() == cut + 1 {
                ControlFlow::Break(seen.len())
            } else {
                ControlFlow::Continue(Visit::Descend)
            }
        },
    );
    assert_eq!(outcome, Some(cut + 1));
    assert_eq!(seen, full_depth[..=cut]);

    let mut seen = Vec::new();
    let outcome = open_breadth_first_events(
        [0_usize],
        OpenBfsConfig::new(),
        table_successors(&table),
        |event| {
            seen.push(match event {
                OpenBfsEvent::Discover(node, depth) => Step::Enter(*node, depth),
                OpenBfsEvent::Refused(node, depth) => Step::Refuse(*node, depth),
            });
            if seen.len() == cut + 1 {
                ControlFlow::Break(seen.len())
            } else {
                ControlFlow::Continue(Visit::Descend)
            }
        },
    );
    assert_eq!(outcome, Some(cut + 1));
    assert_eq!(seen, full_breadth[..=cut]);
}

#[test]
fn a_depth_first_event_walk_over_a_very_deep_space_costs_heap_not_stack() {
    const DEPTH: u64 = 200_000;
    let chain = |node: &u64, out: &mut Vec<u64>| {
        if *node < DEPTH {
            out.push(*node + 1);
        }
    };
    let (mut entered, mut left) = (0_u64, 0_u64);
    let outcome = open_depth_first_events::<_, ()>(
        [0_u64],
        OpenDfsConfig::new(VisitedPolicy::Global),
        chain,
        |event| {
            match event {
                OpenDfsEvent::Discover(..) => entered += 1,
                OpenDfsEvent::Finish(_) => left += 1,
                OpenDfsEvent::Refused(..) => unreachable!("a chain has no second route"),
            }
            ControlFlow::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    assert_eq!((entered, left), (DEPTH + 1, DEPTH + 1));
}
