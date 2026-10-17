//! `search`, `search_with_scratch` and `search_with_marks` over a dense `Graph`.

use std::ops::ControlFlow;

use cfglib::{
    Graph, NodeId, SearchConfig, SearchOrder, TraversalDirection, Visit, VisitedPolicy, search,
};

use crate::Lcg;
use crate::reference::{Table, Visits, Walk, keep_all, nodes_of, prune_multiples_of_five};

pub(crate) fn config(
    order: SearchOrder,
    direction: TraversalDirection,
    visited: VisitedPolicy,
    limit: Option<usize>,
) -> SearchConfig {
    let base = SearchConfig::new(order, direction).with_visited(visited);
    limit.map_or(base, |limit| base.with_max_depth(limit))
}

fn verdict(prune: &dyn Fn(usize) -> bool, node: NodeId) -> ControlFlow<(), Visit> {
    if prune(node.index()) {
        ControlFlow::Continue(Visit::Skip)
    } else {
        ControlFlow::Continue(Visit::Descend)
    }
}

fn run_search(
    graph: &Graph<(), ()>,
    seeds: &[usize],
    config: SearchConfig,
    prune: &dyn Fn(usize) -> bool,
) -> Visits {
    let mut visits = Vec::new();
    let outcome = search(
        graph,
        seeds.iter().map(|&seed| NodeId::from_index(seed)),
        config,
        |node, depth| {
            visits.push((node.index(), depth));
            verdict(prune, node)
        },
    );
    assert_eq!(outcome, None, "no visitor breaks");
    visits
}

fn assert_each_node_once(visits: &Visits) {
    let mut nodes = visits.iter().map(|&(node, _)| node).collect::<Vec<_>>();
    nodes.sort_unstable();
    let before = nodes.len();
    nodes.dedup();
    assert_eq!(
        nodes.len(),
        before,
        "a node was visited twice under global marks"
    );
}

#[test]
fn the_random_tables_make_walks_worth_comparing() {
    for seed in (1..=4).chain(11..=14) {
        let table = Table::random(&mut Lcg(seed), 40, 90);
        let visits = Walk::new(&table, None, &keep_all).breadth_first(&[3, 17, 30]);
        assert!(
            visits.len() > 10,
            "seed {seed} reaches only {} nodes",
            visits.len()
        );
        assert!(
            visits.iter().any(|&(_, depth)| depth >= 3),
            "seed {seed} is too shallow"
        );
    }
}

#[test]
fn depth_first_search_matches_a_recursive_reference_in_order_and_depth() {
    let prunes: [&dyn Fn(usize) -> bool; 2] = [&keep_all, &prune_multiples_of_five];
    for seed in 1..=4 {
        let table = Table::random(&mut Lcg(seed), 40, 90);
        let graph = table.graph();
        let backward = table.reversed();
        for (direction, adjacency) in [
            (TraversalDirection::Outgoing, &table),
            (TraversalDirection::Incoming, &backward),
        ] {
            for limit in [None, Some(0), Some(1), Some(3)] {
                for prune in prunes {
                    let seeds = [3, 17, 3, 30];
                    let expected = Walk::new(adjacency, limit, prune).depth_first(&seeds);
                    let order = SearchOrder::DepthFirst;
                    let actual = run_search(
                        &graph,
                        &seeds,
                        config(order, direction, VisitedPolicy::Global, limit),
                        prune,
                    );
                    assert_each_node_once(&actual);
                    assert_eq!(
                        actual, expected,
                        "seed {seed}, {direction:?}, limit {limit:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn breadth_first_search_matches_a_queue_reference_in_order_and_depth() {
    let prunes: [&dyn Fn(usize) -> bool; 2] = [&keep_all, &prune_multiples_of_five];
    for seed in 1..=4 {
        let table = Table::random(&mut Lcg(seed + 10), 40, 90);
        let graph = table.graph();
        let backward = table.reversed();
        for (direction, adjacency) in [
            (TraversalDirection::Outgoing, &table),
            (TraversalDirection::Incoming, &backward),
        ] {
            for limit in [None, Some(0), Some(1), Some(3)] {
                for prune in prunes {
                    let seeds = [3, 17, 3, 30];
                    let expected = Walk::new(adjacency, limit, prune).breadth_first(&seeds);
                    let order = SearchOrder::BreadthFirst;
                    let actual = run_search(
                        &graph,
                        &seeds,
                        config(order, direction, VisitedPolicy::Global, limit),
                        prune,
                    );
                    assert_each_node_once(&actual);
                    assert_eq!(
                        actual, expected,
                        "seed {seed}, {direction:?}, limit {limit:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn path_marks_visit_a_node_once_per_simple_route_like_a_brute_force_enumeration() {
    let prunes: [&dyn Fn(usize) -> bool; 2] = [&keep_all, &prune_multiples_of_five];
    for seed in 1..=4 {
        let table = Table::random(&mut Lcg(seed + 20), 14, 26);
        let graph = table.graph();
        let backward = table.reversed();
        for (direction, adjacency) in [
            (TraversalDirection::Outgoing, &table),
            (TraversalDirection::Incoming, &backward),
        ] {
            for prune in prunes {
                let seeds = [2, 9, 2];
                let expected = Walk::new(adjacency, Some(5), prune).depth_first_routes(&seeds);
                let actual = run_search(
                    &graph,
                    &seeds,
                    config(
                        SearchOrder::DepthFirst,
                        direction,
                        VisitedPolicy::Path,
                        Some(5),
                    ),
                    prune,
                );
                assert_eq!(actual, expected, "seed {seed}, {direction:?}");
                let distinct_nodes = {
                    let mut nodes = nodes_of(&actual);
                    nodes.sort_unstable();
                    nodes.dedup();
                    nodes.len()
                };
                assert!(
                    actual.len() > distinct_nodes,
                    "seed {seed} never revisits a node"
                );
            }
        }
    }
}

/// Two diamonds in series, so the sink is reached along 2 * 2 = 4 routes.
fn diamond_ladder() -> Table {
    Table {
        successors: vec![
            vec![1, 2],
            vec![3],
            vec![3],
            vec![4, 5],
            vec![6],
            vec![6],
            vec![],
        ],
    }
}

#[test]
fn a_shared_node_is_reached_once_globally_and_once_per_route_under_path_marks() {
    let table = diamond_ladder();
    let graph = table.graph();
    let order = SearchOrder::DepthFirst;
    let outgoing = TraversalDirection::Outgoing;

    let global = run_search(
        &graph,
        &[0],
        config(order, outgoing, VisitedPolicy::Global, None),
        &keep_all,
    );
    assert_eq!(global.iter().filter(|&&(node, _)| node == 6).count(), 1);
    assert_eq!(global.len(), 7);

    let routes = run_search(
        &graph,
        &[0],
        config(order, outgoing, VisitedPolicy::Path, None),
        &keep_all,
    );
    let reached = |node: usize| {
        routes
            .iter()
            .filter(|&&(visited, _)| visited == node)
            .count()
    };
    assert_eq!(
        (0..7).map(reached).collect::<Vec<_>>(),
        [1, 1, 1, 2, 2, 2, 4],
        "visits per node equal the simple routes that reach it"
    );
    assert_eq!(routes.len(), 13);
    assert_eq!(
        routes
            .iter()
            .filter(|&&(node, _)| node == 6)
            .map(|&(_, depth)| depth)
            .collect::<Vec<_>>(),
        [4, 4, 4, 4]
    );
}

#[test]
fn the_path_cycle_guard_stops_a_route_at_a_node_already_on_it() {
    // 0 -> 1 -> 2 -> 0 is a cycle; every route ends where it would close.
    let table = Table {
        successors: vec![vec![1], vec![2], vec![0, 3], vec![]],
    };
    let graph = table.graph();
    let visits = run_search(
        &graph,
        &[0],
        config(
            SearchOrder::DepthFirst,
            TraversalDirection::Outgoing,
            VisitedPolicy::Path,
            None,
        ),
        &keep_all,
    );
    assert_eq!(visits, [(0, 0), (1, 1), (2, 2), (3, 3)]);
}

#[test]
fn duplicate_seeds_are_visited_once_under_global_marks_and_again_under_path_marks() {
    let table = Table {
        successors: vec![vec![1], vec![]],
    };
    let graph = table.graph();
    let outgoing = TraversalDirection::Outgoing;
    let order = SearchOrder::DepthFirst;

    let global = run_search(
        &graph,
        &[0, 0],
        config(order, outgoing, VisitedPolicy::Global, None),
        &keep_all,
    );
    assert_eq!(global, [(0, 0), (1, 1)]);
    let path = run_search(
        &graph,
        &[0, 0],
        config(order, outgoing, VisitedPolicy::Path, None),
        &keep_all,
    );
    assert_eq!(path, [(0, 0), (1, 1), (0, 0), (1, 1)]);
    let breadth = run_search(
        &graph,
        &[0, 1, 0],
        config(
            SearchOrder::BreadthFirst,
            outgoing,
            VisitedPolicy::Global,
            None,
        ),
        &keep_all,
    );
    assert_eq!(
        breadth,
        [(0, 0), (1, 0)],
        "both seeds are at depth zero, each once"
    );
}

#[test]
fn a_break_ends_the_walk_with_its_value_and_visits_nothing_further() {
    let table = Table::random(&mut Lcg(40), 40, 90);
    let graph = table.graph();
    for order in [SearchOrder::DepthFirst, SearchOrder::BreadthFirst] {
        let full = match order {
            SearchOrder::DepthFirst => Walk::new(&table, None, &keep_all).depth_first(&[3]),
            SearchOrder::BreadthFirst => Walk::new(&table, None, &keep_all).breadth_first(&[3]),
        };
        let target = full[full.len() / 2].0;
        let position = full
            .iter()
            .position(|&(node, _)| node == target)
            .expect("the target was visited");

        let mut visits = Vec::new();
        let outcome = search(
            &graph,
            [NodeId::from_index(3)],
            config(
                order,
                TraversalDirection::Outgoing,
                VisitedPolicy::Global,
                None,
            ),
            |node, depth| {
                visits.push((node.index(), depth));
                if node.index() == target {
                    ControlFlow::Break(depth)
                } else {
                    ControlFlow::Continue(Visit::Descend)
                }
            },
        );
        assert_eq!(outcome, Some(full[position].1), "{order:?}");
        assert_eq!(
            visits,
            full[..=position],
            "{order:?}: nothing after the break"
        );
    }
}

#[test]
fn the_first_match_depends_on_the_discipline() {
    // 3 is the only leaf: the long route 0 -> 1 -> 2 -> 3 is listed first, the short one 0 -> 3 second.
    let table = Table {
        successors: vec![vec![1, 3], vec![2], vec![3], vec![]],
    };
    let graph = table.graph();
    let first_leaf = |order| {
        search(
            &graph,
            [NodeId::from_index(0)],
            SearchConfig::new(order, TraversalDirection::Outgoing),
            |node, depth| {
                if table.successors[node.index()].is_empty() {
                    ControlFlow::Break((node.index(), depth))
                } else {
                    ControlFlow::Continue(Visit::Descend)
                }
            },
        )
    };
    assert_eq!(
        first_leaf(SearchOrder::DepthFirst),
        Some((3, 3)),
        "the long route is found first"
    );
    assert_eq!(
        first_leaf(SearchOrder::BreadthFirst),
        Some((3, 1)),
        "the shortest route is found first"
    );
}

#[test]
fn a_skipped_node_counts_as_visited_but_its_successors_are_not_discovered_through_it() {
    // 0 -> {1, 2}, 1 -> 3, 2 -> 3, 3 -> 4
    let table = Table {
        successors: vec![vec![1, 2], vec![3], vec![3], vec![4], vec![]],
    };
    let graph = table.graph();
    let outgoing = TraversalDirection::Outgoing;
    for order in [SearchOrder::DepthFirst, SearchOrder::BreadthFirst] {
        let walk = |prune: &dyn Fn(usize) -> bool| {
            nodes_of(&run_search(
                &graph,
                &[0],
                config(order, outgoing, VisitedPolicy::Global, None),
                prune,
            ))
        };
        assert_eq!(
            walk(&|node| node == 1),
            [0, 1, 2, 3, 4],
            "3 is still reached through 2"
        );
        assert_eq!(
            walk(&|node| node == 1 || node == 2),
            [0, 1, 2],
            "nothing past both pruned nodes"
        );
        assert_eq!(
            walk(&|node| node == 0),
            [0],
            "pruning the seed visits the seed alone"
        );
    }
}

#[test]
fn max_depth_bounds_expansion_not_visiting() {
    let chain = Table {
        successors: vec![vec![1], vec![2], vec![3], vec![]],
    };
    let graph = chain.graph();
    let outgoing = TraversalDirection::Outgoing;
    for order in [SearchOrder::DepthFirst, SearchOrder::BreadthFirst] {
        let walk = |limit| {
            run_search(
                &graph,
                &[0],
                config(order, outgoing, VisitedPolicy::Global, Some(limit)),
                &keep_all,
            )
        };
        assert_eq!(
            walk(0),
            [(0, 0)],
            "{order:?}: a zero bound visits the seeds alone"
        );
        assert_eq!(
            walk(2),
            [(0, 0), (1, 1), (2, 2)],
            "{order:?}: the node at the bound is visited"
        );
        assert_eq!(walk(9).len(), 4);
    }
}

#[test]
fn depth_first_depth_is_the_route_taken_not_the_shortest_one() {
    // 0 -> {1, 3}, 1 -> 2 -> 3: the first route reaches 3 at depth 3.
    let table = Table {
        successors: vec![vec![1, 3], vec![2], vec![3], vec![]],
    };
    let graph = table.graph();
    let outgoing = TraversalDirection::Outgoing;
    let unbounded = |order| {
        run_search(
            &graph,
            &[0],
            config(order, outgoing, VisitedPolicy::Global, None),
            &keep_all,
        )
    };
    assert_eq!(
        unbounded(SearchOrder::DepthFirst),
        [(0, 0), (1, 1), (2, 2), (3, 3)]
    );
    assert_eq!(
        unbounded(SearchOrder::BreadthFirst),
        [(0, 0), (1, 1), (3, 1), (2, 2)]
    );

    let bounded = run_search(
        &graph,
        &[0],
        config(
            SearchOrder::DepthFirst,
            outgoing,
            VisitedPolicy::Global,
            Some(2),
        ),
        &keep_all,
    );
    assert_eq!(
        bounded,
        [(0, 0), (1, 1), (2, 2), (3, 1)],
        "2 sits at the bound, so 3 is reached by the short route"
    );
}

#[test]
fn backward_search_follows_predecessors() {
    let chain = Table {
        successors: vec![vec![1], vec![2], vec![]],
    };
    let graph = chain.graph();
    let visits = run_search(
        &graph,
        &[2],
        config(
            SearchOrder::DepthFirst,
            TraversalDirection::Incoming,
            VisitedPolicy::Global,
            None,
        ),
        &keep_all,
    );
    assert_eq!(visits, [(2, 0), (1, 1), (0, 2)]);
}

#[test]
fn a_store_with_removed_nodes_walks_like_its_reduced_table_before_and_after_compaction() {
    let table = Table::random(&mut Lcg(55), 40, 100);
    let removed = [7, 8, 21, 22];
    let reduced = table.without(&removed);
    let mut graph = table.graph();
    for &node in &removed {
        assert!(graph.remove_node(NodeId::from_index(node)));
    }

    for order in [SearchOrder::DepthFirst, SearchOrder::BreadthFirst] {
        let expected = match order {
            SearchOrder::DepthFirst => {
                Walk::new(&reduced, Some(4), &keep_all).depth_first(&[0, 30])
            }
            SearchOrder::BreadthFirst => {
                Walk::new(&reduced, Some(4), &keep_all).breadth_first(&[0, 30])
            }
        };
        let walk = |graph: &Graph<(), ()>, seeds: &[usize]| {
            run_search(
                graph,
                seeds,
                config(
                    order,
                    TraversalDirection::Outgoing,
                    VisitedPolicy::Global,
                    Some(4),
                ),
                &keep_all,
            )
        };
        assert_eq!(
            walk(&graph, &[0, 30]),
            expected,
            "{order:?} before compaction"
        );
        assert!(expected.iter().all(|(node, _)| !removed.contains(node)));

        let mut compacted = graph.clone();
        let renumbering = compacted.compact();
        let mapped_seeds: Vec<usize> = [0, 30]
            .iter()
            .map(|&seed| {
                renumbering
                    .node(NodeId::from_index(seed))
                    .expect("live")
                    .index()
            })
            .collect();
        let mapped_expected: Visits = expected
            .iter()
            .map(|&(node, depth)| {
                (
                    renumbering
                        .node(NodeId::from_index(node))
                        .expect("live")
                        .index(),
                    depth,
                )
            })
            .collect();
        assert_eq!(
            walk(&compacted, &mapped_seeds),
            mapped_expected,
            "{order:?} after compaction"
        );
    }
}
