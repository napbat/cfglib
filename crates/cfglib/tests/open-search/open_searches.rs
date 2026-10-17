//! `open_search` over lazily generated spaces and over tables.

use std::ops::ControlFlow;

use cfglib::{OpenSearchConfig, SearchOrder, Visit, VisitedPolicy, open_search};

use crate::Lcg;
use crate::reference::{Table, Visits, Walk, keep_all, table_successors};
use crate::spaces::{GRID_SIDE, binomial, collatz_successors, collatz_trajectory, grid_successors};

fn open_visits(
    seeds: &[usize],
    config: OpenSearchConfig,
    table: &Table,
    prune: &dyn Fn(usize) -> bool,
) -> Visits {
    let mut visits = Vec::new();
    let outcome = open_search(
        seeds.iter().copied(),
        config,
        table_successors(table),
        |node, depth| {
            visits.push((*node, depth));
            if prune(*node) {
                ControlFlow::<(), Visit>::Continue(Visit::Skip)
            } else {
                ControlFlow::Continue(Visit::Descend)
            }
        },
    );
    assert_eq!(outcome, None);
    visits
}

fn grid_visits(
    order: SearchOrder,
    visited: VisitedPolicy,
    limit: Option<usize>,
) -> Vec<((u32, u32), usize)> {
    let base = OpenSearchConfig::new(order).with_visited(visited);
    let config = limit.map_or(base, |limit| base.with_max_depth(limit));
    let mut visits = Vec::new();
    let outcome = open_search([(0_u32, 0_u32)], config, grid_successors, |node, depth| {
        visits.push((*node, depth));
        ControlFlow::<(), Visit>::Continue(Visit::Descend)
    });
    assert_eq!(outcome, None);
    visits
}

#[test]
fn a_chase_walks_the_collatz_sequence_of_27_without_any_graph() {
    let trajectory = collatz_trajectory(27);
    assert_eq!(trajectory.len(), 112, "27 takes 111 steps to reach 1");
    assert_eq!(trajectory.iter().max(), Some(&9232));
    let expected: Vec<(u64, usize)> = trajectory.iter().copied().zip(0..).collect();

    for order in [SearchOrder::DepthFirst, SearchOrder::BreadthFirst] {
        let mut visits = Vec::new();
        let outcome = open_search(
            [27_u64],
            OpenSearchConfig::new(order),
            collatz_successors,
            |node, depth| {
                visits.push((*node, depth));
                ControlFlow::<(), Visit>::Continue(Visit::Descend)
            },
        );
        assert_eq!(outcome, None);
        assert_eq!(
            visits, expected,
            "{order:?}: the walk stops where 1 leads back to the visited 4"
        );
    }
}

#[test]
fn skip_stops_expansion_and_the_pruned_node_is_never_probed_for_successors() {
    let trajectory = collatz_trajectory(27);
    let first_large = trajectory
        .iter()
        .position(|&node| node > 1000)
        .expect("27 climbs above 1000");

    let mut probed = 0;
    let mut visited = Vec::new();
    let outcome = open_search(
        [27_u64],
        OpenSearchConfig::new(SearchOrder::DepthFirst),
        |node: &u64, out: &mut Vec<u64>| {
            probed += 1;
            collatz_successors(node, out);
        },
        |node, _| {
            visited.push(*node);
            if *node > 1000 {
                ControlFlow::<(), Visit>::Continue(Visit::Skip)
            } else {
                ControlFlow::Continue(Visit::Descend)
            }
        },
    );
    assert_eq!(outcome, None);
    assert_eq!(visited, trajectory[..=first_large]);
    assert_eq!(
        probed, first_large,
        "every visited node but the pruned last one was probed"
    );
}

#[test]
fn a_break_returns_its_value_at_the_first_match_and_nothing_is_visited_after_it() {
    let trajectory = collatz_trajectory(27);
    let peak_step = trajectory
        .iter()
        .position(|&node| node == 9232)
        .expect("the peak is on the path");

    let mut visited = 0;
    let found = open_search(
        [27_u64],
        OpenSearchConfig::new(SearchOrder::BreadthFirst),
        collatz_successors,
        |node, depth| {
            visited += 1;
            if *node == 9232 {
                ControlFlow::Break(depth)
            } else {
                ControlFlow::Continue(Visit::Descend)
            }
        },
    );
    assert_eq!(found, Some(peak_step));
    assert_eq!(visited, peak_step + 1);
}

#[test]
fn max_depth_bounds_a_chase_by_hops_from_the_seed() {
    let trajectory = collatz_trajectory(27);
    for limit in [0, 1, 10, 50] {
        let config = OpenSearchConfig::new(SearchOrder::DepthFirst).with_max_depth(limit);
        let mut visited = Vec::new();
        let outcome = open_search([27_u64], config, collatz_successors, |node, _| {
            visited.push(*node);
            ControlFlow::<(), Visit>::Continue(Visit::Descend)
        });
        assert_eq!(outcome, None);
        assert_eq!(
            visited,
            trajectory[..=limit],
            "limit {limit}: the node at the bound is visited"
        );
    }
}

#[test]
fn route_counts_through_a_grid_are_binomial_coefficients() {
    let side = u64::from(GRID_SIDE);
    let goal = (GRID_SIDE, GRID_SIDE);
    let total: u64 = (0..=side)
        .flat_map(|x| (0..=side).map(move |y| binomial(x + y, x)))
        .sum();
    assert_eq!(
        (binomial(2 * side, side), total),
        (924, 3431),
        "closed forms: C(12, 6) and C(14, 7) - 1"
    );

    let routes = grid_visits(SearchOrder::DepthFirst, VisitedPolicy::Path, None);
    assert_eq!(
        u64::try_from(routes.len()).expect("fits"),
        total,
        "one visit per simple route"
    );
    let to_goal = routes.iter().filter(|&&(node, _)| node == goal).count();
    assert_eq!(
        u64::try_from(to_goal).expect("fits"),
        binomial(2 * side, side)
    );

    let nodes = grid_visits(SearchOrder::DepthFirst, VisitedPolicy::Global, None);
    assert_eq!(nodes.len(), 49, "each of the 7 x 7 nodes once");
    assert_eq!(
        grid_visits(SearchOrder::BreadthFirst, VisitedPolicy::Global, None).len(),
        49
    );
}

#[test]
fn a_depth_bound_over_the_grid_counts_two_to_the_depth_routes_per_level() {
    let routes = grid_visits(SearchOrder::DepthFirst, VisitedPolicy::Path, Some(5));
    assert_eq!(
        routes.len(),
        63,
        "2^0 + ... + 2^5 routes of at most five hops"
    );
    assert!(routes.iter().all(|&(_, depth)| depth <= 5));

    let nodes = grid_visits(SearchOrder::DepthFirst, VisitedPolicy::Global, Some(5));
    assert_eq!(nodes.len(), 21, "the nodes with x + y <= 5");
    let breadth = grid_visits(SearchOrder::BreadthFirst, VisitedPolicy::Global, Some(5));
    assert_eq!(breadth.len(), 21);
    assert!(
        breadth
            .iter()
            .all(|&((x, y), depth)| usize::try_from(x + y).expect("small") == depth)
    );
}

#[test]
fn open_search_over_a_table_visits_like_the_dense_reference_in_every_discipline() {
    let prune = |node: usize| node.is_multiple_of(6);
    for seed in 1..=4 {
        let table = Table::random(&mut Lcg(seed), 30, 60);
        for limit in [None, Some(2), Some(5)] {
            let with_limit = |config: OpenSearchConfig| {
                limit.map_or(config, |limit| config.with_max_depth(limit))
            };
            let seeds = [4, 11, 4];
            let walk = Walk::new(&table, limit, &prune);
            let depth_first = open_visits(
                &seeds,
                with_limit(OpenSearchConfig::new(SearchOrder::DepthFirst)),
                &table,
                &prune,
            );
            assert_eq!(
                depth_first,
                walk.depth_first(&seeds),
                "seed {seed}, {limit:?}"
            );

            let walk = Walk::new(&table, limit, &prune);
            let breadth_first = open_visits(
                &seeds,
                with_limit(OpenSearchConfig::new(SearchOrder::BreadthFirst)),
                &table,
                &prune,
            );
            assert_eq!(
                breadth_first,
                walk.breadth_first(&seeds),
                "seed {seed}, {limit:?}"
            );
        }

        let routes = |limit| {
            let config = OpenSearchConfig::new(SearchOrder::DepthFirst)
                .with_visited(VisitedPolicy::Path)
                .with_max_depth(limit);
            open_visits(&[4, 11], config, &table, &keep_all)
        };
        assert_eq!(
            routes(4),
            Walk::new(&table, Some(4), &keep_all).depth_first_routes(&[4, 11])
        );
    }
}

#[test]
fn no_seeds_means_no_visits_and_a_repeated_seed_restarts_only_under_path_marks() {
    let table = Table {
        successors: vec![vec![1], vec![]],
    };
    let global = OpenSearchConfig::new(SearchOrder::DepthFirst);
    assert_eq!(open_visits(&[], global, &table, &keep_all).len(), 0);
    assert_eq!(
        open_visits(&[0, 0], global, &table, &keep_all),
        [(0, 0), (1, 1)]
    );
    let path = global.with_visited(VisitedPolicy::Path);
    assert_eq!(
        open_visits(&[0, 0], path, &table, &keep_all),
        [(0, 0), (1, 1), (0, 0), (1, 1)]
    );
}

#[test]
#[should_panic(expected = "VisitedPolicy::Path requires SearchOrder::DepthFirst")]
fn open_breadth_first_search_rejects_path_marks() {
    let config = OpenSearchConfig::new(SearchOrder::BreadthFirst).with_visited(VisitedPolicy::Path);
    let _ = open_search([0_u64], config, collatz_successors, |_, _| {
        ControlFlow::<(), Visit>::Continue(Visit::Descend)
    });
}
