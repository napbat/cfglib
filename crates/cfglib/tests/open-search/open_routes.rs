//! `open_breadth_first_paths`: simple routes through a lazily discovered space, shortest first.

use std::ops::ControlFlow;

use cfglib::{OpenPathsConfig, OpenPathsEvent, Visit, open_breadth_first_paths};

use crate::Lcg;
use crate::reference::{Table, Walk, keep_all, simple_routes, table_successors};
use crate::spaces::{GRID_SIDE, binomial, grid_successors};

/// Every route event of a walk that never prunes, in event order.
fn emitted_routes(table: &Table, seeds: &[usize], config: OpenPathsConfig) -> Vec<Vec<usize>> {
    let mut routes = Vec::new();
    let outcome = open_breadth_first_paths::<_, ()>(
        seeds.iter().copied(),
        config,
        table_successors(table),
        |event| {
            if let OpenPathsEvent::Route(route) = event {
                routes.push(route.to_vec());
            }
            ControlFlow::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    routes
}

#[test]
fn routes_come_out_as_the_multiset_of_simple_routes_a_recursion_enumerates() {
    for seed in 1..=6 {
        let table = Table::random(&mut Lcg(seed), 9, 16);
        for limit in [None, Some(3)] {
            let config = limit.map_or(OpenPathsConfig::new(), |limit| {
                OpenPathsConfig::new().with_max_depth(limit)
            });
            let seeds = [0, 4, 0];

            let mut actual = emitted_routes(&table, &seeds, config);
            let mut expected = Vec::new();
            for &start in &seeds {
                expected.extend(simple_routes(&table, start, limit));
            }
            actual.sort();
            expected.sort();
            assert_eq!(actual, expected, "seed {seed}, limit {limit:?}");
        }
    }
}

#[test]
fn the_number_of_routes_equals_the_number_of_path_marked_depth_first_visits() {
    for seed in 1..=6 {
        let table = Table::random(&mut Lcg(seed + 20), 10, 18);
        let routes = emitted_routes(&table, &[0], OpenPathsConfig::new().with_max_depth(6));
        let visits = Walk::new(&table, Some(6), &keep_all).depth_first_routes(&[0]);
        assert_eq!(routes.len(), visits.len(), "seed {seed}");
    }
}

#[test]
fn routes_arrive_in_nondecreasing_length_and_the_first_route_to_a_node_is_a_shortest_one() {
    for seed in 1..=6 {
        let table = Table::random(&mut Lcg(seed + 40), 25, 45);
        let routes = emitted_routes(&table, &[0], OpenPathsConfig::new().with_max_depth(5));
        let lengths: Vec<usize> = routes.iter().map(Vec::len).collect();
        assert!(
            lengths.is_sorted(),
            "seed {seed}: breadth-first means shortest-first"
        );

        let distance = Walk::new(&table, None, &keep_all).breadth_first(&[0]);
        for &(goal, hops) in distance.iter().filter(|&&(_, hops)| hops <= 5) {
            let first = routes
                .iter()
                .find(|route| route.last() == Some(&goal))
                .expect("a node within five hops has a route");
            assert_eq!(first.len() - 1, hops, "seed {seed}: first route to n{goal}");
        }
    }
}

#[test]
fn the_first_route_to_a_goal_is_a_shortest_one_even_when_a_longer_route_is_listed_first() {
    // 0 -> {1, 4}, 1 -> 2 -> 3 -> 4: the long way round is the first successor.
    let table = Table {
        successors: vec![vec![1, 4], vec![2], vec![3], vec![4], vec![]],
    };
    let shortest = open_breadth_first_paths(
        [0_usize],
        OpenPathsConfig::new(),
        table_successors(&table),
        |event| match event {
            OpenPathsEvent::Route(route) if route.last() == Some(&4) => {
                ControlFlow::Break(route.to_vec())
            }
            _ => ControlFlow::Continue(Visit::Descend),
        },
    );
    assert_eq!(shortest, Some(vec![0, 4]));
}

#[test]
fn routes_through_a_grid_are_counted_by_binomial_coefficients() {
    let side = u64::from(GRID_SIDE);
    let goal = (GRID_SIDE, GRID_SIDE);
    let mut routes: Vec<Vec<(u32, u32)>> = Vec::new();
    let outcome = open_breadth_first_paths::<_, ()>(
        [(0_u32, 0_u32)],
        OpenPathsConfig::new(),
        grid_successors,
        |event| {
            if let OpenPathsEvent::Route(route) = event {
                routes.push(route.to_vec());
            }
            ControlFlow::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);

    let total: u64 = (0..=side)
        .flat_map(|x| (0..=side).map(move |y| binomial(x + y, x)))
        .sum();
    assert_eq!(u64::try_from(routes.len()).expect("fits"), total);
    assert_eq!(total, 3431);

    let mut to_goal: Vec<_> = routes
        .iter()
        .filter(|route| route.last() == Some(&goal))
        .collect();
    assert_eq!(
        u64::try_from(to_goal.len()).expect("fits"),
        binomial(2 * side, side),
        "C(12, 6) = 924"
    );
    to_goal.sort();
    to_goal.dedup();
    assert_eq!(to_goal.len(), 924, "every route to the goal is distinct");
    for route in to_goal {
        assert_eq!(
            route.len(),
            13,
            "a monotone route through the grid has 12 hops"
        );
        assert_eq!(route[0], (0, 0));
        for hop in route.windows(2) {
            let ((x, y), (next_x, next_y)) = (hop[0], hop[1]);
            assert!((next_x, next_y) == (x + 1, y) || (next_x, next_y) == (x, y + 1));
        }
    }

    let lengths: Vec<usize> = routes.iter().map(Vec::len).collect();
    assert!(lengths.is_sorted());
}

#[test]
fn a_depth_bound_over_the_grid_emits_two_to_the_depth_routes_per_level() {
    let mut count = 0;
    let outcome = open_breadth_first_paths::<_, ()>(
        [(0_u32, 0_u32)],
        OpenPathsConfig::new().with_max_depth(5),
        grid_successors,
        |event| {
            if matches!(event, OpenPathsEvent::Route(_)) {
                count += 1;
            }
            ControlFlow::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    assert_eq!(count, 63, "2^0 + ... + 2^5");

    assert_eq!(
        emitted_routes(
            &Table {
                successors: vec![vec![0]]
            },
            &[0],
            OpenPathsConfig::new().with_max_depth(0)
        ),
        [[0]],
        "a zero bound emits the seed route alone"
    );
}

#[test]
fn a_node_already_on_the_route_is_refused_instead_of_extended() {
    let table = Table {
        successors: vec![vec![1], vec![2], vec![0]],
    };
    let mut seen = Vec::new();
    let outcome = open_breadth_first_paths::<_, ()>(
        [0_usize],
        OpenPathsConfig::new(),
        table_successors(&table),
        |event| {
            seen.push(match event {
                OpenPathsEvent::Route(route) => (route.to_vec(), None),
                OpenPathsEvent::Refused(route, node) => (route.to_vec(), Some(*node)),
            });
            ControlFlow::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    assert_eq!(
        seen,
        [
            (vec![0], None),
            (vec![0, 1], None),
            (vec![0, 1, 2], None),
            (vec![0, 1, 2], Some(0)),
        ]
    );
}

#[test]
fn a_refusal_names_a_node_on_its_route_and_its_verdict_is_ignored() {
    for seed in 1..=4 {
        let table = Table::random(&mut Lcg(seed + 60), 8, 16);
        let mut refusals = 0;
        let outcome = open_breadth_first_paths::<_, ()>(
            [0_usize],
            OpenPathsConfig::new().with_max_depth(5),
            table_successors(&table),
            |event| match event {
                OpenPathsEvent::Refused(route, node) => {
                    assert!(
                        route.contains(node),
                        "seed {seed}: refused n{node} is not on {route:?}"
                    );
                    refusals += 1;
                    ControlFlow::Continue(Visit::Skip)
                }
                OpenPathsEvent::Route(_) => ControlFlow::Continue(Visit::Descend),
            },
        );
        assert_eq!(outcome, None);
        assert!(refusals > 0, "seed {seed} never closes a cycle");

        let routes = emitted_routes(&table, &[0], OpenPathsConfig::new().with_max_depth(5));
        let expected = simple_routes(&table, 0, Some(5));
        assert_eq!(
            routes.len(),
            expected.len(),
            "seed {seed}: a Skip verdict on a refusal changes nothing"
        );
    }
}

#[test]
fn skip_seals_a_route_without_extending_it() {
    // 0 -> {1, 2}, 1 -> 3
    let table = Table {
        successors: vec![vec![1, 2], vec![3], vec![], vec![]],
    };
    let mut routes = Vec::new();
    let outcome = open_breadth_first_paths::<_, ()>(
        [0_usize],
        OpenPathsConfig::new(),
        table_successors(&table),
        |event| match event {
            OpenPathsEvent::Route(route) => {
                routes.push(route.to_vec());
                ControlFlow::Continue(if route.last() == Some(&1) {
                    Visit::Skip
                } else {
                    Visit::Descend
                })
            }
            OpenPathsEvent::Refused(..) => ControlFlow::Continue(Visit::Descend),
        },
    );
    assert_eq!(outcome, None);
    assert_eq!(routes, [vec![0], vec![0, 1], vec![0, 2]]);
}

#[test]
fn duplicate_seeds_start_independent_routes_and_a_break_returns_its_value() {
    let table = Table {
        successors: vec![vec![1], vec![]],
    };
    assert_eq!(
        emitted_routes(&table, &[0, 0], OpenPathsConfig::new()),
        [vec![0], vec![0], vec![0, 1], vec![0, 1]]
    );

    let mut seen = 0;
    let outcome = open_breadth_first_paths(
        [0_usize],
        OpenPathsConfig::new(),
        table_successors(&table),
        |event| {
            seen += 1;
            match event {
                OpenPathsEvent::Route(route) if route.len() == 2 => {
                    ControlFlow::Break(route.to_vec())
                }
                _ => ControlFlow::Continue(Visit::Descend),
            }
        },
    );
    assert_eq!(outcome, Some(vec![0, 1]));
    assert_eq!(seen, 2, "nothing is emitted after the break");
}
