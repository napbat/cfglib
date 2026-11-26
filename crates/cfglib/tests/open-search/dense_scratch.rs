//! `search_in` buffer reuse, the sizing contract, and the rejected configurations of `search`.

use std::ops::ControlFlow;

use cfglib::{
    Direction, NodeId, SearchConfig, SearchOrder, SearchScratch, Visit, VisitedPolicy, search,
    search_in,
};

use crate::Lcg;
use crate::dense_search::config;
use crate::reference::{Table, Walk, keep_all};

#[test]
fn one_scratch_serves_every_root_of_a_pass_and_forgets_a_walk_that_broke() {
    let table = Table::random(&mut Lcg(61), 50, 110);
    let graph = table.graph();
    let mut scratch = SearchScratch::default();

    for root in 0..50 {
        for order in [SearchOrder::DepthFirst, SearchOrder::BreadthFirst] {
            if root % 7 == 0 {
                // Leave a frontier behind: the next walk must not inherit any of it.
                let stopped = search_in(
                    &mut scratch,
                    &graph,
                    [NodeId::from_index(root)],
                    Direction::Forward,
                    SearchConfig {
                        order,
                        ..SearchConfig::default()
                    },
                    |_, _| ControlFlow::Break(()),
                );
                assert_eq!(stopped, Some(()));
            }
            let mut visits = Vec::new();
            let discipline = config(order, Direction::Forward, VisitedPolicy::Global, Some(4));
            let outcome = search_in(
                &mut scratch,
                &graph,
                [NodeId::from_index(root)],
                discipline.direction,
                discipline.config,
                |node, depth| {
                    visits.push((node.index(), depth));
                    ControlFlow::<(), Visit>::Continue(Visit::Descend)
                },
            );
            assert_eq!(outcome, None);
            let expected = match order {
                SearchOrder::DepthFirst => {
                    Walk::new(&table, Some(4), &keep_all).depth_first(&[root])
                }
                SearchOrder::BreadthFirst => {
                    Walk::new(&table, Some(4), &keep_all).breadth_first(&[root])
                }
            };
            assert_eq!(visits, expected, "root {root}, {order:?}");
        }
    }
}

#[test]
fn a_scratch_grown_by_a_larger_graph_covers_a_smaller_one_and_the_path_discipline_too() {
    let large = Table::random(&mut Lcg(71), 100, 200).graph();
    let small = Table::random(&mut Lcg(70), 10, 20);
    let graph = small.graph();
    let mut scratch = SearchScratch::default();
    let discipline = config(
        SearchOrder::DepthFirst,
        Direction::Forward,
        VisitedPolicy::Path,
        Some(4),
    );
    let outcome = search_in(
        &mut scratch,
        &large,
        [NodeId::from_index(0)],
        discipline.direction,
        discipline.config,
        |_, _| ControlFlow::<(), Visit>::Continue(Visit::Descend),
    );
    assert_eq!(outcome, None);
    for root in 0..10 {
        let mut visits = Vec::new();
        let outcome = search_in(
            &mut scratch,
            &graph,
            [NodeId::from_index(root)],
            discipline.direction,
            discipline.config,
            |node, depth| {
                visits.push((node.index(), depth));
                ControlFlow::<(), Visit>::Continue(Visit::Descend)
            },
        );
        assert_eq!(outcome, None);
        assert_eq!(
            visits,
            Walk::new(&small, Some(4), &keep_all).depth_first_routes(&[root])
        );
    }
}

#[test]
fn a_reused_scratch_resets_the_marks_for_each_search_under_both_policies() {
    let table = Table::random(&mut Lcg(80), 30, 60);
    let graph = table.graph();
    let mut scratch = SearchScratch::default();
    for policy in [VisitedPolicy::Global, VisitedPolicy::Path] {
        for root in 0..30 {
            let mut visits = Vec::new();
            let discipline = config(SearchOrder::DepthFirst, Direction::Forward, policy, Some(4));
            let outcome = search_in(
                &mut scratch,
                &graph,
                [NodeId::from_index(root)],
                discipline.direction,
                discipline.config,
                |node, depth| {
                    visits.push((node.index(), depth));
                    ControlFlow::<(), Visit>::Continue(Visit::Descend)
                },
            );
            assert_eq!(outcome, None);
            let walk = Walk::new(&table, Some(4), &keep_all);
            let expected = match policy {
                VisitedPolicy::Global => walk.depth_first(&[root]),
                VisitedPolicy::Path => walk.depth_first_routes(&[root]),
            };
            assert_eq!(visits, expected, "root {root}, {policy:?}");
        }
    }
}

#[test]
fn a_scratch_smaller_than_the_graph_grows_on_entry() {
    let table = Table::random(&mut Lcg(3), 12, 20);
    let graph = table.graph();
    let mut scratch = SearchScratch::default();
    let mut visits = Vec::new();
    let outcome = search_in(
        &mut scratch,
        &graph,
        [NodeId::from_index(0)],
        Direction::Forward,
        SearchConfig::default(),
        |node, depth| {
            visits.push((node.index(), depth));
            ControlFlow::<(), Visit>::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    assert_eq!(visits, Walk::new(&table, None, &keep_all).depth_first(&[0]));
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn a_seed_outside_the_graph_panics() {
    let graph = Table::random(&mut Lcg(3), 5, 8).graph();
    let _ = search(
        &graph,
        [NodeId::from_index(5)],
        Direction::Forward,
        SearchConfig::default(),
        |_, _| ControlFlow::<(), Visit>::Continue(Visit::Descend),
    );
}

#[test]
#[should_panic(expected = "VisitedPolicy::Path requires SearchOrder::DepthFirst")]
fn a_breadth_first_frontier_has_no_unwind_so_path_marks_are_rejected() {
    let graph = Table::random(&mut Lcg(3), 5, 8).graph();
    let _ = search(
        &graph,
        [NodeId::from_index(0)],
        Direction::Forward,
        SearchConfig {
            order: SearchOrder::BreadthFirst,
            visited: VisitedPolicy::Path,
            max_depth: None,
        },
        |_, _| ControlFlow::<(), Visit>::Continue(Visit::Descend),
    );
}

#[test]
fn a_configuration_is_plain_data_a_consumer_can_store_and_compare() {
    let literal = SearchConfig {
        order: SearchOrder::BreadthFirst,
        visited: VisitedPolicy::Global,
        max_depth: Some(3),
    };
    let updated = SearchConfig {
        order: SearchOrder::BreadthFirst,
        max_depth: Some(3),
        ..SearchConfig::default()
    };
    assert_eq!(literal, updated);
    assert_eq!(
        SearchConfig::default(),
        SearchConfig {
            order: SearchOrder::DepthFirst,
            visited: VisitedPolicy::Global,
            max_depth: None,
        }
    );
}
