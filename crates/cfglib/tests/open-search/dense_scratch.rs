//! `search_with_scratch` and `search_with_marks` buffer reuse, the sizing contract, and the
//! rejected configurations of `search`.

use std::ops::ControlFlow;

use cfglib::{
    EpochMarks, NodeId, SearchConfig, SearchOrder, SearchScratch, TraversalDirection, Visit,
    VisitedPolicy, search, search_with_marks, search_with_scratch,
};

use crate::Lcg;
use crate::dense_search::config;
use crate::reference::{Table, Walk, keep_all};

#[test]
fn one_scratch_serves_every_root_of_a_pass_and_forgets_a_walk_that_broke() {
    let table = Table::random(&mut Lcg(61), 50, 110);
    let graph = table.graph();
    let mut scratch = SearchScratch::new(graph.node_bound() + 8);
    assert_eq!(scratch.capacity(), graph.node_bound() + 8);

    for root in 0..50 {
        for order in [SearchOrder::DepthFirst, SearchOrder::BreadthFirst] {
            if root % 7 == 0 {
                // Leave a frontier behind: the next walk must not inherit any of it.
                let stopped = search_with_scratch(
                    &graph,
                    [NodeId::from_index(root)],
                    SearchConfig::new(order, TraversalDirection::Outgoing),
                    &mut scratch,
                    |_, _| ControlFlow::Break(()),
                );
                assert_eq!(stopped, Some(()));
            }
            let mut visits = Vec::new();
            let outcome = search_with_scratch(
                &graph,
                [NodeId::from_index(root)],
                config(
                    order,
                    TraversalDirection::Outgoing,
                    VisitedPolicy::Global,
                    Some(4),
                ),
                &mut scratch,
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
fn a_scratch_built_for_a_larger_graph_covers_a_smaller_one_and_the_path_discipline_too() {
    let small = Table::random(&mut Lcg(70), 10, 20);
    let graph = small.graph();
    let mut scratch = SearchScratch::new(100);
    for root in 0..10 {
        let mut visits = Vec::new();
        let outcome = search_with_scratch(
            &graph,
            [NodeId::from_index(root)],
            config(
                SearchOrder::DepthFirst,
                TraversalDirection::Outgoing,
                VisitedPolicy::Path,
                Some(4),
            ),
            &mut scratch,
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
fn search_with_marks_resets_the_marks_for_each_search_under_both_policies() {
    let table = Table::random(&mut Lcg(80), 30, 60);
    let graph = table.graph();
    let mut marks = EpochMarks::new(graph.node_bound());
    for policy in [VisitedPolicy::Global, VisitedPolicy::Path] {
        for root in 0..30 {
            let mut visits = Vec::new();
            let outcome = search_with_marks(
                &graph,
                [NodeId::from_index(root)],
                config(
                    SearchOrder::DepthFirst,
                    TraversalDirection::Outgoing,
                    policy,
                    Some(4),
                ),
                &mut marks,
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
#[should_panic(expected = "visited marks cover")]
fn a_scratch_smaller_than_the_graph_panics_instead_of_resizing() {
    let graph = Table::random(&mut Lcg(3), 12, 20).graph();
    let mut scratch = SearchScratch::new(graph.node_bound() - 1);
    let _ = search_with_scratch(
        &graph,
        [NodeId::from_index(0)],
        SearchConfig::new(SearchOrder::DepthFirst, TraversalDirection::Outgoing),
        &mut scratch,
        |_, _| ControlFlow::<(), Visit>::Continue(Visit::Descend),
    );
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn a_seed_outside_the_graph_panics() {
    let graph = Table::random(&mut Lcg(3), 5, 8).graph();
    let _ = search(
        &graph,
        [NodeId::from_index(5)],
        SearchConfig::new(SearchOrder::DepthFirst, TraversalDirection::Outgoing),
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
        SearchConfig::new(SearchOrder::BreadthFirst, TraversalDirection::Outgoing)
            .with_visited(VisitedPolicy::Path),
        |_, _| ControlFlow::<(), Visit>::Continue(Visit::Descend),
    );
}

#[test]
fn a_configuration_is_plain_data_a_consumer_can_store_and_compare() {
    let literal = SearchConfig {
        order: SearchOrder::BreadthFirst,
        visited: VisitedPolicy::Global,
        direction: TraversalDirection::Incoming,
        max_depth: Some(3),
    };
    let built = SearchConfig::new(SearchOrder::BreadthFirst, TraversalDirection::Incoming)
        .with_max_depth(3);
    assert_eq!(literal, built);
    assert_eq!(
        SearchConfig::new(SearchOrder::DepthFirst, TraversalDirection::Outgoing).max_depth,
        None
    );
}
