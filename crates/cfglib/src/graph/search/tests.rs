use super::*;
use crate::graph::store::{Graph, NodeId};
use alloc::vec;

/// `a -> b -> d`, `a -> c`: depth-first reaches `d` before `c`,
/// breadth-first the other way round.
fn fork() -> (Graph<(), ()>, [NodeId; 4]) {
    let mut graph = Graph::<(), ()>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.add_edge(a, b, ());
    graph.add_edge(a, c, ());
    graph.add_edge(b, d, ());
    (graph, [a, b, c, d])
}

/// `a -> b`, `a -> c`, `b -> d`, `c -> d`: two paths reach `d`.
fn diamond() -> (Graph<(), ()>, [NodeId; 4]) {
    let mut graph = Graph::<(), ()>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.add_edge(a, b, ());
    graph.add_edge(a, c, ());
    graph.add_edge(b, d, ());
    graph.add_edge(c, d, ());
    (graph, [a, b, c, d])
}

fn config(order: SearchOrder) -> SearchConfig {
    SearchConfig::new(order, TraversalDirection::Outgoing)
}

/// Visit order under `config`, seeded at `seeds`, descending everywhere.
fn visit_order<G: GraphView>(
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    config: SearchConfig,
) -> Vec<(G::NodeId, usize)> {
    let mut order = Vec::new();
    let outcome = search(graph, seeds, config, |node, depth| {
        order.push((node, depth));
        ControlFlow::<(), _>::Continue(Visit::Descend)
    });
    assert_eq!(outcome, None, "a descending search never breaks");
    order
}

fn nodes<N: Copy>(order: &[(N, usize)]) -> Vec<N> {
    order.iter().map(|&(node, _)| node).collect()
}

/// The same, over marks the caller owns.
fn marked_order<G: GraphView>(
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    config: SearchConfig,
    marks: &mut EpochMarks,
) -> Vec<(G::NodeId, usize)> {
    let mut order = Vec::new();
    let outcome = search_with_marks(graph, seeds, config, marks, |node, depth| {
        order.push((node, depth));
        ControlFlow::<(), _>::Continue(Visit::Descend)
    });
    assert_eq!(outcome, None, "a descending search never breaks");
    order
}

/// The same, over a scratch the caller owns.
fn scratch_order<G: GraphView>(
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    config: SearchConfig,
    scratch: &mut SearchScratch,
) -> Vec<(G::NodeId, usize)> {
    let mut order = Vec::new();
    let outcome = search_with_scratch(graph, seeds, config, scratch, |node, depth| {
        order.push((node, depth));
        ControlFlow::<(), _>::Continue(Visit::Descend)
    });
    assert_eq!(outcome, None, "a descending search never breaks");
    order
}

/// Every discipline `search` accepts.
fn disciplines() -> [SearchConfig; 3] {
    [
        config(SearchOrder::DepthFirst),
        config(SearchOrder::BreadthFirst),
        config(SearchOrder::DepthFirst).with_visited(VisitedPolicy::Path),
    ]
}

#[test]
fn first_match_depends_on_the_search_order() {
    let (graph, [a, b, c, d]) = fork();
    let first_leaf = |order| {
        search(&graph, [a], config(order), |node, _| {
            if graph.successors(node).count() == 0 {
                return ControlFlow::Break(node);
            }
            ControlFlow::Continue(Visit::Descend)
        })
    };

    // The same graph, the same seed, two disciplines, two answers.
    assert_eq!(first_leaf(SearchOrder::DepthFirst), Some(d));
    assert_eq!(first_leaf(SearchOrder::BreadthFirst), Some(c));

    assert_eq!(
        nodes(&visit_order(&graph, [a], config(SearchOrder::DepthFirst))),
        vec![a, b, d, c]
    );
    assert_eq!(
        nodes(&visit_order(&graph, [a], config(SearchOrder::BreadthFirst))),
        vec![a, b, c, d]
    );
}

#[test]
fn path_policy_reports_every_path_to_a_shared_node() {
    let (graph, [a, b, c, d]) = diamond();
    // Globally marked, `d` is one node reached once.
    assert_eq!(
        nodes(&visit_order(&graph, [a], config(SearchOrder::DepthFirst))),
        vec![a, b, d, c]
    );
    // Path-marked, `d` is reported once per route into it — the shape of
    // an ambiguous base or a symbol reachable through two imports.
    let paths = visit_order(
        &graph,
        [a],
        config(SearchOrder::DepthFirst).with_visited(VisitedPolicy::Path),
    );
    assert_eq!(nodes(&paths), vec![a, b, d, c, d]);
    assert_eq!(paths[2], (d, 2));
    assert_eq!(paths[4], (d, 2));
}

#[test]
fn skip_prunes_the_subtree_under_both_policies() {
    let (graph, [a, b, c, d]) = fork();
    for visited in [VisitedPolicy::Global, VisitedPolicy::Path] {
        let mut order = Vec::new();
        let outcome = search(
            &graph,
            [a],
            config(SearchOrder::DepthFirst).with_visited(visited),
            |node, _| {
                order.push(node);
                if node == b {
                    return ControlFlow::<(), _>::Continue(Visit::Skip);
                }
                ControlFlow::Continue(Visit::Descend)
            },
        );
        // `b` is visited, `d` (only reachable through it) is not.
        assert_eq!(outcome, None);
        assert_eq!(order, vec![a, b, c], "{visited:?}");
    }
    assert_eq!(
        nodes(&visit_order(&graph, [a], config(SearchOrder::DepthFirst))),
        vec![a, b, d, c],
        "without the skip, `d` is reached"
    );
}

#[test]
fn seeds_are_searched_in_the_order_given() {
    let (graph, [a, b, c, d]) = fork();
    // Seeding `c` first puts it before the whole `a` subtree.
    assert_eq!(
        nodes(&visit_order(
            &graph,
            [c, a],
            config(SearchOrder::DepthFirst)
        )),
        vec![c, a, b, d]
    );
    assert_eq!(
        nodes(&visit_order(
            &graph,
            [a, c],
            config(SearchOrder::DepthFirst)
        )),
        vec![a, b, d, c]
    );
    // Breadth-first interleaves the seeds' levels, still in seed order.
    assert_eq!(
        nodes(&visit_order(
            &graph,
            [b, a],
            config(SearchOrder::BreadthFirst)
        )),
        vec![b, a, d, c]
    );
    // Seeds are at depth 0 even when another seed reaches them deeper.
    assert_eq!(
        visit_order(&graph, [d, a], config(SearchOrder::DepthFirst)),
        vec![(d, 0), (a, 0), (b, 1), (c, 1)]
    );
}

#[test]
fn duplicate_seeds_dedup_globally_and_repeat_on_paths() {
    let (graph, [a, b, c, d]) = fork();
    assert_eq!(
        nodes(&visit_order(
            &graph,
            [a, a],
            config(SearchOrder::DepthFirst)
        )),
        vec![a, b, d, c]
    );
    // Each seed starts a fresh path context, so the second one searches
    // again from an empty path.
    assert_eq!(
        nodes(&visit_order(
            &graph,
            [a, a],
            config(SearchOrder::DepthFirst).with_visited(VisitedPolicy::Path)
        )),
        vec![a, b, d, c, a, b, d, c]
    );
}

#[test]
fn cycles_terminate_under_both_policies() {
    // a -> b -> c -> a, with a self-edge on c.
    let mut graph = Graph::<(), ()>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    graph.add_edge(a, b, ());
    graph.add_edge(b, c, ());
    graph.add_edge(c, a, ());
    graph.add_edge(c, c, ());

    assert_eq!(
        nodes(&visit_order(&graph, [a], config(SearchOrder::DepthFirst))),
        vec![a, b, c]
    );
    assert_eq!(
        nodes(&visit_order(&graph, [a], config(SearchOrder::BreadthFirst))),
        vec![a, b, c]
    );
    // The path guard refuses to re-enter a node already on the path, so
    // the walk terminates without a global mark.
    assert_eq!(
        nodes(&visit_order(
            &graph,
            [a],
            config(SearchOrder::DepthFirst).with_visited(VisitedPolicy::Path)
        )),
        vec![a, b, c]
    );
}

#[test]
fn max_depth_bounds_expansion_not_visiting() {
    // A chain a -> b -> c -> d.
    let mut graph = Graph::<(), ()>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.add_edge(a, b, ());
    graph.add_edge(b, c, ());
    graph.add_edge(c, d, ());

    for order in [SearchOrder::DepthFirst, SearchOrder::BreadthFirst] {
        assert_eq!(
            visit_order(&graph, [a], config(order).with_max_depth(0)),
            vec![(a, 0)],
            "{order:?}"
        );
        // The node at the bound is visited; its successor is not
        // discovered through it.
        assert_eq!(
            visit_order(&graph, [a], config(order).with_max_depth(2)),
            vec![(a, 0), (b, 1), (c, 2)],
            "{order:?}"
        );
        assert_eq!(
            visit_order(&graph, [a], config(order).with_max_depth(9)),
            vec![(a, 0), (b, 1), (c, 2), (d, 3)],
            "{order:?}"
        );
    }
}

#[test]
fn break_returns_immediately() {
    let (graph, [a, b, _c, _d]) = fork();
    let mut seen = Vec::new();
    let found = search(
        &graph,
        [a],
        config(SearchOrder::DepthFirst),
        |node, depth| {
            seen.push(node);
            if node == b {
                return ControlFlow::Break(depth);
            }
            ControlFlow::Continue(Visit::Descend)
        },
    );

    assert_eq!(found, Some(1));
    assert_eq!(seen, vec![a, b], "nothing after the break is visited");
}

#[test]
fn the_incoming_direction_searches_predecessors() {
    let (graph, [a, b, _c, d]) = fork();
    assert_eq!(
        nodes(&visit_order(
            &graph,
            [d],
            SearchConfig::new(SearchOrder::DepthFirst, TraversalDirection::Incoming)
        )),
        vec![d, b, a]
    );
}

#[test]
fn every_discipline_walks_the_incoming_axis() {
    // The direction is resolved to a type once per call, so each
    // discipline reaches its reverse axis through its own dispatch arm:
    // a mixed-up arm would silently walk the graph the other way round.
    let (graph, [a, b, c, d]) = diamond();
    let reverse = |order| SearchConfig::new(order, TraversalDirection::Incoming);

    assert_eq!(
        nodes(&visit_order(
            &graph,
            [d],
            reverse(SearchOrder::BreadthFirst)
        )),
        vec![d, b, c, a]
    );
    // Path marks un-mark on unwind, so `a` is reported once per route.
    assert_eq!(
        nodes(&visit_order(
            &graph,
            [d],
            reverse(SearchOrder::DepthFirst).with_visited(VisitedPolicy::Path)
        )),
        vec![d, b, a, c, a]
    );
    // And the forward axis of the same graph is a different answer.
    assert_eq!(
        nodes(&visit_order(&graph, [a], config(SearchOrder::BreadthFirst))),
        vec![a, b, c, d]
    );
}

#[test]
#[should_panic(expected = "VisitedPolicy::Path requires SearchOrder::DepthFirst")]
fn breadth_first_with_path_marks_is_rejected() {
    let (graph, [a, _, _, _]) = fork();
    let _ = search(
        &graph,
        [a],
        config(SearchOrder::BreadthFirst).with_visited(VisitedPolicy::Path),
        |_, _| ControlFlow::<(), _>::Continue(Visit::Descend),
    );
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn an_out_of_range_seed_panics() {
    let (graph, _) = fork();
    let _ = search(
        &graph,
        [NodeId::from_index(9)],
        config(SearchOrder::DepthFirst),
        |_, _| ControlFlow::<(), _>::Continue(Visit::Descend),
    );
}

#[test]
fn reused_marks_equal_one_fresh_search_per_root() {
    let (graph, roots) = diamond();
    for config in disciplines() {
        let mut marks = EpochMarks::new(graph.node_count());
        // Twice around, so every root is also searched over marks a full
        // pass has already written.
        for &root in roots.iter().chain(roots.iter()) {
            assert_eq!(
                marked_order(&graph, [root], config, &mut marks),
                visit_order(&graph, [root], config),
                "{config:?} from {root:?}"
            );
        }
        // Seed handling reads the same marks, so multi-seed searches (and
        // the duplicate-seed rules) have to survive the reuse too.
        for seeds in [[roots[3], roots[0]], [roots[0], roots[0]]] {
            assert_eq!(
                marked_order(&graph, seeds, config, &mut marks),
                visit_order(&graph, seeds, config),
                "{config:?} from {seeds:?}"
            );
        }
    }
}

#[test]
fn a_reused_search_never_inherits_the_previous_marks() {
    let (graph, [a, b, c, d]) = diamond();
    let dfs = config(SearchOrder::DepthFirst);
    let path = dfs.with_visited(VisitedPolicy::Path);
    let mut marks = EpochMarks::new(graph.node_count());

    // A pass over the whole graph, then a search of a part of it: the
    // second search sees none of the first one's marks.
    assert_eq!(
        nodes(&marked_order(&graph, [a], dfs, &mut marks)),
        vec![a, b, d, c]
    );
    assert_eq!(
        nodes(&marked_order(&graph, [b], dfs, &mut marks)),
        vec![b, d]
    );
    assert_eq!(
        nodes(&marked_order(&graph, [c], dfs, &mut marks)),
        vec![c, d]
    );

    // A search that broke mid-walk leaves marks set under both policies —
    // and the walk after it still starts from a clean set.
    for config in [dfs, path] {
        let found = search_with_marks(&graph, [a], config, &mut marks, |node, _| {
            if node == b {
                return ControlFlow::Break(node);
            }
            ControlFlow::Continue(Visit::Descend)
        });
        assert_eq!(found, Some(b), "{config:?}");
        assert_eq!(
            nodes(&marked_order(&graph, [a], dfs, &mut marks)),
            vec![a, b, d, c],
            "{config:?}"
        );
    }
}

#[test]
fn marks_stay_correct_across_an_epoch_wrap() {
    let (graph, [a, b, c, d]) = diamond();
    let dfs = config(SearchOrder::DepthFirst);
    let mut marks = EpochMarks::new(graph.node_count());
    // The state one search short of the wrap, carrying a stamp from the
    // last time the epoch was 1 — the one value bumping alone would not
    // invalidate, so only clearing the buffer keeps `c` visitable.
    marks.epoch = u32::MAX - 1;
    marks.stamps[c.index()] = 1;

    // This search takes the epoch to its last value and never touches `c`.
    assert_eq!(
        nodes(&marked_order(&graph, [b], dfs, &mut marks)),
        vec![b, d]
    );
    assert_eq!(marks.epoch, u32::MAX);
    // This one wraps.
    assert_eq!(
        nodes(&marked_order(&graph, [a], dfs, &mut marks)),
        vec![a, b, d, c]
    );
    assert_eq!(marks.epoch, 1, "the epoch wrapped to its first value");
    // And the buffer keeps working on the far side of the wrap.
    assert_eq!(
        nodes(&marked_order(&graph, [a], dfs, &mut marks)),
        vec![a, b, d, c]
    );
    assert_eq!(marks.epoch, 2);
}

#[test]
fn marks_larger_than_the_graph_are_accepted() {
    // One buffer sized by the largest node space serves the smaller ones.
    let (graph, [a, b, c, d]) = diamond();
    let mut marks = EpochMarks::new(graph.node_count() + 8);
    assert_eq!(
        nodes(&marked_order(
            &graph,
            [a],
            config(SearchOrder::DepthFirst),
            &mut marks
        )),
        vec![a, b, d, c]
    );
}

#[test]
#[should_panic(expected = "visited marks cover 2 nodes but the graph has 4")]
fn marks_smaller_than_the_graph_panic() {
    let (graph, [a, _, _, _]) = diamond();
    let mut marks = EpochMarks::new(2);
    let _ = search_with_marks(
        &graph,
        [a],
        config(SearchOrder::DepthFirst),
        &mut marks,
        |_, _| ControlFlow::<(), _>::Continue(Visit::Descend),
    );
}

#[test]
fn a_reused_scratch_equals_one_fresh_search_per_root() {
    let (graph, roots) = diamond();
    for config in disciplines() {
        let mut scratch = SearchScratch::new(graph.node_count());
        // Twice around, so every root is also searched over buffers a full
        // pass has already written.
        for &root in roots.iter().chain(roots.iter()) {
            assert_eq!(
                scratch_order(&graph, [root], config, &mut scratch),
                visit_order(&graph, [root], config),
                "{config:?} from {root:?}"
            );
        }
        // Seeds are collected into the scratch too, so multi-seed searches
        // (and the duplicate-seed rules) have to survive the reuse.
        for seeds in [[roots[3], roots[0]], [roots[0], roots[0]]] {
            assert_eq!(
                scratch_order(&graph, seeds, config, &mut scratch),
                visit_order(&graph, seeds, config),
                "{config:?} from {seeds:?}"
            );
        }
    }
}

#[test]
fn one_scratch_serves_every_discipline_in_turn() {
    // The two globally marked cores share one frontier buffer — a stack
    // for the depth-first walk, a queue for the breadth-first one — so a
    // pass that changes discipline mid-way is the case that would catch a
    // buffer left in the other core's shape.
    let (graph, [a, b, c, d]) = diamond();
    let mut scratch = SearchScratch::new(graph.node_count());
    for _ in 0..2 {
        assert_eq!(
            nodes(&scratch_order(
                &graph,
                [a],
                config(SearchOrder::DepthFirst),
                &mut scratch
            )),
            vec![a, b, d, c]
        );
        assert_eq!(
            nodes(&scratch_order(
                &graph,
                [a],
                config(SearchOrder::BreadthFirst),
                &mut scratch
            )),
            vec![a, b, c, d]
        );
        assert_eq!(
            nodes(&scratch_order(
                &graph,
                [a],
                config(SearchOrder::DepthFirst).with_visited(VisitedPolicy::Path),
                &mut scratch
            )),
            vec![a, b, d, c, d]
        );
    }
}

#[test]
fn a_reused_scratch_never_inherits_the_previous_frontier() {
    let (graph, [a, b, c, d]) = diamond();
    let dfs = config(SearchOrder::DepthFirst);
    let mut scratch = SearchScratch::new(graph.node_count());

    // A search that broke mid-walk returns with its frontier still loaded
    // — under every discipline — and the walk after it still starts from
    // an empty one.
    for config in disciplines() {
        let found = search_with_scratch(&graph, [a], config, &mut scratch, |node, _| {
            if node == b {
                return ControlFlow::Break(node);
            }
            ControlFlow::Continue(Visit::Descend)
        });
        assert_eq!(found, Some(b), "{config:?}");
        assert_eq!(
            nodes(&scratch_order(&graph, [a], dfs, &mut scratch)),
            vec![a, b, d, c],
            "{config:?}"
        );
    }

    // A whole-graph pass followed by a search of a part of it: neither the
    // marks nor the buffers of the first survive into the second.
    assert_eq!(
        nodes(&scratch_order(&graph, [b], dfs, &mut scratch)),
        vec![b, d]
    );
    assert_eq!(
        nodes(&scratch_order(&graph, [c], dfs, &mut scratch)),
        vec![c, d]
    );
}

#[test]
fn a_scratch_larger_than_the_graph_is_accepted() {
    // One scratch sized by the largest node space serves the smaller ones.
    let (graph, [a, b, c, d]) = diamond();
    let mut scratch = SearchScratch::new(graph.node_count() + 8);
    assert_eq!(scratch.capacity(), graph.node_count() + 8);
    assert_eq!(
        nodes(&scratch_order(
            &graph,
            [a],
            config(SearchOrder::DepthFirst),
            &mut scratch
        )),
        vec![a, b, d, c]
    );
}

#[test]
#[should_panic(expected = "visited marks cover 2 nodes but the graph has 4")]
fn a_scratch_smaller_than_the_graph_panics() {
    let (graph, [a, _, _, _]) = diamond();
    let mut scratch = SearchScratch::new(2);
    let _ = search_with_scratch(
        &graph,
        [a],
        config(SearchOrder::DepthFirst),
        &mut scratch,
        |_, _| ControlFlow::<(), _>::Continue(Visit::Descend),
    );
}
