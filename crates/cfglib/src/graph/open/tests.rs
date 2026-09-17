extern crate alloc;

use super::*;
use crate::graph::search::{
    DfsEvent, SearchConfig, SearchOrder, Visit, VisitedPolicy, depth_first_events, search,
};
use crate::graph::store::{Graph, NodeId};
use crate::graph::traverse::TraversalDirection;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

/// A `(module, name)` node in a re-export space.
type Export = (u32, &'static str);

/// Module 0 re-exports the name from module 1 and stars module 2, which
/// stars module 3. Modules 1 and 3 both define it, so which one the
/// chase reports is decided by the push order alone.
fn barrels(named_first: bool) -> impl FnMut(&Export, &mut Vec<Export>) {
    move |node: &Export, out: &mut Vec<Export>| {
        let (module, name) = *node;
        match module {
            0 if named_first => {
                out.push((1, name));
                out.push((2, name));
            }
            0 => {
                out.push((2, name));
                out.push((1, name));
            }
            2 => out.push((3, name)),
            _ => {}
        }
    }
}

fn first_definition(named_first: bool) -> Option<Export> {
    open_search(
        [(0, "Widget")],
        OpenSearchConfig::new(SearchOrder::DepthFirst),
        barrels(named_first),
        |node, _| {
            if node.0 == 1 || node.0 == 3 {
                return ControlFlow::Break(*node);
            }
            ControlFlow::Continue(Visit::Descend)
        },
    )
}

#[test]
fn an_open_chase_explores_successors_in_push_order() {
    // The named re-export is pushed first, so it wins the ambiguity.
    assert_eq!(first_definition(true), Some((1, "Widget")));
    // Pushing the star first reaches the other definition, through the
    // module that only forwards.
    assert_eq!(first_definition(false), Some((3, "Widget")));
}

#[test]
fn an_open_chase_visits_lazily_minted_nodes_in_order() {
    let mut seen = Vec::new();
    let outcome = open_search(
        [(0, "Widget")],
        OpenSearchConfig::new(SearchOrder::DepthFirst),
        barrels(true),
        |node, depth| {
            seen.push((*node, depth));
            ControlFlow::<(), _>::Continue(Visit::Descend)
        },
    );

    assert_eq!(outcome, None);
    assert_eq!(
        seen,
        vec![
            ((0, "Widget"), 0),
            ((1, "Widget"), 1),
            ((2, "Widget"), 1),
            ((3, "Widget"), 2),
        ]
    );
}

#[test]
fn skip_prunes_before_the_successors_are_discovered() {
    let mut expanded = Vec::new();
    let mut seen = Vec::new();
    let outcome = open_search(
        [(0, "Widget")],
        OpenSearchConfig::new(SearchOrder::DepthFirst),
        |node: &Export, out: &mut Vec<Export>| {
            expanded.push(*node);
            barrels(true)(node, out);
        },
        |node, _| {
            seen.push(*node);
            if node.0 == 2 {
                return ControlFlow::<(), _>::Continue(Visit::Skip);
            }
            ControlFlow::Continue(Visit::Descend)
        },
    );

    assert_eq!(outcome, None);
    assert_eq!(seen, vec![(0, "Widget"), (1, "Widget"), (2, "Widget")]);
    // Module 2 was pruned, so its barrel was never read.
    assert_eq!(expanded, vec![(0, "Widget"), (1, "Widget")]);
}

#[test]
fn max_depth_bounds_the_open_walk() {
    let visits = |max_depth| {
        let mut seen = Vec::new();
        let outcome = open_search(
            [(0, "Widget")],
            OpenSearchConfig::new(SearchOrder::DepthFirst).with_max_depth(max_depth),
            barrels(true),
            |node, _| {
                seen.push(*node);
                ControlFlow::<(), _>::Continue(Visit::Descend)
            },
        );
        assert_eq!(outcome, None);
        seen
    };

    assert_eq!(visits(0), vec![(0, "Widget")]);
    assert_eq!(visits(1), vec![(0, "Widget"), (1, "Widget"), (2, "Widget")]);
}

/// A diamond in an open space: `a` forwards to `b` and `c`, both of which
/// forward to `d`.
fn diamond(node: &&'static str, out: &mut Vec<&'static str>) {
    match *node {
        "a" => {
            out.push("b");
            out.push("c");
        }
        "b" | "c" => out.push("d"),
        _ => {}
    }
}

fn open_visits(config: OpenSearchConfig) -> Vec<(&'static str, usize)> {
    let mut seen = Vec::new();
    let outcome = open_search(["a"], config, diamond, |node, depth| {
        seen.push((*node, depth));
        ControlFlow::<(), _>::Continue(Visit::Descend)
    });
    assert_eq!(outcome, None);
    seen
}

#[test]
fn path_policy_revisits_an_open_node_once_per_route() {
    assert_eq!(
        open_visits(OpenSearchConfig::new(SearchOrder::DepthFirst)),
        vec![("a", 0), ("b", 1), ("d", 2), ("c", 1)]
    );
    // Both routes into `d` are reported — the answer a globally marked
    // walk silently drops.
    assert_eq!(
        open_visits(
            OpenSearchConfig::new(SearchOrder::DepthFirst).with_visited(VisitedPolicy::Path)
        ),
        vec![("a", 0), ("b", 1), ("d", 2), ("c", 1), ("d", 2)]
    );
}

#[test]
fn an_open_cycle_terminates_under_both_policies() {
    let ring = |node: &u32, out: &mut Vec<u32>| out.push((node + 1) % 3);
    let visits = |config| {
        let mut seen = Vec::new();
        let outcome = open_search([0_u32], config, ring, |node, _| {
            seen.push(*node);
            ControlFlow::<(), _>::Continue(Visit::Descend)
        });
        assert_eq!(outcome, None);
        seen
    };

    assert_eq!(
        visits(OpenSearchConfig::new(SearchOrder::DepthFirst)),
        vec![0, 1, 2]
    );
    assert_eq!(
        visits(OpenSearchConfig::new(SearchOrder::BreadthFirst)),
        vec![0, 1, 2]
    );
    assert_eq!(
        visits(OpenSearchConfig::new(SearchOrder::DepthFirst).with_visited(VisitedPolicy::Path)),
        vec![0, 1, 2]
    );
}

/// A node of a source tree: `(offset, label)`.
type Emission = (u32, &'static str);

#[test]
fn ordered_emission_reproduces_a_caller_defined_interleaving() {
    // Two child sources the consumer holds separately — declarations and
    // comments — merged by source offset before they are pushed, so the
    // walk emits one interleaved stream in the caller's order.
    let successors = |node: &Emission, out: &mut Vec<Emission>| {
        let (declarations, comments): (&[Emission], &[Emission]) = match node.0 {
            0 => (
                &[(10, "fn a"), (40, "fn b")],
                &[(5, "// header"), (30, "// between")],
            ),
            10 => (&[(20, "let x")], &[(15, "// inner")]),
            _ => (&[], &[]),
        };
        out.extend_from_slice(declarations);
        out.extend_from_slice(comments);
        out.sort_unstable();
    };

    let mut emitted = Vec::new();
    let outcome = open_search(
        [(0, "file")],
        OpenSearchConfig::new(SearchOrder::DepthFirst),
        successors,
        |node, _| {
            emitted.push(*node);
            ControlFlow::<(), _>::Continue(Visit::Descend)
        },
    );

    assert_eq!(outcome, None);
    assert_eq!(
        emitted,
        vec![
            (0, "file"),
            (5, "// header"),
            (10, "fn a"),
            (15, "// inner"),
            (20, "let x"),
            (30, "// between"),
            (40, "fn b"),
        ]
    );
}

#[test]
#[should_panic(expected = "VisitedPolicy::Path requires SearchOrder::DepthFirst")]
fn breadth_first_with_path_marks_is_rejected() {
    let _ = open_search(
        ["a"],
        OpenSearchConfig::new(SearchOrder::BreadthFirst).with_visited(VisitedPolicy::Path),
        diamond,
        |_, _| ControlFlow::<(), _>::Continue(Visit::Descend),
    );
}

#[test]
fn follow_stops_at_the_end_the_bound_and_the_cycle() {
    let chain = |node: &u32| (*node < 3).then(|| node + 1);
    assert_eq!(follow(0, 16, chain), 3, "runs out of steps to take");
    assert_eq!(follow(0, 2, chain), 2, "the hop bound stops it");
    assert_eq!(
        follow(9, 16, chain),
        9,
        "a seed with no step is its own answer"
    );

    // The guard is the whole path, not the seed: `0 -> 1 -> 2 -> 3 -> 1`
    // closes on a middle node and never returns to where it started.
    let lasso = |node: &u32| Some(if *node == 3 { 1 } else { node + 1 });
    assert_eq!(follow(0, 16, lasso), 3);
    assert_eq!(follow_path(0, 16, lasso), vec![0, 1, 2, 3]);
}

#[test]
fn follow_path_returns_the_chain_including_the_seed() {
    let chain = |node: &u32| (*node < 3).then(|| node + 1);
    assert_eq!(follow_path(0, 16, chain), vec![0, 1, 2, 3]);
    assert_eq!(follow_path(0, 2, chain), vec![0, 1, 2]);
    assert_eq!(follow_path(0, 0, chain), vec![0]);
    assert_eq!(follow_path(9, 16, chain), vec![9]);
    // `follow` is the last node of exactly this chain.
    for hops in 0..5 {
        assert_eq!(
            Some(follow(0, hops, chain)),
            follow_path(0, hops, chain).pop()
        );
    }
}

/// `a -> b`, `a -> c`, `b -> d`, `c -> d`, `d -> a`.
fn dense_diamond() -> (Graph<(), ()>, [NodeId; 4]) {
    let mut graph = Graph::<(), ()>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.add_edge(a, b, ());
    graph.add_edge(a, c, ());
    graph.add_edge(b, d, ());
    graph.add_edge(c, d, ());
    graph.add_edge(d, a, ());
    (graph, [a, b, c, d])
}

#[test]
fn the_open_search_matches_the_dense_search_node_for_node() {
    let (graph, [a, _b, _c, _d]) = dense_diamond();
    let orders = [SearchOrder::DepthFirst, SearchOrder::BreadthFirst];
    let policies = [VisitedPolicy::Global, VisitedPolicy::Path];

    for order in orders {
        for visited in policies {
            if (order, visited) == (SearchOrder::BreadthFirst, VisitedPolicy::Path) {
                continue;
            }
            for max_depth in [None, Some(0), Some(2)] {
                let mut dense = Vec::new();
                let outcome = search(
                    &graph,
                    [a],
                    SearchConfig {
                        order,
                        visited,
                        direction: TraversalDirection::Outgoing,
                        max_depth,
                    },
                    |node, depth| {
                        dense.push((node.index(), depth));
                        ControlFlow::<(), _>::Continue(Visit::Descend)
                    },
                );
                assert_eq!(outcome, None);

                let mut open = Vec::new();
                let outcome = open_search(
                    [a.index()],
                    OpenSearchConfig {
                        order,
                        visited,
                        max_depth,
                    },
                    |node: &usize, out: &mut Vec<usize>| {
                        out.extend(
                            graph
                                .successors(NodeId::from_index(*node))
                                .map(NodeId::index),
                        );
                    },
                    |node, depth| {
                        open.push((*node, depth));
                        ControlFlow::<(), _>::Continue(Visit::Descend)
                    },
                );
                assert_eq!(outcome, None);

                assert_eq!(dense, open, "{order:?} {visited:?} {max_depth:?}");
            }
        }
    }
}

/// An [`OpenDfsEvent`] projected to owned data. The events themselves
/// borrow the walk's frontier, so a pinned sequence has to be copied out.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Seen<N> {
    Discover(N, usize),
    Finish(N),
    Refused(N, usize),
}

fn seen<N: Clone>(event: &OpenDfsEvent<'_, N>) -> Seen<N> {
    match *event {
        OpenDfsEvent::Discover(node, depth) => Seen::Discover(N::clone(node), depth),
        OpenDfsEvent::Finish(node) => Seen::Finish(N::clone(node)),
        OpenDfsEvent::Refused(node, depth) => Seen::Refused(N::clone(node), depth),
    }
}

/// Run an events walk to completion and return the sequence it emitted.
fn open_events<N: Clone + Ord>(
    seeds: impl IntoIterator<Item = N>,
    config: OpenDfsConfig,
    successors: impl FnMut(&N, &mut Vec<N>),
) -> Vec<Seen<N>> {
    let mut events = Vec::new();
    let outcome = open_depth_first_events(seeds, config, successors, |event| {
        events.push(seen(&event));
        ControlFlow::<(), _>::Continue(Visit::Descend)
    });
    assert_eq!(outcome, None);
    events
}

/// `a -> [b, c]`, `b -> [d, e]`: a tree, so every node is reached once
/// under either policy and only the nesting is under test.
fn tree(node: &&'static str, out: &mut Vec<&'static str>) {
    match *node {
        "a" => out.extend(["b", "c"]),
        "b" => out.extend(["d", "e"]),
        _ => {}
    }
}

#[test]
fn an_open_events_walk_finishes_strictly_last_in_first_out() {
    assert_eq!(
        open_events(["a"], OpenDfsConfig::new(VisitedPolicy::Global), tree),
        vec![
            Seen::Discover("a", 0),
            Seen::Discover("b", 1),
            Seen::Discover("d", 2),
            Seen::Finish("d"),
            Seen::Discover("e", 2),
            Seen::Finish("e"),
            Seen::Finish("b"),
            Seen::Discover("c", 1),
            Seen::Finish("c"),
            Seen::Finish("a"),
        ]
    );
}

#[test]
fn the_events_closure_is_handed_a_cleared_buffer() {
    // Every frame's successors share one arena, so the buffer the closure
    // writes into has to be a separate one. A consumer that *reorders*
    // what it pushed — the ordered-emission shape — would otherwise sort
    // its ancestors' not-yet-entered successors along with its own.
    let successors = |node: &u32, out: &mut Vec<u32>| {
        assert!(out.is_empty(), "the closure is handed a cleared buffer");
        match *node {
            0 => out.extend([3, 1]),
            1 => out.extend([5, 4]),
            _ => {}
        }
        out.sort_unstable();
    };

    assert_eq!(
        open_events(
            [0_u32],
            OpenDfsConfig::new(VisitedPolicy::Global),
            successors
        ),
        vec![
            Seen::Discover(0, 0),
            Seen::Discover(1, 1),
            Seen::Discover(4, 2),
            Seen::Finish(4),
            Seen::Discover(5, 2),
            Seen::Finish(5),
            Seen::Finish(1),
            Seen::Discover(3, 1),
            Seen::Finish(3),
            Seen::Finish(0),
        ]
    );
}

#[test]
fn path_marks_refold_a_shared_node_once_per_route() {
    // Globally marked, the second route into `d` is refused: the walk
    // reports it, but the fold below `c` gets nothing.
    assert_eq!(
        open_events(["a"], OpenDfsConfig::new(VisitedPolicy::Global), diamond),
        vec![
            Seen::Discover("a", 0),
            Seen::Discover("b", 1),
            Seen::Discover("d", 2),
            Seen::Finish("d"),
            Seen::Finish("b"),
            Seen::Discover("c", 1),
            Seen::Refused("d", 2),
            Seen::Finish("c"),
            Seen::Finish("a"),
        ]
    );
    // Path marks release at `d`'s finish, so the route through `c` folds
    // it again — the second contribution a C++ ambiguity is made of.
    assert_eq!(
        open_events(["a"], OpenDfsConfig::new(VisitedPolicy::Path), diamond),
        vec![
            Seen::Discover("a", 0),
            Seen::Discover("b", 1),
            Seen::Discover("d", 2),
            Seen::Finish("d"),
            Seen::Finish("b"),
            Seen::Discover("c", 1),
            Seen::Discover("d", 2),
            Seen::Finish("d"),
            Seen::Finish("c"),
            Seen::Finish("a"),
        ]
    );
}

#[test]
fn a_pruned_node_finishes_immediately_and_is_never_expanded() {
    let mut expanded = Vec::new();
    let mut events = Vec::new();
    let outcome = open_depth_first_events(
        ["a"],
        OpenDfsConfig::new(VisitedPolicy::Global),
        |node: &&'static str, out: &mut Vec<&'static str>| {
            expanded.push(*node);
            tree(node, out);
        },
        |event| {
            events.push(seen(&event));
            match event {
                OpenDfsEvent::Discover(node, _) if *node == "b" => {
                    ControlFlow::<(), _>::Continue(Visit::Skip)
                }
                _ => ControlFlow::Continue(Visit::Descend),
            }
        },
    );

    assert_eq!(outcome, None);
    // `b` finishes as a leaf, immediately after its own discovery.
    assert_eq!(
        events,
        vec![
            Seen::Discover("a", 0),
            Seen::Discover("b", 1),
            Seen::Finish("b"),
            Seen::Discover("c", 1),
            Seen::Finish("c"),
            Seen::Finish("a"),
        ]
    );
    // Pruning happened before the successor closure was called at all.
    assert_eq!(expanded, vec!["a", "c"]);
}

#[test]
fn max_depth_bounds_expansion_not_visiting() {
    let mut expanded = Vec::new();
    let mut events = Vec::new();
    let outcome = open_depth_first_events(
        ["a"],
        OpenDfsConfig::new(VisitedPolicy::Global).with_max_depth(1),
        |node: &&'static str, out: &mut Vec<&'static str>| {
            expanded.push(*node);
            tree(node, out);
        },
        |event| {
            events.push(seen(&event));
            ControlFlow::<(), _>::Continue(Visit::Descend)
        },
    );

    assert_eq!(outcome, None);
    assert_eq!(
        events,
        vec![
            Seen::Discover("a", 0),
            Seen::Discover("b", 1),
            Seen::Finish("b"),
            Seen::Discover("c", 1),
            Seen::Finish("c"),
            Seen::Finish("a"),
        ]
    );
    assert_eq!(expanded, vec!["a"]);
}

#[test]
fn a_cycle_is_refused_rather_than_re_entered() {
    let ring = |node: &u32, out: &mut Vec<u32>| out.push((node + 1) % 3);
    let closed = vec![
        Seen::Discover(0, 0),
        Seen::Discover(1, 1),
        Seen::Discover(2, 2),
        Seen::Refused(0, 3),
        Seen::Finish(2),
        Seen::Finish(1),
        Seen::Finish(0),
    ];

    // Under `Path` the refusal is the cycle guard: `0` is on the path
    // being walked, which is what a consumer diagnosing a cyclic base or
    // import graph reports.
    assert_eq!(
        open_events([0_u32], OpenDfsConfig::new(VisitedPolicy::Path), ring),
        closed
    );
    // Under `Global` the same event says only that `0` was entered
    // earlier — here, the same node.
    assert_eq!(
        open_events([0_u32], OpenDfsConfig::new(VisitedPolicy::Global), ring),
        closed
    );
}

#[test]
fn a_repeated_seed_is_refused_globally_and_walked_again_on_a_fresh_path() {
    assert_eq!(
        open_events(
            ["b", "b"],
            OpenDfsConfig::new(VisitedPolicy::Global),
            diamond
        ),
        vec![
            Seen::Discover("b", 0),
            Seen::Discover("d", 1),
            Seen::Finish("d"),
            Seen::Finish("b"),
            Seen::Refused("b", 0),
        ]
    );
    // The stack is empty between seeds, so every path mark has been
    // released and the second seed walks the same route again.
    assert_eq!(
        open_events(["b", "b"], OpenDfsConfig::new(VisitedPolicy::Path), diamond),
        vec![
            Seen::Discover("b", 0),
            Seen::Discover("d", 1),
            Seen::Finish("d"),
            Seen::Finish("b"),
            Seen::Discover("b", 0),
            Seen::Discover("d", 1),
            Seen::Finish("d"),
            Seen::Finish("b"),
        ]
    );
}

#[test]
fn breaking_at_a_discover_or_a_finish_abandons_the_open_frames() {
    let broken_at = |target: Seen<&'static str>| {
        let mut events = Vec::new();
        let outcome = open_depth_first_events(
            ["a"],
            OpenDfsConfig::new(VisitedPolicy::Global),
            tree,
            |event| {
                let event = seen(&event);
                events.push(event.clone());
                if event == target {
                    return ControlFlow::Break("stopped");
                }
                ControlFlow::Continue(Visit::Descend)
            },
        );
        assert_eq!(outcome, Some("stopped"));
        events
    };

    // Breaking on the way down: neither `a` nor `b` ever finishes.
    assert_eq!(
        broken_at(Seen::Discover("d", 2)),
        vec![
            Seen::Discover("a", 0),
            Seen::Discover("b", 1),
            Seen::Discover("d", 2),
        ]
    );
    // Breaking on the way up: the frames above `d` are abandoned too.
    assert_eq!(
        broken_at(Seen::Finish("d")),
        vec![
            Seen::Discover("a", 0),
            Seen::Discover("b", 1),
            Seen::Discover("d", 2),
            Seen::Finish("d"),
        ]
    );
}

#[test]
fn the_open_events_walk_matches_the_dense_one_event_for_event() {
    let (graph, [a, _b, _c, _d]) = dense_diamond();

    let mut dense = Vec::new();
    let mut dense_refused = Vec::new();
    let outcome = depth_first_events(&graph, a, TraversalDirection::Outgoing, |event| {
        match event {
            DfsEvent::Discover(node, depth) => dense.push(Seen::Discover(node.index(), depth)),
            DfsEvent::Finish(node) => dense.push(Seen::Finish(node.index())),
            DfsEvent::BackEdge(_, node) | DfsEvent::ForwardOrCross(_, node) => {
                dense_refused.push(node.index());
            }
            DfsEvent::TreeEdge(..) => {}
        }
        ControlFlow::<()>::Continue(())
    });
    assert_eq!(outcome, None);

    let events = open_events(
        [a.index()],
        OpenDfsConfig::new(VisitedPolicy::Global),
        |node: &usize, out: &mut Vec<usize>| {
            out.extend(
                graph
                    .successors(NodeId::from_index(*node))
                    .map(NodeId::index),
            );
        },
    );
    let open: Vec<Seen<usize>> = events
        .iter()
        .filter(|event| !matches!(event, Seen::Refused(..)))
        .cloned()
        .collect();
    let open_refused: Vec<usize> = events
        .iter()
        .filter_map(|event| match event {
            Seen::Refused(node, _) => Some(*node),
            _ => None,
        })
        .collect();

    assert_eq!(dense, open);
    // A refused re-entry lands exactly where the dense walk classifies a
    // back or forward/cross edge; only the depth is extra, since an edge
    // event carries none.
    assert_eq!(dense_refused, open_refused);
}

/// What a member lookup finds at one class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lookup {
    Nothing,
    Found(&'static str),
    Ambiguous,
}

/// One yielding base subobject answers; two are an ambiguity.
fn merge(answer: Lookup, from_base: Lookup) -> Lookup {
    match (answer, from_base) {
        (Lookup::Nothing, other) | (other, Lookup::Nothing) => other,
        _ => Lookup::Ambiguous,
    }
}

/// Look `class` up over the [`diamond`] base graph, where every class in
/// `declares` declares the member, and report the answer plus the classes
/// whose bases were actually read.
fn member_lookup(
    class: &'static str,
    visited: VisitedPolicy,
    declares: &[&'static str],
) -> (Lookup, Vec<&'static str>) {
    let mut searched = Vec::new();
    let mut frames: Vec<Lookup> = Vec::new();
    let mut answer = Lookup::Nothing;
    let outcome = open_depth_first_events(
        [class],
        OpenDfsConfig::new(visited),
        |node: &&'static str, out: &mut Vec<&'static str>| {
            searched.push(*node);
            diamond(node, out);
        },
        |event| {
            match event {
                OpenDfsEvent::Discover(&node, _) => {
                    if declares.contains(&node) {
                        // A declaration hides every base declaration, so
                        // the bases below it are never searched.
                        frames.push(Lookup::Found(node));
                        return ControlFlow::<(), _>::Continue(Visit::Skip);
                    }
                    frames.push(Lookup::Nothing);
                }
                OpenDfsEvent::Finish(_) => {
                    let found = frames.pop().unwrap_or(Lookup::Nothing);
                    match frames.last_mut() {
                        Some(parent) => *parent = merge(*parent, found),
                        None => answer = found,
                    }
                }
                OpenDfsEvent::Refused(..) => {}
            }
            ControlFlow::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    (answer, searched)
}

#[test]
fn the_member_lookup_fold_answers_a_unique_provider_and_reports_the_ambiguity() {
    // `a` derives from `b` and `c`, both deriving from `d`.
    assert_eq!(
        member_lookup("a", VisitedPolicy::Path, &["b"]).0,
        Lookup::Found("b"),
        "exactly one base subobject yields the name"
    );
    assert_eq!(
        member_lookup("a", VisitedPolicy::Path, &["d"]).0,
        Lookup::Ambiguous,
        "the shared base yields it through both `b` and `c`"
    );
    assert_eq!(
        member_lookup("a", VisitedPolicy::Global, &["d"]).0,
        Lookup::Found("d"),
        "globally marked, the second route never folds and the ambiguity \
         disappears — which is why the policy is the semantics here"
    );
    assert_eq!(
        member_lookup("a", VisitedPolicy::Path, &[]).0,
        Lookup::Nothing,
        "nothing declares it anywhere"
    );

    let (answer, searched) = member_lookup("a", VisitedPolicy::Path, &["a"]);
    assert_eq!(answer, Lookup::Found("a"));
    assert!(
        searched.is_empty(),
        "a class that declares the name hides its bases, so none are read"
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BfsSeen {
    Discover(u32, usize),
    Refused(u32, usize),
}

fn bfs_seen(event: OpenBfsEvent<'_, u32>) -> BfsSeen {
    match event {
        OpenBfsEvent::Discover(&node, depth) => BfsSeen::Discover(node, depth),
        OpenBfsEvent::Refused(&node, depth) => BfsSeen::Refused(node, depth),
    }
}

#[test]
fn events_follow_levels_and_report_every_refused_route() {
    let mut events = Vec::new();
    let outcome = open_breadth_first_events(
        [0],
        OpenBfsConfig::new(),
        |node, out| {
            assert!(out.is_empty());
            match node {
                0 => out.extend([1, 2]),
                1 | 2 => out.push(3),
                3 => out.push(0),
                _ => {}
            }
        },
        |event| {
            events.push(bfs_seen(event));
            ControlFlow::<(), _>::Continue(Visit::Descend)
        },
    );

    assert_eq!(outcome, None);
    assert_eq!(
        events,
        vec![
            BfsSeen::Discover(0, 0),
            BfsSeen::Discover(1, 1),
            BfsSeen::Discover(2, 1),
            BfsSeen::Discover(3, 2),
            BfsSeen::Refused(3, 2),
            BfsSeen::Refused(0, 3),
        ]
    );
}

#[test]
fn skipping_and_depth_bounds_prevent_successor_discovery() {
    let mut expanded = Vec::new();
    let mut events = Vec::new();
    let outcome = open_breadth_first_events(
        [0],
        OpenBfsConfig::new().with_max_depth(2),
        |node, out| {
            expanded.push(*node);
            out.push(node + 1);
        },
        |event| {
            events.push(bfs_seen(event));
            match event {
                OpenBfsEvent::Discover(&1, _) => ControlFlow::<(), _>::Continue(Visit::Skip),
                _ => ControlFlow::Continue(Visit::Descend),
            }
        },
    );

    assert_eq!(outcome, None);
    assert_eq!(
        events,
        vec![BfsSeen::Discover(0, 0), BfsSeen::Discover(1, 1)]
    );
    assert_eq!(expanded, vec![0]);

    let mut bounded = Vec::new();
    let outcome = open_breadth_first_events(
        [0],
        OpenBfsConfig::new().with_max_depth(2),
        |node, out| out.push(node + 1),
        |event| {
            bounded.push(bfs_seen(event));
            ControlFlow::<(), _>::Continue(Visit::Descend)
        },
    );
    assert_eq!(outcome, None);
    assert_eq!(
        bounded,
        vec![
            BfsSeen::Discover(0, 0),
            BfsSeen::Discover(1, 1),
            BfsSeen::Discover(2, 2),
        ]
    );
}

#[test]
fn duplicate_seeds_and_early_break_are_observable() {
    let mut events = Vec::new();
    let outcome = open_breadth_first_events(
        [0, 0],
        OpenBfsConfig::new(),
        |node, out| out.extend([node + 1, node + 2]),
        |event| {
            events.push(bfs_seen(event));
            match event {
                OpenBfsEvent::Discover(&2, _) => ControlFlow::Break("found"),
                _ => ControlFlow::Continue(Visit::Descend),
            }
        },
    );

    assert_eq!(outcome, Some("found"));
    assert_eq!(
        events,
        vec![
            BfsSeen::Discover(0, 0),
            BfsSeen::Refused(0, 0),
            BfsSeen::Discover(1, 1),
            BfsSeen::Discover(2, 1),
        ]
    );
}
