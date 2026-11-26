use super::*;
use crate::graph::traverse::topological_sort;
use crate::test_util::rng::Rng;
use crate::test_util::{DfInst, df_call, df_ff};
use alloc::vec;

#[test]
fn build_from_cfgs_resolves_calls() {
    let mut main_cfg: Cfg<DfInst> = Cfg::new();
    main_cfg
        .block_mut(main_cfg.entry())
        .push(df_call("call", "helper", false));
    let mut helper_cfg: Cfg<DfInst> = Cfg::new();
    helper_cfg.block_mut(helper_cfg.entry()).push(df_ff("ret"));

    let graph = call_graph(&[("main", &main_cfg), ("helper", &helper_cfg)]);
    let main = find_function(&graph, &"main").unwrap();
    let helper = find_function(&graph, &"helper").unwrap();
    assert!(graph.successors(main).any(|callee| callee == helper));
    assert!(!graph.edges().next().unwrap().payload().is_tail_call);
}

#[test]
fn a_key_listed_twice_is_one_function_calling_for_both_cfgs() {
    let calling = |callee: &'static str| {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry())
            .push(df_call("call", callee, false));
        cfg
    };
    let (first, second) = (calling("left"), calling("right"));
    let mut leaf: Cfg<DfInst> = Cfg::new();
    leaf.block_mut(leaf.entry()).push(df_ff("ret"));

    let graph = call_graph(&[
        ("main", &first),
        ("left", &leaf),
        ("main", &second),
        ("right", &leaf),
    ]);

    assert_eq!(graph.node_count(), 3, "one node per distinct key");
    let main = find_function(&graph, &"main").unwrap();
    let callees: Vec<_> = graph
        .successors(main)
        .map(|callee| graph.node(callee).id)
        .collect();
    assert_eq!(callees, ["left", "right"]);
}

#[test]
fn topology_and_recursion_use_shared_algorithms() {
    let mut graph = Graph::new();
    let main = graph.add_node(FunctionNode { id: "main" });
    let helper = graph.add_node(FunctionNode { id: "helper" });
    graph.add_edge(
        main,
        helper,
        CallMetadata {
            is_tail_call: false,
        },
    );
    assert_eq!(topological_sort(&graph), Some(vec![main, helper]));
    assert_eq!(graph.predecessors(main).count(), 0);
    assert_eq!(graph.successors(helper).count(), 0);

    graph.add_edge(
        helper,
        main,
        CallMetadata {
            is_tail_call: false,
        },
    );
    assert!(topological_sort(&graph).is_none());
    assert!(is_recursive_function(&graph, main));
}

#[test]
fn summaries_propagate_callee_first_and_stabilize_cycles() {
    // main calls a; a and b are mutually recursive; a calls leaf.
    let mut graph = Graph::new();
    let main = graph.add_node(FunctionNode { id: "main" });
    let a = graph.add_node(FunctionNode { id: "a" });
    let b = graph.add_node(FunctionNode { id: "b" });
    let leaf = graph.add_node(FunctionNode { id: "leaf" });
    let call = CallMetadata {
        is_tail_call: false,
    };
    graph.add_edge(main, a, call.clone());
    graph.add_edge(a, b, call.clone());
    graph.add_edge(b, a, call.clone());
    graph.add_edge(a, leaf, call);

    // Summary: the set of transitively reachable callees, by id.
    let summaries = propagate_summaries(
        &graph,
        &alloc::collections::BTreeSet::new(),
        |graph, node, summaries| {
            let mut reach = alloc::collections::BTreeSet::new();
            for callee in graph.successors(node) {
                reach.insert(graph.node(callee).id);
                reach.extend(summaries[callee.index()].iter().copied());
            }
            reach
        },
    );
    let names = |node: NodeId| summaries[node.index()].clone();
    assert!(names(leaf).is_empty());
    assert_eq!(names(main), ["a", "b", "leaf"].into_iter().collect());
    assert_eq!(names(a), ["a", "b", "leaf"].into_iter().collect());
    assert_eq!(names(b), ["a", "b", "leaf"].into_iter().collect());
}

/// The walk stops at the function, ignores cycles it never re-enters, and
/// answers for every function what the function's component answers.
#[test]
fn recursion_matches_the_strong_components() {
    let mut graph = Graph::new();
    let call = || CallMetadata {
        is_tail_call: false,
    };
    let ids = ["main", "a", "b", "leaf", "loop", "self", "dead"];
    let nodes: Vec<_> = ids
        .iter()
        .map(|&id| graph.add_node(FunctionNode { id }))
        .collect();
    let [main, a, b, leaf, looping, selfish, dead] = nodes[..] else {
        unreachable!("seven functions");
    };
    for (caller, callee) in [
        (main, a),
        (a, b),
        (b, a),
        (a, leaf),
        (main, looping),
        (looping, selfish),
        (selfish, selfish),
        (dead, main),
    ] {
        graph.add_edge(caller, callee, call());
    }
    let components = SccDecomposition::compute(&graph, SccNumbering::LeavesFirst);
    for node in graph.node_ids() {
        assert_eq!(
            is_recursive_function(&graph, node),
            components.component(node).is_cyclic(&graph),
            "{}",
            graph.node(node).id
        );
    }
    assert!(!is_recursive_function(&graph, main));
    assert!(!is_recursive_function(&graph, looping));
    assert!(is_recursive_function(&graph, selfish));
}

#[test]
#[should_panic(expected = "is not a node of the call graph")]
fn a_removed_function_is_rejected() {
    let mut graph = Graph::<FunctionNode<&str>, CallMetadata>::new();
    let gone = graph.add_node(FunctionNode { id: "gone" });
    graph.remove_node(gone);
    let _ = is_recursive_function(&graph, gone);
}

/// A ring whose only seed sits at its last member, against the id order:
/// the former round-robin schedule ran one pass over the ring per call
/// edge the seed had to cross, about k² transfers.
#[test]
fn a_seed_crosses_a_recursive_ring_in_one_transfer_per_member() {
    const MEMBERS: usize = 1_000;
    let mut graph = Graph::<(), ()>::new();
    let members: Vec<NodeId> = (0..MEMBERS).map(|_| graph.add_node(())).collect();
    for (index, &caller) in members.iter().enumerate() {
        graph.add_edge(caller, members[(index + 1) % MEMBERS], ());
    }
    let mut calls = 0_usize;
    let summaries = propagate_summaries(&graph, &false, |graph, node, summaries| {
        calls += 1;
        node.index() + 1 == MEMBERS
            || graph
                .successors(node)
                .any(|callee| summaries[callee.index()])
    });
    assert!(summaries.iter().all(|&summary| summary));
    assert!(calls <= 3 * MEMBERS, "{calls} transfers");
}

/// The schedule `propagate_summaries` ran before the worklist: every
/// member of a recursive component again, in ascending order, until a
/// whole pass changes nothing.
fn round_robin<S: Clone + PartialEq>(
    graph: &Graph<(), ()>,
    bottom: &S,
    mut compute: impl FnMut(&Graph<(), ()>, NodeId, &[S]) -> S,
) -> Vec<S> {
    let mut summaries = vec![bottom.clone(); graph.node_bound()];
    let components = SccDecomposition::compute(graph, SccNumbering::LeavesFirst);
    for component in components.components() {
        let cyclic = components.component(component[0]).is_cyclic(graph);
        loop {
            let mut changed = false;
            for &node in component {
                let updated = compute(graph, node, &summaries);
                if updated != summaries[node.index()] {
                    summaries[node.index()] = updated;
                    changed = true;
                }
            }
            if !cyclic || !changed {
                break;
            }
        }
    }
    summaries
}

/// A random graph of at most 24 nodes with parallel edges, self-calls,
/// recursive clusters and removed slots.
fn random_graph(rng: &mut Rng) -> Graph<(), ()> {
    let node_count = 1 + rng.below(24);
    let mut graph = Graph::<(), ()>::new();
    let nodes: Vec<NodeId> = (0..node_count).map(|_| graph.add_node(())).collect();
    for _ in 0..rng.below(3 * node_count + 1) {
        graph.add_edge(
            nodes[rng.below(node_count)],
            nodes[rng.below(node_count)],
            (),
        );
    }
    for &node in &nodes {
        if rng.one_in(8) {
            graph.remove_node(node);
        }
    }
    graph
}

/// The ids a function reaches, modulo 64: a monotone summary of finite
/// height whose first `compute` always changes `0`.
fn reached_ids(graph: &Graph<(), ()>, node: NodeId, summaries: &[u64]) -> u64 {
    graph
        .successors(node)
        .fold(1_u64 << (node.index() % 64), |mask, callee| {
            mask | summaries[callee.index()]
        })
}

/// Random graphs with self-calls, recursive clusters and removed slots,
/// under a monotone finite-height summary (the reached ids modulo 64): the
/// worklist reaches the round-robin fixpoint exactly.
#[test]
fn the_worklist_reaches_the_round_robin_fixpoint() {
    let mut rng = Rng::new(93);
    for _ in 0..300 {
        let graph = random_graph(&mut rng);
        assert_eq!(
            propagate_summaries(&graph, &0, reached_ids),
            round_robin(&graph, &0, reached_ids)
        );
    }
}

/// Every first `compute` changes a summary, so `widen` sees every widening
/// point. The points are recursive functions and cut every cycle, and a
/// `widen` that keeps the computed summary changes neither the result nor
/// the `compute` calls of `propagate_summaries`.
#[test]
fn the_widening_points_cut_every_cycle() {
    let mut rng = Rng::new(9_393);
    for _ in 0..300 {
        let graph = random_graph(&mut rng);
        let mut plain_calls = 0_usize;
        let plain = propagate_summaries(&graph, &0, |graph, node, summaries| {
            plain_calls += 1;
            reached_ids(graph, node, summaries)
        });
        let mut widened_calls = 0_usize;
        let mut points = vec![false; graph.node_bound()];
        let widened = propagate_summaries_widened(
            &graph,
            &0,
            |graph, node, summaries| {
                widened_calls += 1;
                reached_ids(graph, node, summaries)
            },
            |node, previous, computed| {
                assert_ne!(previous, computed, "`widen` runs on a change only");
                points[node.index()] = true;
                *computed
            },
        );
        assert_eq!(widened, plain);
        assert_eq!(widened_calls, plain_calls);

        let recursive = recursive_functions(&graph);
        let mut cut = graph.clone();
        for node in graph.node_ids() {
            if points[node.index()] {
                assert!(recursive[node.index()], "a widening point is recursive");
                cut.remove_node(node);
            }
        }
        assert!(
            !recursive_functions(&cut).contains(&true),
            "removing the widening points leaves no cycle"
        );
    }
}

/// The deepest stack over a ring of one-byte frames grows forever. The
/// only widening point is the member whose call closes the ring in the
/// callee-first order, and the depths settle in about two passes.
#[test]
fn a_recursive_ring_widens_at_one_member() {
    const MEMBERS: usize = 1_000;
    let mut graph = Graph::<(), ()>::new();
    let members: Vec<NodeId> = (0..MEMBERS).map(|_| graph.add_node(())).collect();
    for (index, &caller) in members.iter().enumerate() {
        graph.add_edge(caller, members[(index + 1) % MEMBERS], ());
    }
    let mut calls = 0_usize;
    let mut widened = Vec::new();
    let depths = propagate_summaries_widened(
        &graph,
        &None,
        |graph, node, depths: &[Option<u64>]| {
            calls += 1;
            let deepest = graph
                .successors(node)
                .filter_map(|callee| depths[callee.index()])
                .max();
            Some(deepest.unwrap_or(0).saturating_add(1))
        },
        |node, previous, computed| {
            widened.push(node);
            match (*previous, *computed) {
                (Some(previous), Some(computed)) if computed > previous => Some(u64::MAX),
                (previous, computed) => previous.max(computed),
            }
        },
    );
    assert!(depths.iter().all(|&depth| depth == Some(u64::MAX)));
    assert_eq!(
        widened,
        [members[MEMBERS - 1]; 2],
        "set from bottom once, then sent to unbounded"
    );
    assert!(calls <= 3 * MEMBERS, "{calls} transfers");
}

/// A bound that every call raises by one over its callees' highest, and
/// that every third function saturates at 100: a monotone summary of height
/// 256 whose fixpoint is unique, since a cycle either passes a saturating
/// function or climbs to `u8::MAX`.
fn raised_bound(graph: &Graph<(), ()>, node: NodeId, bounds: &[u8]) -> u8 {
    let highest = graph
        .successors(node)
        .map(|callee| bounds[callee.index()])
        .max();
    let bound = highest.map_or(0, |bound| bound.saturating_add(1));
    if node.index().is_multiple_of(3) {
        bound.min(100)
    } else {
        bound
    }
}

/// Sends any increase to the top, as a widening of a bound to infinity.
fn to_top() -> impl FnMut(NodeId, &u8, &u8) -> u8 + Copy {
    |_, previous, computed| {
        if computed > previous {
            u8::MAX
        } else {
            *previous
        }
    }
}

/// With a plain descending pass (`narrow` keeps what `compute` returned), the
/// narrowed summaries are a fixpoint below the widened ones, so on a domain
/// whose fixpoint is unique they are exactly the least fixpoint.
#[test]
fn a_descending_pass_recovers_the_least_fixpoint() {
    let mut rng = Rng::new(7_272);
    let mut recovered = 0;
    for _ in 0..300 {
        let graph = random_graph(&mut rng);
        let least = round_robin(&graph, &0, raised_bound);
        let widened = propagate_summaries_widened(&graph, &0, raised_bound, to_top());
        let narrowed =
            propagate_summaries_narrowed(&graph, &0, raised_bound, to_top(), |_, _, computed| {
                *computed
            });
        assert_eq!(narrowed, least);
        recovered += usize::from(widened != least);
    }
    assert!(recovered > 20, "widening overshot on {recovered} graphs");
}

/// A narrowing that only refines a summary at the top keeps a sound
/// post-fixpoint: at least the least fixpoint, at most the widened result,
/// and at least what `compute` derives from it.
#[test]
fn narrowing_at_the_top_keeps_a_post_fixpoint_below_the_widened_one() {
    let mut rng = Rng::new(7_373);
    let mut lowered = 0;
    for _ in 0..300 {
        let graph = random_graph(&mut rng);
        let least = round_robin(&graph, &0, raised_bound);
        let widened = propagate_summaries_widened(&graph, &0, raised_bound, to_top());
        let narrowed = propagate_summaries_narrowed(
            &graph,
            &0,
            raised_bound,
            to_top(),
            |_, current, computed| {
                if *current == u8::MAX {
                    *computed
                } else {
                    *current
                }
            },
        );
        for node in graph.node_ids() {
            let index = node.index();
            assert!(least[index] <= narrowed[index] && narrowed[index] <= widened[index]);
            assert!(raised_bound(&graph, node, &narrowed) <= narrowed[index]);
        }
        lowered += usize::from(narrowed != widened);
    }
    assert!(lowered > 20, "narrowing lowered {lowered} results");
}

/// A `narrow` that keeps the current summary leaves the widened result as it
/// is, after one more `compute` per member of each recursive component.
#[test]
fn narrowing_that_keeps_the_current_summary_is_the_widened_result() {
    let mut rng = Rng::new(7_474);
    for _ in 0..300 {
        let graph = random_graph(&mut rng);
        assert_eq!(
            propagate_summaries_narrowed(&graph, &0, raised_bound, to_top(), |_, current, _| {
                *current
            }),
            propagate_summaries_widened(&graph, &0, raised_bound, to_top())
        );
    }
}
