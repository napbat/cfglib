use cfglib::{
    BlockId, ControlClosureScratch, FrontierScratch, Graph, NodeId, NonterminationScratch,
    PostDominatorTree, nontermination_sensitive_controllers, weak_control_closure,
};

use super::BenchmarkSuite;
use crate::fixtures::branchy_cfg;
use crate::harness::benchmark_case;

/// Rungs of the ladder of the nontermination-sensitive case.
const LADDER_RUNGS: usize = 1_024;
/// Blocks of the branchy fixture of the closure cases.
const BRANCHY_BLOCKS: usize = 4_096;
/// One block in this many of the branchy fixture is a seed.
const BRANCHY_STRIDE: usize = 100;
/// One block in this many of the ladder's `b` rail is a seed.
const LADDER_STRIDE: usize = 32;

/// Two rails `a_0 .. a_k` and `b_0 .. b_k`, each a path, with a rung from
/// every `a_i` down to `b_i`; `b_k` is the one exit.
///
/// Every `a_i` decides whether each later block of the `b` rail runs, so the
/// nontermination-sensitive relation holds `k(k + 3) / 2` pairs: the ladder
/// is the documented case where the relation itself is quadratic.
fn ladder(rungs: usize) -> Graph<(), ()> {
    let mut graph = Graph::with_capacity(2 * (rungs + 1), 3 * rungs + 1);
    let left: Vec<NodeId> = (0..=rungs).map(|_| graph.add_node(())).collect();
    let right: Vec<NodeId> = (0..=rungs).map(|_| graph.add_node(())).collect();
    for rung in 0..=rungs {
        if rung < rungs {
            graph.add_edge(left[rung], left[rung + 1], ());
            graph.add_edge(right[rung], right[rung + 1], ());
        }
        graph.add_edge(left[rung], right[rung], ());
    }
    graph
}

fn register_nontermination_sensitive(suite: &mut BenchmarkSuite<'_>) {
    let graph = ladder(LADDER_RUNGS);
    let mut scratch = NonterminationScratch::new();
    let mut controllers = Vec::new();
    benchmark_case!(
        suite,
        "api_nontermination_sensitive_controllers_ladder",
        covers[nontermination_sensitive_controllers],
        || {
            // The whole relation, one dependent at a time.
            graph
                .node_ids()
                .map(|node| {
                    nontermination_sensitive_controllers(
                        &graph,
                        node,
                        &mut scratch,
                        &mut controllers,
                    );
                    controllers.len()
                })
                .sum::<usize>()
        },
        |pairs: &usize| assert_eq!(*pairs, LADDER_RUNGS * (LADDER_RUNGS + 3) / 2)
    );
}

/// Weak control-closure on the branchy fixture, of one seed and of one
/// block in a hundred with the entry and the exit, and on the ladder.
fn register_weak_control_closure(suite: &mut BenchmarkSuite<'_>) {
    let cfg = branchy_cfg(BRANCHY_BLOCKS);
    let mut scratch = ControlClosureScratch::new();
    let mut closure = Vec::new();
    let middle = BlockId::from_index(BRANCHY_BLOCKS / 2);
    benchmark_case!(
        suite,
        "api_weak_control_closure_branchy_one_seed",
        covers[weak_control_closure],
        || {
            weak_control_closure(&cfg, [middle], &mut scratch, &mut closure);
            closure.clone()
        },
        // One seed leaves no choice between two seeds.
        |result: &Vec<BlockId>| assert_eq!(*result, [middle])
    );

    let exit = BlockId::from_index(BRANCHY_BLOCKS - 1);
    let seeds: Vec<BlockId> = (0..BRANCHY_BLOCKS)
        .step_by(BRANCHY_STRIDE)
        .map(BlockId::from_index)
        .chain([exit])
        .collect();
    benchmark_case!(
        suite,
        "api_weak_control_closure_branchy_percent",
        || {
            weak_control_closure(&cfg, seeds.iter().copied(), &mut scratch, &mut closure);
            closure.clone()
        },
        |result: &Vec<BlockId>| {
            // Every block reaches the exit and the seeds hold the entry and
            // the exit, so the closure is the seeds with their iterated
            // control-dependence frontier.
            let post = PostDominatorTree::compute(&cfg);
            let mut expected = Vec::new();
            post.iterated_frontier(
                &cfg,
                &mut FrontierScratch::new(),
                seeds.iter().copied(),
                None,
                &mut expected,
            );
            expected.extend(&seeds);
            expected.sort_unstable();
            expected.dedup();
            assert!(expected.len() > seeds.len());
            assert_eq!(*result, expected);
        }
    );

    let ladder = ladder(LADDER_RUNGS);
    let rail = |rung: usize| NodeId::from_index(LADDER_RUNGS + 1 + rung);
    let ladder_seeds: Vec<NodeId> = [NodeId::from_index(0)]
        .into_iter()
        .chain((0..=LADDER_RUNGS).step_by(LADDER_STRIDE).map(rail))
        .collect();
    let mut closure_on_ladder = Vec::new();
    benchmark_case!(
        suite,
        "api_weak_control_closure_ladder",
        || {
            weak_control_closure(
                &ladder,
                ladder_seeds.iter().copied(),
                &mut scratch,
                &mut closure_on_ladder,
            );
            closure_on_ladder.clone()
        },
        |result: &Vec<NodeId>| {
            // Every `a_i` up to one stride before the last seed of the `b`
            // rail chooses between two seeds: directly, or by way of the
            // `a_j` after it that already decides.
            let last = LADDER_RUNGS / LADDER_STRIDE * LADDER_STRIDE;
            let expected: Vec<NodeId> = (0..=last - LADDER_STRIDE)
                .map(NodeId::from_index)
                .chain(ladder_seeds[1..].iter().copied())
                .collect();
            assert_eq!(*result, expected);
        }
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    register_nontermination_sensitive(suite);
    register_weak_control_closure(suite);
}
