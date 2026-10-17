use cfglib::ir::rtl::{BlockSummaries, ParamPlacement, RootLiveness, RootTable};
use cfglib::{
    BlockId, Cfg, DominanceFrontiers, DominatorTree, Graph, PostDominatorTree, PostNode,
    control_dependence_graph,
};

use super::NODE_COUNT;
use crate::analysis_oracles::{
    assert_branchy_dominators, assert_branchy_post_dominators, assert_control_dependence_graph,
    assert_dominance_frontiers,
};
use crate::fixtures::{linear_cfg, many_exit_cfg};
use crate::harness::{BenchmarkSuite, benchmark_case};
use crate::rtl_placement::{SyntheticFunction, assert_param_placement};

fn register_dominators(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>) {
    let cfg_dominators = DominatorTree::compute(cfg);
    let linear = linear_cfg(NODE_COUNT / 2);
    let linear_dominators = DominatorTree::compute(&linear);
    benchmark_case!(
        suite,
        "cfg_dominators",
        || DominatorTree::compute(cfg),
        |result: &DominatorTree| assert_branchy_dominators(result, NODE_COUNT)
    );
    benchmark_case!(
        suite,
        "cfg_dominance_frontiers",
        || { DominanceFrontiers::compute(cfg, &cfg_dominators) },
        |result: &DominanceFrontiers| assert_dominance_frontiers(result, cfg, &cfg_dominators,)
    );
    benchmark_case!(
        suite,
        "cfg_dominator_queries_linear",
        || {
            let root = BlockId::from_index(0);
            (0..NODE_COUNT / 2)
                .filter(|&index| linear_dominators.dominates(root, BlockId::from_index(index)))
                .count()
        },
        |result: &usize| assert_eq!(*result, NODE_COUNT / 2)
    );
}

fn register_post_dominators(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>) {
    let cfg_post_dominators = PostDominatorTree::compute(cfg);
    let many_exits = many_exit_cfg(NODE_COUNT - 1);
    benchmark_case!(
        suite,
        "cfg_post_dominators",
        || PostDominatorTree::compute(cfg),
        |result: &PostDominatorTree| assert_branchy_post_dominators(result, NODE_COUNT)
    );
    benchmark_case!(
        suite,
        "cfg_post_dominators_many_exits",
        || { PostDominatorTree::compute(&many_exits) },
        |result: &PostDominatorTree| {
            for index in 0..NODE_COUNT {
                let block = BlockId::from_index(index);
                assert!(result.contains(block));
                assert_eq!(result.ipdom(block), Some(PostNode::Exit));
            }
        }
    );
    benchmark_case!(
        suite,
        "cfg_control_dependence_graph",
        || { control_dependence_graph(cfg, &cfg_post_dominators) },
        |result: &Graph<BlockId, ()>| assert_control_dependence_graph(
            result,
            cfg,
            &cfg_post_dominators,
        )
    );
}

fn register_rtl(suite: &mut BenchmarkSuite<'_>) {
    let rtl = SyntheticFunction::new(NODE_COUNT);
    let rtl_roots = RootTable::collect(&rtl);
    let rtl_summaries = BlockSummaries::compute(&rtl, &rtl_roots);
    let rtl_liveness = RootLiveness::compute(&rtl, &rtl_roots, &rtl_summaries);
    let rtl_dominators = DominatorTree::compute(&rtl);
    let rtl_frontiers = DominanceFrontiers::compute(&rtl, &rtl_dominators);
    benchmark_case!(
        suite,
        "rtl_param_placement",
        || {
            ParamPlacement::compute(
                &rtl,
                &rtl_dominators,
                &rtl_frontiers,
                &rtl_roots,
                &rtl_summaries,
                &rtl_liveness,
            )
        },
        |result: &ParamPlacement<_>| assert_param_placement(
            result,
            &rtl,
            &rtl_dominators,
            &rtl_roots,
            &rtl_liveness,
        )
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>) {
    register_dominators(suite, cfg);
    register_post_dominators(suite, cfg);
    register_rtl(suite);
}
