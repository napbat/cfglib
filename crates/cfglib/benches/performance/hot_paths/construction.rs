use cfglib::{Cfg, EdgeKind, FlowEffect, Graph};

use super::NODE_COUNT;
use crate::fixtures::{
    BuilderInst, branchy_cfg, branchy_graph, build_conditional_break_chain,
    build_eight_case_switch_chain, build_if_else_chain, build_two_case_switch_chain,
};
use crate::harness::{BenchmarkSuite, benchmark_case};
use crate::structural_oracles::{assert_branchy_cfg, assert_branchy_graph, assert_builder_cfg};

const BUILDER_REGION_COUNT: usize = 2_048;

fn register_branchy(suite: &mut BenchmarkSuite<'_>) {
    benchmark_case!(
        suite,
        "cfg_build_branchy",
        || branchy_cfg(NODE_COUNT),
        |result: &Cfg<u32>| assert_branchy_cfg(result, NODE_COUNT)
    );
    benchmark_case!(
        suite,
        "directed_build_branchy",
        || branchy_graph(NODE_COUNT),
        |result: &Graph<(), ()>| assert_branchy_graph(result, NODE_COUNT)
    );
}

fn register_builder_chains(suite: &mut BenchmarkSuite<'_>) {
    benchmark_case!(
        suite,
        "cfg_builder_if_else_chain",
        || build_if_else_chain(BUILDER_REGION_COUNT),
        |result: &Cfg<BuilderInst>| assert_builder_cfg(
            result,
            1 + 3 * BUILDER_REGION_COUNT,
            4 * BUILDER_REGION_COUNT,
            &[
                (FlowEffect::ConditionalOpen, BUILDER_REGION_COUNT),
                (FlowEffect::ConditionalAlternate, BUILDER_REGION_COUNT),
                (FlowEffect::Fallthrough, 2 * BUILDER_REGION_COUNT),
            ],
            &[
                (EdgeKind::ConditionalTrue, BUILDER_REGION_COUNT),
                (EdgeKind::ConditionalFalse, BUILDER_REGION_COUNT),
                (EdgeKind::Fallthrough, 2 * BUILDER_REGION_COUNT),
            ],
        )
    );
    benchmark_case!(
        suite,
        "cfg_builder_conditional_break_chain",
        || { build_conditional_break_chain(BUILDER_REGION_COUNT) },
        |result: &Cfg<BuilderInst>| assert_builder_cfg(
            result,
            1 + 4 * BUILDER_REGION_COUNT,
            5 * BUILDER_REGION_COUNT,
            &[
                (FlowEffect::LoopOpen, BUILDER_REGION_COUNT),
                (FlowEffect::ConditionalBreak, BUILDER_REGION_COUNT),
            ],
            &[
                (EdgeKind::Fallthrough, BUILDER_REGION_COUNT),
                (EdgeKind::ConditionalTrue, BUILDER_REGION_COUNT),
                (EdgeKind::ConditionalFalse, BUILDER_REGION_COUNT),
                (EdgeKind::Back, BUILDER_REGION_COUNT),
                (EdgeKind::Unconditional, BUILDER_REGION_COUNT),
            ],
        )
    );
    benchmark_case!(
        suite,
        "cfg_builder_two_case_switch_chain",
        || { build_two_case_switch_chain(BUILDER_REGION_COUNT) },
        |result: &Cfg<BuilderInst>| assert_builder_cfg(
            result,
            1 + 3 * BUILDER_REGION_COUNT,
            4 * BUILDER_REGION_COUNT,
            &[
                (FlowEffect::SwitchOpen, BUILDER_REGION_COUNT),
                (FlowEffect::SwitchCase, BUILDER_REGION_COUNT),
                (FlowEffect::Fallthrough, 2 * BUILDER_REGION_COUNT),
            ],
            &[
                (EdgeKind::SwitchCase, 2 * BUILDER_REGION_COUNT),
                (EdgeKind::Fallthrough, 2 * BUILDER_REGION_COUNT),
            ],
        )
    );
    benchmark_case!(
        suite,
        "cfg_builder_eight_case_switch_chain",
        || { build_eight_case_switch_chain(BUILDER_REGION_COUNT / 4) },
        |result: &Cfg<BuilderInst>| {
            let regions = BUILDER_REGION_COUNT / 4;
            assert_builder_cfg(
                result,
                1 + 9 * regions,
                16 * regions,
                &[
                    (FlowEffect::SwitchOpen, regions),
                    (FlowEffect::SwitchCase, 7 * regions),
                    (FlowEffect::Fallthrough, 8 * regions),
                ],
                &[
                    (EdgeKind::SwitchCase, 8 * regions),
                    (EdgeKind::Fallthrough, 8 * regions),
                ],
            );
        }
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    register_branchy(suite);
    register_builder_chains(suite);
}
