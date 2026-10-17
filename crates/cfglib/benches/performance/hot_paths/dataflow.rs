use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, ConstValue, ConstantPropagation, DominatorTree, Facts, Graph, PhiPlacements,
    SccpAnalysis, SsaForm, SsaValue, ValueNumbering, solve_node_problem, solve_problem,
};

use super::NODE_COUNT;
use crate::analysis_oracles::{
    assert_bool_cfg_facts, assert_bool_node_facts, assert_linear_ssa, assert_phi_placements,
    assert_phi_ssa, assert_wide_cfg_facts, assert_wide_node_facts,
};
use crate::fixtures::{
    CfgReachability, Reachability, WideCfgFact, WideNodeFact, branchy_cfg, branchy_graph,
    fixture_u32, independent_constants, linear_constants, phi_storm_cfg, unreachable_constants,
};
use crate::harness::{BenchmarkSuite, benchmark_case};

fn register_fixpoints(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>, graph: &Graph<(), ()>) {
    let wide_cfg = branchy_cfg(1_024);
    let wide_graph = branchy_graph(1_024);
    benchmark_case!(
        suite,
        "directed_node_fixpoint_bool",
        || {
            solve_node_problem(graph, &Reachability)
                .expect("an unbounded solve cannot exceed its step limit")
        },
        |result: &Facts<bool>| assert_bool_node_facts(result, NODE_COUNT)
    );
    benchmark_case!(
        suite,
        "directed_node_fixpoint_wide",
        || {
            solve_node_problem(&wide_graph, &WideNodeFact)
                .expect("an unbounded solve cannot exceed its step limit")
        },
        |result: &Facts<Vec<u64>>| assert_wide_node_facts(result, 1_024)
    );
    benchmark_case!(
        suite,
        "cfg_fixpoint_bool",
        || {
            solve_problem(cfg, &CfgReachability)
                .expect("an unbounded solve cannot exceed its step limit")
        },
        |result: &Facts<bool>| assert_bool_cfg_facts(result, NODE_COUNT)
    );
    benchmark_case!(
        suite,
        "cfg_fixpoint_wide",
        || {
            solve_problem(&wide_cfg, &WideCfgFact)
                .expect("an unbounded solve cannot exceed its step limit")
        },
        |result: &Facts<Vec<u64>>| assert_wide_cfg_facts(result, 1_024)
    );
}

fn register_ssa_and_numbering(suite: &mut BenchmarkSuite<'_>) {
    let unreachable_ssa_cfg = unreachable_constants(256, 256);
    let unreachable_ssa_dominators = DominatorTree::compute(&unreachable_ssa_cfg);
    let linear_constant_cfg = linear_constants(2_048);
    let linear_constant_dominators = DominatorTree::compute(&linear_constant_cfg);
    let phi_storm = phi_storm_cfg(32, 128);
    let phi_storm_dominators = DominatorTree::compute(&phi_storm);
    benchmark_case!(
        suite,
        "cfg_build_ssa_unreachable",
        || SsaForm::compute(&unreachable_ssa_cfg, &unreachable_ssa_dominators),
        |result: &SsaForm<u32>| {
            let block_count = 256 + 2 * 256;
            assert_eq!(result.blocks().len(), block_count);
            for index in 0..block_count {
                let block = BlockId::from_index(index);
                assert_eq!(result.block(block).block, block);
                assert_eq!(result.block(block).instructions.len(), 1);
            }
            assert!(result.max_version(&0) >= block_count);
        }
    );
    benchmark_case!(
        suite,
        "cfg_build_ssa_linear",
        || SsaForm::compute(&linear_constant_cfg, &linear_constant_dominators),
        |result: &SsaForm<u32>| assert_linear_ssa(result, 2_048)
    );
    benchmark_case!(
        suite,
        "cfg_place_phis_phi_storm",
        || PhiPlacements::compute(&phi_storm, &phi_storm_dominators),
        |result: &PhiPlacements<u32>| assert_phi_placements(result, 32, 128)
    );
    benchmark_case!(
        suite,
        "cfg_build_ssa_phi_storm",
        || SsaForm::compute(&phi_storm, &phi_storm_dominators),
        |result: &SsaForm<u32>| assert_phi_ssa(result, &phi_storm, 32, 128)
    );
    benchmark_case!(
        suite,
        "cfg_global_value_numbering_linear",
        || ValueNumbering::compute(&linear_constant_cfg, &linear_constant_dominators),
        |result: &ValueNumbering| {
            assert_eq!(result.blocks.len(), 2_048);
            assert_eq!(result.value_count, 2_048);
            for index in 0..2_048 {
                let block = result
                    .blocks
                    .get(&BlockId::from_index(index))
                    .expect("value numbering omitted a block");
                assert_eq!(block.inst_vn, [Some(fixture_u32(index))]);
                assert_eq!(block.redundant.len(), 0);
            }
        }
    );
}

fn register_constants(suite: &mut BenchmarkSuite<'_>) {
    let constants = independent_constants(1_024);
    let constant_dominators = DominatorTree::compute(&constants);
    let constant_ssa = SsaForm::compute(&constants, &constant_dominators);
    benchmark_case!(
        suite,
        "cfg_sccp_independent_constants",
        || SccpAnalysis::compute(&constants, &constant_ssa),
        |result: &SccpAnalysis<u32, u64>| {
            assert_eq!(result.reachable_blocks, BTreeSet::from([constants.entry()]));
            assert!(result.executable_edges.is_empty());
            assert_eq!(result.values.len(), 1_024);
            for variable in 0..1_024_u32 {
                assert_eq!(
                    result.values.get(&SsaValue::new(variable, 1)),
                    Some(&ConstValue::Const(u64::from(variable)))
                );
            }
        }
    );
    benchmark_case!(
        suite,
        "cfg_constprop_independent_constants",
        || { ConstantPropagation::compute(&constants) },
        |result: &ConstantPropagation<u32, u64>| {
            // Every variable is unknown when the function starts.
            assert_eq!(result.fact_in(constants.entry()).len(), 1_024);
            assert!(
                result
                    .fact_in(constants.entry())
                    .values()
                    .all(|value| *value == ConstValue::Bottom)
            );
            assert_eq!(result.fact_out(constants.entry()).len(), 1_024);
            for variable in 0..1_024_u32 {
                assert_eq!(
                    result.fact_out(constants.entry()).get(&variable),
                    Some(&ConstValue::Const(u64::from(variable)))
                );
            }
        }
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>, graph: &Graph<(), ()>) {
    register_fixpoints(suite, cfg, graph);
    register_ssa_and_numbering(suite);
    register_constants(suite);
}
