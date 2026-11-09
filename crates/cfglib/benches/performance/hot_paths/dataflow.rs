use cfglib::{Cfg, Facts, Graph, solve_node_problem, solve_problem};

use super::NODE_COUNT;
use crate::analysis_oracles::{
    assert_bool_cfg_facts, assert_bool_node_facts, assert_wide_cfg_facts, assert_wide_node_facts,
};
use crate::fixtures::{
    CfgReachability, Reachability, WideCfgFact, WideNodeFact, branchy_cfg, branchy_graph,
};
use crate::harness::{BenchmarkSuite, benchmark_case};

pub(super) fn register(suite: &mut BenchmarkSuite<'_>, cfg: &Cfg<u32>, graph: &Graph<(), ()>) {
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
