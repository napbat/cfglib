use cfglib::{
    Cfg, CfgDiff, EdgeKind, block_nesting_depths, cfg_block_nesting_depths, detect_cfg_patterns,
    detect_patterns,
};

use super::BenchmarkSuite;
use super::fixtures::ApiInst;
use crate::fixtures::{branchy_cfg, linear_cfg};
use crate::harness::benchmark_case;

const BLOCK_COUNT: usize = 256;
/// Cases of the switch in the diff case.
const SWITCH_CASES: usize = 4_096;
/// Blocks of the branchy CFG in the diff case.
const BRANCHY_BLOCKS: usize = 1_024;

fn loop_cfg() -> Cfg<ApiInst> {
    let mut cfg = Cfg::new();
    let header = cfg.new_block();
    let body = cfg.new_block();
    let exit = cfg.new_block();
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, header, EdgeKind::Back);
    cfg
}

fn register_metrics_and_patterns(suite: &mut BenchmarkSuite<'_>) {
    let cfg = loop_cfg();
    let graph = linear_cfg(BLOCK_COUNT);
    let patterns = linear_cfg(BLOCK_COUNT);

    benchmark_case!(
        suite,
        "api_block_nesting_depths",
        covers[block_nesting_depths],
        || block_nesting_depths(&cfg),
        |depths: &Vec<usize>| assert!(depths.contains(&1))
    );
    benchmark_case!(
        suite,
        "api_cfg_block_nesting_depths",
        covers[cfg_block_nesting_depths],
        || cfg_block_nesting_depths(&cfg),
        |depths: &Vec<usize>| assert!(depths.contains(&1))
    );
    benchmark_case!(
        suite,
        "api_detect_patterns",
        covers[detect_patterns],
        || detect_patterns(&graph),
        |result: &Vec<_>| assert_ne!(result.as_slice(), [])
    );
    benchmark_case!(
        suite,
        "api_detect_cfg_patterns",
        covers[detect_cfg_patterns],
        || detect_cfg_patterns(&patterns),
        |result: &Vec<_>| assert_ne!(result.as_slice(), [])
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    register_metrics_and_patterns(suite);
    register_cfg_diff(suite);
}

/// `entry -> dispatch`, `cases` switch edges from `dispatch`, every case
/// falling through to one join.
fn switch_cfg(cases: usize) -> Cfg<u32> {
    let mut cfg = Cfg::new();
    let dispatch = cfg.new_block();
    let join = cfg.new_block();
    cfg.add_edge(cfg.entry(), dispatch, EdgeKind::Fallthrough);
    for _ in 0..cases {
        let case = cfg.new_block();
        cfg.add_edge(dispatch, case, EdgeKind::SwitchCase);
        cfg.add_edge(case, join, EdgeKind::Fallthrough);
    }
    cfg
}

fn register_cfg_diff(suite: &mut BenchmarkSuite<'_>) {
    // A switch that gained a case: the dispatch and the join change shape, so
    // every case block is matched by the fallback, all from one shape bucket.
    let (old, new) = (switch_cfg(SWITCH_CASES), switch_cfg(SWITCH_CASES + 1));
    benchmark_case!(
        suite,
        "cfg_diff_switch_gained_case",
        || CfgDiff::compute(&old, &new),
        |diff: &CfgDiff| assert_eq!(diff.matched.len(), SWITCH_CASES + 1)
    );
    // Two copies of one branchy CFG: the walk from the entry matches it.
    let left = branchy_cfg(BRANCHY_BLOCKS);
    let right = branchy_cfg(BRANCHY_BLOCKS);
    benchmark_case!(
        suite,
        "cfg_diff_branchy_identical",
        || CfgDiff::compute(&left, &right),
        |diff: &CfgDiff| assert!(diff.is_identical())
    );
}
