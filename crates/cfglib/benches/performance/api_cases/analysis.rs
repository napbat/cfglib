use cfglib::{
    Cfg, EdgeKind, block_nesting_depths, cfg_block_nesting_depths, detect_cfg_patterns,
    detect_patterns,
};

use super::BenchmarkSuite;
use super::fixtures::ApiInst;
use crate::fixtures::linear_cfg;
use crate::harness::benchmark_case;

const BLOCK_COUNT: usize = 256;

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
}
