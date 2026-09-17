use cfglib::{
    AstNode, BlockId, Cfg, EdgeKind, FlowEffect, Purity, block_nesting_depths, block_purities,
    block_purity, cfg_block_nesting_depths, cfg_purity, detect_cfg_patterns,
    detect_explicit_tail_calls, detect_patterns, detect_switch_tables, detect_tail_calls, lift,
    lift_borrowed, lift_borrowed_with_report, lift_predicated, lift_with_report,
    recover_block_expressions, recover_expressions, recover_switch_tables,
    set_uniform_edge_weights, verify,
};

use super::BenchmarkSuite;
use super::fixtures::{ApiInst, dataflow_cfg};
use crate::fixtures::{branchy_cfg, linear_cfg};
use crate::harness::benchmark_case;

const BLOCK_COUNT: usize = 256;
const INSTRUCTIONS_PER_BLOCK: usize = 8;

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

fn switch_cfg() -> (Cfg<ApiInst>, Vec<BlockId>) {
    let mut cfg = Cfg::new();
    let targets: Vec<_> = (0..3).map(|_| cfg.new_block()).collect();
    let mut dispatch = ApiInst::control(FlowEffect::IndirectJump);
    dispatch.switch_targets = Some((vec![0, 1], Some(2)));
    cfg.block_mut(cfg.entry()).push(dispatch);
    cfg.add_edge(cfg.entry(), targets[0], EdgeKind::IndirectJump);
    (cfg, targets)
}

fn tail_call_cfg() -> Cfg<ApiInst> {
    let mut cfg = Cfg::new();
    let exit = cfg.new_block();
    let mut call = ApiInst::control(FlowEffect::Call);
    call.callee = Some(7);
    call.tail_call = true;
    cfg.block_mut(cfg.entry()).push(call);
    cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);
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
        |result: &Vec<_>| assert!(!result.is_empty())
    );
    benchmark_case!(
        suite,
        "api_detect_cfg_patterns",
        covers[detect_cfg_patterns],
        || detect_cfg_patterns(&patterns),
        |result: &Vec<_>| assert!(!result.is_empty())
    );
}

fn register_purity(suite: &mut BenchmarkSuite<'_>) {
    let cfg = dataflow_cfg(BLOCK_COUNT, INSTRUCTIONS_PER_BLOCK);
    benchmark_case!(
        suite,
        "api_block_purity",
        covers[block_purity],
        || block_purity(&cfg, cfg.entry()),
        |purity| assert!(matches!(purity, Purity::Impure(_)))
    );
    benchmark_case!(
        suite,
        "api_cfg_purity",
        covers[cfg_purity],
        || cfg_purity(&cfg),
        |purity| assert!(matches!(purity, Purity::Impure(_)))
    );
    benchmark_case!(
        suite,
        "api_block_purities",
        covers[block_purities],
        || block_purities(&cfg),
        |purities: &Vec<_>| assert_eq!(purities.len(), BLOCK_COUNT)
    );
}

fn register_expressions_and_ast(suite: &mut BenchmarkSuite<'_>) {
    let cfg = dataflow_cfg(BLOCK_COUNT, INSTRUCTIONS_PER_BLOCK);
    benchmark_case!(
        suite,
        "api_recover_block_expressions",
        covers[recover_block_expressions],
        || recover_block_expressions(&cfg, cfg.entry()),
        |trees| assert_eq!(trees.block, cfg.entry())
    );
    benchmark_case!(
        suite,
        "api_recover_expressions",
        covers[recover_expressions],
        || recover_expressions(&cfg),
        |trees: &Vec<_>| assert_eq!(trees.len(), BLOCK_COUNT)
    );
    benchmark_case!(
        suite,
        "api_lift",
        covers[lift],
        || lift(&cfg),
        |ast| assert!(matches!(ast, AstNode::Sequence { .. }))
    );
    benchmark_case!(
        suite,
        "api_lift_borrowed",
        covers[lift_borrowed],
        || lift_borrowed(&cfg),
        |ast| assert!(matches!(ast, AstNode::Sequence { .. }))
    );
    benchmark_case!(
        suite,
        "api_lift_borrowed_with_report",
        covers[lift_borrowed_with_report],
        || lift_borrowed_with_report(&cfg),
        |lifted: &(AstNode<&ApiInst>, cfglib::LiftReport)| {
            assert!(matches!(lifted.0, AstNode::Sequence { .. }));
        }
    );
    benchmark_case!(
        suite,
        "api_lift_with_report",
        covers[lift_with_report],
        || lift_with_report(&cfg),
        |lifted: &(AstNode<ApiInst>, cfglib::LiftReport)| {
            assert!(matches!(lifted.0, AstNode::Sequence { .. }));
        }
    );

    let mut predicated = dataflow_cfg(BLOCK_COUNT, INSTRUCTIONS_PER_BLOCK);
    let block_ids: Vec<_> = predicated.block_ids().collect();
    for block in block_ids {
        for instruction in predicated.block_mut(block).instructions_mut() {
            instruction.uses.push(0);
            instruction.predicate = Some((0, true));
        }
    }
    benchmark_case!(
        suite,
        "api_lift_predicated",
        covers[lift_predicated],
        || lift_predicated(&predicated),
        |ast| assert!(matches!(ast, AstNode::Sequence { .. }))
    );
}

fn register_switch_and_calls(suite: &mut BenchmarkSuite<'_>) {
    let (switch, targets) = switch_cfg();
    let tables = detect_switch_tables(&switch);
    benchmark_case!(
        suite,
        "api_detect_switch_tables",
        covers[detect_switch_tables],
        || detect_switch_tables(&switch),
        |result: &Vec<_>| assert_eq!(result.len(), 1)
    );
    benchmark_case!(
        suite,
        "api_recover_switch_tables",
        covers[recover_switch_tables],
        || {
            let mut candidate = switch.clone();
            let recovered = recover_switch_tables(&mut candidate, &tables, |token| {
                usize::try_from(*token)
                    .ok()
                    .and_then(|index| targets.get(index))
                    .copied()
            });
            (candidate, recovered)
        },
        |(candidate, recovered)| {
            assert_eq!(recovered[0].case_count, 2);
            assert!(verify(candidate).is_ok());
        }
    );

    let calls = tail_call_cfg();
    benchmark_case!(
        suite,
        "api_detect_tail_calls",
        covers[detect_tail_calls],
        || detect_tail_calls(&calls),
        |result: &Vec<_>| assert_eq!(result.len(), 1)
    );
    benchmark_case!(
        suite,
        "api_detect_explicit_tail_calls",
        covers[detect_explicit_tail_calls],
        || detect_explicit_tail_calls(&calls),
        |result: &Vec<_>| assert_eq!(result.len(), 1)
    );
}

fn register_profile(suite: &mut BenchmarkSuite<'_>) {
    let cfg = branchy_cfg(BLOCK_COUNT);
    benchmark_case!(
        suite,
        "api_set_uniform_edge_weights",
        covers[set_uniform_edge_weights],
        || {
            let mut candidate = cfg.clone();
            set_uniform_edge_weights(&mut candidate);
            candidate
        },
        |candidate| {
            assert!(verify(candidate).is_ok());
            assert!(candidate.edges().all(|edge| edge.weight().is_some()));
        }
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    register_metrics_and_patterns(suite);
    register_purity(suite);
    register_expressions_and_ast(suite);
    register_switch_and_calls(suite);
    register_profile(suite);
}
