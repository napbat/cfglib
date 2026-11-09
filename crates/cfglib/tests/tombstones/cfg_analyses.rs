//! `CfgMetrics` over a `Cfg` with removed blocks, checked against a reference
//! algorithm and across a compaction.

use cfglib::{BlockId, Cfg, CfgMetrics};

use crate::oracle::Reference;
use crate::topology::{GENERAL, Op, Realizations, SEEDS, Slots, Topology};

fn realize(seed: u64, shape: crate::topology::Shape) -> Realizations {
    Realizations::of_topology(Topology::random(seed, shape)).labeled(format!("seed {seed}"))
}

fn cfg_metrics_match(
    cfg: &Cfg<Op>,
    _slots: &Slots<BlockId>,
    reference: &Reference<'_>,
    label: &str,
) {
    let metrics = CfgMetrics::compute(cfg);
    let expected = reference.metrics();
    assert_eq!(
        metrics.graph.node_count, expected.node_count,
        "{label}: node_count"
    );
    assert_eq!(
        metrics.graph.node_count,
        cfg.block_count(),
        "{label}: node_count is block_count"
    );
    assert_eq!(
        metrics.graph.edge_count, expected.edge_count,
        "{label}: edge_count"
    );
    assert_eq!(
        metrics.graph.edge_count,
        cfg.edge_count(),
        "{label}: edge_count is edge_count"
    );
    assert_eq!(
        metrics.graph.reachable_node_count, expected.reachable_node_count,
        "{label}: reachable_node_count"
    );
    assert_eq!(
        metrics.graph.unreachable_node_count, expected.unreachable_node_count,
        "{label}: unreachable_node_count"
    );
    assert_eq!(
        metrics.graph.cyclomatic_complexity, expected.cyclomatic_complexity,
        "{label}: cyclomatic_complexity"
    );
    assert_eq!(
        metrics.graph.exit_count, expected.exit_count,
        "{label}: exit_count"
    );
    assert_eq!(
        metrics.instruction_count, expected.instruction_count,
        "{label}: instruction_count"
    );
    let instructions = f64::from(u32::try_from(expected.instruction_count).expect("few"));
    let blocks = f64::from(u32::try_from(expected.node_count).expect("few"));
    assert!(
        (metrics.avg_instructions_per_block - instructions / blocks).abs() < f64::EPSILON,
        "{label}: the average divides by the live blocks it reports"
    );
}

#[test]
fn cfg_metrics_count_live_blocks_and_instructions() {
    for seed in SEEDS {
        realize(seed, GENERAL).run_cfgs(cfg_metrics_match);
    }
}

/// Every instruction-level analysis, for a case built elsewhere (for example
/// after a transform).
pub fn check_cfg_analyses(realized: &Realizations) {
    realized.run_cfgs(cfg_metrics_match);
}
