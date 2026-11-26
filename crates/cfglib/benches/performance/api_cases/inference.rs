use cfglib::{
    BlockFrequencies, Cfg, DominatorTree, FREQUENCY_SHIFT, FlowError, FrequencyConfig,
    InferenceCosts, LoopForest,
};

use super::BenchmarkSuite;
use super::frequency::looped_cfg;
use crate::harness::benchmark_case;

/// Noisy, partial block samples of `cfg` run a thousand times: the counts its
/// branch weights imply, with every third block 20% high and every fifth one
/// not sampled.
fn samples(cfg: &Cfg<u32>) -> Vec<Option<u64>> {
    let dominators = DominatorTree::compute(cfg);
    let loops = LoopForest::compute(cfg, &dominators);
    let frequencies = BlockFrequencies::compute(
        cfg,
        &loops,
        |edge| cfg.edge(edge).weight(),
        FrequencyConfig::default(),
    );
    cfg.block_ids()
        .map(|block| {
            let runs = (frequencies.block(block) * 1_000) >> FREQUENCY_SHIFT;
            match block.index() % 15 {
                slot if slot % 5 == 0 => None,
                slot if slot % 3 == 0 => Some(runs + runs / 5),
                _ => Some(runs),
            }
        })
        .collect()
}

/// Inference over the samples, checked: every block but the entry and the
/// exit runs as often as it is entered and as often as it is left.
macro_rules! infer_counts_case {
    ($suite:expr, $name:literal, $blocks:expr) => {{
        let cfg = looped_cfg($blocks);
        let estimate = samples(&cfg);
        let exit = cfg.block_ids().last().expect("the CFG has blocks");
        benchmark_case!(
            $suite,
            $name,
            || BlockFrequencies::infer(
                &cfg,
                |block| block == exit,
                |block| estimate[block.index()],
                |_| None,
                |edge| cfg.edge(edge).weight(),
                &InferenceCosts::default(),
            ),
            |counts: &Result<BlockFrequencies, FlowError>| {
                let counts = counts.as_ref().expect("the samples fit the network");
                for block in cfg.block_ids() {
                    if block == cfg.entry() || block == exit {
                        continue;
                    }
                    let into: u64 = cfg.incoming(block).map(|edge| counts.edge(edge)).sum();
                    let out: u64 = cfg.outgoing(block).map(|edge| counts.edge(edge)).sum();
                    assert_eq!((into, out), (counts.block(block), counts.block(block)));
                }
                assert!(counts.block(exit) > 0);
            }
        );
    }};
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    infer_counts_case!(suite, "api_infer_counts_1k", 1_000);
    infer_counts_case!(suite, "api_infer_counts_10k", 10_000);
}
