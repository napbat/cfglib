use cfglib::{
    BlockFrequencies, Cfg, DominatorTree, EdgeKind, FREQUENCY_SHIFT, FrequencyConfig, LoopForest,
    Probability,
};

use super::BenchmarkSuite;
use crate::harness::benchmark_case;

/// Blocks of one group: an outer header and latch around eight loops of five
/// blocks.
const GROUP_BLOCKS: usize = 42;

/// About `block_count` blocks in groups of loops. Each loop is a header that
/// branches 60/40 into a diamond whose join returns to the header nine times
/// in ten; each group of eight loops sits in an outer loop that returns half
/// the time.
fn looped_cfg(block_count: usize) -> Cfg<u32> {
    let mut cfg = Cfg::new();
    let mut previous = cfg.entry();
    for _ in 0..block_count.div_ceil(GROUP_BLOCKS) {
        let outer = cfg.new_block();
        cfg.add_edge(previous, outer, EdgeKind::Fallthrough);
        previous = outer;
        for _ in 0..8 {
            let [header, left, right, join] = [(); 4].map(|()| cfg.new_block());
            cfg.add_edge(previous, header, EdgeKind::Fallthrough);
            cfg.add_weighted_edge(header, left, EdgeKind::ConditionalTrue, 0.6);
            cfg.add_weighted_edge(header, right, EdgeKind::ConditionalFalse, 0.4);
            cfg.add_edge(left, join, EdgeKind::Fallthrough);
            cfg.add_edge(right, join, EdgeKind::Fallthrough);
            cfg.add_weighted_edge(join, header, EdgeKind::Back, 0.9);
            let next = cfg.new_block();
            cfg.add_weighted_edge(join, next, EdgeKind::ConditionalFalse, 0.1);
            previous = next;
        }
        let latch = cfg.new_block();
        cfg.add_edge(previous, latch, EdgeKind::Fallthrough);
        cfg.add_weighted_edge(latch, outer, EdgeKind::Back, 0.5);
        previous = cfg.new_block();
        cfg.add_weighted_edge(latch, previous, EdgeKind::ConditionalFalse, 0.5);
    }
    cfg
}

/// The frequencies of `cfg`, checked: the entry runs once, and so does the
/// last block, which every path out of the loops reaches.
macro_rules! block_frequencies_case {
    ($suite:expr, $name:literal, $blocks:expr) => {{
        let cfg = looped_cfg($blocks);
        let dominators = DominatorTree::compute(&cfg);
        let loops = LoopForest::compute(&cfg, &dominators);
        let exit = cfg.block_ids().last().expect("the CFG has blocks");
        benchmark_case!(
            $suite,
            $name,
            || BlockFrequencies::compute(
                &cfg,
                &loops,
                |edge| cfg.edge(edge).weight().and_then(Probability::from_f64),
                FrequencyConfig::new(),
            ),
            |frequencies: &BlockFrequencies| {
                let once = 1_u64 << FREQUENCY_SHIFT;
                assert_eq!(frequencies.block(cfg.entry()), once);
                assert!(frequencies.block(exit).abs_diff(once) <= 64);
            }
        );
    }};
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    block_frequencies_case!(suite, "api_block_frequencies_1k", 1_000);
    block_frequencies_case!(suite, "api_block_frequencies_10k", 10_000);
}
