use cfglib::{
    BlockHashes, BlockId, CascadeDiff, Cfg, CfgDiff, DominatorTree, EdgeKind, LoopForest,
    MatchLevel, RegionDiff, detect_patterns, empty_trampolines,
};

use super::BenchmarkSuite;
use super::fixtures::ApiInst;
use crate::fixtures::{branchy_cfg, fixture_u32, linear_cfg};
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

/// `count` blocks in a line, each past the entry without instructions, so
/// every one but the last is an empty trampoline.
fn hollow_line(count: usize) -> Cfg<u32> {
    let mut cfg = Cfg::new();
    let mut previous = cfg.entry();
    cfg.block_mut(previous).push(0);
    for _ in 1..count {
        let next = cfg.new_block();
        cfg.add_edge(previous, next, EdgeKind::Fallthrough);
        previous = next;
    }
    cfg
}

fn register_metrics_and_patterns(suite: &mut BenchmarkSuite<'_>) {
    let cfg = loop_cfg();
    let graph = linear_cfg(BLOCK_COUNT);
    let hollow = hollow_line(BLOCK_COUNT);

    benchmark_case!(
        suite,
        "api_loop_node_depths",
        || LoopForest::compute(&cfg, &DominatorTree::compute(&cfg)).node_depths(),
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
        "api_empty_trampolines",
        covers[empty_trampolines],
        || empty_trampolines(&hollow),
        |result: &Vec<BlockId>| assert_eq!(result.len(), BLOCK_COUNT - 2)
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

/// `switch_cfg(cases)` with every block holding its own index, so each block
/// has content of its own.
fn indexed_switch_cfg(cases: usize) -> Cfg<u32> {
    let mut cfg = switch_cfg(cases);
    let blocks: Vec<BlockId> = cfg.block_ids().collect();
    for block in blocks {
        cfg.block_mut(block).push(fixture_u32(block.index()));
    }
    cfg
}

/// A consumer's content hash: FNV-1a over the instructions.
fn content_hashes(_: BlockId, instructions: &[u32]) -> BlockHashes {
    let strict = instructions
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |hash, &op| {
            (hash ^ u64::from(op)).wrapping_mul(0x0100_0000_01b3)
        });
    BlockHashes {
        strict,
        loose: strict,
    }
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
    // The grown switch again, with content: every case pairs by its full
    // hash, and the changed dispatch and join by their strict hashes.
    let (old, new) = (
        indexed_switch_cfg(SWITCH_CASES),
        indexed_switch_cfg(SWITCH_CASES + 1),
    );
    benchmark_case!(
        suite,
        "cfg_diff_hash_cascade",
        || CascadeDiff::compute(&old, &new, content_hashes, content_hashes),
        |diff: &CascadeDiff| {
            assert_eq!(diff.diff.matched.len(), SWITCH_CASES + 3);
            let full = diff
                .levels
                .iter()
                .filter(|&&level| level == MatchLevel::Full);
            assert_eq!(full.count(), SWITCH_CASES);
        }
    );
    // The branchy CFG with every fourth block split in two: each half pair
    // becomes a chain, and each split comes back as a 1:2 region.
    let mut split = branchy_cfg(BRANCHY_BLOCKS);
    let blocks: Vec<BlockId> = split.block_ids().step_by(4).collect();
    for &block in &blocks {
        split.split_block(block, 1, EdgeKind::Fallthrough);
    }
    benchmark_case!(
        suite,
        "cfg_diff_region_split",
        || RegionDiff::compute(&left, &split),
        |diff: &RegionDiff| {
            assert!(diff.left_only.is_empty() && diff.right_only.is_empty());
            let halves = diff
                .regions
                .iter()
                .filter(|region| region.left.len() == 1 && region.right.len() == 2);
            assert_eq!(halves.count(), blocks.len());
        }
    );
}
