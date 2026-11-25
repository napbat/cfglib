use cfglib::{BlockId, Cfg, EdgeKind, EliminationHeuristic, TreeDecomposition, tree_decomposition};

use super::BenchmarkSuite;
use crate::fixtures::branchy_cfg;
use crate::harness::benchmark_case;

/// Blocks of the hot-path fixture, the largest CFG the benchmarks build.
const BRANCHY_BLOCKS: usize = 4_096;
/// Blocks of the generated CFG with gotos.
const GOTO_BLOCKS: usize = 100_000;
/// How far a goto of the generated CFG reaches, forward or back.
const GOTO_REACH: usize = 8;

/// A fall-through spine where every third block also jumps to a block at
/// most [`GOTO_REACH`] away, from a fixed xorshift sequence: irreducible
/// loops and shared tails throughout.
fn cfg_with_gotos(block_count: usize) -> Cfg<u32> {
    let mut cfg = Cfg::new();
    let mut blocks = Vec::with_capacity(block_count);
    blocks.push(cfg.entry());
    blocks.extend((1..block_count).map(|_| cfg.new_block()));
    for index in 1..block_count {
        cfg.add_edge(blocks[index - 1], blocks[index], EdgeKind::Fallthrough);
    }
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let span = u64::try_from(2 * GOTO_REACH + 1).expect("the reach fits in u64");
    for index in (0..block_count).step_by(3) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let offset = usize::try_from(state % span).expect("an offset fits in usize");
        let target = (index + offset)
            .saturating_sub(GOTO_REACH)
            .min(block_count - 1);
        cfg.add_edge(blocks[index], blocks[target], EdgeKind::Jump);
    }
    cfg
}

/// Whether every block leads its own bag and no bag is wider than `width`.
fn is_sound(cfg: &Cfg<u32>, decomposition: &TreeDecomposition<BlockId>, width: usize) -> bool {
    decomposition.len() == cfg.block_count()
        && decomposition.width() <= width
        && cfg.block_ids().all(|block| {
            decomposition
                .bag_of(block)
                .is_some_and(|bag| decomposition.bag(bag)[0] == block)
        })
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let branchy = branchy_cfg(BRANCHY_BLOCKS);
    benchmark_case!(
        suite,
        "tree_decomposition_branchy_4k_min_degree",
        covers[tree_decomposition],
        || tree_decomposition(&branchy, EliminationHeuristic::MinDegree),
        |decomposition: &TreeDecomposition<BlockId>| {
            assert!(is_sound(&branchy, decomposition, 32));
        }
    );
    benchmark_case!(
        suite,
        "tree_decomposition_branchy_4k_min_fill",
        || tree_decomposition(&branchy, EliminationHeuristic::MinFill),
        |decomposition: &TreeDecomposition<BlockId>| {
            assert!(is_sound(&branchy, decomposition, 32));
        }
    );

    let gotos = cfg_with_gotos(GOTO_BLOCKS);
    benchmark_case!(
        suite,
        "tree_decomposition_gotos_100k_min_degree",
        || tree_decomposition(&gotos, EliminationHeuristic::MinDegree),
        |decomposition: &TreeDecomposition<BlockId>| {
            assert!(is_sound(&gotos, decomposition, 4 * GOTO_REACH));
        }
    );
    benchmark_case!(
        suite,
        "tree_decomposition_gotos_100k_min_fill",
        || tree_decomposition(&gotos, EliminationHeuristic::MinFill),
        |decomposition: &TreeDecomposition<BlockId>| {
            assert!(is_sound(&gotos, decomposition, 4 * GOTO_REACH));
        }
    );
}
