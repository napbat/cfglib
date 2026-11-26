use cfglib::{Cfg, CycleEquivalence, EdgeKind, ProgramStructureTree, SeseEdge};

use super::BenchmarkSuite;
use crate::fixtures::linear_cfg;
use crate::harness::benchmark_case;

/// Blocks of the chain.
const CHAIN_BLOCKS: usize = 100_000;
/// Diamonds nested in one another's `then` arms.
const NEST_DEPTH: usize = 100_000;
/// Blocks of the random CFG.
const RANDOM_BLOCKS: usize = 1_000_000;

/// If/else diamonds nested `depth` deep in their `then` arms.
fn diamond_nest(depth: usize) -> Cfg<u32> {
    let mut cfg = Cfg::new();
    let mut branch = cfg.entry();
    let mut outer_join = None;
    for _ in 0..depth {
        let then_block = cfg.new_block();
        let else_block = cfg.new_block();
        let join = cfg.new_block();
        cfg.add_edge(branch, then_block, EdgeKind::ConditionalTrue);
        cfg.add_edge(branch, else_block, EdgeKind::ConditionalFalse);
        cfg.add_edge(else_block, join, EdgeKind::Fallthrough);
        if let Some(outer) = outer_join {
            cfg.add_edge(join, outer, EdgeKind::Fallthrough);
        }
        branch = then_block;
        outer_join = Some(join);
    }
    if let Some(join) = outer_join {
        cfg.add_edge(branch, join, EdgeKind::Fallthrough);
    }
    cfg
}

/// A fall-through spine to one exit, where a fixed xorshift sequence gives
/// half the blocks a short forward branch and a quarter a short back edge.
fn random_cfg(block_count: usize) -> Cfg<u32> {
    let mut cfg = Cfg::new();
    let mut blocks = Vec::with_capacity(block_count);
    blocks.push(cfg.entry());
    blocks.extend((1..block_count).map(|_| cfg.new_block()));
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for index in 1..block_count {
        let from = index - 1;
        cfg.add_edge(blocks[from], blocks[index], EdgeKind::Fallthrough);
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let jump = usize::try_from(state >> 60).expect("four bits fit in usize");
        match state % 4 {
            0 => {}
            1 | 2 => {
                let target = (from + 2 + jump % 8).min(block_count - 1);
                cfg.add_edge(blocks[from], blocks[target], EdgeKind::ConditionalTrue);
            }
            _ => {
                let target = from.saturating_sub(jump);
                cfg.add_edge(blocks[from], blocks[target], EdgeKind::Back);
            }
        }
    }
    cfg
}

/// Whether every region follows its parent in preorder and holds a block.
fn is_well_formed(tree: &ProgramStructureTree) -> bool {
    tree.regions().all(|region| {
        tree.parent(region).is_none_or(|parent| parent < region) && !tree.blocks(region).is_empty()
    })
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let chain = linear_cfg(CHAIN_BLOCKS);
    benchmark_case!(
        suite,
        "program_structure_chain_100k",
        || ProgramStructureTree::compute(&chain),
        |tree: &ProgramStructureTree| {
            assert_eq!(tree.region_count(), CHAIN_BLOCKS);
            assert!(tree.regions().all(|region| tree.parent(region).is_none()));
        }
    );

    let nest = diamond_nest(NEST_DEPTH);
    benchmark_case!(
        suite,
        "program_structure_diamond_nest_100k",
        || ProgramStructureTree::compute(&nest),
        |tree: &ProgramStructureTree| {
            // The procedure, a diamond per inner level, each `else` arm, and
            // the innermost `then` block.
            assert_eq!(tree.region_count(), 2 * NEST_DEPTH + 1);
            assert!(is_well_formed(tree));
        }
    );

    let random = random_cfg(RANDOM_BLOCKS);
    benchmark_case!(
        suite,
        "program_structure_random_1m",
        || ProgramStructureTree::compute(&random),
        |tree: &ProgramStructureTree| {
            assert!(!tree.is_empty());
            assert!(is_well_formed(tree));
        }
    );
    benchmark_case!(
        suite,
        "cycle_equivalence_random_1m",
        || CycleEquivalence::compute(&random),
        |classes: &CycleEquivalence| {
            assert!(classes.class(SeseEdge::Entry).is_some());
            assert!(classes.class_count() > 1);
        }
    );
}
