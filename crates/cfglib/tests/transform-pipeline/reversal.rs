//! `reverse_cfg`: the transposed graph a post-dominance or backward-analysis
//! consumer builds, checked against brute-force path reasoning.

use std::collections::BTreeSet;

use cfglib::{BlockId, Cfg, DominatorTree, EdgeKind, reverse_cfg};

use crate::oracles::{assert_valid, live_blocks, reachable_blocks};
use crate::programs::random_program;
use crate::toy::{Builder, Inst};

fn edge_triples<I, E>(cfg: &Cfg<I, E>) -> Vec<(BlockId, BlockId, EdgeKind)> {
    let mut triples: Vec<_> = cfg
        .edges()
        .map(|edge| (edge.source(), edge.target(), edge.kind()))
        .collect();
    triples.sort_by_key(|&(source, target, kind)| (source, target, kind.name()));
    triples
}

#[test]
fn reversing_flips_every_edge_and_enters_at_the_only_exit() {
    let before = random_program::<()>(3, false);
    let [exit] = before.exit_blocks().collect::<Vec<_>>()[..] else {
        panic!("one exit")
    };
    let reversed = reverse_cfg(&before);

    assert_eq!(reversed.entry(), exit);
    assert_eq!(reversed.block_count(), before.block_count());
    let mut flipped: Vec<_> = edge_triples(&before)
        .into_iter()
        .map(|(source, target, kind)| (target, source, kind))
        .collect();
    flipped.sort_by_key(|&(source, target, kind)| (source, target, kind.name()));
    assert_eq!(edge_triples(&reversed), flipped);
    for block in before.block_ids() {
        assert_eq!(
            reversed.block(block).instructions(),
            before.block(block).instructions()
        );
    }
    assert_valid(&reversed, "reversed");
}

#[test]
fn reversing_twice_restores_the_graph_when_entry_and_exit_are_unique() {
    for seed in 0..40 {
        let before = random_program::<()>(seed, false);
        let twice = reverse_cfg(&reverse_cfg(&before));
        assert_eq!(twice.entry(), before.entry(), "seed {seed}");
        assert_eq!(edge_triples(&twice), edge_triples(&before), "seed {seed}");
        assert_eq!(live_blocks(&twice), live_blocks(&before), "seed {seed}");
    }
}

#[test]
fn several_exits_are_joined_under_one_synthetic_entry() {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::branch(0));
    let left = builder.block([Inst::ret()]);
    let right = builder.block([Inst::ret()]);
    builder.edge(entry, left, EdgeKind::ConditionalTrue);
    builder.edge(entry, right, EdgeKind::ConditionalFalse);
    let before = builder.finish();

    let reversed = reverse_cfg(&before);

    assert_eq!(reversed.block_count(), before.block_count() + 1);
    assert!(!before.contains_block(reversed.entry()));
    let entries: BTreeSet<BlockId> = reversed.successors(reversed.entry()).collect();
    assert_eq!(entries, BTreeSet::from([left, right]));
    assert_eq!(reachable_blocks(&reversed), live_blocks(&reversed));
}

#[test]
fn reversal_keeps_the_identity_slots_a_removed_block_left_behind() {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    let doomed = builder.block([Inst::print(0)]);
    let last = builder.block([Inst::ret()]);
    builder.edge(entry, last, EdgeKind::Fallthrough);
    let mut before = builder.finish();
    before.remove_block(doomed);

    let reversed = reverse_cfg(&before);

    assert!(!reversed.contains_block(doomed));
    assert_eq!(reversed.block_bound(), before.block_bound());
    assert_eq!(live_blocks(&reversed), live_blocks(&before));
    assert_eq!(reversed.entry(), last);
}

/// `dominator` post-dominates `block` when every path from `block` to `exit`
/// goes through it: removing it must cut `block` off from `exit`.
fn post_dominates(cfg: &Cfg<Inst>, dominator: BlockId, block: BlockId, exit: BlockId) -> bool {
    if dominator == block {
        return true;
    }
    let mut seen = BTreeSet::from([block]);
    let mut pending = vec![block];
    while let Some(current) = pending.pop() {
        if current == exit {
            return false;
        }
        for successor in cfg.successors(current).filter(|&next| next != dominator) {
            if seen.insert(successor) {
                pending.push(successor);
            }
        }
    }
    true
}

#[test]
fn dominators_of_the_reversed_graph_are_the_post_dominators_of_the_original() {
    for seed in 0..40 {
        let before = random_program::<()>(seed, true);
        let [exit] = before.exit_blocks().collect::<Vec<_>>()[..] else {
            panic!("one exit")
        };
        let reversed = reverse_cfg(&before);
        let dominators = DominatorTree::compute(&reversed);
        assert_eq!(reversed.entry(), exit);
        for block in before.block_ids() {
            for candidate in before.block_ids() {
                assert_eq!(
                    dominators.dominates(candidate, block),
                    post_dominates(&before, candidate, block, exit),
                    "seed {seed}: does {candidate} post-dominate {block}?"
                );
            }
        }
    }
}
