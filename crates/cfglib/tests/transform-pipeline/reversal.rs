//! The reversed views a post-dominance or backward-analysis consumer composes
//! instead of copying the graph, checked against brute-force path reasoning.

use std::collections::BTreeSet;

use cfglib::{BlockId, Cfg, DominatorTree, EdgeKind, GraphView, Reversed, Rooted, VirtualRoot};

use crate::programs::random_program;
use crate::toy::{Builder, Inst};

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
fn dominators_of_the_reversed_view_are_the_post_dominators_of_the_original() {
    for seed in 0..40 {
        let before = random_program::<()>(seed, true);
        let [exit] = before.exit_blocks().collect::<Vec<_>>()[..] else {
            panic!("one exit")
        };
        let reversed = Reversed::new(&before);
        let dominators = DominatorTree::compute(&Rooted::new(&reversed, exit));
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

#[test]
fn several_exits_meet_at_a_virtual_root_of_the_reversed_view() {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::branch(0));
    let left = builder.block([Inst::ret()]);
    let right = builder.block([Inst::ret()]);
    builder.edge(entry, left, EdgeKind::ConditionalTrue);
    builder.edge(entry, right, EdgeKind::ConditionalFalse);
    let before = builder.finish();

    let exits: Vec<BlockId> = before.exit_blocks().collect();
    let reversed = Reversed::new(&before);
    let view = VirtualRoot::new(&reversed, &exits);
    let entries: BTreeSet<usize> = view.successors(view.virtual_root()).collect();
    assert_eq!(entries, BTreeSet::from([left.index(), right.index()]));

    let dominators = DominatorTree::compute(&view);
    assert_eq!(
        dominators.idom(entry.index()),
        Some(view.virtual_root()),
        "neither exit post-dominates the branch"
    );
    assert_eq!(dominators.idom(left.index()), Some(view.virtual_root()));
}
