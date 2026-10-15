//! The completed dominator forest of a CFG with unreachable code.

extern crate alloc;
use alloc::vec::Vec;

use super::complete_dominator_forest;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::dominator::{DominatorScratch, DominatorTree};
use crate::test_util::MockInst;

#[test]
fn unreachable_code_gets_numbered_trees_of_its_own() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let entry = cfg.entry();
    let [left, right, join, spin_a, spin_b, dead, feeder] =
        core::array::from_fn(|_| cfg.new_block());
    cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, join, EdgeKind::Fallthrough);
    cfg.add_edge(right, join, EdgeKind::Fallthrough);
    // An unreachable loop, and unreachable code that jumps into the live join.
    cfg.add_edge(spin_a, spin_b, EdgeKind::Fallthrough);
    cfg.add_edge(spin_b, spin_a, EdgeKind::Back);
    cfg.add_edge(dead, feeder, EdgeKind::Fallthrough);
    cfg.add_edge(feeder, join, EdgeKind::Fallthrough);

    let dominators = DominatorTree::compute(&cfg);
    assert_eq!(dominators.idom(join), Some(entry));
    assert!(!dominators.is_reachable(spin_a));

    let mut roots = Vec::new();
    let forest =
        complete_dominator_forest(&mut DominatorScratch::new(), &mut roots, &cfg, &dominators)
            .expect("unreachable blocks call for a forest");

    // The entry, one block of the loop, and the unreachable source start trees.
    assert_eq!(roots.len(), 3);
    let (spin_root, spin_rest) = if roots[1] == spin_a {
        (spin_a, spin_b)
    } else {
        (spin_b, spin_a)
    };
    assert_eq!(roots, [entry, spin_root, dead]);

    // The join is reached from two unrelated roots, so nothing but a root of
    // its own dominates it.
    assert_eq!(forest.idom(join), None);
    assert_eq!(forest.level(join), Some(0));
    assert_eq!(
        forest.preorder(),
        [entry, left, right, join, spin_root, spin_rest, dead, feeder]
    );
    assert_eq!(forest.subtree(entry), [entry, left, right]);
    assert_eq!(forest.children(entry).collect::<Vec<_>>(), [left, right]);
    assert_eq!(forest.level(spin_rest), Some(1));
    assert_eq!(forest.level(feeder), Some(1));

    assert!(forest.dominates(entry, left));
    assert!(!forest.dominates(entry, join));
    assert!(forest.dominates(spin_root, spin_rest));
    assert!(!forest.dominates(spin_rest, spin_root));
    assert!(!forest.dominates(dead, join));
    assert_eq!(forest.nearest_common_dominator(left, right), Some(entry));
    assert_eq!(forest.nearest_common_dominator(left, join), None);
    assert_eq!(
        forest.nearest_common_dominator_of([left, right, join]),
        None
    );
}

#[test]
fn a_cfg_the_entry_reaches_entirely_needs_no_forest() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let next = cfg.new_block();
    cfg.add_edge(cfg.entry(), next, EdgeKind::Fallthrough);
    let dominators = DominatorTree::compute(&cfg);
    let forest = complete_dominator_forest(
        &mut DominatorScratch::new(),
        &mut Vec::new(),
        &cfg,
        &dominators,
    );
    assert!(forest.is_none());
}
