//! Numbering of forests with several roots and of nodes in no tree.

extern crate alloc;
use alloc::vec::Vec;

use super::support::assert_forest_invariants;
use crate::graph::dominator::DominatorTree;

/// A forest over eight nodes. Node 0 and node 7 are in no tree. Node 2 roots a
/// tree and so does node 5, whose children 1 and 4 have smaller indices than
/// its own, and 4 has the child 3.
fn two_trees() -> DominatorTree<u32> {
    let parents = [None, Some(5), None, Some(4), Some(5), None, Some(2), None];
    let reachable = [false, true, true, true, true, true, true, false];
    DominatorTree::from_forest_parts(&parents, &reachable)
}

#[test]
fn roots_and_children_are_numbered_by_ascending_index() {
    let forest = two_trees();
    assert_forest_invariants(&forest, 8);
    assert_eq!(forest.preorder(), [2, 6, 5, 1, 4, 3]);
    assert_eq!(forest.subtree(5), [5, 1, 4, 3]);
    assert_eq!(forest.subtree(2), [2, 6]);
    assert_eq!(forest.children(5).collect::<Vec<_>>(), [1, 4]);
    for (node, level) in [(2, 0), (6, 1), (5, 0), (1, 1), (4, 1), (3, 2)] {
        assert_eq!(forest.level(node), Some(level), "node {node}");
    }
    for node in [0, 7] {
        assert!(!forest.is_reachable(node));
        assert_eq!(forest.subtree(node).len(), 0);
    }
}

#[test]
fn dominance_never_crosses_from_one_tree_to_another() {
    let forest = two_trees();
    assert!(forest.dominates(2, 6));
    assert!(forest.dominates(5, 3));
    assert!(!forest.dominates(2, 5));
    assert!(!forest.dominates(5, 2));
    assert!(!forest.dominates(6, 3));
    assert!(!forest.strictly_dominates(5, 6));
    assert!(forest.dominates(5, 5));
}

#[test]
fn nearest_common_dominators_need_a_shared_tree() {
    let forest = two_trees();
    assert_eq!(forest.nearest_common_dominator(1, 3), Some(5));
    assert_eq!(forest.nearest_common_dominator(3, 3), Some(3));
    assert_eq!(forest.nearest_common_dominator(2, 5), None);
    assert_eq!(forest.nearest_common_dominator(6, 3), None);
    assert_eq!(forest.nearest_common_dominator(6, 0), None);
    assert_eq!(forest.nearest_common_dominator_of([3, 1, 7]), Some(5));
    assert_eq!(forest.nearest_common_dominator_of([6, 3, 7]), None);
    assert_eq!(forest.nearest_common_dominator_of([2, 6]), Some(2));
}

#[test]
fn an_empty_forest_has_nothing_to_number() {
    let forest = DominatorTree::<u32>::from_forest_parts(&[], &[]);
    assert_eq!(forest.preorder().len(), 0);
}
