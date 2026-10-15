//! Graphs that break a naive construction: deep chains, wide dispatchers, and
//! the worst-case families of Georgiadis, Tarjan, and Werneck, "Finding
//! dominators in practice" (2006).
//!
//! The tests assert the exact tree. A deep recursion would overflow the test
//! thread's stack, and a quadratic pass over the larger members would not
//! finish, so a hang or a crash is the failure. A wall-clock bound against a
//! breadth-first search is not asserted, because it flakes on a loaded host.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::support::{EdgeList, assert_forest_invariants, node};
use crate::graph::dominator::DominatorTree;

/// The dominator tree of `edges` from node 0, checked against `parent`.
fn assert_tree(edges: &EdgeList, parent: impl Fn(u32) -> Option<u32>) -> DominatorTree<u32> {
    let tree = DominatorTree::compute_from(&edges.graph(), 0);
    for member in 0..node(edges.node_count) {
        assert_eq!(tree.idom(member), parent(member), "node {member}");
    }
    assert_forest_invariants(&tree, edges.node_count);
    tree
}

/// `edges` agree with the iterative oracle too.
fn assert_matches_oracle(edges: &EdgeList) {
    let graph = edges.graph();
    assert!(
        DominatorTree::compute_from(&graph, 0) == DominatorTree::compute_iterative(&graph, 0),
        "{} nodes",
        edges.node_count
    );
}

/// A path through `count` nodes.
fn chain(count: usize) -> EdgeList {
    let edges = (0..count - 1).map(|i| (node(i), node(i + 1))).collect();
    EdgeList::new(count, edges)
}

/// Control-flow flattening: an entry, a dispatcher, `cases` case blocks that
/// each return to the dispatcher, and an exit. With `chained`, every case
/// also jumps straight to the next one, as a state machine does.
fn dispatcher(cases: usize, chained: bool) -> EdgeList {
    let exit = node(cases + 2);
    let mut edges = vec![(0, 1), (1, exit)];
    for case in 2..node(cases + 2) {
        edges.push((1, case));
        edges.push((case, 1));
        if chained && case + 1 < exit {
            edges.push((case, case + 1));
        }
    }
    EdgeList::new(cases + 3, edges)
}

/// `sncaworst(k)`: a path `x` from the root, and `k` leaves `y` that the root
/// and the end of the path both reach. Node numbers: `x_i = i`, `y_i = k + i`.
fn sncaworst(k: usize) -> EdgeList {
    let x = node;
    let y = |i: usize| node(k + i);
    let mut edges = vec![(0, x(1))];
    edges.extend((1..k).map(|i| (x(i), x(i + 1))));
    for i in 1..=k {
        edges.push((0, y(i)));
        edges.push((x(k), y(i)));
    }
    EdgeList::new(2 * k + 1, edges)
}

/// The tree of `sncaworst(k)`: the path is a chain and every leaf hangs below
/// the root.
fn sncaworst_parent(k: usize, member: u32) -> Option<u32> {
    let index = member as usize;
    match index {
        0 => None,
        1 => Some(0),
        i if i <= k => Some(node(i - 1)),
        _ => Some(0),
    }
}

/// `idfsquad(k)`: a path `x` beside `k` pairs of nodes `y` and `z` that reach
/// each other. Node numbers: `x_i = i`, `y_i = k + i`, `z_i = 2k + i`.
fn idfsquad(k: usize) -> EdgeList {
    let x = node;
    let y = |i: usize| node(k + i);
    let z = |i: usize| node(2 * k + i);
    let mut edges = vec![(0, x(1)), (0, z(1))];
    for i in 1..=k {
        if i < k {
            edges.push((x(i), x(i + 1)));
            edges.push((y(i), z(i + 1)));
        }
        edges.push((x(i), y(i)));
        edges.push((y(i), z(i)));
        edges.push((z(i), y(i)));
    }
    EdgeList::new(3 * k + 1, edges)
}

/// The tree of `idfsquad(k)`: the path is a chain and every other node hangs
/// below the root.
fn idfsquad_parent(k: usize, member: u32) -> Option<u32> {
    let index = member as usize;
    match index {
        0 => None,
        1 => Some(0),
        i if i <= k => Some(node(i - 1)),
        _ => Some(0),
    }
}

/// `itworst(k)`: four paths `w`, `x`, `y`, and `z`, a back arc down `z`, and an
/// arc from every `y` to every `w`. Node numbers: `w_i = i`, `x_i = k + i`,
/// `y_i = 2k + i`, `z_i = 3k + i`.
fn itworst(k: usize) -> EdgeList {
    let w_at = node;
    let x_at = |i: usize| node(k + i);
    let y_at = |i: usize| node(2 * k + i);
    let z_at = |i: usize| node(3 * k + i);
    let mut edges = vec![
        (0, w_at(1)),
        (0, x_at(1)),
        (0, z_at(k)),
        (x_at(k), y_at(1)),
        (y_at(k), z_at(1)),
    ];
    for i in 1..k {
        edges.push((w_at(i), w_at(i + 1)));
        edges.push((x_at(i), x_at(i + 1)));
        edges.push((y_at(i), y_at(i + 1)));
        edges.push((z_at(i), z_at(i + 1)));
    }
    for i in 2..=k {
        edges.push((z_at(i), z_at(i - 1)));
    }
    for i in 1..=k {
        for j in 1..=k {
            edges.push((y_at(i), w_at(j)));
        }
    }
    EdgeList::new(4 * k + 1, edges)
}

/// The tree of `itworst(k)`: `x` and `y` are chains joined at the end of `x`,
/// and every `w` and `z` hangs below the root.
fn itworst_parent(k: usize, member: u32) -> Option<u32> {
    let index = member as usize;
    if index == 0 {
        return None;
    }
    let (path, i) = ((index - 1) / k, (index - 1) % k + 1);
    match (path, i) {
        (1, 1) => Some(0),
        (2, 1) => Some(node(2 * k)),
        (1 | 2, _) => Some(node(index - 1)),
        _ => Some(0),
    }
}

/// `sltworst(k)`: a path of `k` nodes plus one back arc per node chosen so
/// that, when the simple Lengauer-Tarjan algorithm evaluates it, the arc's
/// tail is the deepest node of its link-eval tree. Path compression without
/// balancing then does Θ(k log k) work, the worst it can.
///
/// The generator replays the algorithm's link-eval forest to pick each tail,
/// and returns the number of nodes path compression repointed along the way.
/// Node `i` is the path's `i`-th node, so the depth-first numbering is the
/// index and the algorithm visits nodes from the last to the first.
fn sltworst(k: usize) -> (EdgeList, usize) {
    let mut edges: Vec<_> = (0..k - 1).map(|i| (node(i), node(i + 1))).collect();
    let mut ancestor: Vec<Option<usize>> = vec![None; k];
    let mut depth = vec![0_usize; k];
    let mut repointed = 0;
    for target in (1..k).rev() {
        if target + 1 < k {
            depth[target] = 0;
            for linked in target + 1..k {
                let above = ancestor[linked].expect("a linked node has an ancestor");
                depth[linked] = depth[above] + 1;
            }
            let deepest = (target + 1..k)
                .max_by_key(|&linked| depth[linked])
                .expect("a node is linked below the target");
            edges.push((node(deepest), node(target)));

            // Evaluating `deepest` compresses its path to the root of the tree.
            let mut path = Vec::new();
            let mut current = deepest;
            while let Some(above) = ancestor[current] {
                if ancestor[above].is_none() {
                    break;
                }
                path.push(current);
                current = above;
            }
            repointed += path.len();
            for &compressed in path.iter().rev() {
                let above = ancestor[compressed].expect("a compressed node has an ancestor");
                ancestor[compressed] = ancestor[above];
            }
        }
        ancestor[target] = Some(target - 1);
    }
    (EdgeList::new(k, edges), repointed)
}

fn chain_parent(member: u32) -> Option<u32> {
    member.checked_sub(1)
}

#[test]
fn a_hundred_thousand_block_chain_needs_no_deep_call_stack() {
    const BLOCKS: usize = 100_000;
    let tree = assert_tree(&chain(BLOCKS), chain_parent);
    let last = node(BLOCKS - 1);
    assert_eq!(tree.level(last), Some(last));
    assert!(tree.dominates(0, last));
    assert!(!tree.dominates(last, 0));
    assert_eq!(tree.subtree(node(BLOCKS / 2)).len(), BLOCKS - BLOCKS / 2);
    assert_eq!(
        tree.nearest_common_dominator(last, node(BLOCKS / 2)),
        Some(node(BLOCKS / 2))
    );
    assert_eq!(tree.children(node(BLOCKS / 2)).count(), 1);
}

#[test]
fn a_chain_whose_parents_have_larger_ids_is_numbered_by_structure() {
    const BLOCKS: usize = 100_000;
    let edges = EdgeList::new(
        BLOCKS,
        (1..BLOCKS).map(|i| (node(i), node(i - 1))).collect(),
    );
    let root = node(BLOCKS - 1);
    let tree = DominatorTree::compute_from(&edges.graph(), root);
    assert_forest_invariants(&tree, BLOCKS);
    assert_eq!(tree.idom(0), Some(1));
    assert_eq!(tree.level(0), Some(node(BLOCKS - 1)));
    assert_eq!(tree.preorder().first(), Some(&root));
    assert_eq!(tree.preorder().last(), Some(&0));
}

#[test]
fn a_flattened_dispatcher_puts_every_case_below_the_dispatcher() {
    for chained in [false, true] {
        let cases = 50_000;
        let edges = dispatcher(cases, chained);
        let exit = node(cases + 2);
        let tree = assert_tree(&edges, |member| match member {
            0 => None,
            1 => Some(0),
            _ => Some(1),
        });
        assert_eq!(tree.children(1).count(), cases + 1);
        assert_eq!(tree.level(exit), Some(2));
        assert_eq!(tree.level(2), Some(2));
        assert_eq!(
            tree.nearest_common_dominator(2, node(cases + 1)),
            Some(1),
            "chained: {chained}"
        );
    }
}

#[test]
fn a_small_dispatcher_agrees_with_the_oracle() {
    assert_matches_oracle(&dispatcher(40, false));
    assert_matches_oracle(&dispatcher(40, true));
}

#[test]
fn sncaworst_is_solved_without_climbing_the_path_per_leaf() {
    let k = 50_000;
    let tree = assert_tree(&sncaworst(k), |member| sncaworst_parent(k, member));
    assert_eq!(tree.level(node(k)), Some(node(k)));
    assert_eq!(tree.level(node(2 * k)), Some(1));
    assert_matches_oracle(&sncaworst(300));
}

#[test]
fn idfsquad_is_solved_in_one_pass() {
    let k = 30_000;
    let tree = assert_tree(&idfsquad(k), |member| idfsquad_parent(k, member));
    assert_eq!(tree.level(node(k)), Some(node(k)));
    assert_matches_oracle(&idfsquad(200));
}

#[test]
fn itworst_has_the_tree_its_construction_implies() {
    let k = 150;
    let tree = assert_tree(&itworst(k), |member| itworst_parent(k, member));
    assert_eq!(tree.level(node(3 * k)), Some(node(2 * k)));
    assert_matches_oracle(&itworst(24));
}

#[test]
fn sltworst_is_a_chain_whatever_the_compression_pattern() {
    let (edges, repointed) = sltworst(3_000);
    let tree = assert_tree(&edges, chain_parent);
    assert_eq!(tree.level(2_999), Some(2_999));
    assert_matches_oracle(&sltworst(250).0);
    // The family costs path compression Θ(k log k), which at this size is
    // several repointings per node. A linear cost would fail this bound.
    assert!(repointed >= 8 * 3_000, "only {repointed} nodes repointed");
}

#[test]
fn the_families_keep_their_documented_size() {
    assert_eq!(itworst(7).node_count, 4 * 7 + 1);
    assert_eq!(itworst(7).edges.len(), 7 * 7 + 5 * 7);
    assert_eq!(idfsquad(7).node_count, 3 * 7 + 1);
    assert_eq!(idfsquad(7).edges.len(), 5 * 7);
    assert_eq!(sncaworst(7).node_count, 2 * 7 + 1);
    assert_eq!(sncaworst(7).edges.len(), 3 * 7);
}
