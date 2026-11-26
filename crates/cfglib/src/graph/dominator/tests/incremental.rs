//! Incremental updates against a full recomputation: after every batch the
//! tree is `==` to the one built from scratch, `changed` is exactly the nodes
//! whose immediate dominator differs, and the tree passes its certificate.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::support::{
    EdgeList, Family, Recipe, Rng, assert_certified, for_each_multigraph, node, random_graph,
};
use crate::graph::dominator::incremental::{GATES, Gates};
use crate::graph::dominator::{DominatorScratch, DominatorTree, DominatorUpdate};
use crate::graph::view::GraphView;

/// No gate: every edit is replayed, so the update rules themselves are what
/// is compared with a recomputation.
const UNGATED: Gates = Gates {
    edits: usize::MAX,
    pass_nodes: 0,
    region_divisor: 0,
};

/// Every gate shut: a batch of two edits, or any edit that changes the tree
/// while edits remain, or any subtree to recompute, sends the batch to a full
/// construction.
const SHUT: Gates = Gates {
    edits: 1,
    pass_nodes: usize::MAX,
    region_divisor: usize::MAX,
};

/// The nodes, ascending, whose immediate dominator differs between `before`
/// and `after`, read over `bound` nodes; a node beyond `before`'s bound has
/// none there.
fn differing(before: &DominatorTree<u32>, after: &DominatorTree<u32>, bound: usize) -> Vec<u32> {
    (0..node(bound))
        .filter(|&member| {
            let old = if (member as usize) < before.idom.len() {
                before.idom(member)
            } else {
                None
            };
            old != after.idom(member)
        })
        .collect()
}

/// Apply `updates` to `edges`, in order: an insertion appends the edge, a
/// deletion removes its last copy.
fn edited(edges: &EdgeList, node_count: usize, updates: &[DominatorUpdate<u32>]) -> EdgeList {
    let mut list = edges.edges.clone();
    for update in updates {
        match *update {
            DominatorUpdate::Insert { from, to } => list.push((from, to)),
            DominatorUpdate::Delete { from, to } => {
                let index = list
                    .iter()
                    .rposition(|&edge| edge == (from, to))
                    .expect("a deleted edge is in the graph");
                list.remove(index);
            }
        }
    }
    EdgeList::new(node_count, list)
}

/// Check one batch: the tree of `edges` updated by `updates`, behind each
/// of the gate settings, against the tree of the edited graph.
fn check_batch(
    edges: &EdgeList,
    node_count: usize,
    updates: &[DominatorUpdate<u32>],
    scratch: &mut DominatorScratch,
) {
    let before = DominatorTree::compute(&edges.graph());
    let after_edges = edited(edges, node_count, updates);
    let graph = after_edges.graph();
    let expected = DominatorTree::compute(&graph);
    let moved = differing(&before, &expected, node_count);

    for gates in [UNGATED, GATES, SHUT] {
        let mut tree = before.clone();
        let mut changed = vec![99];
        tree.apply_updates_with(scratch, &graph, updates, &mut changed, gates);
        assert!(
            tree == expected,
            "{edges:?} with {updates:?} behind {gates:?}: {:?} instead of {:?}",
            tree.preorder(),
            expected.preorder()
        );
        assert_eq!(
            changed, moved,
            "{edges:?} with {updates:?} behind {gates:?}"
        );
        assert_certified(&tree, &graph, 0, scratch);
    }
}

/// Every single insertion and deletion on every graph `for_each_multigraph`
/// visits.
fn every_single_edit(node_count: usize, copies: usize, scratch: &mut DominatorScratch) {
    for_each_multigraph(node_count, copies, |edges| {
        for from in 0..node(node_count) {
            for to in 0..node(node_count) {
                check_batch(
                    edges,
                    node_count,
                    &[DominatorUpdate::Insert { from, to }],
                    scratch,
                );
                if edges.edges.contains(&(from, to)) {
                    check_batch(
                        edges,
                        node_count,
                        &[DominatorUpdate::Delete { from, to }],
                        scratch,
                    );
                }
            }
        }
    });
}

#[test]
fn every_single_edit_of_every_digraph_of_up_to_four_nodes_matches_a_recomputation() {
    let mut scratch = DominatorScratch::default();
    for node_count in 1..=4 {
        every_single_edit(node_count, 1, &mut scratch);
    }
}

#[test]
fn every_single_edit_of_every_multigraph_of_up_to_three_nodes_matches_a_recomputation() {
    let mut scratch = DominatorScratch::default();
    for node_count in 1..=3 {
        every_single_edit(node_count, 2, &mut scratch);
    }
}

/// A random edit of `edges` over `node_count` nodes: an insertion between
/// random nodes, or the deletion of a random edge.
fn random_edit(rng: &mut Rng, edges: &EdgeList, node_count: usize) -> DominatorUpdate<u32> {
    if edges.edges.is_empty() || rng.chance(50) {
        DominatorUpdate::Insert {
            from: node(rng.below(node_count)),
            to: node(rng.below(node_count)),
        }
    } else {
        let (from, to) = edges.edges[rng.below(edges.edges.len())];
        DominatorUpdate::Delete { from, to }
    }
}

/// A batch of `size` random edits, each valid on the graph the ones before
/// it leave.
fn random_batch(
    rng: &mut Rng,
    edges: &EdgeList,
    node_count: usize,
    size: usize,
) -> Vec<DominatorUpdate<u32>> {
    let mut current = EdgeList::new(node_count, edges.edges.clone());
    let mut batch = Vec::new();
    for _ in 0..size {
        let update = random_edit(rng, &current, node_count);
        current = edited(&current, node_count, &[update]);
        batch.push(update);
    }
    batch
}

#[test]
fn random_batches_on_random_graphs_match_a_recomputation() {
    let mut scratch = DominatorScratch::default();
    for (span, seeds) in [(6, 300), (12, 200), (40, 80), (200, 30), (1_000, 8)] {
        for seed in 0..seeds {
            let mut rng = Rng::new(seed * 104_729 + u64::try_from(span).expect("fits"));
            let recipe = Recipe {
                family: Family::ALL[rng.below(3)],
                node_count: span,
                parallel: rng.chance(50),
                detached: if rng.chance(25) { span / 4 } else { 0 },
            };
            let mut edges = random_graph(&mut rng, recipe);
            let node_count = edges.node_count;
            for _ in 0..4 {
                // Mostly small batches, which stay incremental, now and then
                // one past either limit, which is finished from scratch.
                let size = match rng.below(10) {
                    0 => 20,
                    1 => 70,
                    _ => 1 + rng.below(16),
                };
                let batch = random_batch(&mut rng, &edges, node_count, size);
                check_batch(&edges, node_count, &batch, &mut scratch);
                edges = edited(&edges, node_count, &batch);
            }
        }
    }
}

#[test]
fn random_batches_on_the_adversarial_families_match_a_recomputation() {
    let mut scratch = DominatorScratch::default();
    let mut rng = Rng::new(59);
    // The path with a reverse subpath of Georgiadis et al., figure 1, where
    // one insertion or deletion moves every node but two, and a dispatcher.
    let n = 300;
    let mut path: Vec<(u32, u32)> = (0..n - 1).map(|i| (node(i), node(i + 1))).collect();
    path.extend((2..n - 1).map(|i| (node(i + 1), node(i))));
    let path = EdgeList::new(n, path);
    let shortcut = DominatorUpdate::Insert {
        from: 0,
        to: node(n - 1),
    };
    check_batch(&path, n, &[shortcut], &mut scratch);
    let with_shortcut = edited(&path, n, &[shortcut]);
    check_batch(
        &with_shortcut,
        n,
        &[DominatorUpdate::Delete {
            from: 0,
            to: node(n - 1),
        }],
        &mut scratch,
    );

    let cases = 200;
    let mut dispatcher = vec![(0, 1), (1, node(cases + 2))];
    for case in 2..node(cases + 2) {
        dispatcher.extend([(1, case), (case, 1)]);
    }
    let dispatcher = EdgeList::new(cases + 3, dispatcher);
    for edges in [&path, &with_shortcut, &dispatcher] {
        for _ in 0..30 {
            let size = 1 + rng.below(8);
            let batch = random_batch(&mut rng, edges, edges.node_count, size);
            check_batch(edges, edges.node_count, &batch, &mut scratch);
        }
    }
}

#[test]
fn edits_may_name_nodes_added_after_the_tree_was_computed() {
    let mut scratch = DominatorScratch::default();
    let mut rng = Rng::new(590);
    for seed in 0..200 {
        let span = 6 + rng.below(30);
        let recipe = Recipe {
            family: Family::ALL[usize::try_from(seed % 3).expect("fits")],
            node_count: span,
            parallel: false,
            detached: 0,
        };
        let edges = random_graph(&mut rng, recipe);
        let grown = edges.node_count + 1 + rng.below(4);
        let mut batch = Vec::new();
        for added in edges.node_count..grown {
            batch.push(DominatorUpdate::Insert {
                from: node(rng.below(added)),
                to: node(added),
            });
            if rng.chance(50) {
                batch.push(DominatorUpdate::Insert {
                    from: node(added),
                    to: node(rng.below(added)),
                });
            }
        }
        check_batch(&edges, grown, &batch, &mut scratch);
    }
}

#[test]
fn a_deletion_that_strands_a_subtree_moves_what_it_led_to() {
    // root -> left -> tail -> cut -> join, and root -> right -> join.
    // Deleting tail -> cut strands `cut`, and `join`, which `cut` and
    // `right` both entered, falls to `right`.
    let (root, left, right, tail, cut, join) = (0, 1, 2, 3, 4, 5);
    let edges = EdgeList::new(
        6,
        vec![
            (root, left),
            (root, right),
            (left, tail),
            (tail, cut),
            (cut, join),
            (right, join),
        ],
    );
    let mut tree = DominatorTree::compute(&edges.graph());
    assert_eq!(tree.idom(join), Some(root));
    let deletion = DominatorUpdate::Delete {
        from: tail,
        to: cut,
    };
    let graph = edited(&edges, 6, &[deletion]).graph();
    let mut changed = Vec::new();
    tree.apply_updates_in(
        &mut DominatorScratch::default(),
        &graph,
        &[deletion],
        &mut changed,
    );
    assert_eq!(tree.idom(join), Some(right));
    assert!(!tree.is_reachable(cut));
    assert_eq!(changed, [cut, join]);
}

#[test]
fn edits_that_change_nothing_report_nothing() {
    // A diamond with a back edge from the join to the entry.
    let edges = EdgeList::new(4, vec![(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)]);
    let tree = DominatorTree::compute(&edges.graph());
    let mut scratch = DominatorScratch::default();
    for update in [
        DominatorUpdate::Insert { from: 1, to: 3 },
        DominatorUpdate::Insert { from: 3, to: 1 },
        DominatorUpdate::Delete { from: 3, to: 0 },
    ] {
        let graph = edited(&edges, 4, &[update]).graph();
        let mut updated = tree.clone();
        let mut changed = vec![7];
        updated.apply_updates_in(&mut scratch, &graph, &[update], &mut changed);
        assert_eq!(updated, tree, "{update:?}");
        assert_eq!(changed, Vec::<u32>::new(), "{update:?}");
    }
}

#[test]
fn an_insertion_and_its_deletion_in_one_batch_cancel() {
    let edges = EdgeList::new(3, vec![(0, 1), (1, 2)]);
    let tree = DominatorTree::compute(&edges.graph());
    let batch = [
        DominatorUpdate::Insert { from: 0, to: 2 },
        DominatorUpdate::Delete { from: 0, to: 2 },
    ];
    let graph = edges.graph();
    let mut updated = tree.clone();
    let mut changed = Vec::new();
    updated.apply_updates_in(
        &mut DominatorScratch::default(),
        &graph,
        &batch,
        &mut changed,
    );
    assert_eq!(updated, tree);
    assert_eq!(changed, Vec::<u32>::new());
    assert_eq!(graph.node_bound(), 3);
}
