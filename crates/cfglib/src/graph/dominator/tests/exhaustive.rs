//! Every small digraph against the definition of dominance.

extern crate alloc;
use alloc::vec::Vec;

use super::support::{EdgeList, Reference, assert_forest_invariants, for_each_multigraph, node};
use crate::graph::dominator::{DominatorScratch, DominatorTree};
use crate::graph::view::GraphView;

/// Check every query of a tree against the brute-force reference.
fn check_against_definition(edges: &EdgeList, scratch: &mut DominatorScratch) {
    let graph = edges.graph();
    let tree = DominatorTree::compute_from_in(scratch, &graph, 0);
    let reference = Reference::new(&graph, 0);
    let count = edges.node_count;
    assert_forest_invariants(&tree, count);
    assert_eq!(
        tree,
        DominatorTree::compute_iterative(&graph, 0),
        "{edges:?}"
    );

    for member in 0..count {
        let id = node(member);
        assert_eq!(tree.is_reachable(id), reference.is_reachable(member));
        assert_eq!(
            tree.idom(id),
            reference.idom(member).map(node),
            "idom of {member} in {edges:?}"
        );
        assert_eq!(
            tree.level(id),
            reference.level(member).map(node),
            "level of {member} in {edges:?}"
        );

        let mut subtree = tree.subtree(id).to_vec();
        subtree.sort_unstable();
        let expected: Vec<u32> = if reference.is_reachable(member) {
            reference
                .dominated_by(member)
                .into_iter()
                .map(node)
                .collect()
        } else {
            Vec::new()
        };
        assert_eq!(subtree, expected, "subtree of {member} in {edges:?}");

        let children: Vec<u32> = (0..count)
            .filter(|&other| reference.idom(other) == Some(member))
            .map(node)
            .collect();
        assert_eq!(
            tree.children(id).collect::<Vec<_>>(),
            children,
            "children of {member} in {edges:?}"
        );
    }

    let preorder: Vec<u32> = reference.preorder().into_iter().map(node).collect();
    assert_eq!(tree.preorder(), preorder, "preorder of {edges:?}");

    for first in 0..count {
        for second in 0..count {
            let (a, b) = (node(first), node(second));
            let dominates = reference.dominates(first, second);
            assert_eq!(
                tree.dominates(a, b),
                dominates,
                "{first} dominates {second} in {edges:?}"
            );
            assert_eq!(
                tree.strictly_dominates(a, b),
                first != second && dominates,
                "{first} strictly dominates {second} in {edges:?}"
            );
            assert_eq!(
                tree.nearest_common_dominator(a, b),
                reference.nearest_common_dominator(first, second).map(node),
                "nearest common dominator of {first} and {second} in {edges:?}"
            );
        }
    }

    for subset in 0..1_u32 << count {
        let members: Vec<usize> = (0..count).filter(|&bit| (subset >> bit) & 1 == 1).collect();
        let expected = members
            .iter()
            .copied()
            .filter(|&member| reference.is_reachable(member))
            .reduce(|common, member| {
                reference
                    .nearest_common_dominator(common, member)
                    .expect("reachable nodes share the root")
            });
        assert_eq!(
            tree.nearest_common_dominator_of(members.iter().copied().map(node)),
            expected.map(node),
            "nearest common dominator of {members:?} in {edges:?}"
        );
    }
}

#[test]
fn every_digraph_of_up_to_four_nodes_matches_the_definition() {
    let mut scratch = DominatorScratch::new();
    for node_count in 1..=4 {
        for_each_multigraph(node_count, 1, |edges| {
            check_against_definition(edges, &mut scratch);
        });
    }
}

#[test]
fn every_multigraph_of_up_to_three_nodes_matches_the_definition() {
    let mut scratch = DominatorScratch::new();
    for node_count in 1..=3 {
        for_each_multigraph(node_count, 2, |edges| {
            check_against_definition(edges, &mut scratch);
        });
    }
}

#[test]
fn doubling_every_edge_of_a_four_node_digraph_keeps_the_tree() {
    for_each_multigraph(4, 1, |edges| {
        let once = DominatorTree::compute_from(&edges.graph(), 0);
        let twice = DominatorTree::compute_from(&edges.repeated(2).graph(), 0);
        assert_eq!(once, twice, "{edges:?}");
    });
}

#[test]
fn a_root_other_than_node_zero_is_honored() {
    for_each_multigraph(3, 1, |edges| {
        let graph = edges.graph();
        for root in 0..edges.node_count {
            let tree = DominatorTree::compute_from(&graph, node(root));
            assert_eq!(tree.idom(node(root)), None, "{edges:?}");
            assert_eq!(tree.level(node(root)), Some(0), "{edges:?}");
            assert_eq!(tree.preorder().first(), Some(&node(root)), "{edges:?}");
            assert_eq!(
                tree,
                DominatorTree::compute_iterative(&graph, node(root)),
                "root {root} of {edges:?}"
            );
            assert_forest_invariants(&tree, graph.node_bound());
        }
    });
}
