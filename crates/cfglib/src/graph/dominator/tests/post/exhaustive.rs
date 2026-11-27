//! Every small digraph against the definition of post-dominance.

use super::certificate::assert_post_certified;
use super::reference::{
    PostReference, all_subsets, assert_contract, assert_frontiers, assert_matches, blocks,
    post_node,
};
use crate::graph::dominator::tests::support::{EdgeList, for_each_multigraph};
use crate::graph::dominator::{DominatorScratch, ExitPolicy, PostDominatorTree};

/// Check every query of a tree against the brute-force reference, with one
/// scratch serving the tree, its certificate and its frontiers.
fn check_against_definition(edges: &EdgeList, scratch: &mut DominatorScratch) {
    let graph = edges.graph();
    let post = PostDominatorTree::compute_in(scratch, &graph, ExitPolicy::CutLoops);
    let reference = PostReference::new(&graph);
    let sets = all_subsets(edges.node_count);
    assert_matches(&post, &reference, edges);
    assert_post_certified(&post, &graph, scratch);
    assert_contract(&post, &reference, edges);
    assert_frontiers(&post, &graph, &reference, scratch, &sets, edges);

    for members in &sets {
        assert_eq!(
            post.nearest_common_post_dominator_among(blocks(members)),
            reference.nearest_common(members).map(post_node),
            "nearest common post-dominator of {members:?} in {edges:?}"
        );
    }
}

#[test]
fn every_digraph_of_up_to_four_nodes_matches_the_definition() {
    let mut scratch = DominatorScratch::default();
    for node_count in 1..=4 {
        for_each_multigraph(node_count, 1, |edges| {
            check_against_definition(edges, &mut scratch);
        });
    }
}

#[test]
fn every_multigraph_of_up_to_three_nodes_matches_the_definition() {
    let mut scratch = DominatorScratch::default();
    for node_count in 1..=3 {
        for_each_multigraph(node_count, 2, |edges| {
            check_against_definition(edges, &mut scratch);
        });
    }
}

#[test]
fn doubling_every_edge_of_a_four_node_digraph_keeps_the_tree() {
    for_each_multigraph(4, 1, |edges| {
        let once = PostDominatorTree::compute(&edges.graph());
        let twice = PostDominatorTree::compute(&edges.repeated(2).graph());
        assert_eq!(once, twice, "{edges:?}");
    });
}
