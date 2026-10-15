//! Every small digraph against the definition of post-dominance.

use super::reference::{
    PostReference, all_subsets, assert_contract, assert_frontiers, assert_matches, blocks,
    post_node,
};
use crate::graph::dominator::tests::support::{EdgeList, for_each_multigraph};
use crate::graph::dominator::{FrontierScratch, PostDominatorScratch, PostDominatorTree};

/// The buffers one sweep reuses for every graph it visits.
#[derive(Default)]
struct Scratches {
    post: PostDominatorScratch,
    frontier: FrontierScratch,
}

/// Check every query of a tree against the brute-force reference.
fn check_against_definition(edges: &EdgeList, scratches: &mut Scratches) {
    let graph = edges.graph();
    let post = PostDominatorTree::compute_in(&mut scratches.post, &graph);
    let reference = PostReference::new(&graph);
    let sets = all_subsets(edges.node_count);
    assert_matches(&post, &reference, edges);
    assert_contract(&post, &reference, edges);
    assert_frontiers(
        &post,
        &graph,
        &reference,
        &mut scratches.frontier,
        &sets,
        edges,
    );

    for members in &sets {
        assert_eq!(
            post.nearest_common_post_dominator_of(blocks(members)),
            reference.nearest_common(members).map(post_node),
            "nearest common post-dominator of {members:?} in {edges:?}"
        );
    }
}

#[test]
fn every_digraph_of_up_to_four_nodes_matches_the_definition() {
    let mut scratches = Scratches::default();
    for node_count in 1..=4 {
        for_each_multigraph(node_count, 1, |edges| {
            check_against_definition(edges, &mut scratches);
        });
    }
}

#[test]
fn every_multigraph_of_up_to_three_nodes_matches_the_definition() {
    let mut scratches = Scratches::default();
    for node_count in 1..=3 {
        for_each_multigraph(node_count, 2, |edges| {
            check_against_definition(edges, &mut scratches);
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
