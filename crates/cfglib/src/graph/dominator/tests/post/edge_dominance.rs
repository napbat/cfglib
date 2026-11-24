//! Edge post-dominance against its definition: an edge post-dominates a
//! block when deleting that one edge leaves the block no way to an exit or a
//! cut root.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::reference::PostReference;
use crate::graph::dominator::tests::support::{EdgeList, for_each_multigraph, node};
use crate::graph::dominator::{PostDominatorScratch, PostDominatorTree};

/// The blocks with a path to one of `joined` over `edges` without the edge
/// at index `skipped`.
fn reaching(edges: &EdgeList, joined: &[usize], skipped: usize) -> Vec<bool> {
    let mut reaches = vec![false; edges.node_count];
    let mut stack = joined.to_vec();
    for &block in joined {
        reaches[block] = true;
    }
    while let Some(to) = stack.pop() {
        for (index, &(tail, head)) in edges.edges.iter().enumerate() {
            let tail = tail as usize;
            if index != skipped && head as usize == to && !reaches[tail] {
                reaches[tail] = true;
                stack.push(tail);
            }
        }
    }
    reaches
}

/// Check every edge, present or not, and every block against the
/// definition, on the cut the reference derives independently.
fn check_against_definition(edges: &EdgeList, scratch: &mut PostDominatorScratch) {
    let graph = edges.graph();
    let post = PostDominatorTree::compute_in(scratch, &graph);
    let reference = PostReference::new(&graph);
    let joined: Vec<usize> = reference
        .exits()
        .iter()
        .chain(reference.cut_roots())
        .copied()
        .collect();
    let count = edges.node_count;
    for from in 0..count {
        for to in 0..count {
            let occurrence = edges
                .edges
                .iter()
                .position(|&edge| edge == (node(from), node(to)));
            let cut = occurrence.map(|index| reaching(edges, &joined, index));
            for block in 0..count {
                let expected = cut
                    .as_ref()
                    .is_some_and(|cut| reference.contains(block) && !cut[block]);
                assert_eq!(
                    post.edge_post_dominates(&graph, node(from), node(to), node(block)),
                    expected,
                    "{from} -> {to} post-dominates {block} in {edges:?}"
                );
            }
        }
    }
}

#[test]
fn every_digraph_of_up_to_four_nodes_matches_the_definition() {
    let mut scratch = PostDominatorScratch::new();
    for node_count in 1..=4 {
        for_each_multigraph(node_count, 1, |edges| {
            check_against_definition(edges, &mut scratch);
        });
    }
}

#[test]
fn every_multigraph_of_up_to_three_nodes_matches_the_definition() {
    let mut scratch = PostDominatorScratch::new();
    for node_count in 1..=3 {
        for_each_multigraph(node_count, 2, |edges| {
            check_against_definition(edges, &mut scratch);
        });
    }
}

#[test]
fn no_edge_out_of_a_cut_root_post_dominates() {
    // entry -> head -> body -> head: the body closes the loop and is cut.
    let edges = EdgeList::new(3, vec![(0, 1), (1, 2), (2, 1)]);
    let graph = edges.graph();
    let post = PostDominatorTree::compute(&graph);
    assert_eq!(post.cut_roots(), [2]);
    for block in 0..3 {
        assert!(!post.edge_post_dominates(&graph, 2, 1, block));
    }
    assert!(post.edge_post_dominates(&graph, 1, 2, 1));
    assert!(post.edge_post_dominates(&graph, 0, 1, 0));
}

#[test]
fn neither_of_two_parallel_edges_post_dominates_their_tail() {
    let edges = EdgeList::new(2, vec![(0, 1), (0, 1)]);
    let graph = edges.graph();
    let post = PostDominatorTree::compute(&graph);
    assert!(post.post_dominates(1, 0));
    assert!(!post.edge_post_dominates(&graph, 0, 1, 0));
}
