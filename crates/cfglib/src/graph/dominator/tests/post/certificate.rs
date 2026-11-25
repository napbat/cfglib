//! The low-high certificate of post-dominator trees.

extern crate alloc;
use alloc::vec;

use crate::graph::dominator::tests::support::{Digraph, EdgeList, for_each_multigraph};
use crate::graph::dominator::{
    CertificateError, LowHighOrder, PostDominatorScratch, PostDominatorTree,
};

/// Check that the certified construction builds `post`, the post-dominator
/// tree of `graph`, and that its certificate is accepted.
pub(super) fn assert_post_certified(
    post: &PostDominatorTree<u32>,
    graph: &Digraph,
    scratch: &mut PostDominatorScratch,
) {
    let (certified, certificate) = PostDominatorTree::compute_certified_in(scratch, graph);
    assert!(&certified == post, "the certified tree differs");
    assert_eq!(certified.verify_certificate(graph, &certificate), Ok(()));
    assert_eq!(certificate.order().len(), post.preorder().len());
}

#[test]
fn a_tree_is_rejected_on_a_graph_one_edge_away_when_their_trees_differ() {
    let mut scratch = PostDominatorScratch::new();
    for_each_multigraph(3, 1, |edges| {
        let graph = edges.graph();
        let (post, certificate) = PostDominatorTree::compute_certified_in(&mut scratch, &graph);
        for from in 0..3 {
            for to in 0..3 {
                let mut more = edges.edges.clone();
                more.push((from, to));
                let other = EdgeList::new(3, more).graph();
                let verdict = post.verify_certificate(&other, &certificate);
                if PostDominatorTree::compute(&other) != post {
                    assert!(verdict.is_err(), "{edges:?} plus {from} -> {to}");
                }
            }
        }
    });
}

#[test]
fn a_changed_exit_structure_or_node_bound_is_reported() {
    // entry -> exit; adding exit -> entry turns the graph into a loop that
    // the cut closes at the old exit.
    let graph = EdgeList::new(2, vec![(0, 1)]).graph();
    let looping = EdgeList::new(2, vec![(0, 1), (1, 0)]).graph();
    let larger = EdgeList::new(3, vec![(0, 1)]).graph();
    let mut scratch = PostDominatorScratch::new();
    let (post, certificate) = PostDominatorTree::compute_certified_in(&mut scratch, &graph);
    assert_eq!(
        post.verify_certificate(&looping, &certificate),
        Err(CertificateError::ExitStructure)
    );
    assert_eq!(
        post.verify_certificate(&larger, &certificate),
        Err(CertificateError::NodeBound {
            graph: 3,
            tree: 2,
            certificate: 2
        })
    );
    let backwards = LowHighOrder::from_order(vec![0, 1], 2);
    assert_eq!(
        post.verify_certificate(&graph, &backwards),
        Err(CertificateError::Preorder(0))
    );
    assert_eq!(certificate.order(), [1, 0]);
}
