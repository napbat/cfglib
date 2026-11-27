use super::*;
use crate::graph::store::NodeId;
use crate::test_util::digraphs::{for_each_multigraph, graph_of};
use crate::test_util::postdominance::{Fixpoint, post_dominance, two_successor_dependences};
use crate::test_util::rng::Rng;
use alloc::vec;

/// Check every dependent of the graph against the two-successor definition
/// over max post-dominance, the least fixpoint of Hecker et al.'s rules,
/// which holds exactly when every maximal path passes the block.
fn check(count: usize, edges: &[(usize, usize)], scratch: &mut NonterminationScratch) {
    let mut successors = vec![Vec::new(); count];
    for &(from, to) in edges {
        successors[from].push(to);
    }
    let relation = post_dominance(&successors, Fixpoint::Max);
    let expected = two_successor_dependences(&successors, &relation);
    let graph = graph_of(count, edges);
    let mut controllers = vec![NodeId::from_index(99)];
    for dependent in 0..count {
        nontermination_sensitive_controllers_in(
            scratch,
            &graph,
            NodeId::from_index(dependent),
            &mut controllers,
        );
        let wanted: Vec<NodeId> = expected
            .iter()
            .filter(|&&(_, target)| target == dependent)
            .map(|&(controller, _)| NodeId::from_index(controller))
            .collect();
        assert_eq!(
            controllers, wanted,
            "{dependent} in {count} nodes, {edges:?}"
        );
    }
}

#[test]
fn every_digraph_of_up_to_four_nodes_matches_the_definition() {
    let mut scratch = NonterminationScratch::default();
    for count in 1..=4 {
        for_each_multigraph(count, 1, |edges| check(count, edges, &mut scratch));
    }
}

#[test]
fn every_multigraph_of_up_to_three_nodes_matches_the_definition() {
    let mut scratch = NonterminationScratch::default();
    for count in 1..=3 {
        for_each_multigraph(count, 2, |edges| check(count, edges, &mut scratch));
    }
}

#[test]
fn random_graphs_match_the_definition() {
    let mut rng = Rng::new(62);
    let mut scratch = NonterminationScratch::default();
    for _ in 0..300 {
        let count = 3 + rng.below(12);
        let edges: Vec<(usize, usize)> = (0..count + rng.below(2 * count))
            .map(|_| (rng.below(count), rng.below(count)))
            .collect();
        check(count, &edges, &mut scratch);
    }
}

#[test]
fn a_branch_that_must_loop_back_controls_itself() {
    // 0 -> 0 | 1: every path from the self-loop passes 0, one from 1 does
    // not, so 0 controls 0; and 0 controls 1, which one arm must reach.
    let graph = graph_of(2, &[(0, 0), (0, 1)]);
    let mut controllers = Vec::new();
    let mut scratch = NonterminationScratch::default();
    nontermination_sensitive_controllers_in(
        &mut scratch,
        &graph,
        NodeId::from_index(0),
        &mut controllers,
    );
    assert_eq!(controllers, [NodeId::from_index(0)]);
    nontermination_sensitive_controllers_in(
        &mut scratch,
        &graph,
        NodeId::from_index(1),
        &mut controllers,
    );
    assert_eq!(controllers, [NodeId::from_index(0)]);
}

#[test]
fn removed_slots_and_a_reused_scratch_change_nothing() {
    let mut graph = graph_of(4, &[(0, 1), (0, 2), (1, 3), (2, 3), (0, 3)]);
    assert!(graph.remove_node(NodeId::from_index(2)));
    let mut scratch = NonterminationScratch::default();
    let mut controllers = Vec::new();
    for _ in 0..2 {
        nontermination_sensitive_controllers_in(
            &mut scratch,
            &graph,
            NodeId::from_index(1),
            &mut controllers,
        );
        assert_eq!(controllers, [NodeId::from_index(0)]);
        nontermination_sensitive_controllers_in(
            &mut scratch,
            &graph,
            NodeId::from_index(3),
            &mut controllers,
        );
        assert_eq!(controllers, Vec::<NodeId>::new());
    }
}
