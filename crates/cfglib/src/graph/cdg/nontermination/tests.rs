use super::*;
use crate::graph::store::{Graph, NodeId};
use crate::test_util::postdominance::{Fixpoint, post_dominance, two_successor_dependences};
use crate::test_util::rng::Rng;
use alloc::vec;

/// The graph over `count` nodes with `edges`, by node index.
fn graph_of(count: usize, edges: &[(usize, usize)]) -> Graph<(), ()> {
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..count).map(|_| graph.add_node(())).collect();
    for &(from, to) in edges {
        graph.add_edge(nodes[from], nodes[to], ());
    }
    graph
}

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
        nontermination_sensitive_controllers(
            &graph,
            NodeId::from_index(dependent),
            scratch,
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

/// Visit every graph over `count` nodes in which each ordered pair, self-loops
/// included, is an edge `0..=copies` times.
fn for_each_multigraph(count: usize, copies: usize, mut visit: impl FnMut(&[(usize, usize)])) {
    let mut multiplicity = vec![0; count * count];
    loop {
        let edges: Vec<(usize, usize)> = multiplicity
            .iter()
            .enumerate()
            .flat_map(|(pair, &times)| core::iter::repeat_n((pair / count, pair % count), times))
            .collect();
        visit(&edges);
        let Some(digit) = multiplicity.iter().position(|&times| times < copies) else {
            return;
        };
        multiplicity[..digit].fill(0);
        multiplicity[digit] += 1;
    }
}

#[test]
fn every_digraph_of_up_to_four_nodes_matches_the_definition() {
    let mut scratch = NonterminationScratch::new();
    for count in 1..=4 {
        for_each_multigraph(count, 1, |edges| check(count, edges, &mut scratch));
    }
}

#[test]
fn every_multigraph_of_up_to_three_nodes_matches_the_definition() {
    let mut scratch = NonterminationScratch::new();
    for count in 1..=3 {
        for_each_multigraph(count, 2, |edges| check(count, edges, &mut scratch));
    }
}

#[test]
fn random_graphs_match_the_definition() {
    let mut rng = Rng::new(62);
    let mut scratch = NonterminationScratch::new();
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
    let mut scratch = NonterminationScratch::new();
    nontermination_sensitive_controllers(
        &graph,
        NodeId::from_index(0),
        &mut scratch,
        &mut controllers,
    );
    assert_eq!(controllers, [NodeId::from_index(0)]);
    nontermination_sensitive_controllers(
        &graph,
        NodeId::from_index(1),
        &mut scratch,
        &mut controllers,
    );
    assert_eq!(controllers, [NodeId::from_index(0)]);
}

#[test]
fn removed_slots_and_a_reused_scratch_change_nothing() {
    let mut graph = graph_of(4, &[(0, 1), (0, 2), (1, 3), (2, 3), (0, 3)]);
    assert!(graph.remove_node(NodeId::from_index(2)));
    let mut scratch = NonterminationScratch::new();
    let mut controllers = Vec::new();
    for _ in 0..2 {
        nontermination_sensitive_controllers(
            &graph,
            NodeId::from_index(1),
            &mut scratch,
            &mut controllers,
        );
        assert_eq!(controllers, [NodeId::from_index(0)]);
        nontermination_sensitive_controllers(
            &graph,
            NodeId::from_index(3),
            &mut scratch,
            &mut controllers,
        );
        assert_eq!(controllers, Vec::<NodeId>::new());
    }
}
