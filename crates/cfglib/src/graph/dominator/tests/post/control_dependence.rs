//! The control-dependence graph, the walk behind it, and the frontier against
//! the definition.
//!
//! The definition is the one of Pingali and Bilardi, read literally off the
//! brute-force post-dominance of [`PostReference`]: a block `w` is control
//! dependent on the arm `u -> t` when `w` post-dominates `t` and does not
//! strictly post-dominate `u`. Nothing here shares code with the walk up the
//! tree or with the frontier, so a disagreement convicts the implementation.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::differential::{SMALL_SEEDS, small_graph};
use super::reference::{PostReference, blocks};
use crate::graph::cdg::control_dependence_graph;
use crate::graph::dominator::PostDominatorTree;
use crate::graph::dominator::tests::support::{Digraph, EdgeList, Rng, for_each_multigraph, node};
use crate::graph::view::GraphView;

/// The blocks control dependent on the arm `controller -> target`, in
/// ascending index, by definition.
fn dependents_by_definition(
    reference: &PostReference,
    count: usize,
    controller: usize,
    target: usize,
) -> Vec<usize> {
    (0..count)
        .filter(|&block| {
            reference.post_dominates(block, target)
                && (block == controller || !reference.post_dominates(block, controller))
        })
        .collect()
}

/// Every edge `(controller, dependent)` of the control-dependence graph by
/// definition: an arm out of a block the entry reaches, and a block that
/// depends on that arm.
fn dependences_by_definition(graph: &Digraph, reference: &PostReference) -> BTreeSet<(u32, u32)> {
    let count = graph.node_bound();
    let mut edges = BTreeSet::new();
    for controller in (0..count).filter(|&block| reference.contains(block)) {
        for target in graph.successors(node(controller)) {
            for dependent in dependents_by_definition(reference, count, controller, target as usize)
            {
                edges.insert((node(controller), node(dependent)));
            }
        }
    }
    edges
}

/// Check the control-dependence graph, the dependents of every arm, and the
/// frontier of every block of one graph against the definition.
///
/// Returns how many blocks the entry reaches that have a record back to
/// themselves, for a sweep to show that it met the case.
fn check_against_definition(edges: &EdgeList) -> usize {
    let graph = edges.graph();
    let post = PostDominatorTree::compute(&graph);
    let reference = PostReference::new(&graph);
    let count = edges.node_count;
    let expected: Vec<(u32, u32)> = dependences_by_definition(&graph, &reference)
        .into_iter()
        .collect();

    let dependences = control_dependence_graph(&graph, &post);
    assert_eq!(dependences.node_bound(), count, "nodes of {edges:?}");
    let actual: Vec<(u32, u32)> = dependences
        .edges()
        .map(|edge| {
            (
                *dependences.node(edge.source()),
                *dependences.node(edge.target()),
            )
        })
        .collect();
    assert_eq!(actual, expected, "control-dependence edges of {edges:?}");

    let looping: Vec<usize> = (0..count)
        .filter(|&block| {
            reference.contains(block) && edges.edges.contains(&(node(block), node(block)))
        })
        .collect();
    for &block in &looping {
        assert!(
            actual.contains(&(node(block), node(block))),
            "the record from {block} to itself gives no edge in {edges:?}"
        );
    }

    for controller in (0..count).filter(|&block| reference.contains(block)) {
        for target in graph.successors(node(controller)) {
            let got: Vec<u32> = post.control_dependents(node(controller), target).collect();
            let wanted = blocks(&dependents_by_definition(
                &reference,
                count,
                controller,
                target as usize,
            ));
            assert!(
                got.first().is_none_or(|&first| first == target),
                "{got:?} does not start at {target} for the arm {controller} -> {target} in {edges:?}"
            );
            assert!(
                got.windows(2).all(|pair| {
                    pair[0] != pair[1]
                        && reference.post_dominates(pair[1] as usize, pair[0] as usize)
                }),
                "{got:?} is not a climb up the tree for the arm {controller} -> {target} in {edges:?}"
            );
            let mut sorted = got;
            sorted.sort_unstable();
            assert_eq!(
                sorted, wanted,
                "dependents of the arm {controller} -> {target} in {edges:?}"
            );
        }
    }

    for from in 0..count {
        for to in 0..count {
            if !reference.contains(from) || !reference.contains(to) {
                assert_eq!(
                    post.control_dependents(node(from), node(to)).count(),
                    0,
                    "dependents of {from} -> {to}, a pair outside the tree, in {edges:?}"
                );
            }
        }
    }

    let mut sources = Vec::new();
    for dependent in 0..count {
        post.frontier(&graph, node(dependent), &mut sources);
        sources.sort_unstable();
        let controllers: Vec<u32> = actual
            .iter()
            .filter(|&&(_, to)| to == node(dependent))
            .map(|&(from, _)| from)
            .collect();
        assert_eq!(
            sources, controllers,
            "frontier of {dependent} against the edges into it in {edges:?}"
        );
    }
    looping.len()
}

#[test]
fn every_digraph_of_up_to_four_nodes_has_the_defined_control_dependences() {
    let (mut visited, mut looping) = (0_usize, 0_usize);
    for node_count in 1..=4 {
        for_each_multigraph(node_count, 1, |edges| {
            visited += 1;
            looping += check_against_definition(edges);
        });
    }
    assert_eq!(visited, 2 + 16 + 512 + 65_536);
    assert_eq!(
        looping, 107_085,
        "blocks the entry reaches that loop to themselves"
    );
}

#[test]
fn every_multigraph_of_up_to_three_nodes_has_the_defined_control_dependences() {
    let (mut visited, mut looping) = (0_usize, 0_usize);
    for node_count in 1..=3 {
        for_each_multigraph(node_count, 2, |edges| {
            visited += 1;
            looping += check_against_definition(edges);
        });
    }
    assert_eq!(visited, 3 + 81 + 19_683);
    assert_eq!(
        looping, 34_598,
        "blocks the entry reaches that loop to themselves"
    );
}

#[test]
fn random_graphs_have_the_defined_control_dependences() {
    let mut looping = 0;
    for seed in 0..SMALL_SEEDS {
        let mut rng = Rng::new(seed);
        let (_, edges) = small_graph(seed, &mut rng);
        looping += check_against_definition(&edges);
    }
    assert!(looping > 0, "no graph had a block that loops to itself");
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn a_block_beyond_the_node_bound_panics_as_contains_does() {
    let post = PostDominatorTree::compute(&EdgeList::new(2, vec![(0, 1)]).graph());
    let _ = post.control_dependents(0, 2);
}
