//! Every small graph, and seeded random larger ones, against the definitions.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::{assert_matches_reference, assert_tree_invariants, build};
use crate::block::BlockId;
use crate::edge::EdgeId;
use crate::graph::program_structure::{ProgramStructureTree, SeseEdge};
use crate::test_util::rng::Rng;

/// Visit every graph over `node_count` nodes in which each ordered pair of
/// nodes, self-loops included, is an edge `0..=copies` times.
fn for_each_graph(node_count: usize, copies: usize, mut visit: impl FnMut(&[(usize, usize)])) {
    let pairs: Vec<(usize, usize)> = (0..node_count)
        .flat_map(|from| (0..node_count).map(move |to| (from, to)))
        .collect();
    let mut counts = vec![0_usize; pairs.len()];
    let mut edges = Vec::new();
    loop {
        edges.clear();
        for (&pair, &count) in pairs.iter().zip(&counts) {
            edges.extend(core::iter::repeat_n(pair, count));
        }
        visit(&edges);
        let Some(position) = counts.iter().position(|&count| count < copies) else {
            return;
        };
        counts[position] += 1;
        counts[..position].fill(0);
    }
}

#[test]
fn every_digraph_of_up_to_four_nodes_matches_the_definitions() {
    for node_count in 1..=4 {
        for_each_graph(node_count, 1, |edges| {
            assert_matches_reference(&build(node_count, edges).0);
        });
    }
}

#[test]
fn every_multigraph_of_up_to_three_nodes_matches_the_definitions() {
    for node_count in 1..=3 {
        for_each_graph(node_count, 2, |edges| {
            assert_matches_reference(&build(node_count, edges).0);
        });
    }
}

/// A random multigraph over `node_count` nodes, self-loops included.
fn random_edges(rng: &mut Rng, node_count: usize, edge_count: usize) -> Vec<(usize, usize)> {
    (0..edge_count)
        .map(|_| (rng.below(node_count), rng.below(node_count)))
        .collect()
}

#[test]
fn random_five_node_multigraphs_match_the_definitions() {
    let mut rng = Rng::new(66);
    for _ in 0..20_000 {
        let edge_count = rng.below(13);
        let edges = random_edges(&mut rng, 5, edge_count);
        let (mut cfg, _, ids) = build(5, &edges);
        if !ids.is_empty() && rng.one_in(4) {
            cfg.remove_edge(ids[rng.below(ids.len())]);
        }
        assert_matches_reference(&cfg);
    }
}

/// The ends of a boundary as block indices, `None` for the virtual exit or
/// start, which do not depend on how edges are numbered.
type Ends = (Option<usize>, Option<usize>);

/// The ends of `edge`, whose identities are `ids` in the order of `list`.
fn ends(list: &[(usize, usize)], ids: &[EdgeId], edge: SeseEdge) -> Ends {
    match edge {
        SeseEdge::Entry => (None, Some(0)),
        SeseEdge::Exit(block) => (Some(block.index()), None),
        SeseEdge::Edge(id) => {
            let (from, to) = list[ids
                .iter()
                .position(|&known| known == id)
                .expect("a boundary is an edge of the graph")];
            (Some(from), Some(to))
        }
    }
}

/// Every region of the tree of `list` by the ends of its boundaries and its
/// sorted blocks.
fn region_shapes(
    node_count: usize,
    list: &[(usize, usize)],
) -> BTreeSet<(Ends, Ends, Vec<BlockId>)> {
    let (cfg, _, ids) = build(node_count, list);
    let tree = ProgramStructureTree::compute(&cfg);
    assert_tree_invariants(&tree, &cfg);
    tree.regions()
        .map(|region| {
            let mut blocks = tree.blocks(region).to_vec();
            blocks.sort_unstable();
            (
                ends(list, &ids, tree.entry_edge(region)),
                ends(list, &ids, tree.exit_edge(region)),
                blocks,
            )
        })
        .collect()
}

#[test]
fn edge_order_changes_no_region() {
    let mut rng = Rng::new(0x5e5e);
    for _ in 0..300 {
        let node_count = 2 + rng.below(30);
        let edge_count = rng.below(3 * node_count);
        let mut edges: BTreeSet<(usize, usize)> = random_edges(&mut rng, node_count, edge_count)
            .into_iter()
            .collect();
        // A spine, so most blocks are reachable.
        edges.extend((1..node_count).map(|next| (rng.below(next), next)));
        let mut edges: Vec<(usize, usize)> = edges.into_iter().collect();
        let expected = region_shapes(node_count, &edges);
        rng.shuffle(&mut edges);
        assert_eq!(region_shapes(node_count, &edges), expected);
    }
}
