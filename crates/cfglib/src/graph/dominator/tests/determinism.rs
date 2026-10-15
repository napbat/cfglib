//! The tree is a function of the set of edges, not of how they are listed.

use super::support::{EdgeList, Family, Recipe, Rng, random_graph};
use crate::graph::dominator::DominatorTree;

/// `edges` with every edge listed again at a random place with probability
/// one in `outcomes`.
pub(super) fn with_extra_copies(rng: &mut Rng, edges: &EdgeList, outcomes: usize) -> EdgeList {
    let mut listed = edges.edges.clone();
    for &edge in &edges.edges {
        if rng.one_in(outcomes) {
            let place = rng.below(listed.len() + 1);
            listed.insert(place, edge);
        }
    }
    EdgeList::new(edges.node_count, listed)
}

#[test]
fn permuting_successor_order_leaves_every_array_identical() {
    for seed in 0..24 {
        let mut rng = Rng::new(seed);
        let recipe = Recipe {
            family: Family::ALL[usize::try_from(seed % 3).expect("a residue fits in usize")],
            node_count: [12, 80, 600][usize::try_from(seed % 3).expect("a residue fits in usize")],
            parallel: false,
            detached: 5,
        };
        let edges = random_graph(&mut rng, recipe);
        let expected = DominatorTree::compute_from(&edges.graph(), 0);
        for _ in 0..4 {
            let mut shuffled = edges.edges.clone();
            rng.shuffle(&mut shuffled);
            let permuted = EdgeList::new(edges.node_count, shuffled);
            assert_eq!(
                DominatorTree::compute_from(&permuted.graph(), 0),
                expected,
                "{recipe:?} seed {seed}"
            );
        }
    }
}

#[test]
fn changing_parallel_edge_multiplicity_leaves_every_array_identical() {
    for seed in 0..24 {
        let mut rng = Rng::new(seed + 1_000);
        let recipe = Recipe {
            family: Family::ALL[usize::try_from(seed % 3).expect("a residue fits in usize")],
            node_count: [12, 80, 600][usize::try_from(seed % 3).expect("a residue fits in usize")],
            parallel: false,
            detached: 5,
        };
        let edges = random_graph(&mut rng, recipe);
        let expected = DominatorTree::compute_from(&edges.graph(), 0);

        let sprinkled = with_extra_copies(&mut rng, &edges, 3);
        assert_eq!(
            DominatorTree::compute_from(&sprinkled.graph(), 0),
            expected,
            "{recipe:?} seed {seed}"
        );
        let tripled = edges.repeated(3);
        assert_eq!(
            DominatorTree::compute_from(&tripled.graph(), 0),
            expected,
            "{recipe:?} seed {seed}"
        );
    }
}

#[test]
fn a_graph_listed_in_another_order_yields_the_same_tree_as_its_duplicate_free_form() {
    let mut rng = Rng::new(7);
    let recipe = Recipe {
        family: Family::Irreducible,
        node_count: 300,
        parallel: true,
        detached: 20,
    };
    let noisy = random_graph(&mut rng, recipe);
    let mut unique = noisy.edges.clone();
    unique.sort_unstable();
    unique.dedup();
    let unique = EdgeList::new(noisy.node_count, unique);
    assert_eq!(
        DominatorTree::compute_from(&noisy.graph(), 0),
        DominatorTree::compute_from(&unique.graph(), 0)
    );
}
