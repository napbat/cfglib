//! Simple Lengauer-Tarjan against the iterative oracle on random graphs.

use super::support::{Family, Recipe, Rng, assert_forest_invariants, node, random_graph};
use crate::graph::dominator::{DominatorScratch, DominatorTree};
use crate::graph::structure::is_reducible;
use crate::graph::view::GraphView;

/// Graph sizes and how many seeds to try at each.
const SWEEP: [(usize, u64); 6] = [
    (10, 150),
    (30, 80),
    (100, 40),
    (500, 12),
    (2_000, 4),
    (10_000, 2),
];

/// The family a seed selects.
fn family_of(seed: u64) -> Family {
    Family::ALL[usize::try_from(seed % 3).expect("a residue fits in usize")]
}

#[test]
fn slt_matches_the_iterative_oracle_on_random_graphs() {
    // One scratch serves the whole sweep, so reuse across graphs of very
    // different sizes and shapes is part of what is compared.
    let mut scratch = DominatorScratch::new();
    for (node_count, seeds) in SWEEP {
        let size = u64::try_from(node_count).expect("a graph size fits in u64");
        for seed in 0..seeds {
            let mut rng = Rng::new(seed * 1_000_003 + size);
            let recipe = Recipe {
                family: family_of(seed),
                node_count,
                parallel: seed % 2 == 1,
                detached: if seed % 4 >= 2 { node_count / 4 } else { 0 },
            };
            let edges = random_graph(&mut rng, recipe);
            let graph = edges.graph();
            let root = if seed % 5 == 4 {
                node(rng.below(edges.node_count))
            } else {
                0
            };

            let tree = DominatorTree::compute_from_in(&mut scratch, &graph, root);
            let oracle = DominatorTree::compute_iterative(&graph, root);
            assert!(tree == oracle, "{recipe:?} seed {seed} root {root}");
            assert_forest_invariants(&tree, graph.node_bound());
        }
    }
}

#[test]
fn the_sweep_reaches_irreducible_graphs_and_unreachable_nodes() {
    let mut irreducible = 0;
    let mut with_unreachable = 0;
    for seed in 0..40 {
        let mut rng = Rng::new(seed);
        let recipe = Recipe {
            family: Family::Irreducible,
            node_count: 100,
            parallel: false,
            detached: 25,
        };
        let graph = random_graph(&mut rng, recipe).graph();
        let tree = DominatorTree::compute_from(&graph, 0);
        irreducible += usize::from(!is_reducible(&graph, &tree));
        with_unreachable += usize::from(tree.preorder().len() < graph.node_bound());
    }
    assert!(
        irreducible >= 30,
        "only {irreducible} of 40 are irreducible"
    );
    assert_eq!(with_unreachable, 40, "detached nodes must stay unreachable");
}

#[test]
fn a_reused_scratch_computes_what_a_fresh_one_does() {
    let mut scratch = DominatorScratch::new();
    for seed in 0..60 {
        let mut rng = Rng::new(seed);
        let sizes = [8, 400, 12, 90];
        let recipe = Recipe {
            family: family_of(seed),
            node_count: sizes[usize::try_from(seed % 4).expect("a residue fits in usize")],
            parallel: seed % 2 == 0,
            detached: 3,
        };
        let graph = random_graph(&mut rng, recipe).graph();
        assert_eq!(
            DominatorTree::compute_from_in(&mut scratch, &graph, 0),
            DominatorTree::compute_from(&graph, 0),
            "{recipe:?} seed {seed}"
        );
    }
}
