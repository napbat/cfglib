//! The post-dominator tree against the definition and the iterative oracle on
//! random graphs, and the invariances the cut promises.

extern crate alloc;
use alloc::vec::Vec;

use super::reference::{PostReference, assert_contract, assert_frontiers, assert_matches};
use crate::graph::dominator::tests::determinism::with_extra_copies;
use crate::graph::dominator::tests::support::{EdgeList, Family, Recipe, Rng, node, random_graph};
use crate::graph::dominator::{
    DominatorTree, FrontierScratch, PostDominatorScratch, PostDominatorTree, PostNode,
};
use crate::graph::traverse::{TraversalDirection, depth_first_preorder};
use crate::graph::view::GraphView;

/// The family a seed selects.
fn family_of(seed: u64) -> Family {
    Family::ALL[usize::try_from(seed % 3).expect("a residue fits in usize")]
}

/// A graph small enough for the brute-force reference, in the family a seed
/// selects, with parallel edges and unreachable blocks on alternate seeds.
/// Every third graph also gets one exit and one spin, hung from random
/// blocks, because the families alone seldom produce both.
pub(super) fn small_graph(seed: u64, rng: &mut Rng) -> (Recipe, EdgeList) {
    let recipe = Recipe {
        family: family_of(seed),
        node_count: [6, 9, 14][usize::try_from(seed % 3).expect("a residue fits in usize")],
        parallel: seed % 2 == 1,
        detached: if seed % 4 >= 2 { 3 } else { 0 },
    };
    let mut edges = random_graph(rng, recipe);
    if seed.is_multiple_of(3) {
        let (exit, spin) = (node(edges.node_count), node(edges.node_count + 1));
        edges.edges.push((node(rng.below(edges.node_count)), exit));
        edges.edges.push((node(rng.below(edges.node_count)), spin));
        edges.edges.push((spin, spin));
        edges.node_count += 2;
    }
    (recipe, edges)
}

/// How many small graphs a sweep visits.
pub(super) const SMALL_SEEDS: u64 = 240;

#[test]
fn random_graphs_match_the_definition() {
    let mut scratch = PostDominatorScratch::new();
    let mut frontier_scratch = FrontierScratch::new();
    for seed in 0..SMALL_SEEDS {
        let mut rng = Rng::new(seed);
        let (recipe, edges) = small_graph(seed, &mut rng);
        let graph = edges.graph();
        let post = PostDominatorTree::compute_in(&mut scratch, &graph);
        let reference = PostReference::new(&graph);
        assert_matches(&post, &reference, &(recipe, seed));
        assert_contract(&post, &reference, &(recipe, seed));

        let sets: Vec<Vec<usize>> = (0..12)
            .map(|_| (0..edges.node_count).filter(|_| rng.one_in(3)).collect())
            .collect();
        let context = &(recipe, seed);
        assert_frontiers(
            &post,
            &graph,
            &reference,
            &mut frontier_scratch,
            &sets,
            context,
        );
    }
}

#[test]
fn the_random_graphs_reach_exits_cuts_and_both() {
    let (mut with_exits, mut with_cuts, mut with_both) = (0, 0, 0);
    for seed in 0..SMALL_SEEDS {
        let mut rng = Rng::new(seed);
        let post = PostDominatorTree::compute(&small_graph(seed, &mut rng).1.graph());
        let (exits, cuts) = (!post.exits().is_empty(), !post.cut_roots().is_empty());
        with_exits += usize::from(exits);
        with_cuts += usize::from(cuts);
        with_both += usize::from(exits && cuts);
    }
    assert!(with_exits >= 40, "only {with_exits} have an exit");
    assert!(with_cuts >= 40, "only {with_cuts} need a cut");
    assert!(with_both >= 30, "only {with_both} have both");
}

/// The blocks `edges` reach from node 0, by a search that shares no code with
/// the tree.
fn reachable_from_entry(edges: &EdgeList) -> Vec<bool> {
    let mut reachable = alloc::vec![false; edges.node_count];
    for block in depth_first_preorder(&edges.graph(), 0, TraversalDirection::Outgoing) {
        reachable[block as usize] = true;
    }
    reachable
}

#[test]
fn large_graphs_match_the_iterative_oracle_on_the_augmented_graph() {
    for (node_count, seeds) in [(100, 20), (500, 8), (2_000, 3), (10_000, 1)] {
        let size = u64::try_from(node_count).expect("a graph size fits in u64");
        for seed in 0..seeds {
            let mut rng = Rng::new(seed * 7_919 + size);
            let recipe = Recipe {
                family: family_of(seed),
                node_count,
                parallel: seed % 2 == 1,
                detached: node_count / 10,
            };
            let edges = random_graph(&mut rng, recipe);
            let graph = edges.graph();
            let post = PostDominatorTree::compute(&graph);
            let reachable = reachable_from_entry(&edges);

            let exits: Vec<u32> = (0..edges.node_count)
                .filter(|&block| reachable[block] && graph.successors(node(block)).next().is_none())
                .map(node)
                .collect();
            assert_eq!(post.exits(), exits, "{recipe:?} seed {seed}");

            // The augmented reverse graph, written out as an edge list: the
            // reverse of every edge that starts at a reachable block, and an
            // edge from the virtual exit to every exit and cut root.
            let virtual_exit = node(edges.node_count);
            let mut augmented: Vec<(u32, u32)> = edges
                .edges
                .iter()
                .filter(|&&(from, _)| reachable[from as usize])
                .map(|&(from, to)| (to, from))
                .collect();
            augmented.extend(
                post.exits()
                    .iter()
                    .chain(post.cut_roots())
                    .map(|&block| (virtual_exit, block)),
            );
            let augmented = EdgeList::new(edges.node_count + 1, augmented).graph();
            let oracle = DominatorTree::compute_iterative(&augmented, virtual_exit);

            for (block, &is_reachable) in reachable.iter().enumerate() {
                let id = node(block);
                if !is_reachable {
                    assert!(!post.contains(id), "{block}: {recipe:?} seed {seed}");
                    continue;
                }
                assert!(
                    oracle.is_reachable(id),
                    "{block} has no path to the virtual exit: {recipe:?} seed {seed}"
                );
                let expected = match oracle.idom(id) {
                    Some(parent) if parent != virtual_exit => PostNode::Node(parent),
                    _ => PostNode::Exit,
                };
                assert_eq!(
                    post.ipdom(id),
                    Some(expected),
                    "ipdom({block}): {recipe:?} seed {seed}"
                );
            }
            assert_eq!(
                post.preorder().len(),
                reachable.iter().filter(|&&reachable| reachable).count()
            );
        }
    }
}

#[test]
fn permuting_successor_order_leaves_the_whole_tree_identical() {
    for seed in 0..24 {
        let mut rng = Rng::new(seed);
        let recipe = Recipe {
            family: family_of(seed),
            node_count: [12, 80, 600][usize::try_from(seed % 3).expect("a residue fits in usize")],
            parallel: false,
            detached: 5,
        };
        let edges = random_graph(&mut rng, recipe);
        let expected = PostDominatorTree::compute(&edges.graph());
        for _ in 0..4 {
            let mut shuffled = edges.edges.clone();
            rng.shuffle(&mut shuffled);
            let permuted = EdgeList::new(edges.node_count, shuffled);
            assert_eq!(
                PostDominatorTree::compute(&permuted.graph()),
                expected,
                "{recipe:?} seed {seed}"
            );
        }
    }
}

#[test]
fn changing_parallel_edge_multiplicity_leaves_the_whole_tree_identical() {
    for seed in 0..24 {
        let mut rng = Rng::new(seed + 1_000);
        let recipe = Recipe {
            family: family_of(seed),
            node_count: [12, 80, 600][usize::try_from(seed % 3).expect("a residue fits in usize")],
            parallel: false,
            detached: 5,
        };
        let edges = random_graph(&mut rng, recipe);
        let expected = PostDominatorTree::compute(&edges.graph());

        let sprinkled = with_extra_copies(&mut rng, &edges, 3);
        assert_eq!(
            PostDominatorTree::compute(&sprinkled.graph()),
            expected,
            "{recipe:?} seed {seed}"
        );
        let tripled = edges.repeated(3);
        assert_eq!(
            PostDominatorTree::compute(&tripled.graph()),
            expected,
            "{recipe:?} seed {seed}"
        );
    }
}

#[test]
fn a_reused_scratch_computes_what_a_fresh_one_does() {
    let mut scratch = PostDominatorScratch::new();
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
            PostDominatorTree::compute_in(&mut scratch, &graph),
            PostDominatorTree::compute(&graph),
            "{recipe:?} seed {seed}"
        );
    }
}

#[test]
fn edits_outside_an_exit_free_sink_component_leave_its_cut_unchanged() {
    // The component is three blocks entered only through `s1`, with latches
    // `s2` and `s3`. Everything else is free to change, including whether the
    // entry reaches the component at all.
    let mut reached = 0;
    for seed in 0..60 {
        let mut rng = Rng::new(seed + 7_000);
        let outside = 12 + rng.below(20);
        let (s1, s2, s3) = (node(outside), node(outside + 1), node(outside + 2));
        let source = node(rng.below(outside));
        let fixed = [(s1, s2), (s2, s3), (s3, s1), (s2, s1), (source, s1)];
        for _ in 0..8 {
            let mut edges: Vec<(u32, u32)> = (0..3 * outside)
                .map(|_| (node(rng.below(outside)), node(rng.below(outside))))
                .collect();
            edges.extend(fixed);
            rng.shuffle(&mut edges);
            let post = PostDominatorTree::compute(&EdgeList::new(outside + 3, edges).graph());

            let cut: Vec<u32> = post
                .cut_roots()
                .iter()
                .copied()
                .filter(|&block| block >= s1)
                .collect();
            if !post.contains(s1) {
                assert!(cut.is_empty(), "seed {seed}");
                continue;
            }
            reached += 1;
            assert_eq!(cut, [s2, s3], "seed {seed}");
            assert_eq!(post.ipdom(s1), Some(PostNode::Node(s2)), "seed {seed}");
            assert_eq!(post.ipdom(s2), Some(PostNode::Exit), "seed {seed}");
            assert_eq!(post.ipdom(s3), Some(PostNode::Exit), "seed {seed}");
            assert!(![s1, s2, s3].iter().any(|&block| post.reaches_exit(block)));
        }
    }
    assert!(
        reached >= 100,
        "only {reached} of 480 graphs reach the component"
    );
}
