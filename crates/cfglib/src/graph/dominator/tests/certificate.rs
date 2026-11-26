//! The low-high certificate: what it rejects that the parent property alone
//! accepts, and that no certificate rescues a wrong tree.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::support::{EdgeList, Family, Recipe, Rng, for_each_multigraph, node, random_graph};
use crate::graph::dominator::{CertificateError, DominatorScratch, DominatorTree, LowHighOrder};
use crate::graph::view::{Rooted, RootedView};

/// The tree whose parent table is `parents`, holding exactly the nodes
/// `reachable` marks.
fn tree_of(parents: &[Option<u32>], reachable: &[bool]) -> DominatorTree<u32> {
    DominatorTree::from_forest_parts(parents, reachable)
}

#[test]
fn a_too_shallow_tree_passes_the_parent_property_and_fails_the_certificate() {
    // s -> a -> b, with b hung directly below s although a dominates it.
    let edges = EdgeList::new(3, vec![(0, 1), (1, 2)]);
    let graph = edges.graph();
    let shallow = tree_of(&[None, Some(0), Some(0)], &[true; 3]);
    assert!(shallow.satisfies_parent_property(&graph));
    // The two preorders of the shallow tree are every candidate there is.
    for order in [vec![0, 1, 2], vec![0, 2, 1]] {
        let certificate = LowHighOrder::from_order(order.clone(), 3);
        assert!(
            matches!(
                shallow.verify_certificate(&graph, &certificate),
                Err(CertificateError::LowHigh(_))
            ),
            "{order:?}"
        );
    }
    let (tree, certificate) =
        DominatorTree::compute_certified_in(&mut DominatorScratch::default(), &graph);
    assert_eq!(tree.idom(2), Some(1));
    assert_eq!(tree.verify_certificate(&graph, &certificate), Ok(()));
}

/// A preorder of `tree` from `root` that visits children in an order `rng`
/// picks.
fn random_preorder(tree: &DominatorTree<u32>, root: u32, rng: &mut Rng) -> Vec<u32> {
    let mut order = Vec::new();
    let mut stack = vec![root];
    while let Some(member) = stack.pop() {
        order.push(member);
        let mut children: Vec<u32> = tree.children(member).collect();
        rng.shuffle(&mut children);
        stack.extend(children);
    }
    order
}

/// Every way to move one reachable node of `tree` below another parent that
/// keeps the tree a tree and changes the node's parent.
fn reparentings(tree: &DominatorTree<u32>) -> Vec<(u32, u32)> {
    let mut moves = Vec::new();
    for &member in tree.preorder() {
        let Some(parent) = tree.idom(member) else {
            continue;
        };
        for &target in tree.preorder() {
            if target != parent && !tree.dominates(member, target) {
                moves.push((member, target));
            }
        }
    }
    moves
}

/// The tree with `member` moved below `target`.
fn moved(
    tree: &DominatorTree<u32>,
    bound: usize,
    (member, target): (u32, u32),
) -> DominatorTree<u32> {
    let reachable: Vec<bool> = (0..node(bound))
        .map(|index| tree.is_reachable(index))
        .collect();
    let mut parents: Vec<Option<u32>> = (0..node(bound)).map(|index| tree.idom(index)).collect();
    parents[member as usize] = Some(target);
    tree_of(&parents, &reachable)
}

/// Check that a moved tree is rejected with the certificate of the true
/// tree, with its own ascending preorder, and with a random preorder of it.
fn assert_rejected(
    graph: &impl RootedView<NodeId = u32>,
    wrong: &DominatorTree<u32>,
    certificate: &LowHighOrder<u32>,
    rng: &mut Rng,
) {
    let bound = graph.node_bound();
    let candidates = [
        certificate.clone(),
        LowHighOrder::from_order(wrong.preorder().to_vec(), bound),
        LowHighOrder::from_order(random_preorder(wrong, 0, rng), bound),
    ];
    for candidate in &candidates {
        assert!(wrong.verify_certificate(graph, candidate).is_err());
    }
}

#[test]
fn every_single_reparenting_of_a_small_tree_is_rejected() {
    let mut rng = Rng::new(61);
    let mut scratch = DominatorScratch::default();
    for node_count in 1..=4 {
        for_each_multigraph(node_count, 1, |edges| {
            let graph = edges.graph();
            let (tree, certificate) = DominatorTree::compute_certified_in(&mut scratch, &graph);
            for change in reparentings(&tree) {
                let wrong = moved(&tree, node_count, change);
                assert_rejected(&graph, &wrong, &certificate, &mut rng);
            }
        });
    }
}

#[test]
fn a_random_reparenting_of_a_large_tree_is_rejected() {
    let mut scratch = DominatorScratch::default();
    for seed in 0..60 {
        let mut rng = Rng::new(seed);
        let recipe = Recipe {
            family: Family::ALL[usize::try_from(seed % 3).expect("a residue fits")],
            node_count: 200 + rng.below(800),
            parallel: seed % 2 == 0,
            detached: 20,
        };
        let edges = random_graph(&mut rng, recipe);
        let graph = edges.graph();
        let (tree, certificate) = DominatorTree::compute_certified_in(&mut scratch, &graph);
        assert_eq!(tree.verify_certificate(&graph, &certificate), Ok(()));
        let reachable: Vec<u32> = tree.preorder().to_vec();
        for _ in 0..20 {
            let member = reachable[rng.below(reachable.len())];
            let target = reachable[rng.below(reachable.len())];
            if tree.idom(member).is_none_or(|parent| parent == target)
                || tree.dominates(member, target)
            {
                continue;
            }
            let wrong = moved(&tree, edges.node_count, (member, target));
            assert_rejected(&graph, &wrong, &certificate, &mut rng);
        }
    }
}

#[test]
fn a_tree_is_rejected_on_a_graph_one_edge_away_when_their_trees_differ() {
    let mut scratch = DominatorScratch::default();
    for_each_multigraph(3, 1, |edges| {
        let graph = edges.graph();
        let (tree, certificate) = DominatorTree::compute_certified_in(&mut scratch, &graph);
        for from in 0..3 {
            for to in 0..3 {
                let mut more = edges.edges.clone();
                more.push((from, to));
                let other = EdgeList::new(3, more).graph();
                let verdict = tree.verify_certificate(&other, &certificate);
                if DominatorTree::compute(&other) != tree {
                    assert!(verdict.is_err(), "{edges:?} plus {from} -> {to}");
                }
            }
        }
    });
}

#[test]
fn the_wrong_root_a_forest_and_a_foreign_certificate_are_rejected() {
    let edges = EdgeList::new(4, vec![(0, 1), (1, 2), (0, 2)]);
    let graph = edges.graph();
    let mut scratch = DominatorScratch::default();
    let (tree, certificate) = DominatorTree::compute_certified_in(&mut scratch, &graph);
    assert_eq!(
        tree.verify_certificate(&Rooted::new(&graph, 1), &certificate),
        Err(CertificateError::Root(1))
    );
    assert_eq!(
        tree.verify_certificate(&Rooted::new(&graph, 7), &certificate),
        Err(CertificateError::Root(7))
    );

    let mut roots = Vec::new();
    let forest = tree
        .complete_forest_in(&mut scratch, &mut roots, &graph)
        .expect("node 3 is unreachable");
    assert_eq!(
        forest.verify_certificate(&graph, &certificate),
        Err(CertificateError::Root(0))
    );

    let smaller = EdgeList::new(3, vec![(0, 1), (1, 2), (0, 2)]).graph();
    let (_, foreign) = DominatorTree::compute_certified_in(&mut scratch, &smaller);
    assert_eq!(
        tree.verify_certificate(&graph, &foreign),
        Err(CertificateError::NodeBound {
            graph: 4,
            tree: 4,
            certificate: 3
        })
    );
    let missing = LowHighOrder::from_order(vec![0, 1], 4);
    assert_eq!(
        tree.verify_certificate(&graph, &missing),
        Err(CertificateError::Preorder(2))
    );
    let unreachable = LowHighOrder::from_order(vec![0, 1, 2, 3], 4);
    assert_eq!(
        tree.verify_certificate(&graph, &unreachable),
        Err(CertificateError::Preorder(3))
    );
}

#[test]
fn the_certificate_follows_successor_order_and_both_orders_are_valid() {
    // A diamond whose arms are listed in both orders.
    let one = EdgeList::new(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let other = EdgeList::new(4, vec![(0, 2), (0, 1), (2, 3), (1, 3)]);
    let mut scratch = DominatorScratch::default();
    let (tree, first) = DominatorTree::compute_certified_in(&mut scratch, &one.graph());
    let (same, second) = DominatorTree::compute_certified_in(&mut scratch, &other.graph());
    assert_eq!(tree, same);
    assert_eq!(first.order()[0], 0);
    assert_eq!(first.position(0), Some(0));
    assert_eq!(first.position(9), None);
    for certificate in [&first, &second] {
        assert_eq!(tree.verify_certificate(&one.graph(), certificate), Ok(()));
        assert_eq!(tree.verify_certificate(&other.graph(), certificate), Ok(()));
    }
}

#[test]
fn every_root_of_every_small_graph_is_certified() {
    let mut scratch = DominatorScratch::default();
    for_each_multigraph(3, 2, |edges| {
        let graph = edges.graph();
        for root in 0..node(edges.node_count) {
            let (tree, certificate) =
                DominatorTree::compute_certified_in(&mut scratch, &Rooted::new(&graph, root));
            assert_eq!(
                tree.verify_certificate(&Rooted::new(&graph, root), &certificate),
                Ok(()),
                "root {root} of {edges:?}"
            );
            assert_eq!(certificate.order().len(), tree.preorder().len());
        }
    });
}
