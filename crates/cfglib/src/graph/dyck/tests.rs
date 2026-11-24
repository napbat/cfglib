extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::{BidirectedDyck, bidirected_dyck_classes};
use crate::graph::store::Graph;
use crate::test_util::rng::Rng;

/// An edge of a model graph: source, target, and the parenthesis it opens.
type ModelEdge = (usize, usize, Option<u32>);

/// Balanced reachability by saturating `S -> S S | (k S )k | ε` over the
/// bidirected graph: `reach[x][y]` when some path from `x` to `y` spells a
/// Dyck word, each edge walked forward with `(k` or backward with `)k`.
fn cfl_saturation(nodes: usize, edges: &[ModelEdge]) -> Vec<Vec<bool>> {
    let mut reach = vec![vec![false; nodes]; nodes];
    for (node, row) in reach.iter_mut().enumerate() {
        row[node] = true;
    }
    for &(a, b, label) in edges {
        if label.is_none() {
            reach[a][b] = true;
            reach[b][a] = true;
        }
    }
    loop {
        let mut changed = false;
        // (k S )k: open along `a --(k--> b`, balance from `b` to `d`, close
        // along the inverse of `c --(k--> d`.
        for &(a, b, open) in edges {
            for &(c, d, close) in edges {
                if open.is_some() && open == close && reach[b][d] && !reach[a][c] {
                    reach[a][c] = true;
                    changed = true;
                }
            }
        }
        // S S.
        for x in 0..nodes {
            for y in 0..nodes {
                if reach[x][y] {
                    for z in 0..nodes {
                        if reach[y][z] && !reach[x][z] {
                            reach[x][z] = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            return reach;
        }
    }
}

fn random_edges(rng: &mut Rng, nodes: usize) -> Vec<ModelEdge> {
    (0..rng.below(13))
        .map(|_| {
            let label = if rng.chance(20) {
                None
            } else {
                Some(u32::try_from(rng.below(2)).expect("a label below two"))
            };
            (rng.below(nodes), rng.below(nodes), label)
        })
        .collect()
}

fn incremental(nodes: usize, edges: &[ModelEdge]) -> BidirectedDyck {
    let mut dyck = BidirectedDyck::new(nodes);
    for &(from, to, label) in edges {
        match label {
            Some(label) => dyck.insert(from, to, label),
            None => dyck.insert_plain(from, to),
        }
    }
    dyck
}

fn batch(nodes: usize, edges: &[ModelEdge]) -> BidirectedDyck {
    let mut graph = Graph::new();
    let ids: Vec<_> = (0..nodes).map(|_| graph.add_node(())).collect();
    for &(from, to, label) in edges {
        graph.add_edge(ids[from], ids[to], label);
    }
    bidirected_dyck_classes(&graph, |edge| *edge.data())
}

/// Every node's class representative.
fn classes(dyck: &mut BidirectedDyck) -> Vec<usize> {
    (0..dyck.node_bound())
        .map(|node| dyck.class_of(node))
        .collect()
}

fn assert_matches_oracle(dyck: &mut BidirectedDyck, reach: &[Vec<bool>], context: &str) {
    for (a, row) in reach.iter().enumerate() {
        for (b, &reaches) in row.iter().enumerate() {
            assert_eq!(dyck.same_class(a, b), reaches, "{context}: {a} and {b}");
        }
        let minimum = row.iter().position(|&reaches| reaches);
        assert_eq!(Some(dyck.class_of(a)), minimum, "{context}: class of {a}");
    }
}

#[test]
fn classes_match_cfl_saturation_on_random_small_graphs() {
    let mut nested = 0;
    for seed in 0..400 {
        let mut rng = Rng::new(seed);
        let nodes = 1 + rng.below(8);
        let edges = random_edges(&mut rng, nodes);
        let reach = cfl_saturation(nodes, &edges);
        let context = alloc::format!("seed {seed}, edges {edges:?}");
        assert_matches_oracle(&mut incremental(nodes, &edges), &reach, &context);
        assert_matches_oracle(&mut batch(nodes, &edges), &reach, &context);

        // Count graphs whose classes need a matched pair of labeled edges,
        // not only the plain ones.
        let plain_only: Vec<_> = edges
            .iter()
            .filter(|edge| edge.2.is_none())
            .copied()
            .collect();
        if cfl_saturation(nodes, &plain_only) != reach {
            nested += 1;
        }
    }
    assert!(
        nested > 100,
        "only {nested} graphs merge through parentheses"
    );
}

#[test]
fn inserting_edges_one_at_a_time_matches_the_batch_build_after_every_prefix() {
    for seed in 0..100 {
        let mut rng = Rng::new(seed + 1_000);
        let nodes = 1 + rng.below(8);
        let mut edges = random_edges(&mut rng, nodes);
        let expected = classes(&mut batch(nodes, &edges));

        rng.shuffle(&mut edges);
        let mut dyck = BidirectedDyck::new(nodes);
        for (inserted, &(from, to, label)) in edges.iter().enumerate() {
            match label {
                Some(label) => dyck.insert(from, to, label),
                None => dyck.insert_plain(from, to),
            }
            let prefix = cfl_saturation(nodes, &edges[..=inserted]);
            assert_matches_oracle(&mut dyck, &prefix, &alloc::format!("seed {seed}"));
        }
        assert_eq!(classes(&mut dyck), expected, "seed {seed}");
    }
}

#[test]
fn nested_parentheses_merge_from_the_inside_out() {
    // outer (0 left (1 inner )1 right )0 back: `left` and `right` join
    // first, both opening label 1 into `inner`; then `outer` and `back` meet
    // through label 0 into the merged class.
    let (outer, left, inner, right, back) = (0, 1, 2, 3, 4);
    let edges = [
        (outer, left, Some(0)),
        (left, inner, Some(1)),
        (right, inner, Some(1)),
        (back, right, Some(0)),
    ];
    let mut dyck = batch(5, &edges);
    assert_eq!(classes(&mut dyck), [outer, left, inner, left, outer]);

    // In the other order, the outer pair waits until the inner one merges.
    let mut late = BidirectedDyck::new(5);
    late.insert(outer, left, 0);
    late.insert(back, right, 0);
    assert!(!late.same_class(outer, back));
    late.insert(left, inner, 1);
    late.insert(right, inner, 1);
    assert_eq!(classes(&mut late), [outer, left, inner, left, outer]);
}

#[test]
fn different_labels_into_one_node_stay_apart() {
    let mut dyck = BidirectedDyck::new(3);
    dyck.insert(0, 2, 0);
    dyck.insert(1, 2, 1);
    assert_eq!(classes(&mut dyck), [0, 1, 2]);
}

#[test]
fn a_labeled_self_loop_is_a_source_of_its_own_node() {
    // `a --(0--> a` and `b --(0--> a`: the path `b (0 a )0 a` is balanced.
    let mut dyck = BidirectedDyck::new(3);
    dyck.insert(0, 0, 0);
    assert_eq!(classes(&mut dyck), [0, 1, 2]);
    dyck.insert(1, 0, 0);
    assert_eq!(classes(&mut dyck), [0, 0, 2]);
}

#[test]
fn a_removed_node_keeps_a_class_of_its_own() {
    let mut graph = Graph::new();
    let nodes: Vec<_> = (0..4).map(|_| graph.add_node(())).collect();
    graph.add_edge(nodes[0], nodes[1], Some(0));
    graph.add_edge(nodes[2], nodes[1], Some(0));
    graph.add_edge(nodes[3], nodes[1], Some(0));
    graph.remove_node(nodes[2]);

    let mut dyck = bidirected_dyck_classes(&graph, |edge| *edge.data());
    assert_eq!(dyck.node_bound(), 4);
    assert_eq!(classes(&mut dyck), [0, 1, 2, 0]);
    assert_eq!(dyck.class_of(nodes[3].index()), 0);
}

#[test]
fn an_empty_graph_has_no_classes() {
    let graph: Graph<(), Option<u32>> = Graph::new();
    let dyck = bidirected_dyck_classes(&graph, |edge| *edge.data());
    assert_eq!(dyck.node_bound(), 0);
}

#[test]
#[should_panic(expected = "edge endpoint is out of range")]
fn a_labeled_edge_past_the_nodes_panics() {
    BidirectedDyck::new(2).insert(0, 2, 0);
}

#[test]
#[should_panic(expected = "edge endpoint is out of range")]
fn a_plain_edge_past_the_nodes_panics() {
    BidirectedDyck::new(2).insert_plain(2, 0);
}
