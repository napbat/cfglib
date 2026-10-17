//! `DenseBits`, `EpochSet` and `EpochMarks`: the dense sets and visited marks kept beside a table.

use std::collections::{BTreeSet, HashSet};
use std::ops::ControlFlow;

use cfglib::{
    DenseBits, EpochMarks, EpochSet, Graph, NodeId, SearchConfig, SearchOrder, TraversalDirection,
    Visit, reachable, search_with_marks,
};

use crate::Lcg;

/// Universes on both sides of every word boundary, and the empty one.
const UNIVERSES: [usize; 10] = [0, 1, 2, 63, 64, 65, 127, 128, 129, 200];

fn random_members(rng: &mut Lcg, universe: usize, draws: usize) -> BTreeSet<usize> {
    if universe == 0 {
        return BTreeSet::new();
    }
    (0..draws).map(|_| rng.below(universe)).collect()
}

fn bits_of(universe: usize, members: &BTreeSet<usize>) -> DenseBits {
    let mut bits = DenseBits::new(universe);
    for &member in members {
        bits.insert(member);
    }
    bits
}

#[test]
fn dense_bits_follow_a_vec_of_bool_oracle_across_word_boundaries() {
    for universe in UNIVERSES {
        let mut rng = Lcg(universe as u64 + 1);
        let mut bits = DenseBits::new(universe);
        let mut oracle = vec![false; universe];
        assert_eq!(bits.len(), universe);
        assert_eq!(bits.is_empty(), universe == 0);

        if universe > 0 {
            for _ in 0..universe * 2 {
                let index = rng.below(universe);
                let newly = !oracle[index];
                assert_eq!(
                    bits.insert(index),
                    newly,
                    "universe {universe}, index {index}"
                );
                oracle[index] = true;
            }
        }

        for (index, &expected) in oracle.iter().enumerate() {
            assert_eq!(
                bits.get(index),
                expected,
                "universe {universe}, index {index}"
            );
        }
        let members: Vec<usize> = oracle
            .iter()
            .enumerate()
            .filter(|&(_, &member)| member)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(bits.ones().collect::<Vec<_>>(), members, "ones ascend");
        assert_eq!(bits.count_ones(), members.len());
        assert_eq!(bits.len(), universe, "the universe never changes");
    }
}

#[test]
fn the_packed_words_hold_sixty_four_indices_each_least_significant_bit_first() {
    for universe in UNIVERSES {
        let mut rng = Lcg(universe as u64 + 77);
        let members = random_members(&mut rng, universe, universe);
        let bits = bits_of(universe, &members);

        assert_eq!(bits.words().len(), universe.div_ceil(64));
        for (word_index, &word) in bits.words().iter().enumerate() {
            for bit in 0..64_usize {
                let index = word_index * 64 + bit;
                let set = (word >> bit) & 1 == 1;
                if index < universe {
                    assert_eq!(
                        set,
                        members.contains(&index),
                        "universe {universe}, index {index}"
                    );
                } else {
                    assert!(
                        !set,
                        "universe {universe}: bit {index} above the universe is set"
                    );
                }
            }
        }
    }
}

#[test]
fn union_and_intersection_match_the_btreeset_algebra() {
    for universe in [1_usize, 64, 65, 130, 257] {
        let mut rng = Lcg(universe as u64 * 3);
        for round in 0..8 {
            let left_members = random_members(&mut rng, universe, universe / 3 + round);
            let right_members = random_members(&mut rng, universe, universe / 2 + round);
            let left = bits_of(universe, &left_members);
            let right = bits_of(universe, &right_members);

            let joined: BTreeSet<usize> = left_members.union(&right_members).copied().collect();
            let mut union = left.clone();
            let changed = union.union_with(&right);
            assert_eq!(
                changed,
                joined != left_members,
                "universe {universe}, round {round}"
            );
            assert_eq!(
                union.ones().collect::<Vec<_>>(),
                joined.iter().copied().collect::<Vec<_>>()
            );
            assert_eq!(union.count_ones(), joined.len());
            assert!(!union.union_with(&right), "a second union changes nothing");
            assert!(
                !union.union_with(&left),
                "the left operand is already inside"
            );

            let common: Vec<usize> = left_members.intersection(&right_members).copied().collect();
            assert_eq!(left.intersection(&right).collect::<Vec<_>>(), common);
            assert_eq!(
                right.intersection(&left).collect::<Vec<_>>(),
                common,
                "symmetric"
            );
            assert_eq!(
                left.intersection(&left).collect::<Vec<_>>(),
                left_members.iter().copied().collect::<Vec<_>>()
            );
            assert_eq!(left.intersection(&DenseBits::new(universe)).count(), 0);
        }
    }
}

#[test]
fn union_with_the_empty_set_changes_nothing_and_into_the_empty_set_changes_it() {
    let mut empty = DenseBits::new(70);
    let mut full = DenseBits::new(70);
    for index in [0, 63, 64, 69] {
        full.insert(index);
    }
    assert!(!full.union_with(&empty));
    assert!(empty.union_with(&full));
    assert_eq!(empty, full);
}

#[test]
fn clearing_keeps_the_universe_and_the_buffer_reusable() {
    let mut bits = DenseBits::new(130);
    for index in [0, 64, 129] {
        bits.insert(index);
    }
    bits.clear();

    assert_eq!(bits.len(), 130);
    assert_eq!(bits.count_ones(), 0);
    assert_eq!(bits.ones().count(), 0);
    assert!(bits.words().iter().all(|&word| word == 0));
    assert!(bits.insert(129), "a cleared index is newly set again");
    assert_eq!(bits, {
        let mut expected = DenseBits::new(130);
        expected.insert(129);
        expected
    });
}

#[test]
fn equal_membership_means_equal_sets_whatever_the_insertion_order() {
    let mut forward = DenseBits::new(100);
    let mut backward = DenseBits::new(100);
    for index in [3, 50, 99] {
        forward.insert(index);
    }
    for index in [99, 50, 3, 50] {
        backward.insert(index);
    }
    let mut other = forward.clone();
    other.insert(4);

    assert_eq!(forward, backward);
    assert_ne!(forward, other);
    let facts: HashSet<DenseBits> = [forward, backward, other].into_iter().collect();
    assert_eq!(facts.len(), 2, "equal rows hash alike");
}

#[test]
#[should_panic(expected = "outside universe")]
fn reading_outside_the_universe_panics() {
    let _ = DenseBits::new(10).get(10);
}

#[test]
#[should_panic(expected = "outside universe")]
fn inserting_outside_the_universe_panics() {
    DenseBits::new(64).insert(64);
}

#[test]
#[should_panic(expected = "share one universe")]
fn unioning_sets_over_different_universes_panics() {
    DenseBits::new(10).union_with(&DenseBits::new(11));
}

#[test]
#[should_panic(expected = "share one universe")]
fn intersecting_sets_over_different_universes_panics() {
    let _ = DenseBits::new(10).intersection(&DenseBits::new(11));
}

#[test]
fn union_with_is_the_join_of_a_reachability_fixpoint_and_its_change_report_ends_the_loop() {
    for seed in [1, 8] {
        let mut rng = Lcg(seed);
        let mut graph = Graph::<(), ()>::new();
        let nodes: Vec<NodeId> = (0..70).map(|_| graph.add_node(())).collect();
        for _ in 0..110 {
            graph.add_edge(nodes[rng.below(70)], nodes[rng.below(70)], ());
        }

        let mut reach: Vec<DenseBits> = (0..70)
            .map(|node| {
                let mut row = DenseBits::new(70);
                row.insert(node);
                row
            })
            .collect();
        let mut rounds = 0;
        loop {
            let mut changed = false;
            for edge in graph.edges() {
                let from_target = reach[edge.target().index()].clone();
                changed |= reach[edge.source().index()].union_with(&from_target);
            }
            rounds += 1;
            if !changed {
                break;
            }
        }
        assert!(
            rounds > 1,
            "the fixpoint took {rounds} rounds, so the change report mattered"
        );

        for &node in &nodes {
            let expected = reachable(&graph, [node], TraversalDirection::Outgoing);
            let actual: Vec<bool> = (0..70)
                .map(|index| reach[node.index()].get(index))
                .collect();
            assert_eq!(
                actual,
                expected,
                "seed {seed}: reach set of n{}",
                node.index()
            );
        }
    }
}

#[test]
fn an_epoch_set_reports_novelty_and_yields_members_in_insertion_order() {
    let mut visited = EpochSet::new(64);
    assert!(visited.insert(7));
    assert!(!visited.insert(7), "a second insert is a no-op");
    assert!(visited.insert(3));
    assert!(visited.insert(60));
    assert!(visited.contains(7) && visited.contains(3) && !visited.contains(4));
    assert_eq!(
        visited.ones().collect::<Vec<_>>(),
        [7, 3, 60],
        "insertion order, not ascending"
    );
    assert_eq!(visited.capacity(), 64);

    visited.clear();
    assert!(!visited.contains(7));
    assert_eq!(visited.ones().count(), 0);
    assert_eq!(visited.capacity(), 64);
    assert!(visited.insert(7), "a cleared member is new again");
}

#[test]
fn one_epoch_set_reused_over_many_problems_matches_a_fresh_oracle_each_time() {
    let universe = 128;
    let mut rng = Lcg(31);
    let mut set = EpochSet::new(universe + 40);
    for round in 0..600 {
        set.clear();
        let draws = rng.below(30);
        let mut order: Vec<usize> = Vec::new();
        let mut oracle = BTreeSet::new();
        for _ in 0..draws {
            let index = rng.below(universe);
            let novel = oracle.insert(index);
            assert_eq!(set.insert(index), novel, "round {round}");
            if novel {
                order.push(index);
            }
        }

        assert_eq!(set.ones().collect::<Vec<_>>(), order, "round {round}");
        for index in 0..universe {
            assert_eq!(
                set.contains(index),
                oracle.contains(&index),
                "round {round}, index {index}"
            );
        }
        assert_eq!(set.capacity(), universe + 40, "reuse never resizes");
    }
}

#[test]
#[should_panic(expected = "outside universe")]
fn inserting_outside_an_epoch_sets_universe_panics() {
    EpochSet::new(4).insert(4);
}

#[test]
#[should_panic(expected = "outside universe")]
fn asking_an_epoch_set_about_an_index_outside_its_universe_panics() {
    let _ = EpochSet::new(4).contains(4);
}

#[test]
fn epoch_marks_report_the_universe_they_were_sized_for() {
    assert_eq!(EpochMarks::new(64).capacity(), 64);
    assert_eq!(
        EpochMarks::default().capacity(),
        0,
        "sized on first use by a consumer that rebuilds it"
    );
    assert_eq!(EpochMarks::new(0).capacity(), 0);
    let marks = EpochMarks::new(9);
    assert_eq!(marks.clone().capacity(), marks.capacity());
}

/// A random graph over `count` nodes.
fn random_graph(rng: &mut Lcg, count: usize, edges: usize) -> Graph<(), ()> {
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..count).map(|_| graph.add_node(())).collect();
    for _ in 0..edges {
        graph.add_edge(nodes[rng.below(count)], nodes[rng.below(count)], ());
    }
    graph
}

fn closure_of(graph: &Graph<(), ()>, root: NodeId, marks: &mut EpochMarks) -> Vec<bool> {
    let config = SearchConfig::new(SearchOrder::BreadthFirst, TraversalDirection::Outgoing);
    let mut closure = vec![false; graph.node_bound()];
    let outcome = search_with_marks(graph, [root], config, marks, |node, _depth| {
        closure[node.index()] = true;
        ControlFlow::<(), Visit>::Continue(Visit::Descend)
    });
    assert_eq!(outcome, None);
    closure
}

#[test]
fn one_marks_buffer_serves_a_closure_per_root_pass_like_a_fresh_buffer_per_root() {
    let mut rng = Lcg(90);
    let graph = random_graph(&mut rng, 70, 120);
    let mut marks = EpochMarks::new(graph.node_bound() + 30);

    for root in graph.node_ids() {
        let closure = closure_of(&graph, root, &mut marks);
        assert_eq!(
            closure,
            reachable(&graph, [root], TraversalDirection::Outgoing),
            "closure of n{}",
            root.index()
        );
    }

    let small = random_graph(&mut rng, 9, 14);
    for root in small.node_ids() {
        assert_eq!(
            closure_of(&small, root, &mut marks),
            reachable(&small, [root], TraversalDirection::Outgoing),
            "a buffer sized for a larger graph covers a smaller one"
        );
    }
}

#[test]
#[should_panic(expected = "visited marks cover")]
fn a_marks_buffer_smaller_than_the_graph_is_a_panic_not_a_resize() {
    let graph = random_graph(&mut Lcg(2), 12, 20);
    let mut marks = EpochMarks::new(graph.node_bound() - 1);
    let _ = closure_of(&graph, graph.node_ids().next().expect("a node"), &mut marks);
}
