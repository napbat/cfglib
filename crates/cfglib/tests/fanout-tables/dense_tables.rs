//! `DenseFanout`: a dense `0..bound` key space to runs of values, built in arrival order.

use std::mem::size_of;

use cfglib::{DenseFanout, Graph, NodeId};

use crate::Lcg;

/// A graph with parallel edges, self loops, removed edges and removed nodes.
fn tombstoned_graph(seed: u64) -> Graph<u32, ()> {
    let mut rng = Lcg(seed);
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..60).map(|label| graph.add_node(label)).collect();
    for _ in 0..200 {
        let (source, target) = (rng.below(60), rng.below(60));
        graph.add_edge(nodes[source], nodes[target], ());
    }
    graph.add_edge(nodes[3], nodes[3], ());
    graph.add_edge(nodes[4], nodes[5], ());
    graph.add_edge(nodes[4], nodes[5], ());
    for _ in 0..5 {
        graph.remove_node(nodes[rng.below(60)]);
    }
    for edge in graph.edge_ids().collect::<Vec<_>>() {
        if rng.below(10) == 0 {
            graph.remove_edge(edge);
        }
    }
    graph
}

#[test]
fn a_reverse_index_keyed_by_node_identity_equals_the_stores_predecessor_lists() {
    for seed in [2, 19, 404] {
        let graph = tombstoned_graph(seed);
        assert!(
            graph.node_count() < graph.node_bound(),
            "the scenario removed nodes"
        );

        let incoming: DenseFanout<NodeId, NodeId> = DenseFanout::from_pairs(
            graph.node_bound(),
            graph.edge_ids().map(|edge| {
                let record = graph.edge(edge);
                (record.target(), record.source())
            }),
        );
        let outgoing: DenseFanout<NodeId, NodeId> = DenseFanout::from_pairs(
            graph.node_bound(),
            graph.edge_ids().map(|edge| {
                let record = graph.edge(edge);
                (record.source(), record.target())
            }),
        );

        assert_eq!(incoming.bound(), graph.node_bound());
        assert_eq!(incoming.value_count(), graph.edge_count());
        assert_eq!(outgoing.value_count(), graph.edge_count());
        for node in graph.node_ids() {
            assert_eq!(
                incoming.get(node),
                graph.predecessors(node).collect::<Vec<_>>()
            );
            assert_eq!(
                outgoing.get(node),
                graph.successors(node).collect::<Vec<_>>()
            );
        }
        for slot in 0..graph.node_bound() {
            let node = NodeId::from_index(slot);
            if !graph.contains_node(node) {
                assert!(incoming.get(node).is_empty(), "removed n{slot} has no run");
            }
        }

        let keys: Vec<NodeId> = incoming.iter().map(|(key, _)| key).collect();
        let every_slot: Vec<NodeId> = (0..graph.node_bound()).map(NodeId::from_index).collect();
        assert_eq!(
            keys, every_slot,
            "iteration yields every key, empty runs included"
        );
        assert_eq!(
            incoming.heap_bytes(),
            (graph.node_bound() + 1) * size_of::<u32>() + graph.edge_count() * size_of::<NodeId>()
        );
    }
}

#[test]
fn pairs_keep_their_arrival_order_inside_each_run_against_a_vec_oracle() {
    for (seed, bound) in [(1, 25_usize), (6, 64), (8, 65)] {
        let mut rng = Lcg(seed);
        let pairs: Vec<(usize, u32)> = (0..400)
            .map(|_| (rng.below(bound), rng.below_u32(1000)))
            .chain([(0, 7), (bound - 1, 8)])
            .collect();
        let mut oracle: Vec<Vec<u32>> = vec![Vec::new(); bound];
        for &(key, value) in &pairs {
            oracle[key].push(value);
        }

        let rows = DenseFanout::from_pairs(bound, pairs);
        assert_eq!(rows.bound(), bound);
        assert_eq!(
            rows.value_count(),
            oracle.iter().map(Vec::len).sum::<usize>()
        );
        for (key, expected) in oracle.iter().enumerate() {
            assert_eq!(rows.get(key), expected.as_slice(), "key {key}");
        }
        let iterated: Vec<(usize, Vec<u32>)> =
            rows.iter().map(|(key, run)| (key, run.to_vec())).collect();
        let expected: Vec<(usize, Vec<u32>)> = oracle.into_iter().enumerate().collect();
        assert_eq!(iterated, expected);
    }
}

#[test]
fn a_key_with_no_pairs_still_owns_an_empty_run() {
    let rows = DenseFanout::from_pairs(5, [(4_usize, "d"), (0, "a")]);
    assert_eq!(rows.get(0), ["a"]);
    for key in 1..4 {
        assert!(rows.get(key).is_empty(), "key {key}");
    }
    assert_eq!(rows.get(4), ["d"]);
    assert_eq!(rows.iter().filter(|(_, run)| run.is_empty()).count(), 3);
}

#[test]
fn grouped_input_equals_the_same_pairs_flattened_in_key_order() {
    for seed in [3, 12] {
        let mut rng = Lcg(seed);
        let groups: Vec<Vec<u32>> = (0..70)
            .map(|_| (0..rng.below(4)).map(|_| rng.below_u32(500)).collect())
            .collect();
        let flattened: Vec<(usize, u32)> = groups
            .iter()
            .enumerate()
            .flat_map(|(key, run)| run.iter().map(move |&value| (key, value)))
            .collect();

        let grouped = DenseFanout::<u32>::from_grouped(groups.clone());
        let counted = DenseFanout::from_pairs(groups.len(), flattened);
        assert_eq!(grouped, counted);
        assert_eq!(grouped.bound(), 70);
        for (key, run) in groups.iter().enumerate() {
            assert_eq!(grouped.get(key), run.as_slice());
        }
    }
}

#[test]
fn a_group_of_no_values_keeps_its_key_in_a_dense_table() {
    let rows: DenseFanout<&str> =
        DenseFanout::from_grouped([vec!["a"], Vec::new(), vec!["c", "b"]]);
    assert_eq!(rows.bound(), 3);
    assert_eq!(rows.get(0), ["a"]);
    assert_eq!(rows.get(1).len(), 0);
    assert_eq!(rows.get(2), ["c", "b"]);
    assert_eq!(
        rows.iter().map(|(key, _)| key).collect::<Vec<_>>(),
        [0, 1, 2]
    );
}

#[test]
fn the_empty_key_space_is_the_default_table_and_costs_nothing() {
    let empty = DenseFanout::<u32>::default();
    assert_eq!(
        empty,
        DenseFanout::from_pairs(0, Vec::<(usize, u32)>::new())
    );
    assert_eq!(
        empty,
        DenseFanout::<u32>::from_grouped(Vec::<Vec<u32>>::new())
    );
    assert_eq!(
        (empty.bound(), empty.value_count(), empty.heap_bytes()),
        (0, 0, 0)
    );
    assert_eq!(empty.iter().count(), 0);
}

#[test]
fn a_nonempty_key_space_without_values_costs_exactly_its_run_bounds() {
    let rows = DenseFanout::<u32>::from_pairs(3, Vec::<(usize, u32)>::new());
    assert_eq!((rows.bound(), rows.value_count()), (3, 0));
    assert_eq!(
        rows.heap_bytes(),
        4 * size_of::<u32>(),
        "bound + 1 run bounds"
    );
    assert_eq!(rows.iter().count(), 3);
    assert_ne!(
        rows,
        DenseFanout::default(),
        "three empty keys are not no keys"
    );
}

#[test]
fn heap_bytes_counts_the_offsets_and_the_values_and_nothing_a_value_owns() {
    let rows = DenseFanout::from_pairs(3, [(2_usize, 7_u64), (0, 8), (2, 9)]);
    assert_eq!(
        rows.heap_bytes(),
        4 * size_of::<u32>() + 3 * size_of::<u64>()
    );

    let short = DenseFanout::from_pairs(2, [(1_usize, "x".to_owned())]);
    let long = DenseFanout::from_pairs(2, [(1_usize, "x".repeat(4096))]);
    assert_eq!(short.heap_bytes(), long.heap_bytes());
}

#[test]
fn a_single_key_space_holds_one_run_in_arrival_order() {
    let rows = DenseFanout::from_pairs(1, [(0_usize, 'b'), (0, 'a'), (0, 'b')]);
    assert_eq!(rows.bound(), 1);
    assert_eq!(rows.get(0), ['b', 'a', 'b'], "arrival order, repeats kept");
    assert_eq!(rows.iter().count(), 1);
}

#[test]
fn the_last_key_of_the_space_is_addressable_at_every_word_boundary() {
    for bound in [2_usize, 3, 63, 64, 65, 128, 129] {
        let rows = DenseFanout::from_pairs(bound, [(bound - 1, bound), (0, 0)]);
        assert_eq!(rows.get(bound - 1), [bound], "bound {bound}");
        assert_eq!(rows.get(0), [0]);
        assert_eq!(rows.iter().last().map(|(key, _)| key), Some(bound - 1));
    }
}

#[test]
#[should_panic(expected = "outside")]
fn a_pair_keyed_at_the_bound_is_rejected_while_building() {
    let _ = DenseFanout::from_pairs(4, [(4_usize, 1_u32)]);
}

#[test]
#[should_panic(expected = "outside")]
fn a_pair_into_the_empty_key_space_is_rejected_while_building() {
    let _ = DenseFanout::from_pairs(0, [(0_usize, 1_u32)]);
}

#[test]
#[should_panic(expected = "outside")]
fn reading_the_bound_key_panics() {
    let rows = DenseFanout::from_pairs(4, [(3_usize, 1_u32)]);
    let _ = rows.get(4);
}

#[test]
#[should_panic(expected = "outside")]
fn reading_any_key_of_the_empty_space_panics() {
    let rows = DenseFanout::<u32>::default();
    let _ = rows.get(0);
}
