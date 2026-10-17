//! `Fanout` (sparse key to a run of values) and `SortedMap` (sparse key to one value).

use std::collections::{BTreeMap, BTreeSet};
use std::mem::size_of;

use cfglib::{Fanout, SortedMap};

use crate::Lcg;

/// Pairs over forty sparse keys, so runs repeat values, plus the two ends of the key space.
fn sparse_pairs(rng: &mut Lcg, count: usize) -> Vec<(u32, u32)> {
    let mut pairs: Vec<(u32, u32)> = (0..count)
        .map(|_| (rng.below_u32(40) * 997, rng.below_u32(30)))
        .collect();
    pairs.extend([(0, 5), (u32::MAX, u32::MAX), (u32::MAX, 0)]);
    pairs
}

fn deduplicated(pairs: &[(u32, u32)]) -> BTreeMap<u32, BTreeSet<u32>> {
    let mut oracle: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for &(key, value) in pairs {
        oracle.entry(key).or_default().insert(value);
    }
    oracle
}

#[test]
fn a_reverse_index_built_from_unordered_pairs_matches_a_btreemap_oracle() {
    for seed in [3, 11, 2026] {
        let pairs = sparse_pairs(&mut Lcg(seed), 600);
        let oracle = deduplicated(&pairs);
        let recorded = pairs.len();
        let table = Fanout::from_pairs(pairs);

        let expected_values: usize = oracle.values().map(BTreeSet::len).sum();
        assert_eq!(table.len(), oracle.len());
        assert_eq!(table.value_count(), expected_values);
        assert!(!table.is_empty());
        assert!(table.value_count() < recorded, "repeated pairs collapsed");

        assert_eq!(
            table.keys().copied().collect::<Vec<_>>(),
            oracle.keys().copied().collect::<Vec<_>>()
        );
        let rows: Vec<(u32, Vec<u32>)> = table
            .iter()
            .map(|(&key, run)| (key, run.to_vec()))
            .collect();
        let expected_rows: Vec<(u32, Vec<u32>)> = oracle
            .iter()
            .map(|(&key, values)| (key, values.iter().copied().collect()))
            .collect();
        assert_eq!(rows, expected_rows);

        for (key, values) in &oracle {
            assert_eq!(table.get(key), values.iter().copied().collect::<Vec<_>>());
            assert!(table.contains_key(key));
        }
        for absent in [1, 998, u32::MAX - 1, 40 * 997] {
            assert!(!oracle.contains_key(&absent));
            assert!(table.get(&absent).is_empty(), "key {absent}");
            assert!(!table.contains_key(&absent));
        }
    }
}

#[test]
fn the_ends_of_the_key_space_are_ordinary_keys() {
    let table = Fanout::from_pairs(vec![
        (u32::MAX, u64::MAX),
        (0, 0),
        (u32::MAX, 1),
        (0, u64::MAX),
    ]);
    assert_eq!(table.keys().copied().collect::<Vec<_>>(), [0, u32::MAX]);
    assert_eq!(table.get(&0), [0, u64::MAX]);
    assert_eq!(table.get(&u32::MAX), [1, u64::MAX]);
    assert_eq!(table.get(&1).len(), 0);
}

#[test]
fn duplicate_pairs_collapse_and_each_run_ends_up_ascending() {
    let sites = Fanout::from_pairs(vec![
        ("read", 7),
        ("write", 4),
        ("read", 1),
        ("read", 7),
        ("read", 1),
    ]);

    assert_eq!(sites.get(&"read"), [1, 7]);
    assert_eq!(sites.get(&"write"), [4]);
    assert_eq!(sites.get(&"call").len(), 0);
    assert_eq!((sites.len(), sites.value_count()), (2, 3));
}

#[test]
fn an_empty_input_builds_the_canonical_empty_table() {
    let table = Fanout::<u32, u32>::from_pairs(Vec::new());
    assert_eq!(table, Fanout::default());
    assert_eq!(
        table,
        Fanout::<u32, u32>::from_grouped(Vec::<(u32, Vec<u32>)>::new())
    );
    assert!(table.is_empty());
    assert_eq!(
        (table.len(), table.value_count(), table.heap_bytes()),
        (0, 0, 0)
    );
    assert_eq!(table.get(&7).len(), 0);
    assert!(!table.contains_key(&7));
    assert_eq!(table.keys().count(), 0);
    assert_eq!(table.iter().count(), 0);
}

#[test]
fn grouped_input_keeps_each_groups_order_and_repeats_and_concatenates_repeated_keys() {
    let table = Fanout::from_grouped(vec![
        (5_u32, vec![30_u32, 10]),
        (2, vec![]),
        (5, vec![10, 20]),
        (1, vec![9]),
    ]);

    assert_eq!(table.get(&5), [30, 10, 10, 20]);
    assert_eq!(table.get(&1), [9]);
    assert!(
        !table.contains_key(&2),
        "a group with no values contributes no key"
    );
    assert_eq!(table.keys().copied().collect::<Vec<_>>(), [1, 5]);
    assert_eq!((table.len(), table.value_count()), (2, 5));
}

#[test]
fn random_groups_match_a_stable_btreemap_concatenation() {
    for seed in [5, 17] {
        let mut rng = Lcg(seed);
        let groups: Vec<(u32, Vec<u32>)> = (0..60)
            .map(|_| {
                let key = rng.below_u32(8);
                let values = (0..rng.below(4)).map(|_| rng.below_u32(50)).collect();
                (key, values)
            })
            .collect();
        let mut oracle: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (key, values) in &groups {
            if !values.is_empty() {
                oracle.entry(*key).or_default().extend(values);
            }
        }

        let table = Fanout::from_grouped(groups);
        assert_eq!(table.len(), oracle.len());
        for (key, values) in &oracle {
            assert_eq!(table.get(key), values.as_slice(), "key {key}");
        }
        assert_eq!(
            table.value_count(),
            oracle.values().map(Vec::len).sum::<usize>()
        );
    }
}

#[test]
fn a_table_of_owned_keys_is_queried_through_the_borrowed_form() {
    let by_name = Fanout::from_pairs(vec![
        ("render".to_owned(), 12_u32),
        ("render".to_owned(), 40),
        ("parse".to_owned(), 7),
    ]);
    assert_eq!(by_name.get("render"), [12, 40]);
    assert!(by_name.contains_key("parse"));
    assert_eq!(by_name.get("missing").len(), 0);

    let tags = SortedMap::from_pairs(
        vec![("render".to_owned(), 1_u32), ("parse".to_owned(), 2)],
        |slot, value| *slot = value,
    );
    assert_eq!(tags.get("parse"), Some(&2));
    assert!(tags.contains_key("render"));
    assert_eq!(tags.get("missing"), None);
}

#[test]
fn compound_keys_iterate_in_lexicographic_order() {
    let table = Fanout::from_pairs(vec![
        ((2_u32, "b"), 1_u8),
        ((1, "z"), 2),
        ((2, "a"), 3),
        ((1, "z"), 4),
    ]);
    let keys: Vec<_> = table.keys().copied().collect();
    assert_eq!(keys, [(1, "z"), (2, "a"), (2, "b")]);
    assert_eq!(table.get(&(1, "z")), [2, 4]);
}

#[test]
fn heap_bytes_counts_the_three_columns_and_nothing_a_key_or_value_owns() {
    let pairs = sparse_pairs(&mut Lcg(9), 300);
    let table = Fanout::from_pairs(pairs);
    // Keys, one closing run bound beyond one per key, and values.
    let expected = table.len() * size_of::<u32>()
        + (table.len() + 1) * size_of::<u32>()
        + table.value_count() * size_of::<u32>();
    assert_eq!(table.heap_bytes(), expected);

    let short = Fanout::from_pairs(vec![(1_u32, "x".to_owned())]);
    let long = Fanout::from_pairs(vec![(1_u32, "x".repeat(4096))]);
    assert_eq!(
        short.heap_bytes(),
        long.heap_bytes(),
        "a string's own heap is not counted"
    );
}

#[test]
fn clones_are_equal_and_tables_with_different_runs_are_not() {
    let table = Fanout::from_pairs(vec![(1_u32, 10_u32), (1, 11), (4, 40)]);
    assert_eq!(table.clone(), table);
    assert_ne!(table, Fanout::from_pairs(vec![(1_u32, 10_u32), (4, 40)]));
    assert_ne!(
        table,
        Fanout::from_pairs(vec![(1_u32, 10_u32), (1, 11), (5, 40)])
    );
}

#[test]
fn a_sorted_map_agrees_with_repeated_btreemap_inserts_for_last_wins_and_for_a_sum() {
    for seed in [4, 21] {
        let pairs = sparse_pairs(&mut Lcg(seed), 400);
        let mut last: BTreeMap<u32, u32> = BTreeMap::new();
        let mut total: BTreeMap<u32, u64> = BTreeMap::new();
        for &(key, value) in &pairs {
            last.insert(key, value);
            *total.entry(key).or_default() += u64::from(value);
        }

        let last_wins = SortedMap::from_pairs(pairs.clone(), |slot, value| *slot = value);
        let summed = SortedMap::from_pairs(
            pairs
                .iter()
                .map(|&(key, value)| (key, u64::from(value)))
                .collect(),
            |slot, value| *slot += value,
        );

        assert_eq!(last_wins.len(), last.len());
        assert_eq!(summed.len(), total.len());
        assert!(!last_wins.is_empty());
        assert_eq!(
            last_wins
                .iter()
                .map(|(&key, &value)| (key, value))
                .collect::<Vec<_>>(),
            last.iter()
                .map(|(&key, &value)| (key, value))
                .collect::<Vec<_>>()
        );
        for (key, value) in &last {
            assert_eq!(last_wins.get(key), Some(value));
            assert!(last_wins.contains_key(key));
        }
        for (key, value) in &total {
            assert_eq!(summed.get(key), Some(value));
        }
        assert_eq!(last_wins.get(&1), None);
        assert!(!last_wins.contains_key(&(u32::MAX - 1)));
    }
}

#[test]
fn a_sorted_map_merge_sees_a_keys_pairs_in_the_order_they_were_recorded() {
    let joined = SortedMap::from_pairs(
        vec![
            ("a", "x".to_owned()),
            ("b", "y".to_owned()),
            ("a", "z".to_owned()),
            ("a", "w".to_owned()),
        ],
        |slot, value| slot.push_str(&value),
    );
    assert_eq!(joined.get(&"a").map(String::as_str), Some("xzw"));
    assert_eq!(joined.get(&"b").map(String::as_str), Some("y"));
}

#[test]
fn a_sorted_map_costs_one_entry_per_key_and_the_empty_map_nothing() {
    let map = SortedMap::from_pairs(vec![(1_u32, 10_u64), (1, 11), (4, 40)], |slot, value| {
        *slot = value;
    });
    assert_eq!(map.len(), 2);
    assert_eq!(map.heap_bytes(), 2 * size_of::<(u32, u64)>());

    let empty = SortedMap::<u32, u64>::default();
    assert!(empty.is_empty());
    assert_eq!((empty.len(), empty.heap_bytes()), (0, 0));
    assert_eq!(empty.iter().count(), 0);
    assert_eq!(empty.get(&1), None);
    assert_eq!(
        empty,
        SortedMap::from_pairs(Vec::new(), |slot: &mut u64, value| *slot = value)
    );
}
