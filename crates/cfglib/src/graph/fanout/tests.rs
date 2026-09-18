//! Behavior tests for the built-once key tables.

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::graph::store::NodeId;

use super::{DenseFanout, Fanout, SortedMap};

/// Sizes of the columns a `Fanout<u32, u32>` owns, in bytes.
const KEY_BYTES: usize = 4;
const OFFSET_BYTES: usize = 4;
const VALUE_BYTES: usize = 4;

fn collected(fanout: &Fanout<&'static str, u32>) -> Vec<(&'static str, Vec<u32>)> {
    fanout
        .iter()
        .map(|(key, values)| (*key, values.to_vec()))
        .collect()
}

#[test]
fn an_empty_table_holds_nothing_and_allocates_nothing() {
    let empty = Fanout::<u32, u32>::from_pairs(Vec::new());

    assert_eq!(empty, Fanout::default());
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
    assert_eq!(empty.value_count(), 0);
    assert_eq!(empty.heap_bytes(), 0);
    assert!(empty.get(&7).is_empty());
    assert!(!empty.contains_key(&7));
    assert_eq!(empty.keys().count(), 0);
    assert_eq!(empty.iter().count(), 0);
}

#[test]
fn one_key_owns_the_whole_value_column() {
    let single = Fanout::from_pairs(vec![("read", 3_u32)]);

    assert_eq!(single.len(), 1);
    assert!(!single.is_empty());
    assert_eq!(single.value_count(), 1);
    assert_eq!(single.get(&"read"), [3]);
    assert!(single.contains_key(&"read"));
    assert_eq!(collected(&single), [("read", vec![3])]);
}

#[test]
fn unsorted_pairs_become_ascending_runs_and_duplicates_collapse() {
    let sites = Fanout::from_pairs(vec![
        ("write", 9_u32),
        ("read", 7),
        ("write", 2),
        ("read", 1),
        ("read", 7),
    ]);

    assert_eq!(sites.len(), 2);
    assert_eq!(sites.value_count(), 4);
    assert_eq!(
        collected(&sites),
        [("read", vec![1, 7]), ("write", vec![2, 9])]
    );
}

#[test]
fn an_absent_key_reads_as_an_empty_run_anywhere_in_the_order() {
    let sites = Fanout::from_pairs(vec![("b", 1_u32), ("d", 2)]);

    for absent in ["a", "c", "e"] {
        assert!(sites.get(&absent).is_empty(), "{absent} should be absent");
        assert!(!sites.contains_key(&absent));
    }
    assert!(sites.contains_key(&"b"));
    assert!(sites.contains_key(&"d"));
}

#[test]
fn grouped_values_keep_the_order_they_were_grouped_in() {
    let sites = Fanout::from_grouped([
        ("write", vec![9_u32, 2]),
        ("read", vec![7, 1, 7]),
        ("write", vec![4]),
    ]);

    assert_eq!(
        collected(&sites),
        [("read", vec![7, 1, 7]), ("write", vec![9, 2, 4])]
    );
    assert_eq!(sites.value_count(), 6);
}

#[test]
fn a_group_without_values_contributes_no_key() {
    let sites = Fanout::from_grouped([("read", vec![1_u32]), ("write", Vec::new())]);

    assert_eq!(sites.len(), 1);
    assert!(!sites.contains_key(&"write"));
    assert!(sites.get(&"write").is_empty());
    assert_eq!(
        Fanout::<&str, u32>::from_grouped([("write", Vec::new())]),
        Fanout::default()
    );
}

#[test]
fn keys_and_values_need_not_be_copyable() {
    let sites = Fanout::from_grouped([
        ("beta".to_string(), vec!["second".to_string()]),
        ("alpha".to_string(), vec!["first".to_string()]),
    ]);

    assert_eq!(sites.get(&"alpha".to_string()), ["first".to_string()]);
    assert_eq!(
        sites.keys().cloned().collect::<Vec<_>>(),
        ["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn heap_bytes_counts_the_three_columns_and_nothing_else() {
    let sites = Fanout::from_pairs(vec![(1_u32, 10_u32), (1, 11), (4, 40)]);

    // Two keys, three run bounds, three values.
    assert_eq!(
        sites.heap_bytes(),
        2 * KEY_BYTES + 3 * OFFSET_BYTES + 3 * VALUE_BYTES
    );

    // A value's own heap is the value's to report, not the table's.
    let short = Fanout::from_pairs(vec![(1_u32, "x".to_string())]);
    let long = Fanout::from_pairs(vec![(1_u32, "x".repeat(4096))]);
    assert_eq!(short.heap_bytes(), long.heap_bytes());
}

#[test]
fn a_dense_table_keeps_each_keys_arrival_order() {
    let rows = DenseFanout::from_pairs(3, [(2_usize, "c"), (0, "a"), (2, "b"), (0, "d")]);

    assert_eq!(rows.bound(), 3);
    assert_eq!(rows.value_count(), 4);
    assert_eq!(rows.get(0), ["a", "d"]);
    assert!(rows.get(1).is_empty());
    assert_eq!(rows.get(2), ["c", "b"]);
}

#[test]
fn a_dense_table_yields_every_key_of_its_space() {
    let rows = DenseFanout::from_pairs(3, [(1_usize, 5_u32)]);

    let walked: Vec<(usize, Vec<u32>)> = rows
        .iter()
        .map(|(key, values)| (key, values.to_vec()))
        .collect();
    assert_eq!(
        walked,
        [(0, Vec::new()), (1, vec![5]), (2, Vec::new())].to_vec()
    );
}

#[test]
fn a_dense_key_space_survives_having_no_values() {
    let rows = DenseFanout::<u32>::from_pairs(4, []);

    assert_eq!(rows.bound(), 4);
    assert_eq!(rows.value_count(), 0);
    assert!(rows.get(3).is_empty());
    assert_eq!(rows.heap_bytes(), 5 * OFFSET_BYTES);
}

#[test]
fn an_empty_dense_key_space_allocates_nothing() {
    let rows = DenseFanout::<u32>::from_pairs(0, []);

    assert_eq!(rows, DenseFanout::default());
    assert_eq!(rows.bound(), 0);
    assert_eq!(rows.value_count(), 0);
    assert_eq!(rows.heap_bytes(), 0);
    assert_eq!(rows.iter().count(), 0);
}

#[test]
fn dense_heap_bytes_counts_both_columns_and_nothing_else() {
    let rows = DenseFanout::from_pairs(3, [(2_usize, 7_u32), (0, 8), (2, 9)]);

    // Four run bounds over a three-key space, three values.
    assert_eq!(rows.heap_bytes(), 4 * OFFSET_BYTES + 3 * VALUE_BYTES);
}

#[test]
#[should_panic(expected = "dense fan-out key is outside 0..2")]
fn a_dense_key_at_the_bound_is_rejected_while_building() {
    let _ = DenseFanout::from_pairs(2, [(2_usize, "past the end")]);
}

#[test]
#[should_panic(expected = "dense fan-out key is outside the empty key space")]
fn a_dense_pair_in_an_empty_key_space_is_rejected() {
    let _ = DenseFanout::from_pairs(0, [(0_usize, "nowhere to put it")]);
}

#[test]
#[should_panic(expected = "dense fan-out key is outside 0..2")]
fn reading_a_dense_key_past_the_bound_panics() {
    let rows = DenseFanout::from_pairs(2, [(1_usize, "here")]);

    let _ = rows.get(2);
}

#[test]
fn a_tagged_identity_keys_a_dense_table() {
    let definition = NodeId::from_index(0);
    let call = NodeId::from_index(2);
    let references: DenseFanout<&str, NodeId> =
        DenseFanout::from_pairs(3, [(call, "tail"), (call, "hot")]);

    assert_eq!(references.get(call), ["tail", "hot"]);
    assert!(references.get(definition).is_empty());
    assert_eq!(references.iter().count(), 3);
}

#[test]
fn an_empty_sorted_map_holds_nothing_and_allocates_nothing() {
    let empty = SortedMap::<u32, u32>::from_pairs(Vec::new(), |slot, value| *slot = value);

    assert_eq!(empty, SortedMap::default());
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
    assert_eq!(empty.heap_bytes(), 0);
    assert_eq!(empty.get(&1), None);
    assert!(!empty.contains_key(&1));
    assert_eq!(empty.iter().count(), 0);
}

#[test]
fn a_replacing_merge_reproduces_repeated_inserts() {
    let last = SortedMap::from_pairs(
        vec![("b", 1_u32), ("a", 2), ("b", 3), ("a", 4)],
        |slot, value| *slot = value,
    );

    assert_eq!(last.len(), 2);
    assert_eq!(last.get(&"a"), Some(&4));
    assert_eq!(last.get(&"b"), Some(&3));
    assert_eq!(last.get(&"c"), None);
    assert!(last.contains_key(&"a"));
    assert!(!last.contains_key(&"c"));
}

#[test]
fn a_merge_sees_one_keys_pairs_in_the_order_they_were_recorded() {
    let folded = SortedMap::from_pairs(
        vec![
            ("b", "1".to_string()),
            ("a", "2".to_string()),
            ("b", "3".to_string()),
            ("b", "4".to_string()),
        ],
        |slot: &mut String, value| slot.push_str(&value),
    );

    assert_eq!(folded.get(&"b"), Some(&"134".to_string()));
    assert_eq!(
        folded
            .iter()
            .map(|(key, value)| (*key, value.clone()))
            .collect::<Vec<_>>(),
        [("a", "2".to_string()), ("b", "134".to_string())]
    );
}

#[test]
fn sorted_map_heap_bytes_counts_its_entry_column_exactly() {
    let map = SortedMap::from_pairs(vec![(1_u32, 10_u32), (1, 11), (4, 40)], |slot, value| {
        *slot = value;
    });

    assert_eq!(map.len(), 2);
    assert_eq!(map.heap_bytes(), 2 * size_of::<(u32, u32)>());
}
