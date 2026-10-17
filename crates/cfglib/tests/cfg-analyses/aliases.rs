//! Caller-populated alias classes, judged against a brute-force transitive
//! closure of the merges made.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{AliasSets, MemoryAlias};

use crate::random::Rng;

/// Everything reachable from `start` through the merged pairs.
fn component(start: u16, merges: &[(u16, u16)]) -> BTreeSet<u16> {
    let mut seen = BTreeSet::from([start]);
    let mut pending = vec![start];
    while let Some(current) = pending.pop() {
        for &(left, right) in merges {
            for (here, there) in [(left, right), (right, left)] {
                if here == current && seen.insert(there) {
                    pending.push(there);
                }
            }
        }
    }
    seen
}

#[test]
fn aliasing_is_the_transitive_closure_of_the_merges_and_nothing_else() {
    for seed in 0..80 {
        let mut rng = Rng::new(seed);
        let mut sets = AliasSets::<u16>::new();
        let mut merges = Vec::new();
        let mut registered = BTreeSet::new();
        for _ in 0..rng.below(14) {
            let (left, right) = (
                u16::try_from(rng.below(10)).unwrap(),
                u16::try_from(rng.below(10)).unwrap(),
            );
            if rng.chance(25) {
                sets.insert(left);
                registered.insert(left);
            } else {
                sets.merge(left, right);
                merges.push((left, right));
                registered.extend([left, right]);
            }
        }

        for left in 0..12_u16 {
            for right in 0..12_u16 {
                assert_eq!(
                    sets.may_alias(&left, &right),
                    component(left, &merges).contains(&right),
                    "seed {seed}: {left} vs {right}"
                );
            }
        }
        let classes: BTreeSet<BTreeSet<u16>> = registered
            .iter()
            .map(|&variable| component(variable, &merges))
            .collect();
        assert_eq!(sets.set_count(), classes.len(), "seed {seed}");

        let mut representative_of: BTreeMap<BTreeSet<u16>, u16> = BTreeMap::new();
        for variable in 0..12_u16 {
            match sets.alias_set(&variable) {
                None => assert!(!registered.contains(&variable), "seed {seed}"),
                Some(&representative) => {
                    let class = component(variable, &merges);
                    assert!(
                        class.contains(&representative),
                        "seed {seed}: the representative is in the class"
                    );
                    assert!(registered.contains(&variable), "seed {seed}");
                    let agreed = *representative_of.entry(class).or_insert(representative);
                    assert_eq!(
                        agreed, representative,
                        "seed {seed}: one representative per class"
                    );
                }
            }
        }
    }
}

#[test]
fn equal_locations_alias_without_being_registered_and_strangers_do_not() {
    let sets = AliasSets::<u16>::new();

    assert!(sets.may_alias(&7, &7));
    assert!(!sets.may_alias(&7, &8));
    assert_eq!(sets.alias_set(&7), None);
    assert_eq!(sets.set_count(), 0);
}

#[test]
fn merging_registers_both_sides_and_registering_alone_makes_a_singleton() {
    let mut sets = AliasSets::<u16>::new();
    sets.insert(1);
    assert_eq!(sets.set_count(), 1);
    assert_eq!(sets.alias_set(&1), Some(&1));

    sets.merge(2, 3);
    assert_eq!(sets.set_count(), 2);
    sets.merge(3, 2);
    sets.merge(2, 2);
    assert_eq!(sets.set_count(), 2, "re-merging a class changes nothing");
    assert_eq!(sets.alias_set(&2), sets.alias_set(&3));
    assert!(!sets.may_alias(&1, &2));

    sets.merge(1, 3);
    assert_eq!(sets.set_count(), 1);
    assert!(sets.may_alias(&1, &2), "joined through 3");
}

fn aliases_through_the_trait<A: MemoryAlias<u16>>(oracle: &A, left: u16, right: u16) -> bool {
    oracle.may_alias(&left, &right)
}

#[test]
fn alias_sets_serve_as_a_memory_alias_oracle() {
    let mut sets = AliasSets::<u16>::default();
    sets.merge(10, 11);
    sets.insert(12);

    assert!(aliases_through_the_trait(&sets, 10, 11));
    assert!(aliases_through_the_trait(&sets, 12, 12));
    assert!(!aliases_through_the_trait(&sets, 11, 12));
}
