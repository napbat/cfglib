//! `DisjointSet` against a plain relabeling partition, for both union strategies.

use std::collections::BTreeSet;

use cfglib::DisjointSet;

use crate::Lcg;

/// The obvious partition: every element carries its class label, and a union relabels a class.
struct Relabeling {
    label: Vec<usize>,
}

impl Relabeling {
    fn new(len: usize) -> Self {
        Self {
            label: (0..len).collect(),
        }
    }

    fn union(&mut self, left: usize, right: usize) -> bool {
        let (keep, drop) = (self.label[left], self.label[right]);
        if keep == drop {
            return false;
        }
        for label in &mut self.label {
            if *label == drop {
                *label = keep;
            }
        }
        true
    }

    fn same(&self, left: usize, right: usize) -> bool {
        self.label[left] == self.label[right]
    }

    fn smallest_member(&self, element: usize) -> usize {
        (0..self.label.len())
            .find(|&candidate| self.same(candidate, element))
            .expect("an element belongs to its own class")
    }

    fn class_count(&self) -> usize {
        let mut labels = self.label.clone();
        labels.sort_unstable();
        labels.dedup();
        labels.len()
    }
}

fn random_pairs(seed: u64, len: usize, count: usize) -> Vec<(usize, usize)> {
    let mut rng = Lcg(seed);
    (0..count)
        .map(|_| (rng.below(len), rng.below(len)))
        .collect()
}

#[test]
fn rank_balanced_unions_partition_elements_like_a_relabeling_oracle() {
    for seed in 1..=10 {
        let len = 40;
        let mut sets = DisjointSet::new(len);
        let mut oracle = Relabeling::new(len);
        for (left, right) in random_pairs(seed, len, 35) {
            assert_eq!(
                sets.union(left, right),
                oracle.union(left, right),
                "seed {seed}: union({left}, {right})"
            );
        }
        for left in 0..len {
            for right in 0..len {
                assert_eq!(
                    sets.find(left) == sets.find(right),
                    oracle.same(left, right),
                    "seed {seed}"
                );
            }
        }
        let representatives: BTreeSet<usize> = (0..len).map(|element| sets.find(element)).collect();
        assert_eq!(
            representatives.len(),
            oracle.class_count(),
            "seed {seed}: one representative per class"
        );
    }
}

#[test]
fn union_toward_min_keeps_the_smallest_member_as_every_representative() {
    for seed in 1..=10 {
        let len = 40;
        let mut sets = DisjointSet::new(len);
        let mut oracle = Relabeling::new(len);
        for (left, right) in random_pairs(seed + 50, len, 35) {
            assert_eq!(
                sets.union_toward_min(left, right),
                oracle.union(left, right)
            );
        }
        for element in 0..len {
            assert_eq!(
                sets.find(element),
                oracle.smallest_member(element),
                "seed {seed}, n{element}"
            );
        }
    }
}

#[test]
fn the_minimum_representatives_do_not_depend_on_the_order_of_the_unions() {
    for seed in 1..=6 {
        let len = 30;
        let pairs = random_pairs(seed + 90, len, 28);
        let mut forward = DisjointSet::new(len);
        let mut backward = DisjointSet::new(len);
        for &(left, right) in &pairs {
            forward.union_toward_min(left, right);
        }
        for &(left, right) in pairs.iter().rev() {
            backward.union_toward_min(right, left);
        }
        for element in 0..len {
            assert_eq!(
                forward.find(element),
                backward.find(element),
                "seed {seed}, n{element}"
            );
        }
    }
}

#[test]
fn a_hand_derived_merge_sequence_names_its_representatives() {
    let mut sets = DisjointSet::new(4);
    assert!(sets.union_toward_min(3, 1));
    assert!(sets.union_toward_min(1, 2));
    assert!(!sets.union_toward_min(2, 3), "already one class");
    assert_eq!(sets.find(3), 1);
    assert_eq!(sets.find(2), 1);
    assert_eq!(sets.find(0), 0);
    assert!(sets.union_toward_min(0, 3));
    assert_eq!((sets.find(1), sets.find(2), sets.find(3)), (0, 0, 0));
}

#[test]
fn a_union_reports_whether_it_merged_two_classes_and_never_merges_an_element_with_itself() {
    let mut sets = DisjointSet::new(3);
    assert!(!sets.union(1, 1));
    assert!(sets.union(0, 1));
    assert!(!sets.union(1, 0));
    assert!(sets.union(1, 2));
    assert!(!sets.union(0, 2));
    assert!(!sets.union_toward_min(0, 2));
}

#[test]
fn a_read_only_root_query_agrees_with_find_without_compressing() {
    let len = 200;
    let mut sets = DisjointSet::new(len);
    // A chain, so representatives are many hops away.
    for element in (1..len).rev() {
        sets.union_toward_min(element - 1, element);
    }
    let frozen = sets.clone();
    let before: Vec<usize> = (0..len).map(|element| frozen.root(element)).collect();
    let after: Vec<usize> = (0..len).map(|element| sets.find(element)).collect();
    assert_eq!(before, after);
    assert!(after.iter().all(|&root| root == 0));
    assert_eq!(frozen.root(len - 1), 0);
}

#[test]
fn pushed_elements_join_as_singletons_and_can_be_merged_later() {
    let mut sets = DisjointSet::default();
    assert!(sets.is_empty());
    assert_eq!(sets.len(), 0);
    let first = sets.push();
    let second = sets.push();
    let third = sets.push();
    assert_eq!((first, second, third), (0, 1, 2));
    assert_eq!(sets.len(), 3);
    assert!(!sets.is_empty());
    assert_ne!(sets.find(first), sets.find(second));

    sets.union_toward_min(third, first);
    let fourth = sets.push();
    assert_eq!(fourth, 3);
    assert_eq!(sets.find(third), first);
    assert_eq!(sets.find(fourth), fourth, "a new element starts alone");
    assert_ne!(sets.find(second), sets.find(first));
}

#[test]
fn a_long_chain_of_unions_is_resolved_without_recursion() {
    let len = 20_000;
    let mut sets = DisjointSet::new(len);
    for element in 1..len {
        assert!(sets.union(element - 1, element));
    }
    let root = sets.find(0);
    assert_eq!(sets.find(len - 1), root);
    assert_eq!(sets.root(len / 2), root);
}

#[test]
fn a_clone_is_independent_of_the_forest_it_came_from() {
    let original = DisjointSet::new(4);
    let mut copy = original.clone();
    copy.union(0, 1);
    let mut untouched = original;
    assert_ne!(untouched.find(0), untouched.find(1));
    assert_eq!(copy.find(0), copy.find(1));
    assert_eq!((untouched.len(), copy.len()), (4, 4));
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn finding_an_element_outside_the_forest_panics() {
    let mut sets = DisjointSet::new(3);
    let _ = sets.find(3);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn asking_the_root_of_an_element_outside_the_forest_panics() {
    let sets = DisjointSet::new(3);
    let _ = sets.root(5);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn merging_with_an_element_outside_the_forest_panics() {
    let mut sets = DisjointSet::new(3);
    sets.union_toward_min(0, 3);
}
