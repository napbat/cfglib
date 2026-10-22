//! Integration tests: the coldness propagation of `ColdnessClasses`.
//!
//! The reference is the definition itself: every reachable node starts at
//! `Normal`, and the bounds (S), (B), and (F) apply one node at a time, in a
//! random order, until no bound lowers a class. The greatest assignment is
//! unique, so every order must reach the classes that the propagation finds.

use cfglib::ColdClass::{Never, Normal, Rare};
use cfglib::{ColdClass, ColdnessClasses, ColdnessGraph};

/// A graph of numbered nodes with node 0 as the entry.
#[derive(Debug, Clone)]
struct Toy {
    seed: Vec<ColdClass>,
    ends: Vec<bool>,
    /// The `(target, cap)` records of each node, in record order.
    records: Vec<Vec<(usize, ColdClass)>>,
    /// The `(source, cap)` records that name each node, from every node.
    incoming: Vec<Vec<(usize, ColdClass)>>,
    /// The nodes that the entry reaches, in the order that `reachable`
    /// yields them.
    order: Vec<usize>,
}

impl Toy {
    /// A graph of `count` nodes, seeded `Normal`, without `ends` or records.
    fn new(count: usize) -> Self {
        Self {
            seed: vec![Normal; count],
            ends: vec![false; count],
            records: vec![Vec::new(); count],
            incoming: Vec::new(),
            order: Vec::new(),
        }
    }

    /// Adds a record from `source` to `target` with the cap `cap`.
    fn record(mut self, source: usize, target: usize, cap: ColdClass) -> Self {
        self.records[source].push((target, cap));
        self
    }

    fn seeded(mut self, node: usize, seed: ColdClass) -> Self {
        self.seed[node] = seed;
        self
    }

    fn ending(mut self, node: usize) -> Self {
        self.ends[node] = true;
        self
    }

    /// Fills `incoming` and `order` from the records: the order is the
    /// depth-first preorder from the entry.
    fn finish(mut self) -> Self {
        let count = self.records.len();
        self.incoming = vec![Vec::new(); count];
        for (source, records) in self.records.iter().enumerate() {
            for &(target, cap) in records {
                self.incoming[target].push((source, cap));
            }
        }
        self.order = reachable(&self.records).into_iter().collect();
        self
    }

    fn compute(&self) -> Vec<ColdClass> {
        let classes = ColdnessClasses::compute(self);
        (0..self.records.len())
            .map(|node| classes.class(node))
            .collect()
    }
}

impl ColdnessGraph for Toy {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.records.len()
    }

    fn entry(&self) -> usize {
        0
    }

    fn reachable(&self) -> impl Iterator<Item = usize> + '_ {
        self.order.iter().copied()
    }

    fn records(&self, node: usize) -> impl Iterator<Item = (usize, ColdClass)> + '_ {
        self.records[node].iter().copied()
    }

    fn incoming(&self, node: usize) -> impl Iterator<Item = (usize, ColdClass)> + '_ {
        self.incoming[node].iter().copied()
    }

    fn seed(&self, node: usize) -> ColdClass {
        self.seed[node]
    }

    fn ends(&self, node: usize) -> bool {
        self.ends[node]
    }
}

/// Returns the nodes that the entry reaches, in depth-first preorder.
fn reachable(records: &[Vec<(usize, ColdClass)>]) -> Vec<usize> {
    let mut seen = vec![false; records.len()];
    let mut order = Vec::new();
    let mut work = vec![0];
    while let Some(node) = work.pop() {
        if core::mem::replace(&mut seen[node], true) {
            continue;
        }
        order.push(node);
        work.extend(records[node].iter().rev().map(|&(target, _)| target));
    }
    order
}

/// A small deterministic generator (xorshift64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % bound as u64).expect("the bound fits usize")
    }

    fn class(&mut self) -> ColdClass {
        [Never, Rare, Normal][self.below(3)]
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for at in (1..items.len()).rev() {
            items.swap(at, self.below(at + 1));
        }
    }
}

/// The hottest of `values`, or `Never` when there is none.
fn hottest(values: impl IntoIterator<Item = ColdClass>) -> ColdClass {
    values.into_iter().max().unwrap_or(Never)
}

/// Applies the bounds one at a time, in an order that `rng` draws, until no
/// class changes.
fn chaotic(toy: &Toy, rng: &mut Rng) -> Vec<ColdClass> {
    let count = toy.records.len();
    let reach = reachable(&toy.records);
    let mut class = vec![Never; count];
    for &node in &reach {
        class[node] = Normal;
    }
    let mut bounds: Vec<(usize, u8)> = reach
        .iter()
        .flat_map(|&node| (0..3).map(move |kind| (node, kind)))
        .collect();
    loop {
        rng.shuffle(&mut bounds);
        let mut changed = false;
        for &(node, kind) in &bounds {
            let limit = match kind {
                0 => toy.seed[node],
                1 if toy.records[node].is_empty() => Normal,
                1 => match hottest(toy.records[node].iter().map(|&(t, cap)| cap.min(class[t]))) {
                    Never if toy.ends[node] => Rare,
                    top => top,
                },
                _ if node == 0 => Normal,
                _ => hottest(
                    (0..count)
                        .filter(|source| reach.contains(source))
                        .flat_map(|source| {
                            toy.records[source]
                                .iter()
                                .filter(move |&&(target, _)| target == node)
                                .map(move |&(_, cap)| (source, cap))
                        })
                        .map(|(source, cap)| cap.min(class[source])),
                ),
            };
            if limit < class[node] {
                class[node] = limit;
                changed = true;
            }
        }
        if !changed {
            return class;
        }
    }
}

/// A random graph of one to six nodes: parallel records, self-loops, every
/// seed and cap, and some nodes with `ends`.
fn random_toy(rng: &mut Rng) -> Toy {
    let count = 1 + rng.below(6);
    let mut toy = Toy::new(count);
    for node in 0..count {
        toy.seed[node] = if rng.below(3) == 0 {
            rng.class()
        } else {
            Normal
        };
        toy.ends[node] = rng.below(3) == 0;
        for _ in 0..rng.below(4) {
            let cap = if rng.below(4) == 0 { Rare } else { Normal };
            toy.records[node].push((rng.below(count), cap));
        }
    }
    toy.finish()
}

#[test]
fn the_classes_are_the_greatest_assignment_under_the_bounds() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for _ in 0..20_000 {
        let toy = random_toy(&mut rng);
        let classes = toy.compute();
        for _ in 0..3 {
            assert_eq!(classes, chaotic(&toy, &mut rng), "{toy:?}");
        }
    }
}

#[test]
fn the_order_of_nodes_and_records_does_not_change_the_classes() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for _ in 0..5_000 {
        let toy = random_toy(&mut rng);
        let expected = toy.compute();
        let mut permuted = toy.clone();
        rng.shuffle(&mut permuted.order);
        for records in &mut permuted.records {
            rng.shuffle(records);
        }
        for incoming in &mut permuted.incoming {
            rng.shuffle(incoming);
        }
        assert_eq!(permuted.compute(), expected, "{toy:?}");
        let mut doubled = permuted.clone();
        for records in &mut doubled.records {
            let copy = records.clone();
            records.extend(copy);
        }
        let doubled = doubled.finish();
        assert_eq!(doubled.compute(), expected, "parallel records: {toy:?}");
    }
}

#[test]
fn the_seed_is_an_upper_bound() {
    let toy = Toy::new(4)
        .record(0, 1, Normal)
        .record(0, 2, Normal)
        .record(0, 3, Normal)
        .seeded(1, Rare)
        .seeded(2, Never)
        .finish();
    assert_eq!(toy.compute(), [Normal, Rare, Never, Normal]);

    // With no hotter record left, the seeds bound the entry through (B).
    let cold = Toy::new(3)
        .record(0, 1, Normal)
        .record(0, 2, Normal)
        .seeded(1, Rare)
        .seeded(2, Never)
        .finish();
    assert_eq!(cold.compute(), [Rare, Rare, Never]);
}

#[test]
fn a_node_whose_records_all_lead_to_never_is_never_unless_it_ends() {
    let silent = Toy::new(2).record(0, 1, Normal).seeded(1, Never).finish();
    assert_eq!(silent.compute(), [Never, Never]);

    let ending = silent.clone().ending(0);
    assert_eq!(ending.compute(), [Rare, Never]);
}

#[test]
fn an_exit_free_loop_stays_normal() {
    let toy = Toy::new(2)
        .record(0, 1, Normal)
        .record(1, 1, Normal)
        .finish();
    assert_eq!(toy.compute(), [Normal, Normal]);
}

#[test]
fn a_loop_whose_only_exit_aborts_stays_normal() {
    let toy = Toy::new(3)
        .record(0, 1, Normal)
        .record(1, 1, Normal)
        .record(1, 2, Normal)
        .seeded(2, Rare)
        .finish();
    assert_eq!(toy.compute(), [Normal, Normal, Rare]);
}

#[test]
fn a_handler_that_normal_flow_also_reaches_stays_normal() {
    // 0 branches to 1 normally and to the handler 2 through a capped record;
    // 1 falls into the handler too.
    let shared = Toy::new(3)
        .record(0, 1, Normal)
        .record(0, 2, Rare)
        .record(1, 2, Normal)
        .finish();
    assert_eq!(shared.compute(), [Normal, Normal, Normal]);

    let only_exceptional = Toy::new(3).record(0, 1, Normal).record(0, 2, Rare).finish();
    assert_eq!(only_exceptional.compute(), [Normal, Normal, Rare]);
}

#[test]
fn the_entry_is_rare_when_every_path_ends_in_a_rare_node() {
    let toy = Toy::new(4)
        .record(0, 1, Normal)
        .record(0, 2, Normal)
        .record(1, 3, Normal)
        .record(2, 3, Normal)
        .seeded(3, Rare)
        .finish();
    assert_eq!(toy.compute(), [Rare, Rare, Rare, Rare]);
}

#[test]
fn successors_of_a_never_node_are_never() {
    let toy = Toy::new(3)
        .record(0, 1, Normal)
        .record(0, 2, Normal)
        .record(1, 2, Normal)
        .seeded(1, Never)
        .finish();
    assert_eq!(toy.compute(), [Normal, Never, Normal]);

    let only_through_never = Toy::new(4)
        .record(0, 1, Normal)
        .record(0, 3, Normal)
        .record(1, 2, Normal)
        .seeded(1, Never)
        .finish();
    assert_eq!(only_through_never.compute(), [Normal, Never, Never, Normal]);
}

#[test]
fn an_irreducible_cycle_supports_itself_unless_a_seed_cools_it() {
    // 0 enters the cycle 1 <-> 2 at both nodes through capped records, and
    // returns through 3. Both nodes of the cycle exit to 3 as well. Each node
    // of the cycle bounds the other only by its own class, so the greatest
    // assignment keeps the cycle `Normal`, as it keeps an exit-free loop.
    let toy = Toy::new(4)
        .record(0, 1, Rare)
        .record(0, 2, Rare)
        .record(0, 3, Normal)
        .record(1, 2, Normal)
        .record(1, 3, Normal)
        .record(2, 1, Normal)
        .record(2, 3, Normal)
        .finish();
    assert_eq!(toy.compute(), [Normal, Normal, Normal, Normal]);

    // When node 2 is seeded `Rare`, node 1 has only `Rare` predecessors
    // left, and (F) lowers it although its exit to 3 is `Normal`.
    let seeded = toy.seeded(2, Rare);
    assert_eq!(seeded.compute(), [Normal, Rare, Rare, Normal]);

    // A normal entry into node 1 keeps it `Normal`.
    let mut entered = seeded.clone();
    entered.records[0][0].1 = Normal;
    let entered = entered.finish();
    assert_eq!(entered.compute(), [Normal, Normal, Rare, Normal]);
}

#[test]
fn records_from_unreached_nodes_are_ignored() {
    // Node 2 is unreachable and names node 1 with a normal record; node 1 is
    // reachable only through a capped record.
    let toy = Toy::new(3).record(0, 1, Rare).record(2, 1, Normal).finish();
    let classes = ColdnessClasses::compute(&toy);
    assert_eq!(classes.class(1), Rare);
    assert_eq!(classes.class(2), Never);
    assert_eq!(classes.class(7), Never, "past the node bound");
}

#[test]
fn a_record_is_as_cold_as_its_ends_and_its_cap() {
    let toy = Toy::new(4)
        .record(0, 1, Normal)
        .record(0, 2, Rare)
        .record(0, 3, Normal)
        .seeded(1, Rare)
        .finish();
    let classes = ColdnessClasses::compute(&toy);
    assert_eq!(toy.compute(), [Normal, Rare, Rare, Normal]);
    assert_eq!(classes.record(0, Normal, 1), Rare, "the target is rare");
    assert_eq!(classes.record(0, Rare, 3), Rare, "the cap is rare");
    assert_eq!(classes.record(0, Normal, 3), Normal);
    assert_eq!(classes.record(5, Normal, 0), Never, "past the node bound");
}

#[test]
fn a_long_chain_falls_without_recursion() {
    let length = 100_000;
    let mut toy = Toy::new(length);
    for node in 1..length {
        toy.records[node - 1].push((node, Normal));
    }
    toy.seed[length - 1] = Never;
    let toy = toy.finish();
    assert!(toy.compute().iter().all(|&class| class == Never));

    let toy = toy.ending(length / 2);
    let classes = toy.compute();
    assert!(classes[..=length / 2].iter().all(|&class| class == Rare));
    assert!(
        classes[length / 2 + 1..]
            .iter()
            .all(|&class| class == Never)
    );
}

#[test]
fn a_hub_with_many_records_reaches_the_definition() {
    // The entry fans out to `width` nodes that all join one hub, and the hub
    // aborts. Every node is `Rare`.
    let width = 50_000;
    let hub = width + 1;
    let mut toy = Toy::new(width + 2);
    for node in 1..=width {
        toy.records[0].push((node, Normal));
        toy.records[node].push((hub, Normal));
    }
    toy.seed[hub] = Rare;
    let toy = toy.finish();
    assert!(toy.compute().iter().all(|&class| class == Rare));
}
