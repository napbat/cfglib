//! Consumer scenario: the reverse-index recipe of semctx and symtree.
//!
//! An analysis records `(key, value)` pairs as it runs — which sites reference this symbol, which
//! scopes declare this name, which files define this module — and then builds one read-only table
//! and queries it. `Fanout`, `DenseFanout` and `SortedMap` are that table for sparse ordered keys,
//! for a dense `0..bound` key space, and for one value per key. `DenseBits`, `EpochSet` and
//! `EpochMarks` are the dense sets and visited marks those passes keep beside them.
//!
//! Every table is checked against a `BTreeMap`, `BTreeSet` or `Vec` oracle built in the test from
//! the same pairs.

mod bit_sets;
mod dense_tables;
mod sparse_tables;

/// A linear congruential generator, so every scenario is reproducible without a dependency.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, bound: usize) -> usize {
        let bound = u64::try_from(bound).expect("the bound fits in u64");
        usize::try_from(self.next_u64() % bound).expect("a value below the bound fits in usize")
    }

    fn below_u32(&mut self, bound: u32) -> u32 {
        u32::try_from(self.below(usize::try_from(bound).expect("the bound fits in usize")))
            .expect("a value below a u32 bound fits in u32")
    }
}
