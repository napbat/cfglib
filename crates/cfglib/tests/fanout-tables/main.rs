//! Consumer scenario: the reverse-index recipe of semctx and symtree.
//!
//! An analysis records `(key, value)` pairs as it runs — which sites reference this symbol, which
//! scopes declare this name, which files define this module — and then builds one read-only table
//! and queries it. `Fanout`, `DenseFanout` and `SortedMap` are that table for sparse ordered keys,
//! for a dense `0..bound` key space, and for one value per key. `DenseBits` and `EpochMarks` are
//! the dense sets and visited marks those passes keep beside them.
//!
//! Every table is checked against a `BTreeMap`, `BTreeSet` or `Vec` oracle built in the test from
//! the same pairs.

#[path = "../common/rng.rs"]
pub mod rng;

mod bit_sets;
mod dense_tables;
mod sparse_tables;

use crate::rng::Rng;

/// A value in `0..bound`, as a `u32`.
fn below_u32(rng: &mut Rng, bound: u32) -> u32 {
    u32::try_from(rng.below(usize::try_from(bound).expect("the bound fits in usize")))
        .expect("a value below a u32 bound fits in u32")
}
