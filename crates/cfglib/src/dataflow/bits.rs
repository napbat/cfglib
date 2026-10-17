//! Dense bit-set lattice elements.
//!
//! Set-of-dense-indices facts (visible files, reachable definitions, owned
//! slots) keep re-implementing the same `Vec<bool>` row with a hand-written
//! changed-tracking union. [`DenseBits`] is that row packed 64 indices per
//! word: `Clone + PartialEq` so it drops into any solver's `Fact`, with
//! [`union_with`](DenseBits::union_with) as the join and its change report
//! driving fixpoint convergence checks.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

/// A fixed-universe set of dense indices, packed 64 per word.
///
/// The universe size is set at construction; every operation stays within
/// it. Use as a dataflow fact with union as the join:
///
/// ```rust
/// use cfglib::DenseBits;
///
/// let mut visible = DenseBits::new(130);
/// assert!(visible.insert(0));
/// assert!(visible.insert(129));
/// let mut merged = DenseBits::new(130);
/// assert!(merged.union_with(&visible));
/// assert!(!merged.union_with(&visible), "a second union changes nothing");
/// assert_eq!(merged.ones().collect::<Vec<_>>(), vec![0, 129]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DenseBits {
    words: Vec<u64>,
    len: usize,
}

impl DenseBits {
    /// Creates the empty set over the universe `0..len`.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }

    /// The universe size the set was created with.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the universe is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether `index` is in the set.
    ///
    /// # Panics
    ///
    /// Panics when `index` is outside the universe.
    #[must_use]
    pub fn get(&self, index: usize) -> bool {
        let (word, bit) = self.slot(index);
        self.words[word] & bit != 0
    }

    /// Inserts `index`, returning whether it was newly set.
    ///
    /// # Panics
    ///
    /// Panics when `index` is outside the universe.
    pub fn insert(&mut self, index: usize) -> bool {
        let (word, bit) = self.slot(index);
        let newly = self.words[word] & bit == 0;
        self.words[word] |= bit;
        newly
    }

    /// Removes `index`, returning whether it had been set.
    ///
    /// # Panics
    ///
    /// Panics when `index` is outside the universe.
    pub fn remove(&mut self, index: usize) -> bool {
        let (word, bit) = self.slot(index);
        let was_set = self.words[word] & bit != 0;
        self.words[word] &= !bit;
        was_set
    }

    /// The word holding `index` and the mask of its bit within that word.
    fn slot(&self, index: usize) -> (usize, u64) {
        assert!(
            index < self.len,
            "index {index} outside universe {}",
            self.len
        );
        bit_slot(index)
    }

    /// Unions `other` into this set, returning whether anything changed.
    ///
    /// This is the lattice join: a fixpoint transfer can report convergence
    /// directly from the return value.
    ///
    /// # Panics
    ///
    /// Panics when the universes differ.
    pub fn union_with(&mut self, other: &Self) -> bool {
        assert_eq!(self.len, other.len, "unioned sets must share one universe");
        let mut changed = false;
        for (word, &other_word) in self.words.iter_mut().zip(&other.words) {
            let merged = *word | other_word;
            changed |= merged != *word;
            *word = merged;
        }
        changed
    }

    /// Removes every index, keeping the universe and the buffer.
    ///
    /// This is a fill of the word array, so it costs O(universe / 64) and no
    /// allocation — which is what lets one row be reused across the many
    /// small problems a whole-codebase pass solves instead of being rebuilt
    /// per problem.
    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    /// The packed words, 64 indices each, least-significant bit first.
    ///
    /// Exposed so a consumer can run its own word-parallel operation over a
    /// row — a population count against a mask, a difference, a three-way
    /// merge — without unpacking it one index at a time. Word `w` holds
    /// indices `64 * w .. 64 * w + 64`; the last word's bits above the
    /// universe are always zero.
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// The number of indices in the set.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// The set's indices in ascending order.
    ///
    /// One step per *set* index, not one per index in the universe: an empty
    /// word is skipped whole.
    pub fn ones(&self) -> impl Iterator<Item = usize> + '_ {
        ones_of(&self.words)
    }

    /// The indices in both sets, in ascending order.
    ///
    /// The two rows are intersected 64 indices at a time, so a pair of large
    /// sparse sets costs one `and` per word rather than one probe per index.
    ///
    /// # Panics
    ///
    /// Panics when the universes differ.
    pub fn intersection<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = usize> + 'a {
        assert_eq!(
            self.len, other.len,
            "intersected sets must share one universe"
        );
        WordOnes::new(
            self.words
                .iter()
                .zip(&other.words)
                .map(|(&left, &right)| left & right),
        )
    }
}

/// The word that holds `index` and the mask of its bit within that word.
///
/// This is the packing every bit row in the crate shares: 64 indices per
/// word, least-significant bit first. A caller that keeps its rows in one
/// flat `u64` buffer instead of in [`DenseBits`] reads it from here.
pub(super) const fn bit_slot(index: usize) -> (usize, u64) {
    (index / 64, 1 << (index % 64))
}

/// The set indices of the packed `words`, in ascending order.
pub(super) fn ones_of(words: &[u64]) -> impl Iterator<Item = usize> + '_ {
    WordOnes::new(words.iter().copied())
}

/// The set indices of a word sequence, in ascending order.
///
/// One iterator for every row-shaped answer: the words may be a set's own, or
/// any combination of two rows computed a word at a time. Each step clears
/// the lowest set bit of the word in hand, so the work is proportional to the
/// answer rather than to the universe.
struct WordOnes<I: Iterator<Item = u64>> {
    words: core::iter::Enumerate<I>,
    /// Set bits of the word in hand that have not been yielded yet.
    remaining: u64,
    /// The index of the word in hand's bit zero.
    base: usize,
}

impl<I: Iterator<Item = u64>> WordOnes<I> {
    fn new(words: I) -> Self {
        Self {
            words: words.enumerate(),
            remaining: 0,
            base: 0,
        }
    }
}

impl<I: Iterator<Item = u64>> Iterator for WordOnes<I> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if self.remaining != 0 {
                let bit = self.remaining.trailing_zeros() as usize;
                self.remaining &= self.remaining - 1;
                return Some(self.base + bit);
            }
            let (position, word) = self.words.next()?;
            self.remaining = word;
            self.base = position * 64;
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec::Vec;

    use super::DenseBits;

    #[test]
    fn insert_get_and_count_agree() {
        let mut bits = DenseBits::new(70);
        assert!(!bits.get(63));
        assert!(bits.insert(63));
        assert!(!bits.insert(63), "a second insert is a no-op");
        assert!(bits.insert(64));
        assert!(bits.get(63));
        assert!(bits.get(64));
        assert!(!bits.get(0));
        assert_eq!(bits.count_ones(), 2);
        assert_eq!(bits.ones().collect::<Vec<_>>(), [63, 64]);
    }

    #[test]
    fn union_reports_change_exactly_when_bits_arrive() {
        let mut left = DenseBits::new(10);
        left.insert(1);
        let mut right = DenseBits::new(10);
        right.insert(1);
        right.insert(9);
        assert!(left.union_with(&right));
        assert!(!left.union_with(&right));
        assert_eq!(left.ones().collect::<Vec<_>>(), [1, 9]);
    }

    #[test]
    fn remove_clears_one_index_and_reports_whether_it_was_set() {
        let mut bits = DenseBits::new(130);
        bits.insert(64);
        bits.insert(65);
        assert!(bits.remove(64));
        assert!(!bits.remove(64), "a second remove is a no-op");
        assert!(!bits.get(64));
        assert!(bits.get(65), "the neighbouring bit is untouched");
        assert_eq!(bits.ones().collect::<Vec<_>>(), [65]);
    }

    #[test]
    fn clear_empties_the_set_and_keeps_the_universe() {
        let mut bits = DenseBits::new(130);
        bits.insert(0);
        bits.insert(129);
        bits.clear();
        assert_eq!(bits.len(), 130);
        assert_eq!(bits.count_ones(), 0);
        assert!(bits.ones().next().is_none());
        assert!(bits.insert(129), "a cleared index is newly set again");
    }

    #[test]
    fn words_expose_the_packed_representation() {
        let mut bits = DenseBits::new(130);
        bits.insert(0);
        bits.insert(64);
        bits.insert(129);
        assert_eq!(bits.words(), [1, 1, 2]);
    }

    #[test]
    fn intersection_yields_the_common_indices_in_order() {
        let mut left = DenseBits::new(200);
        let mut right = DenseBits::new(200);
        for index in [1, 63, 64, 130, 199] {
            left.insert(index);
        }
        for index in [0, 63, 65, 130] {
            right.insert(index);
        }
        assert_eq!(left.intersection(&right).collect::<Vec<_>>(), [63, 130]);
        assert_eq!(
            left.intersection(&right).collect::<Vec<_>>(),
            right.intersection(&left).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn a_disjoint_intersection_is_empty() {
        let mut left = DenseBits::new(70);
        let mut right = DenseBits::new(70);
        left.insert(3);
        right.insert(69);
        assert!(left.intersection(&right).next().is_none());
    }

    #[test]
    #[should_panic(expected = "intersected sets must share one universe")]
    fn intersecting_different_universes_panics() {
        let left = DenseBits::new(70);
        let right = DenseBits::new(71);
        let _ = left.intersection(&right).count();
    }

    #[test]
    fn equality_is_set_equality() {
        let mut left = DenseBits::new(5);
        let mut right = DenseBits::new(5);
        left.insert(2);
        assert_ne!(left, right);
        right.insert(2);
        assert_eq!(left, right);
    }

    #[test]
    fn the_last_index_of_a_universe_lands_in_the_last_word_with_no_bits_above() {
        for len in [1_usize, 63, 64, 65, 128, 129] {
            let mut bits = DenseBits::new(len);
            assert!(bits.insert(len - 1));
            assert_eq!(bits.words().len(), len.div_ceil(64), "len {len}");
            assert_eq!(bits.count_ones(), 1, "len {len}");
            assert_eq!(bits.ones().collect::<Vec<_>>(), [len - 1], "len {len}");
            let used = len % 64;
            if used != 0 {
                assert_eq!(bits.words()[len / 64] >> used, 0, "len {len}");
            }
            assert!(bits.remove(len - 1));
            assert!(bits.words().iter().all(|&word| word == 0), "len {len}");
        }
        let empty = DenseBits::new(0);
        assert_eq!(empty.words().len(), 0);
        assert!(empty.ones().next().is_none());
    }

    #[test]
    #[should_panic(expected = "outside universe")]
    fn reading_outside_the_universe_panics() {
        let _ = DenseBits::new(65).get(65);
    }

    #[test]
    #[should_panic(expected = "outside universe")]
    fn inserting_outside_the_universe_panics() {
        let _ = DenseBits::new(64).insert(64);
    }

    #[test]
    #[should_panic(expected = "outside universe")]
    fn removing_outside_the_universe_panics() {
        let _ = DenseBits::new(3).remove(3);
    }

    #[test]
    #[should_panic(expected = "unioned sets must share one universe")]
    fn unioning_different_universes_panics() {
        let _ = DenseBits::new(70).union_with(&DenseBits::new(71));
    }
}
