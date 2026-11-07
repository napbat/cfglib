//! Dense bit-set lattice elements.
//!
//! Set-of-dense-indices facts (visible files, reachable definitions, owned
//! slots) keep re-implementing the same `Vec<bool>` row with a hand-written
//! changed-tracking union. [`DenseBits`] is that row packed 64 indices per
//! word: `Clone + PartialEq` so it drops into any solver's `Fact`, with
//! [`union_with`](DenseBits::union_with) as the join and its change report
//! driving fixpoint convergence checks. [`DenseRow`] borrows one such row out
//! of an analysis that keeps the rows of every block in one buffer.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

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

    /// The set borrowed as a [`DenseRow`].
    #[must_use]
    pub fn row(&self) -> DenseRow<'_> {
        DenseRow {
            words: &self.words,
            len: self.len,
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
        self.union_with_row(other.row())
    }

    /// Unions the borrowed row `other` into this set, returning whether
    /// anything changed.
    ///
    /// [`union_with`](Self::union_with) for a row that an analysis keeps in a
    /// flat buffer; into a [`clear`](Self::clear)ed set it is a copy that
    /// allocates nothing.
    ///
    /// # Panics
    ///
    /// Panics when the universes differ.
    pub fn union_with_row(&mut self, other: DenseRow<'_>) -> bool {
        assert_eq!(self.len, other.len, "unioned sets must share one universe");
        let mut changed = false;
        for (word, &other_word) in self.words.iter_mut().zip(other.words) {
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

/// A borrowed set of dense indices, in the packing of [`DenseBits`].
///
/// An analysis that keeps the facts of every block in one flat buffer of
/// rows hands out each row as this, so its result holds no allocation for
/// each block and reading a fact allocates nothing. [`DenseBits::row`]
/// borrows an owned set the same way, and
/// [`DenseBits::union_with_row`] copies a row into a reusable set.
///
/// Equality is set equality over the same universe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DenseRow<'a> {
    words: &'a [u64],
    len: usize,
}

impl<'a> DenseRow<'a> {
    /// The row over the universe `0..len` whose packed words are `words`.
    ///
    /// The caller keeps the bits above `len` in the last word clear.
    pub(super) fn from_words(len: usize, words: &'a [u64]) -> Self {
        debug_assert_eq!(words.len(), len.div_ceil(64), "a row of the universe");
        debug_assert!(
            len.is_multiple_of(64) || words.last().is_none_or(|&word| word >> (len % 64) == 0),
            "no bit above the universe"
        );
        Self { words, len }
    }

    /// The universe size of the row.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Whether the universe is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Whether `index` is in the row.
    ///
    /// # Panics
    ///
    /// Panics when `index` is outside the universe.
    #[must_use]
    pub fn get(self, index: usize) -> bool {
        assert!(
            index < self.len,
            "index {index} outside universe {}",
            self.len
        );
        let (word, bit) = bit_slot(index);
        self.words[word] & bit != 0
    }

    /// The packed words, as [`DenseBits::words`] lays them out.
    #[must_use]
    pub const fn words(self) -> &'a [u64] {
        self.words
    }

    /// The number of indices in the row.
    #[must_use]
    pub fn count_ones(self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// The row's indices in ascending order, one step per set index.
    pub fn ones(self) -> impl Iterator<Item = usize> + 'a {
        ones_of(self.words)
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

/// The words of row `row` in a flat buffer of rows `words` long.
pub(super) const fn flat_row(words: usize, row: usize) -> Range<usize> {
    row * words..(row + 1) * words
}

/// The rows of a solve kept as one flat buffer, one row per slot, every row
/// over one universe, so a result holds one allocation however many slots it
/// has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FlatRows {
    /// The universe of every row.
    len: usize,
    /// The number of rows.
    count: usize,
    /// The rows, `len.div_ceil(64)` words each, bits above `len` clear.
    buffer: Vec<u64>,
}

impl FlatRows {
    /// The `count` rows over `0..len` that `buffer` holds back to back.
    pub(super) fn new(len: usize, count: usize, buffer: Vec<u64>) -> Self {
        debug_assert_eq!(buffer.len(), count * len.div_ceil(64), "count rows");
        Self { len, count, buffer }
    }

    /// Row `index`.
    ///
    /// # Panics
    ///
    /// Panics when `index` is not below the row count.
    pub(super) fn row(&self, index: usize) -> DenseRow<'_> {
        assert!(
            index < self.count,
            "row {index} outside {} rows",
            self.count
        );
        DenseRow::from_words(
            self.len,
            &self.buffer[flat_row(self.len.div_ceil(64), index)],
        )
    }
}

/// Sets every index of `range` in the packed `words`.
pub(super) fn set_range(words: &mut [u64], range: Range<usize>) {
    for (word, mask) in range_masks(range) {
        words[word] |= mask;
    }
}

/// Clears every index of `range` in the packed `words`.
pub(super) fn clear_range(words: &mut [u64], range: Range<usize>) {
    for (word, mask) in range_masks(range) {
        words[word] &= !mask;
    }
}

/// Each word that `range` touches, with the mask of the indices of `range` in
/// that word, so a run of indices costs one operation per word.
fn range_masks(range: Range<usize>) -> impl Iterator<Item = (usize, u64)> {
    let Range { start, end } = range;
    let touched = if start < end {
        start / 64..end.div_ceil(64)
    } else {
        0..0
    };
    touched.map(move |word| {
        let low = start.saturating_sub(word * 64);
        let high = (end - word * 64).min(64);
        (word, (u64::MAX >> (64 - (high - low))) << low)
    })
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

    use super::{DenseBits, DenseRow, clear_range, flat_row, set_range};

    /// Every range over a three-word row, the empty ones and those that start
    /// or end on a word boundary included.
    #[test]
    fn a_range_sets_and_clears_exactly_its_indices() {
        let len = 150;
        let mut full = DenseBits::new(len);
        for index in 0..len {
            full.insert(index);
        }
        for start in 0..=len {
            for end in start..=len {
                let mut words = alloc::vec![0; 3];
                set_range(&mut words, start..end);
                let set = DenseRow::from_words(len, &words);
                assert_eq!(
                    set.ones().collect::<Vec<_>>(),
                    (start..end).collect::<Vec<_>>()
                );

                let mut words = full.words().to_vec();
                clear_range(&mut words, start..end);
                let cleared = DenseRow::from_words(len, &words);
                let kept: Vec<usize> = (0..start).chain(end..len).collect();
                assert_eq!(cleared.ones().collect::<Vec<_>>(), kept, "{start}..{end}");
            }
        }
    }

    #[test]
    fn a_flat_row_is_the_words_of_one_row() {
        assert_eq!(flat_row(3, 0), 0..3);
        assert_eq!(flat_row(3, 2), 6..9);
        assert_eq!(flat_row(0, 5), 0..0, "an empty universe has empty rows");
    }

    #[test]
    fn a_row_reads_like_the_set_it_borrows() {
        let mut bits = DenseBits::new(130);
        for index in [0, 63, 64, 129] {
            bits.insert(index);
        }
        let row = bits.row();
        assert_eq!(row.len(), 130);
        assert!(!row.is_empty());
        assert_eq!(row.words(), bits.words());
        assert_eq!(row.count_ones(), 4);
        assert_eq!(row.ones().collect::<Vec<_>>(), [0, 63, 64, 129]);
        assert!(row.get(64));
        assert!(!row.get(65));
        assert_eq!(row, DenseRow::from_words(130, &[1 | (1 << 63), 1, 2]));
    }

    #[test]
    fn a_row_unions_into_a_cleared_set_as_a_copy() {
        let words = [0b1010, 1 << 5];
        let row = DenseRow::from_words(70, &words);
        let mut bits = DenseBits::new(70);
        bits.insert(0);
        bits.clear();
        assert!(bits.union_with_row(row));
        assert_eq!(bits.row(), row);
        assert!(!bits.union_with_row(row), "a second union changes nothing");
    }

    #[test]
    #[should_panic(expected = "outside universe")]
    fn reading_a_row_outside_its_universe_panics() {
        let _ = DenseRow::from_words(65, &[0, 0]).get(65);
    }

    #[test]
    #[should_panic(expected = "unioned sets must share one universe")]
    fn unioning_a_row_of_another_universe_panics() {
        let _ = DenseBits::new(70).union_with_row(DenseRow::from_words(64, &[0]));
    }

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
        assert!(bits.get(65), "the neighboring bit is untouched");
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
