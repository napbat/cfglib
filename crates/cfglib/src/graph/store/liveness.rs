//! A packed liveness bitset over dense slots.
//!
//! One bit per slot answers "was this entity removed?" in constant time and
//! costs 1/64th of the `Vec<bool>` or `Vec<Option<T>>` tombstone the other
//! stores in this crate use. Payloads therefore stay unwrapped: an
//! `Option<T>` per edge pays the niche or a whole discriminant word per entry,
//! and a removed entry's payload stops being readable.

extern crate alloc;

use alloc::vec::Vec;

const BITS_PER_WORD: usize = u64::BITS as usize;

/// A growable bitset whose index space matches a store's dense slot space.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(super) struct LiveSet {
    words: Vec<u64>,
    len: usize,
}

impl LiveSet {
    /// Create an empty set.
    pub(super) const fn new() -> Self {
        Self {
            words: Vec::new(),
            len: 0,
        }
    }

    /// Create an empty set with room for `slots` bits.
    pub(super) fn with_capacity(slots: usize) -> Self {
        Self {
            words: Vec::with_capacity(slots.div_ceil(BITS_PER_WORD)),
            len: 0,
        }
    }

    /// Append one live slot and return its index.
    pub(super) fn push_live(&mut self) -> usize {
        let index = self.len;
        if index % BITS_PER_WORD == 0 {
            self.words.push(0);
        }
        self.len += 1;
        let word = index / BITS_PER_WORD;
        self.words[word] |= 1_u64 << (index % BITS_PER_WORD);
        index
    }

    /// Return whether `index` names a live slot.
    ///
    /// An index outside the slot space is not live.
    pub(super) fn is_live(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        self.words[index / BITS_PER_WORD] & (1_u64 << (index % BITS_PER_WORD)) != 0
    }

    /// Clear `index` and return whether it had been live.
    pub(super) fn clear(&mut self, index: usize) -> bool {
        if !self.is_live(index) {
            return false;
        }
        self.words[index / BITS_PER_WORD] &= !(1_u64 << (index % BITS_PER_WORD));
        true
    }

    /// Replace the contents with `slots` live bits, reusing the allocation.
    pub(super) fn reset_all_live(&mut self, slots: usize) {
        self.words.clear();
        self.words.resize(slots.div_ceil(BITS_PER_WORD), u64::MAX);
        let tail = slots % BITS_PER_WORD;
        if tail != 0 {
            let last = self.words.len() - 1;
            self.words[last] = (1_u64 << tail) - 1;
        }
        self.len = slots;
    }
}
