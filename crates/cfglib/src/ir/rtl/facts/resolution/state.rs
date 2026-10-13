//! The constants that selection resolution tracks.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::ir::rtl::{Bits, ConstBits};

/// The roots with a known constant value at one program point.
///
/// A root that the state does not list has an unknown value. The entries are
/// in ascending order of root index. Each entry holds the words of the root,
/// with bits past the root width zero.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct ConstState {
    /// The known roots with their values.
    entries: Vec<(usize, Box<[u64]>)>,
}

impl ConstState {
    /// Returns the value of root `root`, if it is known.
    pub(super) fn get(&self, root: usize) -> Option<&[u64]> {
        self.entries
            .binary_search_by_key(&root, |&(index, _)| index)
            .ok()
            .map(|position| &*self.entries[position].1)
    }

    /// Sets the value of root `root`.
    pub(super) fn set(&mut self, root: usize, value: Box<[u64]>) {
        match self
            .entries
            .binary_search_by_key(&root, |&(index, _)| index)
        {
            Ok(position) => self.entries[position].1 = value,
            Err(position) => self.entries.insert(position, (root, value)),
        }
    }

    /// Forgets the value of root `root`.
    pub(super) fn remove(&mut self, root: usize) {
        if let Ok(position) = self
            .entries
            .binary_search_by_key(&root, |&(index, _)| index)
        {
            self.entries.remove(position);
        }
    }

    /// Forgets every value.
    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }

    /// Keeps the values that `other` also holds.
    ///
    /// Returns whether the call forgot a value.
    pub(super) fn intersect(&mut self, other: &Self) -> bool {
        let before = self.entries.len();
        self.entries
            .retain(|(root, value)| other.get(*root) == Some(&**value));
        self.entries.len() != before
    }
}

/// The constant values of the values of one record.
///
/// A record names its values with handles that are dense across the whole
/// function, so the table is indexed by handle and cleared through a list of
/// the handles that it set.
#[derive(Debug)]
pub(super) struct ValueConsts {
    /// For each value handle, where its words start in `pool`.
    slots: Vec<Option<usize>>,
    /// The handles that `slots` holds a value for.
    touched: Vec<usize>,
    /// The words of every constant value of the record.
    pool: Vec<u64>,
}

impl ValueConsts {
    /// Creates an empty table for value handles below `bound`.
    pub(super) fn new(bound: usize) -> Self {
        Self {
            slots: alloc::vec![None; bound],
            touched: Vec::new(),
            pool: Vec::new(),
        }
    }

    /// Forgets the constants of the previous record.
    pub(super) fn clear(&mut self) {
        for &handle in &self.touched {
            self.slots[handle] = None;
        }
        self.touched.clear();
        self.pool.clear();
    }

    /// Returns whether value `handle` is a known constant.
    pub(super) fn contains(&self, handle: usize) -> bool {
        self.slots[handle].is_some()
    }

    /// Returns the constant of value `handle`, which has type `width`.
    pub(super) fn get(&self, handle: usize, width: Bits) -> Option<ConstBits<'_>> {
        let start = self.slots[handle]?;
        Some(ConstBits {
            words: &self.pool[start..start + width.words()],
            width,
        })
    }

    /// Makes value `handle` a constant of type `width` and returns its words,
    /// which are zero, for the caller to fill.
    pub(super) fn insert(&mut self, handle: usize, width: Bits) -> &mut [u64] {
        let start = self.pool.len();
        self.pool.resize(start + width.words(), 0);
        if self.slots[handle].replace(start).is_none() {
            self.touched.push(handle);
        }
        &mut self.pool[start..]
    }
}
