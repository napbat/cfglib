//! Rows of items grouped by a dense key, in one flat buffer.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

/// Items grouped by a dense key.
///
/// Two buffers hold the rows, however many keys there are: `starts` marks
/// where each row begins, and `items` holds the rows one after the other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Csr<T> {
    /// Where each row starts in `items`. One longer than the key bound, so a
    /// row is `starts[key]..starts[key + 1]`.
    starts: Vec<usize>,
    /// The items of every row.
    items: Vec<T>,
}

impl<T: Copy> Csr<T> {
    /// Groups the items that `enumerate` emits by key.
    ///
    /// `enumerate` runs twice and emits the same `(key, item)` pairs in the
    /// same order each time: the first run counts the rows, and the second
    /// run places the items. Items of one key keep their emission order.
    ///
    /// # Panics
    ///
    /// Panics when a key is not below `key_bound`.
    pub(super) fn build(key_bound: usize, enumerate: impl Fn(&mut dyn FnMut(usize, T))) -> Self {
        let mut starts = vec![0; key_bound + 1];
        enumerate(&mut |key, _| starts[key + 1] += 1);
        for key in 0..key_bound {
            starts[key + 1] += starts[key];
        }

        let total = starts[key_bound];
        let mut cursors = starts.clone();
        let mut items = Vec::with_capacity(total);
        enumerate(&mut |key, item| {
            if items.is_empty() {
                // Every slot is overwritten below, so the first item is as
                // good a filler as any.
                items.resize(total, item);
            }
            items[cursors[key]] = item;
            cursors[key] += 1;
        });
        Self { starts, items }
    }

    /// Returns the row of `key`.
    ///
    /// # Panics
    ///
    /// Panics when `key` is not below the key bound of the table.
    pub(super) fn row(&self, key: usize) -> &[T] {
        &self.items[self.starts[key]..self.starts[key + 1]]
    }
}
