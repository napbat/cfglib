//! The span of a slice that one pool of an instruction stores.

extern crate alloc;

use alloc::vec::Vec;

/// A slice that a pool holds: the capture, the expression arena, and the
/// tree each keep the slices of their entries in flat pools and store a span
/// in the entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Span {
    start: u32,
    len: u32,
}

impl Span {
    /// Appends `items` to `pool` and returns their span.
    ///
    /// A slice of `Copy` items extends the pool with one copy.
    ///
    /// # Panics
    ///
    /// Panics when the pool grows past `u32::MAX` items.
    pub(crate) fn push<T, I>(pool: &mut Vec<T>, items: I) -> Self
    where
        I: IntoIterator,
        Vec<T>: Extend<I::Item>,
    {
        let start = pool.len();
        pool.extend(items);
        let fits = "a pool of one instruction holds at most u32::MAX items";
        Self {
            start: u32::try_from(start).expect(fits),
            len: u32::try_from(pool.len() - start).expect(fits),
        }
    }

    /// Returns the slice of `pool` that the span names.
    ///
    /// # Panics
    ///
    /// Panics when the span is not a span of `pool`.
    pub(crate) fn get<T>(self, pool: &[T]) -> &[T] {
        let start = self.start as usize;
        &pool[start..start + self.len as usize]
    }
}
