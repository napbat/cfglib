//! Epoch-stamped marking over a dense index space, and the sizing policy of
//! every reusable scratch in the crate.
//!
//! Marking a node visited is the one buffer every repeated walk allocates,
//! and clearing it is the one O(node count) step a walk over five nodes
//! should not be paying. [`EpochMarks`] answers that: a node holds the epoch
//! it was last marked in, so clearing is a bump of the current epoch rather
//! than a pass over the buffer.
//!
//! # Sizing
//!
//! Every reusable scratch of the crate — [`SearchScratch`](crate::SearchScratch),
//! [`ShortestPathScratch`](crate::ShortestPathScratch), the dominator and cut
//! scratches, and the marks here — follows one policy: it starts empty
//! ([`Default`]) and grows on entry to fit the graph it is handed, keeping
//! what it holds. A scratch therefore serves a sequence of graphs of any
//! size, allocates only while it grows to the largest of them, and never
//! panics because a graph outgrew it.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

/// Reusable visited marks over one dense node space.
///
/// Each node holds the epoch it was last marked in, so clearing the marks is
/// a bump of the current epoch rather than a walk over the buffer. A pass
/// that walks once per root over one node space — a nulling closure per
/// grammar nonterminal, a reachable set per definition — keeps one buffer and
/// pays O(1) per root to clear it instead of O(node count).
///
/// # Cost
///
/// The win is exactly the buffer, so it is largest when each walk is small
/// against the graph. Measured on 16,384 nodes whose closures are four nodes
/// each, one search per node: 8.3ms with a fresh buffer per search, 1.5ms over
/// one reused buffer (5.4x). It narrows as walks grow — a walk that visits a
/// large fraction of the graph is dominated by the walk, and reuse lands in
/// the noise.
///
/// # Use
///
/// A pass sizes the marks with [`new`](Self::new) or, as a scratch does on
/// entry, with [`grow_to`](Self::grow_to); marks by hand with
/// [`mark`](Self::mark) and [`is_marked`](Self::is_marked); and clears them
/// with [`reset`](Self::reset) between two walks.
///
/// # Examples
///
/// ```
/// use cfglib::EpochMarks;
///
/// let mut marks = EpochMarks::default();
/// marks.grow_to(64);
/// marks.mark(7);
/// assert!(marks.is_marked(7));
/// marks.reset();
/// assert!(!marks.is_marked(7));
/// ```
#[derive(Debug, Clone)]
pub struct EpochMarks {
    /// Per node, the epoch it was last marked in; marked when it equals
    /// `epoch`.
    pub(crate) stamps: Vec<u32>,
    /// The current epoch. Never zero, so a zero stamp is always unmarked —
    /// which is both the initial state and the un-mark of a walk that marks
    /// per path.
    pub(crate) epoch: u32,
}

impl Default for EpochMarks {
    /// Marks covering nothing, for a scratch that grows them on entry.
    fn default() -> Self {
        Self::new(0)
    }
}

impl EpochMarks {
    /// Marks covering `node_count` nodes, with nothing marked.
    #[must_use]
    pub fn new(node_count: usize) -> Self {
        Self {
            stamps: vec![0; node_count],
            epoch: 1,
        }
    }

    /// Clear every mark in O(1) by moving to a fresh epoch.
    ///
    /// The buffer is only walked when the epoch would wrap, which needs
    /// `u32::MAX` resets of one buffer.
    pub fn reset(&mut self) {
        if self.epoch == u32::MAX {
            self.stamps.fill(0);
            self.epoch = 1;
        } else {
            self.epoch += 1;
        }
    }

    /// Grow the buffer to cover `node_count` nodes, keeping the marks.
    ///
    /// A new stamp is zero, which is never the current epoch, so growing
    /// never marks a node. A buffer that already covers `node_count` nodes is
    /// left as it is.
    pub fn grow_to(&mut self, node_count: usize) {
        grow_to(&mut self.stamps, node_count, 0);
    }

    /// Whether `index` is marked in the current epoch.
    ///
    /// # Panics
    ///
    /// Panics when `index` is beyond the nodes the marks cover.
    #[inline]
    #[must_use]
    pub fn is_marked(&self, index: usize) -> bool {
        self.stamps[index] == self.epoch
    }

    /// Mark `index` for the current epoch.
    ///
    /// # Panics
    ///
    /// Panics when `index` is beyond the nodes the marks cover.
    #[inline]
    pub fn mark(&mut self, index: usize) {
        self.stamps[index] = self.epoch;
    }

    /// Un-mark `index`, the unwind of a walk that marks per path.
    #[inline]
    pub(crate) fn unmark(&mut self, index: usize) {
        self.stamps[index] = 0;
    }
}

/// Grow `table` to `len` entries, filling the new ones with `fill` and keeping
/// the rest: the sizing step of every scratch table (see the module docs). A
/// table that already holds `len` entries is left as it is.
pub(crate) fn grow_to<T: Clone>(table: &mut Vec<T>, len: usize, fill: T) {
    if table.len() < len {
        table.resize(len, fill);
    }
}

#[cfg(test)]
mod tests {
    use super::EpochMarks;

    #[test]
    fn marks_clear_on_reset_and_on_an_epoch_wrap() {
        let mut marks = EpochMarks::new(4);
        marks.mark(1);
        assert!(marks.is_marked(1));
        assert!(!marks.is_marked(0));
        marks.reset();
        assert!(!marks.is_marked(1));

        marks.mark(3);
        marks.epoch = u32::MAX;
        marks.mark(2);
        marks.reset();
        assert_eq!(marks.epoch, 1, "a wrap starts again at the first epoch");
        assert!((0..4).all(|index| !marks.is_marked(index)));
    }

    #[test]
    fn growing_keeps_the_marks_and_never_marks_a_new_node() {
        let mut marks = EpochMarks::default();
        marks.grow_to(3);
        marks.mark(2);
        marks.grow_to(6);
        assert!(marks.is_marked(2));
        assert!((3..6).all(|index| !marks.is_marked(index)));
        marks.grow_to(1);
        assert!(marks.is_marked(2), "growing never shrinks");
        assert!(!marks.is_marked(5));
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn marking_beyond_the_marked_nodes_panics() {
        EpochMarks::new(2).mark(2);
    }
}
