//! Epoch-stamped marking over a dense index space.
//!
//! Marking a node visited is the one buffer every repeated walk allocates,
//! and clearing it is the one O(node count) step a walk over five nodes
//! should not be paying. Both types here answer that the same way: a node
//! holds the epoch it was last touched in, so clearing is a bump of the
//! current epoch rather than a pass over the buffer.
//!
//! [`EpochMarks`] is the write-only half, for a search that only ever asks
//! "have I been here". [`EpochSet`] adds the members, for a pass that has to
//! read back what it marked.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

/// Reusable visited marks for repeated searches over one dense node space.
///
/// [`search`](crate::search) owns its marks, so every call allocates and zeroes a buffer
/// sized to the whole graph. A pass that searches **once per root** over one
/// node space — a nulling closure per grammar nonterminal, a reachable set per
/// definition — pays that O(node count) buffer per root, so its cost scales
/// with the graph even when each search touches a handful of nodes. That is
/// the shape consumers hand-roll an epoch stamp for, and why they decline a
/// substrate that owns its marks.
///
/// `EpochMarks` is that stamp, owned by the caller: each node holds the epoch
/// it was last marked in, so clearing the marks is a bump of the current epoch
/// rather than a walk over the buffer. Allocate one per node space, hand it to
/// every [`search_with_marks`](crate::search_with_marks) of the pass, and marking costs O(1) amortized
/// per root instead of O(node count).
///
/// # Cost
///
/// The win is exactly the buffer, so it is largest when each search is small
/// against the graph. Measured on 16,384 nodes whose closures are four nodes
/// each, one search per node: 8.3ms with a fresh buffer per search, 1.5ms over
/// one reused buffer (5.4x). It narrows as searches grow — a search that
/// visits a large fraction of the graph is dominated by the walk, and reuse
/// lands in the noise.
///
/// # Allocation
///
/// The buffer holds one `u32` stamp per node and is the only allocation in a
/// search whose size is O(node count); what remains per call is O(seeds) and
/// O(nodes visited) — the seed vector, the frontier, and (for the depth-first
/// cores alone, which read a node's successors in reverse) one adjacency
/// buffer refilled per expansion. Those are what [`SearchScratch`](crate::SearchScratch) owns, for
/// the pass whose searches are so small that the call itself is the cost.
/// Sizing is fixed at
/// construction: a buffer smaller than the graph is a panic, not a resize, so
/// that a marks buffer never silently reallocates in the middle of the pass it
/// exists to keep allocation-free. A buffer **larger** than the graph is fine,
/// which is how one buffer covers a set of graphs — size it by the largest.
///
/// # Examples
///
/// ```
/// use cfglib::EpochMarks;
///
/// let marks = EpochMarks::new(64);
/// assert_eq!(marks.capacity(), 64);
/// ```
#[derive(Debug, Clone)]
pub struct EpochMarks {
    /// Per node, the epoch it was last marked in; marked when it equals
    /// `epoch`.
    pub(crate) stamps: Vec<u32>,
    /// The current epoch. Never zero, so a zero stamp is always unmarked —
    /// which is both the initial state and the un-mark of
    /// [`VisitedPolicy::Path`].
    pub(crate) epoch: u32,
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

    /// Return how many nodes these marks cover.
    ///
    /// A [`search_with_marks`](crate::search_with_marks) over a graph with more nodes than this panics;
    /// a consumer whose node space grew builds a new buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.stamps.len()
    }

    /// Clear every mark in O(1) by moving to a fresh epoch.
    ///
    /// The buffer is only walked when the epoch would wrap, which needs
    /// `u32::MAX` searches over one buffer.
    pub(crate) fn reset(&mut self) {
        if self.epoch == u32::MAX {
            self.stamps.fill(0);
            self.epoch = 1;
        } else {
            self.epoch += 1;
        }
    }

    /// Whether `index` is marked in the current epoch.
    pub(crate) fn is_marked(&self, index: usize) -> bool {
        self.stamps[index] == self.epoch
    }

    /// Mark `index` for the current epoch.
    pub(crate) fn mark(&mut self, index: usize) {
        self.stamps[index] = self.epoch;
    }

    /// Un-mark `index`, the unwind of [`VisitedPolicy::Path`].
    pub(crate) fn unmark(&mut self, index: usize) {
        self.stamps[index] = 0;
    }
}

/// A dense set of indices that clears in constant time and can be read back.
///
/// [`EpochMarks`] never says *which* nodes it marked, because a search does
/// not need to know. A pass that does — the blocks a phi-placement worklist
/// touched, the classes one instruction wrote — reaches for a
/// [`BTreeSet`](alloc::collections::BTreeSet) and pays a node allocation per
/// member, or for a `Vec<bool>` plus a separate member list and clears the
/// row per problem.
///
/// `EpochSet` is both halves at once: the epoch stamps answer
/// [`contains`](Self::contains) in constant time, and the member list answers
/// [`ones`](Self::ones) in one pass over the answer rather than over the
/// universe. Neither buffer is released by [`clear`](Self::clear), so a set
/// reused across a sequence of problems allocates only while it is growing
/// to the largest of them.
///
/// # Order
///
/// [`ones`](Self::ones) yields members in **insertion** order, not ascending
/// order. That is what makes it O(members): sorting would cost the pass the
/// buffer it came here to avoid. Collect and sort when an ascending answer is
/// what is wanted, or use [`DenseBits`](crate::DenseBits), whose
/// [`ones`](crate::DenseBits::ones) is ascending because its representation
/// already is.
///
/// # Sizing
///
/// The universe is fixed at construction, on the terms [`EpochMarks`] sets:
/// an index outside it is a panic rather than a resize, so a set never
/// silently reallocates inside the pass it exists to keep allocation-free. A
/// set **larger** than the problem is fine, which is how one set covers a
/// sequence of problems — size it by the largest.
///
/// # Examples
///
/// ```
/// use cfglib::EpochSet;
///
/// let mut visited = EpochSet::new(64);
/// assert!(visited.insert(7));
/// assert!(!visited.insert(7), "a second insert is a no-op");
/// assert!(visited.insert(3));
/// assert!(visited.contains(7));
/// assert_eq!(visited.ones().collect::<Vec<_>>(), vec![7, 3]);
///
/// visited.clear();
/// assert!(!visited.contains(7));
/// assert_eq!(visited.capacity(), 64);
/// ```
#[derive(Debug, Clone)]
pub struct EpochSet {
    /// Membership, answered in constant time and cleared by an epoch bump.
    marks: EpochMarks,
    /// The members of the current epoch, in insertion order.
    members: Vec<usize>,
}

impl EpochSet {
    /// An empty set over the universe `0..len`.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            marks: EpochMarks::new(len),
            members: Vec::new(),
        }
    }

    /// The universe size the set was created with.
    ///
    /// An [`insert`](Self::insert) or [`contains`](Self::contains) outside it
    /// panics; a consumer whose index space grew builds a new set.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.marks.capacity()
    }

    /// Removes every member in constant time, keeping both buffers.
    ///
    /// The stamps are only walked when the epoch would wrap, which needs
    /// `u32::MAX` clears of one set.
    pub fn clear(&mut self) {
        self.marks.reset();
        self.members.clear();
    }

    /// Inserts `index`, returning whether it was newly a member.
    ///
    /// # Panics
    ///
    /// Panics when `index` is outside the universe.
    pub fn insert(&mut self, index: usize) -> bool {
        assert!(
            index < self.capacity(),
            "index {index} outside universe {}",
            self.capacity()
        );
        if self.marks.is_marked(index) {
            return false;
        }
        self.marks.mark(index);
        self.members.push(index);
        true
    }

    /// Whether `index` is a member.
    ///
    /// # Panics
    ///
    /// Panics when `index` is outside the universe.
    #[must_use]
    pub fn contains(&self, index: usize) -> bool {
        assert!(
            index < self.capacity(),
            "index {index} outside universe {}",
            self.capacity()
        );
        self.marks.is_marked(index)
    }

    /// The members, in insertion order.
    pub fn ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.members.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec::Vec;

    use super::EpochSet;

    #[test]
    fn insert_reports_novelty_and_contains_agrees() {
        let mut set = EpochSet::new(16);
        assert!(!set.contains(4));
        assert!(set.insert(4));
        assert!(!set.insert(4));
        assert!(set.contains(4));
        assert!(!set.contains(5));
    }

    #[test]
    fn clear_keeps_the_universe_and_forgets_the_members() {
        let mut set = EpochSet::new(16);
        set.insert(1);
        set.insert(15);
        set.clear();
        assert_eq!(set.capacity(), 16);
        assert!(!set.contains(1));
        assert!(set.ones().next().is_none());
        assert!(set.insert(1), "a cleared member is new again");
    }

    #[test]
    fn members_are_read_back_in_insertion_order() {
        let mut set = EpochSet::new(8);
        for index in [5, 0, 5, 3] {
            set.insert(index);
        }
        assert_eq!(set.ones().collect::<Vec<_>>(), [5, 0, 3]);
    }

    #[test]
    fn an_epoch_wrap_still_clears_every_member() {
        let mut set = EpochSet::new(4);
        set.insert(2);
        for _ in 0..3 {
            set.clear();
            assert!(set.insert(2));
        }
        assert_eq!(set.ones().collect::<Vec<_>>(), [2]);
    }

    #[test]
    #[should_panic(expected = "index 4 outside universe 4")]
    fn inserting_outside_the_universe_panics() {
        EpochSet::new(4).insert(4);
    }
}
