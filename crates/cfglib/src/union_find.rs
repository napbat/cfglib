//! Disjoint-set forest over dense indices.
//!
//! [`DisjointSet`] backs equivalence-class construction wherever elements
//! already carry dense `usize` identities: phi webs, may-alias location
//! classes, and lane webs all reduce to it. Two union strategies are offered
//! because consumers disagree on what the surviving representative means:
//! [`union`](DisjointSet::union) balances by rank and leaves the
//! representative unspecified, while
//! [`union_toward_min`](DisjointSet::union_toward_min) always keeps the
//! smallest member index so iteration order stays deterministic.
//!
//! ```rust
//! use cfglib::DisjointSet;
//!
//! let mut sets = DisjointSet::new(4);
//! sets.union_toward_min(3, 1);
//! sets.union_toward_min(1, 2);
//! assert_eq!(sets.find(3), 1);
//! assert_eq!(sets.find(0), 0);
//! ```

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

/// Disjoint-set forest (union-find) over `0..len` element indices.
///
/// Every element starts in its own singleton set. [`find`](Self::find) uses
/// path halving, so amortized operations are effectively constant time.
///
/// Use one union strategy per instance: mixing
/// [`union`](Self::union) and [`union_toward_min`](Self::union_toward_min)
/// leaves the minimum-representative guarantee unspecified.
#[derive(Debug, Clone, Default)]
pub struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl DisjointSet {
    /// Creates `len` singleton sets.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    /// Returns the forest to `len` singleton sets, keeping both arrays.
    ///
    /// This is [`new`](Self::new) for a caller that solves one merge problem
    /// per procedure over a whole codebase and does not want two allocations
    /// per procedure for it.
    pub(crate) fn reset(&mut self, len: usize) {
        self.parent.clear();
        self.parent.extend(0..len);
        self.rank.clear();
        self.rank.resize(len, 0);
    }

    /// Appends one new singleton set and returns its element index.
    pub fn push(&mut self) -> usize {
        let index = self.parent.len();
        self.parent.push(index);
        self.rank.push(0);
        index
    }

    /// Number of elements across all sets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Whether the forest holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Returns the representative of `index`'s set, compressing the path.
    ///
    /// # Panics
    ///
    /// Panics when `index` is out of bounds.
    pub fn find(&mut self, mut index: usize) -> usize {
        while self.parent[index] != index {
            self.parent[index] = self.parent[self.parent[index]];
            index = self.parent[index];
        }
        index
    }

    /// Returns the representative of `index`'s set without compressing.
    ///
    /// Use this for read-only queries through shared references; prefer
    /// [`find`](Self::find) when the set is mutable.
    ///
    /// # Panics
    ///
    /// Panics when `index` is out of bounds.
    #[must_use]
    pub fn root(&self, mut index: usize) -> usize {
        while self.parent[index] != index {
            index = self.parent[index];
        }
        index
    }

    /// Merges the sets of `left` and `right`, balancing by rank.
    ///
    /// The surviving representative is unspecified. Returns whether the two
    /// were in different sets.
    ///
    /// # Panics
    ///
    /// Panics when either index is out of bounds.
    pub fn union(&mut self, left: usize, right: usize) -> bool {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return false;
        }
        match self.rank[left_root].cmp(&self.rank[right_root]) {
            Ordering::Less => self.parent[left_root] = right_root,
            Ordering::Greater => self.parent[right_root] = left_root,
            Ordering::Equal => {
                self.parent[right_root] = left_root;
                self.rank[left_root] += 1;
            }
        }
        true
    }

    /// Merges the sets of `left` and `right`, keeping the smaller root.
    ///
    /// When only this strategy is used, every set's representative is its
    /// minimum member index, so representatives are deterministic and stable
    /// under insertion order. Returns whether the two were in different sets.
    ///
    /// # Panics
    ///
    /// Panics when either index is out of bounds.
    pub fn union_toward_min(&mut self, left: usize, right: usize) -> bool {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return false;
        }
        let (root, merged) = if left_root < right_root {
            (left_root, right_root)
        } else {
            (right_root, left_root)
        };
        self.parent[merged] = root;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::DisjointSet;

    #[test]
    fn singletons_are_their_own_representatives() {
        let mut sets = DisjointSet::new(3);
        assert_eq!(sets.len(), 3);
        assert!(!sets.is_empty());
        for index in 0..3 {
            assert_eq!(sets.find(index), index);
            assert_eq!(sets.root(index), index);
        }
    }

    #[test]
    fn union_by_rank_connects_transitively() {
        let mut sets = DisjointSet::new(5);
        assert!(sets.union(0, 1));
        assert!(sets.union(1, 2));
        assert!(!sets.union(2, 0));
        assert_eq!(sets.find(0), sets.find(2));
        assert_ne!(sets.find(0), sets.find(3));
    }

    #[test]
    fn union_toward_min_keeps_the_smallest_member() {
        let mut sets = DisjointSet::new(6);
        assert!(sets.union_toward_min(5, 3));
        assert!(sets.union_toward_min(3, 4));
        assert_eq!(sets.find(5), 3);
        assert!(sets.union_toward_min(4, 1));
        assert_eq!(sets.find(5), 1);
    }

    #[test]
    fn push_grows_the_forest() {
        let mut sets = DisjointSet::default();
        assert!(sets.is_empty());
        let first = sets.push();
        let second = sets.push();
        assert_eq!((first, second), (0, 1));
        sets.union(first, second);
        assert_eq!(sets.find(first), sets.find(second));
    }

    #[test]
    fn root_answers_without_mutation() {
        let mut sets = DisjointSet::new(4);
        sets.union_toward_min(2, 3);
        let read_only = sets.clone();
        assert_eq!(read_only.root(3), 2);
    }
}
