//! Placing the loops a construction found in the forest: the preorder runs of
//! the properly nested loops, the run of blocks of each position, and the
//! nesting rule's parent and depth of every loop.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use smallvec::SmallVec;

use super::{LoopForest, LoopId};
use crate::graph::view::DenseId;
use crate::identity::{NO_INDEX, slot, word};

impl<N: DenseId> LoopForest<N> {
    /// A preorder numbering of the properly nested loops that gives each loop's
    /// subtree a run of positions. Children come before parents in `order`, so
    /// the sizes add up in it and the positions are handed out in its reverse.
    pub(super) fn number_subtrees(&mut self, order: &[usize]) {
        let mut subtree: SmallVec<[u32; 8]> = SmallVec::from_elem(1, self.loops.len());
        for &k in order {
            let up = self.loops[k].enclosing;
            if up != NO_INDEX {
                subtree[slot(up)] += subtree[k];
            }
        }
        let mut next: SmallVec<[u32; 8]> = SmallVec::from_elem(0, self.loops.len());
        let mut next_outermost = 0_u32;
        for &k in order.iter().rev() {
            let up = self.loops[k].enclosing;
            let cursor = if up == NO_INDEX {
                &mut next_outermost
            } else {
                &mut next[slot(up)]
            };
            let first = *cursor;
            *cursor += subtree[k];
            next[k] = first + 1;
            self.loops[k].first = first;
            self.loops[k].last = first + subtree[k] - 1;
        }
        self.member_starts = vec![0; slot(next_outermost) + 1];
    }

    /// The blocks of each position, as one run apiece, in ascending order.
    /// `member_starts` holds the number of positions plus one zeros.
    pub(super) fn group_members(&mut self) {
        let positions = self.member_starts.len() - 1;
        for &held in self.innermost.iter().filter(|&&held| held != NO_INDEX) {
            self.member_starts[slot(self.loops[slot(held)].first)] += 1;
        }
        let mut total = 0;
        for count in &mut self.member_starts[..positions] {
            total += *count;
            *count = total;
        }
        self.member_starts[positions] = total;
        self.members = vec![N::from_index(0); slot(total)];
        for (node, &held) in self.innermost.iter().enumerate().rev() {
            if held != NO_INDEX {
                let end = &mut self.member_starts[slot(self.loops[slot(held)].first)];
                *end -= 1;
                self.members[slot(*end)] = N::from_index(node);
            }
        }
    }

    /// Work out the nesting rule's parent and depth of every loop, and the
    /// greatest depth along each properly nested loop's chain of enclosing
    /// loops. `order` is the properly nested loops, innermost first.
    ///
    /// The parent of a loop is the smallest loop with a strictly larger body
    /// that contains its header, the one with the lowest header among equals.
    /// Among the properly nested loops alone, that is the loop directly around
    /// it, so a forest of them needs no search.
    pub(super) fn nest(&mut self, order: &[usize]) {
        if self.explicit.is_empty() {
            for record in &mut self.loops {
                record.parent = record.enclosing;
            }
            self.assign_depths(order.iter().rev().copied());
        } else {
            let (parents, outermost_first) = self.parents_among_explicit_loops();
            for (record, parent) in self.loops.iter_mut().zip(parents) {
                record.parent = parent;
            }
            self.assign_depths(outermost_first.into_iter());
        }

        for &k in order.iter().rev() {
            let around = self.loops[k].enclosing;
            let outer = if around == NO_INDEX {
                0
            } else {
                self.loops[slot(around)].deepest
            };
            self.loops[k].deepest = self.loops[k].depth.max(outer);
        }
    }

    /// Give every loop the depth below its parent, handing out loops
    /// `outermost_first` so that a parent has its depth before its children.
    fn assign_depths(&mut self, outermost_first: impl Iterator<Item = usize>) {
        for k in outermost_first {
            self.loops[k].depth = match self.loops[k].parent {
                NO_INDEX => 0,
                parent => self.loops[slot(parent)].depth + 1,
            };
        }
    }

    /// The parent of every loop, found by the rule when explicit loops may be
    /// the smallest enclosing one, and the loops largest first.
    ///
    /// The properly nested loops that contain a header are a chain, nearest
    /// first, and their bodies grow along it, so the first one larger than the
    /// loop is the best of them. An explicit loop is found by asking each one.
    fn parents_among_explicit_loops(&self) -> (Vec<u32>, Vec<usize>) {
        let sizes: Vec<usize> = self.loops().map(|lp| self.body_count(lp)).collect();
        let explicit: Vec<usize> = (0..self.loops.len())
            .filter(|&k| !self.loops[k].is_nested())
            .collect();

        let parents = (0..self.loops.len())
            .map(|k| {
                let header = self.headers[k];
                let mut best: Option<(usize, N, usize)> = None;
                let mut consider = |candidate: usize| {
                    let key = (sizes[candidate], self.headers[candidate]);
                    if best.is_none_or(|(size, header, _)| key < (size, header)) {
                        best = Some((key.0, key.1, candidate));
                    }
                };
                let mut chain = self.innermost[header.index()];
                while chain != NO_INDEX {
                    let candidate = slot(chain);
                    if sizes[candidate] > sizes[k] {
                        consider(candidate);
                        break;
                    }
                    chain = self.loops[candidate].enclosing;
                }
                for &candidate in &explicit {
                    if candidate != k
                        && sizes[candidate] > sizes[k]
                        && self.contains(LoopId(word(candidate)), header)
                    {
                        consider(candidate);
                    }
                }
                best.map_or(NO_INDEX, |(_, _, candidate)| word(candidate))
            })
            .collect();

        // A parent's body is strictly larger than its child's.
        let mut largest_first: Vec<usize> = (0..self.loops.len()).collect();
        largest_first.sort_unstable_by_key(|&k| Reverse(sizes[k]));
        (parents, largest_first)
    }
}
