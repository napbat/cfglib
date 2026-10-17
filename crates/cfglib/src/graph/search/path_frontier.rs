//! The queue of path prefixes awaiting expansion in a bounded path enumeration.

extern crate alloc;

use alloc::collections::VecDeque;

use super::SearchOrder;

/// Path prefixes waiting to be expanded, popped in a [`SearchOrder`].
///
/// The prefixes pushed between two pops are the successors of the one
/// expansion in between (or the seeds, before the first pop). A breadth-first
/// frontier pops in push order. A depth-first frontier pops the newest
/// expansion's first successor first, so successors are expanded in the order
/// they were pushed, the same convention [`search`](super::search) uses for
/// its seeds and successors. Both pops take constant time per prefix and the
/// frontier allocates nothing beyond its queue.
pub(crate) struct PathFrontier<T> {
    order: SearchOrder,
    queue: VecDeque<T>,
    /// Prefixes pushed since the last pop: the tail a depth-first pop reorders.
    pending: usize,
}

impl<T> PathFrontier<T> {
    /// An empty frontier that pops in `order`.
    pub(crate) const fn new(order: SearchOrder) -> Self {
        Self {
            order,
            queue: VecDeque::new(),
            pending: 0,
        }
    }

    /// Queue one prefix.
    pub(crate) fn push(&mut self, prefix: T) {
        self.queue.push_back(prefix);
        self.pending += 1;
    }

    /// Remove the next prefix to expand, or `None` once the frontier is empty.
    pub(crate) fn pop(&mut self) -> Option<T> {
        let pending = core::mem::take(&mut self.pending);
        match self.order {
            SearchOrder::BreadthFirst => self.queue.pop_front(),
            SearchOrder::DepthFirst => {
                // The newest successors sit at the back in push order; a stack
                // pops the back, so reverse them in place to pop the first
                // pushed one first.
                let len = self.queue.len();
                for offset in 0..pending / 2 {
                    self.queue.swap(len - pending + offset, len - 1 - offset);
                }
                self.queue.pop_back()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec::Vec;

    use super::*;

    /// Expand a three-level tree whose node `n` has successors `10n + 1`,
    /// `10n + 2`, and `10n + 3` (down to depth two), recording the pop order.
    fn expansion_order(order: SearchOrder) -> Vec<u32> {
        let mut frontier = PathFrontier::new(order);
        frontier.push(0_u32);
        let mut popped = Vec::new();
        while let Some(node) = frontier.pop() {
            popped.push(node);
            if node < 10 {
                for successor in 1..=3 {
                    frontier.push(node * 10 + successor);
                }
            }
        }
        popped
    }

    #[test]
    fn breadth_first_expands_level_by_level_in_push_order() {
        assert_eq!(
            expansion_order(SearchOrder::BreadthFirst),
            [0, 1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33]
        );
    }

    #[test]
    fn depth_first_finishes_each_subtree_and_expands_successors_in_push_order() {
        assert_eq!(
            expansion_order(SearchOrder::DepthFirst),
            [0, 1, 11, 12, 13, 2, 21, 22, 23, 3, 31, 32, 33]
        );
    }

    #[test]
    fn an_expansion_without_successors_resumes_the_previous_one() {
        let mut frontier = PathFrontier::new(SearchOrder::DepthFirst);
        for prefix in [1, 2, 3] {
            frontier.push(prefix);
        }
        assert_eq!(frontier.pop(), Some(1));
        assert_eq!(frontier.pop(), Some(2), "no pushes in between");
        frontier.push(4);
        assert_eq!(frontier.pop(), Some(4), "the newest expansion goes first");
        assert_eq!(frontier.pop(), Some(3));
        assert_eq!(frontier.pop(), None);
    }
}
