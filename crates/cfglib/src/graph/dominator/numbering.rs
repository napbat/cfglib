//! Preorder numbering of a dominator forest.
//!
//! The numbering is a pure function of the parent table and the root set.
//! Roots are visited in ascending dense index and so are the children of every
//! node, so the arrays do not depend on the order in which a construction
//! algorithm discovered the tree.
//!
//! The walk keeps no stack. Children are chained as first-child and
//! next-sibling links, and a finished node returns to its parent through the
//! parent table, so a chain of 10⁵ nodes costs no more than a wide fan.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::{ABSENT, slot, word};
use crate::graph::view::DenseId;

/// The arrays a numbered forest adds to its parent table.
pub(super) struct Numbered<N> {
    /// Preorder index of each node, [`ABSENT`] when it is in no tree.
    pub(super) pre: Vec<u32>,
    /// Preorder index of each node's last descendant.
    pub(super) last: Vec<u32>,
    /// Distance from each node to the root of its tree.
    pub(super) level: Vec<u32>,
    /// The nodes of every tree in preorder.
    pub(super) order: Vec<N>,
}

/// The sibling chains [`NumberingBuffers::number`] walks, owned by the caller
/// so that a pass over many procedures builds them once.
#[derive(Debug, Clone, Default)]
pub(super) struct NumberingBuffers {
    first_child: Vec<u32>,
    next_sibling: Vec<u32>,
}

impl NumberingBuffers {
    /// Number the forest whose parent table is `parent` and whose roots are
    /// `roots`, in ascending order.
    ///
    /// `parent` holds [`ABSENT`] for a root and for a node outside the forest.
    /// `reachable` is the number of nodes the forest holds and sizes the
    /// preorder exactly.
    pub(super) fn number<N: DenseId>(
        &mut self,
        parent: &[u32],
        roots: &[u32],
        reachable: usize,
    ) -> Numbered<N> {
        let bound = parent.len();
        self.first_child.clear();
        self.first_child.resize(bound, ABSENT);
        self.next_sibling.clear();
        self.next_sibling.resize(bound, ABSENT);
        for node in (0..bound).rev() {
            let up = parent[node];
            if up != ABSENT {
                self.next_sibling[node] = self.first_child[slot(up)];
                self.first_child[slot(up)] = word(node);
            }
        }

        let mut numbered = Numbered {
            pre: vec![ABSENT; bound],
            last: vec![ABSENT; bound],
            level: vec![ABSENT; bound],
            order: Vec::with_capacity(reachable),
        };
        for &root in roots {
            self.walk(parent, slot(root), &mut numbered);
        }
        debug_assert_eq!(
            numbered.order.len(),
            reachable,
            "every reachable node must hang below a root"
        );
        numbered
    }

    /// Number the tree below `root` in preorder.
    fn walk<N: DenseId>(&self, parent: &[u32], root: usize, numbered: &mut Numbered<N>) {
        let mut node = root;
        numbered.level[node] = 0;
        loop {
            numbered.pre[node] = word(numbered.order.len());
            numbered.order.push(N::from_index(node));
            let child = self.first_child[node];
            if child != ABSENT {
                numbered.level[slot(child)] = numbered.level[node] + 1;
                node = slot(child);
                continue;
            }
            // `node` has no children left to enter, so its subtree ends at the
            // node numbered last. Climb until a sibling is waiting.
            loop {
                numbered.last[node] = word(numbered.order.len() - 1);
                if node == root {
                    return;
                }
                let sibling = self.next_sibling[node];
                if sibling != ABSENT {
                    numbered.level[slot(sibling)] = numbered.level[node];
                    node = slot(sibling);
                    break;
                }
                node = slot(parent[node]);
            }
        }
    }
}
