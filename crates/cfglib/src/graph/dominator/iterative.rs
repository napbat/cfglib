//! The Cooper-Harvey-Kennedy iterative algorithm, kept as a test oracle.
//!
//! It is the construction this crate used before the simple Lengauer-Tarjan
//! algorithm replaced it. Its worst case is quadratic or worse, so production
//! code never calls it. It shares no code with the production construction,
//! which makes it an independent answer to compare that construction against.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::DominatorTree;
use crate::graph::traverse::{TraversalDirection, reverse_postorder};
use crate::graph::view::{DenseId, GraphView};

impl<N: DenseId> DominatorTree<N> {
    /// Compute the dominator tree of `graph` from `root` by iterating the
    /// intersection of predecessor dominators over a reverse postorder until
    /// nothing changes.
    pub(super) fn compute_iterative<G>(graph: &G, root: N) -> Self
    where
        G: GraphView<NodeId = N>,
    {
        let order: Vec<usize> = reverse_postorder(graph, root, TraversalDirection::Outgoing)
            .into_iter()
            .map(DenseId::index)
            .collect();
        let node_count = graph.node_bound();
        let mut order_index = vec![usize::MAX; node_count];
        for (index, node) in order.iter().copied().enumerate() {
            order_index[node] = index;
        }

        let mut dominators: Vec<Option<usize>> = vec![None; order.len()];
        dominators[order_index[root.index()]] = Some(order_index[root.index()]);

        let mut changed = true;
        while changed {
            changed = false;
            for node in order.iter().copied().filter(|&node| node != root.index()) {
                let node_order = order_index[node];
                let mut new_parent_index = None;
                for predecessor in graph.predecessors(N::from_index(node)) {
                    let predecessor_order = order_index[predecessor.index()];
                    if predecessor_order == usize::MAX || dominators[predecessor_order].is_none() {
                        continue;
                    }

                    new_parent_index = Some(match new_parent_index {
                        None => predecessor_order,
                        Some(parent) if predecessor_order == parent => parent,
                        Some(parent) => intersect(&dominators, predecessor_order, parent),
                    });
                }
                let Some(new_parent_index) = new_parent_index else {
                    continue;
                };

                if dominators[node_order] != Some(new_parent_index) {
                    dominators[node_order] = Some(new_parent_index);
                    changed = true;
                }
            }
        }

        let mut immediate = vec![None; node_count];
        for (index, parent) in dominators.iter().copied().enumerate() {
            let node = order[index];
            immediate[node] = parent.map(|parent_index| N::from_index(order[parent_index]));
        }
        immediate[root.index()] = None;
        let mut reachable = vec![false; node_count];
        for &node in &order {
            reachable[node] = true;
        }
        Self::from_forest_parts(&immediate, &reachable)
    }
}

fn intersect(dominators: &[Option<usize>], mut left: usize, mut right: usize) -> usize {
    while left != right {
        while left > right {
            left = dominators[left].expect("processed dominator must have a parent");
        }
        while right > left {
            right = dominators[right].expect("processed dominator must have a parent");
        }
    }
    left
}
