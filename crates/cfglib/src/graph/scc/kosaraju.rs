//! Kosaraju's two passes, numbering the components sources first.

extern crate alloc;
use alloc::vec::Vec;

use super::SccDecomposition;
use crate::graph::traverse::postorder_in;
use crate::graph::view::{DenseId, Direction, GraphView};
use crate::identity::{NO_INDEX, word};

/// Decompose `graph` with Kosaraju's classic two passes.
///
/// 1. A depth-first forest over roots in node-id order, taking each node's
///    successors in adjacency order, recording nodes by finish time.
/// 2. A walk of the reverse graph over those roots in reverse finish order,
///    minting one component per root that is not yet assigned. The first
///    root is in a source component of the condensation, and every root after
///    it is in a source component of what remains, which is what makes the
///    minting order topological.
pub(super) fn sources_first<G: GraphView>(graph: &G) -> SccDecomposition<G::NodeId> {
    let finish_order = postorder_in(graph, graph.node_ids(), Direction::Forward, &mut Vec::new());

    let mut result = SccDecomposition::with_node_bound(graph.node_bound());
    let mut stack = Vec::new();
    for root in finish_order.into_iter().rev() {
        if result.component_of[root.index()] != NO_INDEX {
            continue;
        }

        let index = word(result.component_count());
        result.component_of[root.index()] = index;
        stack.push(root);
        while let Some(node) = stack.pop() {
            result.nodes.push(node);
            for predecessor in graph.predecessors(node) {
                if result.component_of[predecessor.index()] == NO_INDEX {
                    result.component_of[predecessor.index()] = index;
                    stack.push(predecessor);
                }
            }
        }
        result.close_component();
    }
    result
}
