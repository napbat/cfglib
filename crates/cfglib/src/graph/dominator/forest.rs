//! Dominance extended from a tree to a forest, so unreachable code is not left
//! without any.
//!
//! A tree computed from the root of a graph holds only what the root reaches.
//! Code the root misses still has structure worth reading: a handler pad no
//! exception edge leads to, a filter funclet the runtime invokes, the body of
//! a loop that nothing enters. [`complete_dominator_forest`] gives each such
//! piece of code a tree of its own, rooted at the strongly connected components
//! no edge enters, which is where a path into the code can begin.
//!
//! The forest is a whole-graph answer, so it is not the dominance of the live
//! code: unreachable code that jumps into live code is another way in, and a
//! live block it feeds loses the ordinary root as its dominator. A consumer
//! that wants the live code's dominance uses the tree it already has, and
//! reads the forest only for the blocks that tree does not hold.

extern crate alloc;

use alloc::vec::Vec;

use super::{DominatorScratch, DominatorTree};
use crate::graph::scc::kosaraju_scc;
use crate::graph::view::{DenseId, GraphView, RootedView};

#[cfg(test)]
mod tests;

/// Whole-graph dominance view with a private virtual root connected to the
/// ordinary root and every source SCC of otherwise unreachable code.
///
/// Nodes are dense indices, with the virtual root one past the graph's bound.
struct ForestView<'a, G: GraphView> {
    graph: &'a G,
    roots: &'a [G::NodeId],
}

impl<G: GraphView> ForestView<'_, G> {
    fn virtual_root(&self) -> usize {
        self.graph.node_bound()
    }
}

impl<G: GraphView> GraphView for ForestView<'_, G> {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.graph.node_bound() + 1
    }

    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph
            .node_ids()
            .map(DenseId::index)
            .chain(core::iter::once(self.virtual_root()))
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        let original_bound = self.graph.node_bound();
        let original = (node < original_bound).then(|| G::NodeId::from_index(node));
        (node == original_bound)
            .then_some(self.roots)
            .into_iter()
            .flatten()
            .copied()
            .map(DenseId::index)
            .chain(
                original
                    .into_iter()
                    .flat_map(|block| self.graph.successors(block).map(DenseId::index)),
            )
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        let original_bound = self.graph.node_bound();
        let original = (node < original_bound).then(|| G::NodeId::from_index(node));
        let virtual_predecessor = original
            .filter(|block| self.roots.binary_search(block).is_ok())
            .map(|_| original_bound);
        virtual_predecessor.into_iter().chain(
            original
                .into_iter()
                .flat_map(|block| self.graph.predecessors(block).map(DenseId::index)),
        )
    }
}

impl<G: GraphView> RootedView for ForestView<'_, G> {
    fn root(&self) -> Self::NodeId {
        self.virtual_root()
    }
}

/// Extends the dominator tree of a rooted graph into a forest that also covers
/// the code the root does not reach, giving each source component of that code
/// a root of its own.
///
/// `dominators` is the ordinary tree of `graph`. Returns `None` when the root
/// already reaches every node, which is the common case and the one that must
/// stay free: the check is a scan, and the forest construction below it never
/// runs. Otherwise returns the forest and leaves the roots it was built from
/// in `roots`: the graph's own root and one block of each source component,
/// in ascending order. A tree of the forest can start at a block that is not in
/// that list, since a block two unrelated roots lead to has no dominator left.
///
/// A source component is a strongly connected component of otherwise
/// unreachable code that no edge from outside the component enters. A
/// component entered only from other unreachable code is below one of those,
/// and its blocks hang from the tree of the component that leads there; a block
/// that two unrelated components lead to has no dominator but the virtual root
/// that joins them, so it is a root of its own.
pub(crate) fn complete_dominator_forest<G: RootedView>(
    scratch: &mut DominatorScratch,
    roots: &mut Vec<G::NodeId>,
    graph: &G,
    dominators: &DominatorTree<G::NodeId>,
) -> Option<DominatorTree<G::NodeId>> {
    if graph.node_ids().all(|block| dominators.is_reachable(block)) {
        return None;
    }

    let components = kosaraju_scc(graph);
    roots.clear();
    roots.push(graph.root());
    for (component_index, component) in components.components().enumerate() {
        let Some(&first) = component.first() else {
            continue;
        };
        if dominators.is_reachable(first) {
            continue;
        }
        let has_external_predecessor = component.iter().copied().any(|block| {
            graph
                .predecessors(block)
                .any(|predecessor| components.component_index(predecessor) != component_index)
        });
        if !has_external_predecessor {
            roots.push(first);
        }
    }
    roots.sort_unstable();
    roots.dedup();

    let view = ForestView { graph, roots };
    let complete = DominatorTree::<usize>::compute_in(scratch, &view);
    let original_bound = graph.node_bound();
    let idom: Vec<Option<G::NodeId>> = (0..original_bound)
        .map(|node| {
            complete
                .idom(node)
                .filter(|&parent| parent < original_bound)
                .map(G::NodeId::from_index)
        })
        .collect();
    let reachable: Vec<bool> = (0..original_bound)
        .map(|node| complete.is_reachable(node))
        .collect();
    Some(DominatorTree::from_forest_parts(&idom, &reachable))
}
