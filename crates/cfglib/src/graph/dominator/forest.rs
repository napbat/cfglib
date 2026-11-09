//! Dominance extended from a tree to a forest, so unreachable code is not left
//! without any.
//!
//! A tree computed from the root of a graph holds only what the root reaches.
//! Code the root misses still has structure worth reading: a handler pad no
//! exception edge leads to, a filter funclet the runtime invokes, the body of
//! a loop that nothing enters. [`DominatorTree::complete_forest_in`] gives
//! each such piece of code a tree of its own, rooted at the strongly connected
//! components no edge enters, which is where a path into the code can begin.
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
use crate::graph::view::{DenseId, RootedView};
use crate::graph::virtual_root::VirtualRoot;

#[cfg(test)]
mod tests;

impl<N: DenseId> DominatorTree<N> {
    /// Extend this tree, the ordinary dominator tree of `graph`, into a forest
    /// that also covers the code the root does not reach, giving each source
    /// component of that code a root of its own.
    ///
    /// Returns `None` when the root already reaches every node, which is the
    /// common case and the one that must stay free: the check is a scan, and
    /// the forest construction below it never runs. Otherwise returns the
    /// forest, which holds every node of `graph`, and leaves the roots it was
    /// built from in `roots`: the graph's own root and one block of each source
    /// component, in ascending order. A tree of the forest can start at a
    /// block that is not in that list, since a block two unrelated roots lead
    /// to has no dominator left.
    ///
    /// A source component is a strongly connected component of otherwise
    /// unreachable code that no edge from outside the component enters. A
    /// component entered only from other unreachable code is below one of
    /// those, and its blocks hang from the tree of the component that leads
    /// there; a block that two unrelated components lead to has no dominator
    /// but the virtual root that joins them, so it is a root of its own.
    ///
    /// The forest is the dominator tree of `graph` behind one virtual root
    /// that leads to each root, with the virtual root left out. So an edge
    /// between two trees of the forest always ends at the root of its target
    /// tree, and [`frontier`](Self::frontier),
    /// [`iterated_frontier`](Self::iterated_frontier) and
    /// [`DominanceFrontiers`](super::DominanceFrontiers) read the forest like
    /// any other tree. `scratch` holds the buffers of that construction, as it
    /// does for [`compute_in`](Self::compute_in), and `roots` is cleared
    /// first, so both can be reused for the next graph.
    ///
    /// # Panics
    ///
    /// Panics when `graph` has a node at or beyond the node bound this tree
    /// was computed with.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, DominatorScratch, DominatorTree, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let entry = cfg.entry();
    /// let live = cfg.new_block();
    /// let dead = cfg.new_block();
    /// let fed = cfg.new_block();
    /// cfg.add_edge(entry, live, EdgeKind::Fallthrough);
    /// cfg.add_edge(dead, fed, EdgeKind::Fallthrough);
    ///
    /// let tree = DominatorTree::compute(&cfg);
    /// let mut scratch = DominatorScratch::new();
    /// let mut roots = Vec::new();
    /// let forest = tree
    ///     .complete_forest_in(&mut scratch, &mut roots, &cfg)
    ///     .expect("the entry misses two blocks");
    /// assert_eq!(roots, [entry, dead]);
    /// assert_eq!(forest.idom(fed), Some(dead));
    /// assert_eq!(forest.idom(live), Some(entry));
    ///
    /// // A graph the entry reaches entirely is its own forest.
    /// let mut whole = Cfg::<u32>::new();
    /// let next = whole.new_block();
    /// whole.add_edge(whole.entry(), next, EdgeKind::Fallthrough);
    /// let tree = DominatorTree::compute(&whole);
    /// assert!(tree.complete_forest_in(&mut scratch, &mut Vec::new(), &whole).is_none());
    /// ```
    #[must_use]
    pub fn complete_forest_in<G: RootedView<NodeId = N>>(
        &self,
        scratch: &mut DominatorScratch,
        roots: &mut Vec<N>,
        graph: &G,
    ) -> Option<Self> {
        if graph.node_ids().all(|block| self.is_reachable(block)) {
            return None;
        }

        let components = kosaraju_scc(graph);
        roots.clear();
        roots.push(graph.root());
        for (component_index, component) in components.components().enumerate() {
            let Some(&first) = component.first() else {
                continue;
            };
            if self.is_reachable(first) {
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

        let view = VirtualRoot::new(graph, roots);
        let complete = DominatorTree::<usize>::compute_in(scratch, &view);
        let original_bound = graph.node_bound();
        let idom: Vec<Option<N>> = (0..original_bound)
            .map(|node| {
                complete
                    .idom(node)
                    .filter(|&parent| parent < original_bound)
                    .map(N::from_index)
            })
            .collect();
        let reachable: Vec<bool> = (0..original_bound)
            .map(|node| complete.is_reachable(node))
            .collect();
        Some(Self::from_forest_parts(&idom, &reachable))
    }
}
