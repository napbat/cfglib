//! The graph the post-dominator tree is computed on.
//!
//! The simple Lengauer-Tarjan construction runs on the reverse of the blocks
//! the entry reaches, from a virtual exit that has an edge to every exit and
//! to every block the cut joined to it. This module presents that graph as a
//! [`GraphView`], so no node, edge, or adjacency list is copied to build it.

use super::cut::{CutBuffers, ExitStructure};
use crate::graph::view::{DenseId, GraphView};

/// The reverse of the reachable blocks of a graph, with a virtual exit.
///
/// The virtual exit takes the dense index `graph.node_bound()`. The view names
/// its nodes by plain `usize`, so a consumer's identity type is never asked to
/// represent an index past its own bound.
pub(super) struct AugmentedReverse<'a, G: GraphView> {
    graph: &'a G,
    cut: &'a CutBuffers,
    structure: &'a ExitStructure<G::NodeId>,
}

impl<'a, G: GraphView> AugmentedReverse<'a, G> {
    /// The reverse of `graph` over the blocks `cut` found reachable, joined to
    /// a virtual exit as `structure` says.
    pub(super) const fn new(
        graph: &'a G,
        cut: &'a CutBuffers,
        structure: &'a ExitStructure<G::NodeId>,
    ) -> Self {
        Self {
            graph,
            cut,
            structure,
        }
    }

    /// The index of the virtual exit.
    fn exit(&self) -> usize {
        self.graph.node_bound()
    }

    /// The block named by `node`, when `node` is a block the entry reaches.
    fn block(&self, node: usize) -> Option<G::NodeId> {
        (node < self.exit() && self.cut.reachable[node]).then(|| G::NodeId::from_index(node))
    }
}

impl<G: GraphView> GraphView for AugmentedReverse<'_, G> {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.exit() + 1
    }

    fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.exit())
            .filter(|&node| self.cut.reachable[node])
            .chain(core::iter::once(self.exit()))
    }

    /// The virtual exit leads to every exit and cut root. A block leads to
    /// the reachable blocks that precede it in the original graph.
    fn successors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        let from_exit = (node == self.exit()).then(|| {
            self.structure
                .exits
                .iter()
                .chain(&self.structure.cut_roots)
                .map(|&block| block.index())
        });
        let from_block = self.block(node).map(|block| {
            self.graph
                .predecessors(block)
                .map(DenseId::index)
                .filter(move |&before| self.cut.reachable[before])
        });
        from_exit
            .into_iter()
            .flatten()
            .chain(from_block.into_iter().flatten())
    }

    /// A block follows the blocks it leads to in the original graph, and the
    /// virtual exit when it is an exit or a cut root. The virtual exit has no
    /// predecessor.
    fn predecessors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        let block = self.block(node);
        let from_block = block.map(|block| self.graph.successors(block).map(DenseId::index));
        let to_exit = block
            .filter(|&block| self.cut.exit_edge[block.index()])
            .map(|_| self.exit());
        from_block.into_iter().flatten().chain(to_exit)
    }
}
