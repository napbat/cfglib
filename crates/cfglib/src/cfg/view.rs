//! Allocation-free successor and predecessor iterators of [`Cfg`].

use core::ops::Index;
use core::slice;

use crate::block::{BasicBlock, BlockId};
use crate::edge::{Edge, EdgeId};

use super::Cfg;

impl<I, E> crate::graph::view::DirectedGraphView for Cfg<I, E> {
    type NodeId = BlockId;

    fn node_count(&self) -> usize {
        self.block_count()
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        Cfg::successors(self, node)
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        Cfg::predecessors(self, node)
    }
}

impl<I, E> crate::graph::view::NodeGraphView for Cfg<I, E> {
    type NodeData = BasicBlock<I>;

    fn node_ref(&self, node: Self::NodeId) -> &Self::NodeData {
        self.block(node)
    }
}

impl<I, E> crate::graph::view::RootedGraphView for Cfg<I, E> {
    fn root(&self) -> Self::NodeId {
        self.entry()
    }
}

impl<I, E> Index<BlockId> for Cfg<I, E> {
    type Output = BasicBlock<I>;

    /// Index into the CFG by [`BlockId`].
    ///
    /// Equivalent to [`Cfg::block`] but usable with `cfg[id]` syntax.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a block in this CFG.
    #[inline]
    fn index(&self, id: BlockId) -> &BasicBlock<I> {
        &self.blocks[id.index()]
    }
}

impl<I, E> Index<EdgeId> for Cfg<I, E> {
    type Output = Edge<E>;

    /// Index into the CFG by [`EdgeId`].
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a live edge in this CFG.
    #[inline]
    fn index(&self, id: EdgeId) -> &Edge<E> {
        self.edges[id.index()]
            .as_ref()
            .expect("edge has been removed")
    }
}

/// Iterator over successor block ids (zero-allocation).
pub struct Successors<'a, I, E = ()> {
    pub(super) cfg: &'a Cfg<I, E>,
    pub(super) iter: slice::Iter<'a, EdgeId>,
}

impl<I, E> Iterator for Successors<'_, I, E> {
    type Item = BlockId;
    #[inline]
    fn next(&mut self) -> Option<BlockId> {
        self.iter.next().map(|&eid| self.cfg.edge(eid).target)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I, E> ExactSizeIterator for Successors<'_, I, E> {}

/// Iterator over predecessor block ids (zero-allocation).
pub struct Predecessors<'a, I, E = ()> {
    pub(super) cfg: &'a Cfg<I, E>,
    pub(super) iter: slice::Iter<'a, EdgeId>,
}

impl<I, E> Iterator for Predecessors<'_, I, E> {
    type Item = BlockId;
    #[inline]
    fn next(&mut self) -> Option<BlockId> {
        self.iter.next().map(|&eid| self.cfg.edge(eid).source)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I, E> ExactSizeIterator for Predecessors<'_, I, E> {}
