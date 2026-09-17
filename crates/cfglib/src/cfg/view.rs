//! The view-trait and indexing surface of [`Cfg`].

use core::ops::Index;

use crate::block::{BasicBlock, BlockId};
use crate::edge::{Edge, EdgeId};
use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::view::{GraphView, NodeView, RootedView};

use super::Cfg;

impl<I, E> GraphView for Cfg<I, E> {
    type NodeId = BlockId;

    fn node_bound(&self) -> usize {
        self.block_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.block_ids()
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        Cfg::successors(self, node)
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        Cfg::predecessors(self, node)
    }
}

impl<I, E> NodeView for Cfg<I, E> {
    type NodeData = BasicBlock<I>;

    fn node(&self, node: Self::NodeId) -> &Self::NodeData {
        self.block(node)
    }
}

impl<I, E> RootedView for Cfg<I, E> {
    fn root(&self) -> Self::NodeId {
        self.entry()
    }
}

impl<I, E> EdgeView for Cfg<I, E> {
    type EdgeId = EdgeId;
    type EdgeData = Edge<E>;

    fn edge_bound(&self) -> usize {
        Cfg::edge_bound(self)
    }

    fn edge_ids(&self) -> impl Iterator<Item = EdgeId> + '_ {
        Cfg::edge_ids(self)
    }

    fn outgoing(&self, node: BlockId) -> impl Iterator<Item = EdgeId> + '_ {
        Cfg::outgoing(self, node)
    }

    fn incoming(&self, node: BlockId) -> impl Iterator<Item = EdgeId> + '_ {
        Cfg::incoming(self, node)
    }

    fn edge(&self, edge: EdgeId) -> EdgeRef<'_, BlockId, EdgeId, Edge<E>> {
        Cfg::edge(self, edge)
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
    /// Panics if `id` names no block slot in this CFG.
    #[inline]
    fn index(&self, id: BlockId) -> &BasicBlock<I> {
        self.block(id)
    }
}

impl<I, E> Index<EdgeId> for Cfg<I, E> {
    type Output = Edge<E>;

    /// Index into the CFG by [`EdgeId`], yielding the edge's kind, weight,
    /// and consumer payload.
    ///
    /// Endpoints come from [`Cfg::edge`], which borrows them together with
    /// the payload.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a live edge in this CFG.
    #[inline]
    fn index(&self, id: EdgeId) -> &Edge<E> {
        assert!(self.contains_edge(id), "edge {id} has been removed");
        self.graph.edge(id).payload()
    }
}
