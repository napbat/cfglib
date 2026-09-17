//! Read-only view implementations that let the generic algorithms run over
//! [`Graph`] unchanged.
//!
//! The store separates the two numbers the view contract distinguishes: the
//! **bound** is the slot count, which covers every identity the store can
//! yield and is therefore the right size for a dense side table, while
//! [`Graph::node_ids`] yields only the live nodes. A removed node is not a
//! node of the view at all, so a whole-graph partition such as
//! [`tarjan_scc`](crate::tarjan_scc) reports no phantom singleton for it and
//! no analysis has to compact the store first.

use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::view::{GraphView, NodeView};

use super::Graph;
use super::edge::EdgeRecord;
use super::id::{Id, IdTag};

impl<N, E, NT: IdTag, ET: IdTag> GraphView for Graph<N, E, NT, ET> {
    type NodeId = Id<NT>;

    fn node_bound(&self) -> usize {
        Graph::node_bound(self)
    }

    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
        Graph::node_ids(self)
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        Graph::successors(self, node)
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        Graph::predecessors(self, node)
    }
}

impl<N, E, NT: IdTag, ET: IdTag> NodeView for Graph<N, E, NT, ET> {
    type NodeData = N;

    fn node(&self, node: Self::NodeId) -> &Self::NodeData {
        Graph::node(self, node)
    }
}

impl<N, E, NT: IdTag, ET: IdTag> EdgeView for Graph<N, E, NT, ET> {
    type EdgeId = Id<ET>;
    type EdgeData = E;

    fn edge_bound(&self) -> usize {
        Graph::edge_bound(self)
    }

    fn edge_ids(&self) -> impl Iterator<Item = Self::EdgeId> + '_ {
        Graph::edge_ids(self)
    }

    fn outgoing(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        Graph::outgoing(self, node)
    }

    fn incoming(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        Graph::incoming(self, node)
    }

    /// # Panics
    ///
    /// Panics when `edge` has been removed, as the trait requires. The
    /// inherent [`Graph::edge`] keeps a removed record readable instead.
    fn edge(&self, edge: Self::EdgeId) -> EdgeRef<'_, Self::NodeId, Self::EdgeId, Self::EdgeData> {
        assert!(self.contains_edge(edge), "edge has been removed");
        let record: &EdgeRecord<E, NT> = Graph::edge(self, edge);
        EdgeRef::new(edge, record.source(), record.target(), record.payload())
    }
}
