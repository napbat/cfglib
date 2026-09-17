//! Read-only view implementations that let the generic algorithms run over
//! [`Graph`] unchanged.
//!
//! The only subtlety is the meaning of
//! [`DirectedGraphView::node_count`]. The trait uses it as the dense bound —
//! its contract is that every index in `0..node_count()` is yielded exactly
//! once, and the algorithms size node-indexed arrays by it. A store with
//! removed nodes has more slots than nodes, so the honest report is the slot
//! count: an array sized by it covers every identity the view yields, and no
//! analysis can index out of bounds. The alternative, filtering removed slots
//! out of the view while reporting the live count, produces identities above
//! the bound and would make every array in `dominator.rs` and `scc.rs` a
//! latent panic.
//!
//! A removed slot is therefore an isolated node of the view, and
//! [`Graph::node_slot_count`] documents what that costs an analysis.
//! Nothing here asserts compactness: a store that is being updated
//! incrementally has a dense, contiguous identity space already — only
//! *removal* introduces the isolated slots, and refusing to run over them
//! would deny the engine its purpose.

use crate::graph::edge_view::{EdgeGraphView, EdgeRef};
use crate::graph::view::{DirectedGraphView, NodeGraphView};

use super::Graph;
use super::edge::EdgeRecord;
use super::id::{Id, IdTag};

impl<N, E, NT: IdTag, ET: IdTag> DirectedGraphView for Graph<N, E, NT, ET> {
    type NodeId = Id<NT>;

    /// The node **slot** count, removed nodes included — see
    /// [`Graph::node_slot_count`] for why, and for what it costs an analysis.
    fn node_count(&self) -> usize {
        self.node_slot_count()
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        Graph::successors(self, node)
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        Graph::predecessors(self, node)
    }
}

impl<N, E, NT: IdTag, ET: IdTag> NodeGraphView for Graph<N, E, NT, ET> {
    type NodeData = N;

    fn node_ref(&self, node: Self::NodeId) -> &Self::NodeData {
        self.node(node)
    }
}

impl<N, E, NT: IdTag, ET: IdTag> EdgeGraphView for Graph<N, E, NT, ET> {
    type EdgeId = Id<ET>;
    type EdgeData = E;

    fn edge_slot_count(&self) -> usize {
        Graph::edge_slot_count(self)
    }

    fn edge_ids(&self) -> impl Iterator<Item = Self::EdgeId> + '_ {
        Graph::edge_ids(self)
    }

    fn outgoing_edges(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.outgoing(node)
    }

    fn incoming_edges(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.incoming(node)
    }

    /// # Panics
    ///
    /// Panics when `edge` has been removed, as the trait requires. The
    /// inherent [`Graph::edge`] keeps a removed record readable instead.
    fn edge_ref(
        &self,
        edge: Self::EdgeId,
    ) -> EdgeRef<'_, Self::NodeId, Self::EdgeId, Self::EdgeData> {
        assert!(self.is_live_edge(edge), "edge has been removed");
        let record: &EdgeRecord<E, NT> = self.edge(edge);
        EdgeRef::new(edge, record.source(), record.target(), record.payload())
    }
}
