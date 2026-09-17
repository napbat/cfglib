//! Incrementally updatable compressed graph storage.
//!
//! [`Graph`] is one type in two states at once: a **compressed base** that
//! costs nothing per node beyond its payload, and an **incremental delta**
//! holding everything added since the base was built. Both states are always
//! present and always readable; [`compact`](Graph::compact) folds the second
//! into the first when a consumer decides to pay for it.
//!
//! # Why not the arena
//!
//! The arena stores in this crate ([`DirectedGraph`](crate::DirectedGraph),
//! [`Cfg`](crate::Cfg)) keep two `SmallVec` adjacency containers per node.
//! That is the right shape for a few thousand basic blocks and the wrong one
//! for a whole codebase: two `SmallVec<[EdgeId; 4]>` fields cost 32 bytes of
//! inline storage on every node whether or not they are used, and a node
//! whose degree crosses the inline bound moves to its own heap allocation —
//! so a million-node symbol graph pays tens of megabytes of inline slack plus
//! one allocation per high-degree node, and every scan of it chases a pointer
//! per node.
//!
//! A pure compressed-sparse-row index removes both costs but cannot be
//! updated: appending one edge to node 3 means shifting every later node's
//! run. That is the trade this module refuses to make.
//!
//! # The shape
//!
//! ```text
//! base    nodes  [ N N N N N ]            payloads, dense
//!         edges  [ E E E E E E E ]        records, dense, insertion order
//!         out    offsets[i]..offsets[i+1] -> flat edge-id array
//!         in     offsets[i]..offsets[i+1] -> flat edge-id array
//!
//! delta   nodes  appended payloads
//!         edges  appended records
//!         out    one u32 per node: last edge of a circular chain
//!         in     one u32 per node: last edge of a circular chain
//!         next   one u32 per delta edge, per direction
//!
//! live    one bit per node slot, one bit per edge slot
//! ```
//!
//! Adjacency of a node is its base run followed by its delta chain. Base
//! identities are all below delta identities and both parts are ordered, so
//! the concatenation is exactly insertion order — the order
//! [`DirectedGraph`](crate::DirectedGraph) produces today, parallel edges
//! included. Consumers see no reordering when they move.
//!
//! ## Cost per node
//!
//! Two chain slots, eight bytes, and no allocation — against 32 bytes of
//! inline `SmallVec` storage plus a heap allocation on overflow. The chains
//! are circular with the per-node slot naming the chain's **last** element,
//! which is what makes append-order iteration possible from one pointer
//! instead of a head-and-tail pair.
//!
//! ## Removal
//!
//! Removing clears a bit in a liveness bitset — 1/64th the cost of a
//! `Vec<Option<T>>` tombstone, and the payload of a removed entity stays
//! readable through [`node`](Graph::node) and [`edge`](Graph::edge) until the
//! next compaction. Removing a node removes its edges in both directions.
//! Nothing else moves, so every identity a consumer holds stays valid.
//!
//! ## Compaction
//!
//! [`compact`](Graph::compact) rebuilds the base from the live entities and
//! returns a [`Renumbering`]: a total old-to-new mapping that
//! [`compose`](Renumbering::compose)s, so a consumer that skipped several
//! compactions folds them into one lookup.
//!
//! # There is no builder
//!
//! `Graph::new()`, `add_node`, `add_edge`, `compact()` *is* the builder, and
//! it is also the store, so a consumer that keeps mutating does not have to
//! choose a type up front or rebuild to get back to a mutable one.
//! [`CsrDirectedGraphBuilder`](crate::CsrDirectedGraphBuilder) is superseded
//! by this module and kept only until its callers move.
//!
//! # Dense analyses
//!
//! The crate's algorithms index arrays by node index, so a view must report a
//! bound that covers every identity it yields. [`Graph`] reports its **slot**
//! count, tombstones included, through
//! [`DirectedGraphView::node_count`](crate::DirectedGraphView::node_count) —
//! see [`node_slot_count`](Graph::node_slot_count) for what that means for an
//! analysis run over a store with removed nodes in it.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::graph::traverse::TraversalDirection;

use adjacency::{DeltaChains, MAX_SLOTS};
use liveness::LiveSet;

mod adjacency;
mod compact;
mod edge;
mod id;
mod liveness;
mod renumbering;
mod view;

#[cfg(test)]
mod tests;

pub use adjacency::AdjacentEdges;
pub use edge::EdgeRecord;
pub use id::{EdgeId, EdgeTag, Id, IdTag, NodeId, NodeTag};
pub use renumbering::Renumbering;

/// Narrow a slot count to the dense identity space.
fn raw_index(value: usize) -> u32 {
    u32::try_from(value).expect("dense slot count exceeds u32::MAX")
}

fn empty_slice<T>() -> Box<[T]> {
    Vec::new().into_boxed_slice()
}

/// A directed multigraph stored as a compressed base plus an appendable
/// delta, with stable dense identities and explicit compaction.
///
/// Node and edge payloads are consumer-defined; parallel edges and self edges
/// are retained; forward and reverse adjacency are both maintained. The tag
/// parameters exist so a consumer can give its nodes and edges domain
/// identities ([`Id<MyNodeTag>`](Id)) without a newtype and a pair of
/// conversion shims; the defaults cover the ordinary case.
///
/// See the [module documentation](self) for the storage design.
///
/// # Examples
///
/// ```
/// use cfglib::graph::store::Graph;
///
/// let mut graph = Graph::new();
/// let definition = graph.add_node("definition");
/// let call = graph.add_node("call");
/// let flow = graph.add_edge(definition, call, "returns");
///
/// assert_eq!(graph.successors(definition).collect::<Vec<_>>(), [call]);
/// assert_eq!(graph.edge(flow).payload(), &"returns");
///
/// // Appending after a compaction costs no allocation per node.
/// graph.compact();
/// let other = graph.add_node("other");
/// graph.add_edge(definition, other, "escapes");
/// assert_eq!(graph.successors(definition).collect::<Vec<_>>(), [call, other]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "N: serde::Serialize, E: serde::Serialize",
        deserialize = "N: serde::Deserialize<'de>, E: serde::Deserialize<'de>"
    ))
)]
pub struct Graph<N, E, NT: IdTag = NodeTag, ET: IdTag = EdgeTag> {
    base_nodes: Box<[N]>,
    base_edges: Box<[EdgeRecord<E, NT>]>,
    out_offsets: Box<[u32]>,
    out_edges: Box<[u32]>,
    in_offsets: Box<[u32]>,
    in_edges: Box<[u32]>,
    delta_nodes: Vec<N>,
    delta_edges: Vec<EdgeRecord<E, NT>>,
    out_chains: DeltaChains,
    in_chains: DeltaChains,
    live_nodes: LiveSet,
    live_edges: LiveSet,
    live_node_count: usize,
    live_edge_count: usize,
    #[cfg_attr(feature = "serde", serde(skip))]
    edge_tag: PhantomData<fn() -> ET>,
}

impl<N, E> Graph<N, E> {
    /// Create an empty store with the default node and edge tags.
    ///
    /// Tagged stores use [`tagged`](Graph::tagged) instead, for the reason
    /// `HashMap::new` is restricted to the default hasher: a type parameter
    /// with a default is still a parameter, and inference cannot recover it
    /// from the payloads alone.
    #[must_use]
    pub fn new() -> Self {
        Self::tagged()
    }

    /// Create an empty default-tagged store with room for the requested
    /// nodes and edges.
    #[must_use]
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self::tagged_with_capacity(nodes, edges)
    }
}

impl<N, E, NT: IdTag, ET: IdTag> Graph<N, E, NT, ET> {
    /// Create an empty store over consumer-chosen identity tags.
    #[must_use]
    pub fn tagged() -> Self {
        Self {
            base_nodes: empty_slice(),
            base_edges: empty_slice(),
            out_offsets: empty_slice(),
            out_edges: empty_slice(),
            in_offsets: empty_slice(),
            in_edges: empty_slice(),
            delta_nodes: Vec::new(),
            delta_edges: Vec::new(),
            out_chains: DeltaChains::new(),
            in_chains: DeltaChains::new(),
            live_nodes: LiveSet::new(),
            live_edges: LiveSet::new(),
            live_node_count: 0,
            live_edge_count: 0,
            edge_tag: PhantomData,
        }
    }

    /// Create an empty tagged store with room for the requested nodes and
    /// edges.
    ///
    /// The reservation covers the delta, which is where a fresh store puts
    /// everything, so a consumer that knows its size builds without a single
    /// reallocation and then compacts once.
    #[must_use]
    pub fn tagged_with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            delta_nodes: Vec::with_capacity(nodes),
            delta_edges: Vec::with_capacity(edges),
            out_chains: DeltaChains::with_capacity(nodes, edges),
            in_chains: DeltaChains::with_capacity(nodes, edges),
            live_nodes: LiveSet::with_capacity(nodes),
            live_edges: LiveSet::with_capacity(edges),
            ..Self::tagged()
        }
    }

    /// Add a node and return its stable identity.
    ///
    /// Constant time, and no allocation beyond the amortized growth of the
    /// delta arrays: a node costs its payload, eight bytes of chain slots,
    /// and two bits.
    ///
    /// # Panics
    ///
    /// Panics when the store would exceed the dense identity space.
    pub fn add_node(&mut self, payload: N) -> Id<NT> {
        let slot = self.node_slot_count();
        assert!(
            slot < MAX_SLOTS,
            "node count exceeds the dense identity space"
        );
        self.delta_nodes.push(payload);
        self.out_chains.push_node();
        self.in_chains.push_node();
        self.live_nodes.push_live();
        self.live_node_count += 1;
        Id::from_index(slot)
    }

    /// Add a directed edge and return its stable identity.
    ///
    /// Parallel edges and self edges are retained. The edge is appended to
    /// both endpoints' delta chains in constant time; a node that was part of
    /// the compressed base gains delta neighbors without its base run moving.
    ///
    /// # Panics
    ///
    /// Panics when either endpoint is out of range or has been removed, or
    /// when the store would exceed the dense identity space.
    pub fn add_edge(&mut self, source: Id<NT>, target: Id<NT>, payload: E) -> Id<ET> {
        self.assert_live_endpoint(source, "source");
        self.assert_live_endpoint(target, "target");
        let slot = self.edge_slot_count();
        assert!(
            slot < MAX_SLOTS,
            "edge count exceeds the dense identity space"
        );

        self.delta_edges
            .push(EdgeRecord::new(source, target, payload));
        self.out_chains.append(source.index());
        self.in_chains.append(target.index());
        self.live_edges.push_live();
        self.live_edge_count += 1;
        Id::from_index(slot)
    }

    /// Remove an edge, returning whether it had been live.
    ///
    /// Every other identity, in both directions of adjacency, is unaffected.
    pub fn remove_edge(&mut self, edge: Id<ET>) -> bool {
        self.retire_edge(edge.index())
    }

    /// Remove a node and all of its edges, returning whether it had been
    /// live.
    ///
    /// Costs one walk of the node's two adjacency axes. An edge removed this
    /// way is indistinguishable from one removed directly.
    pub fn remove_node(&mut self, node: Id<NT>) -> bool {
        let index = node.index();
        if index >= self.node_slot_count() || !self.live_nodes.is_live(index) {
            return false;
        }
        self.clear_adjacency(node, TraversalDirection::Outgoing);
        self.clear_adjacency(node, TraversalDirection::Incoming);
        self.live_nodes.clear(index);
        self.live_node_count -= 1;
        true
    }

    /// Borrow a node payload.
    ///
    /// A removed node's payload stays readable until the next compaction;
    /// ask [`is_live_node`](Self::is_live_node) when that distinction
    /// matters.
    ///
    /// # Panics
    ///
    /// Panics when `node` names no slot in this store.
    #[must_use]
    pub fn node(&self, node: Id<NT>) -> &N {
        let index = node.index();
        let base = self.base_nodes.len();
        if index < base {
            &self.base_nodes[index]
        } else {
            &self.delta_nodes[index - base]
        }
    }

    /// Mutably borrow a node payload.
    ///
    /// # Panics
    ///
    /// Panics when `node` names no slot in this store.
    pub fn node_mut(&mut self, node: Id<NT>) -> &mut N {
        let index = node.index();
        let base = self.base_nodes.len();
        if index < base {
            &mut self.base_nodes[index]
        } else {
            &mut self.delta_nodes[index - base]
        }
    }

    /// Borrow an edge's endpoints and payload.
    ///
    /// A removed edge's record stays readable until the next compaction.
    ///
    /// # Panics
    ///
    /// Panics when `edge` names no slot in this store.
    #[must_use]
    pub fn edge(&self, edge: Id<ET>) -> &EdgeRecord<E, NT> {
        let index = edge.index();
        let base = self.base_edges.len();
        if index < base {
            &self.base_edges[index]
        } else {
            &self.delta_edges[index - base]
        }
    }

    /// Mutably borrow an edge record, whose payload is the mutable part.
    ///
    /// # Panics
    ///
    /// Panics when `edge` names no slot in this store.
    pub fn edge_mut(&mut self, edge: Id<ET>) -> &mut EdgeRecord<E, NT> {
        let index = edge.index();
        let base = self.base_edges.len();
        if index < base {
            &mut self.base_edges[index]
        } else {
            &mut self.delta_edges[index - base]
        }
    }

    /// Whether `node` names a node that has not been removed.
    #[must_use]
    pub fn is_live_node(&self, node: Id<NT>) -> bool {
        self.live_nodes.is_live(node.index())
    }

    /// Whether `edge` names an edge that has not been removed.
    #[must_use]
    pub fn is_live_edge(&self, edge: Id<ET>) -> bool {
        self.live_edges.is_live(edge.index())
    }

    /// The number of nodes the store holds.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.live_node_count
    }

    /// The number of edges the store holds.
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.live_edge_count
    }

    /// The number of node slots, removed nodes included.
    ///
    /// Every live [`Id`] is below this, so it is the correct size for a
    /// node-indexed side table — and it is what the store reports as
    /// [`DirectedGraphView::node_count`](crate::DirectedGraphView::node_count),
    /// because an analysis sizing an array by that number must cover every
    /// identity the view yields.
    ///
    /// The consequence is worth stating plainly: a removed node remains a
    /// node of the *view*, with no edges. Reachability-based analyses
    /// (dominators, traversals) simply find it unreachable, but a whole-graph
    /// partition such as [`tarjan_scc`](crate::tarjan_scc) reports it as a
    /// singleton component. [`compact`](Self::compact) first when that
    /// matters; [`is_compact`](Self::is_compact) says whether it would change
    /// anything.
    #[must_use]
    pub fn node_slot_count(&self) -> usize {
        self.base_nodes.len() + self.delta_nodes.len()
    }

    /// The number of edge slots, removed edges included.
    #[must_use]
    pub fn edge_slot_count(&self) -> usize {
        self.base_edges.len() + self.delta_edges.len()
    }

    /// Whether the store holds no nodes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live_node_count == 0
    }

    /// Iterate over every live node identity in ascending order.
    pub fn node_ids(&self) -> impl Iterator<Item = Id<NT>> + '_ {
        (0..self.node_slot_count())
            .filter(|&slot| self.live_nodes.is_live(slot))
            .map(Id::from_index)
    }

    /// Iterate over every live edge identity in ascending order, which is
    /// also insertion order.
    pub fn edge_ids(&self) -> impl Iterator<Item = Id<ET>> + '_ {
        (0..self.edge_slot_count())
            .filter(|&slot| self.live_edges.is_live(slot))
            .map(Id::from_index)
    }

    /// Iterate over every live edge record in insertion order.
    pub fn edges(&self) -> impl Iterator<Item = &EdgeRecord<E, NT>> + '_ {
        self.edge_ids().map(|edge| self.edge(edge))
    }

    fn assert_live_endpoint(&self, node: Id<NT>, role: &str) {
        assert!(
            node.index() < self.node_slot_count(),
            "{role} node is out of range"
        );
        assert!(
            self.live_nodes.is_live(node.index()),
            "{role} node has been removed"
        );
    }

    fn retire_edge(&mut self, slot: usize) -> bool {
        if !self.live_edges.clear(slot) {
            return false;
        }
        self.live_edge_count -= 1;
        true
    }
}

impl<N, E, NT: IdTag, ET: IdTag> Default for Graph<N, E, NT, ET> {
    fn default() -> Self {
        Self::tagged()
    }
}
