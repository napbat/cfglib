//! Explicit compaction: fold the delta back into the compressed base.
//!
//! Compaction is the only operation that moves an identity, and it always
//! reports what it moved. Keeping it explicit is the point of the design: a
//! consumer decides when to pay the rebuild, and until it does, every
//! identity it holds stays valid.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use crate::graph::traverse::TraversalDirection;

use super::Graph;
use super::adjacency::{NONE, compress};
use super::edge::EdgeRecord;
use super::id::{Id, IdTag};
use super::liveness::LiveSet;
use super::node_order::NodeOrder;
use super::renumbering::Renumbering;

impl<N, E, NT: IdTag, ET: IdTag> Graph<N, E, NT, ET> {
    /// Whether the store is a pure compressed base: no delta, no tombstones.
    ///
    /// A compact store is the one shape in which an identity's index is also
    /// its position among the live entities, which is what dense-indexed
    /// analyses assume — see the [module documentation](super) on running
    /// analyses over a store that has been mutated since.
    #[must_use]
    pub fn is_compact(&self) -> bool {
        self.delta_nodes.is_empty()
            && self.delta_edges.is_empty()
            && self.live_node_count == self.base_nodes.len()
            && self.live_edge_count == self.base_edges.len()
            && !self.has_relocations()
    }

    /// Rebuild the compressed base from the live entities and report the new
    /// numbering.
    ///
    /// Live entities keep their relative order, so the surviving identities
    /// are renumbered but never reordered: insertion order, adjacency order,
    /// and identity order all mean the same thing afterwards as before. The
    /// delta is emptied and every tombstone disappears. To number the nodes
    /// in an order of your choosing instead, use
    /// [`compact_ordered`](Self::compact_ordered).
    ///
    /// Costs one pass over each slot space plus two counting passes over the
    /// live edges, and allocates the new base once. A node whose adjacency
    /// reads out of identity order, because an edge was redirected to or from
    /// it, additionally costs one walk of that adjacency.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::Graph;
    ///
    /// let mut graph = Graph::<&'static str, ()>::new();
    /// let dropped = graph.add_node("dropped");
    /// let kept = graph.add_node("kept");
    /// graph.remove_node(dropped);
    ///
    /// let renumbering = graph.compact();
    /// assert!(graph.is_compact());
    /// assert_eq!(renumbering.node(dropped), None);
    /// assert_eq!(graph.node(renumbering.node(kept).unwrap()), &"kept");
    /// ```
    pub fn compact(&mut self) -> Renumbering<NT, ET> {
        let node_map = surviving_slots(&self.live_nodes, self.node_bound());
        self.rebuild(node_map, false)
    }

    /// Rebuild the compressed base like [`compact`](Self::compact), numbering
    /// the surviving nodes in `order` rather than in slot order.
    ///
    /// [`compact`](Self::compact) keeps the order the nodes were added in,
    /// which for a whole-program graph is discovery or file order: a node's
    /// neighbors then carry scattered identities, and so do the payloads,
    /// adjacency offsets and node-indexed side tables a scan reads for them.
    /// An order that numbers neighbors close together turns those reads into
    /// reads of nearby memory. A procedure-sized graph fits in cache either
    /// way and has nothing to gain.
    ///
    /// Only nodes move. Afterwards:
    ///
    /// - node identity order is `order`, so it no longer means insertion
    ///   order; a node added later still follows every compacted one;
    /// - edges keep their relative order, so edge identity order still means
    ///   insertion order and the edge renumbering is the one
    ///   [`compact`](Self::compact) would report;
    /// - every node keeps its adjacency order: its edges read in the order
    ///   they read before, under their new identities.
    ///
    /// The delta is emptied and every tombstone disappears, as with
    /// [`compact`](Self::compact), and the returned [`Renumbering`]
    /// [`compose`](Renumbering::compose)s with any other.
    ///
    /// Costs what [`compact`](Self::compact) costs plus the order: one pass
    /// over the live edges for [`FirstTouch`](NodeOrder::FirstTouch), one
    /// search over the reached nodes and their edges for
    /// [`BreadthFirst`](NodeOrder::BreadthFirst) and
    /// [`ReversePostorder`](NodeOrder::ReversePostorder), and one pass over
    /// the order for [`Explicit`](NodeOrder::Explicit). The payloads move
    /// into their new places by swaps, one per payload at most. A search
    /// reads the adjacency in the order it discovers nodes, which is the
    /// scattered order this compaction exists to repair: on a million-node
    /// graph with shuffled identities, `benches/graph-store.rs` measures a
    /// search compaction at several times a plain one.
    ///
    /// # Panics
    ///
    /// Panics when a root of [`BreadthFirst`](NodeOrder::BreadthFirst) or
    /// [`ReversePostorder`](NodeOrder::ReversePostorder) names a node this
    /// store does not hold, and when an [`Explicit`](NodeOrder::Explicit)
    /// order is not a permutation of the live nodes: it names a node the
    /// store does not hold, names a node twice, or leaves one out. The store
    /// is unchanged when it panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Graph, NodeOrder};
    ///
    /// let mut graph = Graph::<&'static str, ()>::new();
    /// let exit = graph.add_node("exit");
    /// let body = graph.add_node("body");
    /// let entry = graph.add_node("entry");
    /// graph.add_edge(entry, body, ());
    /// graph.add_edge(body, body, ());
    /// graph.add_edge(body, exit, ());
    ///
    /// let renumbering = graph.compact_ordered(NodeOrder::ReversePostorder(&[entry]));
    /// assert!(graph.is_compact());
    /// assert_eq!(graph.nodes().copied().collect::<Vec<_>>(), ["entry", "body", "exit"]);
    ///
    /// let body = renumbering.node(body).unwrap();
    /// let exit = renumbering.node(exit).unwrap();
    /// assert_eq!(graph.successors(body).collect::<Vec<_>>(), [body, exit]);
    /// ```
    pub fn compact_ordered(&mut self, order: NodeOrder<'_, NT>) -> Renumbering<NT, ET> {
        let node_map = self.ordered_slots(order);
        self.rebuild(node_map, true)
    }

    /// Rebuild the base under `node_map`, which numbers every live node slot.
    /// Unless the map `reorders`, it must keep the slots' relative order,
    /// which saves placing the payloads one by one.
    fn rebuild(&mut self, node_map: Vec<u32>, reorders: bool) -> Renumbering<NT, ET> {
        // Edge numbers always keep slot order: that is what keeps edge
        // identity order meaning insertion order under any node order, and
        // what lets the counting passes below rebuild every run in order.
        let edge_map = surviving_slots(&self.live_edges, self.edge_bound());
        // The counting passes below place every run in identity order, so the
        // runs that read otherwise are taken now, while the old index is
        // still there to read them from.
        let reordered_out =
            self.reordered_adjacency(TraversalDirection::Outgoing, &node_map, &edge_map);
        let reordered_in =
            self.reordered_adjacency(TraversalDirection::Incoming, &node_map, &edge_map);
        let mut nodes = self.take_live_nodes(&node_map);
        if reorders {
            place(&mut nodes, &node_map);
        }
        let edges = self.take_live_edges(&node_map);

        let (out_offsets, mut out_edges) =
            compress(&edges, nodes.len(), TraversalDirection::Outgoing);
        let (in_offsets, mut in_edges) =
            compress(&edges, nodes.len(), TraversalDirection::Incoming);
        restore_runs(&out_offsets, &mut out_edges, reordered_out);
        restore_runs(&in_offsets, &mut in_edges, reordered_in);

        self.live_node_count = nodes.len();
        self.live_edge_count = edges.len();
        self.live_nodes.reset_all_live(nodes.len());
        self.live_edges.reset_all_live(edges.len());
        self.chains.reset();
        self.relocations_clear();
        self.base_nodes = nodes.into_boxed_slice();
        self.base_edges = edges.into_boxed_slice();
        self.out_offsets = out_offsets.into_boxed_slice();
        self.out_edges = out_edges.into_boxed_slice();
        self.in_offsets = in_offsets.into_boxed_slice();
        self.in_edges = in_edges.into_boxed_slice();

        Renumbering::new(node_map, edge_map)
    }

    /// Consume the store, returning its compacted form and the new numbering.
    ///
    /// The owning counterpart of [`compact`](Self::compact), for a pipeline
    /// stage that hands the store on rather than keeping it.
    #[must_use]
    pub fn compacted(mut self) -> (Self, Renumbering<NT, ET>) {
        let renumbering = self.compact();
        (self, renumbering)
    }

    fn take_live_nodes(&mut self, node_map: &[u32]) -> Vec<N> {
        let base = mem::take(&mut self.base_nodes);
        let delta = mem::take(&mut self.delta_nodes);
        let mut nodes = Vec::with_capacity(self.live_node_count);
        for (slot, payload) in Vec::from(base).into_iter().chain(delta).enumerate() {
            if node_map[slot] != NONE {
                nodes.push(payload);
            }
        }
        nodes
    }

    fn take_live_edges(&mut self, node_map: &[u32]) -> Vec<EdgeRecord<E, NT>> {
        let base = mem::take(&mut self.base_edges);
        let delta = mem::take(&mut self.delta_edges);
        let mut edges = Vec::with_capacity(self.live_edge_count);
        for (slot, mut record) in Vec::from(base).into_iter().chain(delta).enumerate() {
            if !self.live_edges.is_live(slot) {
                continue;
            }
            record.source = remap_endpoint(node_map, record.source);
            record.target = remap_endpoint(node_map, record.target);
            edges.push(record);
        }
        edges
    }
}

/// The new dense number of every live slot, with [`NONE`] for the removed.
fn surviving_slots(live: &LiveSet, slots: usize) -> Vec<u32> {
    let mut mapping = vec![NONE; slots];
    let mut next = 0;
    for (slot, entry) in mapping.iter_mut().enumerate() {
        if live.is_live(slot) {
            *entry = next;
            next += 1;
        }
    }
    mapping
}

/// Move each of `nodes`, the live payloads in slot order, to the number
/// `node_map` gives its slot.
///
/// Following each cycle of the permutation, every swap puts one payload in
/// its final place, so placing takes at most one swap per payload and stages
/// no payload outside the vector.
fn place<N>(nodes: &mut [N], node_map: &[u32]) {
    // `targets[i]` is the new number of the payload now at `i`; a swap moves
    // the targets with the payloads to keep that true.
    let mut targets: Vec<u32> = node_map
        .iter()
        .copied()
        .filter(|&new| new != NONE)
        .collect();
    for position in 0..nodes.len() {
        loop {
            let target = targets[position] as usize;
            if target == position {
                break;
            }
            nodes.swap(position, target);
            targets.swap(position, target);
        }
    }
}

/// Overwrite the runs of the nodes in `reordered` with the order they read in.
///
/// A run holds the same edges whatever their order, so each replacement fills
/// exactly the slice the counting pass gave its node.
fn restore_runs(offsets: &[u32], edges: &mut [u32], reordered: Vec<(u32, Vec<u32>)>) {
    for (node, run) in reordered {
        let start = offsets[node as usize] as usize;
        edges[start..start + run.len()].copy_from_slice(&run);
    }
}

fn remap_endpoint<NT: IdTag>(node_map: &[u32], old: Id<NT>) -> Id<NT> {
    let new = node_map[old.index()];
    debug_assert_ne!(
        new, NONE,
        "a live edge always has live endpoints: removing a node removes its edges"
    );
    Id::from_raw(new)
}
