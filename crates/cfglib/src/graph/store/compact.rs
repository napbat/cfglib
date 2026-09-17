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

use super::adjacency::{NONE, compress};
use super::edge::EdgeRecord;
use super::id::{Id, IdTag};
use super::liveness::LiveSet;
use super::renumbering::Renumbering;
use super::{Graph, raw_index};

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
    }

    /// Rebuild the compressed base from the live entities and report the new
    /// numbering.
    ///
    /// Live entities keep their relative order, so the surviving identities
    /// are renumbered but never reordered: insertion order, adjacency order,
    /// and identity order all mean the same thing afterwards as before. The
    /// delta is emptied and every tombstone disappears.
    ///
    /// Costs one pass over the slot space plus two counting passes over the
    /// live edges, and allocates the new base once.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::graph::store::Graph;
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
        let node_map = surviving_slots(&self.live_nodes, self.node_slot_count());
        let nodes = self.take_live_nodes(&node_map);
        let (edges, edge_map) = self.take_live_edges(&node_map);

        let (out_offsets, out_edges) = compress(&edges, nodes.len(), TraversalDirection::Outgoing);
        let (in_offsets, in_edges) = compress(&edges, nodes.len(), TraversalDirection::Incoming);

        self.live_node_count = nodes.len();
        self.live_edge_count = edges.len();
        self.live_nodes.reset_all_live(nodes.len());
        self.live_edges.reset_all_live(edges.len());
        self.out_chains.reset(nodes.len());
        self.in_chains.reset(nodes.len());
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

    fn take_live_edges(&mut self, node_map: &[u32]) -> (Vec<EdgeRecord<E, NT>>, Vec<u32>) {
        let base = mem::take(&mut self.base_edges);
        let delta = mem::take(&mut self.delta_edges);
        let mut edge_map = vec![NONE; base.len() + delta.len()];
        let mut edges = Vec::with_capacity(self.live_edge_count);
        for (slot, mut record) in Vec::from(base).into_iter().chain(delta).enumerate() {
            if !self.live_edges.is_live(slot) {
                continue;
            }
            record.source = remap_endpoint(node_map, record.source);
            record.target = remap_endpoint(node_map, record.target);
            edge_map[slot] = raw_index(edges.len());
            edges.push(record);
        }
        (edges, edge_map)
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

fn remap_endpoint<NT: IdTag>(node_map: &[u32], old: Id<NT>) -> Id<NT> {
    let new = node_map[old.index()];
    debug_assert_ne!(
        new, NONE,
        "a live edge always has live endpoints: removing a node removes its edges"
    );
    Id::from_raw(new)
}
