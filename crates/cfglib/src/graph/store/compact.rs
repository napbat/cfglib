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

use crate::graph::view::Direction;

use super::Graph;
use super::adjacency::compress;
use super::edge::EdgeRecord;
use super::id::{Id, IdTag};
use super::liveness::LiveSet;
use super::node_order::NodeOrder;
use super::renumbering::Renumbering;
use crate::identity::NO_INDEX;

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

    /// Rebuild the compressed base from the live entities, numbering the
    /// surviving nodes in `order`, and report the new numbering.
    ///
    /// The delta is emptied and every tombstone disappears. Of the surviving
    /// identities, only node identities can change their relative order;
    /// afterwards:
    ///
    /// - node identity order is `order`: under
    ///   [`Insertion`](NodeOrder::Insertion) it still means insertion order,
    ///   and under any other order it no longer does; a node added later
    ///   still follows every compacted one;
    /// - edges keep their relative order, so edge identity order still means
    ///   insertion order and the edge renumbering is the same under every
    ///   order;
    /// - every node keeps its adjacency order: its edges read in the order
    ///   they read before, under their new identities.
    ///
    /// [`Insertion`](NodeOrder::Insertion) keeps the order the nodes were
    /// added in, which for a whole-program graph is discovery or file order:
    /// a node's neighbors then carry scattered identities, and so do the
    /// payloads, adjacency offsets and node-indexed side tables a scan reads
    /// for them. An order that numbers neighbors close together turns those
    /// reads into reads of nearby memory. A procedure-sized graph fits in
    /// cache either way and has nothing to gain. Every returned
    /// [`Renumbering`] [`compose`](Renumbering::compose)s with any other.
    ///
    /// [`Insertion`](NodeOrder::Insertion) costs one pass over each slot
    /// space plus two counting passes over the live edges, and allocates the
    /// new base once. A node whose adjacency reads out of identity order,
    /// because an edge was redirected to or from it, additionally costs one
    /// walk of that adjacency. Every other order adds its own cost: one
    /// search over the reached nodes and their edges for
    /// [`BreadthFirst`](NodeOrder::BreadthFirst) and
    /// [`ReversePostorder`](NodeOrder::ReversePostorder), and one pass over
    /// the order for [`Explicit`](NodeOrder::Explicit); the payloads then
    /// move into their new places by swaps, one per payload at most. On a
    /// million-node graph with shuffled identities, a search compaction costs
    /// several times an insertion-order one (`benches/graph-store.rs`), and
    /// compacting the store first does not remove that cost.
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
    /// let dropped = graph.add_node("dropped");
    /// let kept = graph.add_node("kept");
    /// graph.remove_node(dropped);
    ///
    /// let renumbering = graph.compact(NodeOrder::Insertion);
    /// assert!(graph.is_compact());
    /// assert_eq!(renumbering.node(dropped), None);
    /// assert_eq!(graph.node(renumbering.node(kept).unwrap()), &"kept");
    /// ```
    ///
    /// Reverse postorder from the entry numbers the nodes in a forward
    /// solver's iteration order:
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
    /// let renumbering = graph.compact(NodeOrder::ReversePostorder(&[entry]));
    /// assert!(graph.is_compact());
    /// assert_eq!(graph.nodes().copied().collect::<Vec<_>>(), ["entry", "body", "exit"]);
    ///
    /// let body = renumbering.node(body).unwrap();
    /// let exit = renumbering.node(exit).unwrap();
    /// assert_eq!(graph.successors(body).collect::<Vec<_>>(), [body, exit]);
    /// ```
    pub fn compact(&mut self, order: NodeOrder<'_, NT>) -> Renumbering<NT, ET> {
        let reorders = !matches!(order, NodeOrder::Insertion);
        let node_map = if reorders {
            self.ordered_slots(order)
        } else {
            surviving_slots(&self.live_nodes, self.node_bound())
        };
        self.rebuild(node_map, reorders)
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
        let reordered_out = self.reordered_adjacency(Direction::Forward, &node_map, &edge_map);
        let reordered_in = self.reordered_adjacency(Direction::Backward, &node_map, &edge_map);
        let mut nodes = self.take_live_nodes(&node_map);
        if reorders {
            place(&mut nodes, &node_map);
        }
        let edges = self.take_live_edges(&node_map);

        let (out_offsets, mut out_edges) = compress(&edges, nodes.len(), Direction::Forward);
        let (in_offsets, mut in_edges) = compress(&edges, nodes.len(), Direction::Backward);
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

    fn take_live_nodes(&mut self, node_map: &[u32]) -> Vec<N> {
        let base = mem::take(&mut self.base_nodes);
        let delta = mem::take(&mut self.delta_nodes);
        let mut nodes = Vec::with_capacity(self.live_node_count);
        for (slot, payload) in Vec::from(base).into_iter().chain(delta).enumerate() {
            if node_map[slot] != NO_INDEX {
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

/// The new dense number of every live slot, with [`NO_INDEX`] for the removed.
fn surviving_slots(live: &LiveSet, slots: usize) -> Vec<u32> {
    let mut mapping = vec![NO_INDEX; slots];
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
        .filter(|&new| new != NO_INDEX)
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
        new, NO_INDEX,
        "a live edge always has live endpoints: removing a node removes its edges"
    );
    Id::from_raw(new)
}
