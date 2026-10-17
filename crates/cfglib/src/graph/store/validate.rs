//! Reading a store from its serialized form without trusting it.
//!
//! Every column of the store is encoded as it sits in memory, and the walks
//! over it index those columns by each other: an out-of-range edge number, a
//! chain that never reaches its last link, or a live count that disagrees with
//! the bitset would abort a later read, or never end. A document is therefore
//! decoded into [`Stored`], the same columns with no behavior, and becomes a
//! [`Graph`] only once it has been checked: first that every read stays inside
//! the arrays and terminates, then, by walking the finished store, that the
//! adjacency it reads is the one its edge records describe.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use super::Graph;
use super::adjacency::{DeltaChains, MAX_SLOTS, check_index};
use super::edge::EdgeRecord;
use super::id::IdTag;
use super::liveness::LiveSet;
use super::relocate::Encoded;

/// The columns of a [`Graph`], exactly as they are written.
///
/// A field here without its counterpart in `Graph` (or the reverse) does not
/// compile, because [`TryFrom`] below names every one.
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "N: serde::Deserialize<'de>, E: serde::Deserialize<'de>"))]
pub(super) struct Stored<N, E, NT: IdTag> {
    base_nodes: Box<[N]>,
    base_edges: Box<[EdgeRecord<E, NT>]>,
    out_offsets: Box<[u32]>,
    out_edges: Box<[u32]>,
    in_offsets: Box<[u32]>,
    in_edges: Box<[u32]>,
    delta_nodes: Vec<N>,
    delta_edges: Vec<EdgeRecord<E, NT>>,
    chains: DeltaChains,
    live_nodes: LiveSet,
    live_edges: LiveSet,
    live_node_count: usize,
    live_edge_count: usize,
    #[serde(default)]
    relocations: Option<Box<Encoded>>,
}

impl<N, E, NT: IdTag, ET: IdTag> TryFrom<Stored<N, E, NT>> for Graph<N, E, NT, ET> {
    type Error = &'static str;

    fn try_from(stored: Stored<N, E, NT>) -> Result<Self, Self::Error> {
        let Stored {
            base_nodes,
            base_edges,
            out_offsets,
            out_edges,
            in_offsets,
            in_edges,
            delta_nodes,
            delta_edges,
            chains,
            live_nodes,
            live_edges,
            live_node_count,
            live_edge_count,
            relocations,
        } = stored;

        let node_bound = base_nodes.len() + delta_nodes.len();
        let edge_bound = base_edges.len() + delta_edges.len();
        if node_bound > MAX_SLOTS || edge_bound > MAX_SLOTS {
            return Err("the store holds more slots than the dense identity space");
        }
        if live_nodes.count_live(node_bound) != Some(live_node_count)
            || live_edges.count_live(edge_bound) != Some(live_edge_count)
        {
            return Err("the liveness bitsets disagree with the slots or the live counts");
        }
        let records = || base_edges.iter().chain(&delta_edges);
        if records().any(|record| {
            record.source().index() >= node_bound || record.target().index() >= node_bound
        }) {
            return Err("an edge names a node the store does not hold");
        }
        check_index(&out_offsets, &out_edges, base_nodes.len(), base_edges.len())?;
        check_index(&in_offsets, &in_edges, base_nodes.len(), base_edges.len())?;
        chains.check(node_bound, delta_edges.len(), |edge, chain| {
            let record = &delta_edges[edge];
            match chain {
                0 => record.source().index(),
                _ => record.target().index(),
            }
        })?;
        let relocations = relocations
            .map(|encoded| {
                encoded
                    .into_relocations(node_bound, edge_bound)
                    .map(Box::new)
            })
            .transpose()?;

        let graph = Self {
            base_nodes,
            base_edges,
            out_offsets,
            out_edges,
            in_offsets,
            in_edges,
            delta_nodes,
            delta_edges,
            chains,
            live_nodes,
            live_edges,
            live_node_count,
            live_edge_count,
            relocations,
            edge_tag: PhantomData,
        };
        graph.check_adjacency()?;
        Ok(graph)
    }
}

impl<N, E, NT: IdTag, ET: IdTag> Graph<N, E, NT, ET> {
    /// Check that walking every live node reads each live edge exactly once
    /// from each endpoint, which is the mirrored adjacency the views promise.
    ///
    /// Safe to run on a store the structural checks accepted: every walk stays
    /// inside the arrays and ends.
    fn check_adjacency(&self) -> Result<(), &'static str> {
        let mut outgoing = vec![false; self.edge_bound()];
        let mut incoming = vec![false; self.edge_bound()];
        for node in self.node_ids() {
            for edge in self.outgoing(node) {
                let listed = &mut outgoing[edge.index()];
                if self.edge(edge).source() != node || core::mem::replace(listed, true) {
                    return Err("a node lists an edge that does not leave it, or lists it twice");
                }
            }
            for edge in self.incoming(node) {
                let listed = &mut incoming[edge.index()];
                if self.edge(edge).target() != node || core::mem::replace(listed, true) {
                    return Err("a node lists an edge that does not enter it, or lists it twice");
                }
            }
        }
        for edge in self.edge_ids() {
            if !outgoing[edge.index()] || !incoming[edge.index()] {
                return Err("a live edge is missing from its endpoints' adjacency");
            }
        }
        Ok(())
    }
}
