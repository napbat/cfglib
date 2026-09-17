//! Moving one endpoint of an edge without changing the edge's identity.
//!
//! A control-flow transform that bypasses a block, merges a linear chain, or
//! splits one keeps the edges it did not create: their kind, weight, and
//! consumer payload survive, and so does the identity a caller is holding.
//! That is why [`Rewrite`](crate::Rewrite) spells "redirected" as
//! `old -> [old]` rather than `old -> [new]`.
//!
//! The compressed base cannot express the move in place — an edge's position
//! in a node's compressed run *is* the index, and shifting it would shift
//! every later run.
//!
//! # Why an overlay rather than a filter
//!
//! The obvious repair is a per-edge "has this edge moved?" bit beside the
//! liveness bit, plus a list of arrivals after the indexed run. Both are
//! wrong here, and measurably so: the adjacency walk's `next` is a handful of
//! instructions that a scanning loop inlines whole, and *any* addition to it
//! — a second bitset probe, a second segment, even one `Option` test after
//! the loop — stops that inlining and costs a whole-codebase successor scan
//! about 2.5x. The walk is the hot path of every algorithm in the crate, so
//! it does not change.
//!
//! Instead, a node whose adjacency has changed is served from an **overlay**:
//! one contiguous run that replaces its compressed run, holding the survivors
//! in their original order followed by the arrivals in move order. The walk
//! reads it as an ordinary compressed run and cannot tell the difference;
//! choosing between the index and the overlay happens once per node.
//!
//! The node's incremental chain keeps working beside the overlay, so an edge
//! added *after* the move needs no special handling and the append path stays
//! exactly as fast as it was. Taking the overlay therefore also takes over
//! the chain the node had at that moment.
//!
//! # What it costs
//!
//! A store that never relocates carries one `Option` pointer and no per-edge
//! work at all. A store that does pays one `Vec` per node whose adjacency
//! changed, materialized at that node's degree, and the next
//! [`compact`](super::Graph::compact) folds every overlay back into the
//! index.

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::graph::traverse::TraversalDirection;

use super::Graph;
use super::id::{Id, IdTag};
use super::raw_index;

/// Replacement adjacency runs for the nodes whose edges have moved.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(super) struct Relocations {
    /// Replacement outgoing runs, keyed by node.
    outgoing: BTreeMap<u32, Vec<u32>>,
    /// Replacement incoming runs, keyed by node.
    incoming: BTreeMap<u32, Vec<u32>>,
}

impl Relocations {
    fn runs(&self, direction: TraversalDirection) -> &BTreeMap<u32, Vec<u32>> {
        match direction {
            TraversalDirection::Outgoing => &self.outgoing,
            TraversalDirection::Incoming => &self.incoming,
        }
    }

    fn runs_mut(&mut self, direction: TraversalDirection) -> &mut BTreeMap<u32, Vec<u32>> {
        match direction {
            TraversalDirection::Outgoing => &mut self.outgoing,
            TraversalDirection::Incoming => &mut self.incoming,
        }
    }
}

impl<N, E, NT: IdTag, ET: IdTag> Graph<N, E, NT, ET> {
    /// The replacement adjacency run for `node`, when it has one.
    ///
    /// One `Option` test per node for a store that has never relocated an
    /// edge, and nothing at all inside the walk.
    #[inline]
    pub(super) fn overlay(&self, node: usize, direction: TraversalDirection) -> Option<&[u32]> {
        let relocations = self.relocations.as_deref()?;
        relocations
            .runs(direction)
            .get(&raw_index(node))
            .map(Vec::as_slice)
    }

    /// Forget every overlay, which is what a rebuilt index makes true.
    pub(super) fn relocations_clear(&mut self) {
        self.relocations = None;
    }

    /// Whether any edge has been redirected since the last compaction.
    pub(super) fn has_relocations(&self) -> bool {
        self.relocations.as_deref().is_some_and(|relocations| {
            !relocations.outgoing.is_empty() || !relocations.incoming.is_empty()
        })
    }

    /// Move one edge's source, retaining its identity, kind, and payload.
    ///
    /// Returns the previous source. Adjacency order at the new source places
    /// the relocated edge after the edges already there, which is the order a
    /// caller moving a whole block's outgoing edges expects.
    ///
    /// # Panics
    ///
    /// Panics when the edge has been removed, or when `source` names no live
    /// node in this store.
    pub fn redirect_edge_source(&mut self, edge: Id<ET>, source: Id<NT>) -> Id<NT> {
        self.redirect(edge, source, TraversalDirection::Outgoing)
    }

    /// Move one edge's target, retaining its identity, kind, and payload.
    ///
    /// Returns the previous target.
    ///
    /// # Panics
    ///
    /// Panics when the edge has been removed, or when `target` names no live
    /// node in this store.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::Graph;
    ///
    /// let mut graph = Graph::new();
    /// let source = graph.add_node("source");
    /// let first = graph.add_node("first");
    /// let second = graph.add_node("second");
    /// let edge = graph.add_edge(source, first, "call");
    ///
    /// assert_eq!(graph.redirect_edge_target(edge, second), first);
    /// assert_eq!(graph.successors(source).collect::<Vec<_>>(), [second]);
    /// assert_eq!(graph.edge(edge).payload(), &"call", "the edge is the same edge");
    /// assert!(graph.incoming(first).next().is_none());
    /// ```
    pub fn redirect_edge_target(&mut self, edge: Id<ET>, target: Id<NT>) -> Id<NT> {
        self.redirect(edge, target, TraversalDirection::Incoming)
    }

    fn redirect(
        &mut self,
        edge: Id<ET>,
        endpoint: Id<NT>,
        direction: TraversalDirection,
    ) -> Id<NT> {
        assert!(self.contains_edge(edge), "edge has been removed");
        self.assert_live_endpoint(endpoint, "replacement");
        let record = self.edge_mut(edge);
        let moved = match direction {
            TraversalDirection::Outgoing => &mut record.source,
            TraversalDirection::Incoming => &mut record.target,
        };
        let previous = *moved;
        if previous == endpoint {
            return previous;
        }
        *moved = endpoint;

        let raw = raw_index(edge.index());
        let leaving: Vec<u32> = self
            .materialize(previous, direction)
            .into_iter()
            .filter(|&slot| slot != raw)
            .collect();
        let mut arriving = self.materialize(endpoint, direction);
        arriving.push(raw);

        // The overlay takes over both nodes' chains as well as their runs.
        let previous_index = previous.index();
        let endpoint_index = endpoint.index();
        let chains = self.chains_mut(direction);
        chains.detach_node(previous_index);
        chains.detach_node(endpoint_index);
        let relocations = self.relocations.get_or_insert_default();
        let runs = relocations.runs_mut(direction);
        runs.insert(raw_index(previous.index()), leaving);
        runs.insert(raw_index(endpoint.index()), arriving);
        previous
    }

    /// The node's complete current adjacency along `direction`, which is what
    /// its overlay run has to reproduce.
    fn materialize(&self, node: Id<NT>, direction: TraversalDirection) -> Vec<u32> {
        if let Some(run) = self.overlay(node.index(), direction) {
            return run.to_vec();
        }
        let adjacent: Box<dyn Iterator<Item = Id<ET>> + '_> = match direction {
            TraversalDirection::Outgoing => Box::new(self.outgoing(node)),
            TraversalDirection::Incoming => Box::new(self.incoming(node)),
        };
        adjacent.map(|edge| raw_index(edge.index())).collect()
    }
}
