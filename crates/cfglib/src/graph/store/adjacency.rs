//! Adjacency over a compressed base and an intrusive incremental delta.
//!
//! A node's neighbors live in two places. The base is a classic compressed
//! sparse row index: `offsets[i]..offsets[i + 1]` names a contiguous run of
//! edge identities in a single flat array. Everything added since the last
//! compaction lives in a per-node intrusive chain threaded through one `u32`
//! per delta edge.
//!
//! The chain is circular and the per-node slot holds its **last** element, so
//! the first element is one hop away (`next[last]`). That is what buys
//! constant-time append *and* append-order iteration from a single `u32` per
//! node per direction; a plain head-insertion list would cost the same memory
//! but reverse the order, and a head-plus-tail list would double it.
//!
//! The two directions share one array apiece, holding a `u32` pair per node
//! and per delta edge, because every append joins both chains at once: the
//! pair is what one node or one edge costs, so appending either is a single
//! push rather than one per direction.
//!
//! The per-node array is materialized lazily: it exists only while the delta
//! holds an edge, because a node with nothing appended to it has nothing to
//! say. A compacted store therefore carries no per-node array at all, rather
//! than one all-[`NONE`] pair per node of a graph whose adjacency is entirely
//! in the base — and that is the state a store spends most of its life in.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::graph::traverse::TraversalDirection;

use super::edge::EdgeRecord;
use super::id::{Id, IdTag};
use super::liveness::LiveSet;
use super::{Graph, raw_index};

/// Absence of a link. One value of the dense space is spent so that a chain
/// slot costs four bytes instead of an eight-byte `Option<u32>`.
pub(super) const NONE: u32 = u32::MAX;

/// The largest number of slots a dense store can address.
pub(super) const MAX_SLOTS: usize = NONE as usize;

/// One chain link per direction, in [`chain_of`] order.
type Links = [u32; 2];

/// A pair of chain links belonging to no chain, which is where a node starts.
const UNLINKED: Links = [NONE; 2];

/// Which half of a [`Links`] pair one direction threads.
const fn chain_of(direction: TraversalDirection) -> usize {
    match direction {
        TraversalDirection::Outgoing => 0,
        TraversalDirection::Incoming => 1,
    }
}

/// Per-node circular chains threading both directions of the delta's
/// adjacency.
///
/// [`append`](Self::append) is the only operation that grows either array,
/// and [`reset`](Self::reset) releases both. Three invariants follow, and
/// every reader depends on them: `last.len()` is at most the store's node
/// bound, an absent slot means [`UNLINKED`], and a non-empty `last` implies a
/// non-empty `next`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct DeltaChains {
    /// Last delta edge appended to each materialized node slot's two chains,
    /// or [`NONE`].
    last: Vec<Links>,
    /// Successor of each delta edge within each of its two circular chains.
    next: Vec<Links>,
}

impl DeltaChains {
    pub(super) const fn new() -> Self {
        Self {
            last: Vec::new(),
            next: Vec::new(),
        }
    }

    pub(super) fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            last: Vec::with_capacity(nodes),
            next: Vec::with_capacity(edges),
        }
    }

    /// How many node slots the chains have materialized, which is the whole
    /// of what the delta costs per node.
    #[cfg(test)]
    pub(super) fn node_slots(&self) -> usize {
        self.last.len()
    }

    /// Append the next delta edge to its source's outgoing chain and to its
    /// target's incoming chain.
    ///
    /// One edge joins both chains, so its links are one record and the append
    /// is one push.
    ///
    /// `nodes` is the store's node bound. The first append after a reset
    /// materializes the per-node array to it in one step, and every later one
    /// finds both endpoints already covered — so the threading itself never
    /// asks, and the array is absent exactly while the delta is empty.
    pub(super) fn append(&mut self, source: usize, target: usize, nodes: usize) {
        if self.last.len() < nodes {
            self.materialize(nodes);
        }
        let appended = u32::try_from(self.next.len()).expect("delta edge index exceeds u32::MAX");
        self.next.push(UNLINKED);
        self.thread(source, chain_of(TraversalDirection::Outgoing), appended);
        self.thread(target, chain_of(TraversalDirection::Incoming), appended);
    }

    /// Give every node slot two empty chains, which is what the first append
    /// after a reset owes the nodes the base already describes.
    ///
    /// Out of line because it runs once per delta, against an append path
    /// that runs once per edge.
    #[cold]
    fn materialize(&mut self, nodes: usize) {
        self.last.resize(nodes, UNLINKED);
    }

    /// Splice `appended` in as the last element of one of `node`'s chains,
    /// whose slots [`append`](Self::append) has already materialized.
    fn thread(&mut self, node: usize, chain: usize, appended: u32) {
        let last = self.last[node][chain];
        self.next[appended as usize][chain] = if last == NONE {
            appended
        } else {
            let first = self.next[last as usize][chain];
            self.next[last as usize][chain] = appended;
            first
        };
        self.last[node][chain] = appended;
    }

    /// The first and last delta edges of one of `node`'s chains, when it has
    /// one.
    ///
    /// A node the delta has never touched has no slot at all, which is how a
    /// walk over a compacted store answers from the array's length instead of
    /// reading one per-node entry that is entirely [`NONE`].
    fn ends(&self, node: usize, chain: usize) -> Option<(u32, u32)> {
        let last = self.last.get(node)?[chain];
        if last == NONE {
            return None;
        }
        Some((self.next[last as usize][chain], last))
    }

    /// Forget one of a node's chains, whose edges an overlay has taken over.
    ///
    /// A node the delta has never touched has no chain to forget.
    pub(super) fn detach_node(&mut self, node: usize, direction: TraversalDirection) {
        if let Some(links) = self.last.get_mut(node) {
            links[chain_of(direction)] = NONE;
        }
    }

    /// Drop every chain.
    ///
    /// Both arrays are released rather than cleared: after a compaction the
    /// delta is empty, and holding two `u32` per edge of the graph that was
    /// just folded into the base — or eight bytes per node of it — would
    /// quietly keep a large part of the store's memory alive for nothing.
    pub(super) fn reset(&mut self) {
        self.last = Vec::new();
        self.next = Vec::new();
    }
}

/// Position of a partly consumed adjacency walk.
///
/// The walk is a value rather than a borrow so that a caller which mutates
/// the store between steps — [`Graph::remove_node`] clearing a node's edges —
/// shares one stepping routine with the iterators.
#[derive(Debug, Clone, Copy)]
struct AdjacencyCursor {
    base: u32,
    base_end: u32,
    delta: u32,
    delta_last: u32,
}

impl AdjacencyCursor {
    const EXHAUSTED: Self = Self {
        base: 0,
        base_end: 0,
        delta: NONE,
        delta_last: NONE,
    };
}

/// One direction's adjacency arrays, resolved once for a whole walk.
///
/// Resolving them per step is what the obvious implementation does, and it
/// costs a branch and three pointer loads on every edge of every scan.
#[derive(Clone, Copy)]
struct Axis<'g> {
    offsets: &'g [u32],
    edges: &'g [u32],
    chains: &'g DeltaChains,
    /// Which half of each chain record this axis threads.
    chain: usize,
}

/// Advance one walk to the next live edge, in insertion order.
///
/// Base identities come first in ascending order, then delta identities in
/// append order. Every base identity is below every delta identity, so the
/// sequence is exactly the order in which the edges were added.
///
/// `live` is `None` for a store with no removed edges at all, which spares
/// every scan of a freshly built or freshly compacted store one bitset probe
/// per edge — and that is the store most analyses run on.
///
/// The inline attribute is load-bearing rather than decorative. This is the
/// inner loop of every algorithm in the crate, and it sits right at the
/// inliner's own threshold: leaving the decision open costs a whole-codebase
/// successor scan about twice its time as soon as the body grows, which is
/// the cliff the [relocation module](super::relocate) describes from the
/// other side.
#[inline]
fn advance(
    cursor: &mut AdjacencyCursor,
    axis: &Axis<'_>,
    live: Option<&LiveSet>,
    delta_base: u32,
) -> Option<u32> {
    while cursor.base < cursor.base_end {
        let edge = axis.edges[cursor.base as usize];
        cursor.base += 1;
        if live.is_none_or(|set| set.is_live(edge as usize)) {
            return Some(edge);
        }
    }

    while cursor.delta != NONE {
        let delta = cursor.delta;
        cursor.delta = if delta == cursor.delta_last {
            NONE
        } else {
            axis.chains.next[delta as usize][axis.chain]
        };
        let edge = delta_base + delta;
        if live.is_none_or(|set| set.is_live(edge as usize)) {
            return Some(edge);
        }
    }

    None
}

impl<N, E, NT: IdTag, ET: IdTag> Graph<N, E, NT, ET> {
    fn axis(&self, direction: TraversalDirection) -> Axis<'_> {
        let (offsets, edges) = match direction {
            TraversalDirection::Outgoing => (&self.out_offsets, &self.out_edges),
            TraversalDirection::Incoming => (&self.in_offsets, &self.in_edges),
        };
        Axis {
            offsets,
            edges,
            chains: &self.chains,
            chain: chain_of(direction),
        }
    }

    /// Start a walk over one node's adjacency along `axis`.
    ///
    /// # Panics
    ///
    /// Panics when `node` names no slot in this store.
    fn cursor(&self, node: Id<NT>, axis: &Axis<'_>) -> AdjacencyCursor {
        let index = node.index();
        assert!(index < self.node_bound(), "node identity is out of range");
        if self.tombstoned_nodes() && !self.live_nodes.is_live(index) {
            return AdjacencyCursor::EXHAUSTED;
        }
        let (base, base_end) = if index + 1 < axis.offsets.len() {
            (axis.offsets[index], axis.offsets[index + 1])
        } else {
            (0, 0)
        };
        let (delta, delta_last) = axis.chains.ends(index, axis.chain).unwrap_or((NONE, NONE));
        AdjacencyCursor {
            base,
            base_end,
            delta,
            delta_last,
        }
    }

    fn adjacent(&self, node: Id<NT>, direction: TraversalDirection) -> AdjacentEdges<'_, ET> {
        let mut axis = self.axis(direction);
        let mut cursor = self.cursor(node, &axis);
        // A node whose edges have moved reads its compressed run from a
        // replacement instead; its chain still carries whatever arrived
        // after the move. The relocation module states why this is a whole
        // run rather than a filter and a tail.
        if let Some(run) = self.overlay(node.index(), direction) {
            axis.edges = run;
            cursor.base = 0;
            cursor.base_end = raw_index(run.len());
        }
        AdjacentEdges {
            cursor,
            axis,
            live: self.tombstoned_edges().then_some(&self.live_edges),
            delta_base: raw_index(self.base_edges.len()),
            tag: PhantomData,
        }
    }

    /// Iterate over the live outgoing edges of `node` in insertion order.
    ///
    /// # Panics
    ///
    /// Panics when `node` names no slot in this store.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn outgoing(&self, node: Id<NT>) -> AdjacentEdges<'_, ET> {
        self.adjacent(node, TraversalDirection::Outgoing)
    }

    /// Iterate over the live incoming edges of `node` in insertion order.
    ///
    /// # Panics
    ///
    /// Panics when `node` names no slot in this store.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn incoming(&self, node: Id<NT>) -> AdjacentEdges<'_, ET> {
        self.adjacent(node, TraversalDirection::Incoming)
    }

    /// Iterate over outgoing neighbors, retaining parallel entries.
    ///
    /// # Panics
    ///
    /// Panics when `node` names no slot in this store.
    pub fn successors(&self, node: Id<NT>) -> impl Iterator<Item = Id<NT>> + '_ {
        self.outgoing(node).map(|edge| self.edge(edge).target())
    }

    /// Iterate over incoming neighbors, retaining parallel entries.
    ///
    /// # Panics
    ///
    /// Panics when `node` names no slot in this store.
    pub fn predecessors(&self, node: Id<NT>) -> impl Iterator<Item = Id<NT>> + '_ {
        self.incoming(node).map(|edge| self.edge(edge).source())
    }

    /// Remove every live edge reachable from `node` in `direction`.
    ///
    /// The walk is collected first because the loop body needs the store
    /// mutably. That is the removal path, not a scan, so the buffer never
    /// reaches an analysis.
    pub(super) fn clear_adjacency(&mut self, node: Id<NT>, direction: TraversalDirection) {
        let incident: Vec<Id<ET>> = self.adjacent(node, direction).collect();
        for edge in incident {
            self.retire_edge(edge.index());
        }
    }
}

/// Live edges adjacent to one node, in insertion order.
///
/// Returned by [`Graph::outgoing`] and [`Graph::incoming`]. It is not an
/// [`ExactSizeIterator`]: the length of a delta chain containing tombstones
/// is only known by walking it.
#[derive(Clone)]
pub struct AdjacentEdges<'g, ET: IdTag> {
    axis: Axis<'g>,
    live: Option<&'g LiveSet>,
    cursor: AdjacencyCursor,
    delta_base: u32,
    tag: PhantomData<fn() -> ET>,
}

impl<ET: IdTag> core::fmt::Debug for AdjacentEdges<'_, ET> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("AdjacentEdges")
            .field("cursor", &self.cursor)
            .finish_non_exhaustive()
    }
}

impl<ET: IdTag> Iterator for AdjacentEdges<'_, ET> {
    type Item = Id<ET>;

    /// Inlined for the reason the walk it steps is.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        advance(&mut self.cursor, &self.axis, self.live, self.delta_base).map(Id::from_raw)
    }
}

/// Build one direction's compressed adjacency index over compacted edges.
///
/// One counting pass, then a stable placement pass, so the edges inside every
/// node's run keep their insertion order.
pub(super) fn compress<E, NT: IdTag>(
    edges: &[EdgeRecord<E, NT>],
    nodes: usize,
    direction: TraversalDirection,
) -> (Vec<u32>, Vec<u32>) {
    let endpoint = |edge: &EdgeRecord<E, NT>| match direction {
        TraversalDirection::Outgoing => edge.source.index(),
        TraversalDirection::Incoming => edge.target.index(),
    };

    let mut offsets = vec![0_u32; nodes + 1];
    for edge in edges {
        offsets[endpoint(edge) + 1] += 1;
    }
    for index in 1..offsets.len() {
        offsets[index] += offsets[index - 1];
    }

    let mut next = offsets[..nodes].to_vec();
    let mut adjacency = vec![0_u32; edges.len()];
    for (index, edge) in edges.iter().enumerate() {
        let slot = &mut next[endpoint(edge)];
        adjacency[*slot as usize] =
            u32::try_from(index).expect("compacted edge index exceeds u32::MAX");
        *slot += 1;
    }

    (offsets, adjacency)
}
