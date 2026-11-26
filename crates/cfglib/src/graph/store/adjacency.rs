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
//! than one all-[`NO_INDEX`] pair per node of a graph whose adjacency is entirely
//! in the base — and that is the state a store spends most of its life in.

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::graph::fanout::DenseFanout;
use crate::graph::view::Direction;
use crate::identity::{NO_INDEX, slot, word};

use super::Graph;
use super::edge::EdgeRecord;
use super::id::{Id, IdTag};
use super::liveness::LiveSet;

/// The largest number of slots a dense store can address. A link is a `u32`
/// with [`NO_INDEX`] for its absence, so a chain slot costs four bytes instead
/// of an eight-byte `Option<u32>`.
pub(super) const MAX_SLOTS: usize = slot(NO_INDEX);

/// One chain link per direction, in [`chain_of`] order.
type Links = [u32; 2];

/// A pair of chain links belonging to no chain, which is where a node starts.
const UNLINKED: Links = [NO_INDEX; 2];

/// Which half of a [`Links`] pair one direction threads.
const fn chain_of(direction: Direction) -> usize {
    match direction {
        Direction::Forward => 0,
        Direction::Backward => 1,
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
    /// or [`NO_INDEX`].
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
        let appended = word(self.next.len());
        self.next.push(UNLINKED);
        self.thread(source, chain_of(Direction::Forward), appended);
        self.thread(target, chain_of(Direction::Backward), appended);
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
        self.next[appended as usize][chain] = if last == NO_INDEX {
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
    /// reading one per-node entry that is entirely [`NO_INDEX`].
    fn ends(&self, node: usize, chain: usize) -> Option<(u32, u32)> {
        let last = self.last.get(node)?[chain];
        if last == NO_INDEX {
            return None;
        }
        Some((self.next[last as usize][chain], last))
    }

    /// Forget one of a node's chains, whose edges an overlay has taken over.
    ///
    /// A node the delta has never touched has no chain to forget.
    pub(super) fn detach_node(&mut self, node: usize, direction: Direction) {
        if let Some(links) = self.last.get_mut(node) {
            links[chain_of(direction)] = NO_INDEX;
        }
    }

    /// Move one of a node's chains onto the end of `run` in append order and
    /// forget it, dropping the edges `live` rejects.
    ///
    /// This is how an overlay run takes over the edges appended since it was
    /// built. `delta_base` is the store's base edge count, which turns a delta
    /// position into an edge identity exactly as the walk does.
    pub(super) fn drain_into(
        &mut self,
        node: usize,
        direction: Direction,
        delta_base: u32,
        live: Option<&LiveSet>,
        run: &mut Vec<u32>,
    ) {
        let chain = chain_of(direction);
        let Some((mut delta, last)) = self.ends(node, chain) else {
            return;
        };
        loop {
            let edge = delta_base + delta;
            if live.is_none_or(|set| set.is_live(edge as usize)) {
                run.push(edge);
            }
            if delta == last {
                break;
            }
            delta = self.next[delta as usize][chain];
        }
        self.detach_node(node, direction);
    }

    /// Check chains read from outside: that every walk over them reads inside
    /// the arrays and ends, and that each chain holds only edges whose
    /// endpoint it belongs to.
    ///
    /// `endpoint(edge, chain)` is the node a delta edge is threaded through in
    /// one of its two chains. An edge is in at most one chain per direction,
    /// so the walks cost one step per delta edge between them.
    #[cfg(feature = "serde")]
    pub(super) fn check(
        &self,
        nodes: usize,
        delta_edges: usize,
        endpoint: impl Fn(usize, usize) -> usize,
    ) -> Result<(), &'static str> {
        if self.last.len() > nodes || self.next.len() != delta_edges {
            return Err("the adjacency chains do not match the nodes and delta edges");
        }
        for chain in 0..2 {
            let mut seen = alloc::vec![false; self.next.len()];
            for (node, links) in self.last.iter().enumerate() {
                let last = links[chain];
                if last == NO_INDEX {
                    continue;
                }
                let mut edge = self
                    .next
                    .get(last as usize)
                    .ok_or("an adjacency chain ends outside the delta edges")?[chain];
                loop {
                    let slot = edge as usize;
                    if slot >= seen.len() || seen[slot] || endpoint(slot, chain) != node {
                        return Err("an adjacency chain is not a ring of the node's own edges");
                    }
                    seen[slot] = true;
                    if edge == last {
                        break;
                    }
                    edge = self.next[slot][chain];
                }
            }
        }
        Ok(())
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
        delta: NO_INDEX,
        delta_last: NO_INDEX,
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

/// Advance one walk to the next live edge, in adjacency order.
///
/// The compressed run comes first, then the delta identities in append order.
/// Every base identity is below every delta identity, and a run no edge was
/// redirected into is ascending, so for such a node the sequence is exactly
/// the order in which the edges were added.
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
/// other side. A hint is not enough either: whether the inliner takes it
/// depends on how many other walks the caller's crate instantiates next to a
/// scan, so code that adds walks elsewhere can leave the step out of line in
/// a scan it never touched.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "a hint leaves the step out of line once the caller's crate walks the graph from enough places"
)]
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

    while cursor.delta != NO_INDEX {
        let delta = cursor.delta;
        cursor.delta = if delta == cursor.delta_last {
            NO_INDEX
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
    fn axis(&self, direction: Direction) -> Axis<'_> {
        let (offsets, edges) = match direction {
            Direction::Forward => (&self.out_offsets, &self.out_edges),
            Direction::Backward => (&self.in_offsets, &self.in_edges),
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
    /// Forced inline for the reason [`adjacent`](Self::adjacent) is. The bound
    /// check is a plain `assert!` with a fixed message, not the walks'
    /// `assert_in_bound` with its formatted one: routed through that, as a
    /// call or inlined, `directed_interval_reverse_id_chain` measured 11% and
    /// `cfg_depth_first_preorder` 3 to 7% slower.
    ///
    /// # Panics
    ///
    /// Panics when `node` names no slot in this store.
    #[inline(always)]
    #[expect(
        clippy::inline_always,
        reason = "an out-of-line call keeps the walk in memory; see `adjacent`"
    )]
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
        let (delta, delta_last) = axis
            .chains
            .ends(index, axis.chain)
            .unwrap_or((NO_INDEX, NO_INDEX));
        AdjacencyCursor {
            base,
            base_end,
            delta,
            delta_last,
        }
    }

    /// Start a walk over `node`'s live edges along `direction`, in adjacency
    /// order.
    ///
    /// Forced inline, with [`cursor`](Self::cursor); the stepping layers over
    /// the walk are plain `#[inline]`. The hint is not enough here. A walk is
    /// built from 4- and 8-byte stores, and any copy of it between frames (a
    /// returned value moved into the consumer's loop variable) reads it back
    /// in 16-byte pieces. A load that spans several narrower stores cannot be forwarded,
    /// so every scan waits for its own start to retire: whether the optimizer
    /// elided the copy depended on the surrounding code, and where it did not
    /// a scan cost 15% to 50% more cycles at the same instruction count.
    /// Inlined, the walk is built where the loop that steps it is, and there
    /// is nothing to copy.
    #[inline(always)]
    #[expect(
        clippy::inline_always,
        reason = "a call leaves the walk to be copied out of a return slot, which stalls on store forwarding"
    )]
    pub(super) fn adjacent(&self, node: Id<NT>, direction: Direction) -> AdjacentEdges<'_, ET> {
        let mut axis = self.axis(direction);
        let mut cursor = self.cursor(node, &axis);
        // A node whose edges have moved reads its compressed run from a
        // replacement instead; its chain still carries whatever arrived
        // after the move. The relocation module states why this is a whole
        // run rather than a filter and a tail.
        if let Some(run) = self.overlay(node.index(), direction) {
            axis.edges = run;
            cursor.base = 0;
            cursor.base_end = word(run.len());
        }
        AdjacentEdges {
            cursor,
            axis,
            live: self.tombstoned_edges().then_some(&self.live_edges),
            delta_base: word(self.base_edges.len()),
            tag: PhantomData,
        }
    }

    /// `node`'s compressed run along `direction`, removed edges included.
    fn base_run(&self, node: usize, direction: Direction) -> &[u32] {
        let axis = self.axis(direction);
        if node + 1 < axis.offsets.len() {
            &axis.edges[axis.offsets[node] as usize..axis.offsets[node + 1] as usize]
        } else {
            &[]
        }
    }

    /// The live edges of `node`'s compressed run along `direction`, copied
    /// into a new vector with room for one more.
    ///
    /// The relocation module builds a node's first overlay run from this: it
    /// is the run the overlay replaces, and the one extra slot is the edge a
    /// redirect is about to append. Only the compressed run is copied; the
    /// node's delta chain is [`DeltaChains::drain_into`]'s.
    pub(super) fn live_base_run(&self, node: Id<NT>, direction: Direction) -> Vec<u32> {
        let run = self.base_run(node.index(), direction);
        let mut copy = Vec::with_capacity(run.len() + 1);
        match self.tombstoned_edges().then_some(&self.live_edges) {
            Some(live) => copy.extend(
                run.iter()
                    .copied()
                    .filter(|&edge| live.is_live(edge as usize)),
            ),
            None => copy.extend_from_slice(run),
        }
        copy
    }

    /// The adjacency along `direction` of every surviving node whose edges do
    /// not read in ascending identity order, as it reads now, in the
    /// numbering `node_map` and `edge_map` give.
    ///
    /// A redirected edge reads after the edges its node already had, and a
    /// compaction keeps that order in the base it builds, so a node reads out
    /// of identity order long after its overlay is gone. Every other run is in
    /// identity order, which is all that compaction's counting placement
    /// produces, and compaction restores these from what this returns. The
    /// delta chain never breaks the order: it holds the newest edges, in the
    /// order they were added.
    pub(super) fn reordered_adjacency(
        &self,
        direction: Direction,
        node_map: &[u32],
        edge_map: &[u32],
    ) -> Vec<(u32, Vec<u32>)> {
        (0..self.node_bound())
            .filter(|&slot| node_map[slot] != NO_INDEX)
            .filter(|&slot| {
                let run = self
                    .overlay(slot, direction)
                    .unwrap_or_else(|| self.base_run(slot, direction));
                !run.is_sorted()
            })
            .map(|slot| {
                let run = self
                    .adjacent(Id::from_index(slot), direction)
                    .map(|edge| edge_map[edge.index()])
                    .collect();
                (node_map[slot], run)
            })
            .collect()
    }

    /// Iterate over the live outgoing edges of `node` in adjacency order: the
    /// order they were added to the node, so a redirected edge follows the
    /// edges already there.
    ///
    /// # Panics
    ///
    /// Panics when `node` names no slot in this store.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn outgoing(&self, node: Id<NT>) -> AdjacentEdges<'_, ET> {
        self.adjacent(node, Direction::Forward)
    }

    /// Iterate over the live incoming edges of `node` in adjacency order: the
    /// order they were added to the node, so a redirected edge follows the
    /// edges already there.
    ///
    /// # Panics
    ///
    /// Panics when `node` names no slot in this store.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn incoming(&self, node: Id<NT>) -> AdjacentEdges<'_, ET> {
        self.adjacent(node, Direction::Backward)
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

    /// The number of live outgoing edges of `node`.
    ///
    /// # Panics
    ///
    /// Panics when `node` names no slot in this store.
    #[must_use]
    pub fn out_degree(&self, node: Id<NT>) -> usize {
        self.degree(node, Direction::Forward)
    }

    /// The number of live incoming edges of `node`.
    ///
    /// # Panics
    ///
    /// Panics when `node` names no slot in this store.
    #[must_use]
    pub fn in_degree(&self, node: Id<NT>) -> usize {
        self.degree(node, Direction::Backward)
    }

    /// Count one node's adjacency without materializing it.
    ///
    /// The base run's length is the difference of two offsets, so the
    /// compressed half is constant time however many edges it holds, and only
    /// the delta chain is walked — which is the shape a consumer asking for a
    /// degree wants, because the store it asks spends most of its life
    /// compacted and answers in O(1) there.
    ///
    /// A store holding tombstoned edges cannot do either: whether a run entry
    /// is still live is one bitset probe per edge, and skipping it would count
    /// removed edges. Such a store counts the walk instead, which is what a
    /// caller would have written by hand.
    fn degree(&self, node: Id<NT>, direction: Direction) -> usize {
        let walk = self.adjacent(node, direction);
        if walk.live.is_some() {
            return walk.count();
        }
        let mut count = (walk.cursor.base_end - walk.cursor.base) as usize;
        let mut delta = walk.cursor.delta;
        while delta != NO_INDEX {
            count += 1;
            if delta == walk.cursor.delta_last {
                break;
            }
            delta = walk.axis.chains.next[delta as usize][walk.axis.chain];
        }
        count
    }

    /// Remove every live edge reachable from `node` in `direction`.
    ///
    /// The walk is collected first because the loop body needs the store
    /// mutably. That is the removal path, not a scan, so the buffer never
    /// reaches an analysis.
    pub(super) fn clear_adjacency(&mut self, node: Id<NT>, direction: Direction) {
        let incident: Vec<Id<ET>> = self.adjacent(node, direction).collect();
        for edge in incident {
            self.retire_edge(edge.index());
        }
    }
}

/// Live edges adjacent to one node, in adjacency order.
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
/// The counting sort of [`DenseFanout`], so the edges inside every node's run
/// keep their insertion order.
pub(super) fn compress<E, NT: IdTag>(
    edges: &[EdgeRecord<E, NT>],
    nodes: usize,
    direction: Direction,
) -> (Vec<u32>, Vec<u32>) {
    let endpoint = |edge: &EdgeRecord<E, NT>| match direction {
        Direction::Forward => edge.source.index(),
        Direction::Backward => edge.target.index(),
    };
    let mut offsets = Vec::new();
    let mut adjacency = Vec::new();
    DenseFanout::group_into(
        nodes,
        || {
            edges
                .iter()
                .enumerate()
                .map(|(index, edge)| (endpoint(edge), word(index)))
        },
        &mut offsets,
        &mut adjacency,
    );
    (offsets, adjacency)
}

/// Check one direction's compressed index read from outside: either the empty
/// index of a store that has never been compacted, or one run per base node
/// that together list each base edge once.
#[cfg(feature = "serde")]
pub(super) fn check_index(
    offsets: &[u32],
    edges: &[u32],
    base_nodes: usize,
    base_edges: usize,
) -> Result<(), &'static str> {
    if offsets.is_empty() {
        return if edges.is_empty() && base_nodes == 0 && base_edges == 0 {
            Ok(())
        } else {
            Err("a store with a compressed base has no adjacency index")
        };
    }
    if offsets.len() != base_nodes + 1
        || edges.len() != base_edges
        || offsets[0] != 0
        || offsets[base_nodes] as usize != edges.len()
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err("the adjacency index does not describe the compressed base");
    }
    if edges.iter().any(|&edge| edge as usize >= base_edges) {
        return Err("the adjacency index names an edge outside the compressed base");
    }
    Ok(())
}
