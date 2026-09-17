//! The [`Cfg`] data structure — a control-flow graph parameterised over
//! an instruction type `I` and optional consumer edge payload `E`.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::{BasicBlock, BlockId, BlockTag};
use crate::edge::{Edge, EdgeId, EdgeTag};
use crate::graph::edge_view::EdgeRef;
use crate::graph::store::Graph;
use crate::region::{Cleanup, Region};

mod cleanup;
mod compact;
mod mutation;
mod regions;
mod subgraph;
mod text;
mod view;

/// The store a [`Cfg`] is built on: blocks as nodes, edges as edges.
pub(crate) type CfgStore<I, E> = Graph<BasicBlock<I>, Edge<E>, BlockTag, EdgeTag>;

pub use compact::CfgRenumbering;
pub use text::parse_cfg_text;

/// One borrowed control-flow edge: identity, endpoints, kind, weight, and
/// consumer payload.
pub type CfgEdge<'g, E> = EdgeRef<'g, BlockId, EdgeId, Edge<E>>;

/// Why a requested instruction split-point sequence is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitPointError {
    /// A point lies beyond the original block's instruction count.
    OutOfBounds {
        /// Invalid instruction boundary.
        point: usize,
        /// Number of instructions in the block before any split.
        instruction_count: usize,
    },
    /// Points must be strictly increasing, so duplicates are invalid too.
    NotStrictlyIncreasing {
        /// Earlier point in the supplied sequence.
        previous: usize,
        /// Point that did not follow `previous`.
        point: usize,
    },
}

impl core::fmt::Display for SplitPointError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::OutOfBounds {
                point,
                instruction_count,
            } => write!(
                formatter,
                "split point {point} exceeds instruction count {instruction_count}"
            ),
            Self::NotStrictlyIncreasing { previous, point } => write!(
                formatter,
                "split point {point} does not strictly follow {previous}"
            ),
        }
    }
}

impl core::error::Error for SplitPointError {}

/// A control-flow graph over instruction type `I` and edge payload `E`.
///
/// The graph itself is a [`Graph`] whose nodes are [`BasicBlock`]s and whose
/// edges are [`Edge`]s; the CFG adds the entry block and the exception
/// metadata that make it a *control-flow* graph. Identities are therefore the
/// store's: stable until [`compact`](Self::compact) renumbers them and
/// reports the change.
///
/// `E = ()` retains the compact unannotated form. A frontend can instead use
/// `Cfg<I, E>` to keep format-specific edge provenance without teaching
/// cfglib about that format.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "I: serde::Serialize, E: serde::Serialize",
        deserialize = "I: serde::Deserialize<'de>, E: serde::Deserialize<'de>"
    ))
)]
pub struct Cfg<I, E = ()> {
    pub(crate) graph: CfgStore<I, E>,
    /// Entry block.
    pub(crate) entry: BlockId,
    /// Exception-handler regions (optional; empty for simple ISAs).
    pub(crate) regions: Vec<Region>,
    /// Cleanup records for handlers that continue somewhere once their body
    /// ends (optional; empty unless a frontend records them).
    #[cfg_attr(feature = "serde", serde(default = "Vec::new"))]
    pub(crate) cleanups: Vec<Cleanup>,
}

impl<I, E> Cfg<I, E> {
    /// Create an empty CFG with a single entry block.
    ///
    /// This is the primary constructor for ISA frontends that build
    /// the graph manually (as opposed to [`crate::CfgBuilder::build`] which
    /// processes a structured instruction stream).
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let entry = cfg.entry();
    /// let b1 = cfg.new_block();
    /// cfg.add_edge(entry, b1, EdgeKind::Fallthrough);
    /// assert_eq!(cfg.block_count(), 2);
    /// assert_eq!(cfg.edge_count(), 1);
    /// ```
    #[must_use]
    pub fn with_edge_payload() -> Self {
        let mut graph = CfgStore::<I, E>::tagged();
        let entry = graph.add_node(BasicBlock::new());
        Self {
            graph,
            entry,
            regions: Vec::new(),
            cleanups: Vec::new(),
        }
    }

    /// The entry block of the graph.
    #[inline]
    #[must_use]
    pub const fn entry(&self) -> BlockId {
        self.entry
    }

    /// Change the entry block of the graph.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a live block in this CFG.
    #[inline]
    pub fn set_entry(&mut self, id: BlockId) {
        assert!(
            self.graph.contains_node(id),
            "block {id} is not a live block of this CFG"
        );
        self.entry = id;
    }

    /// Look up a block by id.
    ///
    /// A removed block's contents stay readable until the next
    /// [`compact`](Self::compact); ask
    /// [`contains_block`](Self::contains_block) when that distinction
    /// matters.
    ///
    /// # Panics
    ///
    /// Panics if `id` names no block slot in this CFG.
    #[inline]
    #[must_use]
    pub fn block(&self, id: BlockId) -> &BasicBlock<I> {
        self.graph.node(id)
    }

    /// Mutable access to a block.
    ///
    /// # Panics
    ///
    /// Panics if `id` names no block slot in this CFG.
    #[inline]
    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock<I> {
        self.graph.node_mut(id)
    }

    /// All live blocks in identity order.
    pub fn blocks(&self) -> impl Iterator<Item = &BasicBlock<I>> + '_ {
        self.block_ids().map(|id| self.block(id))
    }

    /// Every live block identity in ascending order.
    pub fn block_ids(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.graph.node_ids()
    }

    /// Whether `id` names a block that has not been removed.
    #[inline]
    #[must_use]
    pub fn contains_block(&self, id: BlockId) -> bool {
        self.graph.contains_node(id)
    }

    /// Look up an edge by id.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a live edge in this CFG.
    #[inline]
    #[must_use]
    pub fn edge(&self, id: EdgeId) -> CfgEdge<'_, E> {
        assert!(self.graph.contains_edge(id), "edge {id} has been removed");
        let record = self.graph.edge(id);
        EdgeRef::new(id, record.source(), record.target(), record.payload())
    }

    /// All live edges in insertion order.
    pub fn edges(&self) -> impl Iterator<Item = CfgEdge<'_, E>> + '_ {
        self.edge_ids().map(|id| self.edge(id))
    }

    /// Every live edge identity in insertion order.
    pub fn edge_ids(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.graph.edge_ids()
    }

    /// Whether `id` names an edge that has not been removed.
    #[inline]
    #[must_use]
    pub fn contains_edge(&self, id: EdgeId) -> bool {
        self.graph.contains_edge(id)
    }

    /// Outgoing edge identities of a block, in insertion order.
    ///
    /// # Panics
    ///
    /// Panics if `id` names no block slot in this CFG.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn outgoing(&self, id: BlockId) -> impl Iterator<Item = EdgeId> + '_ {
        self.graph.outgoing(id)
    }

    /// Incoming edge identities of a block, in insertion order.
    ///
    /// # Panics
    ///
    /// Panics if `id` names no block slot in this CFG.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn incoming(&self, id: BlockId) -> impl Iterator<Item = EdgeId> + '_ {
        self.graph.incoming(id)
    }

    /// Successor block ids (allocation-free), retaining parallel entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let b0 = cfg.entry();
    /// let b1 = cfg.new_block();
    /// let b2 = cfg.new_block();
    /// cfg.add_edge(b0, b1, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(b0, b2, EdgeKind::ConditionalFalse);
    ///
    /// let succs: Vec<_> = cfg.successors(b0).collect();
    /// assert_eq!(succs.len(), 2);
    /// ```
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn successors(&self, id: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        self.graph.successors(id)
    }

    /// Predecessor block ids (allocation-free), retaining parallel entries.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn predecessors(&self, id: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        self.graph.predecessors(id)
    }

    /// Number of live basic blocks.
    ///
    /// A quantity, never an index range: use [`block_bound`](Self::block_bound)
    /// to size or iterate a block-indexed array.
    #[inline]
    #[must_use]
    pub const fn block_count(&self) -> usize {
        self.graph.node_count()
    }

    /// An exclusive upper bound on every live [`BlockId`]'s index.
    ///
    /// The right size for a block-indexed side table — a bound sizes an
    /// array, a count answers "how many". It exceeds
    /// [`block_count`](Self::block_count) exactly when blocks have been
    /// removed without a [`compact`](Self::compact).
    #[inline]
    #[must_use]
    pub fn block_bound(&self) -> usize {
        self.graph.node_bound()
    }

    /// Number of live edges.
    ///
    /// A quantity, never an index range: use [`edge_bound`](Self::edge_bound)
    /// to size or iterate an edge-indexed array.
    #[inline]
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// An exclusive upper bound on every live [`EdgeId`]'s index.
    ///
    /// The right size for an edge-indexed side table — a bound sizes an
    /// array, a count answers "how many".
    #[inline]
    #[must_use]
    pub fn edge_bound(&self) -> usize {
        self.graph.edge_bound()
    }

    /// Returns an iterator over exit blocks — blocks with no outgoing edges.
    ///
    /// These are the natural exit points of the control-flow graph
    /// (return blocks, terminators, etc.).
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let b1 = cfg.new_block();
    /// cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
    /// // b1 has no outgoing edges — it's the only exit block.
    /// let exits: Vec<_> = cfg.exit_blocks().collect();
    /// assert_eq!(exits, vec![b1]);
    /// ```
    pub fn exit_blocks(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.block_ids()
            .filter(|&id| self.outgoing(id).next().is_none())
    }
}

impl<I, E> Default for Cfg<I, E> {
    fn default() -> Self {
        Self::with_edge_payload()
    }
}

impl<I> Cfg<I> {
    /// Create an empty CFG with a single entry block and unit edge payloads.
    ///
    /// Use [`Cfg::with_edge_payload`] when edge metadata has a
    /// consumer-defined type.
    #[must_use]
    pub fn new() -> Self {
        Self::with_edge_payload()
    }
}

#[cfg(test)]
mod tests;
