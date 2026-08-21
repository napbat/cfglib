//! The [`Cfg`] data structure — a control-flow graph parameterised over
//! an instruction type `I`.

extern crate alloc;
use alloc::vec::Vec;
use core::ops::Index;
use core::slice;
use smallvec::SmallVec;

use crate::block::{BasicBlock, BlockId};
use crate::edge::{Edge, EdgeId, EdgeKind};
use crate::region::{Cleanup, Continuation, HandlerRef, Region, RegionId};

/// A control-flow graph over instruction type `I`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cfg<I> {
    pub(crate) blocks: Vec<BasicBlock<I>>,
    /// Edge arena — slots become `None` when removed via [`remove_edge`].
    pub(crate) edges: Vec<Option<Edge>>,
    /// Successor edge ids per block (indexed by `BlockId`).
    pub(crate) succs: Vec<SmallVec<[EdgeId; 2]>>,
    /// Predecessor edge ids per block (indexed by `BlockId`).
    pub(crate) preds: Vec<SmallVec<[EdgeId; 4]>>,
    /// Entry block.
    pub(crate) entry: BlockId,
    /// Exception-handler regions (optional; empty for simple ISAs).
    pub(crate) regions: Vec<Region>,
    /// Cleanup records for handlers that continue somewhere once their body
    /// ends (optional; empty unless a frontend records them).
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) cleanups: Vec<Cleanup>,
}

impl<I> Cfg<I> {
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
    /// assert_eq!(cfg.num_blocks(), 2);
    /// assert_eq!(cfg.num_edges(), 1);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        let entry = BlockId(0);
        Self {
            blocks: alloc::vec![BasicBlock {
                id: entry,
                instructions: Vec::new(),
                label: None,
            }],
            edges: Vec::new(),
            succs: alloc::vec![SmallVec::new()],
            preds: alloc::vec![SmallVec::new()],
            entry,
            regions: Vec::new(),
            cleanups: Vec::new(),
        }
    }

    /// The entry block of the graph.
    #[inline]
    #[must_use]
    pub fn entry(&self) -> BlockId {
        self.entry
    }

    /// Change the entry block of the graph.
    ///
    /// # Panics
    ///
    /// Panics (debug) if `id` does not refer to a block in this CFG.
    #[inline]
    pub fn set_entry(&mut self, id: BlockId) {
        debug_assert!(
            id.index() < self.blocks.len(),
            "BlockId {} out of range (num_blocks = {})",
            id,
            self.blocks.len(),
        );
        self.entry = id;
    }

    /// Look up a block by id.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a block in this CFG.
    #[inline]
    #[must_use]
    pub fn block(&self, id: BlockId) -> &BasicBlock<I> {
        debug_assert!(
            id.index() < self.blocks.len(),
            "BlockId {} out of range (num_blocks = {})",
            id,
            self.blocks.len(),
        );
        &self.blocks[id.index()]
    }

    /// Mutable access to a block.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a block in this CFG.
    #[inline]
    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock<I> {
        debug_assert!(
            id.index() < self.blocks.len(),
            "BlockId {} out of range (num_blocks = {})",
            id,
            self.blocks.len(),
        );
        &mut self.blocks[id.index()]
    }

    /// All blocks in allocation order.
    #[inline]
    #[must_use]
    pub fn blocks(&self) -> &[BasicBlock<I>] {
        &self.blocks
    }

    /// Look up an edge by id.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a live edge in this CFG.
    #[inline]
    #[must_use]
    pub fn edge(&self, id: EdgeId) -> &Edge {
        self.edges[id.index()]
            .as_ref()
            .expect("edge has been removed")
    }

    /// All live edges (skips tombstones left by [`Self::remove_edge`]).
    pub fn edges(&self) -> impl Iterator<Item = &Edge> {
        self.edges.iter().filter_map(|slot| slot.as_ref())
    }

    /// Number of edge slots (including tombstones).
    ///
    /// This is the raw arena length, **not** the count of live edges.
    /// Use `edges().count()` for the live edge count.
    #[inline]
    pub(crate) fn edge_slots(&self) -> usize {
        self.edges.len()
    }

    /// Successor edges for a block.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a block in this CFG.
    #[inline]
    #[must_use]
    pub fn successor_edges(&self, id: BlockId) -> &[EdgeId] {
        debug_assert!(
            id.index() < self.succs.len(),
            "BlockId {} out of range for successor lookup (num_blocks = {})",
            id,
            self.succs.len(),
        );
        &self.succs[id.index()]
    }

    /// Predecessor edges for a block.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a block in this CFG.
    #[inline]
    #[must_use]
    pub fn predecessor_edges(&self, id: BlockId) -> &[EdgeId] {
        debug_assert!(
            id.index() < self.preds.len(),
            "BlockId {} out of range for predecessor lookup (num_blocks = {})",
            id,
            self.preds.len(),
        );
        &self.preds[id.index()]
    }

    /// Successor block ids (allocation-free).
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
    #[must_use]
    pub fn successors(&self, id: BlockId) -> Successors<'_, I> {
        Successors {
            cfg: self,
            iter: self.succs[id.index()].iter(),
        }
    }

    /// Predecessor block ids (allocation-free).
    #[must_use]
    pub fn predecessors(&self, id: BlockId) -> Predecessors<'_, I> {
        Predecessors {
            cfg: self,
            iter: self.preds[id.index()].iter(),
        }
    }

    /// Number of basic blocks.
    #[inline]
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Number of live edges (excludes tombstones).
    #[inline]
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.edges.iter().filter(|e| e.is_some()).count()
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
        self.blocks
            .iter()
            .filter(|b| self.succs[b.id().index()].is_empty())
            .map(super::block::BasicBlock::id)
    }

    // ── Region methods ─────────────────────────────────────────────

    /// All exception-handler regions.
    #[inline]
    #[must_use]
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    /// Add a region and return its id.
    pub fn add_region(&mut self, mut region: Region) -> RegionId {
        let id = RegionId::from_index(self.regions.len());
        region.id = id;
        self.regions.push(region);
        id
    }

    /// Returns the innermost region that protects `block`, if any.
    #[must_use]
    pub fn protecting_region(&self, block: BlockId) -> Option<&Region> {
        // Return the deepest (last-added) region whose protected set
        // contains this block.
        self.regions
            .iter()
            .rev()
            .find(|r| r.protected_blocks.contains(&block))
    }

    // ── Cleanup continuations ─────────────────────────────────────

    /// Every recorded [`Cleanup`], in the order the handlers first recorded
    /// one.
    #[inline]
    #[must_use]
    pub fn cleanups(&self) -> &[Cleanup] {
        &self.cleanups
    }

    /// The cleanup record of `handler`, if it has one.
    #[must_use]
    pub fn cleanup(&self, handler: HandlerRef) -> Option<&Cleanup> {
        self.cleanups
            .iter()
            .find(|cleanup| cleanup.handler == handler)
    }

    /// Record one route out of a cleanup handler: where control resumes once
    /// the cleanup body ends, and the reason that entered it.
    ///
    /// The first call for a handler creates its record. Recording the same
    /// `(reason, resume)` pair twice is a no-op, so a lowering that walks
    /// several transfers into one cleanup keeps one route per distinct
    /// destination, in first-recorded order.
    ///
    /// # Examples
    ///
    /// A `try { ... } finally { ... }` whose body both falls out normally and
    /// `return`s: one cleanup block, two routes, told apart by reason.
    ///
    /// ```
    /// use cfglib::{
    ///     Cfg, CompletionReason, Continuation, Handler, HandlerKind, HandlerRef, Region,
    ///     RegionId, build_eh_model,
    /// };
    ///
    /// let mut cfg = Cfg::<&'static str>::new();
    /// let cleanup_block = cfg.new_block();
    /// let after = cfg.new_block();
    /// let exit = cfg.new_block();
    ///
    /// let region = cfg.add_region(Region {
    ///     id: RegionId::from_raw(0), // overwritten by `add_region`
    ///     protected_blocks: [cfg.entry()].into_iter().collect(),
    ///     handlers: vec![Handler {
    ///         entry: cleanup_block,
    ///         body: [cleanup_block].into_iter().collect(),
    ///         kind: HandlerKind::Finally,
    ///     }],
    ///     parent: None,
    /// });
    ///
    /// let handler = HandlerRef::new(region, 0);
    /// cfg.set_cleanup_resume(handler, cleanup_block);
    /// cfg.add_continuation(handler, Continuation {
    ///     reason: CompletionReason::Normal,
    ///     resume: after,
    /// });
    /// cfg.add_continuation(handler, Continuation {
    ///     reason: CompletionReason::Return,
    ///     resume: exit,
    /// });
    ///
    /// // Both routes leave the same block, and each one is identifiable.
    /// let model = build_eh_model(&cfg);
    /// let recorded = &model.cleanups[&cleanup_block];
    /// assert_eq!(recorded.resume_from, Some(cleanup_block));
    /// assert_eq!(
    ///     recorded.resumes_for(CompletionReason::Return).collect::<Vec<_>>(),
    ///     vec![exit]
    /// );
    /// ```
    ///
    /// # Panics
    ///
    /// Panics (debug) if `handler` does not refer to a handler of a region in
    /// this CFG — register the region first, then attach its routes.
    pub fn add_continuation(&mut self, handler: HandlerRef, continuation: Continuation) {
        debug_assert!(
            self.handler_exists(handler),
            "handler does not exist in this CFG"
        );
        let cleanup = self.cleanup_entry(handler);
        if !cleanup.continuations.contains(&continuation) {
            cleanup.continuations.push(continuation);
        }
    }

    /// Record the block a cleanup handler's body ends in — the block every
    /// continuation edge leaves from.
    ///
    /// A cleanup that diverges never reaches one, so leaving it unset is the
    /// honest description of a `finally` that returns.
    ///
    /// # Panics
    ///
    /// Panics (debug) if `handler` does not refer to a handler of a region in
    /// this CFG, or if `resume_from` does not refer to a block in it.
    pub fn set_cleanup_resume(&mut self, handler: HandlerRef, resume_from: BlockId) {
        debug_assert!(
            self.handler_exists(handler),
            "handler does not exist in this CFG"
        );
        debug_assert!(
            resume_from.index() < self.blocks.len(),
            "resume block does not exist in this CFG"
        );
        self.cleanup_entry(handler).resume_from = Some(resume_from);
    }

    /// The cleanup record of `handler`, created empty when it is the first
    /// route recorded for it.
    fn cleanup_entry(&mut self, handler: HandlerRef) -> &mut Cleanup {
        let existing = self
            .cleanups
            .iter()
            .position(|cleanup| cleanup.handler == handler);
        let at = existing.unwrap_or_else(|| {
            self.cleanups.push(Cleanup {
                handler,
                resume_from: None,
                continuations: Vec::new(),
            });
            self.cleanups.len() - 1
        });
        &mut self.cleanups[at]
    }

    /// Whether `handler` refers to a handler of a region in this CFG.
    fn handler_exists(&self, handler: HandlerRef) -> bool {
        self.regions
            .get(handler.region().index())
            .is_some_and(|region| handler.index() < region.handlers.len())
    }

    // ── Block / edge mutation ─────────────────────────────────────

    /// Allocate a new empty block and return its id.
    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId::from_index(self.blocks.len());
        self.blocks.push(BasicBlock {
            id,
            instructions: Vec::new(),
            label: None,
        });

        self.succs.push(SmallVec::new());
        self.preds.push(SmallVec::new());

        id
    }

    /// Add a directed edge and return its id.
    pub fn add_edge(&mut self, source: BlockId, target: BlockId, kind: EdgeKind) -> EdgeId {
        self.add_edge_inner(source, target, kind, None)
    }

    /// Add a directed edge with a branch weight.
    pub fn add_weighted_edge(
        &mut self,
        source: BlockId,
        target: BlockId,
        kind: EdgeKind,
        weight: f64,
    ) -> EdgeId {
        self.add_edge_inner(source, target, kind, Some(weight))
    }

    fn add_edge_inner(
        &mut self,
        source: BlockId,
        target: BlockId,
        kind: EdgeKind,
        weight: Option<f64>,
    ) -> EdgeId {
        let id = EdgeId::from_index(self.edges.len());
        self.edges.push(Some(Edge {
            id,
            source,
            target,
            kind,
            weight,
        }));

        self.succs[source.index()].push(id);
        self.preds[target.index()].push(id);

        id
    }

    /// Remove an edge by id.
    ///
    /// Returns the removed [`Edge`], or `None` if the id is out of
    /// range or already removed. The edge slot is replaced with a
    /// tombstone (`None`) so that existing [`EdgeId`]s remain valid.
    ///
    /// The successor and predecessor lists of the affected blocks are
    /// updated.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let b0 = cfg.entry();
    /// let b1 = cfg.new_block();
    /// let eid = cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
    ///
    /// assert_eq!(cfg.num_edges(), 1);
    /// let removed = cfg.remove_edge(eid).unwrap();
    /// assert_eq!(removed.kind(), EdgeKind::Fallthrough);
    /// assert_eq!(cfg.num_edges(), 0);
    /// // Double-remove returns None.
    /// assert!(cfg.remove_edge(eid).is_none());
    /// ```
    pub fn remove_edge(&mut self, id: EdgeId) -> Option<Edge> {
        let slot = self.edges.get_mut(id.index())?;
        let edge = slot.take()?;
        self.succs[edge.source.index()].retain(|e| *e != id);
        self.preds[edge.target.index()].retain(|e| *e != id);
        Some(edge)
    }

    /// Split a block at instruction index `at`.
    ///
    /// Instructions `[at..]` are moved into a new block. A
    /// [`Fallthrough`](EdgeKind::Fallthrough) edge is inserted from
    /// the original block to the new one, and all outgoing edges of
    /// the original block are transferred to the new block.
    ///
    /// Returns the id of the newly created block.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range or `at > instructions.len()`.
    pub fn split_block(&mut self, id: BlockId, at: usize) -> BlockId {
        let tail_insts: Vec<I> = self.blocks[id.index()].instructions.split_off(at);
        let new_id = self.new_block();
        self.blocks[new_id.index()].instructions = tail_insts;

        self.move_outgoing_edges(id, new_id);

        // Insert fallthrough edge from original to new block.
        self.add_edge(id, new_id, EdgeKind::Fallthrough);

        new_id
    }

    /// Redirect all edges that target `old` to target `new_target` instead.
    ///
    /// This is useful for bypassing a block before removal.
    ///
    /// # Panics
    ///
    /// Panics if either block is out of range or an incoming edge was removed.
    pub fn redirect_edges_to(&mut self, old: BlockId, new_target: BlockId) {
        let old_index = old.index();
        let new_target_index = new_target.index();
        let _ = &self.preds[old_index];
        let _ = &self.preds[new_target_index];
        if old == new_target {
            return;
        }

        let incoming = core::mem::take(&mut self.preds[old_index]);
        for &eid in &incoming {
            self.edges[eid.index()].as_mut().unwrap().target = new_target;
        }
        self.preds[new_target_index].extend(incoming);
    }

    /// Move every outgoing edge of `old` to `new_source` in adjacency order.
    ///
    /// Only the source endpoint changes: edge identities, targets, kinds,
    /// weights, and predecessor adjacency all remain intact. When
    /// `new_source` has no outgoing edges, ownership of `old`'s complete
    /// adjacency buffer moves without reallocating.
    pub(crate) fn move_outgoing_edges(&mut self, old: BlockId, new_source: BlockId) {
        if old == new_source {
            return;
        }

        let outgoing = core::mem::take(&mut self.succs[old.index()]);
        for &eid in &outgoing {
            self.edges[eid.index()].as_mut().unwrap().source = new_source;
        }
        if self.succs[new_source.index()].is_empty() {
            self.succs[new_source.index()] = outgoing;
        } else {
            self.succs[new_source.index()].extend(outgoing);
        }
    }

    /// Mutable access to an edge.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range or has been removed.
    #[inline]
    pub fn edge_mut(&mut self, id: EdgeId) -> &mut Edge {
        self.edges[id.index()]
            .as_mut()
            .expect("edge has been removed")
    }
}

// ── Default impl ──────────────────────────────────────────────────

impl<I> Default for Cfg<I> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Graph view impls ───────────────────────────────────────────────

impl<I> crate::graph::view::DirectedGraphView for Cfg<I> {
    type NodeId = BlockId;

    fn node_count(&self) -> usize {
        self.num_blocks()
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        Cfg::successors(self, node)
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        Cfg::predecessors(self, node)
    }
}

impl<I> crate::graph::view::RootedGraphView for Cfg<I> {
    fn root(&self) -> Self::NodeId {
        self.entry()
    }
}

// ── Index impls ────────────────────────────────────────────────────

impl<I> Index<BlockId> for Cfg<I> {
    type Output = BasicBlock<I>;

    /// Index into the CFG by [`BlockId`].
    ///
    /// Equivalent to [`Cfg::block`] but usable with `cfg[id]` syntax.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a block in this CFG.
    #[inline]
    fn index(&self, id: BlockId) -> &BasicBlock<I> {
        &self.blocks[id.index()]
    }
}

impl<I> Index<EdgeId> for Cfg<I> {
    type Output = Edge;

    /// Index into the CFG by [`EdgeId`].
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a live edge in this CFG.
    #[inline]
    fn index(&self, id: EdgeId) -> &Edge {
        self.edges[id.index()]
            .as_ref()
            .expect("edge has been removed")
    }
}

/// Iterator over successor block ids (zero-allocation).
pub struct Successors<'a, I> {
    cfg: &'a Cfg<I>,
    iter: slice::Iter<'a, EdgeId>,
}

impl<I> Iterator for Successors<'_, I> {
    type Item = BlockId;
    #[inline]
    fn next(&mut self) -> Option<BlockId> {
        self.iter.next().map(|&eid| self.cfg.edge(eid).target)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I> ExactSizeIterator for Successors<'_, I> {}

/// Iterator over predecessor block ids (zero-allocation).
pub struct Predecessors<'a, I> {
    cfg: &'a Cfg<I>,
    iter: slice::Iter<'a, EdgeId>,
}

impl<I> Iterator for Predecessors<'_, I> {
    type Item = BlockId;
    #[inline]
    fn next(&mut self) -> Option<BlockId> {
        self.iter.next().map(|&eid| self.cfg.edge(eid).source)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I> ExactSizeIterator for Predecessors<'_, I> {}

// ── Convenience dataflow method ────────────────────────────────────
impl<I> Cfg<I> {
    /// Run a fixpoint dataflow analysis on this CFG.
    ///
    /// This is a thin convenience wrapper around
    /// [`dataflow::fixpoint::solve`](crate::dataflow::fixpoint::solve).
    pub fn solve_dataflow<P: crate::dataflow::fixpoint::Problem<I>>(
        &self,
        problem: &P,
    ) -> crate::dataflow::fixpoint::FixpointResult<P::Fact> {
        crate::dataflow::fixpoint::solve(self, problem)
    }
}

// ── Subgraph extraction ───────────────────────────────────────────
impl<I: Clone> Cfg<I> {
    /// Extract a sub-CFG containing only the specified blocks.
    ///
    /// The resulting CFG preserves edges between the selected blocks
    /// and remaps block IDs to be contiguous starting from 0.
    /// The first block in `blocks` becomes the entry.
    ///
    /// Edges that cross the boundary (one endpoint outside the set)
    /// are dropped.
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
    /// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
    /// cfg.add_edge(b1, b2, EdgeKind::Fallthrough);
    ///
    /// let sub = cfg.subgraph(&[b0, b1]);
    /// assert_eq!(sub.num_blocks(), 2);
    /// assert_eq!(sub.num_edges(), 1); // b1→b2 dropped
    /// ```
    #[must_use]
    pub fn subgraph(&self, blocks: &[BlockId]) -> Self {
        if blocks.is_empty() {
            return Self::new();
        }

        let mut new_cfg = Self::new();

        // Map old BlockId → new BlockId via dense Vec (O(1) lookup).
        let mut id_map: Vec<Option<BlockId>> = alloc::vec![None; self.num_blocks()];
        id_map[blocks[0].index()] = Some(new_cfg.entry());

        // Copy instructions into the entry block.
        let src = &self.blocks[blocks[0].index()];
        for inst in src.instructions() {
            new_cfg.block_mut(new_cfg.entry()).push(inst.clone());
        }
        if let Some(lbl) = src.label() {
            new_cfg.block_mut(new_cfg.entry()).set_label(lbl);
        }

        // Create remaining blocks.
        for &bid in &blocks[1..] {
            let new_id = new_cfg.new_block();
            id_map[bid.index()] = Some(new_id);
            let old_block = &self.blocks[bid.index()];
            for inst in old_block.instructions() {
                new_cfg.block_mut(new_id).push(inst.clone());
            }
            if let Some(lbl) = old_block.label() {
                new_cfg.block_mut(new_id).set_label(lbl);
            }
        }

        // Copy live edges that stay within the subgraph.
        for edge in self.edges() {
            let new_src = id_map.get(edge.source().index()).copied().flatten();
            let new_tgt = id_map.get(edge.target().index()).copied().flatten();
            if let (Some(ns), Some(nt)) = (new_src, new_tgt) {
                let eid = new_cfg.add_edge(ns, nt, edge.kind());
                if let Some(w) = edge.weight() {
                    new_cfg.edge_mut(eid).set_weight(Some(w));
                }
            }
        }

        new_cfg
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    extern crate std;
    use super::*;
    use crate::edge::EdgeKind;
    use crate::test_util::MockInst;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn edge_weight_roundtrip() {
        let mut cfg = Cfg::<MockInst>::new();
        let b0 = cfg.entry();
        let b1 = cfg.new_block();
        let eid = cfg.add_weighted_edge(b0, b1, EdgeKind::ConditionalTrue, 0.75);
        assert_eq!(cfg.edge(eid).weight(), Some(0.75));
        // Default edge should have no weight.
        let eid2 = cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
        assert_eq!(cfg.edge(eid2).weight(), None);
    }

    #[test]
    fn subgraph_extraction() {
        let mut cfg = Cfg::<MockInst>::new();
        let b0 = cfg.entry();
        let b1 = cfg.new_block();
        let b2 = cfg.new_block();
        cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
        cfg.add_edge(b1, b2, EdgeKind::Fallthrough);

        // Extract first two blocks.
        let sub = cfg.subgraph(&[b0, b1]);
        assert_eq!(sub.num_blocks(), 2);
        // The subgraph should have an edge from block 0 to block 1.
        let succs: Vec<BlockId> = sub.successors(sub.entry()).collect();
        assert_eq!(succs.len(), 1);
    }

    #[test]
    fn subgraph_empty_input() {
        let sub = Cfg::<MockInst>::new().subgraph(&[]);
        assert_eq!(sub.num_blocks(), 1); // Cfg::new() always has an entry
    }

    #[test]
    fn remove_edge_tombstones_correctly() {
        let mut cfg = Cfg::<MockInst>::new();
        let b0 = cfg.entry();
        let b1 = cfg.new_block();
        let b2 = cfg.new_block();
        let e1 = cfg.add_edge(b0, b1, EdgeKind::ConditionalTrue);
        let e2 = cfg.add_edge(b0, b2, EdgeKind::ConditionalFalse);

        // Both edges are live.
        assert_eq!(cfg.num_edges(), 2);
        assert_eq!(cfg.edges().count(), 2);

        // Remove one edge.
        let removed = cfg.remove_edge(e1).unwrap();
        assert_eq!(removed.kind(), EdgeKind::ConditionalTrue);

        // edges() should now skip the tombstone.
        assert_eq!(cfg.edges().count(), 1);
        let remaining: Vec<&Edge> = cfg.edges().collect();
        assert_eq!(remaining[0].id(), e2);

        // Successor list should only contain e2.
        assert_eq!(cfg.successor_edges(b0).len(), 1);
        assert_eq!(cfg.successor_edges(b0)[0], e2);

        // Double-remove returns None.
        assert!(cfg.remove_edge(e1).is_none());
    }

    #[test]
    fn split_block_preserves_outgoing_edge_identity_and_metadata() {
        let mut cfg = Cfg::<u32>::new();
        let source = cfg.entry();
        let sink = cfg.new_block();
        cfg.block_mut(source).instructions_vec_mut().extend([1, 2]);
        let outgoing = cfg.add_weighted_edge(source, sink, EdgeKind::ConditionalTrue, 0.75);

        let split = cfg.split_block(source, 1);
        let [fallthrough] = cfg.successor_edges(source) else {
            panic!("split source should have one fallthrough edge");
        };

        assert_eq!(cfg.successor_edges(split), &[outgoing]);
        assert_eq!(cfg.edge(outgoing).source(), split);
        assert_eq!(cfg.edge(outgoing).target(), sink);
        assert_eq!(cfg.edge(outgoing).kind(), EdgeKind::ConditionalTrue);
        assert_eq!(cfg.edge(outgoing).weight(), Some(0.75));
        assert_eq!(cfg.edge(*fallthrough).source(), source);
        assert_eq!(cfg.edge(*fallthrough).target(), split);
        assert_eq!(cfg.predecessor_edges(sink), &[outgoing]);
    }

    #[test]
    fn redirect_edges_moves_predecessors_in_order() {
        let mut cfg = Cfg::<MockInst>::new();
        let old = cfg.new_block();
        let new_target = cfg.new_block();
        let first = cfg.add_edge(cfg.entry(), new_target, EdgeKind::Fallthrough);
        let second = cfg.add_edge(cfg.entry(), old, EdgeKind::ConditionalTrue);
        let third = cfg.add_weighted_edge(cfg.entry(), old, EdgeKind::ConditionalFalse, 0.25);

        cfg.redirect_edges_to(old, new_target);

        assert_eq!(cfg.predecessor_edges(old), &[]);
        assert_eq!(cfg.predecessor_edges(new_target), &[first, second, third]);
        assert_eq!(cfg.edge(second).target(), new_target);
        assert_eq!(cfg.edge(third).target(), new_target);
        assert_eq!(cfg.edge(third).weight(), Some(0.25));
    }

    #[test]
    fn redirect_edges_to_same_block_is_a_noop() {
        let mut cfg = Cfg::<MockInst>::new();
        let target = cfg.new_block();
        let edge = cfg.add_edge(cfg.entry(), target, EdgeKind::Fallthrough);

        cfg.redirect_edges_to(target, target);

        assert_eq!(cfg.predecessor_edges(target), &[edge]);
        assert_eq!(cfg.edge(edge).target(), target);
    }

    #[test]
    fn redirect_edges_rejects_an_invalid_target_before_mutating() {
        let mut cfg = Cfg::<MockInst>::new();
        let old_target = cfg.new_block();
        let edge = cfg.add_edge(cfg.entry(), old_target, EdgeKind::Fallthrough);
        let invalid = BlockId::from_raw(u32::MAX);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cfg.redirect_edges_to(old_target, invalid);
        }));

        assert!(panic.is_err());
        assert_eq!(cfg.predecessor_edges(old_target), &[edge]);
        assert_eq!(cfg.edge(edge).target(), old_target);
    }

    #[test]
    fn move_outgoing_edges_preserves_order_identity_and_metadata() {
        let mut cfg = Cfg::<MockInst>::new();
        let old = cfg.new_block();
        let new_source = cfg.new_block();
        let sink = cfg.new_block();
        let existing = cfg.add_edge(new_source, sink, EdgeKind::Fallthrough);
        let first = cfg.add_weighted_edge(old, sink, EdgeKind::ConditionalTrue, 0.25);
        let second = cfg.add_weighted_edge(old, sink, EdgeKind::ConditionalFalse, 0.75);
        let becomes_self = cfg.add_weighted_edge(old, new_source, EdgeKind::Back, 0.875);
        let old_self = cfg.add_weighted_edge(old, old, EdgeKind::Unconditional, 0.5);

        cfg.move_outgoing_edges(old, new_source);

        assert_eq!(cfg.successor_edges(old), &[]);
        assert_eq!(
            cfg.successor_edges(new_source),
            &[existing, first, second, becomes_self, old_self]
        );
        assert_eq!(cfg.edge(first).source(), new_source);
        assert_eq!(cfg.edge(first).target(), sink);
        assert_eq!(cfg.edge(first).kind(), EdgeKind::ConditionalTrue);
        assert_eq!(cfg.edge(first).weight(), Some(0.25));
        assert_eq!(cfg.edge(second).weight(), Some(0.75));
        assert_eq!(cfg.edge(becomes_self).source(), new_source);
        assert_eq!(cfg.edge(becomes_self).target(), new_source);
        assert_eq!(cfg.edge(becomes_self).weight(), Some(0.875));
        assert_eq!(cfg.edge(old_self).source(), new_source);
        assert_eq!(cfg.edge(old_self).target(), old);
        assert_eq!(cfg.edge(old_self).weight(), Some(0.5));
        assert_eq!(cfg.predecessor_edges(sink), &[existing, first, second]);
        assert_eq!(cfg.predecessor_edges(new_source), &[becomes_self]);
        assert_eq!(cfg.predecessor_edges(old), &[old_self]);
    }

    #[test]
    fn exit_blocks_iterator() {
        let mut cfg = Cfg::<MockInst>::new();
        let b1 = cfg.new_block();
        cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
        // b1 has no outgoing edges — it's an exit block.
        let exits: Vec<BlockId> = cfg.exit_blocks().collect();
        assert_eq!(exits, vec![b1]);
    }
}
