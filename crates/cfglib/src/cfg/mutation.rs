//! Structural mutation of [`Cfg`] — block and edge creation, removal,
//! redirection, and block splitting.

extern crate alloc;

use alloc::vec::Vec;

use crate::block::{BasicBlock, BlockId};
use crate::edge::{Edge, EdgeId, EdgeKind};
use crate::rewrite::Rewrite;

use super::{Cfg, SplitPointError};

impl<I, E> Cfg<I, E> {
    /// Allocate a new empty block and return its id.
    pub fn new_block(&mut self) -> BlockId {
        self.graph.add_node(BasicBlock::new())
    }

    /// Remove a block and every edge incident to it.
    ///
    /// Returns whether the block had been live. The block's instructions stay
    /// readable through [`block`](Cfg::block) until the next
    /// [`compact`](Cfg::compact), and every other identity is unaffected.
    ///
    /// # Panics
    ///
    /// Panics when `id` is the entry block: a CFG always has an entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let orphan = cfg.new_block();
    /// cfg.add_edge(cfg.entry(), orphan, EdgeKind::Fallthrough);
    ///
    /// assert!(cfg.remove_block(orphan));
    /// assert_eq!(cfg.block_count(), 1);
    /// assert_eq!(cfg.edge_count(), 0);
    /// assert!(!cfg.contains_block(orphan));
    /// ```
    pub fn remove_block(&mut self, id: BlockId) -> bool {
        assert!(id != self.entry, "the entry block cannot be removed");
        self.graph.remove_node(id)
    }

    /// Remove a block and report every identity the removal retired.
    pub fn remove_block_mapped(&mut self, id: BlockId) -> (bool, Rewrite) {
        let mut mapping = Rewrite::new();
        if !self.contains_block(id) {
            return (false, mapping);
        }
        let incident: Vec<EdgeId> = self.outgoing(id).chain(self.incoming(id)).collect();
        for edge in incident {
            mapping.record_edge(edge, []);
        }
        let removed = self.remove_block(id);
        mapping.record_block(id, []);
        (removed, mapping)
    }

    /// Add a directed edge with the default consumer payload.
    ///
    /// This is the compatibility front door for `Cfg<I>` and is also useful
    /// when a custom payload has a meaningful [`Default`] value.
    pub fn add_edge(&mut self, source: BlockId, target: BlockId, kind: EdgeKind) -> EdgeId
    where
        E: Default,
    {
        self.add_edge_with_payload(source, target, kind, E::default())
    }

    /// Add a weighted directed edge with the default consumer payload.
    pub fn add_weighted_edge(
        &mut self,
        source: BlockId,
        target: BlockId,
        kind: EdgeKind,
        weight: f64,
    ) -> EdgeId
    where
        E: Default,
    {
        self.add_weighted_edge_with_payload(source, target, kind, weight, E::default())
    }

    /// Add a directed edge with consumer metadata and return its id.
    pub fn add_edge_with_payload(
        &mut self,
        source: BlockId,
        target: BlockId,
        kind: EdgeKind,
        payload: E,
    ) -> EdgeId {
        self.graph
            .add_edge(source, target, Edge::new(kind, None, payload))
    }

    /// Add a directed edge with a branch weight and consumer metadata.
    pub fn add_weighted_edge_with_payload(
        &mut self,
        source: BlockId,
        target: BlockId,
        kind: EdgeKind,
        weight: f64,
        payload: E,
    ) -> EdgeId {
        self.graph
            .add_edge(source, target, Edge::new(kind, Some(weight), payload))
    }

    /// Remove an edge by id.
    ///
    /// Returns whether the edge had been live. The successor and predecessor
    /// adjacency of the affected blocks is updated and every other identity
    /// stays valid.
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
    /// assert_eq!(cfg.edge_count(), 1);
    /// assert!(cfg.remove_edge(eid));
    /// assert_eq!(cfg.edge_count(), 0);
    /// // Double-remove reports that there was nothing to do.
    /// assert!(!cfg.remove_edge(eid));
    /// ```
    pub fn remove_edge(&mut self, id: EdgeId) -> bool {
        self.graph.remove_edge(id)
    }

    /// Remove an edge and return both the outcome and the identity mapping.
    pub fn remove_edge_mapped(&mut self, id: EdgeId) -> (bool, Rewrite) {
        let removed = self.remove_edge(id);
        let mut mapping = Rewrite::new();
        if removed {
            mapping.record_edge(id, []);
        }
        (removed, mapping)
    }

    /// Split a block at instruction index `at` using the default payload for
    /// the new fallthrough edge.
    pub fn split_block(&mut self, id: BlockId, at: usize) -> BlockId
    where
        E: Default,
    {
        self.split_block_with_payload(id, at, E::default())
    }

    /// Split a block at instruction index `at` with an explicit payload for
    /// the new fallthrough edge.
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
    pub fn split_block_with_payload(
        &mut self,
        id: BlockId,
        at: usize,
        fallthrough_payload: E,
    ) -> BlockId {
        self.split_block_with_payload_inner(id, at, fallthrough_payload, None)
    }

    /// Split a block and return its new tail plus every affected identity.
    ///
    /// # Panics
    ///
    /// Panics if `id` is invalid or `at` exceeds the instruction count.
    pub fn split_block_with_payload_mapped(
        &mut self,
        id: BlockId,
        at: usize,
        fallthrough_payload: E,
    ) -> (BlockId, Rewrite) {
        let mut mapping = Rewrite::new();
        let new_id =
            self.split_block_with_payload_inner(id, at, fallthrough_payload, Some(&mut mapping));
        (new_id, mapping)
    }

    fn split_block_with_payload_inner(
        &mut self,
        id: BlockId,
        at: usize,
        fallthrough_payload: E,
        mapping: Option<&mut Rewrite>,
    ) -> BlockId {
        let tail_instructions: Vec<I> = self.block_mut(id).instructions.split_off(at);
        let new_id = self.new_block();
        self.block_mut(new_id).instructions = tail_instructions;

        self.move_outgoing_edges(id, new_id);

        let fallthrough =
            self.add_edge_with_payload(id, new_id, EdgeKind::Fallthrough, fallthrough_payload);

        if let Some(mapping) = mapping {
            mapping.record_block(id, [id, new_id]);
            mapping.record_created_block(new_id);
            for edge in self.graph.outgoing(new_id) {
                mapping.record_edge(edge, [edge]);
            }
            mapping.record_created_edge(fallthrough);
        }
        new_id
    }

    /// Split one block at ordered instruction boundaries with explicit edge
    /// payloads and return every resulting block plus an identity mapping.
    ///
    /// Points are offsets in the original block, not in successively shorter
    /// tails. They must be strictly increasing and may include `0` or the
    /// original instruction count. Validation happens before mutation, so an
    /// error leaves the CFG unchanged. The returned blocks are in execution
    /// order and include `id` as the first element.
    ///
    /// # Errors
    ///
    /// Returns [`SplitPointError`] when a point is out of bounds or the points
    /// are not strictly increasing.
    pub fn split_block_at_points_with_payloads(
        &mut self,
        id: BlockId,
        points: impl IntoIterator<Item = (usize, E)>,
    ) -> Result<(Vec<BlockId>, Rewrite), SplitPointError> {
        let points: Vec<_> = points.into_iter().collect();
        let instruction_count = self.block(id).instructions().len();
        let mut previous = None;
        for &(point, _) in &points {
            if point > instruction_count {
                return Err(SplitPointError::OutOfBounds {
                    point,
                    instruction_count,
                });
            }
            if let Some(previous) = previous
                && point <= previous
            {
                return Err(SplitPointError::NotStrictlyIncreasing { previous, point });
            }
            previous = Some(point);
        }

        let mut blocks = alloc::vec![id];
        let mut mapping = Rewrite::new();
        let mut current = id;
        let mut base = 0;
        for (point, payload) in points {
            let (tail, split) =
                self.split_block_with_payload_mapped(current, point - base, payload);
            mapping.compose(split);
            blocks.push(tail);
            current = tail;
            base = point;
        }
        Ok((blocks, mapping))
    }

    /// Split one block at ordered instruction boundaries using default edge
    /// payloads.
    ///
    /// # Errors
    ///
    /// Returns [`SplitPointError`] when a point is out of bounds or the points
    /// are not strictly increasing.
    pub fn split_block_at_points(
        &mut self,
        id: BlockId,
        points: &[usize],
    ) -> Result<(Vec<BlockId>, Rewrite), SplitPointError>
    where
        E: Default,
    {
        self.split_block_at_points_with_payloads(
            id,
            points.iter().copied().map(|point| (point, E::default())),
        )
    }

    /// Redirect all edges that target `old` to target `new_target` instead.
    ///
    /// This is useful for bypassing a block before removal.
    ///
    /// # Panics
    ///
    /// Panics if either block has been removed.
    pub fn redirect_edges_to(&mut self, old: BlockId, new_target: BlockId) {
        self.redirect_edges_to_inner(old, new_target, None);
    }

    /// Redirect every edge targeting `old` and return their stable mapping.
    pub fn redirect_edges_to_mapped(&mut self, old: BlockId, new_target: BlockId) -> Rewrite {
        let mut mapping = Rewrite::new();
        self.redirect_edges_to_inner(old, new_target, Some(&mut mapping));
        mapping
    }

    fn redirect_edges_to_inner(
        &mut self,
        old: BlockId,
        new_target: BlockId,
        mapping: Option<&mut Rewrite>,
    ) {
        if old == new_target {
            return;
        }
        let incoming: Vec<EdgeId> = self.incoming(old).collect();
        for &edge in &incoming {
            self.graph.redirect_edge_target(edge, new_target);
        }
        if let Some(mapping) = mapping {
            for edge in incoming {
                mapping.record_edge(edge, [edge]);
            }
        }
    }

    /// Redirect one edge's source while retaining its identity and payload.
    ///
    /// Returns the previous source.
    ///
    /// # Panics
    ///
    /// Panics if the edge is not live or `new_source` has been removed.
    pub fn redirect_edge_source(&mut self, id: EdgeId, new_source: BlockId) -> BlockId {
        self.graph.redirect_edge_source(id, new_source)
    }

    /// Redirect one edge's source and return its stable identity mapping.
    ///
    /// # Panics
    ///
    /// Panics if the edge is not live or `new_source` has been removed.
    pub fn redirect_edge_source_mapped(
        &mut self,
        id: EdgeId,
        new_source: BlockId,
    ) -> (BlockId, Rewrite) {
        let old_source = self.redirect_edge_source(id, new_source);
        let mut mapping = Rewrite::new();
        if old_source != new_source {
            mapping.record_edge(id, [id]);
        }
        (old_source, mapping)
    }

    /// Redirect one edge's target while retaining its identity and payload.
    ///
    /// Returns the previous target.
    ///
    /// # Panics
    ///
    /// Panics if the edge is not live or `new_target` has been removed.
    pub fn redirect_edge_target(&mut self, id: EdgeId, new_target: BlockId) -> BlockId {
        self.graph.redirect_edge_target(id, new_target)
    }

    /// Redirect one edge's target and return its stable identity mapping.
    ///
    /// # Panics
    ///
    /// Panics if the edge is not live or `new_target` has been removed.
    pub fn redirect_edge_target_mapped(
        &mut self,
        id: EdgeId,
        new_target: BlockId,
    ) -> (BlockId, Rewrite) {
        let old_target = self.redirect_edge_target(id, new_target);
        let mut mapping = Rewrite::new();
        if old_target != new_target {
            mapping.record_edge(id, [id]);
        }
        (old_target, mapping)
    }

    /// Move every outgoing edge of `old` to `new_source` in adjacency order.
    ///
    /// Only the source endpoint changes: edge identities, targets, kinds,
    /// weights, payloads, and predecessor adjacency remain intact.
    pub(crate) fn move_outgoing_edges(&mut self, old: BlockId, new_source: BlockId) {
        if old == new_source {
            return;
        }
        let outgoing: Vec<EdgeId> = self.outgoing(old).collect();
        for edge in outgoing {
            self.graph.redirect_edge_source(edge, new_source);
        }
    }

    /// Mutable access to an edge's kind, weight, and consumer payload.
    ///
    /// Endpoints are not mutable here: moving an edge is
    /// [`redirect_edge_source`](Self::redirect_edge_source) or
    /// [`redirect_edge_target`](Self::redirect_edge_target), which keep the
    /// adjacency index in step.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range or has been removed.
    #[inline]
    pub fn edge_mut(&mut self, id: EdgeId) -> &mut Edge<E> {
        assert!(self.graph.contains_edge(id), "edge {id} has been removed");
        self.graph.edge_mut(id).payload_mut()
    }
}
