//! Structural mutation of [`Cfg`] — block and edge creation, removal,
//! redirection, and block splitting.
//!
//! The methods change the graph and return what they made or what they
//! replaced; none of them keeps an identity record. A pass that reports
//! the identities it touched records them in its own [`Rewrite`], which is
//! what every `_mapped` pass of [`transform`](crate::transform) does.
//! [`split_block_at_points`](Cfg::split_block_at_points) is the one
//! method that returns a record, because the blocks and edges a multi-point
//! split creates are otherwise hard to name.

extern crate alloc;

use alloc::vec::Vec;
use smallvec::SmallVec;

use crate::block::{BasicBlock, BlockId};
use crate::edge::{Edge, EdgeId};
use crate::graph::view::Direction;
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

    /// Add a directed edge and return its id.
    ///
    /// `edge` is anything that converts into an [`Edge`]: a bare
    /// [`EdgeKind`](crate::EdgeKind) when the payload type has a [`Default`],
    /// or an edge built with [`Edge::new`], [`with_weight`](Edge::with_weight)
    /// and [`with_payload`](Edge::with_payload).
    ///
    /// # Panics
    ///
    /// Panics when either block is out of range or has been removed.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, Edge, EdgeKind, Probability};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let exit = cfg.new_block();
    /// let quarter = Probability::from_f64(0.25).expect("0.25 is a probability");
    /// let taken = cfg.add_edge(cfg.entry(), exit, Edge::new(EdgeKind::Jump).with_weight(quarter));
    /// let looped = cfg.add_edge(exit, exit, EdgeKind::Back);
    ///
    /// assert_eq!(cfg.edge(taken).weight(), Some(quarter));
    /// assert_eq!(cfg.edge(looped).weight(), None);
    /// ```
    pub fn add_edge(
        &mut self,
        source: BlockId,
        target: BlockId,
        edge: impl Into<Edge<E>>,
    ) -> EdgeId {
        self.graph.add_edge(source, target, edge.into())
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

    /// Split a block at instruction index `at`.
    ///
    /// Instructions `[at..]` are moved into a new block, every outgoing edge
    /// of the original block is transferred to the new block, and `fallthrough`
    /// is inserted from the original block to the new one: usually
    /// [`EdgeKind::Fallthrough`](crate::EdgeKind::Fallthrough), or an edge of
    /// that kind with a payload.
    ///
    /// The split does not change what any instruction does, so the new block
    /// belongs wherever the original did: it joins every region's protected
    /// set and every known handler body that contains the original, and it
    /// replaces the original as the block a cleanup exits from, because it
    /// now holds the exits. A frontend that wants the boundary to be a
    /// protection boundary edits the region afterwards.
    ///
    /// Returns the id of the newly created block. Use
    /// [`split_block_at_points`](Self::split_block_at_points) to split at
    /// several boundaries at once and to get the identity record.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range or `at > instructions.len()`.
    pub fn split_block(
        &mut self,
        id: BlockId,
        at: usize,
        fallthrough: impl Into<Edge<E>>,
    ) -> BlockId {
        self.split_off(id, at, fallthrough.into()).0
    }

    /// Split one block at ordered instruction boundaries and return every
    /// resulting block plus an identity record.
    ///
    /// Each point comes with the edge inserted in front of the block that
    /// starts there, as for [`split_block`](Self::split_block). Points are
    /// offsets in the original block, not in successively shorter tails.
    /// They must be strictly increasing and may include `0` or the original
    /// instruction count. Validation happens before mutation, so an error
    /// leaves the CFG unchanged. The returned blocks are in execution order
    /// and include `id` as the first element.
    ///
    /// The record maps `id` to every returned block, lists the new blocks and
    /// the inserted edges as created, and maps each outgoing edge of `id` to
    /// itself, since it now leaves the last block.
    ///
    /// # Errors
    ///
    /// Returns [`SplitPointError`] when a point is out of bounds or the points
    /// are not strictly increasing.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range.
    pub fn split_block_at_points<P: Into<Edge<E>>>(
        &mut self,
        id: BlockId,
        points: impl IntoIterator<Item = (usize, P)>,
    ) -> Result<(Vec<BlockId>, Rewrite), SplitPointError> {
        let points: SmallVec<[(usize, P); 4]> = points.into_iter().collect();
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

        // A few points stay on the stack, the blocks are sized once, and the
        // moved edges are read off the last block rather than listed first:
        // with a point list, a growing block list and a moved-edge list, a
        // one-point split (`cfg_split_block_with_payload_mapped`) took 1.15x
        // the time of main's single mapped split; now only the returned block
        // list is left over it.
        let mut rewrite = Rewrite::new();
        let mut blocks = Vec::with_capacity(points.len() + 1);
        blocks.push(id);
        let mut current = id;
        let mut base = 0;
        for (point, fallthrough) in points {
            let (tail, inserted) = self.split_off(current, point - base, fallthrough.into());
            rewrite.record_created_block(tail);
            rewrite.record_created_edge(inserted);
            blocks.push(tail);
            current = tail;
            base = point;
        }
        if current == id {
            return Ok((blocks, rewrite));
        }
        rewrite.record_block(id, blocks.iter().copied());
        // Each split hands every outgoing edge on to its tail, so the last
        // block leaves by exactly the edges `id` left by.
        for edge in self.graph.outgoing(current) {
            rewrite.record_edge(edge, [edge]);
        }
        Ok((blocks, rewrite))
    }

    /// The one split every split method goes through: move `[at..]` and the
    /// outgoing edges of `id` into a new block, and join the two with
    /// `fallthrough`. Returns the new block and the inserted edge.
    fn split_off(&mut self, id: BlockId, at: usize, fallthrough: Edge<E>) -> (BlockId, EdgeId) {
        let tail_instructions: Vec<I> = self.block_mut(id).instructions.split_off(at);
        let tail = self.new_block();
        self.block_mut(tail).instructions = tail_instructions;

        self.move_outgoing_edges(id, tail);
        self.extend_exception_metadata_to_tail(id, tail);

        let inserted = self.graph.add_edge(id, tail, fallthrough);
        (tail, inserted)
    }

    /// Redirect all edges that target `old` to target `new_target` instead.
    ///
    /// Every redirected edge keeps its identity, kind, weight, and payload.
    /// This is useful for bypassing a block before removal.
    ///
    /// # Panics
    ///
    /// Panics if either block has been removed.
    pub fn redirect_edges_to(&mut self, old: BlockId, new_target: BlockId) {
        if old == new_target {
            return;
        }
        self.graph
            .move_adjacency(old, new_target, Direction::Backward);
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

    /// Move every outgoing edge of `old` to `new_source` in adjacency order.
    ///
    /// Only the source endpoint changes: edge identities, targets, kinds,
    /// weights, payloads, and predecessor adjacency remain intact.
    pub(crate) fn move_outgoing_edges(&mut self, old: BlockId, new_source: BlockId) {
        if old == new_source {
            return;
        }
        self.graph
            .move_adjacency(old, new_source, Direction::Forward);
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
