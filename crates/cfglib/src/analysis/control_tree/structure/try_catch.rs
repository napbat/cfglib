//! Exception-region structuring for the control-tree structurer.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use super::super::node::{CatchHandler, ControlNode};
use super::{Structurer, flow_successors};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::traverse::{TraversalDirection, reverse_postorder};
use crate::region::{Handler, HandlerKind, HandlerRef, Region};

/// The blocks of `inner` also admitted by the enclosing bound, so a nested
/// region never escapes the extent it was entered under.
fn intersect_bounds(
    outer: Option<&BTreeSet<BlockId>>,
    inner: &BTreeSet<BlockId>,
) -> BTreeSet<BlockId> {
    match outer {
        None => inner.clone(),
        Some(outer) => inner.intersection(outer).copied().collect(),
    }
}

/// Group regions by anchor — their first protected block in reverse
/// postorder, i.e. the region's entry in flow order rather than by block id.
pub(super) fn region_anchors<'c, I, E>(
    cfg: &'c Cfg<I, E>,
    order: &[BlockId],
) -> BTreeMap<BlockId, Vec<&'c Region>> {
    let mut position: BTreeMap<BlockId, usize> = BTreeMap::new();
    for (index, &block) in order.iter().enumerate() {
        position.insert(block, index);
    }
    let mut anchors: BTreeMap<BlockId, Vec<&Region>> = BTreeMap::new();
    for region in cfg.regions() {
        let anchor = region
            .protected_blocks
            .iter()
            .min_by_key(|block| position.get(block).copied().unwrap_or(usize::MAX))
            .copied();
        if let Some(anchor) = anchor {
            anchors.entry(anchor).or_default().push(region);
        }
    }
    for candidates in anchors.values_mut() {
        candidates.sort_by_key(|region| core::cmp::Reverse(region.protected_blocks.len()));
    }
    anchors
}

/// The blocks the handlers of every region own, in the order the sweep
/// reaches them: each handler's entry, its filter funclet, then the blocks of
/// its known extent that carry code, in flow order.
///
/// A handler body is source code (a catch clause, a finally block, a filter
/// predicate) even when no exception edge enters the pad, and a funclet is
/// invoked by the runtime, so these blocks carry code without any incoming
/// edge from the entry. Protected blocks are not handler code: one the entry
/// cannot reach is dead code.
pub(super) fn handler_roots<I, E>(
    cfg: &Cfg<I, E>,
    carries_code: impl Fn(BlockId) -> bool,
) -> Vec<BlockId> {
    let mut listed: BTreeSet<BlockId> = BTreeSet::new();
    let mut roots = Vec::new();
    for region in cfg.regions() {
        for handler in &region.handlers {
            for block in handler_blocks_in_order(cfg, handler, &carries_code) {
                if listed.insert(block) {
                    roots.push(block);
                }
            }
        }
    }
    roots
}

/// One handler's blocks: its entry, its filter funclet, then the blocks of
/// its known extent that carry code: those the entry reaches in reverse
/// postorder, then the rest by block.
///
/// An extent block without code, such as the jump a source `leave` compiles
/// to, holds nothing to emit; walking from it would only reach the code that
/// follows the handler.
fn handler_blocks_in_order<I, E>(
    cfg: &Cfg<I, E>,
    handler: &Handler,
    carries_code: &impl Fn(BlockId) -> bool,
) -> Vec<BlockId> {
    let mut listed: BTreeSet<BlockId> = BTreeSet::new();
    let mut blocks = alloc::vec![handler.entry];
    listed.insert(handler.entry);
    if let HandlerKind::Filter { filter_block } = handler.kind
        && listed.insert(filter_block)
    {
        blocks.push(filter_block);
    }
    if let Some(extent) = handler.body.blocks() {
        let flow = reverse_postorder(cfg, handler.entry, TraversalDirection::Outgoing);
        for block in flow.into_iter().chain(extent.iter().copied()) {
            if extent.contains(&block) && carries_code(block) && listed.insert(block) {
                blocks.push(block);
            }
        }
    }
    blocks
}

/// The complete handler extents of `region`, or `None` when any handler's
/// extent is unknown and the region must stay unstructured.
///
/// A [`HandlerBody::Known`](crate::region::HandlerBody::Known) that omits
/// its own entry is a frontend bug, distinct from a deliberate `Unknown`:
/// it debug-panics, and degrades to unstructured flow in release builds.
fn complete_handler_bodies(region: &Region) -> Option<Vec<&BTreeSet<BlockId>>> {
    region
        .handlers
        .iter()
        .map(|handler| match handler.body.blocks() {
            Some(blocks) if blocks.contains(&handler.entry) => Some(blocks),
            Some(_) => {
                debug_assert!(
                    false,
                    "HandlerBody::Known must contain its own handler entry"
                );
                None
            }
            None => None,
        })
        .collect()
}

/// The region's normal continuation: the unique target of ordinary flow
/// leaving the region's own blocks — protected and handler alike. `None`
/// when the exits diverge (or none exist), leaving the walk to its
/// conservative boundary handling.
fn region_continuation<I, E>(cfg: &Cfg<I, E>, region: &Region) -> Option<BlockId> {
    let mut inside: BTreeSet<BlockId> = region.protected_blocks.clone();
    for handler in &region.handlers {
        inside.insert(handler.entry);
        if let Some(blocks) = handler.body.blocks() {
            inside.extend(blocks.iter().copied());
        }
    }
    let mut exits: BTreeSet<BlockId> = BTreeSet::new();
    for &block in &inside {
        for target in flow_successors(cfg, block) {
            if !inside.contains(&target) {
                exits.insert(target);
            }
        }
    }
    if exits.len() == 1 {
        exits.into_iter().next()
    } else {
        None
    }
}

impl<I, E> Structurer<'_, I, E> {
    /// Lift a try/catch region anchored at `block`, if a structurable one
    /// exists.
    ///
    /// Among the regions anchored at `block` (outermost first), the first with
    /// complete handler extents is structured; regions with unknown extents
    /// remain ordinary control flow. Bounds intersect with the enclosing
    /// `allowed_blocks`, so a nested region never escapes the extent it was
    /// entered under.
    pub(super) fn lift_try_catch(
        &mut self,
        block: BlockId,
        allowed_blocks: Option<&BTreeSet<BlockId>>,
    ) -> Option<(ControlNode, Option<BlockId>)> {
        let cfg = self.cfg;
        let candidates = self.anchors.get(&block)?;
        let (region, handler_bodies) = candidates
            .iter()
            .filter(|region| !self.structured_regions.contains(&region.id.0))
            .find_map(|region| complete_handler_bodies(region).map(|bodies| (*region, bodies)))?;
        self.structured_regions.insert(region.id.0);

        // The region's normal continuation: the unique block ordinary flow
        // reconverges at, or the anchor's in-bound post-dominator when the
        // exits diverge. Inner walks stop there silently — the region's
        // `leave` is implicit, not a goto — and the enclosing walk resumes
        // there after the structured node.
        let continuation = region_continuation(cfg, region).or_else(|| {
            self.effective_merge(block, allowed_blocks).filter(|merge| {
                !region.protected_blocks.contains(merge)
                    && region.handlers.iter().all(|handler| {
                        handler.entry != *merge
                            && handler
                                .body
                                .blocks()
                                .is_none_or(|blocks| !blocks.contains(merge))
                    })
            })
        });

        // Lift the try body. A nested region can share this anchor (an inner
        // try starting at the same instruction), so the region check runs
        // again first — each pass structures exactly one more region, so the
        // recursion is bounded by the region count. Otherwise the anchor goes
        // through the shared classification path, so an anchor that is itself
        // a branch, dispatch, or loop header keeps its structure.
        let try_bound = intersect_bounds(allowed_blocks, &region.protected_blocks);
        let mut try_nodes = Vec::new();
        if let Some((inner, inner_continuation)) = self.lift_try_catch(block, Some(&try_bound)) {
            try_nodes.push(inner);
            if let Some(next) = inner_continuation.filter(|&next| !self.is_visited(next)) {
                try_nodes.extend(self.lift_region(next, Some(&try_bound), continuation));
            }
        } else {
            let next = self.lift_block(block, Some(&try_bound), continuation, &mut try_nodes);
            if let Some(next) = next
                && !self.is_visited(next)
            {
                try_nodes.extend(self.lift_region(next, Some(&try_bound), continuation));
            }
        }
        // Protected successors the classified walk did not reach (for example
        // the targets of a multi-successor linear anchor) still belong to the
        // try body.
        for succ in cfg.successors(block) {
            if region.protected_blocks.contains(&succ) && !self.is_visited(succ) {
                try_nodes.extend(self.lift_region(succ, Some(&try_bound), continuation));
            }
        }

        let (handlers, finally_nodes) =
            self.lift_handlers(region, handler_bodies, allowed_blocks, continuation);

        Some((
            ControlNode::TryCatch {
                try_body: self.tables.seal(try_nodes),
                handlers: self.tables.push_handlers(handlers),
                finally_body: self.tables.seal(finally_nodes),
            },
            continuation,
        ))
    }

    fn lift_handlers(
        &mut self,
        region: &Region,
        handler_bodies: Vec<&BTreeSet<BlockId>>,
        allowed_blocks: Option<&BTreeSet<BlockId>>,
        continuation: Option<BlockId>,
    ) -> (Vec<CatchHandler>, Vec<ControlNode>) {
        let cfg = self.cfg;
        let mut handlers = Vec::new();
        let mut finally_nodes = Vec::new();
        for ((index, handler), handler_blocks) in
            region.handlers.iter().enumerate().zip(handler_bodies)
        {
            // A synthetic landing entry belongs to its handler even when the
            // enclosing bound does not name it. So do the extent blocks the
            // entry cannot reach: they are the handler's own dead code, and
            // only a block that feeds an enclosing loop is in that loop's
            // bound.
            let mut handler_bound = intersect_bounds(allowed_blocks, handler_blocks);
            handler_bound.insert(handler.entry);
            handler_bound.extend(
                handler_blocks
                    .iter()
                    .copied()
                    .filter(|block| !self.reachable[block.index()]),
            );
            let mut nodes = self.lift_region(handler.entry, Some(&handler_bound), continuation);
            // The walk ends where the pad's flow splits without a branch (a
            // suspension with an exit and a resume edge). The rest of the
            // extent that the entry cannot reach follows in flow order; a
            // reachable block waits for the sweep that always emitted it, and
            // a pad of a nested region waits for that region.
            let owned =
                handler_blocks_in_order(cfg, handler, &|block| self.instruction_count(block) > 0);
            for block in owned {
                if handler_blocks.contains(&block)
                    && !self.reachable[block.index()]
                    && !self.is_visited(block)
                    && !self.awaits_its_handler(region, block)
                {
                    nodes.extend(self.lift_region(block, Some(&handler_bound), continuation));
                }
            }
            if handler.kind == HandlerKind::Finally {
                // Multiple finally handlers concatenate in declaration order.
                finally_nodes.extend(nodes);
            } else {
                handlers.push(CatchHandler {
                    handler: HandlerRef::new(region.id, index),
                    entry: handler.entry,
                    kind: handler.kind,
                    body: self.tables.seal(nodes),
                });
            }
        }
        (handlers, finally_nodes)
    }

    /// Whether `block` is a pad, or in the extent, of a handler of a region
    /// other than `region` that the walk has not structured yet. That region
    /// lifts the block when it reaches the region's anchor.
    fn awaits_its_handler(&self, region: &Region, block: BlockId) -> bool {
        self.cfg.regions().iter().any(|other| {
            other.id != region.id
                && !self.structured_regions.contains(&other.id.0)
                && other.handlers.iter().any(|handler| {
                    handler.entry == block
                        || handler
                            .body
                            .blocks()
                            .is_some_and(|blocks| blocks.contains(&block))
                })
        })
    }
}
