//! Exception-region structuring for the AST lifter.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use super::super::node::{AstNode, CatchHandler};
use super::{LiftState, effective_merge, flow_successors, lift_block, lift_region};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::region::{HandlerKind, HandlerRef, Region};

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
) -> BTreeMap<u32, Vec<&'c Region>> {
    let mut position: BTreeMap<u32, usize> = BTreeMap::new();
    for (index, block) in order.iter().enumerate() {
        position.insert(block.raw(), index);
    }
    let mut anchors: BTreeMap<u32, Vec<&Region>> = BTreeMap::new();
    for region in cfg.regions() {
        let anchor = region
            .protected_blocks
            .iter()
            .min_by_key(|block| position.get(&block.raw()).copied().unwrap_or(usize::MAX))
            .copied();
        if let Some(anchor) = anchor {
            anchors.entry(anchor.raw()).or_default().push(region);
        }
    }
    for candidates in anchors.values_mut() {
        candidates.sort_by_key(|region| core::cmp::Reverse(region.protected_blocks.len()));
    }
    anchors
}

/// Handler entries and filter funclets: sweep roots that can carry code
/// without any incoming CFG edge.
pub(super) fn metadata_roots<I, E>(cfg: &Cfg<I, E>) -> Vec<BlockId> {
    cfg.regions()
        .iter()
        .flat_map(|region| region.handlers.iter())
        .flat_map(|handler| {
            let filter = match handler.kind {
                HandlerKind::Filter { filter_block } => Some(filter_block),
                _ => None,
            };
            core::iter::once(handler.entry).chain(filter)
        })
        .collect()
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

/// Lift a try/catch region anchored at `block`, if a structurable one exists.
///
/// Among the regions anchored at `block` (outermost first), the first with
/// complete handler extents is structured; regions with unknown extents
/// remain ordinary control flow. Bounds intersect with the enclosing
/// `allowed_blocks`, so a nested region never escapes the extent it was
/// entered under.
pub(super) fn lift_try_catch<'a, I, E, O>(
    cfg: &'a Cfg<I, E>,
    state: &mut LiftState<'_>,
    block: BlockId,
    allowed_blocks: Option<&BTreeSet<BlockId>>,
    map: &mut impl FnMut(&'a I) -> O,
) -> Option<(AstNode<O>, Option<BlockId>)> {
    let candidates = state.anchors.get(&block.raw())?;
    let (region, handler_bodies) = candidates
        .iter()
        .filter(|region| !state.structured_regions.contains(&region.id.0))
        .find_map(|region| complete_handler_bodies(region).map(|bodies| (*region, bodies)))?;
    state.structured_regions.insert(region.id.0);

    // The region's normal continuation: the unique block ordinary flow
    // reconverges at, or the anchor's in-bound post-dominator when the
    // exits diverge. Inner walks stop there silently — the region's
    // `leave` is implicit, not a goto — and the enclosing walk resumes
    // there after the structured node.
    let continuation = region_continuation(cfg, region).or_else(|| {
        effective_merge(cfg, state, block, allowed_blocks).filter(|merge| {
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
    let mut try_body = Vec::new();
    if let Some((inner, inner_continuation)) =
        lift_try_catch(cfg, state, block, Some(&try_bound), map)
    {
        try_body.push(inner);
        if let Some(next) = inner_continuation.filter(|&next| !state.is_visited(next)) {
            try_body.extend(lift_region(
                cfg,
                state,
                next,
                Some(&try_bound),
                continuation,
                map,
            ));
        }
    } else {
        let next = lift_block(
            cfg,
            state,
            block,
            Some(&try_bound),
            continuation,
            &mut try_body,
            map,
        );
        if let Some(next) = next {
            if !state.is_visited(next) {
                try_body.extend(lift_region(
                    cfg,
                    state,
                    next,
                    Some(&try_bound),
                    continuation,
                    map,
                ));
            }
        }
    }
    // Protected successors the classified walk did not reach (for example
    // the targets of a multi-successor linear anchor) still belong to the
    // try body.
    for succ in cfg.successors(block) {
        if region.protected_blocks.contains(&succ) && !state.is_visited(succ) {
            try_body.extend(lift_region(
                cfg,
                state,
                succ,
                Some(&try_bound),
                continuation,
                map,
            ));
        }
    }

    let (handlers, finally_body) = lift_handlers(
        cfg,
        state,
        region,
        handler_bodies,
        allowed_blocks,
        continuation,
        map,
    );

    Some((
        AstNode::TryCatch {
            try_body,
            handlers,
            finally_body,
        },
        continuation,
    ))
}

fn lift_handlers<'a, I, E, O>(
    cfg: &'a Cfg<I, E>,
    state: &mut LiftState<'_>,
    region: &Region,
    handler_bodies: Vec<&BTreeSet<BlockId>>,
    allowed_blocks: Option<&BTreeSet<BlockId>>,
    continuation: Option<BlockId>,
    map: &mut impl FnMut(&'a I) -> O,
) -> (Vec<CatchHandler<O>>, Vec<AstNode<O>>) {
    let mut handlers = Vec::new();
    let mut finally_body = Vec::new();
    for ((index, handler), handler_blocks) in region.handlers.iter().enumerate().zip(handler_bodies)
    {
        // A synthetic landing entry belongs to its handler even when the
        // enclosing bound does not name it.
        let mut handler_bound = intersect_bounds(allowed_blocks, handler_blocks);
        handler_bound.insert(handler.entry);
        let body = lift_region(
            cfg,
            state,
            handler.entry,
            Some(&handler_bound),
            continuation,
            map,
        );
        if handler.kind == HandlerKind::Finally {
            // Multiple finally handlers concatenate in declaration order.
            finally_body.extend(body);
        } else {
            handlers.push(CatchHandler {
                handler: HandlerRef::new(region.id, index),
                entry: handler.entry,
                kind: handler.kind,
                body,
            });
        }
    }
    (handlers, finally_body)
}
