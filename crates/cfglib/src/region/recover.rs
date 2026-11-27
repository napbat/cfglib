//! Recovery of unknown handler extents.
//!
//! Two independent recoveries serve different evidence standards.
//! [`promote_handler_extents`] claims the blocks *dominated* by each handler
//! entry — sound, mutating, and silent about why the rest stayed out.
//! [`recover_exclusive_extents`] instead claims the blocks *exclusively
//! reachable* from each handler entry along normal edges, without mutating
//! anything, and reports its boundary and ambiguity evidence so a caller can
//! decide; [`promote_exclusive_extents`] then applies such extents.
//!
//! Both promotions go through one core that applies candidate bodies
//! atomically per region: a region is promoted only when every one of its
//! handlers has an unambiguous body that holds its own entry and no promoted
//! body anywhere overlaps another, because structured lifting owns a distinct
//! body per handler arm.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{Edge, EdgeId};
use crate::graph::dominator::DominatorTree;
use crate::graph::edge_view::FilteredEdges;
use crate::graph::traverse::reachable_in;
use crate::graph::view::Direction;
use crate::region::{Handler, HandlerBody, HandlerRef, RegionId};

/// Promotes every reachable [`HandlerBody::Unknown`] extent to the blocks
/// dominated by its handler entry, atomically per region, and returns the
/// decision for every region in region order.
///
/// A block dominated by the handler entry is reachable only through the
/// handler, so the dominated set is a **sound** extent: it can under-cover
/// (code shared with another handler or with normal flow stays outside and
/// degrades to explicit gotos during structuring) but never claims a block
/// the handler does not own. A handler of known extent proposes its own
/// body. An unreachable handler entry proposes nothing, so its region stays
/// as it is ([`ExtentPromotionStatus::AmbiguousExtent`]); so does a region
/// whose body would overlap another handler's
/// ([`ExtentPromotionStatus::OverlappingExtents`]), as a handler entry
/// dominated by another handler's entry does. See
/// [`promote_exclusive_extents`] for the shared rule.
///
/// Run this on a **derived** graph (a clone lifted for presentation), not
/// on canonical frontend metadata: exact tables that encode only a handler
/// entry should keep saying so.
pub fn promote_handler_extents<I, E>(
    cfg: &mut Cfg<I, E>,
) -> Vec<(RegionId, ExtentPromotionStatus)> {
    let dominators = DominatorTree::compute(cfg);
    let block_ids: Vec<BlockId> = cfg.block_ids().collect();
    promote(cfg, |_, handler| match handler.body.blocks() {
        Some(blocks) => Some(blocks.clone()),
        None => dominators.is_reachable(handler.entry).then(|| {
            block_ids
                .iter()
                .copied()
                .filter(|&block| dominators.dominates(handler.entry, block))
                .collect()
        }),
    })
}

/// Why one exclusive-reachability extent cannot be claimed unambiguously.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExtentIssue {
    /// Ordinary flow from the function entry can reach the handler entry.
    EntryReachableFromNormalFlow,
    /// Another distinct handler entry can reach this handler entry normally.
    EntryReachableFromAnotherHandler,
    /// An exclusively owned interior block has a normal predecessor outside
    /// the recovered body.
    ExternalEntry {
        /// Interior block receiving the external edge.
        block: BlockId,
        /// Source block outside the recovered body.
        predecessor: BlockId,
    },
}

/// Conservative block ownership and boundary evidence for one handler body.
///
/// [`Self::blocks`] contains only blocks reachable from this handler's entry
/// along normal edges that are unreachable that way from the function entry
/// and from every other distinct handler entry. Shared tails are reported in
/// [`Self::boundary_blocks`] rather than claimed: an unambiguous extent with
/// no boundary block is closed under normal flow, one with boundary blocks
/// flows into shared continuations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExclusiveExtent {
    /// The handler this evidence describes.
    pub handler: HandlerRef,
    /// The handler's entry block.
    pub entry: BlockId,
    /// Blocks exclusively reachable from this handler's entry.
    pub blocks: BTreeSet<BlockId>,
    /// Direct normal successors deliberately excluded because they are
    /// shared.
    pub boundary_blocks: BTreeSet<BlockId>,
    /// Deterministically ordered reasons the extent is ambiguous.
    pub issues: Vec<ExtentIssue>,
}

impl ExclusiveExtent {
    /// Whether a structural fact prevents an unambiguous boundary: the
    /// extent has an [`ExtentIssue`]. [`promote_exclusive_extents`] does not
    /// apply an ambiguous extent.
    #[must_use]
    pub fn is_ambiguous(&self) -> bool {
        !self.issues.is_empty()
    }
}

/// Recovers every handler's exclusive-reachability extent without mutating.
///
/// `is_normal` decides which edges ordinary control flow can take; every
/// other edge is invisible to the recovery. Pass
/// `|_, edge| !edge.kind().is_exceptional()` to read normality from the
/// [`EdgeKind`](crate::EdgeKind), or read it from the consumer's edge
/// payload. Extents come back in region-then-handler order.
#[must_use]
pub fn recover_exclusive_extents<I, E>(
    cfg: &Cfg<I, E>,
    is_normal: impl Fn(EdgeId, &Edge<E>) -> bool,
) -> Vec<ExclusiveExtent> {
    let normal = FilteredEdges::new(cfg, &is_normal);
    let mut stack = Vec::new();
    let mut method_reachable = Vec::new();
    reachable_in(
        &normal,
        [cfg.entry()],
        Direction::Forward,
        None,
        &mut method_reachable,
        &mut stack,
    );

    // What each distinct handler entry reaches normally, sorted by entry, the
    // walks sharing one stack. A sorted vector rather than a set of entries
    // and a map of walks, and the handlers read twice rather than collected:
    // with the set, the map and the list, `api_recover_exclusive_extents`
    // took 1.11x its time on main, and it takes 0.88x in this form.
    let mut entry_reachability: Vec<(BlockId, Vec<bool>)> = cfg
        .handlers()
        .map(|(_, stored)| (stored.entry, Vec::new()))
        .collect();
    entry_reachability.sort_unstable_by_key(|&(entry, _)| entry);
    entry_reachability.dedup_by_key(|(entry, _)| *entry);
    for (entry, reached) in &mut entry_reachability {
        reachable_in(
            &normal,
            [*entry],
            Direction::Forward,
            None,
            reached,
            &mut stack,
        );
    }

    cfg.handlers()
        .map(|(handler, stored)| {
            let entry = stored.entry;
            // Every handler entry is listed, so the first position not below
            // `entry` holds its walk.
            let found = entry_reachability.partition_point(|&(other, _)| other < entry);
            let from_entry = &entry_reachability[found].1;
            let blocks: BTreeSet<BlockId> = cfg
                .block_ids()
                .filter(|block| {
                    from_entry[block.index()]
                        && !method_reachable[block.index()]
                        && entry_reachability
                            .iter()
                            .all(|(other, reachable)| *other == entry || !reachable[block.index()])
                })
                .collect();

            let mut boundary_blocks = BTreeSet::new();
            if !blocks.contains(&entry) {
                boundary_blocks.insert(entry);
            }
            for &block in &blocks {
                for edge in cfg.outgoing(block) {
                    let edge = cfg.edge(edge);
                    if is_normal(edge.id(), edge.data()) && !blocks.contains(&edge.target()) {
                        boundary_blocks.insert(edge.target());
                    }
                }
            }

            let mut issues = BTreeSet::new();
            if method_reachable[entry.index()] {
                issues.insert(ExtentIssue::EntryReachableFromNormalFlow);
            }
            if entry_reachability
                .iter()
                .any(|(other, reachable)| *other != entry && reachable[entry.index()])
            {
                issues.insert(ExtentIssue::EntryReachableFromAnotherHandler);
            }
            for &block in &blocks {
                if block == entry {
                    continue;
                }
                for edge in cfg.incoming(block) {
                    let edge = cfg.edge(edge);
                    if is_normal(edge.id(), edge.data()) && !blocks.contains(&edge.source()) {
                        issues.insert(ExtentIssue::ExternalEntry {
                            block,
                            predecessor: edge.source(),
                        });
                    }
                }
            }

            ExclusiveExtent {
                handler,
                entry,
                blocks,
                boundary_blocks,
                issues: issues.into_iter().collect(),
            }
        })
        .collect()
}

/// Whether and why one region's handler extents were promoted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtentPromotionStatus {
    /// Every handler received its unambiguous, non-overlapping body.
    Promoted,
    /// One handler produced no candidate body, or a body that leaves out the
    /// handler's own entry and so cannot be its complete extent.
    AmbiguousExtent {
        /// Handler that prevents promotion of the complete region.
        handler: HandlerRef,
    },
    /// Two handlers claim at least one common body block.
    ///
    /// Shared handler code is valid input, but structured lifting owns a
    /// distinct body per handler arm, so neither claim is applied.
    OverlappingExtents {
        /// First overlapping handler in stable identity order.
        first: HandlerRef,
        /// Second overlapping handler in stable identity order.
        second: HandlerRef,
    },
}

/// Applies exclusive extents as handler bodies, atomically per region, and
/// returns the decision for every region in region order.
///
/// Each handler's candidate is the first extent of `extents` naming it; an
/// [ambiguous](ExclusiveExtent::is_ambiguous) extent, a missing one, or one
/// whose blocks omit the handler's entry withholds the whole region. A body
/// overlapping any other candidate body, in the same region or another,
/// withholds both regions, because structured lifting owns a distinct body per
/// handler arm. Every remaining region has each handler's body set to
/// [`HandlerBody::Known`]; a region without handlers promotes vacuously.
///
/// Run this on a **derived** graph (a clone lifted for presentation), not
/// on canonical frontend metadata, and pair it with
/// [`recover_exclusive_extents`] (or caller-refined evidence) to produce
/// the extents.
pub fn promote_exclusive_extents<I, E>(
    cfg: &mut Cfg<I, E>,
    extents: &[ExclusiveExtent],
) -> Vec<(RegionId, ExtentPromotionStatus)> {
    let mut by_handler: BTreeMap<HandlerRef, &ExclusiveExtent> = BTreeMap::new();
    for extent in extents {
        by_handler.entry(extent.handler).or_insert(extent);
    }
    promote(cfg, |handler, _| {
        by_handler
            .get(&handler)
            .filter(|extent| !extent.is_ambiguous())
            .map(|extent| extent.blocks.clone())
    })
}

/// The promotion core: `candidate` proposes each handler's body, or `None`
/// when there is no unambiguous one.
fn promote<I, E>(
    cfg: &mut Cfg<I, E>,
    mut candidate: impl FnMut(HandlerRef, &Handler) -> Option<BTreeSet<BlockId>>,
) -> Vec<(RegionId, ExtentPromotionStatus)> {
    let mut candidates: BTreeMap<RegionId, Vec<(HandlerRef, BTreeSet<BlockId>)>> = BTreeMap::new();
    let mut statuses: BTreeMap<RegionId, ExtentPromotionStatus> = BTreeMap::new();
    for (region, stored) in cfg.regions() {
        let mut handlers = Vec::with_capacity(stored.handlers.len());
        let mut ambiguous = None;
        for (index, handler) in stored.handlers.iter().enumerate() {
            let reference = HandlerRef::new(region, index);
            match candidate(reference, handler) {
                Some(blocks) if blocks.contains(&handler.entry) => {
                    handlers.push((reference, blocks));
                }
                _ => {
                    ambiguous = Some(reference);
                    break;
                }
            }
        }
        if let Some(handler) = ambiguous {
            statuses.insert(region, ExtentPromotionStatus::AmbiguousExtent { handler });
        } else {
            candidates.insert(region, handlers);
        }
    }

    let flat: Vec<(HandlerRef, &BTreeSet<BlockId>)> = candidates
        .values()
        .flatten()
        .map(|(handler, blocks)| (*handler, blocks))
        .collect();
    for (position, &(first, first_blocks)) in flat.iter().enumerate() {
        for &(second, second_blocks) in &flat[position + 1..] {
            if first_blocks.is_disjoint(second_blocks) {
                continue;
            }
            let status = ExtentPromotionStatus::OverlappingExtents { first, second };
            statuses.entry(first.region()).or_insert(status);
            statuses.entry(second.region()).or_insert(status);
        }
    }
    for (region, handlers) in candidates {
        if statuses.contains_key(&region) {
            continue;
        }
        for (handler, blocks) in handlers {
            cfg.handler_mut(handler).body = HandlerBody::Known(blocks);
        }
        statuses.insert(region, ExtentPromotionStatus::Promoted);
    }

    cfg.regions()
        .map(|(region, _)| {
            // Every region received a status above; an absent entry can only
            // mean the region holds no handlers, which promotes vacuously.
            let status = statuses
                .get(&region)
                .copied()
                .unwrap_or(ExtentPromotionStatus::Promoted);
            (region, status)
        })
        .collect()
}

#[cfg(test)]
mod tests;
