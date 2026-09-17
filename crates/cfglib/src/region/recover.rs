//! Recovery of unknown handler extents.
//!
//! Two independent recoveries serve different evidence standards.
//! [`promote_handler_extents`] claims the blocks *dominated* by each handler
//! entry — sound, mutating, and silent about why the rest stayed out.
//! [`recover_exclusive_extents`] instead claims the blocks *exclusively
//! reachable* from each handler entry along non-exceptional edges, without
//! mutating anything, and reports its boundary and ambiguity evidence so a
//! caller can decide; [`promote_exclusive_extents`] then applies such
//! candidate extents atomically per region — a region is promoted only when
//! every one of its handlers has an unambiguous body and no promoted body
//! anywhere overlaps another, because structured lifting owns a distinct
//! body per handler arm.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{Edge, EdgeId};
use crate::graph::dominator::DominatorTree;
use crate::graph::edge_view::FilteredEdges;
use crate::graph::traverse::{TraversalDirection, reachable};
use crate::region::{HandlerBody, HandlerRef, RegionId};

/// Promotes every reachable [`HandlerBody::Unknown`] extent to the blocks
/// dominated by its handler entry, returning how many were promoted.
///
/// A block dominated by the handler entry is reachable only through the
/// handler, so the dominated set is a **sound** extent: it can under-cover
/// (code shared with another handler or with normal flow stays outside and
/// degrades to explicit gotos during structuring) but never claims a block
/// the handler does not own. Unreachable handler entries stay `Unknown`.
///
/// Run this on a **derived** graph (a clone lifted for presentation), not
/// on canonical frontend metadata: exact tables that encode only a handler
/// entry should keep saying so.
pub fn promote_handler_extents<I, E>(cfg: &mut Cfg<I, E>) -> usize {
    let dominators = DominatorTree::compute(cfg);
    let block_ids: Vec<BlockId> = cfg.block_ids().collect();
    let mut promotions = Vec::new();
    for (region_index, region) in cfg.regions().iter().enumerate() {
        for (handler_index, handler) in region.handlers.iter().enumerate() {
            if handler.body.is_known() {
                continue;
            }
            let entry = handler.entry;
            let reachable = entry == cfg.entry() || dominators.idom(entry).is_some();
            if !reachable {
                continue;
            }
            let extent: BTreeSet<BlockId> = block_ids
                .iter()
                .copied()
                .filter(|&block| dominators.dominates(entry, block))
                .collect();
            promotions.push((region_index, handler_index, extent));
        }
    }
    let promoted = promotions.len();
    for (region_index, handler_index, extent) in promotions {
        let region_id = cfg.regions()[region_index].id;
        if let Some(region) = cfg.region_mut(region_id) {
            if let Some(handler) = region.handlers.get_mut(handler_index) {
                handler.body = HandlerBody::Known(extent);
            }
        }
    }
    promoted
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

/// Confidence category of one exclusive-reachability extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtentStatus {
    /// The handler-owned subgraph is closed under represented normal flow.
    Isolated,
    /// The exclusive blocks flow into one or more shared continuations.
    SharedContinuation,
    /// One or more structural facts prevent an unambiguous boundary.
    Ambiguous,
}

/// Conservative block ownership and boundary evidence for one handler body.
///
/// [`Self::blocks`] contains only blocks reachable from this handler's entry
/// along non-exceptional edges that are unreachable that way from the
/// function entry and from every other distinct handler entry. Shared tails
/// are reported in [`Self::boundary_blocks`] rather than claimed.
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
    /// Classifies the extent from its boundary and ambiguity evidence.
    #[must_use]
    pub fn status(&self) -> ExtentStatus {
        if !self.issues.is_empty() {
            ExtentStatus::Ambiguous
        } else if self.boundary_blocks.is_empty() {
            ExtentStatus::Isolated
        } else {
            ExtentStatus::SharedContinuation
        }
    }
}

/// Recovers every handler's exclusive-reachability extent without mutating.
///
/// Edges whose [`EdgeKind`](crate::EdgeKind) is exceptional are excluded
/// from the walk; use [`recover_exclusive_extents_with`] when normality
/// lives in the consumer's edge payload instead. Extents come back in
/// region-then-handler order.
#[must_use]
pub fn recover_exclusive_extents<I, E>(cfg: &Cfg<I, E>) -> Vec<ExclusiveExtent> {
    recover_exclusive_extents_with(cfg, |_, edge| !edge.kind().is_exceptional())
}

/// Recovers every handler's exclusive-reachability extent under a caller
/// edge classification.
///
/// `is_normal` decides which edges ordinary control flow can take; every
/// other edge is invisible to the recovery. See
/// [`recover_exclusive_extents`] for the extent contract.
#[must_use]
pub fn recover_exclusive_extents_with<I, E>(
    cfg: &Cfg<I, E>,
    is_normal: impl Fn(EdgeId, &Edge<E>) -> bool,
) -> Vec<ExclusiveExtent> {
    let normal = FilteredEdges::new(cfg, &is_normal);
    let method_reachable = reachable(&normal, [cfg.entry()], TraversalDirection::Outgoing);

    let mut handlers: Vec<(HandlerRef, BlockId)> = Vec::new();
    for region in cfg.regions() {
        for (index, handler) in region.handlers.iter().enumerate() {
            handlers.push((HandlerRef::new(region.id, index), handler.entry));
        }
    }
    let entry_reachability: BTreeMap<BlockId, Vec<bool>> = handlers
        .iter()
        .map(|&(_, entry)| entry)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|entry| {
            (
                entry,
                reachable(&normal, [entry], TraversalDirection::Outgoing),
            )
        })
        .collect();

    handlers
        .into_iter()
        .filter_map(|(handler, entry)| {
            let from_entry = entry_reachability.get(&entry)?;
            let blocks: BTreeSet<BlockId> = cfg
                .block_ids()
                .filter(|block| {
                    from_entry[block.index()]
                        && !method_reachable[block.index()]
                        && entry_reachability
                            .iter()
                            .all(|(&other, reachable)| other == entry || !reachable[block.index()])
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
                .any(|(&other, reachable)| other != entry && reachable[entry.index()])
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

            Some(ExclusiveExtent {
                handler,
                entry,
                blocks,
                boundary_blocks,
                issues: issues.into_iter().collect(),
            })
        })
        .collect()
}

/// Whether and why one region's handler extents were promoted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtentPromotionStatus {
    /// Every handler received its unambiguous, non-overlapping body.
    Promoted,
    /// One handler produced no candidate body.
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

/// Promotion decision for one exception region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExtentPromotionDecision {
    /// Stable region identity in the promoted graph.
    pub region: RegionId,
    /// Whether and why the region was promoted.
    pub status: ExtentPromotionStatus,
}

/// Applies candidate handler bodies atomically per region.
///
/// `candidate` answers each handler's proposed body, or `None` when the
/// caller judges it ambiguous; the first `None` in a region withholds the
/// whole region, and a body overlapping any other candidate body — in the
/// same region or another — withholds both regions, because structured
/// lifting owns a distinct body per handler arm. Every remaining region has
/// each handler's body set to [`HandlerBody::Known`]. Decisions come back
/// in region order.
///
/// Run this on a **derived** graph (a clone lifted for presentation), not
/// on canonical frontend metadata, and pair it with
/// [`recover_exclusive_extents`] (or caller-refined evidence) to produce
/// the candidates.
pub fn promote_exclusive_extents<I, E>(
    cfg: &mut Cfg<I, E>,
    mut candidate: impl FnMut(HandlerRef) -> Option<BTreeSet<BlockId>>,
) -> Vec<ExtentPromotionDecision> {
    let mut candidates: BTreeMap<RegionId, Vec<(HandlerRef, BTreeSet<BlockId>)>> = BTreeMap::new();
    let mut statuses: BTreeMap<RegionId, ExtentPromotionStatus> = BTreeMap::new();
    for region in cfg.regions() {
        let mut handlers = Vec::with_capacity(region.handlers.len());
        let mut ambiguous = None;
        for index in 0..region.handlers.len() {
            let handler = HandlerRef::new(region.id, index);
            if let Some(blocks) = candidate(handler) {
                handlers.push((handler, blocks));
            } else {
                ambiguous = Some(handler);
                break;
            }
        }
        if let Some(handler) = ambiguous {
            statuses.insert(
                region.id,
                ExtentPromotionStatus::AmbiguousExtent { handler },
            );
        } else {
            candidates.insert(region.id, handlers);
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
    for (&region, handlers) in &candidates {
        if statuses.contains_key(&region) {
            continue;
        }
        for (handler, blocks) in handlers {
            if let Some(stored) = cfg.handler_mut(*handler) {
                stored.body = HandlerBody::known(blocks.iter().copied());
            }
        }
        statuses.insert(region, ExtentPromotionStatus::Promoted);
    }

    cfg.regions()
        .iter()
        .map(|region| ExtentPromotionDecision {
            region: region.id,
            // Every region received a status above; an absent entry can only
            // mean the region holds no handlers, which promotes vacuously.
            status: statuses
                .get(&region.id)
                .copied()
                .unwrap_or(ExtentPromotionStatus::Promoted),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeKind;
    use crate::region::{Handler, HandlerKind, Region, RegionId};
    use crate::test_util::{MockInst, ff};

    #[test]
    fn promotes_exclusive_handler_code_and_keeps_shared_joins_out() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let protected = cfg.new_block();
        let landing = cfg.new_block();
        let handler_tail = cfg.new_block();
        let join = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("entry"));
        cfg.block_mut(protected).push(ff("try_inst"));
        cfg.block_mut(landing).push(ff("caught"));
        cfg.block_mut(handler_tail).push(ff("handler_tail"));
        cfg.block_mut(join).push(ff("after"));
        cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
        cfg.add_edge(protected, join, EdgeKind::Fallthrough);
        cfg.add_edge(protected, landing, EdgeKind::ExceptionUnwind);
        cfg.add_edge(landing, handler_tail, EdgeKind::Fallthrough);
        cfg.add_edge(handler_tail, join, EdgeKind::Fallthrough);
        cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: [protected].into_iter().collect(),
            handlers: alloc::vec![Handler {
                entry: landing,
                body: HandlerBody::Unknown,
                kind: HandlerKind::CatchAll,
            }],
            parent: None,
        });

        assert_eq!(promote_handler_extents(&mut cfg), 1);
        let body = cfg.regions()[0].handlers[0]
            .body
            .blocks()
            .expect("promoted");
        assert!(body.contains(&landing));
        assert!(body.contains(&handler_tail));
        assert!(
            !body.contains(&join),
            "the shared continuation stays outside the extent"
        );

        // A second run finds nothing left to promote.
        assert_eq!(promote_handler_extents(&mut cfg), 0);
    }

    #[test]
    fn unreachable_handler_entries_stay_unknown() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let protected = cfg.new_block();
        let funclet = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("entry"));
        cfg.block_mut(protected).push(ff("try_inst"));
        cfg.block_mut(funclet).push(ff("funclet"));
        cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
        cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: [protected].into_iter().collect(),
            handlers: alloc::vec![Handler {
                entry: funclet,
                body: HandlerBody::Unknown,
                kind: HandlerKind::CatchAll,
            }],
            parent: None,
        });

        assert_eq!(promote_handler_extents(&mut cfg), 0);
        assert!(!cfg.regions()[0].handlers[0].body.is_known());
    }

    fn handler_region(id: u32, entry: crate::BlockId, protected: crate::BlockId) -> Region {
        Region {
            id: RegionId::from_raw(id),
            protected_blocks: [protected].into_iter().collect(),
            handlers: alloc::vec![Handler {
                entry,
                body: HandlerBody::Unknown,
                kind: HandlerKind::CatchAll,
            }],
            parent: None,
        }
    }

    #[test]
    fn exclusive_extents_report_ownership_boundary_and_status() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let protected = cfg.new_block();
        let landing = cfg.new_block();
        let handler_tail = cfg.new_block();
        let join = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("entry"));
        cfg.block_mut(protected).push(ff("try_inst"));
        cfg.block_mut(landing).push(ff("caught"));
        cfg.block_mut(handler_tail).push(ff("handler_tail"));
        cfg.block_mut(join).push(ff("after"));
        cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
        cfg.add_edge(protected, join, EdgeKind::Fallthrough);
        cfg.add_edge(protected, landing, EdgeKind::ExceptionUnwind);
        cfg.add_edge(landing, handler_tail, EdgeKind::Fallthrough);
        cfg.add_edge(handler_tail, join, EdgeKind::Fallthrough);
        cfg.add_region(handler_region(0, landing, protected));

        let extents = recover_exclusive_extents(&cfg);
        assert_eq!(extents.len(), 1);
        let extent = &extents[0];
        assert_eq!(extent.entry, landing);
        assert_eq!(
            extent.blocks,
            [landing, handler_tail].into_iter().collect(),
            "exclusively reachable handler code is claimed"
        );
        assert_eq!(
            extent.boundary_blocks,
            [join].into_iter().collect(),
            "the shared continuation is boundary evidence, not body"
        );
        assert_eq!(extent.issues, alloc::vec![]);
        assert_eq!(extent.status(), ExtentStatus::SharedContinuation);
    }

    #[test]
    fn a_normally_reachable_entry_is_ambiguous() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let protected = cfg.new_block();
        let landing = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("entry"));
        cfg.block_mut(protected).push(ff("try_inst"));
        cfg.block_mut(landing).push(ff("caught"));
        cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
        // Normal flow falls straight into the handler entry.
        cfg.add_edge(protected, landing, EdgeKind::Fallthrough);
        cfg.add_edge(protected, landing, EdgeKind::ExceptionUnwind);
        cfg.add_region(handler_region(0, landing, protected));

        let extents = recover_exclusive_extents(&cfg);
        assert_eq!(
            extents[0].issues,
            alloc::vec![ExtentIssue::EntryReachableFromNormalFlow]
        );
        assert_eq!(extents[0].status(), ExtentStatus::Ambiguous);
        assert!(
            extents[0].blocks.is_empty(),
            "normally reachable code is never claimed"
        );
        assert_eq!(
            extents[0].boundary_blocks,
            [landing].into_iter().collect(),
            "the unowned entry itself is boundary evidence"
        );
    }

    #[test]
    fn promotion_is_atomic_per_region_and_rejects_overlap() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let protected = cfg.new_block();
        let first_landing = cfg.new_block();
        let second_landing = cfg.new_block();
        let shared_tail = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("entry"));
        cfg.block_mut(protected).push(ff("try_inst"));
        cfg.block_mut(first_landing).push(ff("first"));
        cfg.block_mut(second_landing).push(ff("second"));
        cfg.block_mut(shared_tail).push(ff("shared"));
        cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
        cfg.add_edge(protected, first_landing, EdgeKind::ExceptionUnwind);
        cfg.add_edge(protected, second_landing, EdgeKind::ExceptionUnwind);
        cfg.add_edge(first_landing, shared_tail, EdgeKind::Fallthrough);
        cfg.add_edge(second_landing, shared_tail, EdgeKind::Fallthrough);
        cfg.add_region(handler_region(0, first_landing, protected));
        cfg.add_region(handler_region(1, second_landing, protected));

        // Overlapping candidate bodies withhold both regions.
        let overlapping: BTreeSet<crate::BlockId> =
            [first_landing, shared_tail].into_iter().collect();
        let also_overlapping: BTreeSet<crate::BlockId> =
            [second_landing, shared_tail].into_iter().collect();
        let decisions = promote_exclusive_extents(&mut cfg, |handler| {
            Some(if handler.region().index() == 0 {
                overlapping.clone()
            } else {
                also_overlapping.clone()
            })
        });
        assert!(
            decisions.iter().all(|decision| matches!(
                decision.status,
                ExtentPromotionStatus::OverlappingExtents { .. }
            )),
            "a shared tail withholds both regions"
        );
        assert!(!cfg.regions()[0].handlers[0].body.is_known());
        assert!(!cfg.regions()[1].handlers[0].body.is_known());

        // An ambiguous handler withholds only its own region.
        let exclusive: BTreeSet<crate::BlockId> = [second_landing].into_iter().collect();
        let decisions = promote_exclusive_extents(&mut cfg, |handler| {
            (handler.region().index() == 1).then(|| exclusive.clone())
        });
        assert!(matches!(
            decisions[0].status,
            ExtentPromotionStatus::AmbiguousExtent { .. }
        ));
        assert_eq!(decisions[1].status, ExtentPromotionStatus::Promoted);
        assert!(!cfg.regions()[0].handlers[0].body.is_known());
        assert_eq!(cfg.regions()[1].handlers[0].body.blocks(), Some(&exclusive));
    }

    #[test]
    fn recovered_extents_feed_promotion_end_to_end() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let protected = cfg.new_block();
        let landing = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("entry"));
        cfg.block_mut(protected).push(ff("try_inst"));
        cfg.block_mut(landing).push(ff("caught"));
        cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
        cfg.add_edge(protected, landing, EdgeKind::ExceptionUnwind);
        cfg.add_region(handler_region(0, landing, protected));

        let extents = recover_exclusive_extents(&cfg);
        let decisions = promote_exclusive_extents(&mut cfg, |handler| {
            extents
                .iter()
                .find(|extent| extent.handler == handler)
                .filter(|extent| extent.status() != ExtentStatus::Ambiguous)
                .map(|extent| extent.blocks.clone())
        });
        assert_eq!(decisions[0].status, ExtentPromotionStatus::Promoted);
        let expected: BTreeSet<crate::BlockId> = [landing].into_iter().collect();
        assert_eq!(cfg.regions()[0].handlers[0].body.blocks(), Some(&expected));
    }
}
