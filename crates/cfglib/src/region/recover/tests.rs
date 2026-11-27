use super::*;
use crate::edge::EdgeKind;
use crate::region::{HandlerKind, Region};
use crate::test_util::{MockInst, ff};

fn by_kind(_: EdgeId, edge: &Edge<()>) -> bool {
    !edge.kind().is_exceptional()
}

fn body(cfg: &Cfg<MockInst>, region: u32) -> &HandlerBody {
    &cfg.region(RegionId::from_raw(region)).handlers[0].body
}

fn statuses(decisions: &[(RegionId, ExtentPromotionStatus)]) -> Vec<ExtentPromotionStatus> {
    decisions.iter().map(|&(_, status)| status).collect()
}

/// An unambiguous claim of `blocks` for the only handler of `region`.
fn claim(region: RegionId, entry: BlockId, blocks: &[BlockId]) -> ExclusiveExtent {
    ExclusiveExtent {
        handler: HandlerRef::new(region, 0),
        entry,
        blocks: blocks.iter().copied().collect(),
        boundary_blocks: BTreeSet::new(),
        issues: Vec::new(),
    }
}

/// `entry -> protected -> join`, with `protected` unwinding to `landing ->
/// handler_tail -> join`.
fn try_with_shared_join() -> (Cfg<MockInst>, [BlockId; 4]) {
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
    cfg.add_region(handler_region(landing, protected)).unwrap();
    (cfg, [protected, landing, handler_tail, join])
}

fn handler_region(entry: BlockId, protected: BlockId) -> Region {
    Region {
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
fn promotes_exclusive_handler_code_and_keeps_shared_joins_out() {
    let (mut cfg, [_, landing, handler_tail, join]) = try_with_shared_join();

    assert_eq!(
        statuses(&promote_handler_extents(&mut cfg)),
        [ExtentPromotionStatus::Promoted]
    );
    let promoted = body(&cfg, 0).blocks().expect("promoted").clone();
    assert!(promoted.contains(&landing));
    assert!(promoted.contains(&handler_tail));
    assert!(
        !promoted.contains(&join),
        "the shared continuation stays outside the extent"
    );

    // A second run keeps the promoted body as it is.
    promote_handler_extents(&mut cfg);
    assert_eq!(body(&cfg, 0).blocks(), Some(&promoted));
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
    let region = cfg.add_region(handler_region(funclet, protected)).unwrap();

    assert_eq!(
        statuses(&promote_handler_extents(&mut cfg)),
        [ExtentPromotionStatus::AmbiguousExtent {
            handler: HandlerRef::new(region, 0)
        }]
    );
    assert!(!body(&cfg, 0).is_known());
}

#[test]
fn dominated_bodies_that_overlap_are_not_promoted() {
    // A second handler entered only from inside the first handler's body is
    // dominated by the first entry, so the two dominated sets overlap.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let protected = cfg.new_block();
    let outer = cfg.new_block();
    let inner = cfg.new_block();
    cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
    cfg.add_edge(protected, outer, EdgeKind::ExceptionUnwind);
    cfg.add_edge(outer, inner, EdgeKind::ExceptionUnwind);
    let first = cfg.add_region(handler_region(outer, protected)).unwrap();
    let second = cfg.add_region(handler_region(inner, outer)).unwrap();

    let overlap = ExtentPromotionStatus::OverlappingExtents {
        first: HandlerRef::new(first, 0),
        second: HandlerRef::new(second, 0),
    };
    assert_eq!(
        statuses(&promote_handler_extents(&mut cfg)),
        [overlap, overlap]
    );
    assert!(!body(&cfg, 0).is_known());
    assert!(!body(&cfg, 1).is_known());
}

#[test]
fn exclusive_extents_report_ownership_and_boundary() {
    let (cfg, [_, landing, handler_tail, join]) = try_with_shared_join();

    let extents = recover_exclusive_extents(&cfg, by_kind);
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
    assert!(!extent.is_ambiguous());
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
    cfg.add_region(handler_region(landing, protected)).unwrap();

    let extents = recover_exclusive_extents(&cfg, by_kind);
    assert_eq!(
        extents[0].issues,
        alloc::vec![ExtentIssue::EntryReachableFromNormalFlow]
    );
    assert!(extents[0].is_ambiguous());
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
    let first = cfg
        .add_region(handler_region(first_landing, protected))
        .unwrap();
    let second = cfg
        .add_region(handler_region(second_landing, protected))
        .unwrap();

    // Overlapping candidate bodies withhold both regions.
    let decisions = promote_exclusive_extents(
        &mut cfg,
        &[
            claim(first, first_landing, &[first_landing, shared_tail]),
            claim(second, second_landing, &[second_landing, shared_tail]),
        ],
    );
    assert!(
        decisions
            .iter()
            .all(|(_, status)| matches!(status, ExtentPromotionStatus::OverlappingExtents { .. })),
        "a shared tail withholds both regions"
    );
    assert!(!body(&cfg, 0).is_known());
    assert!(!body(&cfg, 1).is_known());

    // A missing or ambiguous extent withholds only its own region.
    let mut ambiguous = claim(first, first_landing, &[first_landing]);
    ambiguous
        .issues
        .push(ExtentIssue::EntryReachableFromNormalFlow);
    let exclusive = claim(second, second_landing, &[second_landing]);
    for extents in [
        alloc::vec![exclusive.clone()],
        alloc::vec![ambiguous, exclusive.clone()],
    ] {
        let mut candidate = cfg.clone();
        let decisions = promote_exclusive_extents(&mut candidate, &extents);
        assert_eq!(
            decisions,
            [
                (
                    first,
                    ExtentPromotionStatus::AmbiguousExtent {
                        handler: HandlerRef::new(first, 0)
                    }
                ),
                (second, ExtentPromotionStatus::Promoted),
            ]
        );
        assert!(!body(&candidate, 0).is_known());
        assert_eq!(body(&candidate, 1).blocks(), Some(&exclusive.blocks));
    }
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
    cfg.add_region(handler_region(landing, protected)).unwrap();

    let extents = recover_exclusive_extents(&cfg, by_kind);
    let decisions = promote_exclusive_extents(&mut cfg, &extents);
    assert_eq!(statuses(&decisions), [ExtentPromotionStatus::Promoted]);
    let expected: BTreeSet<BlockId> = [landing].into_iter().collect();
    assert_eq!(body(&cfg, 0).blocks(), Some(&expected));
}

#[test]
fn a_candidate_body_without_its_handler_entry_is_not_promoted() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let protected = cfg.new_block();
    let landing = cfg.new_block();
    let tail = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(protected).push(ff("try_inst"));
    cfg.block_mut(landing).push(ff("caught"));
    cfg.block_mut(tail).push(ff("tail"));
    cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
    cfg.add_edge(protected, landing, EdgeKind::ExceptionUnwind);
    cfg.add_edge(landing, tail, EdgeKind::Fallthrough);
    let region = cfg.add_region(handler_region(landing, protected)).unwrap();
    let handler = HandlerRef::new(region, 0);

    // A body that leaves out the pad is no complete extent: the structurer
    // treats a known body without its own entry as a frontend bug.
    for blocks in [&[][..], &[tail]] {
        let decisions = promote_exclusive_extents(&mut cfg, &[claim(region, landing, blocks)]);
        assert_eq!(
            statuses(&decisions),
            [ExtentPromotionStatus::AmbiguousExtent { handler }]
        );
        assert!(!body(&cfg, 0).is_known());
    }
    let decisions =
        promote_exclusive_extents(&mut cfg, &[claim(region, landing, &[landing, tail])]);
    assert_eq!(statuses(&decisions), [ExtentPromotionStatus::Promoted]);
}
