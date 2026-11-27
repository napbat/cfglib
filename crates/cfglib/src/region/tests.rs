use super::*;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::test_util::MockInst;
use alloc::collections::BTreeSet;

fn block_set(ids: &[u32]) -> BTreeSet<BlockId> {
    ids.iter().map(|&i| BlockId::from_raw(i)).collect()
}

fn region(protected: &[u32], handlers: Vec<Handler>, parent: Option<RegionId>) -> Region {
    Region {
        protected_blocks: block_set(protected),
        handlers,
        parent,
    }
}

#[cfg(feature = "serde")]
#[test]
fn handler_body_serde_form_is_pinned() {
    let known = HandlerBody::known([BlockId::from_raw(1), BlockId::from_raw(2)]);
    let json = serde_json::to_string(&known).expect("serialization cannot fail");
    assert_eq!(json, r#"{"Known":[1,2]}"#);
    let round_tripped: HandlerBody =
        serde_json::from_str(&json).expect("the pinned form deserializes");
    assert_eq!(round_tripped, known);

    let unknown = HandlerBody::unknown();
    let json = serde_json::to_string(&unknown).expect("serialization cannot fail");
    assert_eq!(json, r#""Unknown""#);
    let round_tripped: HandlerBody =
        serde_json::from_str(&json).expect("the pinned form deserializes");
    assert_eq!(round_tripped, unknown);
}

#[test]
fn add_region_assigns_sequential_ids() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let r0 = cfg.add_region(region(&[0], alloc::vec![], None)).unwrap();
    let r1 = cfg
        .add_region(region(&[0], alloc::vec![], Some(r0)))
        .unwrap();
    assert_eq!(r0.index(), 0);
    assert_eq!(r1.index(), 1);
    assert_eq!(cfg.region_count(), 2);
    assert_eq!(cfg.region(r1).parent, Some(r0));
    assert_eq!(
        cfg.regions().map(|(id, _)| id).collect::<Vec<_>>(),
        [r0, r1],
        "the regions come with their identities, in identity order"
    );
}

#[test]
fn add_region_refuses_a_missing_parent_and_a_dead_handler_block() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let pad = cfg.new_block();
    let dead = cfg.new_block();
    cfg.remove_block(dead);
    let handler = |kind| Handler {
        entry: pad,
        body: HandlerBody::known([pad]),
        kind,
    };

    assert_eq!(
        cfg.add_region(region(&[0], alloc::vec![], Some(RegionId::from_raw(0)))),
        Err(RegionError::ParentMissing(RegionId::from_raw(0))),
        "a region cannot name itself or a later region as its parent"
    );
    let mut dead_entry = handler(HandlerKind::Catch);
    dead_entry.entry = dead;
    assert_eq!(
        cfg.add_region(region(&[0], alloc::vec![dead_entry], None)),
        Err(RegionError::BlockNotLive(dead))
    );
    assert_eq!(
        cfg.add_region(region(
            &[0],
            alloc::vec![handler(HandlerKind::Filter { filter_block: dead })],
            None
        )),
        Err(RegionError::BlockNotLive(dead))
    );
    assert_eq!(cfg.region_count(), 0, "a refused region is not added");
    assert_eq!(cfg.check_regions(), Ok(()));
}

#[test]
fn region_index_finds_the_innermost_region() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let _b1 = cfg.new_block();
    let _b2 = cfg.new_block();
    // Outer region protects blocks 0,1,2; inner protects block 1 only.
    let outer = cfg
        .add_region(region(&[0, 1, 2], alloc::vec![], None))
        .unwrap();
    let inner = cfg
        .add_region(region(&[1], alloc::vec![], Some(outer)))
        .unwrap();
    let b3 = cfg.new_block();

    let index = RegionIndex::compute(&cfg);
    // Block 1 finds the inner (last-added) region, block 0 the outer one.
    assert_eq!(index.protecting(BlockId::from_raw(1)), Some(inner));
    assert_eq!(index.protecting(BlockId::from_raw(0)), Some(outer));
    assert_eq!(index.protecting(BlockId::from_raw(2)), Some(outer));
    // b3 belongs to no region.
    assert!(index.protecting(b3).is_none());
}

#[test]
fn region_index_answers_for_a_cfg_with_edge_payloads() {
    let mut cfg = Cfg::<u32, &str>::with_edge_payload();
    let inside = cfg.new_block();
    let region = cfg
        .add_region(Region {
            protected_blocks: [inside].into_iter().collect(),
            handlers: alloc::vec![],
            parent: None,
        })
        .unwrap();

    let index = RegionIndex::compute(&cfg);
    assert_eq!(index.protecting(inside), Some(region));
    assert_eq!(index.protecting(cfg.entry()), None);
}

/// A CFG with one protected block, one catch pad, and one cleanup — the
/// shape a source `try`/`catch`/`finally` lowers to.
fn try_catch_finally() -> (Cfg<MockInst>, RegionId, [BlockId; 4]) {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let pad = cfg.new_block();
    let cleanup = cfg.new_block();
    let after = cfg.new_block();
    let exit = cfg.new_block();
    let region = cfg
        .add_region(region(
            &[0],
            alloc::vec![
                Handler {
                    entry: pad,
                    body: HandlerBody::known([pad]),
                    kind: HandlerKind::Catch,
                },
                Handler {
                    entry: cleanup,
                    body: HandlerBody::known([cleanup]),
                    kind: HandlerKind::Finally,
                },
            ],
            None,
        ))
        .unwrap();
    (cfg, region, [pad, cleanup, after, exit])
}

#[test]
fn handler_ref_names_one_handler_of_one_region() {
    let (cfg, region, [pad, cleanup, _, _]) = try_catch_finally();
    let catch = HandlerRef::new(region, 0);
    let finally = HandlerRef::new(region, 1);

    assert_eq!(catch.region(), region);
    assert_eq!(finally.index(), 1);
    assert!(catch < finally, "handlers order by position in the region");
    assert_eq!(cfg.region(region).handlers[catch.index()].entry, pad);
    assert_eq!(alloc::format!("{finally}"), "region0.handler1");
    assert_eq!(cfg.handler(catch).entry, pad);
    assert_eq!(cfg.handler(finally).entry, cleanup);
    assert_eq!(
        cfg.handlers()
            .map(|(handler, stored)| (handler, stored.entry))
            .collect::<Vec<_>>(),
        [(catch, pad), (finally, cleanup)]
    );
    assert_eq!(
        cfg.handlers_at(cleanup)
            .map(|(handler, _)| handler)
            .collect::<Vec<_>>(),
        [finally]
    );
    assert!(cfg.contains_handler(finally));
    assert!(!cfg.contains_handler(HandlerRef::new(region, 2)));
    assert!(!cfg.contains_handler(HandlerRef::new(RegionId::from_raw(99), 0)));
    assert!(!cfg.contains_region(RegionId::from_raw(1)));
}

#[test]
#[should_panic(expected = "region9.handler0 is not a handler of this CFG")]
fn a_missing_handler_is_a_caller_bug() {
    let (cfg, _, _) = try_catch_finally();
    let _ = cfg.handler(HandlerRef::new(RegionId::from_raw(9), 0));
}

#[test]
#[should_panic(expected = "region1 is not a region of this CFG")]
fn a_missing_region_is_a_caller_bug() {
    let (cfg, _, _) = try_catch_finally();
    let _ = cfg.region(RegionId::from_raw(1));
}

#[test]
fn mutable_accessors_update_regions_and_handlers_and_check_regions_sees_it() {
    let (mut cfg, region, [pad, _, _, _]) = try_catch_finally();
    let catch = HandlerRef::new(region, 0);
    let replacement = cfg.new_block();

    cfg.region_mut(region).parent = Some(region);
    cfg.handler_mut(catch).entry = replacement;

    assert_eq!(cfg.region(region).parent, Some(region));
    assert_eq!(cfg.handler(catch).entry, replacement);
    assert_ne!(cfg.handler(catch).entry, pad);
    assert_eq!(cfg.check_regions(), Ok(()));

    cfg.region_mut(region).parent = Some(RegionId::from_raw(4));
    assert_eq!(
        cfg.check_regions(),
        Err(RegionError::ParentMissing(RegionId::from_raw(4)))
    );
    cfg.region_mut(region).parent = None;
    cfg.remove_block(replacement);
    assert_eq!(
        cfg.check_regions(),
        Err(RegionError::BlockNotLive(replacement))
    );
}

#[test]
fn a_cleanup_records_one_route_per_distinct_destination() {
    let (mut cfg, region, [_, cleanup, after, exit]) = try_catch_finally();
    let finally = HandlerRef::new(region, 1);
    assert!(
        cfg.cleanup(finally).is_none(),
        "no record until one is made"
    );

    cfg.set_cleanup_exit(finally, cleanup).unwrap();
    for continuation in [
        Continuation {
            reason: CompletionReason::Normal,
            target: after,
        },
        Continuation {
            reason: CompletionReason::Return,
            target: exit,
        },
        // A second `return` route to the same block is the same route.
        Continuation {
            reason: CompletionReason::Return,
            target: exit,
        },
        // Same destination, different reason: a distinct route.
        Continuation {
            reason: CompletionReason::Unwind,
            target: exit,
        },
    ] {
        cfg.add_continuation(finally, continuation).unwrap();
    }

    let record = cfg.cleanup(finally).expect("recorded");
    assert_eq!(record.exit_block, Some(cleanup));
    assert_eq!(
        record
            .continuations
            .iter()
            .map(|route| (route.reason, route.target))
            .collect::<Vec<_>>(),
        alloc::vec![
            (CompletionReason::Normal, after),
            (CompletionReason::Return, exit),
            (CompletionReason::Unwind, exit),
        ],
        "distinct routes, in first-recorded order"
    );

    // The reason selects the route — the point of the record.
    assert_eq!(
        record
            .targets_for(CompletionReason::Normal)
            .collect::<Vec<_>>(),
        alloc::vec![after]
    );
    assert_eq!(
        record
            .targets_for(CompletionReason::Return)
            .collect::<Vec<_>>(),
        alloc::vec![exit]
    );
    assert_eq!(record.targets_for(CompletionReason::Transfer).next(), None);
    assert_eq!(cfg.cleanups().len(), 1);
}

#[test]
fn cleanup_mutators_refuse_a_missing_handler_and_a_dead_block() {
    let (mut cfg, region, [_, cleanup, after, _]) = try_catch_finally();
    let missing = HandlerRef::new(region, 2);
    let route = Continuation {
        reason: CompletionReason::Normal,
        target: after,
    };
    assert_eq!(
        cfg.add_continuation(missing, route),
        Err(RegionError::HandlerMissing(missing))
    );
    assert_eq!(
        cfg.set_cleanup_exit(missing, cleanup),
        Err(RegionError::HandlerMissing(missing))
    );

    let finally = HandlerRef::new(region, 1);
    let dead = cfg.new_block();
    cfg.remove_block(dead);
    assert_eq!(
        cfg.set_cleanup_exit(finally, dead),
        Err(RegionError::BlockNotLive(dead))
    );
    assert_eq!(
        cfg.add_continuation(
            finally,
            Continuation {
                target: dead,
                ..route
            }
        ),
        Err(RegionError::BlockNotLive(dead))
    );
    assert!(cfg.cleanups().is_empty(), "a refused record is not made");
}

#[test]
fn a_diverging_cleanup_has_routes_but_no_exit_block() {
    let (mut cfg, region, [_, _, after, _]) = try_catch_finally();
    let finally = HandlerRef::new(region, 1);
    cfg.add_continuation(
        finally,
        Continuation {
            reason: CompletionReason::Normal,
            target: after,
        },
    )
    .unwrap();

    // The cleanup body ends in a `return`, so nothing leaves it: the
    // recorded route stays, unreachable, which is what the source says.
    let record = cfg.cleanup(finally).expect("recorded");
    assert_eq!(record.exit_block, None);
    assert_eq!(record.continuations.len(), 1);
}

#[test]
fn cleanups_are_per_handler_and_go_with_their_region() {
    let (mut cfg, region, [_, _, after, exit]) = try_catch_finally();
    let catch = HandlerRef::new(region, 0);
    let finally = HandlerRef::new(region, 1);
    cfg.add_continuation(
        catch,
        Continuation {
            reason: CompletionReason::Normal,
            target: after,
        },
    )
    .unwrap();
    cfg.add_continuation(
        finally,
        Continuation {
            reason: CompletionReason::Unwind,
            target: exit,
        },
    )
    .unwrap();

    assert_eq!(cfg.cleanups().len(), 2);
    assert_eq!(
        cfg.cleanup(catch)
            .map(|record| record.continuations[0].target),
        Some(after)
    );
    assert_eq!(
        cfg.cleanup(finally)
            .map(|record| record.continuations[0].reason),
        Some(CompletionReason::Unwind)
    );

    let removed = cfg.remove_region(region);
    assert_eq!(removed.handlers.len(), 2);
    assert!(cfg.cleanups().is_empty());
    assert!(cfg.region(region).handlers.is_empty(), "a tombstone stays");
}

#[test]
fn verify_reports_every_broken_region_reference_and_check_regions_the_first() {
    use alloc::string::{String, ToString};

    let (mut cfg, region, [pad, cleanup, _, _]) = try_catch_finally();
    let missing = RegionId::from_raw(7);
    cfg.remove_block(pad);
    cfg.remove_block(cleanup);
    cfg.region_mut(region).parent = Some(missing);

    let expected = [
        RegionError::ParentMissing(missing),
        RegionError::BlockNotLive(pad),
        RegionError::BlockNotLive(cleanup),
    ];
    assert_eq!(cfg.check_regions(), Err(expected[0]));
    let messages: Vec<String> = crate::verify(&cfg)
        .errors
        .into_iter()
        .map(|error| error.message)
        .collect();
    assert!(
        messages.ends_with(&expected.map(|error| error.to_string())),
        "{messages:?}"
    );
}

#[test]
fn regions_need_no_optional_cleanup_or_handler_metadata() {
    // The core region remains usable without cleanup continuations or
    // consumer-owned side metadata.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let pad = cfg.new_block();
    let id = cfg
        .add_region(region(
            &[0],
            alloc::vec![Handler {
                entry: pad,
                body: HandlerBody::known([pad]),
                kind: HandlerKind::Finally,
            }],
            None,
        ))
        .unwrap();

    assert_eq!(cfg.region_count(), 1);
    assert_eq!(
        RegionIndex::compute(&cfg).protecting(BlockId::from_raw(0)),
        Some(id)
    );
    // A frontend that records no routes carries no records.
    assert!(cfg.cleanups().is_empty());
    assert!(cfg.cleanup(HandlerRef::new(id, 0)).is_none());
}

#[test]
fn handler_kind_names_round_trip() {
    let filter_block = BlockId::from_raw(4);
    for kind in [
        HandlerKind::Catch,
        HandlerKind::CatchAll,
        HandlerKind::Finally,
        HandlerKind::Fault,
        HandlerKind::Filter { filter_block },
    ] {
        assert_eq!(
            HandlerKind::from_name(kind.name(), kind.filter_block()),
            Some(kind)
        );
    }
    assert_eq!(HandlerKind::from_name("filter", None), None);
    assert_eq!(HandlerKind::from_name("rescue", None), None);
    assert_eq!(
        HandlerKind::Filter { filter_block }.filter_block(),
        Some(filter_block)
    );
    assert_eq!(HandlerKind::Catch.filter_block(), None);
}

#[test]
fn handler_body_distinguishes_unknown_from_complete_extents() {
    let unknown = HandlerBody::unknown();
    assert!(!unknown.is_known());
    assert!(unknown.blocks().is_none());

    let known = HandlerBody::known([
        BlockId::from_raw(2),
        BlockId::from_raw(1),
        BlockId::from_raw(2),
    ]);
    assert!(known.is_known());
    assert_eq!(
        known.blocks().map(BTreeSet::len),
        Some(2),
        "known extents are a deduplicated block set"
    );
}

#[test]
fn a_handler_names_its_entry_filter_and_known_extent() {
    let entry = BlockId::from_raw(1);
    let filter_block = BlockId::from_raw(5);
    let handler = |body, kind| Handler { entry, body, kind };
    let known = handler(
        HandlerBody::known([entry, BlockId::from_raw(3)]),
        HandlerKind::Catch,
    );
    assert_eq!(
        known.blocks().collect::<Vec<_>>(),
        [entry, entry, BlockId::from_raw(3)],
        "the entry, then the extent, which holds the entry again"
    );
    let filtered = handler(HandlerBody::unknown(), HandlerKind::Filter { filter_block });
    assert_eq!(
        filtered.blocks().collect::<Vec<_>>(),
        [entry, filter_block],
        "a handler of unknown extent names its entry and its filter block"
    );
}
