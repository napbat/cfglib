//! The region vocabulary on `Cfg`: identities, lookup, nesting, removal, and
//! the block-granular protection contract.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, ControlNode, ControlTree, EdgeKind, HandlerBody, HandlerRef, Region, RegionId,
    RegionIndex,
};

use crate::shapes::{Stmt, Try, block_of, catch, protect, try_catch};

#[test]
fn add_region_assigns_sequential_identities_and_ignores_the_one_it_was_given() {
    let mut cfg = Cfg::<Stmt>::new();
    let pad = block_of(&mut cfg, &["catch"]);

    let first = cfg.add_region(Region {
        id: RegionId::from_raw(41),
        protected_blocks: BTreeSet::from([cfg.entry()]),
        handlers: vec![catch(pad)],
        parent: None,
    });
    let second = protect(&mut cfg, &[pad], vec![]);

    assert_eq!((first.index(), second.index()), (0, 1));
    assert_eq!(cfg.regions().len(), 2);
    for (index, region) in cfg.regions().iter().enumerate() {
        assert_eq!(region.id.index(), index, "a region knows its own identity");
    }
    assert_eq!(cfg.region(first).map(|region| region.id), Some(first));
    assert!(cfg.region(RegionId::from_raw(2)).is_none());
}

#[test]
fn the_innermost_region_is_the_latest_added_one_and_the_index_agrees_everywhere() {
    let mut cfg = Cfg::<Stmt>::new();
    let blocks: Vec<BlockId> = (0..6).map(|_| block_of(&mut cfg, &["x"])).collect();
    let outer = protect(&mut cfg, &blocks[0..4], vec![]);
    let middle = protect(&mut cfg, &blocks[1..3], vec![]);
    let inner = protect(&mut cfg, &blocks[2..3], vec![]);
    cfg.region_mut(middle).expect("added").parent = Some(outer);
    cfg.region_mut(inner).expect("added").parent = Some(middle);

    let index = RegionIndex::compute(&cfg);

    let expected = [
        (blocks[0], Some(outer)),
        (blocks[1], Some(middle)),
        (blocks[2], Some(inner)),
        (blocks[3], Some(outer)),
        (blocks[4], None),
        (blocks[5], None),
    ];
    for (block, region) in expected {
        assert_eq!(
            cfg.protecting_region(block).map(|found| found.id),
            region,
            "{block}"
        );
        assert_eq!(index.protecting(block), region, "{block}");
    }
    assert_eq!(index.protecting(cfg.entry()), None);
}

#[test]
fn a_region_index_is_a_snapshot_of_the_regions_it_was_computed_from() {
    let mut cfg = Cfg::<Stmt>::new();
    let body = block_of(&mut cfg, &["risky()"]);
    let stale = RegionIndex::compute(&cfg);
    let region = protect(&mut cfg, &[body], vec![]);
    let late = block_of(&mut cfg, &["added after the index"]);
    cfg.region_mut(region)
        .expect("added")
        .protected_blocks
        .insert(late);

    assert_eq!(
        stale.protecting(body),
        None,
        "the index predates the region"
    );
    assert_eq!(stale.protecting(late), None);
    let fresh = RegionIndex::compute(&cfg);
    assert_eq!(fresh.protecting(body), Some(region));
    assert_eq!(fresh.protecting(late), Some(region));
}

#[test]
fn handlers_and_regions_are_reached_by_identity_and_edited_in_place() {
    let Try {
        mut cfg,
        pad,
        region,
        ..
    } = try_catch(&["log(e)"]);
    let handler = HandlerRef::new(region, 0);
    let replacement = block_of(&mut cfg, &["other pad"]);

    assert_eq!(cfg.handler(handler).map(|found| found.entry), Some(pad));
    assert!(cfg.handler(HandlerRef::new(region, 1)).is_none());
    assert!(
        cfg.handler(HandlerRef::new(RegionId::from_raw(9), 0))
            .is_none()
    );

    cfg.handler_mut(handler).expect("exists").entry = replacement;
    cfg.region_mut(region).expect("exists").parent = Some(region);

    assert_eq!(
        cfg.handler(handler).map(|found| found.entry),
        Some(replacement)
    );
    assert_eq!(
        cfg.region(region).and_then(|found| found.parent),
        Some(region)
    );
    assert!(cfg.handler_mut(HandlerRef::new(region, 5)).is_none());
}

#[test]
fn a_handler_ref_orders_by_region_then_position() {
    let region = RegionId::from_raw(0);
    let later = RegionId::from_raw(1);

    let mut refs = vec![
        HandlerRef::new(later, 0),
        HandlerRef::new(region, 2),
        HandlerRef::new(region, 0),
    ];
    refs.sort();

    assert_eq!(
        refs,
        [
            HandlerRef::new(region, 0),
            HandlerRef::new(region, 2),
            HandlerRef::new(later, 0),
        ]
    );
    assert_eq!(refs[1].region(), region);
    assert_eq!(refs[1].index(), 2);
}

#[test]
fn an_empty_known_body_is_a_complete_extent_and_not_an_unknown_one() {
    let nothing = HandlerBody::known([]);
    let unknown = HandlerBody::unknown();
    let some = HandlerBody::known([BlockId::from_raw(3), BlockId::from_raw(1)]);

    assert_ne!(nothing, unknown);
    assert!(nothing.is_known());
    assert!(!unknown.is_known());
    assert_eq!(nothing.blocks().map(BTreeSet::len), Some(0));
    assert_eq!(unknown.blocks(), None);
    assert_eq!(
        some.blocks()
            .map(|blocks| blocks.iter().map(|b| b.index()).collect::<Vec<_>>()),
        Some(vec![1, 3]),
        "a body is a set: sorted, deduplicated"
    );
    let from_set: HandlerBody = BTreeSet::from([BlockId::from_raw(3), BlockId::from_raw(1)]).into();
    assert_eq!(from_set, some);
}

#[test]
fn removing_a_region_leaves_an_inert_tombstone_so_later_identities_stay_put() {
    let mut try_one = try_catch(&["first"]);
    let second_body = block_of(&mut try_one.cfg, &["more()"]);
    let second_pad = block_of(&mut try_one.cfg, &["second"]);
    try_one
        .cfg
        .add_edge(try_one.join, second_body, EdgeKind::Fallthrough);
    try_one
        .cfg
        .add_edge(second_body, second_pad, EdgeKind::ExceptionUnwind);
    let second = protect(&mut try_one.cfg, &[second_body], vec![catch(second_pad)]);
    let cfg = &mut try_one.cfg;

    let removed = cfg.remove_region(try_one.region).expect("it existed");

    assert_eq!(removed.protected_blocks, BTreeSet::from([try_one.body]));
    assert_eq!(removed.handlers.len(), 1);
    assert_eq!(cfg.regions().len(), 2, "identities do not shift");
    let tombstone = cfg.region(try_one.region).expect("the slot stays");
    assert!(tombstone.protected_blocks.is_empty() && tombstone.handlers.is_empty());
    assert_eq!(cfg.region(second).map(|region| region.id), Some(second));
    assert!(cfg.protecting_region(try_one.body).is_none());
    assert!(cfg.remove_region(RegionId::from_raw(7)).is_none());
}

#[test]
fn structuring_ignores_a_removed_region_but_still_structures_the_live_one() {
    let mut first = try_catch(&["first"]);
    let second_body = block_of(&mut first.cfg, &["more()"]);
    let second_pad = block_of(&mut first.cfg, &["second"]);
    first
        .cfg
        .add_edge(first.join, second_body, EdgeKind::Fallthrough);
    first
        .cfg
        .add_edge(second_body, second_pad, EdgeKind::ExceptionUnwind);
    let tail = block_of(&mut first.cfg, &["end"]);
    first.cfg.add_edge(second_body, tail, EdgeKind::Fallthrough);
    first.cfg.add_edge(second_pad, tail, EdgeKind::Fallthrough);
    // `join` now continues into the second try, so it must not return.
    first.cfg.block_mut(first.join).instructions_mut().clear();
    let live = protect(&mut first.cfg, &[second_body], vec![catch(second_pad)]);
    first.cfg.remove_region(first.region);

    let (tree, report) = ControlTree::compute_with_report(&first.cfg);

    assert!(
        !report.unstructured_regions.contains(&first.region),
        "an inert region carries nothing to structure: {report:?}"
    );
    assert!(!report.unstructured_regions.contains(&live), "{report:?}");
    let tries = tree
        .preorder()
        .filter(|&id| matches!(tree.node(id), ControlNode::TryCatch { .. }))
        .count();
    assert_eq!(tries, 1, "only the live region becomes a try/catch");
}

#[test]
fn remove_unreachable_regions_keeps_a_region_while_any_of_its_handlers_can_be_entered() {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    let body = block_of(&mut cfg, &["risky()"]);
    let entered = block_of(&mut cfg, &["entered pad"]);
    let orphan_a = block_of(&mut cfg, &["orphan a"]);
    let orphan_b = block_of(&mut cfg, &["orphan b"]);
    cfg.add_edge(entry, body, EdgeKind::Fallthrough);
    cfg.add_edge(body, entered, EdgeKind::ExceptionUnwind);
    let live = protect(&mut cfg, &[body], vec![catch(entered)]);
    let mixed = protect(&mut cfg, &[body], vec![catch(orphan_a), catch(entered)]);
    let dead = protect(&mut cfg, &[body], vec![catch(orphan_a), catch(orphan_b)]);
    let without_handlers = protect(&mut cfg, &[body], vec![]);

    let removed = cfg.remove_unreachable_regions();

    assert_eq!(removed, 1);
    assert!(cfg.region(live).is_some_and(|r| !r.handlers.is_empty()));
    assert!(cfg.region(mixed).is_some_and(|r| r.handlers.len() == 2));
    assert!(
        cfg.region(dead)
            .is_some_and(|r| r.handlers.is_empty() && r.protected_blocks.is_empty()),
        "no handler of it can be entered"
    );
    assert!(
        cfg.region(without_handlers)
            .is_some_and(|r| r.protected_blocks.contains(&body)),
        "a region with no handler has nothing to judge"
    );
    assert_eq!(cfg.remove_unreachable_regions(), 0, "idempotent");
}

#[test]
fn statement_granular_protection_is_block_granular_once_split_at_its_boundaries() {
    // One block holds `a(); risky(); b()`; only the middle statement is in the
    // try. The frontend splits at the boundaries, then registers the region.
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    for text in ["a()", "risky()", "b()"] {
        cfg.block_mut(entry).push(Stmt::text(text));
    }
    let pad = block_of(&mut cfg, &["recover()"]);
    let tail = cfg.split_block(entry, 2);
    let middle = cfg.split_block(entry, 1);
    cfg.add_edge(middle, pad, EdgeKind::ExceptionUnwind);
    cfg.add_edge(pad, tail, EdgeKind::Fallthrough);
    let region = protect(&mut cfg, &[middle], vec![catch(pad)]);

    assert_eq!(cfg.block(entry).instructions(), [Stmt::text("a()")]);
    assert_eq!(cfg.block(middle).instructions(), [Stmt::text("risky()")]);
    assert_eq!(cfg.block(tail).instructions(), [Stmt::text("b()")]);
    assert_eq!(cfg.protecting_region(middle).map(|r| r.id), Some(region));
    assert!(cfg.protecting_region(entry).is_none());
    assert!(cfg.protecting_region(tail).is_none());

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert!(report.is_fully_structured(), "{report:?}");
    let [try_node] = tree
        .preorder()
        .filter(|&id| matches!(tree.node(id), ControlNode::TryCatch { .. }))
        .collect::<Vec<_>>()[..]
    else {
        panic!("one try/catch: {tree:?}");
    };
    let ControlNode::TryCatch { try_body, .. } = *tree.node(try_node) else {
        unreachable!("filtered to try/catch");
    };
    let protected: Vec<BlockId> = tree
        .body(try_body)
        .iter()
        .filter_map(|&id| match *tree.node(id) {
            ControlNode::Block(span) | ControlNode::Return(span) => Some(span.block()),
            _ => None,
        })
        .collect();
    assert_eq!(
        protected,
        [middle],
        "only the middle statement is protected"
    );
}
