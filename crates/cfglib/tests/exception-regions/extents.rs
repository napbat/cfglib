//! Recovering and promoting handler extents: what a table that names only a
//! handler entry leaves unsaid, and how much of it the graph can prove.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, ControlNode, ControlTree, Edge, EdgeId, EdgeKind, ExclusiveExtent, ExtentIssue,
    ExtentPromotionDecision, ExtentPromotionStatus, ExtentStatus, HandlerBody, HandlerRef,
    RegionId, promote_exclusive_extents, promote_handler_extents, recover_exclusive_extents,
    recover_exclusive_extents_with,
};

use crate::shapes::{Stmt, block_of, protect, unknown_catch};

/// `setup(); risky(); return`: a protected body and its join, no pad yet.
struct Base {
    cfg: Cfg<Stmt>,
    body: BlockId,
    join: BlockId,
}

fn base() -> Base {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(Stmt::text("setup()"));
    let body = block_of(&mut cfg, &["risky()"]);
    let join = block_of(&mut cfg, &["return"]);
    cfg.add_edge(entry, body, EdgeKind::Fallthrough);
    cfg.add_edge(body, join, EdgeKind::Fallthrough);
    Base { cfg, body, join }
}

/// A pad of `statements` the body can raise into, registered with an unknown
/// extent in a region of its own.
fn add_pad(base: &mut Base, statements: &[&str]) -> (BlockId, RegionId) {
    let pad = block_of(&mut base.cfg, statements);
    base.cfg.add_edge(base.body, pad, EdgeKind::ExceptionUnwind);
    let region = protect(&mut base.cfg, &[base.body], vec![unknown_catch(pad)]);
    (pad, region)
}

fn blocks(extent: &ExclusiveExtent) -> Vec<BlockId> {
    extent.blocks.iter().copied().collect()
}

#[test]
fn a_handler_closed_under_normal_flow_is_isolated() {
    let mut fixture = base();
    let (pad, region) = add_pad(&mut fixture, &["log(e)"]);
    let end = block_of(&mut fixture.cfg, &["end of handler"]);
    fixture.cfg.add_edge(pad, end, EdgeKind::Fallthrough);

    let extents = recover_exclusive_extents(&fixture.cfg);

    let [extent] = &extents[..] else {
        panic!("one handler, one extent");
    };
    assert_eq!(extent.handler, HandlerRef::new(region, 0));
    assert_eq!(extent.entry, pad);
    assert_eq!(blocks(extent), [pad, end]);
    assert_eq!(extent.boundary_blocks, BTreeSet::new());
    assert_eq!(extent.issues, []);
    assert_eq!(extent.status(), ExtentStatus::Isolated);
}

#[test]
fn a_handler_that_flows_into_the_normal_join_leaves_the_join_on_its_boundary() {
    let mut fixture = base();
    let (pad, _) = add_pad(&mut fixture, &["log(e)"]);
    fixture
        .cfg
        .add_edge(pad, fixture.join, EdgeKind::Fallthrough);

    let extents = recover_exclusive_extents(&fixture.cfg);

    let [extent] = &extents[..] else {
        panic!("one handler, one extent");
    };
    assert_eq!(blocks(extent), [pad], "the shared join is not claimed");
    assert_eq!(extent.boundary_blocks, BTreeSet::from([fixture.join]));
    assert_eq!(extent.issues, []);
    assert_eq!(extent.status(), ExtentStatus::SharedContinuation);
}

#[test]
fn a_handler_entry_that_normal_flow_reaches_is_ambiguous() {
    let mut fixture = base();
    let (pad, _) = add_pad(&mut fixture, &["log(e)"]);
    // The body also falls into the pad: ordinary flow enters the handler.
    fixture
        .cfg
        .add_edge(fixture.body, pad, EdgeKind::Fallthrough);

    let extents = recover_exclusive_extents(&fixture.cfg);

    let [extent] = &extents[..] else {
        panic!("one handler, one extent");
    };
    assert_eq!(blocks(extent), [], "everything is reachable from the entry");
    assert_eq!(extent.boundary_blocks, BTreeSet::from([pad]));
    assert_eq!(extent.issues, [ExtentIssue::EntryReachableFromNormalFlow]);
    assert_eq!(extent.status(), ExtentStatus::Ambiguous);
}

#[test]
fn one_handler_that_flows_into_another_makes_only_the_second_ambiguous() {
    // pad_b falls into pad_a, so pad_a's code is reachable from both entries.
    let mut fixture = base();
    let (pad_a, region_a) = add_pad(&mut fixture, &["catch a"]);
    let (pad_b, region_b) = add_pad(&mut fixture, &["catch b"]);
    fixture.cfg.add_edge(pad_b, pad_a, EdgeKind::Fallthrough);

    let extents = recover_exclusive_extents(&fixture.cfg);

    let [a, b] = &extents[..] else {
        panic!("two handlers, two extents");
    };
    assert_eq!(a.handler, HandlerRef::new(region_a, 0));
    assert_eq!(blocks(a), []);
    assert_eq!(a.boundary_blocks, BTreeSet::from([pad_a]));
    assert_eq!(a.issues, [ExtentIssue::EntryReachableFromAnotherHandler]);
    assert_eq!(a.status(), ExtentStatus::Ambiguous);

    assert_eq!(b.handler, HandlerRef::new(region_b, 0));
    assert_eq!(blocks(b), [pad_b], "the code both reach is not claimed");
    assert_eq!(b.boundary_blocks, BTreeSet::from([pad_a]));
    assert_eq!(b.issues, []);
    assert_eq!(b.status(), ExtentStatus::SharedContinuation);
}

#[test]
fn an_interior_block_with_an_outside_predecessor_is_an_external_entry() {
    let mut fixture = base();
    let (pad, _) = add_pad(&mut fixture, &["log(e)"]);
    let middle = block_of(&mut fixture.cfg, &["middle"]);
    let stray = block_of(&mut fixture.cfg, &["nothing reaches me"]);
    fixture.cfg.add_edge(pad, middle, EdgeKind::Fallthrough);
    fixture.cfg.add_edge(stray, middle, EdgeKind::Fallthrough);

    let extents = recover_exclusive_extents(&fixture.cfg);

    let [extent] = &extents[..] else {
        panic!("one handler, one extent");
    };
    assert_eq!(blocks(extent), [pad, middle]);
    assert_eq!(
        extent.issues,
        [ExtentIssue::ExternalEntry {
            block: middle,
            predecessor: stray
        }]
    );
    assert_eq!(extent.status(), ExtentStatus::Ambiguous);
}

#[test]
fn extents_come_back_in_region_then_handler_order_and_nothing_is_mutated() {
    let mut fixture = base();
    let first_pad = block_of(&mut fixture.cfg, &["first"]);
    let second_pad = block_of(&mut fixture.cfg, &["second"]);
    let third_pad = block_of(&mut fixture.cfg, &["third"]);
    for pad in [first_pad, second_pad, third_pad] {
        fixture
            .cfg
            .add_edge(fixture.body, pad, EdgeKind::ExceptionUnwind);
    }
    let first = protect(
        &mut fixture.cfg,
        &[fixture.body],
        vec![unknown_catch(first_pad), unknown_catch(second_pad)],
    );
    let second = protect(
        &mut fixture.cfg,
        &[fixture.body],
        vec![unknown_catch(third_pad)],
    );
    let before = fixture.cfg.clone();

    let extents = recover_exclusive_extents(&fixture.cfg);

    assert_eq!(
        extents
            .iter()
            .map(|extent| extent.handler)
            .collect::<Vec<_>>(),
        [
            HandlerRef::new(first, 0),
            HandlerRef::new(first, 1),
            HandlerRef::new(second, 0)
        ]
    );
    assert_eq!(
        fixture.cfg, before,
        "recovery reads the graph and writes nothing"
    );
}

/// How a frontend that does not use the exceptional edge kinds tags its flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Normal,
    Unwind,
}

#[test]
fn normality_in_the_edge_payload_is_honored_by_the_with_variant() {
    // Every edge has an ordinary kind; the payload says which one unwinds.
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    let body = cfg.new_block();
    let pad = cfg.new_block();
    let end = cfg.new_block();
    cfg.add_edge_with_payload(entry, body, EdgeKind::Fallthrough, Route::Normal);
    cfg.add_edge_with_payload(body, pad, EdgeKind::Fallthrough, Route::Unwind);
    cfg.add_edge_with_payload(pad, end, EdgeKind::Fallthrough, Route::Normal);
    let region = cfg.add_region(cfglib::Region {
        id: RegionId::from_raw(0),
        protected_blocks: BTreeSet::from([body]),
        handlers: vec![unknown_catch(pad)],
        parent: None,
    });

    // By kind, the unwind edge is ordinary flow and the pad is reachable from
    // the entry; by payload, it is not.
    let by_kind = recover_exclusive_extents(&cfg);
    let by_payload = recover_exclusive_extents_with(&cfg, |_: EdgeId, edge: &Edge<Route>| {
        *edge.payload() == Route::Normal
    });

    assert_eq!(
        by_kind[0].issues,
        [ExtentIssue::EntryReachableFromNormalFlow]
    );
    assert_eq!(by_payload.len(), 1);
    assert_eq!(by_payload[0].handler, HandlerRef::new(region, 0));
    assert_eq!(blocks(&by_payload[0]), [pad, end]);
    assert_eq!(by_payload[0].status(), ExtentStatus::Isolated);
}

/// Blocks `block` dominates, by definition: those the entry cannot reach once
/// `block` is removed (and `block` itself).
fn dominated_by(cfg: &Cfg<Stmt>, dominator: BlockId) -> BTreeSet<BlockId> {
    let mut reached = BTreeSet::new();
    let mut stack = vec![cfg.entry()];
    while let Some(block) = stack.pop() {
        if block != dominator && reached.insert(block) {
            stack.extend(cfg.successors(block));
        }
    }
    cfg.block_ids()
        .filter(|&block| block == dominator || !reached.contains(&block))
        .collect()
}

#[test]
fn promoting_by_dominance_claims_what_only_the_handler_reaches() {
    let mut fixture = base();
    let (pad, region) = add_pad(&mut fixture, &["log(e)"]);
    let tail = block_of(&mut fixture.cfg, &["tail"]);
    fixture.cfg.add_edge(pad, tail, EdgeKind::Fallthrough);
    fixture
        .cfg
        .add_edge(tail, fixture.join, EdgeKind::Fallthrough);
    let canonical = fixture.cfg.clone();

    let mut derived = fixture.cfg.clone();
    assert_eq!(promote_handler_extents(&mut derived), 1);

    let handler = HandlerRef::new(region, 0);
    let promoted = derived
        .handler(handler)
        .and_then(|found| found.body.blocks())
        .expect("promoted");
    assert_eq!(*promoted, dominated_by(&derived, pad));
    assert_eq!(
        promoted.iter().copied().collect::<Vec<_>>(),
        [pad, tail],
        "the join is also reached directly from the body"
    );
    assert_eq!(
        canonical.handler(handler).map(|found| &found.body),
        Some(&HandlerBody::Unknown),
        "promotion is for a derived graph, and the canonical one is untouched"
    );
    assert_eq!(promote_handler_extents(&mut derived), 0, "idempotent");
}

#[test]
fn promoting_by_dominance_skips_known_bodies_and_unreachable_entries() {
    let mut fixture = base();
    let (pad, _) = add_pad(&mut fixture, &["log(e)"]);
    let orphan = block_of(&mut fixture.cfg, &["no edge reaches this pad"]);
    let known = block_of(&mut fixture.cfg, &["known pad"]);
    fixture
        .cfg
        .add_edge(fixture.body, known, EdgeKind::ExceptionUnwind);
    let region = protect(
        &mut fixture.cfg,
        &[fixture.body],
        vec![unknown_catch(orphan), crate::shapes::catch(known)],
    );

    let promoted = promote_handler_extents(&mut fixture.cfg);

    assert_eq!(promoted, 1, "only the reachable unknown handler");
    let orphan_body = &fixture
        .cfg
        .handler(HandlerRef::new(region, 0))
        .expect("exists")
        .body;
    assert_eq!(
        *orphan_body,
        HandlerBody::Unknown,
        "an unreachable entry stays unknown"
    );
    let known_body = fixture
        .cfg
        .handler(HandlerRef::new(region, 1))
        .expect("exists")
        .body
        .blocks();
    assert_eq!(known_body.map(BTreeSet::len), Some(1));
    assert!(fixture.cfg.regions()[0].handlers[0].body.is_known());
    assert_eq!(
        fixture.cfg.regions()[0].handlers[0].entry,
        pad,
        "the first region's pad is the one promoted"
    );
}

/// Two regions: `a` has two handlers, `b` has one.
struct Candidates {
    cfg: Cfg<Stmt>,
    a: [HandlerRef; 2],
    b: HandlerRef,
    blocks: [BlockId; 4],
}

fn candidates() -> Candidates {
    let mut fixture = base();
    let pads = ["a0", "a1", "b0"].map(|text| block_of(&mut fixture.cfg, &[text]));
    let spare = block_of(&mut fixture.cfg, &["spare"]);
    for &pad in &pads {
        fixture
            .cfg
            .add_edge(fixture.body, pad, EdgeKind::ExceptionUnwind);
    }
    let a = protect(
        &mut fixture.cfg,
        &[fixture.body],
        vec![unknown_catch(pads[0]), unknown_catch(pads[1])],
    );
    let b = protect(
        &mut fixture.cfg,
        &[fixture.body],
        vec![unknown_catch(pads[2])],
    );
    Candidates {
        cfg: fixture.cfg,
        a: [HandlerRef::new(a, 0), HandlerRef::new(a, 1)],
        b: HandlerRef::new(b, 0),
        blocks: [pads[0], pads[1], pads[2], spare],
    }
}

#[test]
fn disjoint_candidates_promote_every_region_atomically() {
    let Candidates {
        mut cfg,
        a,
        b,
        blocks: [a0, a1, b0, spare],
    } = candidates();
    let bodies = [
        (a[0], BTreeSet::from([a0])),
        (a[1], BTreeSet::from([a1, spare])),
        (b, BTreeSet::from([b0])),
    ];

    let decisions = promote_exclusive_extents(&mut cfg, |handler| {
        bodies
            .iter()
            .find(|(candidate, _)| *candidate == handler)
            .map(|(_, body)| body.clone())
    });

    assert_eq!(
        decisions,
        [
            ExtentPromotionDecision {
                region: a[0].region(),
                status: ExtentPromotionStatus::Promoted
            },
            ExtentPromotionDecision {
                region: b.region(),
                status: ExtentPromotionStatus::Promoted
            },
        ]
    );
    for (handler, body) in &bodies {
        assert_eq!(
            cfg.handler(*handler).and_then(|h| h.body.blocks()),
            Some(body)
        );
    }
}

#[test]
fn one_ambiguous_handler_withholds_its_whole_region_and_only_that_region() {
    let Candidates {
        mut cfg,
        a,
        b,
        blocks: [a0, _, b0, _],
    } = candidates();

    let decisions = promote_exclusive_extents(&mut cfg, |handler| {
        if handler == a[0] {
            Some(BTreeSet::from([a0]))
        } else if handler == b {
            Some(BTreeSet::from([b0]))
        } else {
            None
        }
    });

    assert_eq!(
        decisions[0].status,
        ExtentPromotionStatus::AmbiguousExtent { handler: a[1] }
    );
    assert_eq!(decisions[1].status, ExtentPromotionStatus::Promoted);
    assert_eq!(
        cfg.handler(a[0]).map(|h| &h.body),
        Some(&HandlerBody::Unknown),
        "the sibling with a candidate is withheld too"
    );
    assert!(cfg.handler(b).is_some_and(|h| h.body.is_known()));
}

#[test]
fn candidates_that_share_a_block_withhold_both_regions() {
    let Candidates {
        mut cfg,
        a,
        b,
        blocks: [a0, _, b0, spare],
    } = candidates();

    let decisions = promote_exclusive_extents(&mut cfg, |handler| {
        Some(if handler == b {
            BTreeSet::from([b0, spare])
        } else if handler == a[0] {
            BTreeSet::from([a0])
        } else {
            BTreeSet::from([spare])
        })
    });

    // Region `a` is first in order, and its second handler overlaps region `b`.
    let overlap = ExtentPromotionStatus::OverlappingExtents {
        first: a[1],
        second: b,
    };
    assert_eq!(decisions.len(), 2);
    assert_eq!(decisions[0].status, overlap);
    assert_eq!(decisions[1].status, overlap);
    for handler in [a[0], a[1], b] {
        assert_eq!(
            cfg.handler(handler).map(|h| &h.body),
            Some(&HandlerBody::Unknown),
            "{handler}"
        );
    }
}

#[test]
fn recovered_extents_promoted_on_a_derived_graph_let_the_control_tree_emit_the_try() {
    // dasm shape: a table names the handler entry only. Recover what the
    // graph proves, promote it on a clone, and structure the clone.
    let mut fixture = base();
    let (pad, region) = add_pad(&mut fixture, &["log(e)"]);
    let end = block_of(&mut fixture.cfg, &["end of handler"]);
    fixture.cfg.add_edge(pad, end, EdgeKind::Fallthrough);
    fixture
        .cfg
        .add_edge(end, fixture.join, EdgeKind::Fallthrough);
    let (_, before) = ControlTree::compute_with_report(&fixture.cfg);
    assert_eq!(before.unstructured_regions, [region]);

    let extents = recover_exclusive_extents(&fixture.cfg);
    let mut derived = fixture.cfg.clone();
    let decisions = promote_exclusive_extents(&mut derived, |handler| {
        extents
            .iter()
            .find(|extent| extent.handler == handler && extent.status() != ExtentStatus::Ambiguous)
            .map(|extent| extent.blocks.clone())
    });

    assert_eq!(decisions[0].status, ExtentPromotionStatus::Promoted);
    let (tree, report) = ControlTree::compute_with_report(&derived);
    assert!(report.is_fully_structured(), "{report:?}");
    assert_eq!(
        tree.preorder()
            .filter(|&id| matches!(tree.node(id), ControlNode::TryCatch { .. }))
            .count(),
        1
    );
}
