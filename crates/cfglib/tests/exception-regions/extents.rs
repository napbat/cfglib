//! Recovering and promoting handler extents: what a table that names only a
//! handler entry leaves unsaid, and how much of it the graph can prove.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, ControlNode, ControlTree, Edge, EdgeId, EdgeKind, ExclusiveExtent, ExtentIssue,
    ExtentPromotionStatus, HandlerBody, HandlerRef, Region, RegionId, promote_exclusive_extents,
    promote_handler_extents, recover_exclusive_extents,
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

/// The extents along the ordinary edge kinds, the exceptional ones left out.
fn recover_by_kind<E>(cfg: &Cfg<Stmt, E>) -> Vec<ExclusiveExtent> {
    recover_exclusive_extents(cfg, |_, edge| !edge.kind().is_exceptional())
}

/// A caller-proposed body for `handler`, entered at `entry`, with no evidence
/// against it.
fn proposed(handler: HandlerRef, entry: BlockId, blocks: BTreeSet<BlockId>) -> ExclusiveExtent {
    ExclusiveExtent {
        handler,
        entry,
        blocks,
        boundary_blocks: BTreeSet::new(),
        issues: vec![],
    }
}

#[test]
fn a_handler_closed_under_normal_flow_is_isolated() {
    let mut fixture = base();
    let (pad, region) = add_pad(&mut fixture, &["log(e)"]);
    let end = block_of(&mut fixture.cfg, &["end of handler"]);
    fixture.cfg.add_edge(pad, end, EdgeKind::Fallthrough);

    let extents = recover_by_kind(&fixture.cfg);

    let [extent] = &extents[..] else {
        panic!("one handler, one extent");
    };
    assert_eq!(extent.handler, HandlerRef::new(region, 0));
    assert_eq!(extent.entry, pad);
    assert_eq!(blocks(extent), [pad, end]);
    assert_eq!(extent.boundary_blocks, BTreeSet::new());
    assert_eq!(extent.issues, []);
    assert!(!extent.is_ambiguous(), "closed under normal flow: isolated");
}

#[test]
fn a_handler_that_flows_into_the_normal_join_leaves_the_join_on_its_boundary() {
    let mut fixture = base();
    let (pad, _) = add_pad(&mut fixture, &["log(e)"]);
    fixture
        .cfg
        .add_edge(pad, fixture.join, EdgeKind::Fallthrough);

    let extents = recover_by_kind(&fixture.cfg);

    let [extent] = &extents[..] else {
        panic!("one handler, one extent");
    };
    assert_eq!(blocks(extent), [pad], "the shared join is not claimed");
    assert_eq!(extent.boundary_blocks, BTreeSet::from([fixture.join]));
    assert_eq!(extent.issues, []);
    assert!(
        !extent.is_ambiguous(),
        "a shared continuation is no ambiguity"
    );
}

#[test]
fn a_handler_entry_that_normal_flow_reaches_is_ambiguous() {
    let mut fixture = base();
    let (pad, _) = add_pad(&mut fixture, &["log(e)"]);
    // The body also falls into the pad: ordinary flow enters the handler.
    fixture
        .cfg
        .add_edge(fixture.body, pad, EdgeKind::Fallthrough);

    let extents = recover_by_kind(&fixture.cfg);

    let [extent] = &extents[..] else {
        panic!("one handler, one extent");
    };
    assert_eq!(blocks(extent), [], "everything is reachable from the entry");
    assert_eq!(extent.boundary_blocks, BTreeSet::from([pad]));
    assert_eq!(extent.issues, [ExtentIssue::EntryReachableFromNormalFlow]);
    assert!(extent.is_ambiguous());
}

#[test]
fn one_handler_that_flows_into_another_makes_only_the_second_ambiguous() {
    // pad_b falls into pad_a, so pad_a's code is reachable from both entries.
    let mut fixture = base();
    let (pad_a, region_a) = add_pad(&mut fixture, &["catch a"]);
    let (pad_b, region_b) = add_pad(&mut fixture, &["catch b"]);
    fixture.cfg.add_edge(pad_b, pad_a, EdgeKind::Fallthrough);

    let extents = recover_by_kind(&fixture.cfg);

    let [a, b] = &extents[..] else {
        panic!("two handlers, two extents");
    };
    assert_eq!(a.handler, HandlerRef::new(region_a, 0));
    assert_eq!(blocks(a), []);
    assert_eq!(a.boundary_blocks, BTreeSet::from([pad_a]));
    assert_eq!(a.issues, [ExtentIssue::EntryReachableFromAnotherHandler]);
    assert!(a.is_ambiguous());

    assert_eq!(b.handler, HandlerRef::new(region_b, 0));
    assert_eq!(blocks(b), [pad_b], "the code both reach is not claimed");
    assert_eq!(b.boundary_blocks, BTreeSet::from([pad_a]));
    assert_eq!(b.issues, []);
    assert!(!b.is_ambiguous());
}

#[test]
fn an_interior_block_with_an_outside_predecessor_is_an_external_entry() {
    let mut fixture = base();
    let (pad, _) = add_pad(&mut fixture, &["log(e)"]);
    let middle = block_of(&mut fixture.cfg, &["middle"]);
    let stray = block_of(&mut fixture.cfg, &["nothing reaches me"]);
    fixture.cfg.add_edge(pad, middle, EdgeKind::Fallthrough);
    fixture.cfg.add_edge(stray, middle, EdgeKind::Fallthrough);

    let extents = recover_by_kind(&fixture.cfg);

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
    assert!(extent.is_ambiguous());
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

    let extents = recover_by_kind(&fixture.cfg);

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
fn normality_in_the_edge_payload_is_honored_by_the_normality_predicate() {
    // Every edge has an ordinary kind; the payload says which one unwinds.
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    let body = cfg.new_block();
    let pad = cfg.new_block();
    let end = cfg.new_block();
    cfg.add_edge(
        entry,
        body,
        Edge::new(EdgeKind::Fallthrough).with_payload(Route::Normal),
    );
    cfg.add_edge(
        body,
        pad,
        Edge::new(EdgeKind::Fallthrough).with_payload(Route::Unwind),
    );
    cfg.add_edge(
        pad,
        end,
        Edge::new(EdgeKind::Fallthrough).with_payload(Route::Normal),
    );
    let region = cfg
        .add_region(Region {
            protected_blocks: BTreeSet::from([body]),
            handlers: vec![unknown_catch(pad)],
            parent: None,
        })
        .unwrap();

    // By kind, the unwind edge is ordinary flow and the pad is reachable from
    // the entry; by payload, it is not.
    let by_kind = recover_by_kind(&cfg);
    let by_payload = recover_exclusive_extents(&cfg, |_: EdgeId, edge: &Edge<Route>| {
        *edge.payload() == Route::Normal
    });

    assert_eq!(
        by_kind[0].issues,
        [ExtentIssue::EntryReachableFromNormalFlow]
    );
    assert_eq!(by_payload.len(), 1);
    assert_eq!(by_payload[0].handler, HandlerRef::new(region, 0));
    assert_eq!(blocks(&by_payload[0]), [pad, end]);
    assert!(!by_payload[0].is_ambiguous());
    assert_eq!(by_payload[0].boundary_blocks, BTreeSet::new());
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
    let decisions = promote_handler_extents(&mut derived);
    assert_eq!(decisions, [(region, ExtentPromotionStatus::Promoted)]);

    let handler = HandlerRef::new(region, 0);
    let promoted = derived.handler(handler).body.blocks().expect("promoted");
    assert_eq!(*promoted, dominated_by(&derived, pad));
    assert_eq!(
        promoted.iter().copied().collect::<Vec<_>>(),
        [pad, tail],
        "the join is also reached directly from the body"
    );
    assert_eq!(
        canonical.handler(handler).body,
        HandlerBody::Unknown,
        "promotion is for a derived graph, and the canonical one is untouched"
    );
    let once = derived.clone();
    promote_handler_extents(&mut derived);
    assert_eq!(derived, once, "idempotent");
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

    let decisions = promote_handler_extents(&mut fixture.cfg);

    assert_eq!(
        decisions,
        [
            (RegionId::from_raw(0), ExtentPromotionStatus::Promoted),
            (
                region,
                ExtentPromotionStatus::AmbiguousExtent {
                    handler: HandlerRef::new(region, 0)
                }
            ),
        ],
        "only the reachable unknown handler's region"
    );
    assert_eq!(
        fixture.cfg.handler(HandlerRef::new(region, 0)).body,
        HandlerBody::Unknown,
        "an unreachable entry stays unknown"
    );
    let known_body = fixture
        .cfg
        .handler(HandlerRef::new(region, 1))
        .body
        .blocks();
    assert_eq!(known_body.map(BTreeSet::len), Some(1));
    let first = &fixture.cfg.region(RegionId::from_raw(0)).handlers[0];
    assert!(first.body.is_known());
    assert_eq!(
        first.entry, pad,
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
    let extents = [
        proposed(a[0], a0, bodies[0].1.clone()),
        proposed(a[1], a1, bodies[1].1.clone()),
        proposed(b, b0, bodies[2].1.clone()),
    ];

    let decisions = promote_exclusive_extents(&mut cfg, &extents);

    assert_eq!(
        decisions,
        [
            (a[0].region(), ExtentPromotionStatus::Promoted),
            (b.region(), ExtentPromotionStatus::Promoted),
        ]
    );
    for (handler, body) in &bodies {
        assert_eq!(cfg.handler(*handler).body.blocks(), Some(body));
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

    // `a[1]` has no candidate at all.
    let extents = [
        proposed(a[0], a0, BTreeSet::from([a0])),
        proposed(b, b0, BTreeSet::from([b0])),
    ];

    let decisions = promote_exclusive_extents(&mut cfg, &extents);

    assert_eq!(
        decisions[0].1,
        ExtentPromotionStatus::AmbiguousExtent { handler: a[1] }
    );
    assert_eq!(decisions[1].1, ExtentPromotionStatus::Promoted);
    assert_eq!(
        cfg.handler(a[0]).body,
        HandlerBody::Unknown,
        "the sibling with a candidate is withheld too"
    );
    assert!(cfg.handler(b).body.is_known());
}

#[test]
fn candidates_that_share_a_block_withhold_both_regions() {
    let Candidates {
        mut cfg,
        a,
        b,
        blocks: [a0, a1, b0, spare],
    } = candidates();

    let extents = [
        proposed(b, b0, BTreeSet::from([b0, spare])),
        proposed(a[0], a0, BTreeSet::from([a0])),
        proposed(a[1], a1, BTreeSet::from([a1, spare])),
    ];

    let decisions = promote_exclusive_extents(&mut cfg, &extents);

    // Region `a` is first in order, and its second handler overlaps region `b`.
    let overlap = ExtentPromotionStatus::OverlappingExtents {
        first: a[1],
        second: b,
    };
    assert_eq!(decisions.len(), 2);
    assert_eq!(decisions[0].1, overlap);
    assert_eq!(decisions[1].1, overlap);
    for handler in [a[0], a[1], b] {
        assert_eq!(cfg.handler(handler).body, HandlerBody::Unknown, "{handler}");
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
    let before = ControlTree::compute(&fixture.cfg);
    assert_eq!(before.report().unstructured_regions, [region]);

    let extents = recover_by_kind(&fixture.cfg);
    let mut derived = fixture.cfg.clone();
    let decisions = promote_exclusive_extents(&mut derived, &extents);

    assert_eq!(decisions, [(region, ExtentPromotionStatus::Promoted)]);
    let tree = ControlTree::compute(&derived);
    let report = tree.report();
    assert!(report.is_fully_structured(), "{report:?}");
    assert_eq!(
        tree.preorder()
            .filter(|&id| matches!(tree.node(id), ControlNode::TryCatch { .. }))
            .count(),
        1
    );
}
