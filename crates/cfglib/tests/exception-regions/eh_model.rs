//! `EhModel`: how a CFG's exception edges and regions classify its blocks, and
//! the normal-versus-unwind flows a `FilteredEdges` view separates.

use cfglib::{
    BlockId, Cfg, DominatorTree, Edge, EdgeId, EdgeKind, EdgeView, EhBlockKind, EhEdgeKind,
    EhModel, FilteredEdges, HandlerKind, HandlerRef, TraversalDirection, reachable,
};

use crate::shapes::{Stmt, block_of, catch, handler, protect, try_catch};

#[test]
fn a_graph_without_exception_edges_or_regions_has_nothing_to_model() {
    let mut cfg = Cfg::<Stmt>::new();
    let next = block_of(&mut cfg, &["next"]);
    cfg.add_edge(cfg.entry(), next, EdgeKind::Fallthrough);

    let model = EhModel::compute(&cfg);

    assert_eq!(model.eh_edges(), []);
    assert_eq!(model.block_bound(), cfg.block_bound());
    assert!(
        model
            .block_kinds()
            .all(|(_, kind)| kind == EhBlockKind::Normal)
    );
    assert_eq!(model.landing_pads(), []);
    assert_eq!(model.cleanup_blocks(), []);
    assert_eq!(model.resume_blocks(), []);
}

#[test]
fn handler_and_unwind_edges_make_landing_pads_and_record_who_can_raise_into_them() {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    let body = block_of(&mut cfg, &["call()"]);
    let handled = block_of(&mut cfg, &["handled"]);
    let unwound = block_of(&mut cfg, &["unwound"]);
    cfg.add_edge(entry, body, EdgeKind::Fallthrough);
    let handler_edge = cfg.add_edge(entry, handled, EdgeKind::ExceptionHandler);
    let unwind_edge = cfg.add_edge(body, unwound, EdgeKind::ExceptionUnwind);

    let model = EhModel::compute(&cfg);

    assert_eq!(model.landing_pads(), [handled, unwound]);
    assert_eq!(model.block_kind(handled), EhBlockKind::LandingPad);
    assert_eq!(model.block_kind(entry), EhBlockKind::Normal);
    assert_eq!(
        model
            .protected_by(handled)
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [entry]
    );
    assert_eq!(
        model
            .protected_by(unwound)
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [body]
    );
    assert!(model.protected_by(entry).is_empty());

    let edges = model.eh_edges();
    assert_eq!(edges.len(), 2);
    assert_eq!(
        (edges[0].edge_id, edges[0].from, edges[0].to, edges[0].kind),
        (handler_edge, entry, handled, EhEdgeKind::Handler)
    );
    assert_eq!(
        (edges[1].edge_id, edges[1].from, edges[1].to, edges[1].kind),
        (unwind_edge, body, unwound, EhEdgeKind::Unwind)
    );
    assert!(!edges[0].kind.is_unwind() && !edges[0].is_unwind);
    assert!(edges[1].kind.is_unwind() && edges[1].is_unwind);
}

#[test]
fn resume_and_continue_edges_mark_their_sources_not_their_targets() {
    let mut cfg = Cfg::<Stmt>::new();
    let rethrow = block_of(&mut cfg, &["rethrow"]);
    let outer = block_of(&mut cfg, &["outer dispatch"]);
    let resumed = block_of(&mut cfg, &["resume execution"]);
    let fixed_up = block_of(&mut cfg, &["fix up"]);
    cfg.add_edge(cfg.entry(), rethrow, EdgeKind::Fallthrough);
    cfg.add_edge(rethrow, outer, EdgeKind::ExceptionResume);
    cfg.add_edge(cfg.entry(), fixed_up, EdgeKind::Fallthrough);
    cfg.add_edge(fixed_up, resumed, EdgeKind::ExceptionContinue);

    let model = EhModel::compute(&cfg);

    assert_eq!(model.resume_blocks(), [rethrow, fixed_up]);
    assert_eq!(model.block_kind(outer), EhBlockKind::Normal);
    assert_eq!(model.block_kind(resumed), EhBlockKind::Normal);
    let kinds: Vec<EhEdgeKind> = model.eh_edges().iter().map(|edge| edge.kind).collect();
    assert_eq!(kinds, [EhEdgeKind::Resume, EhEdgeKind::Continue]);
    assert!(kinds.iter().all(|kind| !kind.is_unwind()));
}

#[test]
fn a_leave_edge_is_listed_but_classifies_no_block() {
    let mut cfg = Cfg::<Stmt>::new();
    let after = block_of(&mut cfg, &["after"]);
    let leave = cfg.add_edge(cfg.entry(), after, EdgeKind::ExceptionLeave);

    let model = EhModel::compute(&cfg);

    assert_eq!(model.eh_edges().len(), 1);
    assert_eq!(model.eh_edges()[0].edge_id, leave);
    assert_eq!(model.eh_edges()[0].kind, EhEdgeKind::Leave);
    assert!(
        model
            .block_kinds()
            .all(|(_, kind)| kind == EhBlockKind::Normal)
    );
    assert!(model.protected_by(after).is_empty());
}

#[test]
fn region_metadata_decides_the_role_an_unwind_edge_only_guesses() {
    // Every pad is reached by an unwind edge, which alone says `LandingPad`.
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    let catch_pad = block_of(&mut cfg, &["catch"]);
    let catch_all = block_of(&mut cfg, &["catch all"]);
    let finally = block_of(&mut cfg, &["finally"]);
    let fault = block_of(&mut cfg, &["fault"]);
    let filtered = block_of(&mut cfg, &["filtered"]);
    let filter = block_of(&mut cfg, &["filter predicate"]);
    for pad in [catch_pad, catch_all, finally, fault, filtered] {
        cfg.add_edge(entry, pad, EdgeKind::ExceptionUnwind);
    }
    protect(
        &mut cfg,
        &[entry],
        vec![
            handler(catch_pad, HandlerKind::Catch, &[catch_pad]),
            handler(catch_all, HandlerKind::CatchAll, &[catch_all]),
            handler(finally, HandlerKind::Finally, &[finally]),
            handler(fault, HandlerKind::Fault, &[fault]),
            handler(
                filtered,
                HandlerKind::Filter {
                    filter_block: filter,
                },
                &[filtered],
            ),
        ],
    );

    let model = EhModel::compute(&cfg);

    assert_eq!(model.block_kind(catch_pad), EhBlockKind::LandingPad);
    assert_eq!(model.block_kind(catch_all), EhBlockKind::LandingPad);
    assert_eq!(model.block_kind(finally), EhBlockKind::Cleanup);
    assert_eq!(model.block_kind(fault), EhBlockKind::Cleanup);
    assert_eq!(model.block_kind(filtered), EhBlockKind::CatchSwitch);
    assert_eq!(model.landing_pads(), [catch_pad, catch_all]);
    assert_eq!(model.cleanup_blocks(), [finally, fault]);
    assert_eq!(
        model.block_kind(filter),
        EhBlockKind::Normal,
        "the predicate block is named by the handler, not classified by it"
    );
}

#[test]
fn a_handler_no_exception_edge_enters_is_still_classified_by_its_region() {
    let mut fixture = try_catch(&["log(e)"]);
    let unwind: Vec<EdgeId> = fixture
        .cfg
        .edge_ids()
        .filter(|&edge| fixture.cfg.edge(edge).kind() == EdgeKind::ExceptionUnwind)
        .collect();
    for edge in unwind {
        fixture.cfg.remove_edge(edge);
    }

    let model = EhModel::compute(&fixture.cfg);

    assert_eq!(model.eh_edges(), []);
    assert_eq!(model.block_kind(fixture.pad), EhBlockKind::LandingPad);
    assert_eq!(
        model
            .protected_by(fixture.pad)
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [fixture.body],
        "the region alone says what the pad protects"
    );
}

#[test]
fn every_handler_entered_at_a_block_is_listed_in_region_order() {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    let shared = block_of(&mut cfg, &["shared pad"]);
    let other = block_of(&mut cfg, &["other pad"]);
    cfg.add_edge(entry, shared, EdgeKind::ExceptionUnwind);
    let first = protect(&mut cfg, &[entry], vec![catch(shared)]);
    let second = protect(&mut cfg, &[entry], vec![catch(other), catch(shared)]);

    let model = EhModel::compute(&cfg);

    assert_eq!(
        model.handlers(shared),
        [HandlerRef::new(first, 0), HandlerRef::new(second, 1)]
    );
    assert_eq!(model.handlers(other), [HandlerRef::new(second, 0)]);
    assert_eq!(model.handlers(entry), []);
}

#[test]
fn protection_unions_the_region_with_the_sources_of_edges_into_the_pad() {
    let mut fixture = try_catch(&["log(e)"]);
    let extra = block_of(&mut fixture.cfg, &["also raises"]);
    fixture
        .cfg
        .add_edge(fixture.join, extra, EdgeKind::Fallthrough);
    fixture
        .cfg
        .add_edge(extra, fixture.pad, EdgeKind::ExceptionUnwind);

    let model = EhModel::compute(&fixture.cfg);

    let protected: Vec<BlockId> = model.protected_by(fixture.pad).iter().copied().collect();
    assert_eq!(protected, [fixture.body, extra]);
}

#[test]
fn a_block_added_after_the_model_reads_as_normal() {
    let mut fixture = try_catch(&["log(e)"]);
    let model = EhModel::compute(&fixture.cfg);
    let late = block_of(&mut fixture.cfg, &["late"]);

    assert_eq!(model.block_kind(late), EhBlockKind::Normal);
    assert!(model.block_bound() <= late.index());
    assert_eq!(model.handlers(late), []);
    assert!(model.protected_by(late).is_empty());
    assert!(model.cleanup(late).is_none());
}

#[test]
fn model_edges_lead_back_to_the_callers_payloads_by_edge_identity() {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Dispatch {
        Normal,
        Order(u8),
    }

    let mut cfg = Cfg::<Stmt, Dispatch>::with_edge_payload();
    let entry = cfg.entry();
    let first = cfg.new_block();
    let second = cfg.new_block();
    cfg.add_edge_with_payload(entry, entry, EdgeKind::Back, Dispatch::Normal);
    let to_first =
        cfg.add_edge_with_payload(entry, first, EdgeKind::ExceptionHandler, Dispatch::Order(0));
    let to_second = cfg.add_edge_with_payload(
        entry,
        second,
        EdgeKind::ExceptionHandler,
        Dispatch::Order(1),
    );

    let model = EhModel::compute(&cfg);

    let order: Vec<(EdgeId, Dispatch)> = model
        .eh_edges()
        .iter()
        .map(|edge| (edge.edge_id, *cfg.edge(edge.edge_id).payload()))
        .collect();
    assert_eq!(
        order,
        [
            (to_first, Dispatch::Order(0)),
            (to_second, Dispatch::Order(1))
        ]
    );
}

fn is_normal(_: EdgeId, edge: &Edge<()>) -> bool {
    !edge.kind().is_exceptional()
}

/// Blocks reachable from the entry along the edges `accept` keeps, by a plain
/// depth-first search over the edge list.
fn reference_reachable(cfg: &Cfg<Stmt>, accept: impl Fn(EdgeKind) -> bool) -> Vec<bool> {
    let mut seen = vec![false; cfg.block_bound()];
    let mut stack = vec![cfg.entry()];
    while let Some(block) = stack.pop() {
        if !std::mem::replace(&mut seen[block.index()], true) {
            stack.extend(
                cfg.outgoing(block)
                    .map(|edge| cfg.edge(edge))
                    .filter(|edge| accept(edge.kind()))
                    .map(|edge| edge.target()),
            );
        }
    }
    seen
}

#[test]
fn normal_flow_never_reaches_a_pad_that_the_full_graph_reaches() {
    let fixture = try_catch(&["log(e)"]);
    let cfg = &fixture.cfg;
    let normal = FilteredEdges::new(cfg, is_normal);

    let along_normal = reachable(&normal, [cfg.entry()], TraversalDirection::Outgoing);
    let along_all = reachable(cfg, [cfg.entry()], TraversalDirection::Outgoing);

    assert_eq!(
        along_normal,
        reference_reachable(cfg, |kind| !kind.is_exceptional())
    );
    assert_eq!(along_all, reference_reachable(cfg, |_| true));
    assert!(!along_normal[fixture.pad.index()]);
    assert!(along_all[fixture.pad.index()]);
    assert!(along_normal[fixture.join.index()]);
}

#[test]
fn dominance_over_the_normal_view_leaves_the_handler_out() {
    let fixture = try_catch(&["log(e)"]);
    let cfg = &fixture.cfg;

    let full = DominatorTree::compute(cfg);
    let normal = DominatorTree::compute(&FilteredEdges::new(cfg, is_normal));

    assert!(full.is_reachable(fixture.pad));
    assert!(
        full.dominates(fixture.body, fixture.pad),
        "only an unwind enters it"
    );
    assert!(!normal.is_reachable(fixture.pad));
    assert!(normal.dominates(fixture.body, fixture.join));
    assert!(
        !full.dominates(fixture.pad, fixture.join),
        "the join is also reached directly from the body"
    );
}

#[test]
fn the_view_rejects_exactly_the_exceptional_edges_the_model_lists() {
    let mut fixture = try_catch(&["log(e)"]);
    let after = block_of(&mut fixture.cfg, &["after"]);
    fixture
        .cfg
        .add_edge(fixture.join, after, EdgeKind::ExceptionLeave);
    let cfg = &fixture.cfg;

    let normal = FilteredEdges::new(cfg, is_normal);
    let kept: Vec<EdgeId> = normal.edge_ids().collect();
    let rejected: Vec<EdgeId> = cfg.edge_ids().filter(|edge| !kept.contains(edge)).collect();
    let model = EhModel::compute(cfg);

    let listed_exceptional: Vec<EdgeId> = model
        .eh_edges()
        .iter()
        .filter(|edge| edge.kind != EhEdgeKind::Leave)
        .map(|edge| edge.edge_id)
        .collect();
    assert_eq!(rejected, listed_exceptional);
    assert_eq!(model.eh_edges().len(), rejected.len() + 1, "plus the leave");
    assert_eq!(
        normal.edge_bound(),
        cfg.edge_bound(),
        "identities are not renumbered"
    );
}
