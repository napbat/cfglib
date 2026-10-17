//! A frame-based SEH dispatch drawn as `Cfg` edges: search and unwind phases,
//! the filter's three dispositions, and normal-only versus full-flow views.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, DominatorTree, Edge, EdgeId, EdgeKind, EdgeView, EhBlockKind, EhEdgeKind,
    EhModel, ExceptionDisposition, ExceptionFlow, ExceptionPhase, FilteredEdges, HandlerBody,
    HandlerKind, RootedView, SehExceptionRegion, SehHandler, SehHandlerKind, install_seh_region,
    verify, verify_edge_view,
};

use crate::{LiftedCfg, set};

/// A function with a frame-based SEH `__try { ... } __except (filter) { ... }`
/// and the dispatch edges a lifter draws for it.
struct SehShape {
    cfg: LiftedCfg,
    prologue: BlockId,
    guarded: BlockId,
    filter: BlockId,
    handler: BlockId,
    after: BlockId,
    outer: BlockId,
    leave: EdgeId,
    dispatch: EdgeId,
    execute: EdgeId,
    resume: EdgeId,
    search_outward: EdgeId,
}

fn seh_function() -> SehShape {
    let mut cfg = LiftedCfg::with_edge_payload();
    let prologue = cfg.entry();
    let guarded = cfg.new_block();
    let filter = cfg.new_block();
    let handler = cfg.new_block();
    let after = cfg.new_block();
    let outer = cfg.new_block();
    for (block, text) in [
        (prologue, "prologue"),
        (guarded, "guarded call"),
        (filter, "filter: classify record"),
        (handler, "handle"),
        (after, "epilogue"),
        (outer, "propagate to caller"),
    ] {
        cfg.block_mut(block).push(text);
    }
    install_seh_region(
        &mut cfg,
        SehExceptionRegion {
            protected_blocks: set(&[guarded]),
            handlers: vec![SehHandler {
                entry: handler,
                body: HandlerBody::known([handler]),
                kind: SehHandlerKind::Except {
                    filter_block: filter,
                },
            }],
            parent: None,
        },
    );
    let search =
        |disposition| ExceptionFlow::exceptional(ExceptionPhase::Search, disposition, "filter");
    cfg.add_edge_with_payload(
        prologue,
        guarded,
        EdgeKind::Fallthrough,
        ExceptionFlow::normal("enter"),
    );
    // A normal exit from the guarded body is a leave, which stays sequential.
    let leave = cfg.add_edge_with_payload(
        guarded,
        after,
        EdgeKind::ExceptionLeave,
        ExceptionFlow::normal("leave"),
    );
    // First pass: the guarded call raises and the dispatcher evaluates the
    // filter. Nothing has been selected yet, so this is the search phase.
    let dispatch =
        cfg.add_edge_with_payload(guarded, filter, EdgeKind::ExceptionHandler, search(None));
    // The filter's three answers.
    let execute = cfg.add_edge_with_payload(
        filter,
        handler,
        EdgeKind::ExceptionHandler,
        ExceptionFlow::exceptional(
            ExceptionPhase::Unwind,
            Some(ExceptionDisposition::ExecuteHandler),
            "filter",
        ),
    );
    let resume = cfg.add_edge_with_payload(
        filter,
        guarded,
        EdgeKind::ExceptionContinue,
        search(Some(ExceptionDisposition::ContinueExecution)),
    );
    let search_outward = cfg.add_edge_with_payload(
        filter,
        outer,
        EdgeKind::ExceptionResume,
        search(Some(ExceptionDisposition::ContinueSearch)),
    );
    cfg.add_edge_with_payload(
        handler,
        after,
        EdgeKind::Fallthrough,
        ExceptionFlow::normal("handled"),
    );
    SehShape {
        cfg,
        prologue,
        guarded,
        filter,
        handler,
        after,
        outer,
        leave,
        dispatch,
        execute,
        resume,
        search_outward,
    }
}

/// The blocks `view` reaches from its root, by explicit search.
fn reachable_from_root<G: RootedView<NodeId = BlockId>>(
    view: &G,
    removed: Option<BlockId>,
) -> BTreeSet<BlockId> {
    let mut seen = BTreeSet::new();
    let mut pending = vec![view.root()];
    while let Some(block) = pending.pop() {
        if Some(block) != removed && seen.insert(block) {
            pending.extend(view.successors(block));
        }
    }
    seen
}

/// Every pair's dominance by removal, compared with `DominatorTree`.
fn assert_dominance_matches_removal<G: RootedView<NodeId = BlockId>>(view: &G) {
    let tree = DominatorTree::compute(view);
    let reachable = reachable_from_root(view, None);
    for dominator in view.node_ids() {
        let without = reachable_from_root(view, Some(dominator));
        for node in view.node_ids() {
            let expected =
                dominator == node || (reachable.contains(&node) && !without.contains(&node));
            assert_eq!(
                tree.dominates(dominator, node),
                expected,
                "{dominator} dominates {node}"
            );
        }
    }
}

#[test]
fn seh_filter_edges_carry_the_documented_phase_and_disposition_semantics() {
    let shape = seh_function();
    let cfg = &shape.cfg;
    assert!(verify(cfg).is_ok());

    // The payload and the structural kind agree on which edges are
    // exceptional; a leave is a normal transfer out of the guarded body.
    for edge in cfg.edges() {
        assert_eq!(
            edge.payload().phase().is_some(),
            edge.kind().is_exceptional()
        );
    }
    assert!(!cfg.edge(shape.leave).kind().is_exceptional());

    // Search happens before a handler is selected; unwinding after.
    let in_phase = |phase| {
        cfg.edges()
            .filter(|edge| edge.payload().phase() == Some(phase))
            .map(|edge| edge.id())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        in_phase(ExceptionPhase::Search),
        [shape.dispatch, shape.resume, shape.search_outward]
    );
    assert_eq!(in_phase(ExceptionPhase::Unwind), [shape.execute]);

    // Each disposition lands where its documentation says.
    let target_of = |edge| cfg.edge(edge).target();
    assert_eq!(
        cfg.edge(shape.execute).payload().disposition(),
        Some(ExceptionDisposition::ExecuteHandler)
    );
    assert_eq!(target_of(shape.execute), cfg.regions()[0].handlers[0].entry);
    assert_eq!(
        cfg.edge(shape.resume).payload().disposition(),
        Some(ExceptionDisposition::ContinueExecution)
    );
    assert_eq!(
        target_of(shape.resume),
        shape.guarded,
        "execution resumes in the protected block"
    );
    assert!(
        cfg.regions()[0]
            .protected_blocks
            .contains(&target_of(shape.resume))
    );
    assert_eq!(
        cfg.edge(shape.search_outward).payload().disposition(),
        Some(ExceptionDisposition::ContinueSearch)
    );
    assert!(
        !cfg.regions()[0]
            .protected_blocks
            .contains(&target_of(shape.search_outward))
    );
    assert_eq!(target_of(shape.search_outward), shape.outer);
    assert_eq!(cfg.edge(shape.dispatch).payload().disposition(), None);
    assert_eq!(target_of(shape.dispatch), shape.filter);
}

#[test]
fn the_exception_model_reads_a_seh_dispatch_as_handler_resume_continue_and_leave_edges() {
    let shape = seh_function();
    let model = EhModel::compute(&shape.cfg);
    // Edge identities are allocated in insertion order, so sorting by them
    // lists the edges the way the lifter drew them.
    let mut kinds: Vec<_> = model
        .eh_edges()
        .iter()
        .map(|edge| (edge.edge_id, edge.kind))
        .collect();
    kinds.sort_by_key(|&(edge, _)| edge.index());
    assert_eq!(
        kinds,
        [
            (shape.leave, EhEdgeKind::Leave),
            (shape.dispatch, EhEdgeKind::Handler),
            (shape.execute, EhEdgeKind::Handler),
            (shape.resume, EhEdgeKind::Continue),
            (shape.search_outward, EhEdgeKind::Resume),
        ]
    );
    assert!(model.eh_edges().iter().all(|edge| !edge.is_unwind));
    // The route from the handler entry back to the installed handler.
    let route = model.handlers(shape.handler);
    assert_eq!(route.len(), 1);
    assert_eq!(
        shape.cfg.handler(route[0]).unwrap().kind,
        HandlerKind::Filter {
            filter_block: shape.filter
        }
    );
    assert_eq!(model.block_kind(shape.handler), EhBlockKind::CatchSwitch);
    assert!(model.protected_by(shape.handler).contains(&shape.guarded));
    assert_eq!(model.block_kind(shape.prologue), EhBlockKind::Normal);
    assert_eq!(model.block_kind(shape.after), EhBlockKind::Normal);
}

#[test]
fn a_block_that_is_both_a_dispatch_target_and_a_resume_point_is_classified_independently_of_edge_order()
 {
    // The filter block receives a handler edge and also resumes the search:
    // the same graph with its edges inserted in the other order must still
    // classify every block the same way.
    let build = |reverse: bool| {
        let mut cfg = Cfg::<&str>::new();
        let guarded = cfg.entry();
        let filter = cfg.new_block();
        let outer = cfg.new_block();
        let mut edges = [
            (guarded, filter, EdgeKind::ExceptionHandler),
            (filter, outer, EdgeKind::ExceptionResume),
        ];
        if reverse {
            edges.reverse();
        }
        for (source, target, kind) in edges {
            cfg.add_edge(source, target, kind);
        }
        (EhModel::compute(&cfg), filter)
    };
    let (forward, filter) = build(false);
    let (backward, _) = build(true);
    assert_eq!(forward.block_kind(filter), backward.block_kind(filter));
    assert_eq!(forward.landing_pads(), backward.landing_pads());
    assert_eq!(forward.resume_blocks(), backward.resume_blocks());
}

#[test]
fn the_normal_only_view_hides_the_handler_the_full_flow_reaches() {
    let shape = seh_function();
    let cfg = &shape.cfg;
    let normal = FilteredEdges::new(cfg, |_: EdgeId, edge: &Edge<ExceptionFlow<&str>>| {
        edge.payload().phase().is_none()
    });
    assert!(verify_edge_view(cfg).is_ok());
    assert!(verify_edge_view(&normal).is_ok());

    let full_reach = reachable_from_root(cfg, None);
    let normal_reach = reachable_from_root(&normal, None);
    assert_eq!(
        full_reach,
        set(&[
            shape.prologue,
            shape.guarded,
            shape.filter,
            shape.handler,
            shape.after,
            shape.outer
        ])
    );
    // Normal flow enters the guarded body and leaves it: the filter, the
    // handler, and the propagation block are exceptional-only.
    assert_eq!(
        normal_reach,
        set(&[shape.prologue, shape.guarded, shape.after])
    );

    assert_dominance_matches_removal(cfg);
    assert_dominance_matches_removal(&normal);
    let full = DominatorTree::compute(cfg);
    assert!(full.dominates(shape.guarded, shape.filter));
    assert!(full.dominates(shape.filter, shape.handler));
    assert_eq!(full.idom(shape.after), Some(shape.guarded));
    let only_normal = DominatorTree::compute(&normal);
    assert!(!only_normal.is_reachable(shape.handler));
    assert!(only_normal.dominates(shape.guarded, shape.after));
}

#[test]
fn payload_and_kind_predicates_select_the_same_normal_edges() {
    let shape = seh_function();
    let cfg = &shape.cfg;
    let by_payload = FilteredEdges::new(cfg, |_: EdgeId, edge: &Edge<ExceptionFlow<&str>>| {
        edge.payload().phase().is_none()
    });
    let by_kind = FilteredEdges::new(cfg, |_: EdgeId, edge: &Edge<ExceptionFlow<&str>>| {
        !edge.kind().is_exceptional()
    });
    assert_eq!(
        by_payload.edge_ids().collect::<Vec<_>>(),
        by_kind.edge_ids().collect::<Vec<_>>()
    );
    assert_eq!(by_payload.edge_ids().count(), 3);

    // Phase views partition the exceptional edges.
    let search = FilteredEdges::new(cfg, |_: EdgeId, edge: &Edge<ExceptionFlow<&str>>| {
        edge.payload().phase() == Some(ExceptionPhase::Search)
    });
    let unwind = FilteredEdges::new(cfg, |_: EdgeId, edge: &Edge<ExceptionFlow<&str>>| {
        edge.payload().phase() == Some(ExceptionPhase::Unwind)
    });
    assert_eq!(
        search.edge_ids().count() + unwind.edge_ids().count() + by_payload.edge_ids().count(),
        cfg.edge_count()
    );
    // Every view keeps the full graph's edge numbering.
    assert_eq!(unwind.edge_bound(), cfg.edge_bound());
    assert_eq!(unwind.edge_ids().collect::<Vec<_>>(), [shape.execute]);
}
