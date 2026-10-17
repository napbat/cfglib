//! CLR clause installation and the exception model over the installed shape.

use cfglib::{
    BlockId, Cfg, ClrExceptionRegion, ClrHandler, ClrHandlerKind, CompletionReason, Continuation,
    EdgeKind, EhBlockKind, EhEdgeKind, EhModel, ExceptionFlow, ExceptionPhase, Handler,
    HandlerBody, HandlerKind, HandlerRef, HandlerTypes, RegionId, install_clr_region, verify,
};

use crate::{LiftedCfg, set};

fn clr_handler<T>(entry: BlockId, kind: ClrHandlerKind<T>) -> ClrHandler<T> {
    ClrHandler {
        entry,
        body: HandlerBody::known([entry]),
        kind,
    }
}

#[test]
fn clr_clauses_keep_their_kind_and_only_typed_catches_get_a_type_entry() {
    let mut cfg = LiftedCfg::with_edge_payload();
    let protected = cfg.entry();
    let catch_io = cfg.new_block();
    let catch_all = cfg.new_block();
    let finally = cfg.new_block();
    let fault = cfg.new_block();
    let filter_predicate = cfg.new_block();
    let filtered = cfg.new_block();
    let catch_net = cfg.new_block();
    let mut types = HandlerTypes::new();

    let region = install_clr_region(
        &mut cfg,
        &mut types,
        ClrExceptionRegion {
            protected_blocks: set(&[protected]),
            handlers: vec![
                clr_handler(
                    catch_io,
                    ClrHandlerKind::Catch {
                        ty: "System.IO.IOException",
                    },
                ),
                clr_handler(catch_all, ClrHandlerKind::CatchAll),
                clr_handler(finally, ClrHandlerKind::Finally),
                clr_handler(fault, ClrHandlerKind::Fault),
                ClrHandler {
                    entry: filtered,
                    body: HandlerBody::Unknown,
                    kind: ClrHandlerKind::Filter {
                        filter_block: filter_predicate,
                    },
                },
                clr_handler(
                    catch_net,
                    ClrHandlerKind::Catch {
                        ty: "System.Net.WebException",
                    },
                ),
            ],
            parent: None,
        },
    );

    assert_eq!(region, RegionId::from_raw(0));
    let installed = cfg
        .region(region)
        .expect("the returned id names the region");
    assert_eq!(installed.id, region);
    assert_eq!(installed.protected_blocks, set(&[protected]));
    assert_eq!(installed.parent, None);
    assert_eq!(
        installed.handlers,
        [
            Handler {
                entry: catch_io,
                body: HandlerBody::known([catch_io]),
                kind: HandlerKind::Catch,
            },
            Handler {
                entry: catch_all,
                body: HandlerBody::known([catch_all]),
                kind: HandlerKind::CatchAll,
            },
            Handler {
                entry: finally,
                body: HandlerBody::known([finally]),
                kind: HandlerKind::Finally,
            },
            Handler {
                entry: fault,
                body: HandlerBody::known([fault]),
                kind: HandlerKind::Fault,
            },
            Handler {
                entry: filtered,
                body: HandlerBody::Unknown,
                kind: HandlerKind::Filter {
                    filter_block: filter_predicate,
                },
            },
            Handler {
                entry: catch_net,
                body: HandlerBody::known([catch_net]),
                kind: HandlerKind::Catch,
            },
        ]
    );

    // Type tokens are keyed by the clause's position in the table, not by its
    // position among the typed clauses.
    let io = HandlerRef::new(region, 0);
    let net = HandlerRef::new(region, 5);
    assert_eq!(types.len(), 2);
    assert_eq!(types.metadata(io), Some(&"System.IO.IOException"));
    assert_eq!(types.metadata(net), Some(&"System.Net.WebException"));
    for untyped in 1..5 {
        assert_eq!(types.metadata(HandlerRef::new(region, untyped)), None);
    }
    assert_eq!(
        types.iter().collect::<Vec<_>>(),
        [
            (io, &"System.IO.IOException"),
            (net, &"System.Net.WebException")
        ]
    );
    assert_eq!(
        cfg.handler(net).map(|handler| handler.entry),
        Some(catch_net)
    );

    // Installing only records metadata: no block or edge appears.
    assert_eq!(cfg.block_count(), 8);
    assert_eq!(cfg.edge_count(), 0);
    assert!(verify(&cfg).is_ok());
}

#[test]
fn nested_clr_regions_get_sequential_ids_and_the_later_one_is_innermost() {
    let mut cfg = Cfg::<&str>::new();
    let outer_body = cfg.entry();
    let inner_body = cfg.new_block();
    let outer_catch = cfg.new_block();
    let inner_catch = cfg.new_block();
    let mut types = HandlerTypes::new();

    let outer = install_clr_region(
        &mut cfg,
        &mut types,
        ClrExceptionRegion {
            protected_blocks: set(&[outer_body, inner_body]),
            handlers: vec![clr_handler(
                outer_catch,
                ClrHandlerKind::Catch {
                    ty: "T".to_string(),
                },
            )],
            parent: None,
        },
    );
    let inner = install_clr_region(
        &mut cfg,
        &mut types,
        ClrExceptionRegion {
            protected_blocks: set(&[inner_body]),
            handlers: vec![clr_handler(
                inner_catch,
                ClrHandlerKind::Catch {
                    ty: "T".to_string(),
                },
            )],
            parent: Some(outer),
        },
    );

    assert_eq!((outer.index(), inner.index()), (0, 1));
    assert_eq!(cfg.regions().len(), 2);
    assert_eq!(cfg.region(inner).unwrap().parent, Some(outer));
    // Two regions can carry an equal token: handlers, not tokens, are keys.
    assert_eq!(types.len(), 2);
    assert_eq!(
        types
            .metadata(HandlerRef::new(outer, 0))
            .map(String::as_str),
        Some("T")
    );
    assert_eq!(
        types
            .metadata(HandlerRef::new(inner, 0))
            .map(String::as_str),
        Some("T")
    );
    assert_eq!(
        cfg.protecting_region(outer_body).map(|region| region.id),
        Some(outer)
    );
    assert_eq!(
        cfg.protecting_region(inner_body).map(|region| region.id),
        Some(inner)
    );
    assert_eq!(
        cfg.protecting_region(outer_catch).map(|region| region.id),
        None
    );
}

#[test]
fn clr_regions_install_into_a_cfg_with_any_edge_payload() {
    let mut unit = Cfg::<u32>::new();
    let body = unit.entry();
    let handler = unit.new_block();
    let mut types = HandlerTypes::new();
    let region = install_clr_region(
        &mut unit,
        &mut types,
        ClrExceptionRegion {
            protected_blocks: set(&[body]),
            handlers: vec![clr_handler(
                handler,
                ClrHandlerKind::Catch {
                    ty: 0x0200_0001_u32,
                },
            )],
            parent: None,
        },
    );
    assert_eq!(
        types.metadata(HandlerRef::new(region, 0)),
        Some(&0x0200_0001)
    );

    let mut flowing = Cfg::<u32, ExceptionFlow<u64>>::with_edge_payload();
    let body = flowing.entry();
    let handler = flowing.new_block();
    let mut types = HandlerTypes::<u32>::new();
    let region = install_clr_region(
        &mut flowing,
        &mut types,
        ClrExceptionRegion {
            protected_blocks: set(&[body]),
            handlers: vec![clr_handler(handler, ClrHandlerKind::<u32>::CatchAll)],
            parent: None,
        },
    );
    assert!(types.is_empty());
    assert_eq!(flowing.region(region).unwrap().handlers.len(), 1);
}

/// A CLR method with every clause kind, each handler reached by one unwind
/// edge from the protected block, as a lifter wires them.
fn clr_method_with_all_clauses() -> (
    LiftedCfg,
    HandlerTypes<&'static str>,
    RegionId,
    [BlockId; 6],
) {
    let mut cfg = LiftedCfg::with_edge_payload();
    let protected = cfg.entry();
    let handlers = [(); 6].map(|()| cfg.new_block());
    let [catch_io, catch_all, finally, fault, filtered, catch_net] = handlers;
    let filter_predicate = cfg.new_block();
    let mut types = HandlerTypes::new();
    let region = install_clr_region(
        &mut cfg,
        &mut types,
        ClrExceptionRegion {
            protected_blocks: set(&[protected]),
            handlers: vec![
                clr_handler(catch_io, ClrHandlerKind::Catch { ty: "IOException" }),
                clr_handler(catch_all, ClrHandlerKind::CatchAll),
                clr_handler(finally, ClrHandlerKind::Finally),
                clr_handler(fault, ClrHandlerKind::Fault),
                clr_handler(
                    filtered,
                    ClrHandlerKind::Filter {
                        filter_block: filter_predicate,
                    },
                ),
                clr_handler(catch_net, ClrHandlerKind::Catch { ty: "WebException" }),
            ],
            parent: None,
        },
    );
    for handler in handlers {
        cfg.add_edge_with_payload(
            protected,
            handler,
            EdgeKind::ExceptionUnwind,
            ExceptionFlow::exceptional(ExceptionPhase::Unwind, None, "clause"),
        );
    }
    (cfg, types, region, handlers)
}

#[test]
fn the_exception_model_classifies_installed_clauses_by_kind_and_routes_back_to_them() {
    let (cfg, types, region, handlers) = clr_method_with_all_clauses();
    let [catch_io, catch_all, finally, fault, filtered, catch_net] = handlers;
    let model = EhModel::compute(&cfg);

    assert_eq!(model.landing_pads(), [catch_io, catch_all, catch_net]);
    assert_eq!(model.cleanup_blocks(), [finally, fault]);
    assert_eq!(model.block_kind(filtered), EhBlockKind::CatchSwitch);
    assert_eq!(model.block_kind(cfg.entry()), EhBlockKind::Normal);
    assert_eq!(model.resume_blocks(), Vec::<BlockId>::new());

    assert_eq!(model.eh_edges().len(), 6);
    for edge in model.eh_edges() {
        assert_eq!(edge.kind, EhEdgeKind::Unwind);
        assert!(edge.is_unwind);
        assert_eq!(edge.from, cfg.entry());
        // The model keeps the identity of the lifter's own edge, so the
        // phase and metadata the lifter attached stay reachable.
        let payload = cfg.edge(edge.edge_id).payload();
        assert_eq!(payload.phase(), Some(ExceptionPhase::Unwind));
        assert_eq!(*payload.metadata(), "clause");
    }
    for (index, handler) in handlers.into_iter().enumerate() {
        assert_eq!(model.protected_by(handler), &set(&[cfg.entry()]));
        assert_eq!(model.handlers(handler), [HandlerRef::new(region, index)]);
    }
    // From a handler entry back to the clause's type token.
    let route = model.handlers(catch_net)[0];
    assert_eq!(types.metadata(route), Some(&"WebException"));
    assert_eq!(cfg.handler(route).unwrap().kind, HandlerKind::Catch);
}

#[test]
fn a_clr_finally_records_where_each_completion_reason_resumes() {
    let mut cfg = Cfg::<&str>::new();
    let body = cfg.entry();
    let cleanup = cfg.new_block();
    let after = cfg.new_block();
    let exit = cfg.new_block();
    let mut types = HandlerTypes::<()>::new();
    let region = install_clr_region(
        &mut cfg,
        &mut types,
        ClrExceptionRegion {
            protected_blocks: set(&[body]),
            handlers: vec![clr_handler(cleanup, ClrHandlerKind::Finally)],
            parent: None,
        },
    );
    let finally = HandlerRef::new(region, 0);
    cfg.set_cleanup_resume(finally, cleanup);
    for route in [
        Continuation {
            reason: CompletionReason::Normal,
            resume: after,
        },
        Continuation {
            reason: CompletionReason::Return,
            resume: exit,
        },
        // Recording a route twice keeps one.
        Continuation {
            reason: CompletionReason::Normal,
            resume: after,
        },
    ] {
        cfg.add_continuation(finally, route);
    }

    let model = EhModel::compute(&cfg);
    let recorded = model
        .cleanup(cleanup)
        .expect("the finally has a cleanup record");
    assert_eq!(recorded.handler, finally);
    assert_eq!(recorded.resume_from, Some(cleanup));
    assert_eq!(recorded.continuations.len(), 2);
    assert_eq!(
        recorded
            .resumes_for(CompletionReason::Normal)
            .collect::<Vec<_>>(),
        [after]
    );
    assert_eq!(
        recorded
            .resumes_for(CompletionReason::Return)
            .collect::<Vec<_>>(),
        [exit]
    );
    assert_eq!(recorded.resumes_for(CompletionReason::Unwind).count(), 0);
    assert_eq!(
        model.cleanups().map(|(block, _)| block).collect::<Vec<_>>(),
        [cleanup]
    );
    assert_eq!(model.block_kind(cleanup), EhBlockKind::Cleanup);
}
