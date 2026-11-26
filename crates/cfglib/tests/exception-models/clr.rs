//! CLR clause tables registered as regions, and the exception model over the
//! registered shape.
//!
//! A lifter maps each clause 1:1 onto a [`HandlerKind`] and keeps the
//! catch-type token of every typed `catch` in a [`HandlerMetadata`] side
//! table keyed by the clause's position in the table.

use cfglib::{
    BlockId, Cfg, CompletionReason, Continuation, EdgeKind, EhBlockKind, EhModel, ExceptionFlow,
    ExceptionPhase, Handler, HandlerBody, HandlerKind, HandlerMetadata, HandlerRef, Region,
    RegionId, RegionIndex, verify,
};

use crate::{LiftedCfg, set};

/// A clause whose body is just its entry block.
fn clause(entry: BlockId, kind: HandlerKind) -> Handler {
    Handler {
        entry,
        body: HandlerBody::known([entry]),
        kind,
    }
}

/// The blocks that the clause table of the clause-kind test refers to: the
/// protected block, the entry of every clause, and the filter's predicate.
struct ClauseBlocks {
    protected: BlockId,
    catch_io: BlockId,
    catch_all: BlockId,
    finally: BlockId,
    fault: BlockId,
    filter_predicate: BlockId,
    filtered: BlockId,
    catch_net: BlockId,
}

/// Adds the blocks of the clause table, in the order of its fields.
fn clause_blocks(cfg: &mut LiftedCfg) -> ClauseBlocks {
    ClauseBlocks {
        protected: cfg.entry(),
        catch_io: cfg.new_block(),
        catch_all: cfg.new_block(),
        finally: cfg.new_block(),
        fault: cfg.new_block(),
        filter_predicate: cfg.new_block(),
        filtered: cfg.new_block(),
        catch_net: cfg.new_block(),
    }
}

/// The handlers of the clause table, one of every kind, in table order.
fn every_clause_kind(blocks: &ClauseBlocks) -> Vec<Handler> {
    vec![
        clause(blocks.catch_io, HandlerKind::Catch),
        clause(blocks.catch_all, HandlerKind::CatchAll),
        clause(blocks.finally, HandlerKind::Finally),
        clause(blocks.fault, HandlerKind::Fault),
        Handler {
            entry: blocks.filtered,
            body: HandlerBody::Unknown,
            kind: HandlerKind::Filter {
                filter_block: blocks.filter_predicate,
            },
        },
        clause(blocks.catch_net, HandlerKind::Catch),
    ]
}

/// Registers one region with a clause of every kind, in table order, keeps
/// the type token of each typed catch, and returns the region's id.
fn register_every_clause_kind(
    cfg: &mut LiftedCfg,
    types: &mut HandlerMetadata<&'static str>,
    blocks: &ClauseBlocks,
) -> RegionId {
    let region = cfg
        .add_region(Region {
            protected_blocks: set(&[blocks.protected]),
            handlers: every_clause_kind(blocks),
            parent: None,
        })
        .unwrap();
    types.set_all(
        region,
        [(0, "System.IO.IOException"), (5, "System.Net.WebException")],
    );
    region
}

/// The registered region has the returned id and protected blocks, and every
/// clause keeps its native kind and table position.
fn assert_registered_clauses_keep_their_kind(
    cfg: &LiftedCfg,
    region: RegionId,
    blocks: &ClauseBlocks,
) {
    assert_eq!(region, RegionId::from_raw(0));
    let registered = cfg.region(region);
    assert_eq!(registered.protected_blocks, set(&[blocks.protected]));
    assert_eq!(registered.parent, None);
    assert_eq!(registered.handlers, every_clause_kind(blocks));
    assert_eq!(
        cfg.handlers()
            .map(|(handler, registered)| (handler.index(), registered.kind.name()))
            .collect::<Vec<_>>(),
        [
            (0, "catch"),
            (1, "catch_all"),
            (2, "finally"),
            (3, "fault"),
            (4, "filter"),
            (5, "catch"),
        ]
    );
}

/// Only the two typed catches have a type entry, and the handler behind a
/// type token is the clause it names.
fn assert_only_typed_catches_get_a_type_entry(
    cfg: &LiftedCfg,
    types: &HandlerMetadata<&'static str>,
    region: RegionId,
    catch_net: BlockId,
) {
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
    assert_eq!(cfg.handler(net).entry, catch_net);
    assert_eq!(cfg.handler(net).kind, HandlerKind::Catch);
}

#[test]
fn clr_clauses_keep_their_kind_and_only_typed_catches_get_a_type_entry() {
    let mut cfg = LiftedCfg::with_edge_payload();
    let blocks = clause_blocks(&mut cfg);
    let mut types = HandlerMetadata::new();

    let region = register_every_clause_kind(&mut cfg, &mut types, &blocks);

    assert_registered_clauses_keep_their_kind(&cfg, region, &blocks);
    assert_only_typed_catches_get_a_type_entry(&cfg, &types, region, blocks.catch_net);

    // Registering only records metadata: no block or edge appears.
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
    let mut types = HandlerMetadata::new();

    let outer = cfg
        .add_region(Region {
            protected_blocks: set(&[outer_body, inner_body]),
            handlers: vec![clause(outer_catch, HandlerKind::Catch)],
            parent: None,
        })
        .unwrap();
    types.set_all(outer, [(0, "T".to_string())]);
    let inner = cfg
        .add_region(Region {
            protected_blocks: set(&[inner_body]),
            handlers: vec![clause(inner_catch, HandlerKind::Catch)],
            parent: Some(outer),
        })
        .unwrap();
    types.set_all(inner, [(0, "T".to_string())]);

    assert_eq!((outer.index(), inner.index()), (0, 1));
    assert_eq!(cfg.region_count(), 2);
    assert_eq!(cfg.region(inner).parent, Some(outer));
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
    let index = RegionIndex::compute(&cfg);
    assert_eq!(index.protecting(outer_body), Some(outer));
    assert_eq!(index.protecting(inner_body), Some(inner));
    assert_eq!(index.protecting(outer_catch), None);
}

/// A CLR method with every clause kind, each handler reached by one unwind
/// edge from the protected block, as a lifter wires them.
fn clr_method_with_all_clauses() -> (
    LiftedCfg,
    HandlerMetadata<&'static str>,
    RegionId,
    [BlockId; 6],
) {
    let mut cfg = LiftedCfg::with_edge_payload();
    let protected = cfg.entry();
    let handlers = [(); 6].map(|()| cfg.new_block());
    let [catch_io, catch_all, finally, fault, filtered, catch_net] = handlers;
    let filter_predicate = cfg.new_block();
    let mut types = HandlerMetadata::new();
    let region = cfg
        .add_region(Region {
            protected_blocks: set(&[protected]),
            handlers: vec![
                clause(catch_io, HandlerKind::Catch),
                clause(catch_all, HandlerKind::CatchAll),
                clause(finally, HandlerKind::Finally),
                clause(fault, HandlerKind::Fault),
                clause(
                    filtered,
                    HandlerKind::Filter {
                        filter_block: filter_predicate,
                    },
                ),
                clause(catch_net, HandlerKind::Catch),
            ],
            parent: None,
        })
        .unwrap();
    types.set_all(region, [(0, "IOException"), (5, "WebException")]);
    for handler in handlers {
        cfg.add_edge_with_payload(
            protected,
            handler,
            EdgeKind::ExceptionUnwind,
            ExceptionFlow {
                phase: Some(ExceptionPhase::Unwind),
                disposition: None,
                metadata: "clause",
            },
        );
    }
    (cfg, types, region, handlers)
}

#[test]
fn the_exception_model_classifies_registered_clauses_by_kind_and_routes_back_to_them() {
    let (cfg, types, region, handlers) = clr_method_with_all_clauses();
    let [catch_io, catch_all, finally, fault, filtered, catch_net] = handlers;
    let model = EhModel::compute(&cfg);

    assert_eq!(
        model.blocks_of(EhBlockKind::LandingPad).collect::<Vec<_>>(),
        [catch_io, catch_all, catch_net]
    );
    assert_eq!(
        model.blocks_of(EhBlockKind::Cleanup).collect::<Vec<_>>(),
        [finally, fault]
    );
    assert_eq!(model.block_kind(filtered), EhBlockKind::CatchSwitch);
    assert_eq!(model.block_kind(cfg.entry()), EhBlockKind::Normal);
    assert_eq!(model.blocks_of(EhBlockKind::Resume).count(), 0);

    assert_eq!(model.eh_edges().len(), 6);
    for &edge in model.eh_edges() {
        let edge = cfg.edge(edge);
        assert_eq!(edge.kind(), EdgeKind::ExceptionUnwind);
        assert_eq!(edge.source(), cfg.entry());
        // The model keeps the identity of the lifter's own edge, so the
        // phase and metadata the lifter attached stay reachable.
        let payload = edge.payload();
        assert_eq!(payload.phase, Some(ExceptionPhase::Unwind));
        assert_eq!(payload.metadata, "clause");
    }
    for (index, handler) in handlers.into_iter().enumerate() {
        assert_eq!(model.protected_by(handler), &set(&[cfg.entry()]));
        assert_eq!(
            cfg.handlers_at(handler)
                .map(|(route, _)| route)
                .collect::<Vec<_>>(),
            [HandlerRef::new(region, index)]
        );
    }
    // From a handler entry back to the clause's type token.
    let (route, handler) = cfg
        .handlers_at(catch_net)
        .next()
        .expect("the entry starts a handler");
    assert_eq!(types.metadata(route), Some(&"WebException"));
    assert_eq!(handler.kind, HandlerKind::Catch);
}

#[test]
fn a_clr_finally_records_where_each_completion_reason_resumes() {
    let mut cfg = Cfg::<&str>::new();
    let body = cfg.entry();
    let cleanup = cfg.new_block();
    let after = cfg.new_block();
    let exit = cfg.new_block();
    let region = cfg
        .add_region(Region {
            protected_blocks: set(&[body]),
            handlers: vec![clause(cleanup, HandlerKind::Finally)],
            parent: None,
        })
        .unwrap();
    let finally = HandlerRef::new(region, 0);
    cfg.set_cleanup_exit(finally, cleanup).unwrap();
    for route in [
        Continuation {
            reason: CompletionReason::Normal,
            target: after,
        },
        Continuation {
            reason: CompletionReason::Return,
            target: exit,
        },
        // Recording a route twice keeps one.
        Continuation {
            reason: CompletionReason::Normal,
            target: after,
        },
    ] {
        cfg.add_continuation(finally, route).unwrap();
    }

    let recorded = cfg
        .cleanup(finally)
        .expect("the finally has a cleanup record");
    assert_eq!(recorded.exit_block, Some(cleanup));
    assert_eq!(recorded.continuations.len(), 2);
    assert_eq!(
        recorded
            .targets_for(CompletionReason::Normal)
            .collect::<Vec<_>>(),
        [after]
    );
    assert_eq!(
        recorded
            .targets_for(CompletionReason::Return)
            .collect::<Vec<_>>(),
        [exit]
    );
    assert_eq!(recorded.targets_for(CompletionReason::Unwind).count(), 0);
    assert_eq!(
        cfg.cleanups()
            .iter()
            .map(|(handler, _)| handler)
            .collect::<Vec<_>>(),
        [finally]
    );
    let model = EhModel::compute(&cfg);
    assert_eq!(model.block_kind(cleanup), EhBlockKind::Cleanup);
}
