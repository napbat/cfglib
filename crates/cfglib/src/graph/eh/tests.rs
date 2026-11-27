extern crate alloc;

use alloc::vec::Vec;

use super::*;
use crate::cfg::Cfg;
use crate::edge::{Edge, EdgeKind};
use crate::region::{
    CompletionReason, Continuation, Handler, HandlerBody, HandlerKind, HandlerRef, Region,
};
use crate::test_util::ff;

fn region(protected: BlockId, entry: BlockId, kind: HandlerKind) -> Region {
    Region {
        protected_blocks: [protected].into_iter().collect(),
        handlers: alloc::vec![Handler {
            entry,
            body: HandlerBody::known([entry]),
            kind,
        }],
        parent: None,
    }
}

#[test]
fn handler_kinds_map_to_precise_block_roles() {
    assert_eq!(
        EhBlockKind::from(HandlerKind::Catch),
        EhBlockKind::LandingPad
    );
    assert_eq!(
        EhBlockKind::from(HandlerKind::CatchAll),
        EhBlockKind::LandingPad
    );
    assert_eq!(
        EhBlockKind::from(HandlerKind::Finally),
        EhBlockKind::Cleanup
    );
    assert_eq!(EhBlockKind::from(HandlerKind::Fault), EhBlockKind::Cleanup);
    assert_eq!(
        EhBlockKind::from(HandlerKind::Filter {
            filter_block: BlockId::from_raw(7),
        }),
        EhBlockKind::CatchSwitch
    );
}

#[test]
fn no_eh_all_normal() {
    let mut cfg = Cfg::new();
    let b = cfg.new_block();
    cfg.block_mut(cfg.entry()).instructions_mut().push(ff("a"));
    cfg.block_mut(b).instructions_mut().push(ff("b"));
    cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
    let model = EhModel::compute(&cfg);
    assert_eq!(model.eh_edges(), []);
    assert!(
        model
            .block_kinds()
            .all(|(_, kind)| kind == EhBlockKind::Normal)
    );
}

#[test]
fn exception_edge_creates_landing_pad() {
    let mut cfg = Cfg::new();
    let handler = cfg.new_block();
    cfg.block_mut(cfg.entry())
        .instructions_mut()
        .push(ff("call"));
    cfg.block_mut(handler).instructions_mut().push(ff("catch"));
    cfg.add_edge(cfg.entry(), handler, EdgeKind::ExceptionHandler);
    let model = EhModel::compute(&cfg);
    assert_eq!(model.eh_edges().len(), 1);
    assert_eq!(model.block_kind(handler), EhBlockKind::LandingPad);
    assert_eq!(
        model.blocks_of(EhBlockKind::LandingPad).collect::<Vec<_>>(),
        [handler]
    );
    assert!(model.protected_by(handler).contains(&cfg.entry()));
}

#[test]
fn edge_order_does_not_decide_the_role_of_a_dispatch_target_that_also_resumes() {
    // A filter-shaped block: the target of a handler edge and the source
    // of a resume edge.
    for dispatch_first in [true, false] {
        let mut cfg = Cfg::<()>::new();
        let filter = cfg.new_block();
        let next = cfg.new_block();
        if dispatch_first {
            cfg.add_edge(cfg.entry(), filter, EdgeKind::ExceptionHandler);
            cfg.add_edge(filter, next, EdgeKind::ExceptionResume);
        } else {
            cfg.add_edge(filter, next, EdgeKind::ExceptionResume);
            cfg.add_edge(cfg.entry(), filter, EdgeKind::ExceptionHandler);
        }

        let model = EhModel::compute(&cfg);
        assert_eq!(
            model.block_kind(filter),
            EhBlockKind::LandingPad,
            "dispatch edge first: {dispatch_first}"
        );
        assert_eq!(
            model.blocks_of(EhBlockKind::LandingPad).collect::<Vec<_>>(),
            [filter]
        );
        assert_eq!(model.blocks_of(EhBlockKind::Resume).next(), None);
    }
}

#[test]
fn unknown_handler_body_still_has_region_and_edge_identity() {
    let mut cfg = Cfg::<()>::new();
    let handler = cfg.new_block();
    let entry = cfg.entry();
    let edge = cfg.add_edge(entry, handler, EdgeKind::ExceptionHandler);
    let mut catch = region(entry, handler, HandlerKind::Catch);
    catch.handlers[0].body = HandlerBody::unknown();
    let region = cfg.add_region(catch).unwrap();

    let model = EhModel::compute(&cfg);
    assert_eq!(model.eh_edges(), [edge]);
    assert_eq!(model.block_kind(handler), EhBlockKind::LandingPad);
    assert_eq!(
        cfg.handlers_at(handler)
            .map(|(handler, _)| handler)
            .collect::<Vec<_>>(),
        [HandlerRef::new(region, 0)]
    );
    assert!(model.protected_by(handler).contains(&entry));
}

#[test]
fn a_pad_entered_by_several_handlers_keeps_every_cleanup_record() {
    // Two regions share one cleanup pad, and each handler records a route of
    // its own. The CFG answers per handler, so neither record hides the other.
    let mut cfg = Cfg::<()>::new();
    let inner_body = cfg.new_block();
    let pad = cfg.new_block();
    let after_inner = cfg.new_block();
    let after_outer = cfg.new_block();
    let outer = cfg
        .add_region(region(cfg.entry(), pad, HandlerKind::Finally))
        .unwrap();
    let inner = cfg
        .add_region(region(inner_body, pad, HandlerKind::Finally))
        .unwrap();
    for (handler, target) in [(outer, after_outer), (inner, after_inner)] {
        let handler = HandlerRef::new(handler, 0);
        cfg.set_cleanup_exit(handler, pad).unwrap();
        cfg.add_continuation(
            handler,
            Continuation {
                reason: CompletionReason::Normal,
                target,
            },
        )
        .unwrap();
    }

    let targets: Vec<Vec<BlockId>> = cfg
        .handlers_at(pad)
        .map(|(handler, _)| {
            cfg.cleanup(handler)
                .expect("each handler recorded its route")
                .targets_for(CompletionReason::Normal)
                .collect()
        })
        .collect();
    assert_eq!(targets, [[after_outer], [after_inner]]);
    assert_eq!(EhModel::compute(&cfg).block_kind(pad), EhBlockKind::Cleanup);
}

#[test]
fn payload_cfg_retains_every_exception_transfer_and_edge_identity() {
    use crate::exception::{ExceptionDisposition, ExceptionFlow, ExceptionPhase};

    let mut cfg = Cfg::<(), ExceptionFlow<u32>>::with_edge_payload();
    let handler = cfg.new_block();
    let leave = cfg.new_block();
    let rethrow = cfg.new_block();
    let outer = cfg.new_block();
    let continue_decision = cfg.new_block();
    let resume_target = cfg.new_block();
    let entry = cfg.entry();
    let decision = |phase, disposition, metadata| ExceptionFlow {
        phase: Some(phase),
        disposition: Some(disposition),
        metadata,
    };

    let handler_edge = cfg.add_edge(
        entry,
        handler,
        Edge::new(EdgeKind::ExceptionHandler).with_payload(decision(
            ExceptionPhase::Unwind,
            ExceptionDisposition::ExecuteHandler,
            11,
        )),
    );
    cfg.add_edge(
        entry,
        leave,
        Edge::new(EdgeKind::ExceptionLeave).with_payload(ExceptionFlow::new(12)),
    );
    cfg.add_edge(
        rethrow,
        outer,
        Edge::new(EdgeKind::ExceptionResume).with_payload(decision(
            ExceptionPhase::Search,
            ExceptionDisposition::ContinueSearch,
            13,
        )),
    );
    cfg.add_edge(
        continue_decision,
        resume_target,
        Edge::new(EdgeKind::ExceptionContinue).with_payload(decision(
            ExceptionPhase::Search,
            ExceptionDisposition::ContinueExecution,
            14,
        )),
    );

    let model = EhModel::compute(&cfg);
    assert_eq!(
        model
            .eh_edges()
            .iter()
            .map(|&edge| cfg.edge(edge).kind())
            .collect::<Vec<_>>(),
        [
            EdgeKind::ExceptionHandler,
            EdgeKind::ExceptionLeave,
            EdgeKind::ExceptionResume,
            EdgeKind::ExceptionContinue,
        ]
    );
    assert_eq!(model.eh_edges()[0], handler_edge);
    assert_eq!(cfg[model.eh_edges()[0]].payload().metadata, 11);
    assert_eq!(
        model.blocks_of(EhBlockKind::Resume).collect::<Vec<_>>(),
        [rethrow, continue_decision]
    );
}

#[test]
fn explicit_cleanup_region_overrides_incoming_unwind_inference() {
    let mut cfg = Cfg::<()>::new();
    let cleanup = cfg.new_block();
    let entry = cfg.entry();
    cfg.add_edge(entry, cleanup, EdgeKind::ExceptionUnwind);
    cfg.add_region(region(entry, cleanup, HandlerKind::Finally))
        .unwrap();

    let model = EhModel::compute(&cfg);
    assert_eq!(model.block_kind(cleanup), EhBlockKind::Cleanup);
    assert_eq!(
        model.blocks_of(EhBlockKind::Cleanup).collect::<Vec<_>>(),
        [cleanup]
    );
}
