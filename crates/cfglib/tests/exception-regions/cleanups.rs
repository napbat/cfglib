//! Cleanup records: the routes a shared `finally` block continues along, told
//! apart by the reason that entered it.

use cfglib::{Cfg, CompletionReason, Continuation, EdgeKind, EhModel, HandlerKind, HandlerRef};

use crate::shapes::{Stmt, block_of, catch, handler, protect, try_catch};

/// A try whose finally block is lowered once and entered by every route out of
/// the protected body: it ends the function (`return`), leaves for `after_loop`
/// (a `break`), falls out normally, or is unwound through.
struct SharedCleanup {
    cfg: Cfg<Stmt>,
    cleanup: cfglib::BlockId,
    handler: HandlerRef,
    after: cfglib::BlockId,
    exit: cfglib::BlockId,
    after_loop: cfglib::BlockId,
}

fn shared_cleanup() -> SharedCleanup {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    let body = block_of(&mut cfg, &["work()"]);
    let cleanup = block_of(&mut cfg, &["close()"]);
    let after = block_of(&mut cfg, &["after()"]);
    let exit = block_of(&mut cfg, &["return"]);
    let after_loop = block_of(&mut cfg, &["after_loop()"]);
    cfg.add_edge(entry, body, EdgeKind::Fallthrough);
    cfg.add_edge(body, cleanup, EdgeKind::ExceptionLeave);
    cfg.add_edge(cleanup, after, EdgeKind::Fallthrough);
    cfg.add_edge(cleanup, exit, EdgeKind::Fallthrough);
    cfg.add_edge(cleanup, after_loop, EdgeKind::Fallthrough);
    let region = protect(
        &mut cfg,
        &[body],
        vec![handler(cleanup, HandlerKind::Finally, &[cleanup])],
    );
    SharedCleanup {
        cfg,
        cleanup,
        handler: HandlerRef::new(region, 0),
        after,
        exit,
        after_loop,
    }
}

#[test]
fn routes_are_recorded_by_reason_and_read_back_in_recorded_order() {
    let SharedCleanup {
        mut cfg,
        cleanup,
        handler,
        after,
        exit,
        after_loop,
    } = shared_cleanup();
    assert!(cfg.cleanups().is_empty(), "nothing is recorded until asked");
    assert!(cfg.cleanup(handler).is_none());

    cfg.set_cleanup_resume(handler, cleanup);
    cfg.add_continuation(
        handler,
        Continuation {
            reason: CompletionReason::Normal,
            resume: after,
        },
    );
    cfg.add_continuation(
        handler,
        Continuation {
            reason: CompletionReason::Return,
            resume: exit,
        },
    );
    cfg.add_continuation(
        handler,
        Continuation {
            reason: CompletionReason::Transfer,
            resume: after_loop,
        },
    );
    cfg.add_continuation(
        handler,
        Continuation {
            reason: CompletionReason::Transfer,
            resume: after,
        },
    );

    let record = cfg.cleanup(handler).expect("recorded");
    assert_eq!(record.handler, handler);
    assert_eq!(record.resume_from, Some(cleanup));
    assert_eq!(record.continuations.len(), 4);
    assert_eq!(
        record
            .resumes_for(CompletionReason::Normal)
            .collect::<Vec<_>>(),
        [after]
    );
    assert_eq!(
        record
            .resumes_for(CompletionReason::Return)
            .collect::<Vec<_>>(),
        [exit]
    );
    assert_eq!(
        record
            .resumes_for(CompletionReason::Transfer)
            .collect::<Vec<_>>(),
        [after_loop, after],
        "several transfers keep their recorded order"
    );
    assert_eq!(
        record
            .resumes_for(CompletionReason::Unwind)
            .collect::<Vec<_>>(),
        []
    );
    assert_eq!(cfg.cleanups().len(), 1);
}

#[test]
fn recording_the_same_route_twice_keeps_one_route() {
    let SharedCleanup {
        mut cfg,
        handler,
        after,
        ..
    } = shared_cleanup();
    let route = Continuation {
        reason: CompletionReason::Normal,
        resume: after,
    };

    cfg.add_continuation(handler, route);
    cfg.add_continuation(handler, route);
    cfg.add_continuation(
        handler,
        Continuation {
            reason: CompletionReason::Return,
            resume: after,
        },
    );

    let record = cfg.cleanup(handler).expect("recorded");
    assert_eq!(
        record.continuations.len(),
        2,
        "the same destination under another reason is a distinct route"
    );
    assert_eq!(record.continuations[0], route);
}

#[test]
fn a_diverging_cleanup_records_no_resume_block() {
    // `finally { return 1 }`: the cleanup never reaches a block it continues
    // from, so leaving the resume block unset is the honest description.
    let SharedCleanup {
        mut cfg,
        handler,
        exit,
        ..
    } = shared_cleanup();

    cfg.add_continuation(
        handler,
        Continuation {
            reason: CompletionReason::Return,
            resume: exit,
        },
    );

    assert_eq!(cfg.cleanup(handler).expect("recorded").resume_from, None);
}

#[test]
fn the_eh_model_carries_each_record_under_its_handler_entry() {
    let SharedCleanup {
        mut cfg,
        cleanup,
        handler,
        after,
        exit,
        ..
    } = shared_cleanup();
    cfg.set_cleanup_resume(handler, cleanup);
    cfg.add_continuation(
        handler,
        Continuation {
            reason: CompletionReason::Normal,
            resume: after,
        },
    );
    cfg.add_continuation(
        handler,
        Continuation {
            reason: CompletionReason::Return,
            resume: exit,
        },
    );

    let model = EhModel::compute(&cfg);

    let record = model.cleanup(cleanup).expect("keyed by the entry block");
    assert_eq!(Some(record), cfg.cleanup(handler));
    assert_eq!(
        record
            .resumes_for(CompletionReason::Return)
            .collect::<Vec<_>>(),
        [exit]
    );
    assert!(model.cleanup(after).is_none());
    let all: Vec<_> = model.cleanups().collect();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].0, cleanup);
}

#[test]
fn a_handler_without_a_record_has_no_cleanup_in_the_model() {
    let fixture = try_catch(&["log(e)"]);

    let model = EhModel::compute(&fixture.cfg);

    assert!(model.cleanup(fixture.pad).is_none());
    assert_eq!(model.cleanups().count(), 0);
    assert!(
        fixture
            .cfg
            .cleanup(HandlerRef::new(fixture.region, 0))
            .is_none()
    );
}

#[test]
fn records_of_different_handlers_are_kept_apart_in_first_recorded_order() {
    let mut cfg = Cfg::<Stmt>::new();
    let body = block_of(&mut cfg, &["risky()"]);
    let catch_pad = block_of(&mut cfg, &["catch"]);
    let cleanup = block_of(&mut cfg, &["cleanup"]);
    let after = block_of(&mut cfg, &["after"]);
    let region = protect(
        &mut cfg,
        &[body],
        vec![
            catch(catch_pad),
            handler(cleanup, HandlerKind::Finally, &[cleanup]),
        ],
    );
    let finally = HandlerRef::new(region, 1);
    let catch_ref = HandlerRef::new(region, 0);

    cfg.add_continuation(
        finally,
        Continuation {
            reason: CompletionReason::Normal,
            resume: after,
        },
    );
    cfg.add_continuation(
        catch_ref,
        Continuation {
            reason: CompletionReason::Unwind,
            resume: after,
        },
    );

    let order: Vec<HandlerRef> = cfg.cleanups().iter().map(|record| record.handler).collect();
    assert_eq!(order, [finally, catch_ref], "in the order first recorded");
    assert_eq!(
        cfg.cleanup(catch_ref)
            .expect("recorded")
            .resumes_for(CompletionReason::Normal)
            .count(),
        0
    );
    let model = EhModel::compute(&cfg);
    assert!(model.cleanup(catch_pad).is_some());
    assert!(model.cleanup(cleanup).is_some());
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "handler does not exist")]
fn recording_a_route_for_a_handler_the_graph_lacks_panics() {
    let mut cfg = Cfg::<Stmt>::new();
    let after = block_of(&mut cfg, &["after"]);

    cfg.add_continuation(
        HandlerRef::new(cfglib::RegionId::from_raw(3), 0),
        Continuation {
            reason: CompletionReason::Normal,
            resume: after,
        },
    );
}
