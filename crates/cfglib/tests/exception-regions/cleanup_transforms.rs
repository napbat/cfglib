//! The cleanup and structural transforms on a graph that carries regions.
//!
//! The contract any sane pass owes such a graph: afterwards every block a
//! region, handler, or cleanup names is a live block, `compact` does not panic,
//! a handler is still entered through an unwind edge from the protected code,
//! it still does what it did, and the set of statements that can unwind into it
//! is exactly what it was, neither widened nor narrowed.

use cfglib::{
    Cfg, EhBlockKind, EhEdgeKind, EhModel, HandlerBody, HandlerKind, HandlerRef, contract_edge,
    dead_code_elimination, merge_blocks, remove_dead_code, remove_empty_blocks, remove_unreachable,
    simplify, split_critical_edges, verify,
};

use crate::shapes::{
    Stmt, Try, always_throwing_try, dead_and_live_handlers, throwing_entry_try, trace_from,
    try_catch, try_catch_with,
};

/// Every block a region, handler, or cleanup of `cfg` names is live.
fn assert_references_live(cfg: &Cfg<Stmt>, when: &str) {
    for region in cfg.regions() {
        for &block in &region.protected_blocks {
            assert!(
                cfg.contains_block(block),
                "{when}: {} protects the removed block {block}",
                region.id
            );
        }
        if let Some(parent) = region.parent {
            assert!(
                cfg.region(parent).is_some(),
                "{when}: parent of {}",
                region.id
            );
        }
        for (index, handler) in region.handlers.iter().enumerate() {
            let which = HandlerRef::new(region.id, index);
            assert!(
                cfg.contains_block(handler.entry),
                "{when}: {which} enters the removed block {}",
                handler.entry
            );
            if let HandlerBody::Known(body) = &handler.body {
                for &block in body {
                    assert!(
                        cfg.contains_block(block),
                        "{when}: the body of {which} holds the removed block {block}"
                    );
                }
            }
            if let HandlerKind::Filter { filter_block } = handler.kind {
                assert!(
                    cfg.contains_block(filter_block),
                    "{when}: the filter of {which} is the removed block {filter_block}"
                );
            }
        }
    }
    for cleanup in cfg.cleanups() {
        let named = cleanup
            .resume_from
            .into_iter()
            .chain(cleanup.continuations.iter().map(|route| route.resume));
        for block in named {
            assert!(
                cfg.contains_block(block),
                "{when}: the cleanup of {} names the removed block {block}",
                cleanup.handler
            );
        }
    }
}

/// The statements inside the blocks a region protects, sorted.
fn protected_statements(cfg: &Cfg<Stmt>, handler: HandlerRef) -> Vec<String> {
    let region = cfg.region(handler.region()).expect("the region exists");
    let mut statements: Vec<String> = region
        .protected_blocks
        .iter()
        .flat_map(|&block| cfg.block(block).instructions())
        .map(|stmt| stmt.as_str().to_owned())
        .collect();
    statements.sort();
    statements
}

/// What must hold after a pass: references live, compaction safe, and the
/// handler `locate` finds still a handler of the same code over the same
/// statements.
fn assert_try_survives(
    cfg: &mut Cfg<Stmt>,
    locate: impl Fn(&Cfg<Stmt>) -> HandlerRef,
    protected: &[&str],
    handler_trace: &[&str],
) {
    assert_references_live(cfg, "right after the pass");

    cfg.compact();

    assert!(cfg.is_compact());
    assert_references_live(cfg, "after compaction");
    assert!(verify(cfg).is_ok());
    let handler = locate(cfg);
    let entry = cfg.handler(handler).expect("the handler survives").entry;
    let region = cfg.region(handler.region()).expect("the region survives");
    let model = EhModel::compute(cfg);
    assert_eq!(model.block_kind(entry), EhBlockKind::LandingPad);
    assert!(model.handlers(entry).contains(&handler));
    assert!(
        model.eh_edges().iter().any(|edge| {
            edge.to == entry
                && edge.kind == EhEdgeKind::Unwind
                && region.protected_blocks.contains(&edge.from)
        }),
        "no unwind edge from the protected code enters the handler: {:?}",
        model.eh_edges()
    );
    assert_eq!(
        trace_from(cfg, entry),
        handler_trace,
        "what the handler does"
    );
    assert_eq!(
        protected_statements(cfg, handler),
        protected,
        "which statements can unwind into it"
    );
}

fn first_handler(region: cfglib::RegionId) -> impl Fn(&Cfg<Stmt>) -> HandlerRef {
    move |_| HandlerRef::new(region, 0)
}

#[test]
fn remove_empty_blocks_does_not_bypass_an_empty_catch_pad() {
    let Try {
        mut cfg, region, ..
    } = try_catch(&[]);

    remove_empty_blocks(&mut cfg);

    assert_try_survives(&mut cfg, first_handler(region), &["risky()"], &["return"]);
}

#[test]
fn simplify_keeps_an_empty_catch_pad_and_the_protection_boundary() {
    let Try {
        mut cfg, region, ..
    } = try_catch(&[]);

    simplify(&mut cfg);

    assert_try_survives(&mut cfg, first_handler(region), &["risky()"], &["return"]);
}

#[test]
fn simplify_does_not_fold_the_handler_of_a_body_that_always_throws() {
    let Try {
        mut cfg, region, ..
    } = always_throwing_try();

    simplify(&mut cfg);

    assert_try_survives(
        &mut cfg,
        first_handler(region),
        &["throw e"],
        &["log(e)", "return"],
    );
}

#[test]
fn merge_blocks_does_not_merge_a_handler_into_the_block_that_raises_into_it() {
    // The throwing block is the entry, so its only candidate merge is the one
    // that folds the pad into it: a sole successor over a sole predecessor.
    let Try {
        mut cfg, region, ..
    } = throwing_entry_try();

    merge_blocks(&mut cfg);

    assert_try_survives(
        &mut cfg,
        first_handler(region),
        &["throw e"],
        &["log(e)", "return"],
    );
}

#[test]
fn merge_blocks_does_not_widen_protection_over_the_unprotected_predecessor() {
    // entry -> body is a sole-successor, sole-predecessor pair; merging them
    // would put `setup()` under the handler.
    let Try {
        mut cfg, region, ..
    } = try_catch(&["log(e)"]);

    merge_blocks(&mut cfg);

    assert_try_survives(
        &mut cfg,
        first_handler(region),
        &["risky()"],
        &["log(e)", "return"],
    );
}

#[test]
fn contract_edge_refuses_to_fold_a_handler_entry_into_its_raising_block() {
    let Try {
        mut cfg,
        body,
        pad,
        region,
        ..
    } = always_throwing_try();

    // Whether or not the contraction is refused, the graph must stay sound.
    contract_edge(&mut cfg, body, pad);

    assert_try_survives(
        &mut cfg,
        first_handler(region),
        &["throw e"],
        &["log(e)", "return"],
    );
}

#[test]
fn remove_unreachable_keeps_the_handler_that_can_be_entered_when_its_sibling_cannot() {
    let fixture = dead_and_live_handlers();
    let mut cfg = fixture.cfg;
    let region = fixture.region;

    remove_unreachable(&mut cfg);

    // The dead handler may be dropped from the region or kept with a live
    // entry; the live one is found by the code it runs.
    let live = move |cfg: &Cfg<Stmt>| {
        let regions = cfg.region(region).expect("the region survives");
        let index = regions
            .handlers
            .iter()
            .position(|handler| {
                cfg.block(handler.entry)
                    .instructions()
                    .first()
                    .is_some_and(|stmt| stmt.as_str() == "catch live")
            })
            .expect("the reachable handler survives");
        HandlerRef::new(region, index)
    };
    assert_try_survives(&mut cfg, live, &["risky()"], &["catch live", "return"]);
}

#[test]
fn remove_dead_code_keeps_a_catch_pad_whose_only_statements_were_dead() {
    // The pad computes a value nothing reads: it is dead, so the pass empties
    // the pad and then simplifies. The handler must outlive its body.
    let Try {
        mut cfg, region, ..
    } = try_catch_with(vec![Stmt::text("unused = 1").defining("unused")]);

    let removed = remove_dead_code(&mut cfg);

    assert!(removed > 0, "the dead statement goes");
    assert_try_survives(&mut cfg, first_handler(region), &["risky()"], &["return"]);
}

#[test]
fn dead_code_elimination_alone_leaves_the_emptied_pad_in_place() {
    let Try {
        mut cfg,
        pad,
        region,
        ..
    } = try_catch_with(vec![Stmt::text("unused = 1").defining("unused")]);

    let removed = dead_code_elimination(&mut cfg);

    assert_eq!(removed, 1);
    assert_eq!(cfg.block(pad).instructions(), []);
    assert_try_survives(&mut cfg, first_handler(region), &["risky()"], &["return"]);
}

#[test]
fn split_critical_edges_leaves_regions_and_unwind_edges_alone() {
    // body -> join is critical: the body has two successors, the join two
    // predecessors.
    let Try {
        mut cfg, region, ..
    } = try_catch(&["log(e)"]);

    let split = split_critical_edges(&mut cfg);

    assert_eq!(split, 1);
    assert_try_survives(
        &mut cfg,
        first_handler(region),
        &["risky()"],
        &["log(e)", "return"],
    );
}

#[test]
fn splitting_a_protected_block_keeps_both_halves_protected() {
    // Protection is block-granular: every instruction of a protected block may
    // unwind. After a split, both blocks hold instructions that did.
    let Try {
        mut cfg,
        body,
        region,
        ..
    } = try_catch(&["log(e)"]);
    cfg.block_mut(body)
        .instructions_mut()
        .insert(0, Stmt::text("prepare()"));

    let tail = cfg.split_block(body, 1);

    for half in [body, tail] {
        assert_eq!(
            cfg.protecting_region(half).map(|found| found.id),
            Some(region),
            "{half} holds instructions that used to be protected"
        );
    }
    assert_try_survives(
        &mut cfg,
        first_handler(region),
        &["prepare()", "risky()"],
        &["log(e)", "return"],
    );
}
