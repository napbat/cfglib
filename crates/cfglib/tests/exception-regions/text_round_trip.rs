//! The text form of a region-bearing graph, and `verify` over region shapes.

use cfglib::{
    BlockId, Cfg, EdgeKind, HandlerBody, HandlerKind, HandlerRef, Region, RegionId,
    SemanticValidator, parse_cfg_text, verify, verify_with,
};

use crate::golden::assert_golden;
use crate::shapes::{
    Stmt, always_throwing_try, block_of, catch, dead_and_live_handlers, handler, protect,
    try_catch, unknown_catch,
};

/// Every region feature the text form names, with a hole among the blocks:
/// the returned block was removed.
fn kitchen_sink() -> (Cfg<Stmt>, BlockId) {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(Stmt::text("setup()"));
    let gone = block_of(&mut cfg, &["removed before printing"]);
    let body = block_of(&mut cfg, &["risky()"]);
    let inner = block_of(&mut cfg, &["inner()"]);
    let catch_pad = block_of(&mut cfg, &["catch (e)"]);
    let catch_tail = block_of(&mut cfg, &["log(e)"]);
    let filter_pad = block_of(&mut cfg, &["filtered"]);
    let funclet = block_of(&mut cfg, &["is_recoverable(e)"]);
    let cleanup = block_of(&mut cfg, &["close()"]);
    let after = block_of(&mut cfg, &["return"]);
    cfg.add_edge(entry, gone, EdgeKind::Fallthrough);
    cfg.add_edge(entry, body, EdgeKind::Fallthrough);
    cfg.add_edge(body, inner, EdgeKind::Fallthrough);
    cfg.add_edge(inner, after, EdgeKind::Fallthrough);
    cfg.add_edge(body, catch_pad, EdgeKind::ExceptionHandler);
    cfg.add_edge(inner, filter_pad, EdgeKind::ExceptionUnwind);
    cfg.add_edge(inner, cleanup, EdgeKind::ExceptionUnwind);
    cfg.add_edge(catch_pad, catch_tail, EdgeKind::Fallthrough);
    cfg.add_edge(catch_tail, after, EdgeKind::Fallthrough);
    cfg.add_edge(filter_pad, after, EdgeKind::Fallthrough);
    cfg.add_edge(cleanup, after, EdgeKind::Fallthrough);
    cfg.remove_block(gone);
    let outer = protect(
        &mut cfg,
        &[body, inner],
        vec![
            handler(catch_pad, HandlerKind::Catch, &[catch_pad, catch_tail]),
            filtered_handler(filter_pad, funclet),
        ],
    );
    let nested = protect(
        &mut cfg,
        &[inner],
        vec![handler(cleanup, HandlerKind::Finally, &[cleanup])],
    );
    cfg.region_mut(nested).expect("added").parent = Some(outer);
    (cfg, gone)
}

/// A filtered catch whose extent is not known.
fn filtered_handler(entry: BlockId, filter_block: BlockId) -> cfglib::Handler {
    cfglib::Handler {
        entry,
        body: HandlerBody::unknown(),
        kind: HandlerKind::Filter { filter_block },
    }
}

fn parse(text: &str) -> Cfg<Stmt> {
    parse_cfg_text(text, |line| Ok::<_, String>(Stmt::text(line))).expect("well-formed text")
}

#[test]
fn a_region_bearing_graph_reads_back_equal_and_prints_the_same_text() {
    let (cfg, _) = kitchen_sink();

    let text = cfg.to_text();
    let parsed = parse(&text);

    assert_eq!(parsed.regions(), cfg.regions());
    assert_eq!(parsed.entry(), cfg.entry());
    for block in cfg.block_ids() {
        assert_eq!(parsed.block(block), cfg.block(block), "{block}");
    }
    let edges = |cfg: &Cfg<Stmt>| -> Vec<_> {
        cfg.edges()
            .map(|edge| (edge.source(), edge.target(), edge.kind()))
            .collect()
    };
    assert_eq!(edges(&parsed), edges(&cfg));
    assert_eq!(
        parsed.to_text(),
        text,
        "write, read, write is a fixed point"
    );
}

#[test]
fn block_holes_survive_the_text_form() {
    let (cfg, gone) = kitchen_sink();
    assert!(!cfg.contains_block(gone));

    let parsed = parse(&cfg.to_text());

    assert!(
        !parsed.contains_block(gone),
        "the hole is reproduced, not filled"
    );
    assert_eq!(parsed.block_bound(), cfg.block_bound());
    assert_eq!(parsed.block_count(), cfg.block_count());
    assert_eq!(
        parsed.block_ids().collect::<Vec<_>>(),
        cfg.block_ids().collect::<Vec<_>>()
    );
}

#[test]
fn unknown_extents_filters_and_nesting_are_named_in_the_text() {
    let text = kitchen_sink().0.to_text();

    let regions: Vec<&str> = text
        .lines()
        .filter(|line| line.starts_with("region ") || line.trim_start().starts_with("handler "))
        .collect();

    assert_eq!(regions.len(), 5, "two regions and three handlers: {text}");
    assert!(
        text.contains("body=?"),
        "an unknown extent is not an empty one: {text}"
    );
    assert!(text.contains("filter=bb"), "{text}");
    assert!(text.contains("parent=r0"), "{text}");
}

#[test]
fn a_try_catch_prints_as_the_documented_text_form() {
    let mut fixture = try_catch(&["log(e)"]);
    let end = block_of(&mut fixture.cfg, &["cleanup()"]);
    fixture
        .cfg
        .add_edge(fixture.pad, end, EdgeKind::Fallthrough);
    fixture
        .cfg
        .region_mut(fixture.region)
        .expect("exists")
        .handlers
        .push(handler(end, HandlerKind::Finally, &[end]));

    assert_golden(
        "exception-regions/try_catch_finally.txt",
        &fixture.cfg.to_text(),
    );
}

#[test]
fn cleanup_records_are_not_part_of_the_text_projection() {
    // The text form is a projection; `serde` is the fidelity path. A frontend
    // that needs the routes keeps them.
    let mut fixture = try_catch(&["log(e)"]);
    let handler = HandlerRef::new(fixture.region, 0);
    fixture.cfg.add_continuation(
        handler,
        cfglib::Continuation {
            reason: cfglib::CompletionReason::Normal,
            resume: fixture.join,
        },
    );

    let parsed = parse(&fixture.cfg.to_text());

    assert_eq!(fixture.cfg.cleanups().len(), 1);
    assert_eq!(parsed.cleanups().len(), 0);
    assert_eq!(parsed.regions(), fixture.cfg.regions());
}

#[test]
fn verify_accepts_the_region_shapes_a_frontend_builds() {
    let shapes: Vec<(&str, Cfg<Stmt>)> = vec![
        ("try catch", try_catch(&["log(e)"]).cfg),
        ("empty pad", try_catch(&[]).cfg),
        ("always throws", always_throwing_try().cfg),
        ("dead and live handlers", dead_and_live_handlers().cfg),
        ("kitchen sink", kitchen_sink().0),
    ];

    for (name, cfg) in shapes {
        let report = verify(&cfg);
        assert!(report.is_ok(), "{name}: {report:?}");
    }
}

/// A consumer check `verify` does not do for it: every handler entry names a
/// live block.
struct HandlerEntriesAreLive;

impl SemanticValidator<Stmt, ()> for HandlerEntriesAreLive {
    type Error = (HandlerRef, BlockId);

    fn finish(&self, cfg: &Cfg<Stmt>, errors: &mut Vec<Self::Error>) {
        for region in cfg.regions() {
            for (index, handler) in region.handlers.iter().enumerate() {
                if !cfg.contains_block(handler.entry) {
                    errors.push((HandlerRef::new(region.id, index), handler.entry));
                }
            }
        }
    }
}

#[test]
fn a_consumer_validator_reports_a_handler_entry_the_graph_no_longer_has() {
    let mut fixture = try_catch(&["log(e)"]);
    let handler = HandlerRef::new(fixture.region, 0);

    let sound = verify_with(&fixture.cfg, &HandlerEntriesAreLive);
    assert!(sound.is_ok(), "{sound:?}");

    fixture.cfg.remove_block(fixture.pad);
    let broken = verify_with(&fixture.cfg, &HandlerEntriesAreLive);
    assert_eq!(broken.semantic_errors, [(handler, fixture.pad)]);
    assert!(!broken.is_ok());
}

#[test]
fn a_region_may_protect_blocks_with_no_handler_or_handlers_with_unknown_bodies() {
    let mut cfg = Cfg::<Stmt>::new();
    let body = block_of(&mut cfg, &["risky()"]);
    let pad = block_of(&mut cfg, &["pad"]);
    cfg.add_edge(cfg.entry(), body, EdgeKind::Fallthrough);
    cfg.add_edge(body, pad, EdgeKind::ExceptionUnwind);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [body].into_iter().collect(),
        handlers: vec![unknown_catch(pad), catch(pad)],
        parent: None,
    });

    let parsed = parse(&cfg.to_text());

    assert_eq!(parsed.regions(), cfg.regions());
    assert!(verify(&cfg).is_ok());
}
