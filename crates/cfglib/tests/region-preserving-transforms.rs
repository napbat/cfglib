//! Integration tests: structural passes keep exception metadata consistent.
//!
//! A pass that removes, merges, or splits blocks must never leave a handler
//! entry, filter block, cleanup resume block, or protected extent describing
//! a block that no longer matches the code, so `compact` and the exception
//! model stay usable after any sequence of passes.

use cfglib::{
    BlockId, Cfg, CompletionReason, Continuation, EdgeKind, EhBlockKind, EhModel, Handler,
    HandlerBody, HandlerKind, HandlerRef, Region, RegionId, contract_edge, merge_blocks,
    remove_empty_blocks, remove_unreachable, simplify,
};

fn region(protected: impl IntoIterator<Item = BlockId>, handlers: Vec<Handler>) -> Region {
    Region {
        id: RegionId::from_raw(0),
        protected_blocks: protected.into_iter().collect(),
        handlers,
        parent: None,
    }
}

fn catch(entry: BlockId) -> Handler {
    Handler {
        entry,
        body: HandlerBody::known([entry]),
        kind: HandlerKind::Catch,
    }
}

/// `try { body } catch { }` where the catch body is an empty block that falls
/// through to the join, so a naive bypass of empty blocks removes the entry.
/// Returns the CFG and the handler entry.
fn empty_catch() -> (Cfg<u32>, BlockId) {
    let mut cfg = Cfg::<u32>::new();
    let body = cfg.entry();
    let handler = cfg.new_block();
    let join = cfg.new_block();
    cfg.block_mut(body).push(1);
    cfg.block_mut(join).push(2);
    cfg.add_edge(body, join, EdgeKind::Fallthrough);
    cfg.add_edge(body, handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(handler, join, EdgeKind::Fallthrough);
    cfg.add_region(region([body], vec![catch(handler)]));
    (cfg, handler)
}

fn assert_metadata_survives_compaction(cfg: &mut Cfg<u32>) {
    cfg.compact();
    assert!(cfg.is_compact());
    for region in cfg.regions() {
        for handler in &region.handlers {
            assert!(cfg.contains_block(handler.entry));
        }
    }
}

#[test]
fn bypassing_empty_blocks_keeps_a_handler_entry_alive() {
    let (mut cfg, handler) = empty_catch();

    assert_eq!(remove_empty_blocks(&mut cfg), 0);
    assert!(cfg.contains_block(handler));
    assert_metadata_survives_compaction(&mut cfg);
}

#[test]
fn simplify_then_compact_keeps_the_empty_catch() {
    let (mut cfg, _) = empty_catch();

    simplify(&mut cfg);
    assert_metadata_survives_compaction(&mut cfg);
    let handler = cfg.regions()[0].handlers[0].entry;
    assert_eq!(
        EhModel::compute(&cfg).block_kind(handler),
        EhBlockKind::LandingPad
    );
}

#[test]
fn removing_unreachable_blocks_keeps_a_declared_handler_entry() {
    let mut cfg = Cfg::<u32>::new();
    let body = cfg.entry();
    let entered = cfg.new_block();
    let never_entered = cfg.new_block();
    let after = cfg.new_block();
    cfg.add_edge(body, after, EdgeKind::Fallthrough);
    cfg.add_edge(body, entered, EdgeKind::ExceptionHandler);
    cfg.add_edge(entered, after, EdgeKind::Fallthrough);
    cfg.add_edge(never_entered, after, EdgeKind::Fallthrough);
    cfg.block_mut(entered).push(7);
    cfg.block_mut(never_entered).push(8);
    cfg.add_region(region([body], vec![catch(entered), catch(never_entered)]));

    assert_eq!(remove_unreachable(&mut cfg), 0);
    assert!(cfg.contains_block(never_entered));
    assert_metadata_survives_compaction(&mut cfg);
}

#[test]
fn removing_a_dead_region_releases_its_blocks() {
    let mut cfg = Cfg::<u32>::new();
    let body = cfg.entry();
    let dead_body = cfg.new_block();
    let dead_handler = cfg.new_block();
    let after = cfg.new_block();
    cfg.add_edge(body, after, EdgeKind::Fallthrough);
    cfg.add_edge(dead_body, dead_handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(dead_handler, after, EdgeKind::Fallthrough);
    cfg.block_mut(dead_body).push(1);
    cfg.block_mut(dead_handler).push(2);
    let dead = cfg.add_region(region([dead_body], vec![catch(dead_handler)]));
    let handler = HandlerRef::new(dead, 0);
    cfg.set_cleanup_resume(handler, dead_handler);
    cfg.add_continuation(
        handler,
        Continuation {
            reason: CompletionReason::Normal,
            resume: after,
        },
    );

    assert_eq!(cfg.remove_unreachable_regions(), 1);
    assert!(
        cfg.cleanup(handler).is_none(),
        "the region's cleanup went with it"
    );
    assert_eq!(remove_unreachable(&mut cfg), 2);
    assert_metadata_survives_compaction(&mut cfg);
}

#[test]
fn merging_never_moves_code_across_a_protection_boundary() {
    let mut cfg = Cfg::<u32>::new();
    let protected = cfg.entry();
    let unprotected = cfg.new_block();
    let handler = cfg.new_block();
    cfg.block_mut(protected).push(1);
    cfg.block_mut(unprotected).push(2);
    cfg.block_mut(handler).push(3);
    cfg.add_edge(protected, unprotected, EdgeKind::Fallthrough);
    cfg.add_region(region([protected], vec![catch(handler)]));

    assert_eq!(merge_blocks(&mut cfg), 0);
    assert!(!contract_edge(&mut cfg, protected, unprotected));
    assert_eq!(cfg.block(protected).instructions(), &[1]);
}

#[test]
fn merging_inside_one_region_still_happens() {
    let mut cfg = Cfg::<u32>::new();
    let first = cfg.entry();
    let second = cfg.new_block();
    let handler = cfg.new_block();
    cfg.block_mut(first).push(1);
    cfg.block_mut(second).push(2);
    cfg.add_edge(first, second, EdgeKind::Fallthrough);
    cfg.add_region(region([first, second], vec![catch(handler)]));

    assert_eq!(merge_blocks(&mut cfg), 1);
    assert_eq!(cfg.block(first).instructions(), &[1, 2]);
}

#[test]
fn merging_never_follows_an_exception_edge() {
    let mut cfg = Cfg::<u32>::new();
    let thrower = cfg.entry();
    let handler = cfg.new_block();
    cfg.block_mut(thrower).push(1);
    cfg.block_mut(handler).push(2);
    cfg.add_edge(thrower, handler, EdgeKind::ExceptionUnwind);

    assert_eq!(merge_blocks(&mut cfg), 0);
    assert!(!contract_edge(&mut cfg, thrower, handler));
    assert!(cfg.contains_block(handler));
}

#[test]
fn merging_into_a_handler_entry_is_refused() {
    let mut cfg = Cfg::<u32>::new();
    let body = cfg.entry();
    let before = cfg.new_block();
    let handler = cfg.new_block();
    cfg.block_mut(before).push(1);
    cfg.block_mut(handler).push(2);
    cfg.add_edge(body, before, EdgeKind::Fallthrough);
    cfg.add_edge(before, handler, EdgeKind::Fallthrough);
    cfg.add_region(region([], vec![catch(handler)]));

    // `before` folds into the entry; the handler entry, which a handler names,
    // stays a block of its own.
    assert_eq!(merge_blocks(&mut cfg), 1);
    assert_eq!(cfg.block(body).instructions(), &[1]);
    assert!(cfg.contains_block(handler));
    assert_eq!(cfg.block(handler).instructions(), &[2]);
    assert_metadata_survives_compaction(&mut cfg);
}

#[test]
fn splitting_a_protected_block_protects_both_halves() {
    let mut cfg = Cfg::<u32>::new();
    let body = cfg.entry();
    let handler = cfg.new_block();
    let after = cfg.new_block();
    cfg.block_mut(body).instructions_mut().extend([1, 2, 3]);
    cfg.block_mut(handler).push(9);
    cfg.add_edge(body, after, EdgeKind::Fallthrough);
    cfg.add_edge(body, handler, EdgeKind::ExceptionHandler);
    let owner = cfg.add_region(region([body], vec![catch(handler)]));

    let tail = cfg.split_block(body, 1);

    let protected = &cfg.region(owner).unwrap().protected_blocks;
    assert!(protected.contains(&body) && protected.contains(&tail));
    assert_eq!(cfg.protecting_region(tail).map(|r| r.id), Some(owner));
    assert_eq!(cfg.outgoing(tail).count(), 2, "the exits moved to the tail");
}

#[test]
fn splitting_a_handler_body_block_keeps_the_whole_body_known() {
    let mut cfg = Cfg::<u32>::new();
    let body = cfg.entry();
    let handler = cfg.new_block();
    cfg.block_mut(handler).instructions_mut().extend([1, 2]);
    cfg.add_edge(body, handler, EdgeKind::ExceptionHandler);
    let owner = cfg.add_region(region([body], vec![catch(handler)]));

    let tail = cfg.split_block(handler, 1);

    let known = cfg.region(owner).unwrap().handlers[0]
        .body
        .blocks()
        .unwrap();
    assert!(known.contains(&handler) && known.contains(&tail));
}

#[test]
fn splitting_a_cleanup_resume_block_moves_the_resume_point_to_the_tail() {
    let mut cfg = Cfg::<u32>::new();
    let body = cfg.entry();
    let cleanup = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cleanup).instructions_mut().extend([1, 2]);
    cfg.add_edge(body, cleanup, EdgeKind::Fallthrough);
    cfg.add_edge(cleanup, exit, EdgeKind::Fallthrough);
    let owner = cfg.add_region(region(
        [body],
        vec![Handler {
            entry: cleanup,
            body: HandlerBody::known([cleanup]),
            kind: HandlerKind::Finally,
        }],
    ));
    let handler = HandlerRef::new(owner, 0);
    cfg.set_cleanup_resume(handler, cleanup);
    cfg.add_continuation(
        handler,
        Continuation {
            reason: CompletionReason::Normal,
            resume: exit,
        },
    );

    let tail = cfg.split_block(cleanup, 1);

    assert_eq!(cfg.cleanup(handler).unwrap().resume_from, Some(tail));
    assert_eq!(cfg.successors(tail).collect::<Vec<_>>(), [exit]);
}

#[test]
fn merging_keeps_the_label_of_the_block_it_absorbs() {
    let mut cfg = Cfg::<u32>::new();
    let entry = cfg.entry();
    let looped = cfg.new_block();
    cfg.block_mut(looped).set_label("loop");
    cfg.add_edge(entry, looped, EdgeKind::Fallthrough);

    assert_eq!(merge_blocks(&mut cfg), 1);
    assert_eq!(cfg.block(entry).label(), Some("loop"));
}
