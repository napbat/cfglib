//! `compact` on a graph with regions: every block a region, handler, or cleanup
//! names is renumbered exactly as the blocks are, checked against an
//! independent mapping written from the returned `Renumbering`.

use std::collections::BTreeSet;
use std::panic::{AssertUnwindSafe, catch_unwind};

use cfglib::{
    BlockId, Cfg, CfgRenumbering, Cleanup, CompletionReason, Continuation, EdgeKind, EhModel,
    Handler, HandlerBody, HandlerKind, HandlerRef,
};

use crate::shapes::{Stmt, block_of, catch, handler, protect, try_catch, unknown_catch};

/// Blocks of the holey graph, named.
struct Holey {
    cfg: Cfg<Stmt>,
    removed: Vec<BlockId>,
}

/// Every reference kind a region graph holds, with removed blocks both before
/// and among the blocks they point at, and stale references to some of them.
fn holey() -> Holey {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(Stmt::text("setup()"));
    let junk_before = block_of(&mut cfg, &["junk before"]);
    let body = block_of(&mut cfg, &["risky()"]);
    let junk_between = block_of(&mut cfg, &["junk between"]);
    let inner = block_of(&mut cfg, &["inner()"]);
    let catch_pad = block_of(&mut cfg, &["catch (e)"]);
    let junk_in_extent = block_of(&mut cfg, &["junk in the catch extent"]);
    let catch_tail = block_of(&mut cfg, &["log(e)"]);
    let filter_pad = block_of(&mut cfg, &["filtered"]);
    let funclet = block_of(&mut cfg, &["is_recoverable(e)"]);
    let cleanup = block_of(&mut cfg, &["close()"]);
    let after = block_of(&mut cfg, &["return"]);

    cfg.add_edge(entry, junk_before, EdgeKind::Fallthrough);
    cfg.add_edge(entry, body, EdgeKind::Fallthrough);
    cfg.add_edge(body, junk_between, EdgeKind::Fallthrough);
    cfg.add_edge(body, inner, EdgeKind::Fallthrough);
    cfg.add_edge(inner, after, EdgeKind::Fallthrough);
    cfg.add_edge(body, catch_pad, EdgeKind::ExceptionHandler);
    cfg.add_edge(inner, filter_pad, EdgeKind::ExceptionUnwind);
    cfg.add_edge(inner, cleanup, EdgeKind::ExceptionUnwind);
    cfg.add_edge(catch_pad, junk_in_extent, EdgeKind::Fallthrough);
    cfg.add_edge(catch_pad, catch_tail, EdgeKind::Fallthrough);
    cfg.add_edge(catch_tail, after, EdgeKind::Fallthrough);
    cfg.add_edge(filter_pad, after, EdgeKind::Fallthrough);
    cfg.add_edge(cleanup, after, EdgeKind::Fallthrough);
    let removed = vec![junk_before, junk_between, junk_in_extent];
    for &block in &removed {
        cfg.remove_block(block);
    }

    let outer = protect(
        &mut cfg,
        &[junk_before, body, junk_between, inner],
        vec![
            handler(
                catch_pad,
                HandlerKind::Catch,
                &[catch_pad, junk_in_extent, catch_tail],
            ),
            Handler {
                entry: filter_pad,
                body: HandlerBody::unknown(),
                kind: HandlerKind::Filter {
                    filter_block: funclet,
                },
            },
        ],
    );
    let nested = protect(
        &mut cfg,
        &[inner],
        vec![handler(cleanup, HandlerKind::Finally, &[cleanup])],
    );
    cfg.region_mut(nested).expect("added").parent = Some(outer);
    let finally = HandlerRef::new(nested, 0);
    cfg.set_cleanup_resume(finally, cleanup);
    for (reason, resume) in [
        (CompletionReason::Normal, after),
        (CompletionReason::Transfer, catch_tail),
        (CompletionReason::Unwind, inner),
    ] {
        cfg.add_continuation(finally, Continuation { reason, resume });
    }
    Holey { cfg, removed }
}

/// `block` mapped through the renumbering, which it must survive.
fn survivor(renumbering: &CfgRenumbering, block: BlockId) -> BlockId {
    renumbering
        .node(block)
        .unwrap_or_else(|| panic!("{block} should have survived"))
}

/// A set of blocks as compaction leaves it: the members that survived, renamed.
fn surviving(renumbering: &CfgRenumbering, blocks: &BTreeSet<BlockId>) -> BTreeSet<BlockId> {
    blocks
        .iter()
        .filter_map(|&block| renumbering.node(block))
        .collect()
}

#[test]
fn compaction_renumbers_every_block_reference_as_it_renumbers_the_blocks() {
    let Holey { mut cfg, removed } = holey();
    let before = cfg.clone();

    let renumbering = cfg.compact();

    for &block in &removed {
        assert_eq!(renumbering.node(block), None);
    }
    let live: Vec<BlockId> = before.block_ids().collect();
    for (position, &block) in live.iter().enumerate() {
        assert_eq!(
            survivor(&renumbering, block).index(),
            position,
            "survivors keep their relative order"
        );
    }
    assert_eq!(cfg.entry(), survivor(&renumbering, before.entry()));

    assert_eq!(cfg.regions().len(), before.regions().len());
    for (old, new) in before.regions().iter().zip(cfg.regions()) {
        assert_eq!(new.id, old.id, "region identities are not renumbered");
        assert_eq!(new.parent, old.parent);
        assert_eq!(
            new.protected_blocks,
            surviving(&renumbering, &old.protected_blocks)
        );
        assert_eq!(new.handlers.len(), old.handlers.len());
        for (old_handler, new_handler) in old.handlers.iter().zip(&new.handlers) {
            assert_eq!(new_handler.entry, survivor(&renumbering, old_handler.entry));
            match (&old_handler.body, &new_handler.body) {
                (HandlerBody::Unknown, HandlerBody::Unknown) => {}
                (HandlerBody::Known(old_body), HandlerBody::Known(new_body)) => {
                    assert_eq!(*new_body, surviving(&renumbering, old_body));
                }
                (old_body, new_body) => panic!("{old_body:?} became {new_body:?}"),
            }
            match (old_handler.kind, new_handler.kind) {
                (
                    HandlerKind::Filter { filter_block: old },
                    HandlerKind::Filter { filter_block: new },
                ) => assert_eq!(new, survivor(&renumbering, old)),
                (old_kind, new_kind) => assert_eq!(old_kind, new_kind),
            }
        }
    }

    assert_eq!(cfg.cleanups().len(), before.cleanups().len());
    for (old, new) in before.cleanups().iter().zip(cfg.cleanups()) {
        let expected = Cleanup {
            handler: old.handler,
            resume_from: old.resume_from.map(|block| survivor(&renumbering, block)),
            continuations: old
                .continuations
                .iter()
                .map(|route| Continuation {
                    reason: route.reason,
                    resume: survivor(&renumbering, route.resume),
                })
                .collect(),
        };
        assert_eq!(*new, expected);
    }
}

#[test]
fn the_eh_model_before_and_after_compaction_agrees_through_the_renumbering() {
    let Holey { mut cfg, .. } = holey();
    let before_cfg = cfg.clone();
    let before = EhModel::compute(&cfg);

    let renumbering = cfg.compact();
    let after = EhModel::compute(&cfg);

    assert_eq!(after.block_bound(), cfg.block_count());
    for block in before_cfg.block_ids() {
        let renamed = survivor(&renumbering, block);
        assert_eq!(
            after.block_kind(renamed),
            before.block_kind(block),
            "{block}"
        );
        assert_eq!(after.handlers(renamed), before.handlers(block), "{block}");
        assert_eq!(
            *after.protected_by(renamed),
            surviving(&renumbering, before.protected_by(block)),
            "{block}"
        );
        assert_eq!(
            after
                .cleanup(renamed)
                .map(|cleanup| cleanup.continuations.len()),
            before
                .cleanup(block)
                .map(|cleanup| cleanup.continuations.len()),
            "{block}"
        );
    }
    let expected_edges: Vec<_> = before
        .eh_edges()
        .iter()
        .map(|edge| {
            (
                renumbering.edge(edge.edge_id).expect("the edge survived"),
                survivor(&renumbering, edge.from),
                survivor(&renumbering, edge.to),
                edge.kind,
            )
        })
        .collect();
    let actual_edges: Vec<_> = after
        .eh_edges()
        .iter()
        .map(|edge| (edge.edge_id, edge.from, edge.to, edge.kind))
        .collect();
    assert_eq!(actual_edges, expected_edges);
}

#[test]
fn a_second_compaction_finds_nothing_to_move() {
    let Holey { mut cfg, .. } = holey();
    cfg.compact();
    let settled = cfg.clone();

    let again = cfg.compact();

    assert!(cfg.is_compact());
    assert!(again.is_identity());
    assert_eq!(cfg, settled);
}

#[test]
fn a_graph_with_no_removed_block_compacts_to_itself() {
    let mut cfg = try_catch(&["log(e)"]).cfg;
    let before = cfg.clone();

    let renumbering = cfg.compact();

    assert!(renumbering.is_identity());
    assert!(cfg.is_compact());
    assert_eq!(cfg.regions(), before.regions());
    assert_eq!(cfg.entry(), before.entry());
    assert_eq!(cfg.to_text(), before.to_text());
}

/// One graph per reference kind that names a single block, and that block.
fn single_references() -> Vec<(&'static str, Cfg<Stmt>, BlockId)> {
    let mut graphs = Vec::new();

    let entry_pad = try_catch(&["log(e)"]);
    let pad = entry_pad.pad;
    graphs.push(("handler entry", entry_pad.cfg, pad));

    let mut filtered = try_catch(&["log(e)"]);
    let funclet = block_of(&mut filtered.cfg, &["is_recoverable(e)"]);
    let filtered_pad = block_of(&mut filtered.cfg, &["filtered"]);
    filtered
        .cfg
        .add_edge(filtered.body, filtered_pad, EdgeKind::ExceptionUnwind);
    protect(
        &mut filtered.cfg,
        &[filtered.body],
        vec![handler(
            filtered_pad,
            HandlerKind::Filter {
                filter_block: funclet,
            },
            &[filtered_pad],
        )],
    );
    graphs.push(("filter block", filtered.cfg, funclet));

    let mut resume = try_catch(&["log(e)"]);
    let resume_block = block_of(&mut resume.cfg, &["resume here"]);
    resume
        .cfg
        .set_cleanup_resume(HandlerRef::new(resume.region, 0), resume_block);
    graphs.push(("cleanup resume block", resume.cfg, resume_block));

    let mut route = try_catch(&["log(e)"]);
    let target = block_of(&mut route.cfg, &["continue here"]);
    route.cfg.add_continuation(
        HandlerRef::new(route.region, 0),
        Continuation {
            reason: CompletionReason::Normal,
            resume: target,
        },
    );
    graphs.push(("cleanup continuation", route.cfg, target));

    graphs
}

#[test]
fn compaction_panics_when_a_single_block_reference_names_a_removed_block() {
    for (name, mut cfg, block) in single_references() {
        assert!(cfg.remove_block(block), "{name}");

        let outcome = catch_unwind(AssertUnwindSafe(|| cfg.compact()));

        assert!(
            outcome.is_err(),
            "{name}: a reference with no replacement is documented to panic"
        );
    }
}

#[test]
fn compaction_keeps_a_graph_whose_removed_blocks_are_not_single_references() {
    for (name, mut cfg, block) in single_references() {
        let spare = block_of(&mut cfg, &["spare"]);
        cfg.remove_block(spare);
        assert!(cfg.contains_block(block), "{name}");

        let renumbering = cfg.compact();

        assert_eq!(renumbering.node(spare), None, "{name}");
        assert!(cfg.is_compact(), "{name}");
    }
}

#[test]
fn a_set_of_blocks_loses_a_removed_member_instead_of_failing() {
    let mut fixture = try_catch(&["log(e)"]);
    let spare = block_of(&mut fixture.cfg, &["spare"]);
    {
        let region = fixture.cfg.region_mut(fixture.region).expect("exists");
        region.protected_blocks.insert(spare);
        region.handlers.push(Handler {
            entry: fixture.pad,
            body: HandlerBody::known([fixture.pad, spare]),
            kind: HandlerKind::Catch,
        });
    }
    fixture.cfg.remove_block(spare);

    let renumbering = fixture.cfg.compact();

    let region = fixture.cfg.region(fixture.region).expect("exists");
    assert_eq!(region.protected_blocks.len(), 1);
    assert!(
        region
            .protected_blocks
            .contains(&survivor(&renumbering, fixture.body))
    );
    assert_eq!(
        region.handlers[1].body.blocks().map(BTreeSet::len),
        Some(1),
        "the known body shrinks to the survivor"
    );
}

#[test]
fn an_unknown_extent_has_no_block_set_to_shrink_and_stays_unknown() {
    let mut fixture = try_catch(&["log(e)"]);
    let extra = block_of(&mut fixture.cfg, &["extra pad"]);
    fixture
        .cfg
        .add_edge(fixture.body, extra, EdgeKind::ExceptionUnwind);
    protect(
        &mut fixture.cfg,
        &[fixture.body],
        vec![unknown_catch(extra), catch(fixture.pad)],
    );
    let hole = block_of(&mut fixture.cfg, &["hole"]);
    fixture.cfg.remove_block(hole);

    fixture.cfg.compact();

    assert_eq!(
        fixture.cfg.regions()[1].handlers[0].body,
        HandlerBody::Unknown
    );
}
