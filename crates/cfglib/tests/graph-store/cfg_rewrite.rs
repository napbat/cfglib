//! A CFG pass followed by `compact`: `Rewrite::then_renumbered` carries the pass's record across
//! the renumbering, so a consumer still holds one answer for every identity it held before the
//! pass.
//!
//! The record only lists identities the pass touched, so an identity it left alone is found in
//! the `Renumbering` instead. `resolve_block` and `resolve_edge` are that two-step lookup.

use cfglib::{BlockId, Cfg, CfgRenumbering, EdgeId, EdgeKind, Rewrite, simplify_mapped};

/// Edge provenance a frontend keeps beside the control flow: owned, not `Copy`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Origin(String);

fn origin(text: &str) -> Origin {
    Origin(text.to_owned())
}

/// ```text
/// entry --true--> then --> tail --> join
///   |                               ^ ^
///   +---false--> empty ------------+ |
/// orphan (unreachable) -------------+
/// ```
struct Fixture {
    cfg: Cfg<String, Origin>,
    entry: BlockId,
    then_block: BlockId,
    tail: BlockId,
    empty: BlockId,
    join: BlockId,
    orphan: BlockId,
    entry_then: EdgeId,
    entry_empty: EdgeId,
    then_tail: EdgeId,
    tail_join: EdgeId,
    empty_join: EdgeId,
    orphan_join: EdgeId,
}

fn fixture() -> Fixture {
    let mut cfg = Cfg::<String, Origin>::with_edge_payload();
    let entry = cfg.entry();
    let then_block = cfg.new_block();
    let tail = cfg.new_block();
    let empty = cfg.new_block();
    let join = cfg.new_block();
    let orphan = cfg.new_block();
    for (block, text) in [
        (entry, "entry"),
        (then_block, "then"),
        (tail, "tail"),
        (join, "join"),
        (orphan, "orphan"),
    ] {
        cfg.block_mut(block).push(text.to_owned());
    }
    let entry_then = cfg.add_edge_with_payload(
        entry,
        then_block,
        EdgeKind::ConditionalTrue,
        origin("entry_then"),
    );
    let entry_empty = cfg.add_edge_with_payload(
        entry,
        empty,
        EdgeKind::ConditionalFalse,
        origin("entry_empty"),
    );
    let then_tail =
        cfg.add_edge_with_payload(then_block, tail, EdgeKind::Fallthrough, origin("then_tail"));
    let tail_join =
        cfg.add_edge_with_payload(tail, join, EdgeKind::Fallthrough, origin("tail_join"));
    let empty_join =
        cfg.add_edge_with_payload(empty, join, EdgeKind::Fallthrough, origin("empty_join"));
    let orphan_join =
        cfg.add_edge_with_payload(orphan, join, EdgeKind::Fallthrough, origin("orphan_join"));
    Fixture {
        cfg,
        entry,
        then_block,
        tail,
        empty,
        join,
        orphan,
        entry_then,
        entry_empty,
        then_tail,
        tail_join,
        empty_join,
        orphan_join,
    }
}

fn resolve_block(rewrite: &Rewrite, renumbering: &CfgRenumbering, old: BlockId) -> Vec<BlockId> {
    rewrite.blocks(old).map_or_else(
        || renumbering.node(old).into_iter().collect(),
        <[BlockId]>::to_vec,
    )
}

fn resolve_edge(rewrite: &Rewrite, renumbering: &CfgRenumbering, old: EdgeId) -> Vec<EdgeId> {
    rewrite.edges(old).map_or_else(
        || renumbering.edge(old).into_iter().collect(),
        <[EdgeId]>::to_vec,
    )
}

fn instructions(cfg: &Cfg<String, Origin>, block: BlockId) -> Vec<&str> {
    cfg.block(block)
        .instructions()
        .iter()
        .map(String::as_str)
        .collect()
}

#[test]
fn a_cleanup_pass_followed_by_compact_leaves_every_old_identity_with_one_answer() {
    let Fixture {
        mut cfg,
        entry,
        then_block,
        tail,
        empty,
        join,
        orphan,
        entry_then,
        entry_empty,
        then_tail,
        tail_join,
        empty_join,
        orphan_join,
    } = fixture();

    let (changes, rewrite) = simplify_mapped(&mut cfg);
    assert_eq!(
        changes, 3,
        "one unreachable block, one empty block, one merge"
    );
    assert_eq!((cfg.block_count(), cfg.block_bound()), (3, 6));
    let renumbering = cfg.compact();
    let rewrite = rewrite.then_renumbered(&renumbering);
    assert!(cfg.is_compact());
    assert_eq!((cfg.block_count(), cfg.edge_count()), (3, 3));

    let blocks = |old| resolve_block(&rewrite, &renumbering, old);
    let new_entry = blocks(entry);
    let new_then = blocks(then_block);
    let new_join = blocks(join);
    assert_eq!((new_entry.len(), new_then.len(), new_join.len()), (1, 1, 1));
    let (new_entry, new_then, new_join) = (new_entry[0], new_then[0], new_join[0]);
    assert_eq!(new_entry, cfg.entry());
    assert_eq!(instructions(&cfg, new_entry), ["entry"]);
    assert_eq!(
        instructions(&cfg, new_then),
        ["then", "tail"],
        "the tail was merged into `then`"
    );
    assert_eq!(instructions(&cfg, new_join), ["join"]);
    assert_eq!(
        blocks(tail),
        [new_then],
        "the merged block's identity is its absorber's"
    );
    assert!(blocks(empty).is_empty(), "the bypassed block is gone");
    assert!(blocks(orphan).is_empty(), "the unreachable block is gone");

    let edges = |old| resolve_edge(&rewrite, &renumbering, old);
    let assert_edge = |old: EdgeId, source, target, kind, text: &str| {
        let new = edges(old);
        assert_eq!(new.len(), 1, "edge {old} survives as one edge");
        let edge = cfg.edge(new[0]);
        assert_eq!(
            (edge.source(), edge.target(), edge.kind()),
            (source, target, kind)
        );
        assert_eq!(edge.payload(), &origin(text), "payload of edge {old}");
    };
    assert_edge(
        entry_then,
        new_entry,
        new_then,
        EdgeKind::ConditionalTrue,
        "entry_then",
    );
    assert_edge(
        entry_empty,
        new_entry,
        new_join,
        EdgeKind::ConditionalFalse,
        "entry_empty",
    );
    assert_edge(
        tail_join,
        new_then,
        new_join,
        EdgeKind::Fallthrough,
        "tail_join",
    );
    assert!(edges(then_tail).is_empty(), "the merged edge is gone");
    assert!(edges(empty_join).is_empty(), "the bypassing edge is gone");
    assert!(
        edges(orphan_join).is_empty(),
        "the unreachable block's edge is gone"
    );

    assert!(rewrite.created_blocks().is_empty() && rewrite.created_edges().is_empty());
}

#[test]
fn the_carried_record_never_names_an_identity_the_compaction_dropped() {
    let mut fixture = fixture();
    let (_, rewrite) = simplify_mapped(&mut fixture.cfg);
    let renumbering = fixture.cfg.compact();
    let rewrite = rewrite.then_renumbered(&renumbering);

    for (_, replacements) in rewrite.block_mappings() {
        for &block in replacements {
            assert!(block.index() < fixture.cfg.block_bound());
            assert!(fixture.cfg.contains_block(block));
        }
    }
    for (_, replacements) in rewrite.edge_mappings() {
        for &edge in replacements {
            assert!(edge.index() < fixture.cfg.edge_bound());
            assert!(fixture.cfg.contains_edge(edge));
        }
    }
    assert_eq!(rewrite.blocks(fixture.orphan), Some([].as_slice()));
    assert_eq!(rewrite.blocks(fixture.empty), Some([].as_slice()));
    assert_eq!(rewrite.edges(fixture.then_tail), Some([].as_slice()));
}

#[test]
fn carrying_a_record_across_two_compactions_equals_carrying_it_across_their_composition() {
    let mut fixture = fixture();
    let (_, rewrite) = simplify_mapped(&mut fixture.cfg);
    let first = fixture.cfg.compact();

    // A later pass drops the merged block, so the first record's replacement for `tail` is
    // dropped by the second compaction.
    let merged = first
        .node(fixture.then_block)
        .expect("`then` survived the first compaction");
    assert!(fixture.cfg.remove_block(merged));
    let second = fixture.cfg.compact();

    let stepwise = rewrite.then_renumbered(&first).then_renumbered(&second);
    let composed = rewrite.then_renumbered(&first.compose(&second));
    assert_eq!(stepwise, composed);

    assert_eq!(
        stepwise.blocks(fixture.tail),
        Some([].as_slice()),
        "its absorber was dropped"
    );
    assert_eq!(stepwise.edges(fixture.tail_join), Some([].as_slice()));
    let surviving = stepwise
        .edges(fixture.entry_empty)
        .expect("the redirected edge is recorded");
    assert_eq!(surviving.len(), 1);
    let edge = fixture.cfg.edge(surviving[0]);
    assert_eq!(edge.payload(), &origin("entry_empty"));
    let join = first
        .compose(&second)
        .node(fixture.join)
        .expect("`join` survived both compactions");
    assert_eq!((edge.source(), edge.target()), (fixture.cfg.entry(), join));
}

#[test]
fn a_pass_that_changes_nothing_records_nothing_and_carries_an_empty_record() {
    let mut cfg = Cfg::<String, Origin>::with_edge_payload();
    let left = cfg.new_block();
    let right = cfg.new_block();
    let join = cfg.new_block();
    for (block, text) in [
        (cfg.entry(), "entry"),
        (left, "left"),
        (right, "right"),
        (join, "join"),
    ] {
        cfg.block_mut(block).push(text.to_owned());
    }
    cfg.add_edge_with_payload(
        cfg.entry(),
        left,
        EdgeKind::ConditionalTrue,
        origin("to_left"),
    );
    cfg.add_edge_with_payload(
        cfg.entry(),
        right,
        EdgeKind::ConditionalFalse,
        origin("to_right"),
    );
    cfg.add_edge_with_payload(left, join, EdgeKind::Fallthrough, origin("left_join"));
    cfg.add_edge_with_payload(right, join, EdgeKind::Fallthrough, origin("right_join"));

    let (changes, nothing) = simplify_mapped(&mut cfg);
    assert_eq!(changes, 0);
    assert!(nothing.is_empty());
    let renumbering = cfg.compact();
    assert!(renumbering.is_identity());
    assert!(nothing.then_renumbered(&renumbering).is_empty());
}
