//! Integration tests: natural loops, nesting, reducibility and loop
//! canonicalization, through the public facade only.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, CfgMetrics, DominatorTree, EdgeKind, GraphMetrics, LoopNestingTree,
    block_nesting_depths, canonicalize_loops, cfg_block_nesting_depths, detect_loops,
    detect_loops_tagged, find_back_edges, insert_preheader, loop_exit_blocks, verify,
};

fn set(blocks: impl IntoIterator<Item = BlockId>) -> BTreeSet<BlockId> {
    blocks.into_iter().collect()
}

#[test]
fn a_dead_predecessor_is_not_part_of_the_loop_it_feeds() {
    // entry -> header <-> body, and an unreachable block that jumps into the body.
    let mut cfg = Cfg::<u32>::new();
    let [header, body, dead] = [(); 3].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, body, EdgeKind::Fallthrough);
    cfg.add_edge(body, header, EdgeKind::Back);
    cfg.add_edge(dead, body, EdgeKind::Fallthrough);

    let dom = DominatorTree::compute(&cfg);
    let loops = detect_loops(&cfg, &dom);
    assert_eq!(loops.len(), 1);
    assert_eq!(loops[0].body, set([header, body]));

    let depths = block_nesting_depths(&cfg);
    assert_eq!(depths[header.index()], 1);
    assert_eq!(depths[body.index()], 1);
    assert_eq!(depths[dead.index()], 0, "dead code is in no loop");
}

#[test]
fn unreachable_code_has_no_loops_however_it_branches() {
    // A dead self-loop and a dead two-cycle: neither runs, so neither is a loop.
    let mut cfg = Cfg::<u32>::new();
    let [dead_self, dead_a, dead_b] = [(); 3].map(|()| cfg.new_block());
    cfg.add_edge(dead_self, dead_self, EdgeKind::ConditionalTrue);
    cfg.add_edge(dead_a, dead_b, EdgeKind::Fallthrough);
    cfg.add_edge(dead_b, dead_a, EdgeKind::Back);

    let dom = DominatorTree::compute(&cfg);
    assert!(find_back_edges(&cfg, &dom).is_empty());
    assert!(detect_loops_tagged(&cfg, &dom).is_empty());
    for depths in [block_nesting_depths(&cfg), cfg_block_nesting_depths(&cfg)] {
        assert_eq!(depths, [0; 4]);
    }
}

#[test]
fn a_tagged_irreducible_loop_covers_its_cycle_and_nothing_before_it() {
    // entry -> {h, y}, h -> x -> y, and the tagged edge y -> h. Dominance
    // cannot see the loop, the tag can; the entry leads into the cycle without
    // being part of it.
    let mut cfg = Cfg::<u32>::new();
    let entry = cfg.entry();
    let [h, x, y] = [(); 3].map(|()| cfg.new_block());
    cfg.add_edge(entry, h, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, y, EdgeKind::ConditionalFalse);
    cfg.add_edge(h, x, EdgeKind::Fallthrough);
    cfg.add_edge(x, y, EdgeKind::Fallthrough);
    cfg.add_edge(y, h, EdgeKind::Back);

    let dom = DominatorTree::compute(&cfg);
    assert!(detect_loops(&cfg, &dom).is_empty());
    let loops = detect_loops_tagged(&cfg, &dom);
    assert_eq!(loops.len(), 1);
    assert_eq!(loops[0].header, h);
    assert_eq!(loops[0].latches, set([y]));
    assert_eq!(loops[0].body, set([h, x, y]));

    let depths = cfg_block_nesting_depths(&cfg);
    assert_eq!(depths[entry.index()], 0);
    assert_eq!(depths[h.index()], 1);
    assert_eq!(depths[x.index()], 1);
}

/// `entry -> outer`, `outer -> inner -> body -> inner`, `inner -> outer`, and
/// the exit `outer -> exit`.
struct NestedLoops {
    cfg: Cfg<u32>,
    outer: BlockId,
    inner: BlockId,
    body: BlockId,
    exit: BlockId,
}

fn nested_loops() -> NestedLoops {
    let mut cfg = Cfg::<u32>::new();
    let [outer, inner, body, exit] = [(); 4].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), outer, EdgeKind::Fallthrough);
    cfg.add_edge(outer, inner, EdgeKind::ConditionalTrue);
    cfg.add_edge(outer, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(inner, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, inner, EdgeKind::Back);
    cfg.add_edge(inner, outer, EdgeKind::Back);
    NestedLoops {
        cfg,
        outer,
        inner,
        body,
        exit,
    }
}

#[test]
fn nested_loops_form_a_tree_whose_depths_the_loops_share() {
    let NestedLoops {
        cfg,
        outer,
        inner,
        body,
        exit,
    } = nested_loops();
    let dom = DominatorTree::compute(&cfg);
    let loops = detect_loops(&cfg, &dom);
    assert_eq!(
        loops.iter().map(|lp| lp.header).collect::<Vec<_>>(),
        [outer, inner],
        "outermost first"
    );

    let tree = LoopNestingTree::compute(&loops);
    assert_eq!(tree.roots(), [0]);
    assert_eq!(tree.nodes[1].parent, Some(0));
    assert_eq!(tree.nodes[0].children, [1]);
    for (index, lp) in loops.iter().enumerate() {
        assert_eq!(
            lp.depth,
            tree.depth(index),
            "depth of the loop at {:?}",
            lp.header
        );
    }
    assert_eq!(tree.innermost_loop(body), Some(1));
    assert_eq!(tree.innermost_loop(outer), Some(0));
    assert_eq!(tree.innermost_loop(exit), None);
    assert_eq!(tree.innermost_loop(inner), Some(1));
}

#[test]
fn an_irreducible_cycle_tagged_both_ways_nests_without_hanging() {
    // entry -> {b1, b2}, b1 -> b2 and b2 -> b1 both tagged back edges: two
    // loops over one body, each header inside the other loop.
    let mut cfg = Cfg::<u32>::new();
    let [b1, b2] = [(); 2].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), b1, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), b2, EdgeKind::ConditionalFalse);
    cfg.add_edge(b1, b2, EdgeKind::Back);
    cfg.add_edge(b2, b1, EdgeKind::Back);

    let dom = DominatorTree::compute(&cfg);
    let loops = detect_loops_tagged(&cfg, &dom);
    assert_eq!(loops.len(), 2);
    assert_eq!(loops[0].body, loops[1].body);

    let tree = LoopNestingTree::compute(&loops);
    assert_eq!(tree.roots(), [0, 1], "neither loop contains the other");
    assert!(tree.nodes.iter().all(|node| node.depth == 0));
    assert_eq!(cfg_block_nesting_depths(&cfg), [0, 1, 1]);
    assert_eq!(CfgMetrics::compute(&cfg).graph.max_nesting_depth, 1);
}

#[test]
fn the_deepest_loop_is_the_deepest_block() {
    let NestedLoops {
        cfg,
        outer,
        inner,
        body,
        exit,
    } = nested_loops();

    let depths = block_nesting_depths(&cfg);
    assert_eq!(depths[outer.index()], 1);
    assert_eq!(depths[inner.index()], 2);
    assert_eq!(depths[body.index()], 2);
    assert_eq!(depths[exit.index()], 0);
    for max in [
        CfgMetrics::compute(&cfg).graph.max_nesting_depth,
        GraphMetrics::compute(&cfg).max_nesting_depth,
    ] {
        assert_eq!(max, depths.iter().copied().max().unwrap_or(0));
        assert_eq!(max, 2);
    }
}

#[test]
fn one_loop_is_depth_one_and_no_loop_is_depth_zero() {
    let mut cfg = Cfg::<u32>::new();
    let head = cfg.new_block();
    cfg.add_edge(cfg.entry(), head, EdgeKind::Fallthrough);
    assert_eq!(CfgMetrics::compute(&cfg).graph.max_nesting_depth, 0);

    cfg.add_edge(head, head, EdgeKind::ConditionalTrue);
    assert_eq!(
        CfgMetrics::compute(&cfg).graph.max_nesting_depth,
        1,
        "a block that branches to itself is a loop"
    );
    assert_eq!(GraphMetrics::compute(&cfg).max_nesting_depth, 1);
}

#[test]
fn metrics_count_live_blocks_whether_or_not_the_store_was_compacted() {
    let NestedLoops { mut cfg, exit, .. } = nested_loops();
    let spare = cfg.new_block();
    assert!(cfg.remove_block(spare));
    assert!(cfg.remove_block(exit));

    let before = CfgMetrics::compute(&cfg);
    assert_eq!(before.graph.node_count, cfg.block_count());
    assert_eq!(before.graph.unreachable_node_count, 0);
    assert_eq!(before.graph.reachable_node_count, before.graph.node_count);

    cfg.compact();
    assert_eq!(CfgMetrics::compute(&cfg), before);
}

/// A route label an edge-aware consumer hangs on its edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Route {
    #[default]
    Plain,
    Hot,
    Cold,
}

#[test]
fn a_preheader_takes_over_the_entering_edges_with_their_identity_and_payload() {
    // entry -> header twice over, a hot weighted conditional arm and a cold
    // one through `side`, and the exit `header -> done`.
    let mut cfg = Cfg::<u32, Route>::with_edge_payload();
    let entry = cfg.entry();
    let [header, body, side, done] = [(); 4].map(|()| cfg.new_block());
    let hot = cfg.add_weighted_edge_with_payload(
        entry,
        header,
        EdgeKind::ConditionalTrue,
        0.8,
        Route::Hot,
    );
    let to_done = cfg.add_edge(entry, done, EdgeKind::ConditionalFalse);
    let to_side = cfg.add_edge(entry, side, EdgeKind::SwitchCase);
    let cold = cfg.add_edge_with_payload(side, header, EdgeKind::Fallthrough, Route::Cold);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, header, EdgeKind::Back);
    cfg.add_edge(header, done, EdgeKind::ConditionalFalse);

    let before: Vec<_> = cfg.outgoing(entry).collect();
    let dom = DominatorTree::compute(&cfg);
    let lp = detect_loops(&cfg, &dom).remove(0);
    let preheader = insert_preheader(&mut cfg, &lp).expect("two entries need a preheader");

    assert_eq!(before, [hot, to_done, to_side]);
    assert_eq!(
        cfg.outgoing(entry).collect::<Vec<_>>(),
        before,
        "the entry's arms keep their order"
    );
    assert_eq!(cfg.edge(hot).target(), preheader);
    assert_eq!(cfg.edge(hot).weight(), Some(0.8));
    assert_eq!(*cfg.edge(hot).payload(), Route::Hot);
    assert_eq!(cfg.edge(cold).target(), preheader);
    assert_eq!(*cfg.edge(cold).payload(), Route::Cold);
    assert_eq!(cfg.successors(preheader).collect::<Vec<_>>(), [header]);
    assert!(verify(&cfg).is_ok());
}

#[test]
fn canonicalizing_gives_every_entered_loop_a_dedicated_preheader() {
    // The loop is entered from a block that also branches around it.
    let mut cfg = Cfg::<u32>::new();
    let [header, body, exit] = [(); 3].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), header, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, header, EdgeKind::Back);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    let dom = DominatorTree::compute(&cfg);

    let canonical = canonicalize_loops(&mut cfg, &dom);

    assert_eq!(canonical.len(), 1);
    let preheader = canonical[0]
        .preheader
        .expect("the loop is entered from outside");
    assert_eq!(cfg.successors(preheader).collect::<Vec<_>>(), [header]);
    assert_eq!(canonical[0].exits, set([exit]));
    assert!(verify(&cfg).is_ok());

    // Canonical form is stable: the same loop, and nothing left to insert.
    let dom = DominatorTree::compute(&cfg);
    let again = canonicalize_loops(&mut cfg, &dom);
    assert_eq!(again.len(), 1);
    assert_eq!(again[0].preheader, Some(preheader));
    assert_eq!(again[0].natural_loop.body, canonical[0].natural_loop.body);
}

#[test]
fn a_loop_headed_by_the_entry_has_no_preheader() {
    let mut cfg = Cfg::<u32>::new();
    let [body, exit] = [(); 2].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), body, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, cfg.entry(), EdgeKind::Back);
    cfg.add_edge(cfg.entry(), exit, EdgeKind::ConditionalFalse);
    let dom = DominatorTree::compute(&cfg);

    let canonical = canonicalize_loops(&mut cfg, &dom);

    assert_eq!(canonical.len(), 1);
    assert_eq!(canonical[0].preheader, None);
    assert_eq!(cfg.block_count(), 3);
}

#[test]
fn canonicalized_nested_loops_report_current_bodies_and_exits() {
    // outer -> {inner, side}, side -> inner: the inner header has two entries,
    // and its preheader lies inside the outer loop. A second loop follows the
    // first, entered from the first loop's exit and from the entry.
    let mut cfg = Cfg::<u32>::new();
    let [outer, inner, side, work, next, next_body, done] = [(); 7].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), outer, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), next, EdgeKind::ConditionalFalse);
    cfg.add_edge(outer, inner, EdgeKind::ConditionalTrue);
    cfg.add_edge(outer, side, EdgeKind::SwitchCase);
    cfg.add_edge(outer, next, EdgeKind::ConditionalFalse);
    cfg.add_edge(side, inner, EdgeKind::Fallthrough);
    cfg.add_edge(inner, work, EdgeKind::ConditionalTrue);
    cfg.add_edge(work, inner, EdgeKind::Back);
    cfg.add_edge(inner, outer, EdgeKind::Back);
    cfg.add_edge(next, next_body, EdgeKind::ConditionalTrue);
    cfg.add_edge(next_body, next, EdgeKind::Back);
    cfg.add_edge(next, done, EdgeKind::ConditionalFalse);
    let dom = DominatorTree::compute(&cfg);

    let canonical = canonicalize_loops(&mut cfg, &dom);

    // What canonicalization reports is what detection finds afterwards.
    let dom = DominatorTree::compute(&cfg);
    let detected = detect_loops_tagged(&cfg, &dom);
    assert_eq!(canonical.len(), detected.len());
    for report in &canonical {
        let now = detected
            .iter()
            .find(|lp| lp.header == report.natural_loop.header)
            .expect("the loop is still there");
        assert_eq!(
            report.natural_loop.body, now.body,
            "body of {:?}",
            now.header
        );
        assert_eq!(
            report.exits,
            loop_exit_blocks(&cfg, now),
            "exits of {:?}",
            now.header
        );
        let preheader = report
            .preheader
            .expect("every header here is entered from outside");
        assert_eq!(cfg.successors(preheader).collect::<Vec<_>>(), [now.header]);
    }
    assert!(verify(&cfg).is_ok());
}
