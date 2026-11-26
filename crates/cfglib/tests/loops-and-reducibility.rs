//! Integration tests: natural loops, nesting, reducibility and loop
//! canonicalization, through the public facade only.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, CfgMetrics, DominatorTree, EdgeKind, GraphMetrics, LoopForest, NodeOrder,
    canonicalize_loops, find_back_edges, insert_preheader, is_reducible, make_reducible, verify,
};

fn set(blocks: impl IntoIterator<Item = BlockId>) -> BTreeSet<BlockId> {
    blocks.into_iter().collect()
}

/// The loops of `cfg` as it is now.
fn loops_of<E>(cfg: &Cfg<u32, E>) -> LoopForest {
    LoopForest::compute(cfg, &DominatorTree::compute(cfg))
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

    let loops = loops_of(&cfg);
    assert_eq!(loops.loop_count(), 1);
    let lp = loops.loop_of_header(header).expect("header heads the loop");
    assert_eq!(loops.body(lp).collect::<BTreeSet<_>>(), set([header, body]));

    let depths = loops.node_depths();
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
    assert_eq!(find_back_edges(&cfg, &dom), Vec::new());
    let loops = LoopForest::compute(&cfg, &dom);
    assert!(loops.is_empty());
    assert_eq!(loops.node_depths(), [0; 4]);
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
    assert!(!dom.dominates(h, y), "dominance cannot see the loop");
    let loops = LoopForest::compute(&cfg, &dom);
    assert_eq!(loops.loop_count(), 1);
    let lp = loops.loop_of_header(h).expect("the tag makes `h` a header");
    assert_eq!(loops.latches(lp), [y]);
    assert_eq!(loops.body(lp).collect::<BTreeSet<_>>(), set([h, x, y]));

    let depths = loops.node_depths();
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
    let loops = loops_of(&cfg);
    assert_eq!(
        loops.loops().map(|lp| loops.header(lp)).collect::<Vec<_>>(),
        [outer, inner],
        "by header"
    );

    let [outer_loop, inner_loop] =
        [outer, inner].map(|header| loops.loop_of_header(header).expect("a header"));
    assert_eq!(loops.parent(outer_loop), None);
    assert_eq!(loops.parent(inner_loop), Some(outer_loop));
    assert_eq!([outer_loop, inner_loop].map(|lp| loops.depth(lp)), [0, 1]);
    assert_eq!(loops.innermost(body), Some(inner_loop));
    assert_eq!(loops.innermost(outer), Some(outer_loop));
    assert_eq!(loops.innermost(exit), None);
    assert_eq!(loops.innermost(inner), Some(inner_loop));
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

    let loops = loops_of(&cfg);
    assert_eq!(loops.loop_count(), 2);
    let [first, second] =
        [b1, b2].map(|header| loops.loop_of_header(header).expect("a tagged header"));
    assert_eq!(
        loops.body(first).collect::<BTreeSet<_>>(),
        loops.body(second).collect::<BTreeSet<_>>()
    );
    assert_eq!(
        [first, second].map(|lp| loops.parent(lp)),
        [None, None],
        "neither loop contains the other"
    );
    assert!(loops.loops().all(|lp| loops.depth(lp) == 0));
    assert_eq!(loops.node_depths(), [0, 1, 1]);
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

    let depths = loops_of(&cfg).node_depths();
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
    assert_eq!(before.graph.reachable_node_count, before.graph.node_count);

    cfg.compact(NodeOrder::Insertion);
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
    let loops = loops_of(&cfg);
    let lp = loops.loop_of_header(header).expect("header heads the loop");
    let preheader = insert_preheader(&mut cfg, &loops, lp).expect("two entries need a preheader");

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
    assert_eq!(canonical[0].exits, [exit]);
    assert!(verify(&cfg).is_ok());

    // Canonical form is stable: the same loop, and nothing left to insert.
    let dom = DominatorTree::compute(&cfg);
    let again = canonicalize_loops(&mut cfg, &dom);
    assert_eq!(again.len(), 1);
    assert_eq!(again[0].preheader, Some(preheader));
    assert_eq!(again[0], canonical[0]);
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
fn canonicalized_nested_loops_report_current_exits() {
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

    // What canonicalization reports is what the loops found afterwards say.
    let loops = loops_of(&cfg);
    assert_eq!(canonical.len(), loops.loop_count());
    for report in &canonical {
        let now = loops
            .loop_of_header(report.header)
            .expect("the loop is still there");
        assert_eq!(
            report.exits,
            loops.exits(&cfg, now),
            "exits of {:?}",
            report.header
        );
        let preheader = report
            .preheader
            .expect("every header here is entered from outside");
        assert_eq!(
            cfg.successors(preheader).collect::<Vec<_>>(),
            [report.header]
        );
        assert!(!loops.contains(now, preheader));
    }
    assert!(verify(&cfg).is_ok());
}

#[test]
fn splitting_makes_an_irreducible_cfg_reducible() {
    // entry -> {a, b}, a <-> b: a cycle with two entries.
    let mut cfg = Cfg::<u32>::new();
    let [a, b] = [(); 2].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
    cfg.add_edge(a, b, EdgeKind::Fallthrough);
    cfg.add_edge(b, a, EdgeKind::Fallthrough);
    assert!(!is_reducible(&cfg, &DominatorTree::compute(&cfg)));

    let copies = make_reducible(&mut cfg);

    assert_ne!(copies.len(), 0);
    assert!(is_reducible(&cfg, &DominatorTree::compute(&cfg)));
    assert!(verify(&cfg).is_ok());
}

#[test]
fn a_back_tag_that_closes_no_cycle_is_not_a_loop() {
    // `entry -Back-> sink`, and the sink returns.
    let mut cfg = Cfg::<u32>::new();
    let sink = cfg.new_block();
    cfg.add_edge(cfg.entry(), sink, EdgeKind::Back);
    let dom = DominatorTree::compute(&cfg);

    let loops = LoopForest::compute(&cfg, &dom);
    assert!(loops.is_empty());
    assert_eq!(loops.node_depths(), [0, 0]);
    assert_eq!(CfgMetrics::compute(&cfg).graph.max_nesting_depth, 0);
}

/// `entry -> {r, q}`, the cycle `q -> w -> s -> {w, q}`, and `r -> s`: a cycle
/// with two entries (`q` from the entry, `s` from `r`) whose first retreating
/// edge ends at `w`, in its interior. Splitting one block at a time never
/// finished on it.
#[test]
fn make_reducible_finishes_on_a_cycle_whose_witness_is_not_an_entry() {
    let mut cfg = Cfg::<u32, Route>::with_edge_payload();
    let entry = cfg.entry();
    let [q, w, s, r] = [(); 4].map(|()| cfg.new_block());
    cfg.add_edge_with_payload(entry, r, EdgeKind::ConditionalTrue, Route::Hot);
    cfg.add_edge_with_payload(entry, q, EdgeKind::ConditionalFalse, Route::Cold);
    cfg.add_edge(q, w, EdgeKind::Fallthrough);
    cfg.add_edge(w, s, EdgeKind::Fallthrough);
    cfg.add_edge(s, w, EdgeKind::Back);
    cfg.add_edge(s, q, EdgeKind::Back);
    cfg.add_edge(r, s, EdgeKind::Fallthrough);
    assert!(!is_reducible(&cfg, &DominatorTree::compute(&cfg)));
    let blocks_before = cfg.block_count();

    let copies = make_reducible(&mut cfg);

    assert_ne!(copies.len(), 0);
    assert_eq!(cfg.block_count(), blocks_before + copies.len());
    assert!(is_reducible(&cfg, &DominatorTree::compute(&cfg)));
    assert!(verify(&cfg).is_ok());
    // The two edges out of the entry keep their payloads and order.
    let from_entry: Vec<Route> = cfg
        .outgoing(entry)
        .map(|edge| *cfg.edge(edge).payload())
        .collect();
    assert_eq!(from_entry, [Route::Hot, Route::Cold]);
}

/// `entry -> outer`; `outer -> middle` or `exit`; `middle -> inner` or
/// `middle_latch`; `inner <-> inner_body`; the latches return to their headers;
/// and a second loop beside the first, entered from the exit.
fn nested_and_sibling() -> (Cfg<u32>, [BlockId; 8]) {
    let mut cfg = Cfg::<u32>::new();
    let [
        outer,
        middle,
        inner,
        inner_body,
        middle_latch,
        exit,
        again,
        again_body,
    ] = [(); 8].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), outer, EdgeKind::Fallthrough);
    cfg.add_edge(outer, middle, EdgeKind::ConditionalTrue);
    cfg.add_edge(outer, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(middle, inner, EdgeKind::ConditionalTrue);
    cfg.add_edge(middle, middle_latch, EdgeKind::ConditionalFalse);
    cfg.add_edge(inner, inner_body, EdgeKind::ConditionalTrue);
    cfg.add_edge(inner_body, inner, EdgeKind::Back);
    cfg.add_edge(inner, middle_latch, EdgeKind::ConditionalFalse);
    cfg.add_edge(middle_latch, middle, EdgeKind::Back);
    cfg.add_edge(middle_latch, outer, EdgeKind::Back);
    cfg.add_edge(exit, again, EdgeKind::Fallthrough);
    cfg.add_edge(again, again_body, EdgeKind::ConditionalTrue);
    cfg.add_edge(again_body, again, EdgeKind::Back);
    (
        cfg,
        [
            outer,
            middle,
            inner,
            inner_body,
            middle_latch,
            exit,
            again,
            again_body,
        ],
    )
}

#[test]
fn the_forest_states_the_nesting_rule() {
    let (
        cfg,
        [
            outer,
            middle,
            inner,
            inner_body,
            middle_latch,
            exit,
            again,
            again_body,
        ],
    ) = nested_and_sibling();
    let dom = DominatorTree::compute(&cfg);

    let forest = LoopForest::compute(&cfg, &dom);

    assert_eq!(forest.loop_count(), 4);
    let [outer_loop, middle_loop, inner_loop, again_loop] =
        [outer, middle, inner, again].map(|header| {
            forest
                .loop_of_header(header)
                .expect("every header heads a loop")
        });
    assert_eq!(forest.loop_of_header(exit), None);
    assert_eq!(forest.parent(outer_loop), None);
    assert_eq!(forest.parent(middle_loop), Some(outer_loop));
    assert_eq!(forest.parent(inner_loop), Some(middle_loop));
    assert_eq!(forest.parent(again_loop), None);
    assert_eq!(
        [outer_loop, middle_loop, inner_loop, again_loop].map(|lp| forest.depth(lp)),
        [0, 1, 2, 0]
    );
    assert_eq!(forest.max_nesting_depth(), 3);
    assert_eq!(forest.latches(inner_loop), [inner_body]);
    assert_eq!(forest.latches(outer_loop), [middle_latch]);

    assert_eq!(
        forest.body(middle_loop).collect::<BTreeSet<_>>(),
        set([middle, inner, inner_body, middle_latch])
    );
    assert_eq!(forest.body_count(outer_loop), 5);
    assert!(forest.contains(outer_loop, inner_body));
    assert!(!forest.contains(inner_loop, middle));
    assert!(!forest.contains(outer_loop, exit));
    assert!(!forest.contains(again_loop, outer));
    assert!(forest.contains(again_loop, again_body));
    assert!(forest.encloses(outer_loop, middle_loop));
    assert!(forest.encloses(outer_loop, inner_loop));
    assert!(
        forest.encloses(middle_loop, middle_loop),
        "a loop encloses itself"
    );
    assert!(!forest.encloses(inner_loop, middle_loop));
    assert!(!forest.encloses(outer_loop, again_loop));
    assert!(!forest.encloses(again_loop, outer_loop));
    let depths = forest.node_depths();
    assert_eq!(depths.len(), cfg.block_ids().count());
    assert_eq!(depths[inner_body.index()], 3);
    assert_eq!(depths[middle_latch.index()], 2);
    assert_eq!(depths[exit.index()], 0);
}

#[test]
fn a_tag_names_a_loop_dominance_cannot_prove() {
    // Two entries into one cycle: dominance finds no loop, and the tag names one.
    let mut cfg = Cfg::<u32>::new();
    let [left, right] = [(); 2].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, right, EdgeKind::Fallthrough);
    cfg.add_edge(right, left, EdgeKind::Back);
    let dom = DominatorTree::compute(&cfg);
    assert!(!dom.dominates(left, right) && !dom.dominates(right, left));

    let tagged = LoopForest::compute(&cfg, &dom);

    assert_eq!(tagged.loop_count(), 1);
    let cycle = tagged.loop_of_header(left).expect("the tag names the loop");
    assert_eq!(
        tagged.body(cycle).collect::<BTreeSet<_>>(),
        set([left, right])
    );
    assert!(tagged.contains(cycle, right));
    assert!(!tagged.contains(cycle, cfg.entry()));
}

/// `depth` loops, each inside the one before, found, queried and measured.
/// The flat loops of this graph hold `depth * (depth + 1) / 2` set entries, five
/// billion at this size, so only the forest can answer for it.
#[test]
fn a_hundred_thousand_nested_loops_are_found_and_measured() {
    let depth = 100_000_usize;
    let mut cfg = Cfg::<u32>::new();
    let headers: Vec<BlockId> = std::iter::once(cfg.entry())
        .chain((1..depth).map(|_| cfg.new_block()))
        .collect();
    let latches: Vec<BlockId> = (0..depth).map(|_| cfg.new_block()).collect();
    let exit = cfg.new_block();
    for index in 0..depth {
        let inward = headers.get(index + 1).copied().unwrap_or(latches[index]);
        cfg.add_edge(headers[index], inward, EdgeKind::ConditionalTrue);
        cfg.add_edge(headers[index], exit, EdgeKind::ConditionalFalse);
        if index + 1 < depth {
            cfg.add_edge(latches[index + 1], latches[index], EdgeKind::Fallthrough);
        }
        cfg.add_edge(latches[index], headers[index], EdgeKind::Back);
    }
    let dom = DominatorTree::compute(&cfg);

    let forest = LoopForest::compute(&cfg, &dom);

    assert_eq!(forest.loop_count(), depth);
    assert_eq!(forest.max_nesting_depth(), depth);
    let outermost = forest
        .loop_of_header(headers[0])
        .expect("the entry heads a loop");
    let innermost = forest
        .loop_of_header(headers[depth - 1])
        .expect("the last header heads a loop");
    assert_eq!(forest.depth(innermost), depth - 1);
    assert_eq!(forest.body_count(outermost), 2 * depth);
    assert!(forest.contains(outermost, latches[depth - 1]));
    assert!(forest.encloses(outermost, innermost));
    assert!(!forest.encloses(innermost, outermost));
    assert!(!forest.contains(outermost, exit));

    // The metrics read the forest too, so they scale with it.
    assert_eq!(CfgMetrics::compute(&cfg).graph.max_nesting_depth, depth);
    let depths = forest.node_depths();
    assert_eq!(depths[headers[depth - 1].index()], depth);
    assert_eq!(depths[exit.index()], 0);
}
