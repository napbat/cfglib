//! Integration tests: natural loops, nesting, reducibility and loop
//! canonicalization, through the public facade only.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, CfgMetrics, DominatorTree, EdgeKind, GraphMetrics, IntervalAnalysis,
    LoopNestingTree, block_nesting_depths, canonicalize_loops, cfg_block_nesting_depths,
    detect_loops, detect_loops_tagged, find_back_edges, insert_preheader, is_reducible,
    loop_exit_blocks, make_reducible, verify,
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
    assert_eq!(find_back_edges(&cfg, &dom), Vec::new());
    assert_eq!(detect_loops_tagged(&cfg, &dom), Vec::new());
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
    assert_eq!(detect_loops(&cfg, &dom), Vec::new());
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

#[test]
fn intervals_partition_the_blocks_the_entry_reaches() {
    // entry -> header <-> body -> exit, and a dead block into the exit.
    let mut cfg = Cfg::<u32>::new();
    let [header, body, exit, dead] = [(); 4].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, header, EdgeKind::Back);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(dead, exit, EdgeKind::Fallthrough);

    let analysis = IntervalAnalysis::compute(&cfg);

    let covered: BTreeSet<BlockId> = analysis
        .intervals
        .iter()
        .flat_map(|interval| interval.blocks.iter().copied())
        .collect();
    assert_eq!(covered, set([cfg.entry(), header, body, exit]));
    assert_eq!(
        analysis
            .intervals
            .iter()
            .map(|i| i.blocks.len())
            .sum::<usize>(),
        covered.len(),
        "no block is in two intervals"
    );
    assert_eq!(analysis.intervals[0].header, cfg.entry());
}

#[test]
fn a_dead_block_does_not_split_an_interval() {
    let mut cfg = Cfg::<u32>::new();
    let [join, dead] = [(); 2].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), join, EdgeKind::Fallthrough);
    cfg.add_edge(dead, join, EdgeKind::Fallthrough);

    let analysis = IntervalAnalysis::compute(&cfg);

    assert_eq!(analysis.intervals.len(), 1);
    assert_eq!(analysis.intervals[0].blocks, set([cfg.entry(), join]));
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

    let split = make_reducible(&mut cfg);

    assert!(split > 0);
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

    assert_eq!(detect_loops_tagged(&cfg, &dom), Vec::new());
    assert_eq!(cfg_block_nesting_depths(&cfg), [0, 0]);
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

    let duplicated = make_reducible(&mut cfg);

    assert!(duplicated > 0);
    assert_eq!(cfg.block_count(), blocks_before + duplicated);
    assert!(is_reducible(&cfg, &DominatorTree::compute(&cfg)));
    assert!(verify(&cfg).is_ok());
    // The two edges out of the entry keep their payloads and order.
    let from_entry: Vec<Route> = cfg
        .outgoing(entry)
        .map(|edge| *cfg.edge(edge).payload())
        .collect();
    assert_eq!(from_entry, [Route::Hot, Route::Cold]);
}

use cfglib::LoopForest;

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
fn the_forest_states_what_the_flat_loops_and_the_nesting_tree_state() {
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

    let forest = LoopForest::compute_tagged(&cfg, &dom);

    assert_eq!(forest.natural_loops(), detect_loops_tagged(&cfg, &dom));
    assert_eq!(forest.len(), 4);
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
    assert_eq!(forest.body_len(outer_loop), 5);
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

    // The nesting tree of the flat loops agrees loop for loop.
    let flat = forest.natural_loops();
    let tree = LoopNestingTree::compute(&flat);
    for (position, lp) in flat.iter().enumerate() {
        let id = forest
            .loop_of_header(lp.header)
            .expect("a loop of the forest");
        assert_eq!(tree.depth(position), forest.depth(id));
        assert_eq!(
            tree.nodes[position]
                .parent
                .map(|parent| flat[parent].header),
            forest.parent(id).map(|parent| forest.header(parent)),
        );
    }

    let depths = forest.node_depths();
    assert_eq!(depths.len(), cfg.block_ids().count());
    assert_eq!(depths, cfg_block_nesting_depths(&cfg));
    assert_eq!(depths[inner_body.index()], 3);
    assert_eq!(depths[middle_latch.index()], 2);
    assert_eq!(depths[exit.index()], 0);
}

#[test]
fn the_untagged_forest_misses_what_dominance_cannot_prove_like_detect_loops() {
    // Two entries into one cycle: dominance finds no loop, and the tag names one.
    let mut cfg = Cfg::<u32>::new();
    let [left, right] = [(); 2].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, right, EdgeKind::Fallthrough);
    cfg.add_edge(right, left, EdgeKind::Back);
    let dom = DominatorTree::compute(&cfg);

    let untagged = LoopForest::compute(&cfg, &dom);
    let tagged = LoopForest::compute_tagged(&cfg, &dom);

    assert!(untagged.is_empty());
    assert_eq!(untagged.natural_loops(), detect_loops(&cfg, &dom));
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

    let forest = LoopForest::compute_tagged(&cfg, &dom);

    assert_eq!(forest.len(), depth);
    assert_eq!(forest.max_nesting_depth(), depth);
    let outermost = forest
        .loop_of_header(headers[0])
        .expect("the entry heads a loop");
    let innermost = forest
        .loop_of_header(headers[depth - 1])
        .expect("the last header heads a loop");
    assert_eq!(forest.depth(innermost), depth - 1);
    assert_eq!(forest.body_len(outermost), 2 * depth);
    assert!(forest.contains(outermost, latches[depth - 1]));
    assert!(forest.encloses(outermost, innermost));
    assert!(!forest.encloses(innermost, outermost));
    assert!(!forest.contains(outermost, exit));

    // The metrics read the forest too, so they scale with it.
    assert_eq!(CfgMetrics::compute(&cfg).graph.max_nesting_depth, depth);
    let depths = cfg_block_nesting_depths(&cfg);
    assert_eq!(depths[headers[depth - 1].index()], depth);
    assert_eq!(depths[exit.index()], 0);
}
