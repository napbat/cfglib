//! Integration tests: natural loops, nesting, reducibility and loop
//! canonicalization, through the public facade only.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, CfgMetrics, DominatorTree, EdgeKind, GraphMetrics, LoopNestingTree,
    block_nesting_depths, cfg_block_nesting_depths, detect_loops, detect_loops_tagged,
    find_back_edges,
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
