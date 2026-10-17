//! Integration tests: the loop forest answers what the flat loops do, and at
//! nesting depths the flat loops cannot be built at, through the public facade
//! only.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, CfgMetrics, DominatorTree, EdgeKind, LoopForest, LoopNestingTree,
    cfg_block_nesting_depths, detect_loops, detect_loops_tagged,
};

fn set(blocks: impl IntoIterator<Item = BlockId>) -> BTreeSet<BlockId> {
    blocks.into_iter().collect()
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
    assert!(!forest.contains(outermost, exit));

    // The metrics read the forest too, so they scale with it.
    assert_eq!(CfgMetrics::compute(&cfg).graph.max_nesting_depth, depth);
    let depths = cfg_block_nesting_depths(&cfg);
    assert_eq!(depths[headers[depth - 1].index()], depth);
    assert_eq!(depths[exit.index()], 0);
}
