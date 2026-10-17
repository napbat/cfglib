//! Integration tests: natural loops, nesting, reducibility and loop
//! canonicalization, through the public facade only.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, DominatorTree, EdgeKind, block_nesting_depths, cfg_block_nesting_depths,
    detect_loops, detect_loops_tagged, find_back_edges,
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
