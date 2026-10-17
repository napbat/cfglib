//! Integration tests: dominance frontiers, iterated frontiers, and the
//! control-dependence frontier of the reverse graph.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, DominanceFrontiers, DominatorTree, EdgeKind, FrontierScratch, NodeId,
    PostDominatorTree, control_dependence_graph,
};

/// `entry -> head`, a diamond `head -> {left, right} -> join`, the back edge
/// `join -> head`, and the exit `head -> exit`.
struct DiamondLoop {
    cfg: Cfg<u32>,
    head: BlockId,
    left: BlockId,
    right: BlockId,
    join: BlockId,
    exit: BlockId,
}

fn diamond_loop() -> DiamondLoop {
    let mut cfg = Cfg::<u32>::new();
    let entry = cfg.entry();
    let [head, left, right, join, exit] = [(); 5].map(|()| cfg.new_block());
    cfg.add_edge(entry, head, EdgeKind::Fallthrough);
    cfg.add_edge(head, left, EdgeKind::ConditionalTrue);
    cfg.add_edge(head, right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, join, EdgeKind::Fallthrough);
    cfg.add_edge(right, join, EdgeKind::Fallthrough);
    cfg.add_edge(join, head, EdgeKind::Back);
    cfg.add_edge(head, exit, EdgeKind::SwitchCase);
    DiamondLoop {
        cfg,
        head,
        left,
        right,
        join,
        exit,
    }
}

fn per_query(dominators: &DominatorTree, cfg: &Cfg<u32>, block: BlockId) -> BTreeSet<BlockId> {
    let mut out = Vec::new();
    dominators.frontier(cfg, block, &mut out);
    out.into_iter().collect()
}

#[test]
fn the_table_and_the_per_query_frontier_state_one_rule() {
    let DiamondLoop {
        cfg,
        head,
        left,
        right,
        join,
        exit,
    } = diamond_loop();
    let dominators = DominatorTree::compute(&cfg);
    let table = DominanceFrontiers::compute(&cfg, &dominators);

    for block in cfg.block_ids() {
        assert_eq!(
            table
                .frontier(block)
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
            per_query(&dominators, &cfg, block),
            "frontier of {block:?}"
        );
    }
    assert_eq!(table.frontier(left), [join]);
    assert_eq!(table.frontier(right), [join]);
    assert_eq!(table.frontier(join), [head]);
    assert_eq!(
        table.frontier(head),
        [head],
        "a loop header is its own frontier"
    );
    assert!(table.frontier(exit).is_empty());
}

#[test]
fn an_iterated_frontier_is_the_closure_of_the_table() {
    let DiamondLoop {
        cfg,
        head,
        left,
        join,
        ..
    } = diamond_loop();
    let dominators = DominatorTree::compute(&cfg);
    let table = DominanceFrontiers::compute(&cfg, &dominators);

    let mut closure: BTreeSet<BlockId> = BTreeSet::new();
    let mut pending = vec![left];
    while let Some(block) = pending.pop() {
        for &next in table.frontier(block) {
            if closure.insert(next) {
                pending.push(next);
            }
        }
    }

    let mut scratch = FrontierScratch::new();
    let mut placed = Vec::new();
    dominators.iterated_frontier(&cfg, &mut scratch, [left], None, &mut placed);
    assert_eq!(placed.iter().copied().collect::<BTreeSet<_>>(), closure);
    assert_eq!(closure, BTreeSet::from([join, head]));

    let only_the_header = |block: BlockId| block == head;
    dominators.iterated_frontier(
        &cfg,
        &mut scratch,
        [left],
        Some(&only_the_header),
        &mut placed,
    );
    assert!(
        placed.is_empty(),
        "a rejected join is neither placed nor explored, so the header stays out"
    );
}

/// `entry -> merge`, `entry -> arm -> merge`, plus two dead blocks: one that
/// joins live code at `merge`, one that branches into the entry.
#[test]
fn dead_code_has_no_frontier_and_adds_none_to_live_code() {
    let mut cfg = Cfg::<u32>::new();
    let entry = cfg.entry();
    let [arm, merge, dead_into_merge, dead_into_entry] = [(); 4].map(|()| cfg.new_block());
    cfg.add_edge(entry, merge, EdgeKind::ConditionalFalse);
    cfg.add_edge(entry, arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(arm, merge, EdgeKind::Fallthrough);
    cfg.add_edge(dead_into_merge, merge, EdgeKind::Fallthrough);
    cfg.add_edge(dead_into_entry, entry, EdgeKind::Fallthrough);

    let dominators = DominatorTree::compute(&cfg);
    let table = DominanceFrontiers::compute(&cfg, &dominators);
    for block in cfg.block_ids() {
        assert_eq!(
            table
                .frontier(block)
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
            per_query(&dominators, &cfg, block),
            "frontier of {block:?}"
        );
    }
    assert_eq!(table.frontier(arm), [merge]);
    assert_eq!(table.frontier(entry), []);
    assert_eq!(table.frontier(dead_into_merge), []);
    assert_eq!(table.frontier(dead_into_entry), []);
}

#[test]
fn a_frontier_scratch_gives_the_answers_of_a_fresh_one_across_graphs() {
    let DiamondLoop {
        cfg: large, left, ..
    } = diamond_loop();
    let mut small = Cfg::<u32>::new();
    let only_arm = small.new_block();
    let small_merge = small.new_block();
    small.add_edge(small.entry(), only_arm, EdgeKind::ConditionalTrue);
    small.add_edge(small.entry(), small_merge, EdgeKind::ConditionalFalse);
    small.add_edge(only_arm, small_merge, EdgeKind::Fallthrough);

    let mut shared = FrontierScratch::new();
    for _ in 0..2 {
        for (cfg, definition) in [(&large, left), (&small, only_arm), (&large, left)] {
            let dominators = DominatorTree::compute(cfg);
            let (mut fresh, mut reused) = (Vec::new(), Vec::new());
            dominators.iterated_frontier(
                cfg,
                &mut FrontierScratch::new(),
                [definition],
                None,
                &mut fresh,
            );
            dominators.iterated_frontier(cfg, &mut shared, [definition], None, &mut reused);
            assert_eq!(reused, fresh);
            assert!(!reused.is_empty());
        }
    }
}

#[test]
fn the_post_dominance_frontier_lists_the_branches_that_decide_a_block() {
    let DiamondLoop {
        cfg,
        head,
        left,
        right,
        join,
        exit,
    } = diamond_loop();
    let post = PostDominatorTree::compute(&cfg);
    let dependences = control_dependence_graph(&cfg, &post);

    for block in cfg.block_ids() {
        let mut sources = Vec::new();
        post.frontier(&cfg, block, &mut sources);
        let from_graph: BTreeSet<BlockId> = dependences
            .predecessors(NodeId::from_index(block.index()))
            .map(|controller| dependences[controller])
            .collect();
        assert_eq!(
            sources.iter().copied().collect::<BTreeSet<_>>(),
            from_graph,
            "control-dependence sources of {block:?}"
        );
    }

    let mut sources = Vec::new();
    post.frontier(&cfg, left, &mut sources);
    assert_eq!(sources, [head]);
    post.frontier(&cfg, right, &mut sources);
    assert_eq!(sources, [head]);
    post.frontier(&cfg, exit, &mut sources);
    assert!(
        sources.is_empty(),
        "the only exit runs whatever the header decides"
    );
    post.frontier(&cfg, join, &mut sources);
    assert_eq!(
        sources,
        [head],
        "the loop keeps running until the header exits"
    );

    let mut decided = Vec::new();
    post.iterated_frontier(
        &cfg,
        &mut FrontierScratch::new(),
        [left, right],
        None,
        &mut decided,
    );
    assert_eq!(decided, [head]);
}
