//! Integration tests: dominator-tree queries and recomputation after an edit.

use cfglib::{BlockId, Cfg, DominatorTree, EdgeKind};

/// `entry -> head`, `head -> {left, right} -> join`, `join -> head`, and
/// `head -> exit`.
fn diamond_loop() -> (Cfg<u32>, [BlockId; 6]) {
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
    (cfg, [entry, head, left, right, join, exit])
}

#[test]
fn the_tree_answers_ancestry_levels_and_common_dominators() {
    let (cfg, [entry, head, left, right, join, exit]) = diamond_loop();
    let dom = DominatorTree::compute(&cfg);

    assert_eq!(dom.idom(entry), None);
    assert_eq!(dom.idom(join), Some(head), "neither arm dominates the join");
    assert!(dom.dominates(head, join));
    assert!(dom.strictly_dominates(entry, exit));
    assert!(!dom.dominates(left, join));
    assert_eq!(dom.level(entry), Some(0));
    assert_eq!(dom.level(join), Some(2));

    assert_eq!(dom.nearest_common_dominator(left, right), Some(head));
    assert_eq!(dom.nearest_common_dominator(left, left), Some(left));
    assert_eq!(
        dom.nearest_common_dominator_of([left, right, exit]),
        Some(head)
    );
    assert_eq!(dom.nearest_common_dominator_of([join]), Some(join));

    assert_eq!(dom.preorder()[0], entry);
    assert_eq!(dom.subtree(head).len(), cfg.block_count() - 1);
    assert_eq!(
        dom.children(head).collect::<Vec<_>>(),
        [left, right, join, exit],
        "children come in ascending index order"
    );
}

#[test]
fn unreachable_blocks_are_in_no_tree() {
    let (mut cfg, [_, head, left, ..]) = diamond_loop();
    let dead = cfg.new_block();
    cfg.add_edge(dead, left, EdgeKind::Fallthrough);

    let dom = DominatorTree::compute(&cfg);

    assert!(!dom.is_reachable(dead));
    assert_eq!(dom.idom(dead), None);
    assert_eq!(dom.level(dead), None);
    assert_eq!(dom.nearest_common_dominator(dead, head), None);
    assert_eq!(dom.nearest_common_dominator_of([dead]), None);
    assert_eq!(dom.subtree(dead), []);
    assert_eq!(
        dom.idom(left),
        Some(head),
        "an edge from dead code adds no path to the arm"
    );
}

#[test]
fn recomputing_after_an_edit_reports_the_blocks_whose_idom_changed() {
    let (mut cfg, [entry, head, left, right, join, exit]) = diamond_loop();
    let before = DominatorTree::compute(&cfg);

    // A second way into the join from the entry takes it out from under the
    // head: the head no longer dominates the join.
    cfg.add_edge(entry, join, EdgeKind::SwitchCase);
    let (after, changed) = DominatorTree::compute_with_diff(&cfg, &before);

    assert_eq!(after, DominatorTree::compute(&cfg));
    assert_eq!(after.idom(join), Some(entry));
    assert_eq!(changed, [join]);
    for unchanged in [head, left, right, exit] {
        assert_eq!(after.idom(unchanged), before.idom(unchanged));
    }
}
