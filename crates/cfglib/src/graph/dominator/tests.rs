//! Dominator and post-dominator trees over graphs, CFGs, and
//! consumer-defined views.

use super::*;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::store::{Graph, NodeId};
use crate::test_util::MockInst;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct BoundedNode(u8);

impl DenseId for BoundedNode {
    fn from_index(index: usize) -> Self {
        assert!(index < 4, "bounded ID cannot represent a synthetic node");
        Self(u8::try_from(index).expect("test node index fits in u8"))
    }

    fn index(self) -> usize {
        usize::from(self.0)
    }
}

struct BoundedDiamond;

impl GraphView for BoundedDiamond {
    type NodeId = BoundedNode;

    fn node_bound(&self) -> usize {
        4
    }

    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
        (0..4).map(BoundedNode)
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        const EMPTY: &[u8] = &[];
        const ENTRY: &[u8] = &[1, 2];
        const TO_EXIT: &[u8] = &[3];
        let successors = match node.0 {
            0 => ENTRY,
            1 | 2 => TO_EXIT,
            _ => EMPTY,
        };
        successors.iter().copied().map(BoundedNode)
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        const EMPTY: &[u8] = &[];
        const FROM_ENTRY: &[u8] = &[0];
        const MERGE: &[u8] = &[1, 2];
        let predecessors = match node.0 {
            1 | 2 => FROM_ENTRY,
            3 => MERGE,
            _ => EMPTY,
        };
        predecessors.iter().copied().map(BoundedNode)
    }
}

#[test]
fn single_block_cfg() {
    let cfg: Cfg<MockInst> = Cfg::new();
    let dom = DominatorTree::compute(&cfg);
    assert_eq!(dom.idom(cfg.entry()), None);
    assert!(dom.dominates(cfg.entry(), cfg.entry()));
    assert_eq!(dom.children(cfg.entry()).len(), 0);
}

#[test]
fn linear_chain_dominance() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let b1 = cfg.new_block();
    let b2 = cfg.new_block();
    cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
    cfg.add_edge(b1, b2, EdgeKind::Fallthrough);
    let dom = DominatorTree::compute(&cfg);
    assert!(dom.dominates(cfg.entry(), b1));
    assert!(dom.dominates(cfg.entry(), b2));
    assert!(dom.dominates(b1, b2));
    assert!(!dom.dominates(b2, b1));
    assert_eq!(dom.idom(b1), Some(cfg.entry()));
    assert_eq!(dom.idom(b2), Some(b1));
}

#[test]
fn diamond_idom_at_merge() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let a = cfg.new_block();
    let b = cfg.new_block();
    let merge = cfg.new_block();
    cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
    cfg.add_edge(a, merge, EdgeKind::Fallthrough);
    cfg.add_edge(b, merge, EdgeKind::Fallthrough);
    let dom = DominatorTree::compute(&cfg);
    // Merge block's idom should be entry (not a or b).
    assert_eq!(dom.idom(merge), Some(cfg.entry()));
    assert!(dom.dominates(cfg.entry(), a));
    assert!(dom.dominates(cfg.entry(), b));
    assert!(!dom.dominates(a, b));
    assert!(!dom.dominates(b, a));
}

#[test]
fn self_loop_dominance() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    cfg.add_edge(cfg.entry(), cfg.entry(), EdgeKind::Back);
    let dom = DominatorTree::compute(&cfg);
    assert_eq!(dom.idom(cfg.entry()), None);
    assert!(dom.dominates(cfg.entry(), cfg.entry()));
}

#[test]
fn unreachable_block_not_dominated() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let unreachable = cfg.new_block();
    let dom = DominatorTree::compute(&cfg);
    // Entry still dominates itself.
    assert!(dom.dominates(cfg.entry(), cfg.entry()));
    // Unreachable block has no idom.
    assert_eq!(dom.idom(unreachable), None);
}

#[test]
fn depth_computation() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let b1 = cfg.new_block();
    let b2 = cfg.new_block();
    cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
    cfg.add_edge(b1, b2, EdgeKind::Fallthrough);
    let dom = DominatorTree::compute(&cfg);
    assert_eq!(dom.depth(cfg.entry()), Some(0));
    assert_eq!(dom.depth(b1), Some(1));
    assert_eq!(dom.depth(b2), Some(2));
}

#[test]
fn depth_tables_match_queries_when_parents_have_larger_ids() {
    let mut graph = Graph::<(), ()>::new();
    let leaf = graph.add_node(());
    let middle = graph.add_node(());
    let child = graph.add_node(());
    let root = graph.add_node(());
    let unreachable = graph.add_node(());
    graph.add_edge(root, child, ());
    graph.add_edge(child, middle, ());
    graph.add_edge(middle, leaf, ());

    let dom = DominatorTree::compute_from(&graph, root);
    let depths = dom.depths();
    let AnalysisDepths::Compact(compact_depths) = dom.analysis_depths() else {
        panic!("small test graph should use compact depths");
    };
    for node in [root, child, middle, leaf] {
        let expected = dom.depth(node).expect("reachable node has a depth");
        assert_eq!(depths[node.index()], expected);
        assert_eq!(
            compact_depths[node.index()],
            u32::try_from(expected).expect("test depth fits u32")
        );
    }
    assert_eq!(depths[unreachable.index()], usize::MAX);
    assert_eq!(compact_depths[unreachable.index()], u32::MAX);
}

#[test]
fn full_analysis_depths_match_public_dominance_queries() {
    let mut graph = Graph::<(), ()>::new();
    let root = graph.add_node(());
    let left = graph.add_node(());
    let right = graph.add_node(());
    let merge = graph.add_node(());
    let unreachable = graph.add_node(());
    graph.add_edge(root, left, ());
    graph.add_edge(root, right, ());
    graph.add_edge(left, merge, ());
    graph.add_edge(right, merge, ());

    let dom = DominatorTree::compute_from(&graph, root);
    let full_depths = AnalysisDepths::Full(dom.depths());
    let nodes = [root, left, right, merge, unreachable];
    for dominator in nodes {
        for node in nodes {
            assert_eq!(
                dom.dominates_with_analysis_depths(dominator, node, &full_depths),
                dom.dominates(dominator, node),
                "mismatch for {dominator:?} dominating {node:?}"
            );
        }
    }
}

#[test]
fn child_links_follow_the_selected_sibling_order() {
    let mut graph = Graph::<(), ()>::new();
    let root = graph.add_node(());
    let first = graph.add_node(());
    let second = graph.add_node(());
    let grandchild = graph.add_node(());
    let unreachable = graph.add_node(());
    graph.add_edge(root, first, ());
    graph.add_edge(root, second, ());
    graph.add_edge(first, grandchild, ());

    let dom = DominatorTree::compute_from(&graph, root);
    let ascending = dom.child_links(DominatorChildOrder::Ascending);
    let descending = dom.child_links(DominatorChildOrder::Descending);

    let collect = |links: &DominatorChildLinks<NodeId>, parent| {
        let mut children = Vec::new();
        let mut child = links.first_child(parent);
        while let Some(next) = child {
            children.push(next);
            child = links.next_sibling(next);
        }
        children
    };

    assert_eq!(collect(&ascending, root), vec![first, second]);
    assert_eq!(collect(&descending, root), vec![second, first]);
    assert_eq!(collect(&ascending, first), vec![grandchild]);
    assert_eq!(collect(&ascending, unreachable).len(), 0);
}

#[test]
fn compact_depth_selection_falls_back_before_the_sentinel_can_be_a_depth() {
    let largest_compact = usize::try_from(u32::MAX).expect("u32 fits supported usize targets");
    assert!(compact_depths_supported(largest_compact));
    if let Some(too_large) = largest_compact.checked_add(1) {
        assert!(!compact_depths_supported(too_large));
    }
}

#[test]
fn post_dominators_over_a_consumer_view() {
    // Diamond in consumer storage: a -> {b, c} -> d. Everything is
    // post-dominated by d; the branch is post-dominated by the merge.
    let mut graph = Graph::<&str, ()>::new();
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let c = graph.add_node("c");
    let d = graph.add_node("d");
    graph.add_edge(a, b, ());
    graph.add_edge(a, c, ());
    graph.add_edge(b, d, ());
    graph.add_edge(c, d, ());

    let post = DominatorTree::compute_post_from(&graph, &[d]);
    assert_eq!(post.idom(a), Some(d));
    assert_eq!(post.idom(b), Some(d));
    assert_eq!(post.idom(c), Some(d));
    assert!(post.dominates(d, a), "d post-dominates the entry");

    // No exits: nothing is reachable on the reverse graph.
    let empty = DominatorTree::compute_post_from(&graph, &[]);
    assert_eq!(empty.idom(a), None);
    assert_eq!(empty.depth(a), None);
}

#[test]
#[should_panic(expected = "post-dominator exit index is outside the graph")]
fn post_dominators_reject_an_exit_outside_the_graph() {
    let mut graph = Graph::<(), ()>::new();
    graph.add_node(());

    let outside =
        NodeId::from_raw(u32::try_from(graph.node_bound()).expect("test graph size fits in u32"));
    let _ = DominatorTree::compute_post_from(&graph, &[outside]);
}

#[test]
fn post_dominator_view_preserves_duplicate_exit_edges() {
    let mut graph = Graph::<(), ()>::new();
    let exit = graph.add_node(());
    let exits = [exit, exit];
    let reverse = PostDominatorView {
        graph: &graph,
        exits: &exits,
        binary_search_exits: false,
    };
    let virtual_exit = graph.node_bound();

    assert_eq!(
        reverse.successors(virtual_exit).collect::<Vec<_>>(),
        vec![exit.index(), exit.index()]
    );
    assert_eq!(
        reverse.predecessors(exit.index()).collect::<Vec<_>>(),
        vec![virtual_exit, virtual_exit]
    );
}

#[test]
fn post_dominators_do_not_require_consumer_ids_for_the_virtual_exit() {
    let post = DominatorTree::compute_post_from(&BoundedDiamond, &[BoundedNode(3)]);
    assert_eq!(post.idom(BoundedNode(0)), Some(BoundedNode(3)));
    assert_eq!(post.idom(BoundedNode(1)), Some(BoundedNode(3)));
    assert_eq!(post.idom(BoundedNode(2)), Some(BoundedNode(3)));
    assert!(post.dominates(BoundedNode(3), BoundedNode(0)));
}

#[test]
fn post_dominators_accept_large_unsorted_exit_lists() {
    let mut graph = Graph::<(), ()>::new();
    let entry = graph.add_node(());
    let mut exits = Vec::new();
    for _ in 0..16 {
        let exit = graph.add_node(());
        graph.add_edge(entry, exit, ());
        exits.push(exit);
    }

    let ordered = DominatorTree::compute_post_from(&graph, &exits);
    exits.reverse();
    let reversed = DominatorTree::compute_post_from(&graph, &exits);
    assert_eq!(reversed, ordered);
}

#[test]
fn children_returns_immediate_children() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let a = cfg.new_block();
    let b = cfg.new_block();
    let c = cfg.new_block();
    cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
    cfg.add_edge(a, c, EdgeKind::Fallthrough);
    let dom = DominatorTree::compute(&cfg);
    let mut entry_children = dom.children(cfg.entry());
    entry_children.sort();
    assert_eq!(entry_children.len(), 2);
    assert!(entry_children.contains(&a));
    assert!(entry_children.contains(&b));
    assert_eq!(dom.children(a), vec![c]);
}

#[test]
fn compute_with_diff_detects_an_idom_change() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let a = cfg.new_block();
    let b = cfg.new_block();
    cfg.add_edge(cfg.entry(), a, EdgeKind::Fallthrough);
    cfg.add_edge(a, b, EdgeKind::Fallthrough);

    let dom = DominatorTree::compute(&cfg);
    // Add a shortcut edge from entry directly to b.
    cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalTrue);
    let (next, changed) = DominatorTree::compute_with_diff(&cfg, &dom);

    // b's idom should have changed from a to entry.
    assert!(changed.contains(&b));
    assert_eq!(next.idom(b), Some(cfg.entry()));
}

#[test]
fn compute_with_diff_reports_no_change_for_a_redundant_edge() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let a = cfg.new_block();
    cfg.add_edge(cfg.entry(), a, EdgeKind::Fallthrough);

    let dom = DominatorTree::compute(&cfg);
    // A second entry→a edge doesn't change dominators.
    cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
    let (_next, changed) = DominatorTree::compute_with_diff(&cfg, &dom);
    assert!(changed.is_empty());
}

#[test]
fn one_scratch_reused_down_a_sequence_computes_the_allocating_answer() {
    let sequence = crate::test_util::shapes::scratch_sequence::<MockInst>();
    let mut scratch = DominatorScratch::new();
    // Twice, so the second pass sees a scratch every buffer of which is
    // already at the sequence's high-water mark.
    for _ in 0..2 {
        for cfg in &sequence {
            assert_eq!(
                DominatorTree::compute_in(&mut scratch, cfg),
                DominatorTree::compute(cfg),
                "{} blocks",
                cfg.block_count()
            );
        }
    }
}

#[test]
fn a_reused_scratch_computes_the_allocating_answer_from_any_root() {
    let cfg = crate::test_util::shapes::diamond_chain::<MockInst>(3);
    let mut scratch = DominatorScratch::new();
    let large = crate::test_util::shapes::diamond_chain::<MockInst>(30);
    // Grow the scratch past the graph under test first: a stale entry left
    // beyond the node bound must not reach the answer.
    drop(DominatorTree::compute_in(&mut scratch, &large));
    for root in cfg.block_ids() {
        assert_eq!(
            DominatorTree::compute_from_in(&mut scratch, &cfg, root),
            DominatorTree::compute_from(&cfg, root),
            "root {root:?}"
        );
    }
}

#[test]
fn a_scratch_crosses_node_id_types() {
    // One scratch, two graphs whose node identities are unrelated types,
    // which is the reuse a worker thread over a whole corpus needs.
    let cfg = crate::test_util::shapes::diamond_chain::<MockInst>(6);
    let mut graph = Graph::<(), ()>::new();
    let nodes: Vec<_> = (0..4).map(|_| graph.add_node(())).collect();
    graph.add_edge(nodes[0], nodes[1], ());
    graph.add_edge(nodes[0], nodes[2], ());
    graph.add_edge(nodes[1], nodes[3], ());

    let mut scratch = DominatorScratch::new();
    assert_eq!(
        DominatorTree::compute_in(&mut scratch, &cfg),
        DominatorTree::compute(&cfg)
    );
    assert_eq!(
        DominatorTree::compute_from_in(&mut scratch, &graph, nodes[0]),
        DominatorTree::compute_from(&graph, nodes[0])
    );
    assert_eq!(
        DominatorTree::compute_in(&mut scratch, &cfg),
        DominatorTree::compute(&cfg)
    );
}
