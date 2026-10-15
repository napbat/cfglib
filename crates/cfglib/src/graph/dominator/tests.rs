//! Dominator and post-dominator trees over graphs, CFGs, and
//! consumer-defined views.

extern crate alloc;
use super::*;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::store::{Graph, NodeId};
use crate::test_util::MockInst;
use alloc::vec;

mod adversarial;
mod determinism;
mod differential;
mod exhaustive;
mod forest;
mod frontier;
mod post;
mod support;

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
    assert_eq!(dom.children(cfg.entry()).count(), 0);
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
fn level_computation() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let b1 = cfg.new_block();
    let b2 = cfg.new_block();
    cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
    cfg.add_edge(b1, b2, EdgeKind::Fallthrough);
    let dom = DominatorTree::compute(&cfg);
    assert_eq!(dom.level(cfg.entry()), Some(0));
    assert_eq!(dom.level(b1), Some(1));
    assert_eq!(dom.level(b2), Some(2));
}

#[test]
fn levels_and_preorder_follow_the_tree_when_parents_have_larger_ids() {
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
    for (node, level) in [(root, 0), (child, 1), (middle, 2), (leaf, 3)] {
        assert_eq!(dom.level(node), Some(level));
    }
    assert_eq!(dom.level(unreachable), None);
    assert_eq!(dom.preorder(), [root, child, middle, leaf]);
    assert_eq!(dom.subtree(child), [child, middle, leaf]);
    assert_eq!(dom.subtree(unreachable).len(), 0);
}

#[test]
fn dominance_keeps_its_answers_for_unreachable_nodes() {
    let mut graph = Graph::<(), ()>::new();
    let root = graph.add_node(());
    let child = graph.add_node(());
    let unreachable = graph.add_node(());
    let other = graph.add_node(());
    graph.add_edge(root, child, ());
    graph.add_edge(unreachable, child, ());
    graph.add_edge(unreachable, other, ());

    let dom = DominatorTree::compute_from(&graph, root);
    assert!(dom.dominates(unreachable, unreachable));
    assert!(!dom.strictly_dominates(unreachable, unreachable));
    assert!(!dom.dominates(root, unreachable));
    assert!(!dom.dominates(unreachable, root));
    assert!(!dom.dominates(unreachable, child));
    assert!(!dom.dominates(unreachable, other));
    assert_eq!(
        dom.idom(child),
        Some(root),
        "an unreachable predecessor is skipped"
    );
    assert_eq!(dom.nearest_common_dominator(root, unreachable), None);
}

#[test]
fn nearest_common_dominator_of_skips_unreachable_nodes() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let a = cfg.new_block();
    let b = cfg.new_block();
    let merge = cfg.new_block();
    let unreachable = cfg.new_block();
    cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
    cfg.add_edge(a, merge, EdgeKind::Fallthrough);
    cfg.add_edge(b, merge, EdgeKind::Fallthrough);
    let dom = DominatorTree::compute(&cfg);

    assert_eq!(dom.nearest_common_dominator(a, b), Some(cfg.entry()));
    assert_eq!(dom.nearest_common_dominator(a, merge), Some(cfg.entry()));
    assert_eq!(dom.nearest_common_dominator(merge, merge), Some(merge));
    assert_eq!(
        dom.nearest_common_dominator_of([merge, unreachable, a]),
        Some(cfg.entry())
    );
    assert_eq!(
        dom.nearest_common_dominator_of([unreachable, merge]),
        Some(merge)
    );
    assert_eq!(dom.nearest_common_dominator_of([unreachable]), None);
    assert_eq!(dom.nearest_common_dominator_of([]), None);
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

#[cfg(target_pointer_width = "64")]
struct Oversized;

#[cfg(target_pointer_width = "64")]
impl GraphView for Oversized {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        usize::try_from(u32::MAX).expect("u32 fits a 64-bit usize") + 1
    }

    fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        core::iter::once(0)
    }

    fn successors(&self, _node: usize) -> impl Iterator<Item = usize> + '_ {
        core::iter::empty()
    }

    fn predecessors(&self, _node: usize) -> impl Iterator<Item = usize> + '_ {
        core::iter::empty()
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "node bound exceeds the dense u32 range")]
fn a_node_bound_beyond_u32_is_rejected() {
    let _ = DominatorTree::compute_from(&Oversized, 0);
}

#[test]
#[should_panic(expected = "dominator root is not a node of the graph")]
fn a_root_outside_the_graph_is_rejected() {
    let mut graph = Graph::<(), ()>::new();
    graph.add_node(());
    let _ = DominatorTree::compute_from(&graph, NodeId::from_raw(1));
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
    assert_eq!(dom.children(cfg.entry()).collect::<Vec<_>>(), [a, b]);
    assert_eq!(dom.children(a).collect::<Vec<_>>(), [c]);
    assert_eq!(dom.children(c).count(), 0);
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
    assert_eq!(changed, [] as [crate::BlockId; 0]);
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
