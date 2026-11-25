//! Edge dominance against its definition: an edge dominates a node when
//! deleting that one edge cuts the node off from the root.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::support::{EdgeList, for_each_multigraph, node};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{EdgeId, EdgeKind};
use crate::graph::dominator::{DominatorScratch, DominatorTree};

/// The nodes the root, node 0, reaches over `edges` without the edge at
/// index `skipped`.
pub(super) fn reached_without(edges: &EdgeList, skipped: Option<usize>) -> Vec<bool> {
    let mut reached = vec![false; edges.node_count];
    let mut stack = vec![0_usize];
    reached[0] = true;
    while let Some(from) = stack.pop() {
        for (index, &(tail, head)) in edges.edges.iter().enumerate() {
            let head = head as usize;
            if Some(index) != skipped && tail as usize == from && !reached[head] {
                reached[head] = true;
                stack.push(head);
            }
        }
    }
    reached
}

/// Check every edge, present or not, and every node against the definition.
fn check_node_form(edges: &EdgeList, scratch: &mut DominatorScratch) {
    let graph = edges.graph();
    let tree = DominatorTree::compute_from_in(scratch, &graph, 0);
    let count = edges.node_count;
    let reached = reached_without(edges, None);

    for to in 0..count {
        let mut bridge = None;
        for from in 0..count {
            let occurrence = edges
                .edges
                .iter()
                .position(|&edge| edge == (node(from), node(to)));
            let cut = occurrence.map(|index| reached_without(edges, Some(index)));
            for target in 0..count {
                let expected = cut
                    .as_ref()
                    .is_some_and(|cut| reached[target] && !cut[target]);
                assert_eq!(
                    tree.edge_dominates(&graph, node(from), node(to), node(target)),
                    expected,
                    "{from} -> {to} dominates {target} in {edges:?}"
                );
                if expected && target == to {
                    assert_eq!(bridge, None, "two bridges into {to} in {edges:?}");
                    bridge = Some(node(from));
                }
            }
        }
        assert_eq!(
            tree.dominating_in_edge(&graph, node(to)),
            bridge,
            "bridge into {to} in {edges:?}"
        );
    }
}

/// The CFG with the edges of `edges`, block `i` standing for node `i`, and
/// the identity each edge of the list received.
fn cfg_of(edges: &EdgeList) -> (Cfg<()>, Vec<EdgeId>) {
    let mut cfg = Cfg::new();
    for _ in 1..edges.node_count {
        cfg.new_block();
    }
    let ids = edges
        .edges
        .iter()
        .map(|&(from, to)| {
            cfg.add_edge(
                BlockId::from_index(from as usize),
                BlockId::from_index(to as usize),
                EdgeKind::Unconditional,
            )
        })
        .collect();
    (cfg, ids)
}

/// Check every edge identity and every node against the definition.
fn check_id_form(edges: &EdgeList, scratch: &mut DominatorScratch) {
    let (cfg, ids) = cfg_of(edges);
    let tree = DominatorTree::compute_in(scratch, &cfg);
    let reached = reached_without(edges, None);
    for (index, &id) in ids.iter().enumerate() {
        let cut = reached_without(edges, Some(index));
        for target in 0..edges.node_count {
            assert_eq!(
                tree.edge_id_dominates(&cfg, id, BlockId::from_index(target)),
                reached[target] && !cut[target],
                "edge {index} dominates {target} in {edges:?}"
            );
        }
    }
}

#[test]
fn every_digraph_of_up_to_four_nodes_matches_the_definition() {
    let mut scratch = DominatorScratch::new();
    for node_count in 1..=4 {
        for_each_multigraph(node_count, 1, |edges| {
            check_node_form(edges, &mut scratch);
            check_id_form(edges, &mut scratch);
        });
    }
}

#[test]
fn every_multigraph_of_up_to_three_nodes_matches_the_definition() {
    let mut scratch = DominatorScratch::new();
    for node_count in 1..=3 {
        for_each_multigraph(node_count, 2, |edges| {
            check_node_form(edges, &mut scratch);
            check_id_form(edges, &mut scratch);
        });
    }
}

/// A CFG over `count` blocks with `edges` between them, by block index.
fn cfg_with(count: usize, edges: &[(usize, usize)]) -> (Cfg<()>, Vec<BlockId>, Vec<EdgeId>) {
    let list = EdgeList::new(
        count,
        edges
            .iter()
            .map(|&(from, to)| (node(from), node(to)))
            .collect(),
    );
    let (cfg, ids) = cfg_of(&list);
    let blocks = (0..count).map(BlockId::from_index).collect();
    (cfg, blocks, ids)
}

#[test]
fn each_arm_of_a_diamond_dominates_its_own_block_and_not_the_join() {
    let (cfg, b, ids) = cfg_with(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    let tree = DominatorTree::compute(&cfg);
    for (arm, id) in [(1, ids[0]), (2, ids[1])] {
        assert!(tree.edge_dominates(&cfg, b[0], b[arm], b[arm]));
        assert!(tree.edge_id_dominates(&cfg, id, b[arm]));
        assert!(!tree.edge_dominates(&cfg, b[0], b[arm], b[3]));
        assert!(!tree.edge_id_dominates(&cfg, id, b[3]));
        assert!(!tree.edge_dominates(&cfg, b[0], b[arm], b[0]));
    }
    assert_eq!(tree.dominating_in_edge(&cfg, b[1]), Some(b[0]));
    assert_eq!(tree.dominating_in_edge(&cfg, b[3]), None);
    assert!(!tree.edge_dominates(&cfg, b[1], b[3], b[3]));
}

#[test]
fn the_entry_edge_of_a_loop_dominates_the_loop_and_the_back_edge_nothing() {
    // entry -> header -> body -> header, header -> exit
    let (cfg, b, ids) = cfg_with(4, &[(0, 1), (1, 2), (2, 1), (1, 3)]);
    let tree = DominatorTree::compute(&cfg);
    for target in [1, 2, 3] {
        assert!(tree.edge_dominates(&cfg, b[0], b[1], b[target]));
        assert!(tree.edge_id_dominates(&cfg, ids[0], b[target]));
    }
    assert_eq!(tree.dominating_in_edge(&cfg, b[1]), Some(b[0]));
    for target in 0..4 {
        assert!(!tree.edge_dominates(&cfg, b[2], b[1], b[target]));
        assert!(!tree.edge_id_dominates(&cfg, ids[2], b[target]));
    }
    assert!(tree.edge_dominates(&cfg, b[1], b[2], b[2]));
    assert!(!tree.edge_dominates(&cfg, b[1], b[2], b[3]));
}

#[test]
fn neither_of_two_parallel_edges_dominates_their_head() {
    let (cfg, b, ids) = cfg_with(2, &[(0, 1), (0, 1)]);
    let tree = DominatorTree::compute(&cfg);
    assert!(tree.dominates(b[0], b[1]));
    assert!(!tree.edge_dominates(&cfg, b[0], b[1], b[1]));
    assert_eq!(tree.dominating_in_edge(&cfg, b[1]), None);
    for &id in &ids {
        assert!(!tree.edge_id_dominates(&cfg, id, b[1]));
    }
}

#[test]
fn an_edge_into_the_root_dominates_nothing() {
    let (cfg, b, ids) = cfg_with(2, &[(0, 1), (1, 0)]);
    let tree = DominatorTree::compute(&cfg);
    for target in 0..2 {
        assert!(!tree.edge_dominates(&cfg, b[1], b[0], b[target]));
        assert!(!tree.edge_id_dominates(&cfg, ids[1], b[target]));
    }
    assert_eq!(tree.dominating_in_edge(&cfg, b[0]), None);
    assert!(tree.edge_dominates(&cfg, b[0], b[1], b[1]));
}

#[test]
fn an_unreachable_tail_or_target_dominates_nothing() {
    // Block 2 is unreachable and has an edge into block 1.
    let (cfg, b, ids) = cfg_with(3, &[(0, 1), (2, 1)]);
    let tree = DominatorTree::compute(&cfg);
    assert!(tree.edge_dominates(&cfg, b[0], b[1], b[1]));
    assert!(tree.edge_id_dominates(&cfg, ids[0], b[1]));
    assert!(!tree.edge_dominates(&cfg, b[2], b[1], b[1]));
    assert!(!tree.edge_id_dominates(&cfg, ids[1], b[1]));
    assert!(!tree.edge_dominates(&cfg, b[0], b[1], b[2]));
    assert_eq!(tree.dominating_in_edge(&cfg, b[2]), None);
}

#[test]
fn a_removed_edge_slot_is_not_an_in_edge() {
    let (mut cfg, b, ids) = cfg_with(2, &[(0, 1), (0, 1)]);
    assert!(cfg.remove_edge(ids[1]));
    let tree = DominatorTree::compute(&cfg);
    assert!(tree.edge_dominates(&cfg, b[0], b[1], b[1]));
    assert!(tree.edge_id_dominates(&cfg, ids[0], b[1]));
}
