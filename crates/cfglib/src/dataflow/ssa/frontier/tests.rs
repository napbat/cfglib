//! Dominance frontiers over a `Cfg` and over a consumer-owned view.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::DominanceFrontiers;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::dominator::DominatorTree;
use crate::graph::view::{GraphView, RootedView};

/// Forward adjacency lists over `u32` nodes, rooted at node zero. The node
/// identity is not `BlockId`, so the tests prove the algorithm reads the view
/// and not a `Cfg`.
struct Adjacency {
    successors: Vec<Vec<u32>>,
}

impl GraphView for Adjacency {
    type NodeId = u32;

    fn node_bound(&self) -> usize {
        self.successors.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = u32> + '_ {
        0..u32::try_from(self.successors.len()).expect("a toy graph has few nodes")
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.successors[node as usize].iter().copied()
    }

    fn predecessors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.node_ids().flat_map(move |source| {
            self.successors[source as usize]
                .iter()
                .filter(move |&&target| target == node)
                .map(move |_| source)
        })
    }
}

impl RootedView for Adjacency {
    fn root(&self) -> u32 {
        0
    }
}

/// An [`Adjacency`] that yields its nodes from the highest identity down, so
/// the walk meets the blocks out of ascending order.
struct Descending(Adjacency);

impl GraphView for Descending {
    type NodeId = u32;

    fn node_bound(&self) -> usize {
        self.0.node_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = u32> + '_ {
        (0..u32::try_from(self.0.node_bound()).expect("a toy graph has few nodes")).rev()
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.0.successors(node)
    }

    fn predecessors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.0.predecessors(node)
    }
}

impl RootedView for Descending {
    fn root(&self) -> u32 {
        self.0.root()
    }
}

/// `0 → 1, 2` and `1, 2 → 3`.
fn diamond_edges() -> Vec<(u32, u32)> {
    vec![(0, 1), (0, 2), (1, 3), (2, 3)]
}

/// `0 → 1`, a loop `1 → 2 → 1`, and the exit `1 → 3`.
fn loop_edges() -> Vec<(u32, u32)> {
    vec![(0, 1), (1, 2), (2, 1), (1, 3)]
}

fn adjacency(count: usize, edges: &[(u32, u32)]) -> Adjacency {
    let mut successors = vec![Vec::new(); count];
    for &(source, target) in edges {
        successors[source as usize].push(target);
    }
    Adjacency { successors }
}

fn cfg(count: usize, edges: &[(u32, u32)]) -> Cfg<u32> {
    let mut cfg = Cfg::<u32>::new();
    let mut blocks = vec![cfg.entry()];
    blocks.extend((1..count).map(|_| cfg.new_block()));
    for &(source, target) in edges {
        cfg.add_edge(
            blocks[source as usize],
            blocks[target as usize],
            EdgeKind::Fallthrough,
        );
    }
    cfg
}

fn frontier_indices(frontiers: &DominanceFrontiers<u32>, count: u32) -> Vec<Vec<u32>> {
    (0..count)
        .map(|node| frontiers.frontier(node).to_vec())
        .collect()
}

#[test]
fn the_join_of_a_diamond_is_the_frontier_of_both_arms() {
    let graph = adjacency(4, &diamond_edges());
    let frontiers = DominanceFrontiers::compute(&graph, &DominatorTree::compute(&graph));
    assert_eq!(
        frontier_indices(&frontiers, 4),
        [vec![], vec![3], vec![3], vec![]]
    );
}

#[test]
fn a_loop_header_is_in_its_own_frontier() {
    let graph = adjacency(4, &loop_edges());
    let frontiers = DominanceFrontiers::compute(&graph, &DominatorTree::compute(&graph));
    assert_eq!(
        frontier_indices(&frontiers, 4),
        [vec![], vec![1], vec![1], vec![]]
    );
}

#[test]
fn a_cfg_and_a_foreign_view_of_one_graph_agree() {
    for edges in [diamond_edges(), loop_edges()] {
        let graph = adjacency(4, &edges);
        let foreign = DominanceFrontiers::compute(&graph, &DominatorTree::compute(&graph));

        let cfg = cfg(4, &edges);
        let native = DominanceFrontiers::compute(&cfg, &DominatorTree::compute(&cfg));
        for (node, block) in cfg.block_ids().enumerate() {
            let expected: BTreeSet<usize> = foreign
                .frontier(u32::try_from(node).unwrap())
                .iter()
                .map(|&member| member as usize)
                .collect();
            let actual: BTreeSet<usize> = native
                .frontier(block)
                .iter()
                .map(|&member| member.index())
                .collect();
            assert_eq!(actual, expected, "frontier of node {node}");
        }
    }
}

#[test]
fn a_block_with_one_predecessor_has_no_frontier_entry() {
    let graph = adjacency(3, &[(0, 1), (1, 2)]);
    let frontiers = DominanceFrontiers::compute(&graph, &DominatorTree::compute(&graph));
    assert_eq!(
        frontier_indices(&frontiers, 3),
        [Vec::<u32>::new(), Vec::new(), Vec::new()]
    );
}

#[test]
fn an_entry_with_one_back_edge_is_a_join() {
    // `0 → 1 → 0` and `1 → 2`: the start of the function is the second path
    // into the entry.
    let graph = adjacency(3, &[(0, 1), (1, 0), (1, 2)]);
    let frontiers = DominanceFrontiers::compute(&graph, &DominatorTree::compute(&graph));
    assert_eq!(
        frontier_indices(&frontiers, 3),
        [vec![0], vec![0], Vec::new()]
    );
}

#[test]
fn an_entry_with_two_back_edges_is_in_its_own_frontier() {
    let graph = adjacency(4, &[(0, 1), (1, 0), (0, 2), (2, 0), (1, 3)]);
    let frontiers = DominanceFrontiers::compute(&graph, &DominatorTree::compute(&graph));
    assert_eq!(
        frontier_indices(&frontiers, 4),
        [vec![0], vec![0], vec![0], Vec::new()]
    );
}

#[test]
fn dead_code_has_no_frontier_even_when_it_branches_into_live_code() {
    // `2` is unreachable and joins live code at `1`; `3` is unreachable and
    // branches into the entry. A frontier is a property of code that runs.
    let graph = adjacency(4, &[(0, 1), (2, 1), (3, 0)]);
    let table = DominanceFrontiers::compute(&graph, &DominatorTree::compute(&graph));
    assert_eq!(frontier_indices(&table, 4), vec![Vec::<u32>::new(); 4]);
}

/// Every digraph on up to four nodes, self-loops included, as a node count
/// and an edge list.
fn small_digraphs() -> impl Iterator<Item = (u32, Vec<(u32, u32)>)> {
    (1..=4_u32).flat_map(|count| {
        let pairs: Vec<(u32, u32)> = (0..count)
            .flat_map(|source| (0..count).map(move |target| (source, target)))
            .collect();
        (0..1_u32 << pairs.len()).map(move |mask| {
            let edges = pairs
                .iter()
                .enumerate()
                .filter(|&(bit, _)| mask & (1 << bit) != 0)
                .map(|(_, &edge)| edge)
                .collect();
            (count, edges)
        })
    })
}

/// Every small digraph: the table agrees with the on-demand frontier for
/// every node, an unreachable one included, which has an empty frontier in
/// both. The two are one rule computed two ways.
#[test]
fn the_table_agrees_with_the_on_demand_frontier_on_every_small_digraph() {
    let mut on_demand = Vec::new();
    for (count, edges) in small_digraphs() {
        let graph = adjacency(count as usize, &edges);
        let dominators = DominatorTree::compute(&graph);
        let table = DominanceFrontiers::compute(&graph, &dominators);
        for node in 0..count {
            dominators.frontier(&graph, node, &mut on_demand);
            let expected: BTreeSet<u32> = on_demand.iter().copied().collect();
            assert!(
                table.frontier(node).iter().eq(&expected),
                "frontier of {node} in {edges:?}"
            );
        }
    }
}

/// A frontier reads back in strictly ascending order for every block of every
/// small digraph, including the blocks that the walk reaches only through an
/// unreachable predecessor.
#[test]
fn every_frontier_is_strictly_ascending_on_every_small_digraph() {
    for (count, edges) in small_digraphs() {
        let graph = adjacency(count as usize, &edges);
        let table = DominanceFrontiers::compute(&graph, &DominatorTree::compute(&graph));
        for node in 0..count {
            assert!(
                table
                    .frontier(node)
                    .is_sorted_by(|left, right| left < right),
                "frontier of {node} in {edges:?}"
            );
        }
    }
}

/// Two edges into one join walk the same runners twice, and a runner is
/// listed once.
#[test]
fn parallel_edges_do_not_repeat_a_frontier_member() {
    let graph = adjacency(4, &[(0, 1), (0, 2), (1, 3), (1, 3), (2, 3), (2, 3)]);
    let frontiers = DominanceFrontiers::compute(&graph, &DominatorTree::compute(&graph));
    assert_eq!(
        frontier_indices(&frontiers, 4),
        [vec![], vec![3], vec![3], vec![]]
    );
}

/// The order a view yields its nodes in changes the order the walk emits its
/// pairs in and nothing else: the frontiers are the ones the ascending view
/// gives.
#[test]
fn a_view_that_yields_its_nodes_in_descending_order_gets_the_same_frontiers() {
    for (count, edges) in small_digraphs() {
        let ascending = adjacency(count as usize, &edges);
        let expected = DominanceFrontiers::compute(&ascending, &DominatorTree::compute(&ascending));
        let descending = Descending(adjacency(count as usize, &edges));
        let table = DominanceFrontiers::compute(&descending, &DominatorTree::compute(&descending));
        assert_eq!(
            frontier_indices(&table, count),
            frontier_indices(&expected, count),
            "frontiers of {edges:?}"
        );
    }
}

#[test]
fn a_cloned_table_reads_the_same_frontiers() {
    let graph = adjacency(4, &loop_edges());
    let table = DominanceFrontiers::compute(&graph, &DominatorTree::compute(&graph));
    assert_eq!(
        frontier_indices(&table.clone(), 4),
        [vec![], vec![1], vec![1], vec![]]
    );
}

#[test]
fn a_removed_block_has_an_empty_frontier_and_leaves_the_others_in_place() {
    let edges = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)];
    let mut cfg = cfg(5, &edges);
    let orphan = BlockId::from_index(3);
    let before = DominanceFrontiers::compute(&cfg, &DominatorTree::compute(&cfg));
    assert_eq!(before.frontier(orphan), [BlockId::from_index(4)]);

    assert!(cfg.remove_block(orphan));
    assert!(cfg.block_bound() > cfg.block_count());
    let after = DominanceFrontiers::compute(&cfg, &DominatorTree::compute(&cfg));
    assert_eq!(after.frontier(orphan), []);
    for node in [1, 2] {
        assert_eq!(
            after.frontier(BlockId::from_index(node)),
            [BlockId::from_index(4)]
        );
    }
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn a_block_past_the_node_bound_panics() {
    let graph = adjacency(2, &[(0, 1)]);
    let table = DominanceFrontiers::compute(&graph, &DominatorTree::compute(&graph));
    let _ = table.frontier(2);
}
