//! `follow` and `follow_path` over out-degree-one chases, and `scan_predecessors` for a view that
//! stores only forward adjacency.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{
    GraphView, NodeId, SearchConfig, SearchOrder, TraversalDirection, Visit, breadth_first,
    depth_first_preorder, follow, follow_path, reachable, scan_predecessors, search, shortest_path,
    topological_sort,
};
use std::ops::ControlFlow;

use crate::Lcg;
use crate::reference::Table;
use crate::spaces::{collatz_step, collatz_trajectory};

#[test]
fn a_chase_over_the_collatz_sequence_ends_at_one_where_the_next_node_is_already_on_the_path() {
    let step = |node: &u64| Some(collatz_step(*node));
    let trajectory = collatz_trajectory(27);

    assert_eq!(
        follow(27, 1000, step),
        1,
        "the next node, 4, is on the path"
    );
    assert_eq!(follow_path(27, 1000, step), trajectory);
    assert_eq!(trajectory.len(), 112);
}

#[test]
fn the_hop_bound_stops_a_chase_after_that_many_steps() {
    let step = |node: &u64| Some(collatz_step(*node));
    let trajectory = collatz_trajectory(27);

    for hops in [0, 1, 7, 50, 111] {
        assert_eq!(follow(27, hops, step), trajectory[hops], "{hops} hops");
        let path = follow_path(27, hops, step);
        assert_eq!(path, trajectory[..=hops]);
        assert_eq!(
            path.len(),
            hops + 1,
            "at most max_hops + 1 nodes, seed first"
        );
    }
    assert_eq!(
        follow_path(27, 5000, step).len(),
        112,
        "a bound beyond the cycle changes nothing"
    );
}

#[test]
fn a_chase_that_runs_out_of_successors_returns_its_last_node() {
    let chain = |node: &u32| (*node < 4).then_some(node + 1);
    assert_eq!(follow(0, 100, chain), 4);
    assert_eq!(follow_path(0, 100, chain), [0, 1, 2, 3, 4]);

    let nowhere = |_: &u32| None;
    assert_eq!(follow(9, 100, nowhere), 9);
    assert_eq!(follow_path(9, 100, nowhere), [9]);
}

#[test]
fn the_cycle_guard_is_the_whole_path_not_just_the_seed() {
    // 0 -> 1 -> 2 -> 3 -> 1: the chain re-enters itself three hops in.
    let next = [1_usize, 2, 3, 1];
    let tail_cycle = |node: &usize| Some(next[*node]);
    assert_eq!(follow(0, 100, tail_cycle), 3);
    assert_eq!(follow_path(0, 100, tail_cycle), [0, 1, 2, 3]);

    let around = |node: &u32| Some((node + 1) % 3);
    assert_eq!(follow(0, 16, around), 2);
    assert_eq!(follow_path(0, 16, around), [0, 1, 2]);
    assert_eq!(follow(0, 1, around), 1);
}

#[test]
fn random_chases_match_a_manual_walk_with_a_visited_set() {
    for seed in 1..=8 {
        let mut rng = Lcg(seed);
        let next: Vec<Option<usize>> = (0..40)
            .map(|_| (rng.below(5) != 0).then(|| rng.below(40)))
            .collect();
        let step = |node: &usize| next[*node];
        for start in 0..40 {
            for hops in [0, 3, 100] {
                let mut path = vec![start];
                let mut seen = BTreeSet::from([start]);
                while path.len() <= hops {
                    let Some(following) = next[*path.last().expect("non-empty")] else {
                        break;
                    };
                    if !seen.insert(following) {
                        break;
                    }
                    path.push(following);
                }
                assert_eq!(
                    follow_path(start, hops, step),
                    path,
                    "seed {seed}, start {start}, {hops} hops"
                );
                assert_eq!(follow(start, hops, step), *path.last().expect("non-empty"));
            }
        }
    }
}

#[test]
fn nodes_need_only_equality_and_cloning_so_aliases_by_name_chase_like_numbers() {
    let aliases: BTreeMap<String, String> = [
        ("Handle", "Ref"),
        ("Ref", "Pointer"),
        ("Pointer", "Address"),
    ]
    .into_iter()
    .map(|(alias, target)| (alias.to_owned(), target.to_owned()))
    .collect();
    let step = |name: &String| aliases.get(name).cloned();

    assert_eq!(follow("Handle".to_owned(), 8, step), "Address");
    assert_eq!(
        follow_path("Handle".to_owned(), 8, step),
        ["Handle", "Ref", "Pointer", "Address"]
    );
    assert_eq!(follow("Handle".to_owned(), 2, step), "Pointer");
}

/// A consumer view that stores only forward adjacency and answers predecessor queries by scan.
struct ForwardOnly {
    successors: Vec<Vec<usize>>,
    live: Vec<bool>,
}

impl ForwardOnly {
    fn new(table: &Table) -> Self {
        Self {
            successors: table.successors.clone(),
            live: vec![true; table.node_count()],
        }
    }

    fn without(table: &Table, removed: &[usize]) -> Self {
        let mut view = Self::new(&table.without(removed));
        for &node in removed {
            view.live[node] = false;
        }
        view
    }
}

impl GraphView for ForwardOnly {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.successors.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.successors.len()).filter(|&node| self.live[node])
    }

    fn successors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.successors[node].iter().copied()
    }

    fn predecessors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        scan_predecessors(self, node)
    }
}

/// The table with repeated edges dropped, keeping the first occurrence of each.
fn simple(table: &Table) -> Table {
    let successors = table
        .successors
        .iter()
        .map(|targets| {
            let mut seen = BTreeSet::new();
            targets
                .iter()
                .copied()
                .filter(|&target| seen.insert(target))
                .collect()
        })
        .collect();
    Table { successors }
}

fn indexes(nodes: Vec<NodeId>) -> Vec<usize> {
    nodes.into_iter().map(NodeId::index).collect()
}

#[test]
fn scanning_finds_the_same_incoming_neighbors_a_store_keeps_in_its_reverse_adjacency() {
    for seed in 1..=6 {
        let table = simple(&Table::random(&mut Lcg(seed), 25, 70));
        let view = ForwardOnly::new(&table);
        let graph = table.graph();
        for node in 0..25 {
            let scanned: Vec<usize> = view.predecessors(node).collect();
            let stored = indexes(graph.predecessors(NodeId::from_index(node)).collect());
            assert_eq!(scanned, stored, "seed {seed}, n{node}");
        }
    }
}

#[test]
fn a_self_loop_is_its_own_predecessor_in_the_scan() {
    let table = Table {
        successors: vec![vec![0, 1], vec![]],
    };
    let view = ForwardOnly::new(&table);
    assert_eq!(view.predecessors(0).collect::<Vec<_>>(), [0]);
    assert_eq!(view.predecessors(1).collect::<Vec<_>>(), [0]);
}

#[test]
fn a_removed_node_is_neither_a_node_nor_a_predecessor_of_the_view() {
    let table = simple(&Table::random(&mut Lcg(9), 20, 50));
    let removed = [3, 4];
    let view = ForwardOnly::without(&table, &removed);
    let mut graph = table.graph();
    for &node in &removed {
        graph.remove_node(NodeId::from_index(node));
    }

    assert_eq!(view.node_ids().count(), 18);
    for node in graph.node_ids() {
        let scanned: Vec<usize> = view.predecessors(node.index()).collect();
        assert!(scanned.iter().all(|source| !removed.contains(source)));
        assert_eq!(scanned, indexes(graph.predecessors(node).collect()));
    }
}

#[test]
fn the_backward_algorithms_agree_between_a_forward_only_view_and_the_store() {
    for seed in 1..=5 {
        let table = simple(&Table::random(&mut Lcg(seed + 10), 25, 60));
        let view = ForwardOnly::new(&table);
        let graph = table.graph();
        let incoming = TraversalDirection::Incoming;
        for start in [0, 6, 17] {
            let id = NodeId::from_index(start);
            assert_eq!(
                breadth_first(&view, start, incoming),
                indexes(breadth_first(&graph, id, incoming)),
                "seed {seed}, breadth-first from n{start}"
            );
            assert_eq!(
                depth_first_preorder(&view, start, incoming),
                indexes(depth_first_preorder(&graph, id, incoming)),
                "seed {seed}, depth-first from n{start}"
            );
            assert_eq!(
                reachable(&view, [start], incoming),
                reachable(&graph, [id], incoming),
                "seed {seed}, reachable from n{start}"
            );
            for goal in [2, 11, 24] {
                let along_view = shortest_path(&view, start, goal, incoming).map(|path| path.len());
                let along_graph = shortest_path(&graph, id, NodeId::from_index(goal), incoming)
                    .map(|path| path.len());
                assert_eq!(
                    along_view, along_graph,
                    "seed {seed}, n{start} back to n{goal}"
                );
            }
        }
    }
}

#[test]
fn a_backward_search_over_a_forward_only_view_visits_like_one_over_the_store() {
    let table = simple(&Table::random(&mut Lcg(31), 25, 60));
    let view = ForwardOnly::new(&table);
    let graph = table.graph();
    for order in [SearchOrder::DepthFirst, SearchOrder::BreadthFirst] {
        let config = SearchConfig::new(order, TraversalDirection::Incoming).with_max_depth(4);
        let mut along_view = Vec::new();
        let outcome = search(&view, [5_usize], config, |node, depth| {
            along_view.push((node, depth));
            ControlFlow::<(), Visit>::Continue(Visit::Descend)
        });
        assert_eq!(outcome, None);
        let mut along_graph = Vec::new();
        let outcome = search(&graph, [NodeId::from_index(5)], config, |node, depth| {
            along_graph.push((node.index(), depth));
            ControlFlow::<(), Visit>::Continue(Visit::Descend)
        });
        assert_eq!(outcome, None);
        assert_eq!(along_view, along_graph, "{order:?}");
    }
}

#[test]
fn a_scan_keeps_every_parallel_edge_the_forward_adjacency_lists() {
    // The view contract: forward and reverse adjacency describe one edge multiset.
    let table = Table {
        successors: vec![vec![1, 1, 2], vec![2], vec![]],
    };
    let view = ForwardOnly::new(&table);
    assert_eq!(
        view.predecessors(1).collect::<Vec<_>>(),
        [0, 0],
        "two edges 0 -> 1"
    );
    assert_eq!(view.predecessors(2).collect::<Vec<_>>(), [0, 1]);
}

#[test]
fn a_topological_order_of_a_forward_only_view_with_parallel_edges_exists() {
    let table = Table {
        successors: vec![vec![1, 1, 2], vec![2], vec![]],
    };
    let view = ForwardOnly::new(&table);
    assert_eq!(topological_sort(&view), Some(vec![0, 1, 2]));
}
