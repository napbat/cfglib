//! Tests of the greedy tree decompositions.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::{EliminationHeuristic, TreeDecomposition};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::store::{Graph, NodeId};
use crate::graph::view::{DenseId, GraphView};
use crate::test_util::rng::Rng;
use crate::test_util::shapes::{diamond_chain, scratch_sequence};
use crate::{ControlTree, SplDecomposition};

const HEURISTICS: [EliminationHeuristic; 2] = [
    EliminationHeuristic::MinDegree,
    EliminationHeuristic::MinFill,
];

/// Check `decomposition` against the definition of a tree decomposition of
/// the undirected simple graph underlying `graph`: every node in a bag, both
/// ends of every edge in one bag, and the bags holding a node connected. Also
/// check the shape this one promises: a bag per node that holds it first, and
/// parents after their children.
fn assert_valid<G: GraphView>(graph: &G, decomposition: &TreeDecomposition<G::NodeId>) {
    let live: Vec<G::NodeId> = graph.node_ids().collect();
    assert_eq!(decomposition.bag_count(), live.len(), "a bag per node");
    let bags: Vec<BTreeSet<usize>> = (0..decomposition.bag_count())
        .map(|bag| {
            let members: BTreeSet<usize> = decomposition
                .bag(bag)
                .iter()
                .map(|node| node.index())
                .collect();
            assert_eq!(
                members.len(),
                decomposition.bag(bag).len(),
                "a bag lists a node once"
            );
            if let Some(parent) = decomposition.parent(bag) {
                assert!(parent > bag, "a parent comes after its child");
            }
            members
        })
        .collect();
    for &node in &live {
        let own = decomposition.bag_of(node).expect("a node has a bag");
        assert_eq!(decomposition.bag(own)[0], node, "a node leads its own bag");
        for next in graph.successors(node) {
            let (a, b) = (node.index(), next.index());
            assert!(
                bags.iter().any(|bag| bag.contains(&a) && bag.contains(&b)),
                "a bag holds both ends of {node:?} -> {next:?}"
            );
        }
        // The bags that hold the node form a subtree: exactly one of them has
        // no parent that holds it too.
        let holding: Vec<usize> = (0..bags.len())
            .filter(|&bag| bags[bag].contains(&node.index()))
            .collect();
        let tops = holding
            .iter()
            .filter(|&&bag| {
                decomposition
                    .parent(bag)
                    .is_none_or(|parent| !bags[parent].contains(&node.index()))
            })
            .count();
        assert_eq!(tops, 1, "the bags holding {node:?} are connected");
    }
    let width = bags.iter().map(BTreeSet::len).max().unwrap_or(1) - 1;
    assert_eq!(decomposition.width(), width);
}

/// The decomposition the definition of greedy elimination gives, computed
/// with no incremental bookkeeping: every step recomputes every key.
fn reference(
    node_count: usize,
    edges: &[(usize, usize)],
    heuristic: EliminationHeuristic,
) -> (Vec<Vec<usize>>, Vec<Option<usize>>) {
    let mut neighbors: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); node_count];
    for &(a, b) in edges {
        if a != b {
            neighbors[a].insert(b);
            neighbors[b].insert(a);
        }
    }
    let mut remaining: BTreeSet<usize> = (0..node_count).collect();
    let mut position = vec![usize::MAX; node_count];
    let mut bags = Vec::new();
    while !remaining.is_empty() {
        let key = |node: usize| -> u64 {
            let around = &neighbors[node];
            match heuristic {
                EliminationHeuristic::MinDegree => around.len() as u64,
                EliminationHeuristic::MinFill => {
                    let mut missing = 0;
                    for &a in around {
                        for &b in around {
                            if a < b && !neighbors[a].contains(&b) {
                                missing += 1;
                            }
                        }
                    }
                    missing
                }
            }
        };
        let node = *remaining
            .iter()
            .min_by_key(|&&node| (key(node), node))
            .expect("a node remains");
        remaining.remove(&node);
        position[node] = bags.len();
        let around: Vec<usize> = neighbors[node].iter().copied().collect();
        for &a in &around {
            neighbors[a].remove(&node);
            for &b in &around {
                if a != b {
                    neighbors[a].insert(b);
                }
            }
        }
        let mut bag = vec![node];
        bag.extend(around);
        bags.push(bag);
    }
    let parents = bags
        .iter()
        .map(|bag| bag[1..].iter().map(|&member| position[member]).min())
        .collect();
    (bags, parents)
}

/// A graph over `node_count` nodes with `edges` in order.
fn graph_of(node_count: usize, edges: &[(usize, usize)]) -> (Graph<(), ()>, Vec<NodeId>) {
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..node_count).map(|_| graph.add_node(())).collect();
    for &(a, b) in edges {
        graph.add_edge(nodes[a], nodes[b], ());
    }
    (graph, nodes)
}

/// A random multigraph, self-loops included.
fn random_edges(rng: &mut Rng, node_count: usize, edge_count: usize) -> Vec<(usize, usize)> {
    (0..edge_count)
        .map(|_| (rng.below(node_count), rng.below(node_count)))
        .collect()
}

#[test]
fn random_graphs_match_the_definition_of_greedy_elimination() {
    let mut rng = Rng::new(81);
    for _ in 0..400 {
        let node_count = 1 + rng.below(24);
        let edge_count = rng.below(3 * node_count);
        let edges = random_edges(&mut rng, node_count, edge_count);
        let (graph, _) = graph_of(node_count, &edges);
        for heuristic in HEURISTICS {
            let decomposition = TreeDecomposition::compute(&graph, heuristic);
            assert_valid(&graph, &decomposition);
            let (bags, parents) = reference(node_count, &edges, heuristic);
            for (bag, wanted) in bags.iter().enumerate() {
                let found: Vec<usize> = decomposition
                    .bag(bag)
                    .iter()
                    .map(|node| node.index())
                    .collect();
                assert_eq!(&found, wanted, "bag {bag} of {edges:?} by {heuristic:?}");
                assert_eq!(decomposition.parent(bag), parents[bag]);
            }
        }
    }
}

#[test]
fn known_graphs_have_their_treewidth() {
    // A path, a star, a cycle, a complete graph, and a lone node.
    let path: Vec<(usize, usize)> = (1..6).map(|next| (next - 1, next)).collect();
    let star: Vec<(usize, usize)> = (1..6).map(|leaf| (0, leaf)).collect();
    let cycle: Vec<(usize, usize)> = (0..6).map(|node| (node, (node + 1) % 6)).collect();
    let complete: Vec<(usize, usize)> = (0..5)
        .flat_map(|a| (0..5).filter(move |&b| b != a).map(move |b| (a, b)))
        .collect();
    for (node_count, edges, width) in [
        (6, &path[..], 1),
        (6, &star[..], 1),
        (6, &cycle[..], 2),
        (5, &complete[..], 4),
        (1, &[][..], 0),
    ] {
        let (graph, _) = graph_of(node_count, edges);
        for heuristic in HEURISTICS {
            let decomposition = TreeDecomposition::compute(&graph, heuristic);
            assert_valid(&graph, &decomposition);
            assert_eq!(decomposition.width(), width, "{edges:?} by {heuristic:?}");
        }
    }
}

#[test]
fn an_empty_graph_has_no_bag() {
    let graph = Graph::<(), ()>::new();
    let decomposition = TreeDecomposition::compute(&graph, EliminationHeuristic::MinFill);
    assert!(decomposition.is_empty());
    assert_eq!(decomposition.width(), 0);
}

#[test]
fn removed_nodes_have_no_bag_and_components_are_separate_trees() {
    let (mut graph, nodes) = graph_of(5, &[(0, 1), (1, 2), (3, 4)]);
    assert!(graph.remove_node(nodes[2]));
    let decomposition = TreeDecomposition::compute(&graph, EliminationHeuristic::MinDegree);
    assert_valid(&graph, &decomposition);
    assert_eq!(decomposition.bag_of(nodes[2]), None);
    assert_eq!(decomposition.bag_of(NodeId::from_index(9)), None);
    let roots = (0..decomposition.bag_count())
        .filter(|&bag| decomposition.parent(bag).is_none())
        .count();
    assert_eq!(roots, 2, "one tree per component");
}

#[test]
fn edge_order_parallel_edges_and_direction_change_nothing() {
    let mut rng = Rng::new(0x81);
    for _ in 0..100 {
        let node_count = 2 + rng.below(30);
        let edge_count = rng.below(3 * node_count);
        let mut edges = random_edges(&mut rng, node_count, edge_count);
        let (graph, _) = graph_of(node_count, &edges);
        rng.shuffle(&mut edges);
        let doubled: Vec<(usize, usize)> =
            edges.iter().flat_map(|&(a, b)| [(b, a), (a, b)]).collect();
        let (shuffled, _) = graph_of(node_count, &doubled);
        for heuristic in HEURISTICS {
            assert_eq!(
                TreeDecomposition::compute(&graph, heuristic),
                TreeDecomposition::compute(&shuffled, heuristic)
            );
        }
    }
}

/// A fall-through spine of `block_count` blocks with a jump out of every
/// third block to a block at most `reach` away, forward or back: gotos that
/// make irreducible loops and shared tails.
fn cfg_with_gotos(rng: &mut Rng, block_count: usize, reach: usize) -> Cfg<()> {
    let mut cfg = Cfg::new();
    let mut blocks = vec![cfg.entry()];
    blocks.extend((1..block_count).map(|_| cfg.new_block()));
    for index in 1..block_count {
        cfg.add_edge(blocks[index - 1], blocks[index], EdgeKind::Fallthrough);
    }
    for index in (0..block_count).step_by(3) {
        let low = index.saturating_sub(reach);
        let high = (index + reach).min(block_count - 1);
        let target = low + rng.below(high - low + 1);
        cfg.add_edge(blocks[index], blocks[target], EdgeKind::Jump);
    }
    cfg
}

#[test]
fn cfg_fixtures_and_cfgs_with_gotos_decompose() {
    let mut fixtures: Vec<Cfg<()>> = scratch_sequence();
    fixtures.push(diamond_chain(50));
    let mut rng = Rng::new(7);
    for _ in 0..40 {
        let block_count = 2 + rng.below(200);
        fixtures.push(cfg_with_gotos(&mut rng, block_count, 6));
    }
    for cfg in &fixtures {
        for heuristic in HEURISTICS {
            let decomposition = TreeDecomposition::compute(cfg, heuristic);
            assert_valid(cfg, &decomposition);
            let block: BlockId = cfg.entry();
            assert!(decomposition.bag_of(block).is_some());
        }
    }
}

/// `if (t0) { if (t1) { ... leaf } }; merge`.
fn nested_then_arms(depth: usize) -> Cfg<()> {
    let mut cfg = Cfg::new();
    let mut tests = vec![cfg.entry()];
    tests.extend((1..depth).map(|_| cfg.new_block()));
    let leaf = cfg.new_block();
    let merge = cfg.new_block();
    for (index, &test) in tests.iter().enumerate() {
        let next = tests.get(index + 1).copied().unwrap_or(leaf);
        cfg.add_edge(test, next, EdgeKind::ConditionalTrue);
        cfg.add_edge(test, merge, EdgeKind::ConditionalFalse);
    }
    cfg.add_edge(leaf, merge, EdgeKind::Fallthrough);
    cfg
}

/// `while (h0) { while (h1) { ... work } x1 } x0`: loop `i` tests in its
/// header, runs loop `i + 1`, and then `x(i + 1)`, which jumps back to it.
fn nested_loops(depth: usize) -> Cfg<()> {
    let mut cfg = Cfg::new();
    let headers: Vec<BlockId> = (0..depth).map(|_| cfg.new_block()).collect();
    let work = cfg.new_block();
    let exits: Vec<BlockId> = (0..depth).map(|_| cfg.new_block()).collect();
    cfg.add_edge(cfg.entry(), headers[0], EdgeKind::Fallthrough);
    for (level, &header) in headers.iter().enumerate() {
        let body = headers.get(level + 1).copied().unwrap_or(work);
        cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exits[level], EdgeKind::ConditionalFalse);
        if let Some(&inner_exit) = exits.get(level + 1) {
            cfg.add_edge(inner_exit, header, EdgeKind::Jump);
        }
    }
    cfg.add_edge(work, headers[depth - 1], EdgeKind::Jump);
    cfg
}

#[test]
fn heuristic_widths_against_the_spl_path_width_on_structured_fixtures() {
    // Not a bound: the heuristics carry no guarantee, and the path width is a
    // path decomposition's. The table records where each lands.
    let mut chain = Cfg::<()>::new();
    let mut previous = chain.entry();
    for _ in 0..5 {
        let next = chain.new_block();
        chain.add_edge(previous, next, EdgeKind::Fallthrough);
        previous = next;
    }
    let fixtures: [(&str, Cfg<()>); 7] = [
        ("chain", chain),
        ("diamond", diamond_chain(1)),
        ("four diamonds", diamond_chain(4)),
        ("then arms 4 deep", nested_then_arms(4)),
        ("then arms 8 deep", nested_then_arms(8)),
        ("loops 3 deep", nested_loops(3)),
        ("loops 6 deep", nested_loops(6)),
    ];
    let widths: Vec<(&str, [usize; 3])> = fixtures
        .iter()
        .map(|(name, cfg)| {
            let spl = SplDecomposition::compute(&ControlTree::compute(cfg), cfg)
                .expect("the fixture is structured");
            let widths = [
                TreeDecomposition::compute(cfg, EliminationHeuristic::MinDegree).width(),
                TreeDecomposition::compute(cfg, EliminationHeuristic::MinFill).width(),
                spl.path_decomposition().width(),
            ];
            (*name, widths)
        })
        .collect();
    // Min-degree, min-fill, and the SPL path decomposition.
    assert_eq!(
        widths,
        [
            ("chain", [1, 1, 1]),
            ("diamond", [2, 2, 2]),
            ("four diamonds", [2, 2, 2]),
            ("then arms 4 deep", [2, 2, 5]),
            ("then arms 8 deep", [2, 2, 9]),
            ("loops 3 deep", [2, 2, 5]),
            ("loops 6 deep", [2, 2, 8]),
        ]
    );
}
