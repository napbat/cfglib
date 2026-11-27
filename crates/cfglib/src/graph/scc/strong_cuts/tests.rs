use super::*;
use crate::graph::scc::{SccDecomposition, SccNumbering};
use crate::graph::store::{Graph, NodeId};
use crate::test_util::rng::Rng;
use alloc::vec;

/// The graph over `count` nodes with `edges`, by node index.
fn graph_of(count: usize, edges: &[(usize, usize)]) -> Graph<(), ()> {
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..count).map(|_| graph.add_node(())).collect();
    for &(from, to) in edges {
        graph.add_edge(nodes[from], nodes[to], ());
    }
    graph
}

/// Whether what is left of a component is strongly connected, counting its
/// components with the leaves-first decomposition: the component's nodes are those `local`
/// numbers, less `removed_node`, and its edges are those of `edges` between
/// them, less the edge at index `removed_edge`.
fn strongly_connected(
    local: &[Option<usize>],
    size: usize,
    edges: &[(usize, usize)],
    removed_node: Option<usize>,
    removed_edge: Option<usize>,
) -> bool {
    let kept = |node: usize| local[node].filter(|_| Some(node) != removed_node);
    let rest: Vec<(usize, usize)> = edges
        .iter()
        .enumerate()
        .filter(|&(index, _)| Some(index) != removed_edge)
        .filter_map(|(_, &(from, to))| Some((kept(from)?, kept(to)?)))
        .collect();
    let components = SccDecomposition::compute(&graph_of(size, &rest), SccNumbering::LeavesFirst)
        .component_count();
    // The removed node stays behind as a component of its own.
    components <= 1 + usize::from(removed_node.is_some())
}

/// The strong cuts of every component, from their definition: delete each
/// node, and each edge occurrence, and recount the components of what is
/// left of the component.
fn brute_force(count: usize, edges: &[(usize, usize)]) -> StrongCuts<NodeId> {
    let graph = graph_of(count, edges);
    let mut cuts = StrongCuts::default();
    for component in SccDecomposition::compute(&graph, SccNumbering::LeavesFirst).components() {
        if component.len() < 2 {
            continue;
        }
        let mut local = vec![None; count];
        for (position, node) in component.iter().enumerate() {
            local[node.index()] = Some(position);
        }
        for &node in component {
            if !strongly_connected(&local, component.len(), edges, Some(node.index()), None) {
                cuts.articulation_points.push(node);
            }
        }
        for (index, &(from, to)) in edges.iter().enumerate() {
            if local[from].is_some()
                && local[to].is_some()
                && !strongly_connected(&local, component.len(), edges, None, Some(index))
            {
                cuts.bridges
                    .push((NodeId::from_index(from), NodeId::from_index(to)));
            }
        }
    }
    sort(&mut cuts);
    cuts
}

/// The cuts of every component of `graph`, merged: the cuts of the graph.
fn all_cuts(
    graph: &Graph<(), ()>,
    numbering: SccNumbering,
    mut cuts_of: impl FnMut(&[NodeId]) -> StrongCuts<NodeId>,
) -> StrongCuts<NodeId> {
    let mut cuts = StrongCuts::default();
    for component in SccDecomposition::compute(graph, numbering).components() {
        let piece = cuts_of(component);
        cuts.articulation_points.extend(piece.articulation_points);
        cuts.bridges.extend(piece.bridges);
    }
    sort(&mut cuts);
    cuts
}

/// The cuts of every component of `graph` over one scratch.
fn cuts_in(graph: &Graph<(), ()>, scratch: &mut StrongCutScratch) -> StrongCuts<NodeId> {
    all_cuts(graph, SccNumbering::LeavesFirst, |component| {
        StrongCuts::compute_in(scratch, graph, component)
    })
}

/// Check both entry points, over both numberings, on one graph against the
/// brute force.
fn check(count: usize, edges: &[(usize, usize)], scratch: &mut StrongCutScratch) {
    let graph = graph_of(count, edges);
    let expected = brute_force(count, edges);
    assert_eq!(
        cuts_in(&graph, scratch),
        expected,
        "{count} nodes, {edges:?}"
    );
    let fresh = all_cuts(&graph, SccNumbering::SourcesFirst, |component| {
        StrongCuts::compute(&graph, component)
    });
    assert_eq!(fresh, expected, "fresh buffers: {count} nodes, {edges:?}");
}

/// Visit every graph over `count` nodes in which each ordered pair, self-loops
/// included, is an edge `0..=copies` times.
fn for_each_multigraph(count: usize, copies: usize, mut visit: impl FnMut(&[(usize, usize)])) {
    let mut multiplicity = vec![0; count * count];
    loop {
        let edges: Vec<(usize, usize)> = multiplicity
            .iter()
            .enumerate()
            .flat_map(|(pair, &times)| core::iter::repeat_n((pair / count, pair % count), times))
            .collect();
        visit(&edges);
        let Some(digit) = multiplicity.iter().position(|&times| times < copies) else {
            return;
        };
        multiplicity[..digit].fill(0);
        multiplicity[digit] += 1;
    }
}

#[test]
fn every_digraph_of_up_to_four_nodes_matches_the_definition() {
    let mut scratch = StrongCutScratch::default();
    for count in 1..=4 {
        for_each_multigraph(count, 1, |edges| check(count, edges, &mut scratch));
    }
}

#[test]
fn every_multigraph_of_up_to_three_nodes_matches_the_definition() {
    let mut scratch = StrongCutScratch::default();
    for count in 1..=3 {
        for_each_multigraph(count, 2, |edges| check(count, edges, &mut scratch));
    }
}

#[test]
fn random_sparse_graphs_match_the_definition() {
    let mut rng = Rng::new(0x0091_5eed);
    let mut scratch = StrongCutScratch::default();
    for count in [12, 40, 150, 600, 1_500] {
        for _ in 0..2 {
            // A Hamiltonian cycle keeps most nodes in one large component, and
            // the other edges, about one per node, add parallel edges,
            // self-loops and chords at random.
            let mut order: Vec<usize> = (0..count).collect();
            rng.shuffle(&mut order);
            let mut edges: Vec<(usize, usize)> = order
                .iter()
                .zip(order.iter().cycle().skip(1))
                .map(|(&from, &to)| (from, to))
                .filter(|_| !rng.one_in(8))
                .collect();
            for _ in 0..count {
                edges.push((rng.below(count), rng.below(count)));
            }
            check(count, &edges, &mut scratch);
        }
    }
}

#[test]
fn every_node_and_edge_of_a_simple_cycle_is_a_cut() {
    let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
    let graph = graph_of(4, &edges);
    let cuts = cuts_in(&graph, &mut StrongCutScratch::default());
    assert_eq!(
        cuts.articulation_points,
        (0..4).map(NodeId::from_index).collect::<Vec<_>>()
    );
    assert_eq!(cuts.bridges.len(), 4);
}

#[test]
fn a_two_cycle_has_bridges_but_no_articulation_point() {
    let graph = graph_of(2, &[(0, 1), (1, 0)]);
    let cuts = cuts_in(&graph, &mut StrongCutScratch::default());
    assert_eq!(cuts.articulation_points, []);
    let (a, b) = (NodeId::from_index(0), NodeId::from_index(1));
    assert_eq!(cuts.bridges, [(a, b), (b, a)]);
}

#[test]
fn a_chord_spares_the_node_it_bypasses_and_is_no_bridge_itself() {
    // 0 -> 1 -> 2 -> 3 -> 0, and the chord 0 -> 2 bypasses node 1. Node 1
    // still has one way in and one way out, so both its edges are bridges.
    let graph = graph_of(4, &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]);
    let cuts = cuts_in(&graph, &mut StrongCutScratch::default());
    let node = NodeId::from_index;
    assert_eq!(cuts.articulation_points, [node(0), node(2), node(3)]);
    assert_eq!(
        cuts.bridges,
        [
            (node(0), node(1)),
            (node(1), node(2)),
            (node(2), node(3)),
            (node(3), node(0)),
        ]
    );
}

#[test]
fn only_the_shared_node_of_two_two_cycles_is_an_articulation_point() {
    let graph = graph_of(3, &[(0, 1), (1, 0), (1, 2), (2, 1)]);
    let cuts = cuts_in(&graph, &mut StrongCutScratch::default());
    assert_eq!(cuts.articulation_points, [NodeId::from_index(1)]);
    assert_eq!(cuts.bridges.len(), 4);
}

#[test]
fn a_parallel_edge_is_never_a_strong_bridge() {
    let graph = graph_of(3, &[(0, 1), (0, 1), (1, 2), (2, 0)]);
    let cuts = cuts_in(&graph, &mut StrongCutScratch::default());
    let node = NodeId::from_index;
    assert_eq!(cuts.bridges, [(node(1), node(2)), (node(2), node(0))]);
    assert_eq!(cuts.articulation_points, [node(0), node(1), node(2)]);
}

#[test]
fn separate_components_are_answered_alone() {
    // A 3-cycle, an acyclic node between, and a 2-cycle with a self-loop.
    let graph = graph_of(
        6,
        &[
            (0, 1),
            (1, 2),
            (2, 0),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 4),
            (5, 5),
        ],
    );
    let cuts = cuts_in(&graph, &mut StrongCutScratch::default());
    let node = NodeId::from_index;
    assert_eq!(cuts.articulation_points, [node(0), node(1), node(2)]);
    assert_eq!(
        cuts.bridges,
        [
            (node(0), node(1)),
            (node(1), node(2)),
            (node(2), node(0)),
            (node(4), node(5)),
            (node(5), node(4)),
        ]
    );
}

#[test]
fn a_removed_node_slot_belongs_to_no_component() {
    let mut graph = graph_of(4, &[(0, 1), (1, 3), (3, 0), (1, 2), (2, 1)]);
    assert!(graph.remove_node(NodeId::from_index(2)));
    let cuts = cuts_in(&graph, &mut StrongCutScratch::default());
    let node = NodeId::from_index;
    assert_eq!(cuts.articulation_points, [node(0), node(1), node(3)]);
    assert_eq!(cuts.bridges.len(), 3);
}

#[test]
#[should_panic(expected = "not strongly connected")]
fn a_set_that_is_not_strongly_connected_is_rejected() {
    let graph = graph_of(2, &[(0, 1)]);
    let both = [NodeId::from_index(0), NodeId::from_index(1)];
    let _ = StrongCuts::compute(&graph, &both);
}

#[test]
fn an_empty_graph_and_a_lone_node_have_no_cuts() {
    let graph = graph_of(0, &[]);
    assert_eq!(
        cuts_in(&graph, &mut StrongCutScratch::default()),
        StrongCuts::default()
    );
    assert_eq!(StrongCuts::compute(&graph, &[]), StrongCuts::default());
    let graph = graph_of(1, &[(0, 0)]);
    assert_eq!(
        StrongCuts::compute(&graph, &[NodeId::from_index(0)]),
        StrongCuts::default()
    );
}
