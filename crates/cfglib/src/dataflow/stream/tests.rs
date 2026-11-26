//! When a stream frees facts (Kim, Venet and Thakur, *Memory-efficient
//! fixpoint computation*, FMSD 2025, Fig. 1), and what it hands out.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::convert::Infallible;

use super::{StreamedNodes, stream_edge_problem, stream_node_problem};
use crate::dataflow::edge_fixpoint::TryEdgeProblem;
use crate::dataflow::fixpoint::{IterationOrder, SolveConfig, SolveError, TrySolveError};
use crate::dataflow::node_fixpoint::{TryNodeProblem, try_solve_node_problem_with_config};
use crate::graph::store::{Graph, NodeId};
use crate::graph::traverse::Outgoing;
use crate::graph::view::Direction;
use crate::graph::weak_topological_order::WeakTopologicalOrder;

fn node(index: usize) -> NodeId {
    NodeId::from_index(index)
}

/// A graph over nodes `0..count` with `edges` added in order.
fn graph(count: usize, edges: &[(usize, usize)]) -> Graph<(), ()> {
    let mut graph = Graph::new();
    for _ in 0..count {
        graph.add_node(());
    }
    for &(source, target) in edges {
        graph.add_edge(node(source), node(target), ());
    }
    graph
}

/// The graphs of FMSD 2025 Fig. 1, nodes `1..=9` at indices `1..=9`,
/// successors listed so the walk gives the paper's orders.
fn figure_one(loops: &[(usize, usize)]) -> Graph<(), ()> {
    let mut edges = vec![
        (1, 2),
        (2, 3),
        (3, 7),
        (3, 4),
        (4, 6),
        (4, 5),
        (7, 9),
        (7, 8),
    ];
    edges.extend_from_slice(loops);
    graph(10, &edges)
}

/// For each node of the order, the node whose instruction frees its OUT
/// fact in the paper's terms: the head of the outermost element (or the
/// loose node) after which the stream releases it.
fn release_points(graph: &Graph<(), ()>) -> Vec<(usize, usize)> {
    let order = WeakTopologicalOrder::compute(graph, node(1), Direction::Forward);
    let nodes: StreamedNodes<(), _> = StreamedNodes::new(
        graph,
        &order,
        Outgoing,
        (),
        Direction::Forward,
        |_: NodeId, &(), &()| {},
    );
    let mut points = Vec::new();
    for end in 1..=order.order().len() {
        let row = nodes.release_start[end] as usize..nodes.release_start[end + 1] as usize;
        let releaser = order.order()[order.outermost(end - 1).start];
        points.extend(
            nodes.release[row]
                .iter()
                .map(|&index| (index as usize, releaser.index())),
        );
    }
    points.sort_unstable();
    points
}

/// The release points of FMSD 2025, Examples 22 to 24, for nodes 1 to 9.
fn paper(points: [usize; 9]) -> Vec<(usize, usize)> {
    (1..=9).zip(points).collect()
}

#[test]
fn release_points_of_the_three_graphs_of_figure_one_match_the_paper() {
    let g1 = figure_one(&[]);
    assert_eq!(release_points(&g1), paper([2, 3, 7, 6, 5, 6, 9, 8, 9]));

    let g2 = figure_one(&[(2, 8), (5, 4), (8, 7)]);
    assert_eq!(release_points(&g2), paper([2, 7, 7, 6, 4, 6, 9, 7, 9]));

    // The paper frees Post[4] after Inst[6] in every iteration of the loop
    // headed by 3 (Dpost(4) = 6), because Bourdoncle's recursive strategy
    // recomputes 4 before 5 and 6 read it again. A worklist solve can revisit
    // 5 or 6 without 4, so it keeps OUT[4] until the loop of 3 is final.
    let g3 = figure_one(&[(2, 8), (5, 3), (5, 4), (6, 3), (8, 7)]);
    assert_eq!(release_points(&g3), paper([2, 7, 7, 3, 3, 3, 9, 7, 9]));
}

/// Forward reachability with a gen set: what reaches a node, as a bit row.
struct Reaching;

impl TryNodeProblem<Graph<(), ()>> for Reaching {
    type Fact = u64;
    type Error = usize;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &Graph<(), ()>) -> u64 {
        0
    }

    fn boundary(&self, _graph: &Graph<(), ()>) -> Result<u64, usize> {
        Ok(0)
    }

    fn meet(&self, _: &Graph<(), ()>, _: NodeId, a: &u64, b: &u64) -> Result<u64, usize> {
        Ok(a | b)
    }

    fn transfer(&self, _: &Graph<(), ()>, node: NodeId, input: &u64) -> Result<u64, usize> {
        if node.index() == 7 {
            return Err(7);
        }
        Ok(input | 1 << node.index())
    }
}

#[test]
fn every_node_is_visited_once_with_the_facts_of_the_solve_that_keeps_them() {
    // Two nested loops, a self-loop and an unreachable pair.
    let graph = graph(
        7,
        &[
            (0, 1),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 3),
            (3, 0),
            (3, 4),
            (5, 6),
            (6, 5),
        ],
    );
    for passes in [0, 2] {
        let config = SolveConfig::new()
            .with_order(IterationOrder::WeakTopological)
            .with_narrowing_passes(passes);
        let expected = try_solve_node_problem_with_config(&graph, &Reaching, config).unwrap();
        let mut seen = Vec::new();
        // The stream runs in the weak topological order whatever the config
        // names; the passes carry over.
        let streamed = SolveConfig::new().with_narrowing_passes(passes);
        let stats = stream_node_problem(&graph, &Reaching, streamed, |id, input, output| {
            assert_eq!(input, expected.fact_in(id));
            assert_eq!(output, expected.fact_out(id));
            seen.push(id.index());
        })
        .unwrap();
        assert_eq!(stats.steps(), expected.steps());
        seen.sort_unstable();
        assert_eq!(seen, [0, 1, 2, 3, 4, 5, 6]);
        assert!(stats.peak_live_facts() < 2 * 7);
    }
}

#[test]
fn an_error_ends_the_stream_after_the_nodes_already_final() {
    // 0 -> 1 -> 7: node 7 rejects its input, after 0 and 1 are final.
    let graph = graph(8, &[(0, 1), (1, 7)]);
    let mut seen = Vec::new();
    let error = stream_node_problem(&graph, &Reaching, SolveConfig::new(), |id, _, _| {
        seen.push(id.index());
    })
    .unwrap_err();
    assert_eq!(error, TrySolveError::Problem(7));
    assert!(seen.contains(&0) && seen.contains(&1) && !seen.contains(&7));

    let limited = SolveConfig::with_step_limit(1);
    let error = stream_node_problem(&graph, &Reaching, limited, |_, _, _| {}).unwrap_err();
    assert!(matches!(
        error,
        TrySolveError::Solver(SolveError::StepLimitExceeded { limit: 1, .. })
    ));
}

/// Reachability carried on edges, with every node a boundary of its own
/// bit so that every edge fact differs from bottom.
struct Marks;

impl TryEdgeProblem<Graph<(), ()>> for Marks {
    type Fact = u64;
    type Error = Infallible;

    fn direction(&self) -> Direction {
        Direction::Backward
    }

    fn bottom(&self, _graph: &Graph<(), ()>) -> u64 {
        0
    }

    fn boundary(&self, _: &Graph<(), ()>, node: NodeId) -> Result<Option<u64>, Infallible> {
        Ok(Some(1 << (node.index() % 64)))
    }

    fn meet(&self, _: &Graph<(), ()>, _: NodeId, a: &u64, b: &u64) -> Result<u64, Infallible> {
        Ok(a | b)
    }

    fn transfer_node(&self, _: &Graph<(), ()>, _: NodeId, flow: &u64) -> Result<u64, Infallible> {
        Ok(*flow)
    }
}

#[test]
fn an_edge_stream_frees_node_and_edge_facts_along_a_chain() {
    let count = 500;
    let edges: Vec<(usize, usize)> = (1..count).map(|index| (index - 1, index)).collect();
    let graph = graph(count, &edges);
    let mut visited = 0;
    let stats = stream_edge_problem(&graph, &Marks, SolveConfig::new(), |id, input, _| {
        // Backward: a node's input is what it transferred, every later bit.
        let expected = (id.index()..count).fold(0_u64, |bits, at| bits | 1 << (at % 64));
        assert_eq!(*input, expected);
        visited += 1;
    })
    .unwrap();
    assert_eq!(visited, count);
    // At most the facts of the node being solved, the edge it reads and the
    // edge it writes: a constant, whatever the length of the chain.
    assert_eq!(stats.peak_live_facts(), 4, "{stats:?}");
}
