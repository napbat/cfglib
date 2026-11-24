use cfglib::{
    Cfg, Direction, EdgeProblem, Facts, Graph, IterationOrder, NodeId, NodeProblem, SolveConfig,
    SolveError, SolveStats, TraversalDirection, TryEdgeProblem, TryNodeProblem, TryProblem,
    TrySolveError, WeakTopologicalOrder, meet_options, solve_edge_problem, solve_edge_problem_from,
    solve_edge_problem_from_with_config, solve_edge_problem_with_config, solve_node_problem,
    solve_node_problem_from, solve_node_problem_from_with_config, solve_node_problem_with_config,
    solve_problem_from, solve_problem_from_with_config, stream_edge_problem, stream_node_problem,
    stream_problem, try_solve_edge_problem, try_solve_edge_problem_from,
    try_solve_edge_problem_from_with_config, try_solve_edge_problem_with_config,
    try_solve_node_problem, try_solve_node_problem_from, try_solve_node_problem_from_with_config,
    try_solve_node_problem_with_config, try_solve_problem, try_solve_problem_from,
    try_solve_problem_from_with_config, try_solve_problem_with_config,
};

use super::BenchmarkSuite;
use crate::fixtures::{CfgReachability, Reachability, branchy_cfg, branchy_graph, fixture_u32};
use crate::harness::benchmark_case;

const NODE_COUNT: usize = 512;

struct TryCfgReachability;

impl TryProblem<Cfg<u32>> for TryCfgReachability {
    type Fact = bool;
    type Error = ();

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &Cfg<u32>) -> Self::Fact {
        false
    }

    fn boundary(&self, _graph: &Cfg<u32>) -> Result<Self::Fact, Self::Error> {
        Ok(true)
    }

    fn meet(
        &self,
        _graph: &Cfg<u32>,
        _block: cfglib::BlockId,
        left: &Self::Fact,
        right: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(*left || *right)
    }

    fn transfer(
        &self,
        _cfg: &Cfg<u32>,
        _block: cfglib::BlockId,
        input: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(*input)
    }
}

struct TryGraphReachability;

impl TryNodeProblem<Graph<(), ()>> for TryGraphReachability {
    type Fact = bool;
    type Error = ();

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &Graph<(), ()>) -> Self::Fact {
        false
    }

    fn boundary(&self, _graph: &Graph<(), ()>) -> Result<Self::Fact, Self::Error> {
        Ok(true)
    }

    fn meet(
        &self,
        _graph: &Graph<(), ()>,
        _node: NodeId,
        left: &Self::Fact,
        right: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(*left || *right)
    }

    fn transfer(
        &self,
        _graph: &Graph<(), ()>,
        _node: NodeId,
        input: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(*input)
    }
}

struct EdgeReachability;

impl EdgeProblem<Graph<(), ()>> for EdgeReachability {
    type Fact = bool;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &Graph<(), ()>) -> Self::Fact {
        false
    }

    fn boundary(&self, graph: &Graph<(), ()>, node: NodeId) -> Option<Self::Fact> {
        graph.predecessors(node).next().is_none().then_some(true)
    }

    fn meet(&self, left: &Self::Fact, right: &Self::Fact) -> Self::Fact {
        *left || *right
    }

    fn transfer_node(
        &self,
        _graph: &Graph<(), ()>,
        _node: NodeId,
        flow_fact: &Self::Fact,
    ) -> Self::Fact {
        *flow_fact
    }
}

struct TryEdgeReachability;

impl TryEdgeProblem<Graph<(), ()>> for TryEdgeReachability {
    type Fact = bool;
    type Error = ();

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &Graph<(), ()>) -> Self::Fact {
        false
    }

    fn boundary(
        &self,
        graph: &Graph<(), ()>,
        node: NodeId,
    ) -> Result<Option<Self::Fact>, Self::Error> {
        Ok(graph.predecessors(node).next().is_none().then_some(true))
    }

    fn meet(
        &self,
        _graph: &Graph<(), ()>,
        _node: NodeId,
        left: &Self::Fact,
        right: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(*left || *right)
    }

    fn transfer_node(
        &self,
        _graph: &Graph<(), ()>,
        _node: NodeId,
        flow_fact: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(*flow_fact)
    }
}

macro_rules! solver_case {
    ($suite:expr, $name:literal, $api:path, $operation:expr) => {
        benchmark_case!($suite, $name, covers[$api], $operation, |result| assert!(
            result.is_ok()
        ));
    };
}

fn register_cfg_solvers(suite: &mut BenchmarkSuite<'_>) {
    let cfg = branchy_cfg(NODE_COUNT);
    let seeds = [cfg.entry()];
    let config = SolveConfig::new();

    solver_case!(suite, "api_solve_problem_from", solve_problem_from, || {
        solve_problem_from(&cfg, &CfgReachability, &seeds)
    });
    solver_case!(
        suite,
        "api_solve_problem_from_with_config",
        solve_problem_from_with_config,
        || solve_problem_from_with_config(&cfg, &CfgReachability, &seeds, config)
    );
    solver_case!(suite, "api_try_solve_problem", try_solve_problem, || {
        try_solve_problem(&cfg, &TryCfgReachability)
    });
    solver_case!(
        suite,
        "api_try_solve_problem_from",
        try_solve_problem_from,
        || try_solve_problem_from(&cfg, &TryCfgReachability, &seeds)
    );
    solver_case!(
        suite,
        "api_try_solve_problem_with_config",
        try_solve_problem_with_config,
        || try_solve_problem_with_config(&cfg, &TryCfgReachability, config)
    );
    solver_case!(
        suite,
        "api_try_solve_problem_from_with_config",
        try_solve_problem_from_with_config,
        || try_solve_problem_from_with_config(&cfg, &TryCfgReachability, &seeds, config)
    );
}

fn register_node_solvers(suite: &mut BenchmarkSuite<'_>) {
    let graph = branchy_graph(NODE_COUNT);
    let seeds = [NodeId::from_raw(0)];
    let config = SolveConfig::new();

    solver_case!(
        suite,
        "api_solve_node_problem_from",
        solve_node_problem_from,
        || solve_node_problem_from(&graph, &Reachability, &seeds)
    );
    solver_case!(
        suite,
        "api_solve_node_problem_from_with_config",
        solve_node_problem_from_with_config,
        || solve_node_problem_from_with_config(&graph, &Reachability, &seeds, config)
    );
    solver_case!(
        suite,
        "api_try_solve_node_problem",
        try_solve_node_problem,
        || try_solve_node_problem(&graph, &TryGraphReachability)
    );
    solver_case!(
        suite,
        "api_try_solve_node_problem_from",
        try_solve_node_problem_from,
        || try_solve_node_problem_from(&graph, &TryGraphReachability, &seeds)
    );
    solver_case!(
        suite,
        "api_try_solve_node_problem_with_config",
        try_solve_node_problem_with_config,
        || try_solve_node_problem_with_config(&graph, &TryGraphReachability, config)
    );
    solver_case!(
        suite,
        "api_try_solve_node_problem_from_with_config",
        try_solve_node_problem_from_with_config,
        || try_solve_node_problem_from_with_config(&graph, &TryGraphReachability, &seeds, config)
    );
}

fn register_edge_solvers(suite: &mut BenchmarkSuite<'_>) {
    let graph = branchy_graph(NODE_COUNT);
    let seeds = [NodeId::from_raw(0)];
    let config = SolveConfig::new();

    solver_case!(suite, "api_solve_edge_problem", solve_edge_problem, || {
        solve_edge_problem(&graph, &EdgeReachability)
    });
    solver_case!(
        suite,
        "api_solve_edge_problem_from",
        solve_edge_problem_from,
        || solve_edge_problem_from(&graph, &EdgeReachability, &seeds)
    );
    solver_case!(
        suite,
        "api_solve_edge_problem_with_config",
        solve_edge_problem_with_config,
        || solve_edge_problem_with_config(&graph, &EdgeReachability, config)
    );
    solver_case!(
        suite,
        "api_solve_edge_problem_from_with_config",
        solve_edge_problem_from_with_config,
        || solve_edge_problem_from_with_config(&graph, &EdgeReachability, &seeds, config)
    );
    solver_case!(
        suite,
        "api_try_solve_edge_problem",
        try_solve_edge_problem,
        || try_solve_edge_problem(&graph, &TryEdgeReachability)
    );
    solver_case!(
        suite,
        "api_try_solve_edge_problem_from",
        try_solve_edge_problem_from,
        || try_solve_edge_problem_from(&graph, &TryEdgeReachability, &seeds)
    );
    solver_case!(
        suite,
        "api_try_solve_edge_problem_with_config",
        try_solve_edge_problem_with_config,
        || try_solve_edge_problem_with_config(&graph, &TryEdgeReachability, config)
    );
    solver_case!(
        suite,
        "api_try_solve_edge_problem_from_with_config",
        try_solve_edge_problem_from_with_config,
        || try_solve_edge_problem_from_with_config(&graph, &TryEdgeReachability, &seeds, config,)
    );
}

/// The number of facts the option-meet helper folds together.
const MEET_COUNT: u64 = 1_024;

fn register_lattice_helpers(suite: &mut BenchmarkSuite<'_>) {
    // A heap fixture the optimizer cannot fold the meet loop away against.
    let facts: Vec<u64> = (0..MEET_COUNT).rev().collect();

    benchmark_case!(
        suite,
        "api_meet_options",
        covers[meet_options],
        || {
            let mut merged: Option<u64> = None;
            for &value in &facts {
                merged = meet_options(&merged, &Some(value), |left, right| *left.min(right));
            }
            merged
        },
        |merged: &Option<u64>| assert_eq!(*merged, Some(0))
    );
}

/// Nodes in the permuted loop nest: 128 nests of eight.
const NEST_NODE_COUNT: usize = 1_024;

/// A loop-nest CFG and its entry. In program order, each eight positions
/// hold an outer loop around an inner one, and every fourth nest closes into
/// a third level; the node ids are a fixed pseudo-random permutation of the
/// positions, as after renumbering, merging, or building a value-flow graph
/// in no flow order.
fn permuted_loop_nest(node_count: usize) -> (Graph<(), ()>, NodeId) {
    let mut ids: Vec<usize> = (0..node_count).collect();
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for last in (1..node_count).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bound = u64::try_from(last + 1).expect("a fixture size fits u64");
        let pick = usize::try_from(state % bound).expect("below the fixture size");
        ids.swap(last, pick);
    }
    let mut graph = Graph::with_capacity(node_count, node_count * 2);
    let nodes: Vec<NodeId> = (0..node_count).map(|_| graph.add_node(())).collect();
    let at = |position: usize| nodes[ids[position]];
    for start in (0..node_count).step_by(8) {
        for offset in 0..7 {
            graph.add_edge(at(start + offset), at(start + offset + 1), ());
        }
        graph.add_edge(at(start + 3), at(start + 1), ());
        graph.add_edge(at(start + 6), at(start), ());
        if start + 8 < node_count {
            graph.add_edge(at(start + 7), at(start + 8), ());
        }
        if start % 32 == 24 {
            graph.add_edge(at(start + 7), at(start - 24), ());
        }
    }
    (graph, at(0))
}

/// Definitions reaching each node, one bit per node id modulo 64.
struct Definitions;

impl NodeProblem<Graph<(), ()>> for Definitions {
    type Fact = u64;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &Graph<(), ()>) -> u64 {
        0
    }

    fn boundary(&self, _graph: &Graph<(), ()>) -> u64 {
        0
    }

    fn meet(&self, left: &u64, right: &u64) -> u64 {
        left | right
    }

    fn transfer(&self, _graph: &Graph<(), ()>, node: NodeId, input: &u64) -> u64 {
        input | 1 << (node.index() % 64)
    }
}

fn register_iteration_orders(suite: &mut BenchmarkSuite<'_>) {
    let (graph, entry) = permuted_loop_nest(NEST_NODE_COUNT);
    let index = solve_node_problem(&graph, &Definitions).expect("an unbounded solve converges");
    // Every order reaches the facts of the index order. The step count goes
    // next to the time, because the steps are what an order changes.
    let check = |name: &str, facts: &Result<Facts<u64>, SolveError>| {
        let facts = facts.as_ref().expect("an unbounded solve converges");
        for node in graph.node_ids() {
            assert_eq!(facts.fact_out(node), index.fact_out(node));
        }
        println!(
            "{name:<36} {:>12} steps  index order {}",
            facts.steps(),
            index.steps()
        );
        facts.steps()
    };
    let reverse_postorder = SolveConfig::new().with_order(IterationOrder::ReversePostorder);
    let weak_topological = SolveConfig::new().with_order(IterationOrder::WeakTopological);

    benchmark_case!(
        suite,
        "api_solve_node_problem_rpo",
        || solve_node_problem_with_config(&graph, &Definitions, reverse_postorder),
        |facts: &Result<Facts<u64>, SolveError>| {
            check("api_solve_node_problem_rpo", facts);
        }
    );
    benchmark_case!(
        suite,
        "api_solve_node_problem_wto",
        || solve_node_problem_with_config(&graph, &Definitions, weak_topological),
        |facts: &Result<Facts<u64>, SolveError>| {
            let steps = check("api_solve_node_problem_wto", facts);
            assert!(
                steps < index.steps(),
                "the weak topological order saves steps"
            );
        }
    );
    benchmark_case!(
        suite,
        "api_weak_topological_order",
        || WeakTopologicalOrder::compute(&graph, entry, TraversalDirection::Outgoing),
        |order: &WeakTopologicalOrder<NodeId>| assert_eq!(order.order().len(), NEST_NODE_COUNT)
    );
}

impl TryNodeProblem<Graph<(), ()>> for Definitions {
    type Fact = u64;
    type Error = ();

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &Graph<(), ()>) -> u64 {
        0
    }

    fn boundary(&self, _graph: &Graph<(), ()>) -> Result<u64, ()> {
        Ok(0)
    }

    fn meet(&self, _: &Graph<(), ()>, _: NodeId, left: &u64, right: &u64) -> Result<u64, ()> {
        Ok(left | right)
    }

    fn transfer(&self, _: &Graph<(), ()>, node: NodeId, input: &u64) -> Result<u64, ()> {
        Ok(input | 1 << (node.index() % 64))
    }
}

/// Nodes in the loop nest that the node stream solves.
const STREAM_NODE_COUNT: usize = 100_000;

/// What a stream run hands back to its oracle: its stats, how many nodes it
/// visited, and every OUT fact folded into one word.
type Streamed = Result<(SolveStats, usize, u64), TrySolveError<()>>;

fn register_streams(suite: &mut BenchmarkSuite<'_>) {
    let (nest, _) = permuted_loop_nest(STREAM_NODE_COUNT);
    let weak_topological = SolveConfig::new().with_order(IterationOrder::WeakTopological);
    let kept = try_solve_node_problem_with_config(&nest, &Definitions, weak_topological)
        .expect("an unbounded solve converges");
    let folded = nest.node_ids().fold(0, |folded, node| {
        folded ^ kept.fact_out(node).rotate_left(fixture_u32(node.index()))
    });
    benchmark_case!(
        suite,
        "api_stream_node_problem",
        covers[stream_node_problem],
        || {
            let (mut visited, mut fold) = (0, 0_u64);
            let config = SolveConfig::new();
            stream_node_problem(&nest, &Definitions, config, |node, _, out| {
                visited += 1;
                fold ^= out.rotate_left(fixture_u32(node.index()));
            })
            .map(|stats| (stats, visited, fold))
        },
        |streamed: &Streamed| {
            let (stats, visited, fold) = streamed.as_ref().expect("the stream converges");
            assert_eq!((*visited, *fold), (STREAM_NODE_COUNT, folded));
            assert_eq!(stats.steps(), kept.steps());
            // Peak memory goes next to the time: it is what a stream saves.
            println!(
                "{:<36} {:>12} peak live facts  solve keeps {}",
                "api_stream_node_problem",
                stats.peak_live_facts(),
                2 * STREAM_NODE_COUNT
            );
        }
    );

    let cfg = branchy_cfg(NODE_COUNT);
    benchmark_case!(
        suite,
        "api_stream_problem",
        covers[stream_problem],
        || {
            let mut visited = 0;
            stream_problem(
                &cfg,
                &TryCfgReachability,
                SolveConfig::new(),
                |_, _, out| {
                    visited += usize::from(*out);
                },
            )
            .map(|stats| (stats, visited, 0))
        },
        |streamed: &Streamed| {
            let (_, visited, _) = streamed.as_ref().expect("the stream converges");
            assert_eq!(*visited, NODE_COUNT);
        }
    );

    let graph = branchy_graph(NODE_COUNT);
    benchmark_case!(
        suite,
        "api_stream_edge_problem",
        covers[stream_edge_problem],
        || {
            let mut visited = 0;
            stream_edge_problem(
                &graph,
                &TryEdgeReachability,
                SolveConfig::new(),
                |_, _, out| {
                    visited += usize::from(*out);
                },
            )
            .map(|stats| (stats, visited, 0))
        },
        |streamed: &Streamed| {
            let (_, visited, _) = streamed.as_ref().expect("the stream converges");
            assert_eq!(*visited, NODE_COUNT);
        }
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    register_lattice_helpers(suite);
    register_cfg_solvers(suite);
    register_node_solvers(suite);
    register_edge_solvers(suite);
    register_iteration_orders(suite);
    register_streams(suite);
}
