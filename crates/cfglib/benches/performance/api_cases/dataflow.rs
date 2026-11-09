use cfglib::{
    Cfg, Direction, EdgeProblem, Graph, NodeId, SolveConfig, TryEdgeProblem, TryNodeProblem,
    TryProblem, meet_options, solve_edge_problem, solve_edge_problem_from,
    solve_edge_problem_from_with_config, solve_edge_problem_with_config, solve_node_problem_from,
    solve_node_problem_from_with_config, solve_problem_from, solve_problem_from_with_config,
    try_solve_edge_problem, try_solve_edge_problem_from, try_solve_edge_problem_from_with_config,
    try_solve_edge_problem_with_config, try_solve_node_problem, try_solve_node_problem_from,
    try_solve_node_problem_from_with_config, try_solve_node_problem_with_config, try_solve_problem,
    try_solve_problem_from, try_solve_problem_from_with_config, try_solve_problem_with_config,
};

use super::BenchmarkSuite;
use crate::fixtures::{CfgReachability, Reachability, branchy_cfg, branchy_graph};
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

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    register_lattice_helpers(suite);
    register_cfg_solvers(suite);
    register_node_solvers(suite);
    register_edge_solvers(suite);
}
