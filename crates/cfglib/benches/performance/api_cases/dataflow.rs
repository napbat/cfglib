use std::convert::Infallible;

use cfglib::{
    Cfg, Direction, EdgeFacts, EdgeProblem, Facts, Graph, IterationOrder, NodeId, NodeProblem,
    Resume, ResumeMode, Seeds, Solve, SolveConfig, SolveStats, TryEdgeProblem, TryNodeProblem,
    TrySolveError, WeakTopologicalOrder, meet_options, resume_edge_problem, resume_node_problem,
    solve_edge_problem, solve_node_problem, stream_edge_problem, stream_node_problem,
    try_meet_options,
};

use super::BenchmarkSuite;
use crate::fixtures::{CfgReachability, Reachability, branchy_cfg, branchy_graph, fixture_u32};
use crate::harness::benchmark_case;

const NODE_COUNT: usize = 512;

struct TryCfgReachability;

impl TryNodeProblem<Cfg<u32>> for TryCfgReachability {
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
    ($suite:expr, $name:literal, covers[$api:path], $operation:expr) => {
        benchmark_case!($suite, $name, covers[$api], $operation, |result| assert!(
            result.is_ok()
        ));
    };
    ($suite:expr, $name:literal, $operation:expr) => {
        benchmark_case!($suite, $name, $operation, |result| assert!(result.is_ok()));
    };
}

/// The rooted solve of a control-flow graph: what its entry reaches, with
/// the boundary in front of the entry.
fn rooted(cfg: &Cfg<u32>) -> Solve<'static, cfglib::BlockId> {
    Solve {
        seeds: Seeds::Reachable(cfg.entry()),
        ..Solve::default()
    }
}

fn register_cfg_solvers(suite: &mut BenchmarkSuite<'_>) {
    let cfg = branchy_cfg(NODE_COUNT);
    let seeds = [cfg.entry()];
    let seeded = Solve {
        seeds: Seeds::Of(&seeds),
        ..Solve::default()
    };
    let rooted = rooted(&cfg);

    solver_case!(suite, "api_solve_problem_from", || {
        solve_node_problem(&cfg, &CfgReachability, seeded)
    });
    solver_case!(suite, "api_solve_problem_from_with_config", || {
        solve_node_problem(&cfg, &CfgReachability, seeded)
    });
    solver_case!(suite, "api_try_solve_problem", || {
        solve_node_problem(&cfg, &TryCfgReachability, rooted)
    });
    solver_case!(suite, "api_try_solve_problem_from", || {
        solve_node_problem(&cfg, &TryCfgReachability, seeded)
    });
    solver_case!(suite, "api_try_solve_problem_with_config", || {
        solve_node_problem(&cfg, &TryCfgReachability, rooted)
    });
    solver_case!(suite, "api_try_solve_problem_from_with_config", || {
        solve_node_problem(&cfg, &TryCfgReachability, seeded)
    });
}

fn register_node_solvers(suite: &mut BenchmarkSuite<'_>) {
    let graph = branchy_graph(NODE_COUNT);
    let seeds = [NodeId::from_raw(0)];
    let seeded = Solve {
        seeds: Seeds::Of(&seeds),
        ..Solve::default()
    };

    solver_case!(suite, "api_solve_node_problem_from", || {
        solve_node_problem(&graph, &Reachability, seeded)
    });
    solver_case!(suite, "api_solve_node_problem_from_with_config", || {
        solve_node_problem(&graph, &Reachability, seeded)
    });
    solver_case!(suite, "api_try_solve_node_problem", || {
        solve_node_problem(&graph, &TryGraphReachability, Solve::default())
    });
    solver_case!(suite, "api_try_solve_node_problem_from", || {
        solve_node_problem(&graph, &TryGraphReachability, seeded)
    });
    solver_case!(suite, "api_try_solve_node_problem_with_config", || {
        solve_node_problem(&graph, &TryGraphReachability, Solve::default())
    });
    solver_case!(suite, "api_try_solve_node_problem_from_with_config", || {
        solve_node_problem(&graph, &TryGraphReachability, seeded)
    });
}

fn register_edge_solvers(suite: &mut BenchmarkSuite<'_>) {
    let graph = branchy_graph(NODE_COUNT);
    let seeds = [NodeId::from_raw(0)];
    let seeded = Solve {
        seeds: Seeds::Of(&seeds),
        ..Solve::default()
    };

    solver_case!(
        suite,
        "api_solve_edge_problem",
        covers[solve_edge_problem],
        || solve_edge_problem(&graph, &EdgeReachability, Solve::default())
    );
    solver_case!(suite, "api_solve_edge_problem_from", || {
        solve_edge_problem(&graph, &EdgeReachability, seeded)
    });
    solver_case!(suite, "api_solve_edge_problem_with_config", || {
        solve_edge_problem(&graph, &EdgeReachability, Solve::default())
    });
    solver_case!(suite, "api_solve_edge_problem_from_with_config", || {
        solve_edge_problem(&graph, &EdgeReachability, seeded)
    });
    solver_case!(suite, "api_try_solve_edge_problem", || {
        solve_edge_problem(&graph, &TryEdgeReachability, Solve::default())
    });
    solver_case!(suite, "api_try_solve_edge_problem_from", || {
        solve_edge_problem(&graph, &TryEdgeReachability, seeded)
    });
    solver_case!(suite, "api_try_solve_edge_problem_with_config", || {
        solve_edge_problem(&graph, &TryEdgeReachability, Solve::default())
    });
    solver_case!(suite, "api_try_solve_edge_problem_from_with_config", || {
        solve_edge_problem(&graph, &TryEdgeReachability, seeded)
    });
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
    benchmark_case!(
        suite,
        "api_try_meet_options",
        covers[try_meet_options],
        || -> Result<Option<u64>, ()> {
            let mut merged: Option<u64> = None;
            for &value in &facts {
                merged =
                    try_meet_options(&merged, &Some(value), |left, right| Ok(*left.min(right)))?;
            }
            Ok(merged)
        },
        |merged: &Result<Option<u64>, ()>| assert_eq!(*merged, Ok(Some(0)))
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

/// What a solve of [`Definitions`] returns.
type Solved = Result<Facts<u64>, TrySolveError<Infallible>>;

/// A solve of every live node in `order`.
fn in_order(order: IterationOrder) -> Solve<'static, NodeId> {
    Solve {
        config: SolveConfig {
            order,
            ..SolveConfig::default()
        },
        ..Solve::default()
    }
}

fn register_iteration_orders(suite: &mut BenchmarkSuite<'_>) {
    let (graph, entry) = permuted_loop_nest(NEST_NODE_COUNT);
    let index = solve_node_problem(&graph, &Definitions, Solve::default())
        .expect("an unbounded solve converges");
    // Every order reaches the facts of the index order. The step count goes
    // next to the time, because the steps are what an order changes.
    let check = |name: &str, facts: &Solved| {
        let facts = facts.as_ref().expect("an unbounded solve converges");
        for node in graph.node_ids() {
            assert_eq!(facts.fact_out(node), index.fact_out(node));
        }
        println!(
            "{name:<36} {:>12} steps  index order {}",
            facts.stats().step_count(),
            index.stats().step_count()
        );
        facts.stats().step_count()
    };
    let reverse_postorder = in_order(IterationOrder::ReversePostorder);
    let weak_topological = in_order(IterationOrder::WeakTopological);

    benchmark_case!(
        suite,
        "api_solve_node_problem_rpo",
        || solve_node_problem(&graph, &Definitions, reverse_postorder),
        |facts: &Solved| {
            check("api_solve_node_problem_rpo", facts);
        }
    );
    benchmark_case!(
        suite,
        "api_solve_node_problem_wto",
        || solve_node_problem(&graph, &Definitions, weak_topological),
        |facts: &Solved| {
            let steps = check("api_solve_node_problem_wto", facts);
            assert!(
                steps < index.stats().step_count(),
                "the weak topological order saves steps"
            );
        }
    );
    benchmark_case!(
        suite,
        "api_weak_topological_order",
        || WeakTopologicalOrder::compute(&graph, entry, Direction::Forward),
        |order: &WeakTopologicalOrder<NodeId>| assert_eq!(order.order().len(), NEST_NODE_COUNT)
    );
}

/// Nodes in the loop nest that the node stream solves.
const STREAM_NODE_COUNT: usize = 100_000;

/// What a stream run hands back to its oracle: its stats, how many nodes it
/// visited, and every OUT fact folded into one word.
type Streamed<E = ()> = Result<(SolveStats, usize, u64), TrySolveError<E>>;

fn register_streams(suite: &mut BenchmarkSuite<'_>) {
    let (nest, _) = permuted_loop_nest(STREAM_NODE_COUNT);
    let weak_topological = in_order(IterationOrder::WeakTopological);
    let kept = solve_node_problem(&nest, &Definitions, weak_topological)
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
            stream_node_problem(&nest, &Definitions, Solve::default(), |node, _, out| {
                visited += 1;
                fold ^= out.rotate_left(fixture_u32(node.index()));
            })
            .map(|stats| (stats, visited, fold))
        },
        |streamed: &Streamed<Infallible>| {
            let (stats, visited, fold) = streamed.as_ref().expect("the stream converges");
            assert_eq!((*visited, *fold), (STREAM_NODE_COUNT, folded));
            assert_eq!(stats.step_count(), kept.stats().step_count());
            // Peak memory goes next to the time: it is what a stream saves.
            println!(
                "{:<36} {:>12} peak live facts  solve keeps {}",
                "api_stream_node_problem",
                stats.peak_count(),
                2 * STREAM_NODE_COUNT
            );
        }
    );

    let cfg = branchy_cfg(NODE_COUNT);
    let rooted = rooted(&cfg);
    benchmark_case!(
        suite,
        "api_stream_problem",
        || {
            let mut visited = 0;
            stream_node_problem(&cfg, &TryCfgReachability, rooted, |_, _, out| {
                visited += usize::from(*out);
            })
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
                Solve::default(),
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

/// Nodes in the chain whose tail a resume re-solves.
const RESUME_NODE_COUNT: usize = 100_000;

/// Nodes at the end of the chain that the edit reaches.
const RESUME_REGION: usize = 100;

fn register_resumes(suite: &mut BenchmarkSuite<'_>) {
    let mut chain = Graph::with_capacity(RESUME_NODE_COUNT, RESUME_NODE_COUNT - 1);
    let nodes: Vec<NodeId> = (0..RESUME_NODE_COUNT).map(|_| chain.add_node(())).collect();
    for pair in nodes.windows(2) {
        chain.add_edge(pair[0], pair[1], ());
    }
    let solved = solve_node_problem(&chain, &TryGraphReachability, Solve::default())
        .expect("a solve converges");
    let changed = [nodes[RESUME_NODE_COUNT - RESUME_REGION]];
    let last = nodes[RESUME_NODE_COUNT - 1];
    let reset = ResumeMode::Reset;
    let edit = Resume {
        changed: &changed,
        mode: reset,
    };
    benchmark_case!(
        suite,
        "api_resume_node_problem",
        covers[resume_node_problem],
        || {
            let previous = solved.clone();
            resume_node_problem(
                &chain,
                &TryGraphReachability,
                previous,
                edit,
                Solve::default(),
            )
        },
        |resumed: &Result<Facts<bool>, TrySolveError<()>>| {
            let resumed = resumed.as_ref().expect("a resume converges");
            assert!(*resumed.fact_out(last));
            assert_eq!(resumed.stats().step_count(), RESUME_REGION);
            // The steps go next to the time: the full solve takes one per
            // node, the resume one per node of the region.
            println!(
                "{:<36} {:>12} steps  full solve {}",
                "api_resume_node_problem",
                resumed.stats().step_count(),
                solved.stats().step_count()
            );
        }
    );

    let cfg = branchy_cfg(NODE_COUNT);
    let rooted = rooted(&cfg);
    let solved_cfg =
        solve_node_problem(&cfg, &TryCfgReachability, rooted).expect("a solve converges");
    let block = [cfg
        .block_ids()
        .nth(NODE_COUNT - 16)
        .expect("the fixture has the block")];
    let edit = Resume {
        changed: &block,
        mode: reset,
    };
    benchmark_case!(
        suite,
        "api_resume_problem",
        || {
            let previous = solved_cfg.clone();
            resume_node_problem(&cfg, &TryCfgReachability, previous, edit, rooted)
        },
        |resumed: &Result<Facts<bool>, TrySolveError<()>>| {
            let resumed = resumed.as_ref().expect("a resume converges");
            assert!(cfg.block_ids().all(|id| *resumed.fact_out(id)));
        }
    );

    let graph = branchy_graph(NODE_COUNT);
    let solved_graph = solve_edge_problem(&graph, &TryEdgeReachability, Solve::default())
        .expect("a solve converges");
    let node = [NodeId::from_raw(fixture_u32(NODE_COUNT - 16))];
    let edit = Resume {
        changed: &node,
        mode: reset,
    };
    benchmark_case!(
        suite,
        "api_resume_edge_problem",
        covers[resume_edge_problem],
        || {
            let previous = solved_graph.clone();
            resume_edge_problem(
                &graph,
                &TryEdgeReachability,
                previous,
                edit,
                Solve::default(),
            )
        },
        |resumed: &Result<EdgeFacts<bool>, TrySolveError<()>>| {
            let resumed = resumed.as_ref().expect("a resume converges");
            assert!(graph.node_ids().all(|id| *resumed.fact_out(id)));
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
    register_resumes(suite);
}
