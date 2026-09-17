//! Node-level fixpoint dataflow over any [`GraphView`].
//!
//! The instruction fixpoint in [`fixpoint`](super::fixpoint) is bound to
//! [`Cfg`](crate::Cfg) blocks; this is its graph-shaped counterpart: one
//! fact per node, meet over the in-edges (out-edges backward), transfer
//! per node. It serves analyses over non-CFG graphs — taint or
//! reachability-with-facts over a value-flow graph, closure over an import
//! or include graph — anywhere the graph is the program representation.
//!
//! [`solve_node_problem`] queues every node; [`solve_node_problem_from`]
//! queues a chosen subset, so an incremental or dirty-region analysis pays
//! for the part of the graph its change actually reaches. The solver carries
//! the same facility matrix as its instruction-level and edge-sensitive
//! counterparts: `_with_config` bounds the solve deterministically via
//! [`SolveConfig`], and the `try_` variants run a fallible
//! [`TryNodeProblem`].

extern crate alloc;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;
use core::convert::Infallible;

use super::fixpoint::{Direction, SolveConfig, SolveError, TrySolveError, collapse_infallible};
use crate::graph::traverse::{Adjacency, Incoming, Outgoing};
use crate::graph::view::{DenseId, GraphView};

/// A node-level dataflow problem over a graph view `G`.
///
/// Termination requires the usual contract: `meet` and `transfer` monotone
/// over a finite-height fact lattice.
pub trait NodeProblem<G: GraphView> {
    /// The per-node dataflow fact.
    type Fact: Clone + PartialEq;

    /// Whether facts flow along edges ([`Direction::Forward`]) or against
    /// them ([`Direction::Backward`]).
    fn direction(&self) -> Direction;

    /// The initial fact for every node.
    fn bottom(&self, graph: &G) -> Self::Fact;

    /// The fact entering boundary nodes (no in-edges forward; no out-edges
    /// backward).
    fn boundary(&self, graph: &G) -> Self::Fact;

    /// Combine two facts at a join point.
    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact;

    /// The fact leaving `node`, given the met fact entering it.
    fn transfer(&self, graph: &G, node: G::NodeId, input: &Self::Fact) -> Self::Fact;
}

/// A fallible node-level dataflow problem over a graph view `G`.
///
/// This is the error-preserving counterpart of [`NodeProblem`], for
/// verification and abstract interpretation where a boundary, merge, or
/// transfer can reject the input graph. The solver reports those consumer
/// errors separately from its own configured step limit.
pub trait TryNodeProblem<G: GraphView> {
    /// The per-node dataflow fact.
    type Fact: Clone + PartialEq;

    /// Consumer error produced by a boundary, merge, or transfer operation.
    type Error;

    /// Whether facts flow along edges ([`Direction::Forward`]) or against
    /// them ([`Direction::Backward`]).
    fn direction(&self) -> Direction;

    /// The initial fact for every node.
    fn bottom(&self, graph: &G) -> Self::Fact;

    /// The fact entering boundary nodes (no in-edges forward; no out-edges
    /// backward).
    ///
    /// # Errors
    ///
    /// Returns a consumer error when constructing the boundary fact fails.
    fn boundary(&self, graph: &G) -> Result<Self::Fact, Self::Error>;

    /// Combine two facts at a join point.
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the facts are incompatible.
    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Result<Self::Fact, Self::Error>;

    /// The fact leaving `node`, given the met fact entering it.
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the node rejects the incoming fact.
    fn transfer(
        &self,
        graph: &G,
        node: G::NodeId,
        input: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error>;
}

/// The solved per-node facts of a [`NodeProblem`].
#[derive(Debug, Clone)]
pub struct NodeFacts<F> {
    input: Vec<F>,
    output: Vec<F>,
    steps: usize,
}

impl<F> NodeFacts<F> {
    /// The met fact entering `node`.
    #[must_use]
    pub fn fact_in<N: DenseId>(&self, node: N) -> &F {
        &self.input[node.index()]
    }

    /// The transferred fact leaving `node`.
    #[must_use]
    pub fn fact_out<N: DenseId>(&self, node: N) -> &F {
        &self.output[node.index()]
    }

    /// Number of worklist entries processed.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }
}

/// Solve a [`NodeProblem`] to fixpoint with a worklist, without a step limit.
///
/// # Errors
///
/// The unbounded configuration cannot produce a solver-limit error. The
/// `Result` matches [`solve_node_problem_with_config`] so callers can switch
/// configurations without changing result handling.
pub fn solve_node_problem<G, P>(graph: &G, problem: &P) -> Result<NodeFacts<P::Fact>, SolveError>
where
    G: GraphView,
    P: NodeProblem<G>,
{
    solve_node_problem_with_config(graph, problem, SolveConfig::new())
}

/// Solve a [`NodeProblem`] with only `seeds` on the initial worklist.
///
/// Every node still starts at `bottom`, but only the seeds — and whatever
/// their transfers reach — are ever visited. That is the difference between
/// re-solving a whole graph and re-solving the part of it that changed: an
/// incremental or dirty-region analysis seeds the nodes whose inputs moved
/// and lets the worklist carry the effect exactly as far as the facts
/// actually travel.
///
/// Seeding every node is exactly [`solve_node_problem`]; duplicate seeds are
/// queued once. With no seeds nothing is visited and every fact stays
/// `bottom` — an empty change set costs an allocation, not a traversal.
///
/// The facts of unvisited nodes are `bottom`, not stale values from a
/// previous solve: this solves a fresh problem from a subset of entry
/// points, it does not resume one. A consumer holding previous results
/// merges them itself, which is why the seeded solve keeps the same
/// `NodeFacts` shape.
///
/// # Examples
///
/// ```
/// use cfglib::{Graph, Direction, NodeId, NodeProblem, solve_node_problem_from};
///
/// // Forward taint over a value-flow graph.
/// struct Taint(Vec<NodeId>);
/// impl NodeProblem<Graph<&'static str, ()>> for Taint {
///     type Fact = bool;
///     fn direction(&self) -> Direction {
///         Direction::Forward
///     }
///     fn bottom(&self, _: &Graph<&'static str, ()>) -> bool {
///         false
///     }
///     fn boundary(&self, _: &Graph<&'static str, ()>) -> bool {
///         false
///     }
///     fn meet(&self, a: &bool, b: &bool) -> bool {
///         *a || *b
///     }
///     fn transfer(
///         &self,
///         _: &Graph<&'static str, ()>,
///         node: NodeId,
///         input: &bool,
///     ) -> bool {
///         *input || self.0.contains(&node)
///     }
/// }
///
/// let mut graph = Graph::<&'static str, ()>::new();
/// let edited = graph.add_node("edited");
/// let downstream = graph.add_node("downstream");
/// let untouched = graph.add_node("untouched");
/// graph.add_edge(edited, downstream, ());
///
/// // Re-solving from the node that changed still reaches everything its
/// // facts flow to, and never visits the rest of the graph.
/// let facts = solve_node_problem_from(&graph, &Taint(vec![edited]), &[edited]).unwrap();
/// assert!(*facts.fact_out(downstream));
/// assert!(!*facts.fact_out(untouched));
/// ```
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`.
///
/// # Errors
///
/// The unbounded configuration cannot produce a solver-limit error.
pub fn solve_node_problem_from<G, P>(
    graph: &G,
    problem: &P,
    seeds: &[G::NodeId],
) -> Result<NodeFacts<P::Fact>, SolveError>
where
    G: GraphView,
    P: NodeProblem<G>,
{
    solve_node_problem_from_with_config(graph, problem, seeds, SolveConfig::new())
}

/// Solve a [`NodeProblem`] with deterministic bounded iteration.
///
/// # Errors
///
/// Returns [`SolveError::StepLimitExceeded`] when work remains at the
/// configured limit.
pub fn solve_node_problem_with_config<G, P>(
    graph: &G,
    problem: &P,
    config: SolveConfig,
) -> Result<NodeFacts<P::Fact>, SolveError>
where
    G: GraphView,
    P: NodeProblem<G>,
{
    let fallible = InfallibleNodeProblem(problem);
    collapse_infallible(try_solve_with_worklist(
        graph,
        &fallible,
        NodeWorklist::all(graph.node_bound()),
        config,
    ))
}

/// Solve a [`NodeProblem`] from `seeds` with a deterministic step limit.
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`.
///
/// # Errors
///
/// Returns [`SolveError::StepLimitExceeded`] when work remains at the
/// configured limit.
pub fn solve_node_problem_from_with_config<G, P>(
    graph: &G,
    problem: &P,
    seeds: &[G::NodeId],
    config: SolveConfig,
) -> Result<NodeFacts<P::Fact>, SolveError>
where
    G: GraphView,
    P: NodeProblem<G>,
{
    let fallible = InfallibleNodeProblem(problem);
    collapse_infallible(try_solve_with_worklist(
        graph,
        &fallible,
        seed_worklist(graph, seeds),
        config,
    ))
}

/// Solve a fallible [`TryNodeProblem`] without a step limit.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error. The unbounded
/// configuration cannot produce a solver-limit error.
pub fn try_solve_node_problem<G, P>(
    graph: &G,
    problem: &P,
) -> Result<NodeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
{
    try_solve_node_problem_with_config(graph, problem, SolveConfig::new())
}

/// Solve a fallible [`TryNodeProblem`] from only the initial `seeds`.
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error. The unbounded
/// configuration cannot produce a solver-limit error.
pub fn try_solve_node_problem_from<G, P>(
    graph: &G,
    problem: &P,
    seeds: &[G::NodeId],
) -> Result<NodeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
{
    try_solve_node_problem_from_with_config(graph, problem, seeds, SolveConfig::new())
}

/// Solve a fallible [`TryNodeProblem`] with a deterministic step limit.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error or
/// [`TrySolveError::Solver`] when work remains at the configured limit.
pub fn try_solve_node_problem_with_config<G, P>(
    graph: &G,
    problem: &P,
    config: SolveConfig,
) -> Result<NodeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
{
    try_solve_with_worklist(
        graph,
        problem,
        NodeWorklist::all(graph.node_bound()),
        config,
    )
}

/// Solve a fallible [`TryNodeProblem`] from `seeds` with a deterministic step
/// limit.
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error or
/// [`TrySolveError::Solver`] when work remains at the configured limit.
pub fn try_solve_node_problem_from_with_config<G, P>(
    graph: &G,
    problem: &P,
    seeds: &[G::NodeId],
    config: SolveConfig,
) -> Result<NodeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
{
    try_solve_with_worklist(graph, problem, seed_worklist(graph, seeds), config)
}

fn seed_worklist<G: GraphView>(graph: &G, seeds: &[G::NodeId]) -> NodeWorklist {
    let mut nodes: Vec<_> = seeds
        .iter()
        .map(|seed| {
            assert!(
                seed.index() < graph.node_bound(),
                "seed node is out of range"
            );
            seed.index()
        })
        .collect();
    nodes.sort_unstable();
    nodes.dedup();
    NodeWorklist::from_sorted(nodes, graph.node_bound())
}

struct NodeWorklist {
    queue: VecDeque<usize>,
    queued: Vec<bool>,
}

impl NodeWorklist {
    fn all(node_count: usize) -> Self {
        Self {
            queue: (0..node_count).collect(),
            queued: vec![true; node_count],
        }
    }

    fn from_sorted(nodes: Vec<usize>, node_count: usize) -> Self {
        let mut queued = vec![false; node_count];
        for &node in &nodes {
            queued[node] = true;
        }
        Self {
            queue: nodes.into_iter().collect(),
            queued,
        }
    }

    fn pop(&mut self) -> Option<usize> {
        let node = self.queue.pop_front()?;
        self.queued[node] = false;
        Some(node)
    }

    fn insert(&mut self, node: usize) {
        if !self.queued[node] {
            self.queued[node] = true;
            self.queue.push_back(node);
        }
    }
}

/// The worklist solver every entry point runs, differing only in which nodes
/// start queued.
fn try_solve_with_worklist<G, P>(
    graph: &G,
    problem: &P,
    worklist: NodeWorklist,
    config: SolveConfig,
) -> Result<NodeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
{
    match problem.direction() {
        Direction::Forward => {
            try_solve_by_axis(graph, problem, Incoming, Outgoing, worklist, config)
        }
        Direction::Backward => {
            try_solve_by_axis(graph, problem, Outgoing, Incoming, worklist, config)
        }
    }
}

fn try_solve_by_axis<G, P, U, D>(
    graph: &G,
    problem: &P,
    upstream: U,
    downstream: D,
    mut worklist: NodeWorklist,
    config: SolveConfig,
) -> Result<NodeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
    U: Adjacency,
    D: Adjacency,
{
    let node_count = graph.node_bound();
    let boundary = problem.boundary(graph).map_err(TrySolveError::Problem)?;
    let mut input = vec![problem.bottom(graph); node_count];
    let mut output = vec![problem.bottom(graph); node_count];

    let mut steps = 0;
    while let Some(node_index) = worklist.pop() {
        if let Some(limit) = config.max_steps() {
            if steps >= limit {
                return Err(SolveError::StepLimitExceeded {
                    limit,
                    steps,
                    pending_node: node_index,
                }
                .into());
            }
        }
        steps += 1;
        let node = G::NodeId::from_index(node_index);

        let mut neighbors = upstream.neighbors(graph, node);
        let met = if let Some(first) = neighbors.next() {
            let mut current = output[first.index()].clone();
            for from in neighbors {
                current = problem
                    .meet(&current, &output[from.index()])
                    .map_err(TrySolveError::Problem)?;
            }
            current
        } else {
            boundary.clone()
        };

        let transferred = problem
            .transfer(graph, node, &met)
            .map_err(TrySolveError::Problem)?;
        input[node_index] = met;
        if transferred != output[node_index] {
            output[node_index] = transferred;
            for next in downstream.neighbors(graph, node) {
                worklist.insert(next.index());
            }
        }
    }

    Ok(NodeFacts {
        input,
        output,
        steps,
    })
}

/// Adapter that runs an infallible [`NodeProblem`] on the fallible solver
/// core.
struct InfallibleNodeProblem<'p, P>(&'p P);

impl<G, P> TryNodeProblem<G> for InfallibleNodeProblem<'_, P>
where
    G: GraphView,
    P: NodeProblem<G>,
{
    type Fact = P::Fact;
    type Error = Infallible;

    fn direction(&self) -> Direction {
        NodeProblem::direction(self.0)
    }

    fn bottom(&self, graph: &G) -> Self::Fact {
        NodeProblem::bottom(self.0, graph)
    }

    fn boundary(&self, graph: &G) -> Result<Self::Fact, Self::Error> {
        Ok(NodeProblem::boundary(self.0, graph))
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Result<Self::Fact, Self::Error> {
        Ok(NodeProblem::meet(self.0, a, b))
    }

    fn transfer(
        &self,
        graph: &G,
        node: G::NodeId,
        input: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(NodeProblem::transfer(self.0, graph, node, input))
    }
}

#[cfg(test)]
mod tests;
