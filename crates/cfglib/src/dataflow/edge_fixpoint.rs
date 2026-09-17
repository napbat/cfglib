//! Edge-sensitive fixpoint dataflow over borrowed graph views.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::convert::Infallible;

use crate::dataflow::fixpoint::{
    Direction, SolveConfig, SolveError, TrySolveError, collapse_infallible,
};
use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::view::DenseId;

/// A dataflow problem whose transfer can distinguish individual edges.
///
/// In forward analysis, `node_input` is the physical pre-state and
/// `node_output` is the physical post-state. An exceptional edge can therefore
/// transfer from `node_input`, while a normal edge uses `node_output`. The same
/// two physical facts are supplied during backward analysis; the default edge
/// transfer uses the fact flowing in the selected analysis direction.
pub trait EdgeProblem<G: EdgeView> {
    /// Lattice element propagated through nodes and edges.
    type Fact: Clone + PartialEq;

    /// Analysis direction.
    fn direction(&self) -> Direction;

    /// Bottom value for nodes and live edges.
    fn bottom(&self, graph: &G) -> Self::Fact;

    /// Optional boundary fact for a node.
    ///
    /// Forward problems normally seed entries; backward problems normally
    /// seed exits. Returning facts for several nodes supports handler entries,
    /// coroutine resumptions, and graphs with multiple roots.
    fn boundary(&self, _graph: &G, _node: G::NodeId) -> Option<Self::Fact> {
        None
    }

    /// Meet/join operator, invoked in stable adjacency order.
    fn meet(&self, left: &Self::Fact, right: &Self::Fact) -> Self::Fact;

    /// Transfer through one node in the selected analysis direction.
    fn transfer_node(&self, graph: &G, node: G::NodeId, flow_fact: &Self::Fact) -> Self::Fact;

    /// Transfer onto one stable edge.
    ///
    /// The borrowed edge exposes its identity, view-oriented endpoints, and
    /// consumer data. Implementations may select either physical node state or
    /// construct a distinct edge fact from payload metadata.
    fn transfer_edge(
        &self,
        _graph: &G,
        _edge: EdgeRef<'_, G::NodeId, G::EdgeId, G::EdgeData>,
        node_input: &Self::Fact,
        node_output: &Self::Fact,
    ) -> Self::Fact {
        match self.direction() {
            Direction::Forward => node_output.clone(),
            Direction::Backward => node_input.clone(),
        }
    }
}

/// A fallible dataflow problem whose transfers can distinguish individual
/// edges.
///
/// This is the error-preserving counterpart of [`EdgeProblem`]. It is intended
/// for verification and abstract interpretation where a transfer or lattice
/// merge can reject the input program. The solver reports those consumer
/// errors separately from its own configured step limit.
pub trait TryEdgeProblem<G: EdgeView> {
    /// Lattice element propagated through nodes and edges.
    type Fact: Clone + PartialEq;

    /// Consumer error produced by a boundary, merge, or transfer operation.
    type Error;

    /// Analysis direction.
    fn direction(&self) -> Direction;

    /// Bottom value for nodes and live edges.
    fn bottom(&self, graph: &G) -> Self::Fact;

    /// Optional boundary fact for a node.
    ///
    /// # Errors
    ///
    /// Returns a consumer error when constructing the boundary fact fails.
    fn boundary(&self, _graph: &G, _node: G::NodeId) -> Result<Option<Self::Fact>, Self::Error> {
        Ok(None)
    }

    /// Meet or join two facts in stable adjacency order.
    ///
    /// `node` is the physical merge point, allowing consumer errors to retain
    /// the exact destination identity or source location.
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the facts are incompatible.
    fn meet(
        &self,
        graph: &G,
        node: G::NodeId,
        left: &Self::Fact,
        right: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error>;

    /// Transfer through one node in the selected analysis direction.
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the node rejects the incoming fact.
    fn transfer_node(
        &self,
        graph: &G,
        node: G::NodeId,
        flow_fact: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error>;

    /// Transfer onto one stable edge.
    ///
    /// The borrowed edge exposes its identity, view-oriented endpoints, and
    /// consumer data. Implementations may select either physical node state or
    /// construct a distinct edge fact from payload metadata.
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the edge rejects the physical node facts.
    fn transfer_edge(
        &self,
        _graph: &G,
        _edge: EdgeRef<'_, G::NodeId, G::EdgeId, G::EdgeData>,
        node_input: &Self::Fact,
        node_output: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(match self.direction() {
            Direction::Forward => node_output.clone(),
            Direction::Backward => node_input.clone(),
        })
    }
}

/// An edge analysis whose facts exist only where execution reaches.
///
/// Verification-style analyses (stack maps, register frames) have no fact
/// at all for code no path has reached yet — bottom is *unreached*, not an
/// empty state. Wrapping an implementation in [`Reachable`] lifts it to a
/// [`TryEdgeProblem`] over `Option` facts with the standard plumbing
/// supplied once: `None` is unreached bottom and the identity of every
/// merge, transfers short-circuit through unreached nodes, entry facts
/// start the flow, and each edge chooses whether it observes the node's
/// pre-state — an exceptional edge leaving before the node's effect
/// commits — or its post-state.
pub trait ReachableEdgeProblem<G: EdgeView> {
    /// Lattice element for reached program points.
    type Fact: Clone + PartialEq;

    /// Consumer error produced by an entry, merge, or transfer operation.
    type Error;

    /// Analysis direction.
    fn direction(&self) -> Direction;

    /// The fact `node` starts with before any flow, when it has one.
    ///
    /// # Errors
    ///
    /// Returns a consumer error when constructing the entry fact fails.
    fn entry_fact(&self, graph: &G, node: G::NodeId) -> Result<Option<Self::Fact>, Self::Error>;

    /// Merges two reached facts at the physical merge point.
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the facts are incompatible.
    fn merge(
        &self,
        graph: &G,
        node: G::NodeId,
        left: &Self::Fact,
        right: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error>;

    /// Transfers one reached fact through `node`.
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

    /// Whether `edge` observes the node's pre-state instead of the
    /// direction-appropriate post-state.
    ///
    /// The classic case is an exceptional edge in a forward analysis: it
    /// leaves before the node's effect commits, so the handler observes the
    /// state the node received. Defaults to `false`.
    fn edge_observes_input(
        &self,
        _graph: &G,
        _edge: EdgeRef<'_, G::NodeId, G::EdgeId, G::EdgeData>,
    ) -> bool {
        false
    }
}

/// The reachability lifting of a [`ReachableEdgeProblem`].
///
/// Pass `Reachable(problem)` to [`try_solve_edge_problem`] or its seeded
/// variants; the resulting facts are `Option`s whose `None` means the
/// point was never reached.
#[derive(Debug, Clone, Copy)]
pub struct Reachable<P>(pub P);

impl<G: EdgeView, P: ReachableEdgeProblem<G>> TryEdgeProblem<G> for Reachable<P> {
    type Fact = Option<P::Fact>;
    type Error = P::Error;

    fn direction(&self) -> Direction {
        self.0.direction()
    }

    fn bottom(&self, _graph: &G) -> Self::Fact {
        None
    }

    fn boundary(&self, graph: &G, node: G::NodeId) -> Result<Option<Self::Fact>, Self::Error> {
        Ok(self.0.entry_fact(graph, node)?.map(Some))
    }

    fn meet(
        &self,
        graph: &G,
        node: G::NodeId,
        left: &Self::Fact,
        right: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        match (left, right) {
            (None, None) => Ok(None),
            (Some(fact), None) | (None, Some(fact)) => Ok(Some(fact.clone())),
            (Some(left), Some(right)) => self.0.merge(graph, node, left, right).map(Some),
        }
    }

    fn transfer_node(
        &self,
        graph: &G,
        node: G::NodeId,
        flow_fact: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        match flow_fact {
            None => Ok(None),
            Some(fact) => self.0.transfer(graph, node, fact).map(Some),
        }
    }

    fn transfer_edge(
        &self,
        graph: &G,
        edge: EdgeRef<'_, G::NodeId, G::EdgeId, G::EdgeData>,
        node_input: &Self::Fact,
        node_output: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        if self.0.edge_observes_input(graph, edge) {
            return Ok(node_input.clone());
        }
        Ok(match self.0.direction() {
            Direction::Forward => node_output.clone(),
            Direction::Backward => node_input.clone(),
        })
    }
}

/// Node and edge facts at a completed fixpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeFacts<F> {
    node_input: Vec<F>,
    node_output: Vec<F>,
    edge: Vec<Option<F>>,
    steps: usize,
}

impl<F> EdgeFacts<F> {
    /// Physical input fact for `node`.
    #[must_use]
    pub fn fact_in<N: DenseId>(&self, node: N) -> &F {
        &self.node_input[node.index()]
    }

    /// Physical output fact for `node`.
    #[must_use]
    pub fn fact_out<N: DenseId>(&self, node: N) -> &F {
        &self.node_output[node.index()]
    }

    /// Fact on a live edge in the solved view, or `None` for a tombstone or an
    /// edge excluded by a filtered view.
    #[must_use]
    pub fn fact_on<E: DenseId>(&self, edge: E) -> Option<&F> {
        self.edge.get(edge.index()).and_then(Option::as_ref)
    }

    /// Edge facts indexed by dense edge slot, retaining `None` tombstones.
    #[must_use]
    pub fn edge_facts(&self) -> &[Option<F>] {
        &self.edge
    }

    /// Number of worklist entries processed.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }
}

/// Solve an edge-sensitive problem to a fixpoint without a step limit.
///
/// # Errors
///
/// The unbounded configuration cannot produce a solver-limit error. The
/// `Result` matches [`solve_edge_problem_with_config`] so callers can switch
/// configurations without changing result handling.
pub fn solve_edge_problem<G, P>(graph: &G, problem: &P) -> Result<EdgeFacts<P::Fact>, SolveError>
where
    G: EdgeView,
    P: EdgeProblem<G>,
{
    solve_edge_problem_with_config(graph, problem, SolveConfig::new())
}

/// Solve an edge-sensitive problem from only the initial `seeds`.
///
/// Every fact starts at bottom, but only the seeds and nodes reached by changed
/// edge facts enter the deterministic worklist. Duplicate seeds are ignored.
/// This supports reachable-only verification and incremental analysis without
/// manufacturing facts for disconnected nodes.
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`.
///
/// # Errors
///
/// The unbounded configuration cannot produce a solver-limit error.
pub fn solve_edge_problem_from<G, P>(
    graph: &G,
    problem: &P,
    seeds: &[G::NodeId],
) -> Result<EdgeFacts<P::Fact>, SolveError>
where
    G: EdgeView,
    P: EdgeProblem<G>,
{
    solve_edge_problem_from_with_config(graph, problem, seeds, SolveConfig::new())
}

/// Solve an edge-sensitive problem with deterministic bounded iteration.
///
/// # Errors
///
/// Returns [`SolveError::StepLimitExceeded`] when work remains at the
/// configured limit.
pub fn solve_edge_problem_with_config<G, P>(
    graph: &G,
    problem: &P,
    config: SolveConfig,
) -> Result<EdgeFacts<P::Fact>, SolveError>
where
    G: EdgeView,
    P: EdgeProblem<G>,
{
    let fallible = InfallibleProblem(problem);
    collapse_infallible(try_solve_with_worklist(
        graph,
        &fallible,
        (0..graph.node_bound()).collect(),
        config,
    ))
}

/// Solve an edge-sensitive problem from `seeds` with a deterministic step
/// limit.
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`.
///
/// # Errors
///
/// Returns [`SolveError::StepLimitExceeded`] when work remains at the
/// configured limit.
pub fn solve_edge_problem_from_with_config<G, P>(
    graph: &G,
    problem: &P,
    seeds: &[G::NodeId],
    config: SolveConfig,
) -> Result<EdgeFacts<P::Fact>, SolveError>
where
    G: EdgeView,
    P: EdgeProblem<G>,
{
    let fallible = InfallibleProblem(problem);
    collapse_infallible(try_solve_with_worklist(
        graph,
        &fallible,
        seed_worklist(graph, seeds),
        config,
    ))
}

/// Solve a fallible edge-sensitive problem without a step limit.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error. The unbounded
/// configuration cannot produce a solver-limit error.
pub fn try_solve_edge_problem<G, P>(
    graph: &G,
    problem: &P,
) -> Result<EdgeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
{
    try_solve_edge_problem_with_config(graph, problem, SolveConfig::new())
}

/// Solve a fallible edge-sensitive problem from only the initial `seeds`.
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error. The unbounded
/// configuration cannot produce a solver-limit error.
pub fn try_solve_edge_problem_from<G, P>(
    graph: &G,
    problem: &P,
    seeds: &[G::NodeId],
) -> Result<EdgeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
{
    try_solve_edge_problem_from_with_config(graph, problem, seeds, SolveConfig::new())
}

/// Solve a fallible edge-sensitive problem with a deterministic step limit.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error or
/// [`TrySolveError::Solver`] when work remains at the configured limit.
pub fn try_solve_edge_problem_with_config<G, P>(
    graph: &G,
    problem: &P,
    config: SolveConfig,
) -> Result<EdgeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
{
    try_solve_with_worklist(graph, problem, (0..graph.node_bound()).collect(), config)
}

/// Solve a fallible edge-sensitive problem from `seeds` with a deterministic
/// step limit.
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error or
/// [`TrySolveError::Solver`] when work remains at the configured limit.
pub fn try_solve_edge_problem_from_with_config<G, P>(
    graph: &G,
    problem: &P,
    seeds: &[G::NodeId],
    config: SolveConfig,
) -> Result<EdgeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
{
    try_solve_with_worklist(graph, problem, seed_worklist(graph, seeds), config)
}

fn seed_worklist<G: EdgeView>(graph: &G, seeds: &[G::NodeId]) -> BTreeSet<usize> {
    seeds
        .iter()
        .map(|seed| {
            assert!(
                seed.index() < graph.node_bound(),
                "seed node is out of range"
            );
            seed.index()
        })
        .collect()
}

fn try_solve_with_worklist<G, P>(
    graph: &G,
    problem: &P,
    mut worklist: BTreeSet<usize>,
    config: SolveConfig,
) -> Result<EdgeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
{
    let bottom = problem.bottom(graph);
    let mut node_input = vec![bottom.clone(); graph.node_bound()];
    let mut node_output = vec![bottom.clone(); graph.node_bound()];
    let mut edge = vec![None; graph.edge_bound()];
    for edge_id in graph.edge_ids() {
        edge[edge_id.index()] = Some(bottom.clone());
    }

    let mut steps = 0;
    while let Some(node_index) = worklist.pop_first() {
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

        match problem.direction() {
            Direction::Forward => try_solve_forward_node(
                graph,
                problem,
                node,
                &bottom,
                &mut node_input,
                &mut node_output,
                &mut edge,
                &mut worklist,
            ),
            Direction::Backward => try_solve_backward_node(
                graph,
                problem,
                node,
                &bottom,
                &mut node_input,
                &mut node_output,
                &mut edge,
                &mut worklist,
            ),
        }
        .map_err(TrySolveError::Problem)?;
    }

    Ok(EdgeFacts {
        node_input,
        node_output,
        edge,
        steps,
    })
}

fn try_meet_edges<G, P>(
    graph: &G,
    problem: &P,
    node: G::NodeId,
    bottom: &P::Fact,
    edge_facts: &[Option<P::Fact>],
    incoming: bool,
) -> Result<P::Fact, P::Error>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
{
    let mut merged = problem.boundary(graph, node)?;
    let edges: Vec<_> = if incoming {
        graph.incoming(node).collect()
    } else {
        graph.outgoing(node).collect()
    };
    for edge in edges {
        let fact = edge_facts[edge.index()]
            .as_ref()
            .expect("adjacency contains an edge excluded from the view");
        merged = Some(match merged {
            Some(current) => problem.meet(graph, node, &current, fact)?,
            None => fact.clone(),
        });
    }
    Ok(merged.unwrap_or_else(|| bottom.clone()))
}

// The solver's per-node step borrows each state slice separately so the
// worklist loop can keep disjoint mutable borrows; bundling them into a
// struct would force the loop to re-borrow the whole solver state.
#[allow(clippy::too_many_arguments)]
fn try_solve_forward_node<G, P>(
    graph: &G,
    problem: &P,
    node: G::NodeId,
    bottom: &P::Fact,
    node_input: &mut [P::Fact],
    node_output: &mut [P::Fact],
    edge_facts: &mut [Option<P::Fact>],
    worklist: &mut BTreeSet<usize>,
) -> Result<(), P::Error>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
{
    let input = try_meet_edges(graph, problem, node, bottom, edge_facts, true)?;
    let output = problem.transfer_node(graph, node, &input)?;
    node_input[node.index()] = input;
    node_output[node.index()] = output;

    let outgoing: Vec<_> = graph.outgoing(node).collect();
    for edge_id in outgoing {
        let edge = graph.edge(edge_id);
        let new_fact = problem.transfer_edge(
            graph,
            edge,
            &node_input[node.index()],
            &node_output[node.index()],
        )?;
        let fact = edge_facts[edge_id.index()]
            .as_mut()
            .expect("adjacency contains an edge excluded from the view");
        if new_fact != *fact {
            *fact = new_fact;
            worklist.insert(edge.target().index());
        }
    }
    Ok(())
}

// Same disjoint-borrow constraint as `try_solve_forward_node`.
#[allow(clippy::too_many_arguments)]
fn try_solve_backward_node<G, P>(
    graph: &G,
    problem: &P,
    node: G::NodeId,
    bottom: &P::Fact,
    node_input: &mut [P::Fact],
    node_output: &mut [P::Fact],
    edge_facts: &mut [Option<P::Fact>],
    worklist: &mut BTreeSet<usize>,
) -> Result<(), P::Error>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
{
    let output = try_meet_edges(graph, problem, node, bottom, edge_facts, false)?;
    let input = problem.transfer_node(graph, node, &output)?;
    node_output[node.index()] = output;
    node_input[node.index()] = input;

    let incoming: Vec<_> = graph.incoming(node).collect();
    for edge_id in incoming {
        let edge = graph.edge(edge_id);
        let new_fact = problem.transfer_edge(
            graph,
            edge,
            &node_input[node.index()],
            &node_output[node.index()],
        )?;
        let fact = edge_facts[edge_id.index()]
            .as_mut()
            .expect("adjacency contains an edge excluded from the view");
        if new_fact != *fact {
            *fact = new_fact;
            worklist.insert(edge.source().index());
        }
    }
    Ok(())
}

struct InfallibleProblem<'problem, P>(&'problem P);

impl<G, P> TryEdgeProblem<G> for InfallibleProblem<'_, P>
where
    G: EdgeView,
    P: EdgeProblem<G>,
{
    type Fact = P::Fact;
    type Error = Infallible;

    fn direction(&self) -> Direction {
        EdgeProblem::direction(self.0)
    }

    fn bottom(&self, graph: &G) -> Self::Fact {
        EdgeProblem::bottom(self.0, graph)
    }

    fn boundary(&self, graph: &G, node: G::NodeId) -> Result<Option<Self::Fact>, Self::Error> {
        Ok(EdgeProblem::boundary(self.0, graph, node))
    }

    fn meet(
        &self,
        _graph: &G,
        _node: G::NodeId,
        left: &Self::Fact,
        right: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(EdgeProblem::meet(self.0, left, right))
    }

    fn transfer_node(
        &self,
        graph: &G,
        node: G::NodeId,
        flow_fact: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(EdgeProblem::transfer_node(self.0, graph, node, flow_fact))
    }

    fn transfer_edge(
        &self,
        graph: &G,
        edge: EdgeRef<'_, G::NodeId, G::EdgeId, G::EdgeData>,
        node_input: &Self::Fact,
        node_output: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(EdgeProblem::transfer_edge(
            self.0,
            graph,
            edge,
            node_input,
            node_output,
        ))
    }
}

#[cfg(test)]
mod tests;
