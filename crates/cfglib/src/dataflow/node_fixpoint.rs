//! Node-level fixpoint dataflow over any [`GraphView`].
//!
//! The instruction fixpoint in [`fixpoint`](super::fixpoint) solves one fact
//! per block of a rooted view; this is its graph-shaped counterpart: one
//! fact per node of any [`GraphView`], meet over the in-edges (out-edges
//! backward), transfer per node. It serves analyses over non-CFG graphs —
//! taint or reachability-with-facts over a value-flow graph, closure over an
//! import or include graph — anywhere the graph is the program
//! representation.
//!
//! [`solve_node_problem`] queues every node; [`solve_node_problem_from`]
//! queues a chosen subset, so an incremental or dirty-region analysis pays
//! for the part of the graph its change actually reaches. The solver carries
//! the same facility matrix as its instruction-level and edge-sensitive
//! counterparts: `_with_config` bounds the solve deterministically via
//! [`SolveConfig`], and the `try_` variants run a fallible
//! [`TryNodeProblem`].

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::convert::Infallible;

use super::bits::DenseBits;
use super::fixpoint::{
    Direction, Facts, SolveConfig, SolveError, TrySolveError, collapse_infallible,
};
use super::schedule::{Bound, Changed, Leaving, Sweep, schedule};
use super::worklist::Worklist;
use crate::graph::traverse::{Adjacency, by_axis};
use crate::graph::view::{DenseId, GraphView};

/// A node-level dataflow problem over a graph view `G`.
///
/// # Termination
///
/// In [`IterationOrder::Index`](crate::IterationOrder::Index) and
/// [`IterationOrder::ReversePostorder`](crate::IterationOrder::ReversePostorder),
/// termination requires the usual contract: `meet` and `transfer` monotone
/// over a finite-height fact lattice. In
/// [`IterationOrder::WeakTopological`](crate::IterationOrder::WeakTopological)
/// the solver widens at the head of every loop with [`widen`](Self::widen),
/// so a monotone problem over an infinite-height lattice terminates too, and
/// [narrowing passes](SolveConfig::with_narrowing_passes) can then win back
/// precision with [`narrow`](Self::narrow). The defaults of both leave the
/// facts unchanged.
///
/// # Examples
///
/// An interval counter `i = 0; loop { i += 1 }` climbs forever without
/// widening; widening at the loop head jumps to the unbounded interval.
///
/// ```
/// use cfglib::{
///     Direction, Graph, IterationOrder, NodeId, NodeProblem, SolveConfig, meet_options,
///     solve_node_problem_with_config,
/// };
///
/// /// `None` is unreached, `(low, high)` an interval; `u32::MAX` is unbounded.
/// struct Counter {
///     init: NodeId,
///     step: NodeId,
/// }
///
/// type Range = Option<(u32, u32)>;
///
/// impl NodeProblem<Graph<(), ()>> for Counter {
///     type Fact = Range;
///     fn direction(&self) -> Direction {
///         Direction::Forward
///     }
///     fn bottom(&self, _: &Graph<(), ()>) -> Range {
///         None
///     }
///     fn boundary(&self, _: &Graph<(), ()>) -> Range {
///         None
///     }
///     fn meet(&self, a: &Range, b: &Range) -> Range {
///         meet_options(a, b, |a, b| (a.0.min(b.0), a.1.max(b.1)))
///     }
///     fn transfer(&self, _: &Graph<(), ()>, node: NodeId, input: &Range) -> Range {
///         if node == self.init {
///             Some((0, 0))
///         } else if node == self.step {
///             input.map(|(low, high)| (low + 1, high.saturating_add(1)))
///         } else {
///             *input
///         }
///     }
///     fn widen(&self, previous: &Range, next: &Range) -> Range {
///         match (previous, next) {
///             (Some((low, high)), Some((next_low, next_high))) => Some((
///                 if next_low < low { 0 } else { *low },
///                 if next_high > high { u32::MAX } else { *high },
///             )),
///             _ => *next,
///         }
///     }
/// }
///
/// let mut graph = Graph::<(), ()>::new();
/// let [init, head, step] = [(); 3].map(|()| graph.add_node(()));
/// graph.add_edge(init, head, ());
/// graph.add_edge(head, step, ());
/// graph.add_edge(step, head, ());
/// let counter = Counter { init, step };
///
/// let bounded = SolveConfig::with_step_limit(1_000);
/// assert!(solve_node_problem_with_config(&graph, &counter, bounded).is_err());
///
/// let widened = bounded.with_order(IterationOrder::WeakTopological);
/// let facts = solve_node_problem_with_config(&graph, &counter, widened).unwrap();
/// assert_eq!(*facts.fact_in(head), Some((0, u32::MAX)));
/// ```
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

    /// Widens at a loop head, as [`Problem::widen`](super::fixpoint::Problem::widen).
    ///
    /// The default returns `next`, which is no widening.
    fn widen(&self, _previous: &Self::Fact, next: &Self::Fact) -> Self::Fact {
        next.clone()
    }

    /// Narrows at a loop head, as
    /// [`Problem::narrow`](super::fixpoint::Problem::narrow).
    ///
    /// The default returns `next`, a plain descending step.
    fn narrow(&self, _previous: &Self::Fact, next: &Self::Fact) -> Self::Fact {
        next.clone()
    }
}

/// A fallible node-level dataflow problem over a graph view `G`.
///
/// This is the error-preserving counterpart of [`NodeProblem`], for
/// verification and abstract interpretation where a boundary, merge, or
/// transfer can reject the input graph. The solver reports those consumer
/// errors separately from its own configured step limit. Termination follows
/// the rules of [`NodeProblem`].
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

    /// Combine two facts at the join point `node`.
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the facts are incompatible.
    fn meet(
        &self,
        graph: &G,
        node: G::NodeId,
        a: &Self::Fact,
        b: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error>;

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

    /// Widens at the loop head `node`, as
    /// [`Problem::widen`](super::fixpoint::Problem::widen).
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the facts are incompatible.
    fn widen(
        &self,
        _graph: &G,
        _node: G::NodeId,
        _previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(next.clone())
    }

    /// Narrows at the loop head `node`, as
    /// [`Problem::narrow`](super::fixpoint::Problem::narrow).
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the facts are incompatible.
    fn narrow(
        &self,
        _graph: &G,
        _node: G::NodeId,
        _previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(next.clone())
    }
}

/// Solve a [`NodeProblem`] to fixpoint with a worklist, without a step limit.
///
/// # Errors
///
/// The unbounded configuration cannot produce a solver-limit error. The
/// `Result` matches [`solve_node_problem_with_config`] so callers can switch
/// configurations without changing result handling.
pub fn solve_node_problem<G, P>(graph: &G, problem: &P) -> Result<Facts<P::Fact>, SolveError>
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
/// `Facts` shape.
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
) -> Result<Facts<P::Fact>, SolveError>
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
) -> Result<Facts<P::Fact>, SolveError>
where
    G: GraphView,
    P: NodeProblem<G>,
{
    let fallible = InfallibleNodeProblem(problem);
    collapse_infallible(try_solve_with_worklist(
        graph,
        &fallible,
        Worklist::live(graph),
        None,
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
) -> Result<Facts<P::Fact>, SolveError>
where
    G: GraphView,
    P: NodeProblem<G>,
{
    let fallible = InfallibleNodeProblem(problem);
    collapse_infallible(try_solve_with_worklist(
        graph,
        &fallible,
        Worklist::seeded(graph, seeds),
        None,
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
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>>
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
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>>
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
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
{
    try_solve_with_worklist(graph, problem, Worklist::live(graph), None, config)
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
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
{
    try_solve_with_worklist(graph, problem, Worklist::seeded(graph, seeds), None, config)
}

/// The worklist solver every entry point runs, differing only in which nodes
/// start queued and in `entry`, a node that receives the boundary fact in
/// front of its upstream flow even when it has upstream neighbors — the
/// root of a rooted view that something branches back to.
pub(super) fn try_solve_with_worklist<G, P>(
    graph: &G,
    problem: &P,
    worklist: Worklist,
    entry: Option<G::NodeId>,
    config: SolveConfig,
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
{
    by_axis!(
        problem.direction().traversal(),
        try_solve_along(graph, problem, worklist, entry, config)
    )
}

/// The solver core, generic over the axis along which facts flow: facts move
/// from a node to its neighbors along `downstream`, and a node meets those of
/// its neighbors along the opposite axis.
fn try_solve_along<G, P, D>(
    downstream: D,
    graph: &G,
    problem: &P,
    mut worklist: Worklist,
    entry: Option<G::NodeId>,
    config: SolveConfig,
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
    D: Adjacency,
{
    let node_count = graph.node_bound();
    let mut solver = Solver {
        graph,
        problem,
        downstream,
        entry,
        boundary: problem.boundary(graph).map_err(TrySolveError::Problem)?,
        input: vec![problem.bottom(graph); node_count],
        output: vec![problem.bottom(graph); node_count],
    };
    let direction = problem.direction();
    let order = schedule(
        graph,
        &mut worklist,
        entry,
        direction.traversal(),
        config.order(),
    );
    let narrowing = order.as_ref().filter(|_| config.narrowing_passes() > 0);
    let mut leaving = narrowing.map(Leaving::new);
    let mut visited = narrowing.map(|_| DenseBits::new(node_count));

    let mut steps = 0;
    loop {
        let popped = worklist.pop_first();
        if let (Some(order), Some(visited)) = (narrowing, &visited)
            && let Some(component) = leaving.as_mut().and_then(|leaving| leaving.leave(popped))
        {
            if let Some(node_index) = popped {
                worklist.insert(node_index);
            }
            let sweep = Sweep {
                order,
                component,
                visited,
            };
            solver.narrow(&sweep, &mut worklist, config, &mut steps)?;
            continue;
        }
        let Some(node_index) = popped else {
            break;
        };
        config.count_step(&mut steps, node_index)?;
        let node = G::NodeId::from_index(node_index);
        let widen = order.as_ref().is_some_and(|order| order.is_head(node));
        let changed = solver
            .visit(node, widen.then_some(Bound::Widen))
            .map_err(TrySolveError::Problem)?;
        if let Some(visited) = &mut visited {
            visited.insert(node_index);
        }
        if changed.output {
            for next in downstream.neighbors(graph, node) {
                worklist.insert(next.index());
            }
        }
    }

    let (input, output) = direction.program_order(solver.input, solver.output);
    Ok(Facts::new(input, output, steps))
}

/// The facts of one solve so far, in flow order: `input` is what flows into
/// a node and `output` what its transfer produced.
struct Solver<'a, G: GraphView, P: TryNodeProblem<G>, D> {
    graph: &'a G,
    problem: &'a P,
    downstream: D,
    entry: Option<G::NodeId>,
    boundary: P::Fact,
    input: Vec<P::Fact>,
    output: Vec<P::Fact>,
}

impl<G: GraphView, P: TryNodeProblem<G>, D: Adjacency> Solver<'_, G, P, D> {
    /// Recomputes `node` from its upstream neighbors' outputs. With a
    /// `bound`, a fact flowing in that differs from the one the node holds is
    /// widened or narrowed against it first.
    fn visit(&mut self, node: G::NodeId, bound: Option<Bound>) -> Result<Changed, P::Error> {
        let (graph, problem) = (self.graph, self.problem);
        let index = node.index();
        let mut met = (self.entry == Some(node)).then(|| self.boundary.clone());
        for from in self.downstream.against().neighbors(graph, node) {
            met = Some(match met {
                None => self.output[from.index()].clone(),
                Some(current) => problem.meet(graph, node, &current, &self.output[from.index()])?,
            });
        }
        let mut met = met.unwrap_or_else(|| self.boundary.clone());
        let mut bounded = false;
        if let Some(bound) = bound {
            let held = &self.input[index];
            if met != *held {
                met = match bound {
                    Bound::Widen => problem.widen(graph, node, held, &met)?,
                    Bound::Narrow => problem.narrow(graph, node, held, &met)?,
                };
                bounded = met != *held;
            }
        }
        let transferred = problem.transfer(graph, node, &met)?;
        self.input[index] = met;
        let output = transferred != self.output[index];
        if output {
            self.output[index] = transferred;
        }
        Ok(Changed { bounded, output })
    }

    /// Runs the descending passes over a component the solve has left: each
    /// node of it that the solve visited, once per pass in order, narrowing
    /// at heads, until a pass changes nothing or the configured passes are
    /// spent. A changed output re-queues only the nodes past the component,
    /// so nothing in it widens again.
    fn narrow(
        &mut self,
        sweep: &Sweep<'_, G::NodeId>,
        worklist: &mut Worklist,
        config: SolveConfig,
        steps: &mut usize,
    ) -> Result<(), TrySolveError<P::Error>> {
        for _ in 0..config.narrowing_passes() {
            let mut changed = false;
            for node in sweep.nodes() {
                config.count_step(steps, node.index())?;
                let bound = sweep.order.is_head(node).then_some(Bound::Narrow);
                let visit = self.visit(node, bound).map_err(TrySolveError::Problem)?;
                changed |= visit.bounded || visit.output;
                if visit.output {
                    for next in self.downstream.neighbors(self.graph, node) {
                        worklist.insert_from(next.index(), sweep.component.end);
                    }
                }
            }
            if !changed {
                break;
            }
        }
        Ok(())
    }
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

    fn meet(
        &self,
        _graph: &G,
        _node: G::NodeId,
        a: &Self::Fact,
        b: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
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

    fn widen(
        &self,
        _graph: &G,
        _node: G::NodeId,
        previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(NodeProblem::widen(self.0, previous, next))
    }

    fn narrow(
        &self,
        _graph: &G,
        _node: G::NodeId,
        previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(NodeProblem::narrow(self.0, previous, next))
    }
}

#[cfg(test)]
mod tests;
