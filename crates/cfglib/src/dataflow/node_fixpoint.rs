//! Node-level fixpoint dataflow over any [`GraphView`].
//!
//! One fact per node, meet over the in-edges (out-edges backward), transfer
//! per node. It serves the per-block problems of a control-flow graph —
//! liveness, reaching definitions, rooted forward analyses through
//! [`Seeds::Reachable`](super::solve::Seeds::Reachable) — and analyses over graphs that are not CFGs: taint
//! or reachability-with-facts over a value-flow graph, closure over an import
//! or include graph, anywhere the graph is the program representation.
//!
//! [`solve_node_problem`] is the one entry point. Its [`Solve`] descriptor
//! names the [`Seeds`](super::solve::Seeds) the solve starts from and its
//! [`SolveConfig`]; the
//! [`stream`](super::stream) and [`resume`](super::resume) forms take the
//! same descriptor. A problem implements [`NodeProblem`], or
//! [`TryNodeProblem`] when a boundary, merge or transfer can fail; every
//! [`NodeProblem`] is a [`TryNodeProblem`] whose error is
//! [`Infallible`].

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::convert::Infallible;

use super::schedule::{Bound, Changed, Mode, Visit, drive};
use super::solve::{Facts, Solve, SolveConfig, TrySolveError};
use super::worklist::{Queue, Worklist};
use crate::graph::traverse::{Adjacency, by_axis};
use crate::graph::view::Direction;
use crate::graph::view::{DenseId, GraphView};
use crate::graph::weak_topological_order::WeakTopologicalOrder;

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
/// [narrowing passes](SolveConfig::narrowing_passes) can then win back
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
///     Direction, Graph, IterationOrder, NodeId, NodeProblem, Solve, SolveConfig, meet_options,
///     solve_node_problem,
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
/// let bounded = SolveConfig { max_steps: Some(1_000), ..SolveConfig::default() };
/// let solve = |config| Solve { config, ..Solve::default() };
/// assert!(solve_node_problem(&graph, &counter, solve(bounded)).is_err());
///
/// let widened = SolveConfig { order: IterationOrder::WeakTopological, ..bounded };
/// let facts = solve_node_problem(&graph, &counter, solve(widened)).unwrap();
/// assert_eq!(*facts.fact_in(head), Some((0, u32::MAX)));
/// ```
pub trait NodeProblem<G: GraphView> {
    /// The per-node dataflow fact.
    type Fact: Clone + PartialEq;

    /// Whether facts flow along edges ([`Direction::Forward`]) or against
    /// them ([`Direction::Backward`]).
    fn direction(&self) -> Direction;

    /// The initial fact for every node.
    ///
    /// The graph is given so a fact whose size depends on it, such as a
    /// [`DenseBits`](super::bits::DenseBits) over its nodes, can be built.
    fn bottom(&self, graph: &G) -> Self::Fact;

    /// The fact entering boundary nodes: those with no in-edges forward or
    /// no out-edges backward, and the root of
    /// [`Seeds::Reachable`](super::solve::Seeds::Reachable), merged
    /// with whatever flows back into it.
    fn boundary(&self, graph: &G) -> Self::Fact;

    /// Combine two facts at a join point.
    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact;

    /// The fact leaving `node`, given the met fact entering it.
    fn transfer(&self, graph: &G, node: G::NodeId, input: &Self::Fact) -> Self::Fact;

    /// Widens at a head of the weak topological order: combines the fact
    /// that entered the head before with the one that flows in now.
    ///
    /// The solver calls it only in
    /// [`IterationOrder::WeakTopological`](crate::IterationOrder::WeakTopological),
    /// only at heads, and only when the fact flowing in differs from the one
    /// the head holds: the result replaces the head's entering fact and is
    /// what the head transfers. It must be at least both arguments, and
    /// every chain `x0, x1 = x0 ∇ y1, x2 = x1 ∇ y2, ...` must become
    /// stationary, which is what bounds the solve. Heads cut every cycle, so
    /// widening there suffices.
    ///
    /// The default returns `next`, which is no widening.
    fn widen(&self, _previous: &Self::Fact, next: &Self::Fact) -> Self::Fact {
        next.clone()
    }

    /// Narrows at a head during a descending pass: refines the fact that
    /// entered the head with the one that flows in now.
    ///
    /// The solver calls it at heads during the passes of
    /// [`SolveConfig::narrowing_passes`], once the widened ascent of the
    /// head's outermost loop has converged, and only when the fact flowing
    /// in differs from the one the head holds. The result must lie between
    /// `next` and `previous`, so the facts stay sound while they shrink back
    /// from what widening overshot.
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
/// the rules of [`NodeProblem`]. Every [`NodeProblem`] implements this trait
/// with `Error = Infallible`, so the solver takes this form.
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

    /// The fact entering boundary nodes, as [`NodeProblem::boundary`].
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

    /// Widens at the head `node`, as [`NodeProblem::widen`].
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

    /// Narrows at the head `node`, as [`NodeProblem::narrow`].
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

impl<G: GraphView, P: NodeProblem<G>> TryNodeProblem<G> for P {
    type Fact = P::Fact;
    type Error = Infallible;

    fn direction(&self) -> Direction {
        NodeProblem::direction(self)
    }

    fn bottom(&self, graph: &G) -> Self::Fact {
        NodeProblem::bottom(self, graph)
    }

    fn boundary(&self, graph: &G) -> Result<Self::Fact, Self::Error> {
        Ok(NodeProblem::boundary(self, graph))
    }

    fn meet(
        &self,
        _graph: &G,
        _node: G::NodeId,
        a: &Self::Fact,
        b: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(NodeProblem::meet(self, a, b))
    }

    fn transfer(
        &self,
        graph: &G,
        node: G::NodeId,
        input: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(NodeProblem::transfer(self, graph, node, input))
    }

    fn widen(
        &self,
        _graph: &G,
        _node: G::NodeId,
        previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(NodeProblem::widen(self, previous, next))
    }

    fn narrow(
        &self,
        _graph: &G,
        _node: G::NodeId,
        previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(NodeProblem::narrow(self, previous, next))
    }
}

/// Solve a node problem to a fixpoint with a worklist, from the seeds and
/// within the limits of `solve`.
///
/// Every node starts at `bottom`.
/// [`Seeds::Live`](super::solve::Seeds::Live) queues every live node,
/// [`Seeds::Reachable`](super::solve::Seeds::Reachable) the nodes the root
/// reaches along the analysis direction, and
/// [`Seeds::Of`](super::solve::Seeds::Of) only the named nodes; after that a node runs
/// again when an upstream neighbor's output changed. An infallible problem
/// reports `TrySolveError<Infallible>`, which `?` converts into a
/// [`SolveError`](super::solve::SolveError).
///
/// # Examples
///
/// ```
/// use cfglib::{Direction, Graph, NodeId, NodeProblem, Seeds, Solve, solve_node_problem};
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
///     fn transfer(&self, _: &Graph<&'static str, ()>, node: NodeId, input: &bool) -> bool {
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
/// let seeded = Solve { seeds: Seeds::Of(&[edited]), ..Solve::default() };
/// let facts = solve_node_problem(&graph, &Taint(vec![edited]), seeded).unwrap();
/// assert!(*facts.fact_out(downstream));
/// assert!(!*facts.fact_out(untouched));
/// assert_eq!(facts.stats().step_count(), 2);
/// ```
///
/// # Panics
///
/// Panics when a seed or the root is not a live node of `graph`.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error or
/// [`TrySolveError::Solver`] when work remains at the configured limit.
pub fn solve_node_problem<G, P>(
    graph: &G,
    problem: &P,
    solve: Solve<'_, G::NodeId>,
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
{
    let direction = problem.direction();
    let (worklist, entry) = solve.seeds.start(graph, direction);
    by_axis!(
        direction,
        solve_from_bottom(graph, problem, worklist, entry, solve.config)
    )
}

fn solve_from_bottom<G, P, D>(
    downstream: D,
    graph: &G,
    problem: &P,
    worklist: Worklist,
    entry: Option<G::NodeId>,
    config: SolveConfig,
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
    D: Adjacency,
{
    let node_count = graph.node_bound();
    let every = || Dense {
        input: vec![problem.bottom(graph); node_count],
        output: vec![problem.bottom(graph); node_count],
    };
    solve_dense(downstream, graph, problem, worklist, entry, config, every)
}

/// The solver core over every fact, from the facts in flow order that
/// `facts` builds, reported in program order.
pub(super) fn solve_dense<G, P, D>(
    downstream: D,
    graph: &G,
    problem: &P,
    worklist: Worklist,
    entry: Option<G::NodeId>,
    config: SolveConfig,
    facts: impl FnOnce() -> Dense<P::Fact>,
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
    D: Adjacency,
{
    let store = |_: Option<&WeakTopologicalOrder<G::NodeId>>| facts();
    let (facts, steps) = solve_along(downstream, graph, problem, worklist, entry, config, store)?;
    let (input, output) = problem.direction().program_order(facts.input, facts.output);
    Ok(Facts::new(input, output, steps, 0))
}

/// Where a node solve keeps the facts of its nodes in flow order: what flows
/// into a node and what its transfer produced.
pub(super) trait NodeStore<N, F> {
    /// Whether the solve calls [`finish`](Self::finish) as it leaves each
    /// outermost element of its weak topological order.
    const FINISHES: bool;

    /// The fact flowing into the node at `index`.
    fn input(&self, index: usize) -> &F;

    /// The fact that the node at `index` transferred.
    fn output(&self, index: usize) -> &F;

    fn set_input(&mut self, index: usize, fact: F);

    fn set_output(&mut self, index: usize, fact: F);

    /// Every node before position `end` of `order` holds its final facts,
    /// and no visit reads the facts of what is final unless a node at or
    /// after `end` reads it.
    fn finish(&mut self, order: &WeakTopologicalOrder<N>, end: usize);
}

/// Every node's facts in flow order, for a solve that returns them all.
pub(super) struct Dense<F> {
    pub(super) input: Vec<F>,
    pub(super) output: Vec<F>,
}

impl<N, F> NodeStore<N, F> for Dense<F> {
    const FINISHES: bool = false;

    fn input(&self, index: usize) -> &F {
        &self.input[index]
    }

    fn output(&self, index: usize) -> &F {
        &self.output[index]
    }

    fn set_input(&mut self, index: usize, fact: F) {
        self.input[index] = fact;
    }

    fn set_output(&mut self, index: usize, fact: F) {
        self.output[index] = fact;
    }

    fn finish(&mut self, _order: &WeakTopologicalOrder<N>, _end: usize) {}
}

/// The solver core, generic over the axis along which facts flow and over
/// where the facts live: facts move from a node to its neighbors along
/// `downstream`, and a node meets those of its neighbors along the opposite
/// axis. `store` builds the fact store once the iteration order is known;
/// the store comes back with the number of steps taken.
pub(super) fn solve_along<G, P, D, S>(
    downstream: D,
    graph: &G,
    problem: &P,
    worklist: Worklist,
    entry: Option<G::NodeId>,
    config: SolveConfig,
    store: impl FnOnce(Option<&WeakTopologicalOrder<G::NodeId>>) -> S,
) -> Result<(S, usize), TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
    D: Adjacency,
    S: NodeStore<G::NodeId, P::Fact>,
{
    let boundary = problem.boundary(graph).map_err(TrySolveError::Problem)?;
    let start = |order: Option<&WeakTopologicalOrder<G::NodeId>>| Solver {
        graph,
        problem,
        downstream,
        entry,
        boundary,
        store: store(order),
    };
    let direction = problem.direction();
    let (solver, steps) = drive(graph, worklist, entry, direction, config, start)?;
    Ok((solver.store, steps))
}

/// One solve in progress.
struct Solver<'a, G: GraphView, P: TryNodeProblem<G>, D, S> {
    graph: &'a G,
    problem: &'a P,
    downstream: D,
    entry: Option<G::NodeId>,
    boundary: P::Fact,
    store: S,
}

impl<G, P, D, S> Visit<G::NodeId> for Solver<'_, G, P, D, S>
where
    G: GraphView,
    P: TryNodeProblem<G>,
    D: Adjacency,
    S: NodeStore<G::NodeId, P::Fact>,
{
    type Error = P::Error;

    const FINISHES: bool = S::FINISHES;

    fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    /// Recomputes `node` from its upstream neighbors' outputs, and queues its
    /// downstream neighbors when its output changed.
    fn visit<M: Mode, Q: Queue>(
        &mut self,
        _mode: M,
        queue: &mut Q,
        floor: usize,
        node: G::NodeId,
        bound: Option<Bound>,
    ) -> Result<Changed, P::Error> {
        let (graph, problem) = (self.graph, self.problem);
        let index = node.index();
        let mut met = (self.entry == Some(node)).then(|| self.boundary.clone());
        for from in self.downstream.against().neighbors(graph, node) {
            let output = self.store.output(from.index());
            met = Some(match met {
                None => output.clone(),
                Some(current) => problem.meet(graph, node, &current, output)?,
            });
        }
        let mut met = met.unwrap_or_else(|| self.boundary.clone());
        let mut bounded = false;
        if M::ORDERED
            && let Some(bound) = bound
        {
            let held = self.store.input(index);
            if met != *held {
                met = match bound {
                    Bound::Widen => problem.widen(graph, node, held, &met)?,
                    Bound::Narrow => problem.narrow(graph, node, held, &met)?,
                };
                bounded = met != *held;
            }
        }
        let transferred = problem.transfer(graph, node, &met)?;
        self.store.set_input(index, met);
        let output = transferred != *self.store.output(index);
        if output {
            self.store.set_output(index, transferred);
            for next in self.downstream.neighbors(graph, node) {
                if M::ORDERED {
                    queue.push_from(next.index(), floor);
                } else {
                    queue.push(next.index());
                }
            }
        }
        Ok(Changed { bounded, output })
    }

    fn finish(&mut self, order: &WeakTopologicalOrder<G::NodeId>, end: usize) {
        self.store.finish(order, end);
    }
}

#[cfg(test)]
mod tests;
