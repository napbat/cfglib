//! Edge-sensitive fixpoint dataflow over borrowed graph views.
//!
//! [`solve_edge_problem`] is the one entry point. Its [`Solve`] descriptor
//! names the seeds the solve
//! starts from and its [`SolveConfig`]; the [`stream`](super::stream) and
//! [`resume`](super::resume) forms take the same descriptor. A problem
//! implements [`EdgeProblem`], or [`TryEdgeProblem`] when a boundary, merge
//! or transfer can fail; every [`EdgeProblem`] is a [`TryEdgeProblem`] whose
//! error is [`Infallible`].
//!
//! A verification-style analysis whose facts exist only where execution
//! reaches (stack maps, register frames) uses `Option` facts: `bottom` and
//! the default boundary are `None`, [`try_meet_options`](crate::try_meet_options)
//! merges only reached facts, the transfers pass `None` through, and an
//! exceptional edge in a forward analysis returns the node's input fact from
//! [`TryEdgeProblem::transfer_edge`] because it leaves before the node's
//! effect commits.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::convert::Infallible;

mod store;

pub(super) use store::{Dense, EdgeStore};

use crate::dataflow::schedule::{Bound, Changed, Mode, Visit, drive};
use crate::dataflow::solve::{Facts, Solve, SolveConfig, SolveStats, TrySolveError};
use crate::dataflow::worklist::{Queue, Worklist};
use crate::graph::edge_traverse::EdgeAdjacency;
use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::traverse::by_axis;
use crate::graph::view::DenseId;
use crate::graph::view::Direction;
use crate::graph::weak_topological_order::WeakTopologicalOrder;

/// A dataflow problem whose transfer can distinguish individual edges.
///
/// In forward analysis, `node_input` is the physical pre-state and
/// `node_output` is the physical post-state. An exceptional edge can therefore
/// transfer from `node_input`, while a normal edge uses `node_output`. The same
/// two physical facts are supplied during backward analysis; the default edge
/// transfer uses the fact flowing in the selected analysis direction.
///
/// # Termination
///
/// In [`IterationOrder::Index`](crate::IterationOrder::Index) and
/// [`IterationOrder::ReversePostorder`](crate::IterationOrder::ReversePostorder),
/// termination requires `meet`, `transfer_node` and `transfer_edge` to be
/// monotone over a finite-height fact lattice. In
/// [`IterationOrder::WeakTopological`](crate::IterationOrder::WeakTopological)
/// the solver widens the fact flowing into the head of every loop with
/// [`widen`](Self::widen), so a monotone problem over an infinite-height
/// lattice terminates too, and
/// [narrowing passes](SolveConfig::narrowing_passes) can then win back
/// precision with [`narrow`](Self::narrow). The defaults of both leave the
/// facts unchanged.
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

    /// Widens the fact flowing into a loop head, as
    /// [`NodeProblem::widen`](crate::NodeProblem::widen).
    ///
    /// The default returns `next`, which is no widening.
    fn widen(&self, _previous: &Self::Fact, next: &Self::Fact) -> Self::Fact {
        next.clone()
    }

    /// Narrows the fact flowing into a loop head, as
    /// [`NodeProblem::narrow`](crate::NodeProblem::narrow).
    ///
    /// The default returns `next`, a plain descending step.
    fn narrow(&self, _previous: &Self::Fact, next: &Self::Fact) -> Self::Fact {
        next.clone()
    }
}

/// A fallible dataflow problem whose transfers can distinguish individual
/// edges.
///
/// This is the error-preserving counterpart of [`EdgeProblem`]. It is intended
/// for verification and abstract interpretation where a transfer or lattice
/// merge can reject the input program. The solver reports those consumer
/// errors separately from its own configured step limit. Termination follows
/// the rules of [`EdgeProblem`]. Every [`EdgeProblem`] implements this trait
/// with `Error = Infallible`, so the solver takes this form.
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

    /// Widens the fact flowing into the loop head `node`, as
    /// [`NodeProblem::widen`](crate::NodeProblem::widen).
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

    /// Narrows the fact flowing into the loop head `node`, as
    /// [`NodeProblem::narrow`](crate::NodeProblem::narrow).
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

impl<G: EdgeView, P: EdgeProblem<G>> TryEdgeProblem<G> for P {
    type Fact = P::Fact;
    type Error = Infallible;

    fn direction(&self) -> Direction {
        EdgeProblem::direction(self)
    }

    fn bottom(&self, graph: &G) -> Self::Fact {
        EdgeProblem::bottom(self, graph)
    }

    fn boundary(&self, graph: &G, node: G::NodeId) -> Result<Option<Self::Fact>, Self::Error> {
        Ok(EdgeProblem::boundary(self, graph, node))
    }

    fn meet(
        &self,
        _graph: &G,
        _node: G::NodeId,
        left: &Self::Fact,
        right: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(EdgeProblem::meet(self, left, right))
    }

    fn transfer_node(
        &self,
        graph: &G,
        node: G::NodeId,
        flow_fact: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(EdgeProblem::transfer_node(self, graph, node, flow_fact))
    }

    fn transfer_edge(
        &self,
        graph: &G,
        edge: EdgeRef<'_, G::NodeId, G::EdgeId, G::EdgeData>,
        node_input: &Self::Fact,
        node_output: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(EdgeProblem::transfer_edge(
            self,
            graph,
            edge,
            node_input,
            node_output,
        ))
    }

    fn widen(
        &self,
        _graph: &G,
        _node: G::NodeId,
        previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(EdgeProblem::widen(self, previous, next))
    }

    fn narrow(
        &self,
        _graph: &G,
        _node: G::NodeId,
        previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(EdgeProblem::narrow(self, previous, next))
    }
}

/// Node and edge facts at a completed fixpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeFacts<F> {
    nodes: Facts<F>,
    edge: Vec<Option<F>>,
}

impl<F> EdgeFacts<F> {
    /// Physical input fact for `node`: the fact at its start in program
    /// order, whichever way the analysis flows.
    #[must_use]
    pub fn fact_in<N: DenseId>(&self, node: N) -> &F {
        self.nodes.fact_in(node)
    }

    /// Physical output fact for `node`: the fact at its end in program
    /// order.
    #[must_use]
    pub fn fact_out<N: DenseId>(&self, node: N) -> &F {
        self.nodes.fact_out(node)
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

    /// What the solve did. It held every node and live edge fact
    /// throughout.
    #[must_use]
    pub const fn stats(&self) -> SolveStats {
        self.nodes.stats()
    }

    /// The physical input and output facts of every node and the fact on
    /// every edge slot, for a solve that resumes from them.
    pub(super) fn into_parts(self) -> (Vec<F>, Vec<F>, Vec<Option<F>>) {
        let (input, output) = self.nodes.into_parts();
        (input, output, self.edge)
    }
}

/// Solve an edge-sensitive problem to a fixpoint with a worklist, from the
/// seeds and within the limits of `solve`.
///
/// Every node and live edge starts at bottom. [`Seeds::Live`] queues every
/// live node, [`Seeds::Reachable`] the nodes the root reaches along the
/// analysis direction (the problem's `boundary` still decides which nodes
/// start with a fact), and [`Seeds::Of`] only the named nodes: reachable-only
/// verification and incremental analysis without manufacturing facts for
/// disconnected nodes. After that a node runs again when a fact on one of its
/// upstream edges changed. An infallible problem reports
/// `TrySolveError<Infallible>`, which `?` converts into a
/// [`SolveError`](super::solve::SolveError).
///
/// [`Seeds::Live`]: super::solve::Seeds::Live
/// [`Seeds::Reachable`]: super::solve::Seeds::Reachable
/// [`Seeds::Of`]: super::solve::Seeds::Of
///
/// # Panics
///
/// Panics when a seed or the root is not a live node of `graph`.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error or
/// [`TrySolveError::Solver`] when work remains at the configured limit.
pub fn solve_edge_problem<G, P>(
    graph: &G,
    problem: &P,
    solve: Solve<'_, G::NodeId>,
) -> Result<EdgeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
{
    let direction = problem.direction();
    let (worklist, entry) = solve.seeds.start(graph, direction);
    let every = || {
        let bottom = problem.bottom(graph);
        let mut edge = vec![None; graph.edge_bound()];
        for edge_id in graph.edge_ids() {
            edge[edge_id.index()] = Some(bottom.clone());
        }
        Dense {
            input: vec![bottom.clone(); graph.node_bound()],
            output: vec![bottom.clone(); graph.node_bound()],
            edge,
            bottom,
        }
    };
    by_axis!(
        direction,
        solve_dense(graph, problem, worklist, entry, solve.config, every)
    )
}

/// The edge solver over every fact, from the facts that `facts` builds.
pub(super) fn solve_dense<G, P, D>(
    downstream: D,
    graph: &G,
    problem: &P,
    worklist: Worklist,
    entry: Option<G::NodeId>,
    config: SolveConfig,
    facts: impl FnOnce() -> Dense<P::Fact>,
) -> Result<EdgeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
    D: EdgeAdjacency,
    D::Against: EdgeAdjacency,
{
    let store = |_: Option<&WeakTopologicalOrder<G::NodeId>>| facts();
    let (facts, steps) = solve_edges(downstream, graph, problem, worklist, entry, config, store)?;
    let held = graph.edge_ids().count();
    Ok(EdgeFacts {
        nodes: Facts::new(facts.input, facts.output, steps, held),
        edge: facts.edge,
    })
}

/// The edge solver core, generic over the axis along which facts flow, over
/// the facts that `store` builds once the iteration order is known; the
/// store comes back with the number of steps taken.
pub(super) fn solve_edges<G, P, D, S>(
    downstream: D,
    graph: &G,
    problem: &P,
    worklist: Worklist,
    entry: Option<G::NodeId>,
    config: SolveConfig,
    store: impl FnOnce(Option<&WeakTopologicalOrder<G::NodeId>>) -> S,
) -> Result<(S, usize), TrySolveError<P::Error>>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
    D: EdgeAdjacency,
    D::Against: EdgeAdjacency,
    S: EdgeStore<G::NodeId, P::Fact>,
{
    let start = |order: Option<&WeakTopologicalOrder<G::NodeId>>| Solver {
        graph,
        problem,
        downstream,
        store: store(order),
    };
    let direction = problem.direction();
    let (solver, steps) = drive(graph, worklist, entry, direction, config, start)?;
    Ok((solver.store, steps))
}

/// One solve in progress: facts move along the edges of `downstream`.
struct Solver<'a, G, P, D, S> {
    graph: &'a G,
    problem: &'a P,
    downstream: D,
    store: S,
}

impl<G, P, D, S> Solver<'_, G, P, D, S>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
    D: EdgeAdjacency,
    D::Against: EdgeAdjacency,
    S: EdgeStore<G::NodeId, P::Fact>,
{
    /// The node's boundary fact met with the fact on every upstream edge, in
    /// adjacency order; bottom when there is neither.
    fn meet_edges(&self, node: G::NodeId) -> Result<P::Fact, P::Error> {
        let mut merged = self.problem.boundary(self.graph, node)?;
        for edge in self.downstream.against().edges(self.graph, node) {
            let fact = self.store.edge(edge.index());
            merged = Some(match merged {
                Some(current) => self.problem.meet(self.graph, node, &current, fact)?,
                None => fact.clone(),
            });
        }
        Ok(merged.unwrap_or_else(|| self.store.bottom().clone()))
    }
}

impl<G, P, D, S> Visit<G::NodeId> for Solver<'_, G, P, D, S>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
    D: EdgeAdjacency,
    D::Against: EdgeAdjacency,
    S: EdgeStore<G::NodeId, P::Fact>,
{
    type Error = P::Error;

    const FINISHES: bool = S::FINISHES;

    fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    /// Recomputes `node` from the edges flowing into it, and queues the far
    /// end of every downstream edge whose fact changed.
    fn visit<M: Mode, Q: Queue>(
        &mut self,
        _mode: M,
        queue: &mut Q,
        floor: usize,
        node: G::NodeId,
        bound: Option<Bound>,
    ) -> Result<Changed, P::Error> {
        let (graph, problem) = (self.graph, self.problem);
        let direction = problem.direction();
        let index = node.index();
        let mut flowing_in = self.meet_edges(node)?;
        let mut bounded = false;
        if M::ORDERED
            && let Some(bound) = bound
        {
            let (held, _) =
                direction.program_order(self.store.input(index), self.store.output(index));
            if flowing_in != *held {
                flowing_in = match bound {
                    Bound::Widen => problem.widen(graph, node, held, &flowing_in)?,
                    Bound::Narrow => problem.narrow(graph, node, held, &flowing_in)?,
                };
                bounded = flowing_in != *held;
            }
        }
        let flowing_out = problem.transfer_node(graph, node, &flowing_in)?;
        let (input, output) = direction.program_order(flowing_in, flowing_out);
        self.store.set_node(index, input, output);

        let mut edges_changed = false;
        for edge_id in self.downstream.edges(graph, node) {
            let edge = graph.edge(edge_id);
            let new_fact = problem.transfer_edge(
                graph,
                edge,
                self.store.input(index),
                self.store.output(index),
            )?;
            if self.store.update_edge(edge_id.index(), new_fact) {
                let next = self.downstream.next(edge).index();
                if M::ORDERED {
                    queue.push_from(next, floor);
                } else {
                    queue.push(next);
                }
                edges_changed = true;
            }
        }
        Ok(Changed {
            bounded,
            output: edges_changed,
        })
    }

    fn finish(&mut self, order: &WeakTopologicalOrder<G::NodeId>, end: usize) {
        self.store.finish(order, end);
    }
}

#[cfg(test)]
mod tests;
