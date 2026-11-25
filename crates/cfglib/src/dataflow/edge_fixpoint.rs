//! Edge-sensitive fixpoint dataflow over borrowed graph views.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::convert::Infallible;

mod store;

use store::Dense;
pub(super) use store::EdgeStore;

use crate::dataflow::bits::DenseBits;
use crate::dataflow::fixpoint::{
    Direction, Facts, SolveConfig, SolveError, TrySolveError, collapse_infallible,
};
use crate::dataflow::schedule::{
    Bound, Changed, Leaving, Mode, Ordered, Plain, Schedule, Sweep, schedule,
};
use crate::dataflow::worklist::{Queue, Ranked, Worklist};
use crate::graph::edge_traverse::EdgeAdjacency;
use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::traverse::by_axis;
use crate::graph::view::DenseId;
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
/// [narrowing passes](SolveConfig::with_narrowing_passes) can then win back
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
    /// [`Problem::widen`](crate::Problem::widen).
    ///
    /// The default returns `next`, which is no widening.
    fn widen(&self, _previous: &Self::Fact, next: &Self::Fact) -> Self::Fact {
        next.clone()
    }

    /// Narrows the fact flowing into a loop head, as
    /// [`Problem::narrow`](crate::Problem::narrow).
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
/// the rules of [`EdgeProblem`].
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
    /// [`Problem::widen`](crate::Problem::widen).
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
    /// [`Problem::narrow`](crate::Problem::narrow).
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
    fn boundary(&self, graph: &G, node: G::NodeId) -> Result<Option<Self::Fact>, Self::Error>;

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
        Ok(ReachableEdgeProblem::boundary(&self.0, graph, node)?.map(Some))
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

    /// Number of worklist entries processed.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.nodes.steps()
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
        Worklist::live(graph),
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
        Worklist::seeded(graph, seeds),
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
    try_solve_with_worklist(graph, problem, Worklist::live(graph), config)
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
    try_solve_with_worklist(graph, problem, Worklist::seeded(graph, seeds), config)
}

fn try_solve_with_worklist<G, P>(
    graph: &G,
    problem: &P,
    worklist: Worklist,
    config: SolveConfig,
) -> Result<EdgeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
{
    let every = |_: Option<&WeakTopologicalOrder<G::NodeId>>| {
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
    let (facts, steps) = solve_edges(graph, problem, worklist, config, every)?;
    Ok(EdgeFacts {
        nodes: Facts::new(facts.input, facts.output, steps),
        edge: facts.edge,
    })
}

/// The edge solver core over the facts that `store` builds once the
/// iteration order is known; the store comes back with the number of steps
/// taken.
pub(super) fn solve_edges<G, P, S>(
    graph: &G,
    problem: &P,
    worklist: Worklist,
    config: SolveConfig,
    store: impl FnOnce(Option<&WeakTopologicalOrder<G::NodeId>>) -> S,
) -> Result<(S, usize), TrySolveError<P::Error>>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
    S: EdgeStore<G::NodeId, P::Fact>,
{
    let traversal = problem.direction().traversal();
    let solver = |order| Solver {
        graph,
        problem,
        store: store(order),
    };
    match schedule(graph, worklist, None, traversal, config.order()) {
        Schedule::Index(mut worklist) => solver(None).solve_plain(&mut worklist, config),
        Schedule::Ranked(mut ranked) => solver(None).solve_plain(&mut ranked, config),
        Schedule::Weak(mut ranked, order) => {
            solver(Some(&order)).solve_ordered(&mut ranked, &order, config)
        }
    }
}

/// One solve in progress.
struct Solver<'a, G, P, S> {
    graph: &'a G,
    problem: &'a P,
    store: S,
}

/// Recomputes `node` from the edges flowing into it against `downstream`, and
/// re-queues the neighbor of every edge leaving it along `downstream` whose
/// fact changed. In [`Ordered`] mode, a `bound` widens or narrows a fact
/// flowing in that differs from the one the node holds, and a neighbor
/// before position `floor` of the iteration order is not re-queued.
fn visit_along<G, P, S, D, M, Q>(
    downstream: D,
    solver: &mut Solver<'_, G, P, S>,
    _mode: M,
    queue: &mut Q,
    floor: usize,
    node: G::NodeId,
    bound: Option<Bound>,
) -> Result<Changed, P::Error>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
    S: EdgeStore<G::NodeId, P::Fact>,
    D: EdgeAdjacency,
    D::Against: EdgeAdjacency,
    M: Mode,
    Q: Queue,
{
    let (graph, problem) = (solver.graph, solver.problem);
    let direction = problem.direction();
    let index = node.index();
    let mut flowing_in = solver.meet_edges(node, downstream.against())?;
    let mut bounded = false;
    if M::ORDERED
        && let Some(bound) = bound
    {
        let (held, _) =
            direction.program_order(solver.store.input(index), solver.store.output(index));
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
    solver.store.set_node(index, input, output);

    let mut edges_changed = false;
    for edge_id in downstream.edges(graph, node) {
        let edge = graph.edge(edge_id);
        let new_fact = problem.transfer_edge(
            graph,
            edge,
            solver.store.input(index),
            solver.store.output(index),
        )?;
        if solver.store.update_edge(edge_id.index(), new_fact) {
            let next = downstream.next(edge).index();
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

impl<G, P, S> Solver<'_, G, P, S>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
    S: EdgeStore<G::NodeId, P::Fact>,
{
    /// The node's boundary fact met with the fact on every edge along
    /// `upstream`, in adjacency order; bottom when there is neither.
    fn meet_edges<U: EdgeAdjacency>(
        &self,
        node: G::NodeId,
        upstream: U,
    ) -> Result<P::Fact, P::Error> {
        let mut merged = self.problem.boundary(self.graph, node)?;
        for edge in upstream.edges(self.graph, node) {
            let fact = self.store.edge(edge.index());
            merged = Some(match merged {
                Some(current) => self.problem.meet(self.graph, node, &current, fact)?,
                None => fact.clone(),
            });
        }
        Ok(merged.unwrap_or_else(|| self.store.bottom().clone()))
    }

    /// The worklist loop of a solve that never bounds a fact: the index or
    /// reverse-postorder order. A solve whose store finishes elements runs
    /// in a weak topological order, so none comes here.
    fn solve_plain(
        mut self,
        queue: &mut impl Queue,
        config: SolveConfig,
    ) -> Result<(S, usize), TrySolveError<P::Error>> {
        let traversal = self.problem.direction().traversal();
        let mut steps = 0;
        while let Some(node_index) = queue.pop() {
            config.count_step(&mut steps, node_index)?;
            let node = G::NodeId::from_index(node_index);
            by_axis!(
                traversal,
                visit_along(&mut self, Plain, queue, 0, node, None)
            )
            .map_err(TrySolveError::Problem)?;
        }
        Ok((self.store, steps))
    }

    /// The worklist loop of a solve in a weak topological order: it widens
    /// at heads, narrows each outermost component once it has left it, and
    /// finishes each outermost element in the store as it leaves it.
    ///
    /// Out of line, so that the default solve's entry point holds only the
    /// plain loop.
    #[inline(never)]
    fn solve_ordered(
        mut self,
        queue: &mut Ranked,
        order: &WeakTopologicalOrder<G::NodeId>,
        config: SolveConfig,
    ) -> Result<(S, usize), TrySolveError<P::Error>> {
        let traversal = self.problem.direction().traversal();
        let narrowing = config.narrowing_passes() > 0;
        let mut leaving = (S::FINISHES || narrowing).then(|| Leaving::new(order, S::FINISHES));
        let mut visited = narrowing.then(|| DenseBits::new(self.graph.node_bound()));
        let mut steps = 0;
        loop {
            let popped = queue.pop();
            if let Some(element) = leaving.as_mut().and_then(|leaving| leaving.leave(popped)) {
                if let Some(node_index) = popped {
                    queue.push(node_index);
                }
                if let Some(visited) = &visited
                    && order.heads_at(element.start)
                {
                    let sweep = Sweep {
                        order,
                        component: element.clone(),
                        visited,
                    };
                    self.narrow(&sweep, queue, config, &mut steps)?;
                }
                self.store.finish(order, element.end);
                continue;
            }
            let Some(node_index) = popped else {
                break;
            };
            config.count_step(&mut steps, node_index)?;
            let node = G::NodeId::from_index(node_index);
            let bound = order.is_head(node).then_some(Bound::Widen);
            by_axis!(
                traversal,
                visit_along(&mut self, Ordered, queue, 0, node, bound)
            )
            .map_err(TrySolveError::Problem)?;
            if let Some(visited) = &mut visited {
                visited.insert(node_index);
            }
        }
        if S::FINISHES {
            self.store.finish(order, order.order().len());
        }
        Ok((self.store, steps))
    }

    /// Runs the descending passes over a component the solve has left: each
    /// node of it that the solve visited, once per pass in order, narrowing
    /// at heads, until a pass changes nothing or the configured passes are
    /// spent. A changed edge re-queues only a node past the component, so
    /// nothing in it widens again.
    fn narrow(
        &mut self,
        sweep: &Sweep<'_, G::NodeId>,
        queue: &mut Ranked,
        config: SolveConfig,
        steps: &mut usize,
    ) -> Result<(), TrySolveError<P::Error>> {
        let traversal = self.problem.direction().traversal();
        let floor = sweep.component.end;
        for _ in 0..config.narrowing_passes() {
            let mut changed = false;
            for node in sweep.nodes() {
                config.count_step(steps, node.index())?;
                let bound = sweep.order.is_head(node).then_some(Bound::Narrow);
                let visit = by_axis!(
                    traversal,
                    visit_along(self, Ordered, queue, floor, node, bound)
                )
                .map_err(TrySolveError::Problem)?;
                changed |= visit.bounded || visit.output;
            }
            if !changed {
                break;
            }
        }
        Ok(())
    }
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

    fn widen(
        &self,
        _graph: &G,
        _node: G::NodeId,
        previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(EdgeProblem::widen(self.0, previous, next))
    }

    fn narrow(
        &self,
        _graph: &G,
        _node: G::NodeId,
        previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(EdgeProblem::narrow(self.0, previous, next))
    }
}

#[cfg(test)]
mod tests;
