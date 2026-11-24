//! Generic fixpoint iteration engine for data flow analysis.
//!
//! Supports both **forward** and **backward** analyses via a worklist
//! algorithm that iterates until the solution stabilizes.
//!
//! The crate's three generic solvers — this one, the node-level
//! [`node_fixpoint`], and the edge-sensitive
//! [`edge_fixpoint`](super::edge_fixpoint) — carry the same facility matrix:
//! a full solve, a seeded solve (`_from`), a configured solve
//! (`_with_config`), and fallible `try_` counterparts, all sharing
//! [`SolveConfig`], [`SolveError`], and [`TrySolveError`]. This one runs
//! per-block facts on the node solver's core, and the node solver and this
//! one report their facts in the same [`Facts`] type.
//!
//! A [`SolveConfig`] bounds a solve deterministically and selects its
//! [`IterationOrder`]. In [`IterationOrder::WeakTopological`] every solver
//! widens at the heads of a
//! [`WeakTopologicalOrder`](crate::WeakTopologicalOrder) and can narrow
//! afterwards, through the `widen` and `narrow` methods that every problem
//! trait carries with defaults that change nothing.
//!
//! # One vocabulary for the three problem families
//!
//! Every problem names its starting fact `boundary`, its unreached or
//! initial fact `bottom`, and its direction [`Direction`]. Two shapes of
//! `boundary` exist, for one reason:
//!
//! - [`Problem`] and [`NodeProblem`](super::node_fixpoint::NodeProblem)
//!   return one fact, and the *solver* decides which nodes receive it: a
//!   node with no upstream neighbor along the analysis direction, and for
//!   [`Problem`] also the root of a forward solve. A block or node graph has
//!   a structural boundary, so naming the nodes would only restate it.
//! - [`EdgeProblem`](super::edge_fixpoint::EdgeProblem) returns an optional
//!   fact *per node*, and the *problem* decides which nodes get one, merged
//!   with whatever flows in. Exceptional-edge analyses seed handler entries,
//!   coroutine resumptions, and several roots, none of which is structural.
//!
//! A fallible problem's `meet` also receives the view and the merge point,
//! so an error can name where two facts disagreed; an infallible `meet` is
//! a pure lattice operation and receives only the two facts.

extern crate alloc;
use alloc::vec::Vec;
use core::convert::Infallible;

use super::node_fixpoint::{self, TryNodeProblem};
use super::worklist::Worklist;
use crate::graph::traverse::{TraversalDirection, depth_first_preorder};
use crate::graph::view::{DenseId, RootedView};

/// Direction of the data flow analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Forward: information flows from predecessors to successors.
    Forward,
    /// Backward: information flows from successors to predecessors.
    Backward,
}

impl Direction {
    /// The direction in which the solver follows edges to carry a fact from
    /// a node to the nodes that consume it.
    ///
    /// A forward analysis carries facts along outgoing edges, a backward
    /// analysis along incoming ones. Every solver derives its adjacency
    /// from this one conversion.
    #[must_use]
    pub const fn traversal(self) -> TraversalDirection {
        match self {
            Self::Forward => TraversalDirection::Outgoing,
            Self::Backward => TraversalDirection::Incoming,
        }
    }

    /// Orders what a node receives and what it transfers by program order.
    ///
    /// A forward analysis receives its fact at the node's start and transfers
    /// it to the node's end, so `entering` is the input and `leaving` the
    /// output. A backward analysis receives at the end and transfers to the
    /// start, so they swap. [`Facts`] reports program order whichever way
    /// the analysis flows.
    pub(crate) const fn program_order<T>(self, entering: T, leaving: T) -> (T, T) {
        match self {
            Self::Forward => (entering, leaving),
            Self::Backward => (leaving, entering),
        }
    }
}

/// The order in which a solve takes queued nodes off its worklist.
///
/// The least fixpoint of a monotone problem over a finite-height lattice does
/// not depend on the order, so every order returns the same facts for such a
/// problem; the order decides how many steps it takes to get there. Index
/// order is program order for a CFG whose blocks were created in order, and
/// far from it for a backward problem, for a graph that is not a CFG, or for
/// a CFG whose blocks were split, merged or renumbered.
///
/// [`ReversePostorder`](Self::ReversePostorder) and
/// [`WeakTopological`](Self::WeakTopological) order the nodes by a
/// depth-first walk along the analysis direction. The walk starts at the
/// root of a forward block solve when it is queued, then at every queued node
/// with no upstream neighbor, then at the lowest-index queued node it has not
/// reached, so it covers exactly the nodes that the solve can queue. Building
/// the order costs one walk and a table of `u32`s per node on every solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum IterationOrder {
    /// The lowest dense index first: the order of a solve that names none.
    #[default]
    Index,
    /// The earliest node in reverse postorder first, so a node runs after
    /// the upstream nodes that do not close a loop onto it.
    ReversePostorder,
    /// The earliest node in a
    /// [`WeakTopologicalOrder`](crate::WeakTopologicalOrder) first.
    ///
    /// Each component is contiguous and its head comes first, so a node after
    /// a component runs only once nothing in the component is queued. This is
    /// the one order that widens: the solve calls the problem's `widen` at
    /// every head, which makes infinite-height lattices terminate, and runs
    /// the configured [narrowing passes](SolveConfig::with_narrowing_passes)
    /// over each outermost component once it has converged.
    WeakTopological,
}

/// Deterministic solver limits and iteration order, shared by every fixpoint
/// solver.
///
/// The default is an unbounded solve in [`IterationOrder::Index`] with no
/// narrowing passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SolveConfig {
    max_steps: Option<usize>,
    order: IterationOrder,
    narrowing_passes: usize,
}

impl SolveConfig {
    /// An unbounded solve.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_steps: None,
            order: IterationOrder::Index,
            narrowing_passes: 0,
        }
    }

    /// Stop before processing more than `limit` worklist entries.
    #[must_use]
    pub const fn with_step_limit(limit: usize) -> Self {
        Self {
            max_steps: Some(limit),
            ..Self::new()
        }
    }

    /// Configured worklist-entry limit, if any.
    #[must_use]
    pub const fn max_steps(self) -> Option<usize> {
        self.max_steps
    }

    /// The same configuration, iterating in `order`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{IterationOrder, SolveConfig};
    ///
    /// let config = SolveConfig::with_step_limit(10_000)
    ///     .with_order(IterationOrder::WeakTopological)
    ///     .with_narrowing_passes(1);
    /// assert_eq!(config.order(), IterationOrder::WeakTopological);
    /// assert_eq!(config.max_steps(), Some(10_000));
    /// assert_eq!(SolveConfig::new().order(), IterationOrder::Index);
    /// ```
    #[must_use]
    pub const fn with_order(self, order: IterationOrder) -> Self {
        Self { order, ..self }
    }

    /// Configured iteration order.
    #[must_use]
    pub const fn order(self) -> IterationOrder {
        self.order
    }

    /// The same configuration, running at most `passes` descending passes
    /// over each outermost loop once its widened ascent has converged.
    ///
    /// A solve in [`IterationOrder::WeakTopological`] finishes an outermost
    /// component of the order before it moves past it, and nothing past it
    /// can queue the component again. At that point a pass recomputes every
    /// node of the component that the solve visited, once each and in order,
    /// and replaces the fact entering each head by the problem's `narrow` of
    /// the fact it held and the one now flowing in; the nodes past the
    /// component then start from the narrowed facts. The passes stop early
    /// when one changes nothing, and each recomputation counts as a step
    /// against the step limit. They run only in
    /// [`IterationOrder::WeakTopological`]: without widening, the ascending
    /// solve already ends at the least fixpoint.
    #[must_use]
    pub const fn with_narrowing_passes(self, passes: usize) -> Self {
        Self {
            narrowing_passes: passes,
            ..self
        }
    }

    /// Configured number of narrowing passes.
    #[must_use]
    pub const fn narrowing_passes(self) -> usize {
        self.narrowing_passes
    }

    /// Counts one more processed node against the step limit.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::StepLimitExceeded`], naming `pending_node`, when
    /// `steps` already reached the limit.
    pub(crate) fn count_step(
        self,
        steps: &mut usize,
        pending_node: usize,
    ) -> Result<(), SolveError> {
        if let Some(limit) = self.max_steps
            && *steps >= limit
        {
            return Err(SolveError::StepLimitExceeded {
                limit,
                steps: *steps,
                pending_node,
            });
        }
        *steps += 1;
        Ok(())
    }
}

/// A bounded solve did not reach a fixpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveError {
    /// The deterministic worklist still contained a node at the limit.
    StepLimitExceeded {
        /// Configured limit.
        limit: usize,
        /// Worklist entries already processed.
        steps: usize,
        /// Dense index of the next node that would have been processed.
        pending_node: usize,
    },
}

impl core::fmt::Display for SolveError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::StepLimitExceeded {
                limit,
                steps,
                pending_node,
            } => write!(
                formatter,
                "dataflow step limit {limit} exceeded after {steps} steps; next node is {pending_node}"
            ),
        }
    }
}

impl core::error::Error for SolveError {}

/// Failure from a fallible solve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrySolveError<E> {
    /// A boundary, merge, or transfer operation rejected the input.
    Problem(E),
    /// The solver reached its configured deterministic work limit.
    Solver(SolveError),
}

impl<E> From<SolveError> for TrySolveError<E> {
    fn from(error: SolveError) -> Self {
        Self::Solver(error)
    }
}

impl<E: core::fmt::Display> core::fmt::Display for TrySolveError<E> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Problem(error) => error.fmt(formatter),
            Self::Solver(error) => error.fmt(formatter),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for TrySolveError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Problem(error) => Some(error),
            Self::Solver(error) => Some(error),
        }
    }
}

/// Collapse a fallible-solver result whose consumer error is [`Infallible`].
pub(crate) fn collapse_infallible<T>(
    result: Result<T, TrySolveError<Infallible>>,
) -> Result<T, SolveError> {
    match result {
        Ok(facts) => Ok(facts),
        Err(TrySolveError::Solver(error)) => Err(error),
        Err(TrySolveError::Problem(error)) => match error {},
    }
}

/// A data flow problem to be solved by the fixpoint engine.
///
/// `V` is the rooted graph view that the problem runs on. A problem over
/// instructions implements the trait for every
/// [`CfgView`](super::cfg_view::CfgView) whose instruction type it reads, so
/// one problem serves [`Cfg`](crate::Cfg) and every consumer view alike.
///
/// `F` is the flow fact type (e.g. `BTreeSet<DefSite>` for reaching
/// definitions, or `BTreeSet<I::Variable>` for liveness).
///
/// # Termination
///
/// In [`IterationOrder::Index`] and [`IterationOrder::ReversePostorder`],
/// termination requires `meet` and `transfer` to be monotone over a
/// finite-height fact lattice. In [`IterationOrder::WeakTopological`] the
/// solver also calls [`widen`](Self::widen) at the head of every loop, and a
/// widening operator makes a monotone problem over an infinite-height lattice
/// (intervals, congruences, numeric bounds) terminate as well. The defaults of
/// `widen` and [`narrow`](Self::narrow) leave the facts unchanged, so a problem
/// that implements neither solves exactly as before in every order.
pub trait Problem<V: RootedView> {
    /// The flow fact (lattice element) type.
    type Fact: Clone + PartialEq;

    /// Analysis direction.
    fn direction(&self) -> Direction;

    /// Initial (bottom) value for each block.
    ///
    /// The view is given so a fact whose size depends on the graph, such as
    /// a [`DenseBits`](super::bits::DenseBits) over its blocks, can be built.
    fn bottom(&self, view: &V) -> Self::Fact;

    /// The fact entering the analysis boundary.
    ///
    /// A forward analysis starts the root with it, merged with whatever
    /// flows back into a root that is a branch target. A backward analysis
    /// starts every block without successors with it. A seeded solve also
    /// gives it to a seeded block with no upstream in the analysis
    /// direction.
    fn boundary(&self, view: &V) -> Self::Fact;

    /// Meet/join operator: merge information from multiple paths.
    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact;

    /// Transfer function: given the incoming fact for a block, compute
    /// the outgoing fact after the block's instructions are applied.
    fn transfer(&self, view: &V, block: V::NodeId, input: &Self::Fact) -> Self::Fact;

    /// Widens at a head of the weak topological order: combines the fact
    /// that entered the head before with the one that flows in now.
    ///
    /// The solver calls it only in [`IterationOrder::WeakTopological`], only
    /// at heads, and only when the fact flowing in differs from the one the
    /// head holds; the result replaces the head's IN fact (its OUT fact for a
    /// backward problem) and is what the head transfers. It must be at least
    /// both arguments, and every chain `x₀, x₁ = x₀ ∇ y₁, x₂ = x₁ ∇ y₂, …`
    /// must become stationary, which is what bounds the solve. Heads cut every
    /// cycle, so widening there suffices.
    ///
    /// The default returns `next`, which is no widening.
    fn widen(&self, _previous: &Self::Fact, next: &Self::Fact) -> Self::Fact {
        next.clone()
    }

    /// Narrows at a head during a descending pass: refines the fact that
    /// entered the head with the one that flows in now.
    ///
    /// The solver calls it at heads during the passes of
    /// [`SolveConfig::with_narrowing_passes`], once the widened ascent of the
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

/// A fallible data flow problem.
///
/// This is the error-preserving counterpart of [`Problem`], for verification
/// and abstract interpretation where a boundary, merge, or transfer can reject
/// the input program. The solver reports those consumer errors separately from
/// its own configured step limit. Termination without a step limit requires
/// what [`Problem`] requires: a monotone problem over a finite-height lattice,
/// or a [`widen`](Self::widen) in [`IterationOrder::WeakTopological`].
pub trait TryProblem<V: RootedView> {
    /// The flow fact (lattice element) type.
    type Fact: Clone + PartialEq;

    /// Consumer error produced by a boundary, merge, or transfer operation.
    type Error;

    /// Analysis direction.
    fn direction(&self) -> Direction;

    /// Initial (bottom) value for each block.
    fn bottom(&self, view: &V) -> Self::Fact;

    /// The fact entering the analysis boundary, as [`Problem::boundary`].
    ///
    /// # Errors
    ///
    /// Returns a consumer error when constructing the boundary fact fails.
    fn boundary(&self, view: &V) -> Result<Self::Fact, Self::Error>;

    /// Meet/join operator: merge information from multiple paths into
    /// `block`.
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the facts are incompatible.
    fn meet(
        &self,
        view: &V,
        block: V::NodeId,
        a: &Self::Fact,
        b: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error>;

    /// Transfer function: given the incoming fact for a block, compute
    /// the outgoing fact after the block's instructions are applied.
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the block rejects the incoming fact.
    fn transfer(
        &self,
        view: &V,
        block: V::NodeId,
        input: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error>;

    /// Widens at the head `block`, as [`Problem::widen`].
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the facts are incompatible.
    fn widen(
        &self,
        _view: &V,
        _block: V::NodeId,
        _previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(next.clone())
    }

    /// Narrows at the head `block`, as [`Problem::narrow`].
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the facts are incompatible.
    fn narrow(
        &self,
        _view: &V,
        _block: V::NodeId,
        _previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(next.clone())
    }
}

/// Per-node facts computed by a fixpoint solve, in program order: the IN fact
/// is the fact at a node's start and the OUT fact the fact at its end,
/// whichever way the analysis flows.
///
/// The block, node and edge solvers all report their node facts in this one
/// type. For a backward analysis the IN fact is therefore what the node's
/// transfer produced and the OUT fact what the node's successors handed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Facts<F> {
    /// The IN fact for each node (indexed by `DenseId::index()`).
    input: Vec<F>,
    /// The OUT fact for each node (indexed by `DenseId::index()`).
    output: Vec<F>,
    /// Number of worklist entries processed.
    steps: usize,
}

impl<F> Facts<F> {
    /// Get the IN fact for a node.
    #[must_use]
    pub fn fact_in<N: DenseId>(&self, node: N) -> &F {
        &self.input[node.index()]
    }

    /// Get the OUT fact for a node.
    #[must_use]
    pub fn fact_out<N: DenseId>(&self, node: N) -> &F {
        &self.output[node.index()]
    }

    /// Number of worklist entries processed.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }

    /// Creates the facts of a solve, from vectors indexed by
    /// `DenseId::index()` up to the node bound of the graph.
    pub(crate) const fn new(input: Vec<F>, output: Vec<F>, steps: usize) -> Self {
        Self {
            input,
            output,
            steps,
        }
    }
}

/// Meets two optional facts, treating `None` as unreachable bottom.
///
/// This is the standard reachability lifting of a lattice: `None` means "no
/// path reaches this point yet", so it is the identity of the meet and only
/// two reached facts invoke the inner `meet`. Use it as the
/// [`Problem::meet`] of an analysis whose `Fact` is `Option<F>` and whose
/// `bottom` is `None`, so unreachable code never fabricates a fact.
#[must_use]
pub fn meet_options<F: Clone>(
    left: &Option<F>,
    right: &Option<F>,
    meet: impl FnOnce(&F, &F) -> F,
) -> Option<F> {
    match (left, right) {
        (None, None) => None,
        (Some(fact), None) | (None, Some(fact)) => Some(fact.clone()),
        (Some(left), Some(right)) => Some(meet(left, right)),
    }
}

/// Run the fixpoint iteration for the given problem on the CFG without a
/// step limit.
///
/// # Errors
///
/// The unbounded configuration cannot produce a solver-limit error. The
/// `Result` matches [`solve_problem_with_config`] so callers can switch
/// configurations without changing result handling.
///
/// # Examples
///
/// A problem names its direction, its lattice and its transfer. This one
/// marks the blocks that flow from the entry reaches. Analyses whose
/// exceptional edges carry their own facts run on the
/// [edge-sensitive solver](crate::dataflow::edge_fixpoint) instead.
///
/// ```
/// use cfglib::{BlockId, Cfg, Direction, EdgeKind, Problem, solve_problem};
///
/// struct Reached;
///
/// impl Problem<Cfg<()>> for Reached {
///     type Fact = bool;
///
///     fn direction(&self) -> Direction {
///         Direction::Forward
///     }
///     fn bottom(&self, _cfg: &Cfg<()>) -> bool {
///         false
///     }
///     fn boundary(&self, _cfg: &Cfg<()>) -> bool {
///         true
///     }
///     fn meet(&self, a: &bool, b: &bool) -> bool {
///         *a || *b
///     }
///     fn transfer(&self, _cfg: &Cfg<()>, _block: BlockId, input: &bool) -> bool {
///         *input
///     }
/// }
///
/// let mut cfg = Cfg::<()>::new();
/// let body = cfg.new_block();
/// let orphan = cfg.new_block();
/// cfg.add_edge(cfg.entry(), body, EdgeKind::Fallthrough);
///
/// let facts = solve_problem(&cfg, &Reached).unwrap();
/// assert!(*facts.fact_out(body));
/// assert!(!*facts.fact_out(orphan));
/// ```
pub fn solve_problem<V: RootedView, P: Problem<V>>(
    view: &V,
    problem: &P,
) -> Result<Facts<P::Fact>, SolveError> {
    solve_problem_with_config(view, problem, SolveConfig::new())
}

/// Run the fixpoint iteration from only the initial `seeds`.
///
/// Every fact starts at `bottom`, but only the seeds — and whatever their
/// transfers reach — are ever visited, so an incremental or dirty-region
/// analysis pays for the part of the CFG its change actually reaches. A block
/// with no upstream in the analysis direction receives the problem's entry
/// fact when visited; unvisited blocks stay at `bottom`.
///
/// # Panics
///
/// Panics when a seed is not a live node of `view`.
///
/// # Errors
///
/// The unbounded configuration cannot produce a solver-limit error.
pub fn solve_problem_from<V: RootedView, P: Problem<V>>(
    view: &V,
    problem: &P,
    seeds: &[V::NodeId],
) -> Result<Facts<P::Fact>, SolveError> {
    solve_problem_from_with_config(view, problem, seeds, SolveConfig::new())
}

/// Run the fixpoint iteration with deterministic bounded iteration.
///
/// # Errors
///
/// Returns [`SolveError::StepLimitExceeded`] when work remains at the
/// configured limit.
pub fn solve_problem_with_config<V: RootedView, P: Problem<V>>(
    view: &V,
    problem: &P,
    config: SolveConfig,
) -> Result<Facts<P::Fact>, SolveError> {
    let fallible = InfallibleProblem(problem);
    collapse_infallible(try_solve_with_worklist(
        view,
        &fallible,
        initial_worklist(view, Problem::direction(problem)),
        config,
    ))
}

/// Run the fixpoint iteration from `seeds` with a deterministic step limit.
///
/// # Panics
///
/// Panics when a seed is not a live node of `view`.
///
/// # Errors
///
/// Returns [`SolveError::StepLimitExceeded`] when work remains at the
/// configured limit.
pub fn solve_problem_from_with_config<V: RootedView, P: Problem<V>>(
    view: &V,
    problem: &P,
    seeds: &[V::NodeId],
    config: SolveConfig,
) -> Result<Facts<P::Fact>, SolveError> {
    let fallible = InfallibleProblem(problem);
    collapse_infallible(try_solve_with_worklist(
        view,
        &fallible,
        Worklist::seeded(view, seeds),
        config,
    ))
}

/// Run a fallible fixpoint iteration without a step limit.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error. The unbounded
/// configuration cannot produce a solver-limit error.
pub fn try_solve_problem<V: RootedView, P: TryProblem<V>>(
    view: &V,
    problem: &P,
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>> {
    try_solve_problem_with_config(view, problem, SolveConfig::new())
}

/// Run a fallible fixpoint iteration from only the initial `seeds`.
///
/// # Panics
///
/// Panics when a seed is not a live node of `view`.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error. The unbounded
/// configuration cannot produce a solver-limit error.
pub fn try_solve_problem_from<V: RootedView, P: TryProblem<V>>(
    view: &V,
    problem: &P,
    seeds: &[V::NodeId],
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>> {
    try_solve_problem_from_with_config(view, problem, seeds, SolveConfig::new())
}

/// Run a fallible fixpoint iteration with a deterministic step limit.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error or
/// [`TrySolveError::Solver`] when work remains at the configured limit.
pub fn try_solve_problem_with_config<V: RootedView, P: TryProblem<V>>(
    view: &V,
    problem: &P,
    config: SolveConfig,
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>> {
    try_solve_with_worklist(
        view,
        problem,
        initial_worklist(view, problem.direction()),
        config,
    )
}

/// Run a fallible fixpoint iteration from `seeds` with a deterministic step
/// limit.
///
/// # Panics
///
/// Panics when a seed is not a live node of `view`.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error or
/// [`TrySolveError::Solver`] when work remains at the configured limit.
pub fn try_solve_problem_from_with_config<V: RootedView, P: TryProblem<V>>(
    view: &V,
    problem: &P,
    seeds: &[V::NodeId],
    config: SolveConfig,
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>> {
    try_solve_with_worklist(view, problem, Worklist::seeded(view, seeds), config)
}

/// The blocks the full solve starts from: those reachable from the root going
/// forward, every block going backward.
fn initial_worklist<V: RootedView>(view: &V, direction: Direction) -> Worklist {
    match direction {
        Direction::Forward => Worklist::of(
            view.node_bound(),
            depth_first_preorder(view, view.root(), direction.traversal())
                .into_iter()
                .map(DenseId::index),
        ),
        Direction::Backward => Worklist::live(view),
    }
}

fn try_solve_with_worklist<V: RootedView, P: TryProblem<V>>(
    view: &V,
    problem: &P,
    worklist: Worklist,
    config: SolveConfig,
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>> {
    let direction = problem.direction();
    let entry = matches!(direction, Direction::Forward).then(|| view.root());
    node_fixpoint::try_solve_with_worklist(view, &BlockProblem(problem), worklist, entry, config)
}

/// Adapter that runs a [`TryProblem`] on the node solver's core.
struct BlockProblem<'p, P>(&'p P);

impl<V: RootedView, P: TryProblem<V>> TryNodeProblem<V> for BlockProblem<'_, P> {
    type Fact = P::Fact;
    type Error = P::Error;

    fn direction(&self) -> Direction {
        TryProblem::direction(self.0)
    }

    fn bottom(&self, view: &V) -> Self::Fact {
        TryProblem::bottom(self.0, view)
    }

    fn boundary(&self, view: &V) -> Result<Self::Fact, Self::Error> {
        TryProblem::boundary(self.0, view)
    }

    fn meet(
        &self,
        view: &V,
        block: V::NodeId,
        a: &Self::Fact,
        b: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        TryProblem::meet(self.0, view, block, a, b)
    }

    fn transfer(
        &self,
        view: &V,
        block: V::NodeId,
        input: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        TryProblem::transfer(self.0, view, block, input)
    }

    fn widen(
        &self,
        view: &V,
        block: V::NodeId,
        previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        TryProblem::widen(self.0, view, block, previous, next)
    }

    fn narrow(
        &self,
        view: &V,
        block: V::NodeId,
        previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        TryProblem::narrow(self.0, view, block, previous, next)
    }
}

/// Adapter that runs an infallible [`Problem`] on the fallible solver core.
struct InfallibleProblem<'p, P>(&'p P);

impl<V: RootedView, P: Problem<V>> TryProblem<V> for InfallibleProblem<'_, P> {
    type Fact = P::Fact;
    type Error = Infallible;

    fn direction(&self) -> Direction {
        Problem::direction(self.0)
    }

    fn bottom(&self, view: &V) -> Self::Fact {
        Problem::bottom(self.0, view)
    }

    fn boundary(&self, view: &V) -> Result<Self::Fact, Self::Error> {
        Ok(Problem::boundary(self.0, view))
    }

    fn meet(
        &self,
        _view: &V,
        _block: V::NodeId,
        a: &Self::Fact,
        b: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(Problem::meet(self.0, a, b))
    }

    fn transfer(
        &self,
        view: &V,
        block: V::NodeId,
        input: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(Problem::transfer(self.0, view, block, input))
    }

    fn widen(
        &self,
        _view: &V,
        _block: V::NodeId,
        previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(Problem::widen(self.0, previous, next))
    }

    fn narrow(
        &self,
        _view: &V,
        _block: V::NodeId,
        previous: &Self::Fact,
        next: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(Problem::narrow(self.0, previous, next))
    }
}

#[cfg(test)]
mod tests;
