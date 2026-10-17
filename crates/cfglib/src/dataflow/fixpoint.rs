//! Generic fixpoint iteration engine for data flow analysis.
//!
//! Supports both **forward** and **backward** analyses via a worklist
//! algorithm that iterates until the solution stabilizes.
//!
//! The crate's three generic solvers — this one, the node-level
//! [`node_fixpoint`], and the edge-sensitive
//! [`edge_fixpoint`](super::edge_fixpoint) — carry the same facility matrix:
//! a full solve, a seeded solve (`_from`), a deterministically bounded solve
//! (`_with_config`), and fallible `try_` counterparts, all sharing
//! [`SolveConfig`], [`SolveError`], and [`TrySolveError`]. This one runs
//! per-block facts on the node solver's core.

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

/// Deterministic solver limits, shared by every fixpoint solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SolveConfig {
    max_steps: Option<usize>,
}

impl SolveConfig {
    /// An unbounded solve.
    #[must_use]
    pub const fn new() -> Self {
        Self { max_steps: None }
    }

    /// Stop before processing more than `limit` worklist entries.
    #[must_use]
    pub const fn with_step_limit(limit: usize) -> Self {
        Self {
            max_steps: Some(limit),
        }
    }

    /// Configured worklist-entry limit, if any.
    #[must_use]
    pub const fn max_steps(self) -> Option<usize> {
        self.max_steps
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
/// Termination requires `meet` and `transfer` to be monotone over a
/// finite-height fact lattice.
pub trait Problem<V: RootedView> {
    /// The flow fact (lattice element) type.
    type Fact: Clone + PartialEq;

    /// Analysis direction.
    fn direction(&self) -> Direction;

    /// Initial (bottom) value for each block.
    fn bottom(&self) -> Self::Fact;

    /// The fact entering the analysis boundary.
    ///
    /// A forward analysis starts the root with it, merged with whatever
    /// flows back into a root that is a branch target. A backward analysis
    /// starts every block without successors with it. A seeded solve also
    /// gives it to a seeded block with no upstream in the analysis
    /// direction.
    fn entry_fact(&self) -> Self::Fact;

    /// Meet/join operator: merge information from multiple paths.
    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact;

    /// Transfer function: given the incoming fact for a block, compute
    /// the outgoing fact after the block's instructions are applied.
    fn transfer(&self, view: &V, block: V::NodeId, input: &Self::Fact) -> Self::Fact;
}

/// A fallible data flow problem.
///
/// This is the error-preserving counterpart of [`Problem`], for verification
/// and abstract interpretation where a boundary, merge, or transfer can reject
/// the input program. The solver reports those consumer errors separately from
/// its own configured step limit. Termination without a step limit requires
/// `meet` and `transfer` to be monotone over a finite-height fact lattice.
pub trait TryProblem<V: RootedView> {
    /// The flow fact (lattice element) type.
    type Fact: Clone + PartialEq;

    /// Consumer error produced by a boundary, merge, or transfer operation.
    type Error;

    /// Analysis direction.
    fn direction(&self) -> Direction;

    /// Initial (bottom) value for each block.
    fn bottom(&self) -> Self::Fact;

    /// The fact entering the analysis boundary, as [`Problem::entry_fact`].
    ///
    /// # Errors
    ///
    /// Returns a consumer error when constructing the boundary fact fails.
    fn entry_fact(&self) -> Result<Self::Fact, Self::Error>;

    /// Meet/join operator: merge information from multiple paths.
    ///
    /// # Errors
    ///
    /// Returns a consumer error when the facts are incompatible.
    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Result<Self::Fact, Self::Error>;

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
}

/// Per-block facts computed by a fixpoint solve, in program order: the IN fact
/// is the fact at a block's start and the OUT fact the fact at its end,
/// whichever way the analysis flows.
#[derive(Debug, Clone)]
pub struct Facts<F> {
    /// The IN fact for each block (indexed by `DenseId::index()`).
    block_in: Vec<F>,
    /// The OUT fact for each block (indexed by `DenseId::index()`).
    block_out: Vec<F>,
    /// Number of worklist entries processed.
    steps: usize,
}

impl<F> Facts<F> {
    /// Get the IN fact for a block.
    #[must_use]
    pub fn fact_in<N: DenseId>(&self, block: N) -> &F {
        &self.block_in[block.index()]
    }

    /// Get the OUT fact for a block.
    #[must_use]
    pub fn fact_out<N: DenseId>(&self, block: N) -> &F {
        &self.block_out[block.index()]
    }

    /// Number of worklist entries processed.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }

    /// Creates the facts of a solve that a specialized solver performed.
    ///
    /// `block_in` and `block_out` are indexed by `DenseId::index()` up to
    /// the node bound of the graph, as the generic solver indexes them.
    pub(crate) const fn from_parts(block_in: Vec<F>, block_out: Vec<F>, steps: usize) -> Self {
        Self {
            block_in,
            block_out,
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
/// [`LivenessProblem`](crate::dataflow::liveness::LivenessProblem) is a
/// concrete problem. Analyses whose exceptional edges carry their own facts,
/// such as [`ReachingDefs`](crate::dataflow::reaching::ReachingDefs), run on
/// the [edge-sensitive solver](crate::dataflow::edge_fixpoint) instead.
///
/// ```
/// # use cfglib::{Cfg, EdgeKind, InstrInfo};
/// # #[derive(Debug, Clone)]
/// # struct Inst { uses: Vec<u16>, defs: Vec<u16> }
/// # impl InstrInfo for Inst {
/// #     type Variable = u16;
/// #     fn uses(&self) -> &[u16] { &self.uses }
/// #     fn defs(&self) -> &[u16] { &self.defs }
/// # }
/// use cfglib::{LivenessProblem, solve_problem};
///
/// let mut cfg = Cfg::<Inst>::new();
/// cfg.block_mut(cfg.entry()).push(Inst { uses: vec![], defs: vec![0] });
///
/// let facts = solve_problem(&cfg, &LivenessProblem).unwrap();
/// assert!(facts.fact_in(cfg.entry()).is_empty()); // r0 defined, not used
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
            depth_first_preorder(view, view.root(), TraversalDirection::Outgoing)
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
        .map(|facts| facts.into_block_facts(direction))
}

/// Adapter that runs a [`TryProblem`] on the node solver's core.
struct BlockProblem<'p, P>(&'p P);

impl<V: RootedView, P: TryProblem<V>> TryNodeProblem<V> for BlockProblem<'_, P> {
    type Fact = P::Fact;
    type Error = P::Error;

    fn direction(&self) -> Direction {
        TryProblem::direction(self.0)
    }

    fn bottom(&self, _view: &V) -> Self::Fact {
        TryProblem::bottom(self.0)
    }

    fn boundary(&self, _view: &V) -> Result<Self::Fact, Self::Error> {
        TryProblem::entry_fact(self.0)
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Result<Self::Fact, Self::Error> {
        TryProblem::meet(self.0, a, b)
    }

    fn transfer(
        &self,
        view: &V,
        block: V::NodeId,
        input: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        TryProblem::transfer(self.0, view, block, input)
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

    fn bottom(&self) -> Self::Fact {
        Problem::bottom(self.0)
    }

    fn entry_fact(&self) -> Result<Self::Fact, Self::Error> {
        Ok(Problem::entry_fact(self.0))
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Result<Self::Fact, Self::Error> {
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
}

#[cfg(test)]
mod tests;
