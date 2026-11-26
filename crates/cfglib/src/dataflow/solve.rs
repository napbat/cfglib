//! The vocabulary that every fixpoint solver shares: which nodes a solve
//! starts from, how it is bounded and ordered, what it reports, and how it
//! fails.
//!
//! The node solver ([`solve_node_problem`](super::node_fixpoint::solve_node_problem))
//! and the edge solver
//! ([`solve_edge_problem`](super::edge_fixpoint::solve_edge_problem)) each
//! have one entry point. It takes the problem and one [`Solve`] descriptor:
//! the [`Seeds`] the solve starts from and its [`SolveConfig`]. Their
//! [`stream`](super::stream) and [`resume`](super::resume) forms take the
//! same descriptor. Every problem trait comes in an infallible and a
//! fallible form; the fallible form has a blanket implementation for every
//! infallible problem, and the solvers take the fallible form, so an
//! infallible problem reports [`TrySolveError<Infallible>`], which converts
//! into a [`SolveError`] with `?`.
//!
//! A [`SolveConfig`] bounds a solve deterministically and selects its
//! [`IterationOrder`]. In [`IterationOrder::WeakTopological`] every solver
//! widens at the heads of a
//! [`WeakTopologicalOrder`](crate::WeakTopologicalOrder) and can narrow
//! afterwards, through the `widen` and `narrow` methods that every problem
//! trait carries with defaults that change nothing.
//!
//! Every solver returns all its facts, two per node and one per edge for the
//! edge solver, so it holds them all until it returns. A consumer that reads
//! each node's facts once — to check a property, report a diagnostic, or copy
//! a projection into its own table — and whose facts are large prefers the
//! [`stream`](super::stream) of the same solve, which hands each node out
//! once its facts are final and frees them as soon as nothing reads them.
//!
//! # One vocabulary for the problem families
//!
//! Every problem names its starting fact `boundary`, its unreached or
//! initial fact `bottom`, and its direction [`Direction`]. Two shapes of
//! `boundary` exist, for one reason:
//!
//! - [`NodeProblem`](super::node_fixpoint::NodeProblem) returns one fact, and
//!   the *solver* decides which nodes receive it: a node with no upstream
//!   neighbor along the analysis direction, and the root of
//!   [`Seeds::Reachable`]. A node graph has a structural boundary, so naming
//!   the nodes would only restate it.
//! - [`EdgeProblem`](super::edge_fixpoint::EdgeProblem) returns an optional
//!   fact *per node*, and the *problem* decides which nodes get one, merged
//!   with whatever flows in. Exceptional-edge analyses seed handler entries,
//!   coroutine resumptions, and several roots, none of which is structural.
//!
//! A fallible problem's `meet` also receives the view and the merge point,
//! so an error can name where two facts disagreed; an infallible `meet` is
//! a pure lattice operation and receives only the two facts. A problem whose
//! facts exist only where execution reaches uses `Option` facts with `None`
//! as bottom, and meets them with [`meet_options`] or [`try_meet_options`].

extern crate alloc;
use alloc::vec::Vec;
use core::convert::Infallible;

use super::worklist::Worklist;
use crate::graph::traverse::reachable_in;
use crate::graph::view::{DenseId, Direction, GraphView};

#[cfg(test)]
mod tests;

impl Direction {
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
/// root of [`Seeds::Reachable`], then at every queued node with no upstream
/// neighbor, then at the lowest-index queued node it has not reached, so it
/// covers exactly the nodes that the solve can queue. Building the order
/// costs one walk and a table of `u32`s per node on every solve.
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
    /// the configured [narrowing passes](SolveConfig::narrowing_passes) over
    /// each outermost component once it has converged.
    WeakTopological,
}

/// Deterministic solver limits and iteration order.
///
/// The default is an unbounded solve in [`IterationOrder::Index`] with no
/// narrowing passes.
///
/// # Examples
///
/// ```
/// use cfglib::{IterationOrder, SolveConfig};
///
/// let config = SolveConfig {
///     max_steps: Some(10_000),
///     order: IterationOrder::WeakTopological,
///     narrowing_passes: 1,
/// };
/// assert_eq!(SolveConfig::default().order, IterationOrder::Index);
/// assert_ne!(config, SolveConfig::default());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SolveConfig {
    /// The most worklist entries the solve processes; with work left at the
    /// limit it stops with [`SolveError::StepLimitExceeded`]. `None` is
    /// unbounded.
    pub max_steps: Option<usize>,
    /// The order in which queued nodes run.
    pub order: IterationOrder,
    /// The most descending passes over each outermost loop once its widened
    /// ascent has converged.
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
    pub narrowing_passes: usize,
}

/// The nodes a solve starts from.
///
/// Every node starts at the problem's `bottom`, and a solve visits a node
/// only when it is queued: at the start, or after an upstream neighbor's
/// output changed. The seeds decide what is queued at the start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Seeds<'a, N> {
    /// Every live node. A node with no upstream neighbor along the analysis
    /// direction receives the boundary fact.
    #[default]
    Live,
    /// Every node that the root reaches along the analysis direction, the
    /// rooted solve of a control-flow graph: forward from its entry, every
    /// other block stays at `bottom`.
    ///
    /// The node solver gives the root its boundary fact in front of whatever
    /// flows back into it, so a root that is a branch target still starts
    /// from it. The edge solver's problem names its boundary nodes itself,
    /// so there the root only chooses what is queued.
    Reachable(N),
    /// Only these nodes, duplicates once; with none, nothing is visited and
    /// every fact stays `bottom`.
    ///
    /// The seeds and whatever their changed outputs reach are visited, and
    /// nothing else: an incremental or dirty-region analysis seeds the nodes
    /// whose inputs moved and pays for the part of the graph that its change
    /// actually reaches. This solves a fresh problem from a subset of entry
    /// points; it does not resume one, so facts downstream of the seeds miss
    /// what flows in from the nodes it never visits. To re-solve after an
    /// edit from the facts of the solve before it, use
    /// [`resume`](super::resume).
    Of(&'a [N]),
}

impl<N: DenseId> Seeds<'_, N> {
    /// The worklist a solve along `direction` starts from, and the node that
    /// receives the boundary fact in front of its upstream flow.
    ///
    /// # Panics
    ///
    /// Panics when a seed or the root is not a live node of `graph`.
    pub(super) fn start<G: GraphView<NodeId = N>>(
        self,
        graph: &G,
        direction: Direction,
    ) -> (Worklist, Option<N>) {
        match self {
            Self::Live => (Worklist::live(graph), None),
            Self::Of(seeds) => (Worklist::seeded(graph, seeds), None),
            Self::Reachable(root) => {
                assert!(
                    graph.node_ids().any(|node| node == root),
                    "seed node is out of range"
                );
                let mut reached = Vec::new();
                reachable_in(
                    graph,
                    [root],
                    direction,
                    None,
                    &mut reached,
                    &mut Vec::new(),
                );
                let indices = reached.iter().enumerate().filter(|&(_, &hit)| hit);
                let worklist = Worklist::of(graph.node_bound(), indices.map(|(index, _)| index));
                (worklist, Some(root))
            }
        }
    }

    /// The nodes that a solve from these seeds computes, as a mask over the
    /// node bound of `graph`, and the root that receives the boundary fact:
    /// every live node, or what the seeds reach along `direction`.
    pub(super) fn domain<G: GraphView<NodeId = N>>(
        self,
        graph: &G,
        direction: Direction,
    ) -> (Option<Vec<bool>>, Option<N>) {
        let root;
        let (starts, entry): (&[N], Option<N>) = match self {
            Self::Live => return (None, None),
            Self::Reachable(node) => {
                root = node;
                (core::slice::from_ref(&root), Some(node))
            }
            Self::Of(seeds) => (seeds, None),
        };
        let mut reached = Vec::new();
        reachable_in(
            graph,
            starts.iter().copied(),
            direction,
            None,
            &mut reached,
            &mut Vec::new(),
        );
        (Some(reached), entry)
    }
}

/// The seeds and limits of one solve: what every solver entry point takes
/// besides the graph and the problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Solve<'a, N> {
    /// The nodes the solve starts from.
    pub seeds: Seeds<'a, N>,
    /// Its limits and iteration order.
    pub config: SolveConfig,
}

impl<N> Default for Solve<'_, N> {
    /// Every live node, unbounded, in index order.
    fn default() -> Self {
        Self {
            seeds: Seeds::Live,
            config: SolveConfig::default(),
        }
    }
}

/// Counts one more step against `max_steps`.
///
/// Every bounded fixpoint of the crate counts its steps through
/// [`count_steps`]: the node and edge solvers in worklist entries, IFDS in
/// path edges, the slot drivers in evaluations. It runs once per step inside
/// cores that the caller's crate instantiates, so it is inline: without the
/// attribute, every step would pay a call across crates for one comparison.
///
/// # Errors
///
/// Returns [`SolveError::StepLimitExceeded`], naming `pending_node`, when
/// `steps` already reached the limit.
#[inline]
pub(crate) fn count_step(
    max_steps: Option<usize>,
    steps: &mut usize,
    pending_node: usize,
) -> Result<(), SolveError> {
    count_steps(max_steps, steps, 1, pending_node)
}

/// Counts `amount` more steps against `max_steps`, all or none: a scheduler
/// that runs several steps at once (a wave) runs them only when all fit.
///
/// # Errors
///
/// Returns [`SolveError::StepLimitExceeded`], naming `pending_node`, when
/// the steps would pass the limit.
#[inline]
pub(crate) fn count_steps(
    max_steps: Option<usize>,
    steps: &mut usize,
    amount: usize,
    pending_node: usize,
) -> Result<(), SolveError> {
    if let Some(limit) = max_steps
        && *steps + amount > limit
    {
        return Err(SolveError::StepLimitExceeded {
            limit,
            steps: *steps,
            pending_node,
        });
    }
    *steps += amount;
    Ok(())
}

/// A bounded fixpoint ran out of steps before it converged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveError {
    /// Work was still pending at the step limit.
    StepLimitExceeded {
        /// Configured limit.
        limit: usize,
        /// Steps already taken.
        steps: usize,
        /// Dense index of what would have run next: a node, the node of a
        /// path edge, or a unit.
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
                "step limit {limit} exceeded after {steps} steps; next is {pending_node}"
            ),
        }
    }
}

impl core::error::Error for SolveError {}

impl From<TrySolveError<Infallible>> for SolveError {
    /// The solver error of an infallible problem, which can fail only by its
    /// step limit.
    fn from(error: TrySolveError<Infallible>) -> Self {
        match error {
            TrySolveError::Solver(error) => error,
            TrySolveError::Problem(error) => match error {},
        }
    }
}

/// Failure from a solve of a fallible problem.
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

/// What a fixpoint did, in one shape for every bounded fixpoint of the
/// crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SolveStats {
    steps: usize,
    rounds: usize,
    peak: usize,
}

impl SolveStats {
    pub(crate) const fn new(steps: usize, rounds: usize, peak: usize) -> Self {
        Self {
            steps,
            rounds,
            peak,
        }
    }

    /// The steps taken, the unit that a step limit counts: worklist entries
    /// of a node or edge solve (narrowing passes included), path edges of
    /// IFDS, evaluations of a slot fixpoint.
    #[must_use]
    pub const fn step_count(self) -> usize {
        self.steps
    }

    /// The rounds a scheduler ran: the waves of
    /// [`wave_fixpoint`](crate::wave_fixpoint) and the sweeps of
    /// [`ordered_fixpoint`](crate::ordered_fixpoint). A worklist solve runs
    /// no rounds and reports zero.
    #[must_use]
    pub const fn round_count(self) -> usize {
        self.rounds
    }

    /// The most facts the fixpoint held at once: IN and OUT facts of nodes,
    /// and facts on edges for an edge problem, or the path edges of IFDS. A
    /// solve that returns every fact holds them all from start to end; a
    /// [stream](super::stream) does not hold a fact that still equals bottom
    /// because nothing reached it. A slot fixpoint holds no facts of its own
    /// and reports zero.
    #[must_use]
    pub const fn peak_count(self) -> usize {
        self.peak
    }
}

/// Per-node facts computed by a fixpoint solve, in program order: the IN fact
/// is the fact at a node's start and the OUT fact the fact at its end,
/// whichever way the analysis flows.
///
/// The node and edge solvers both report their node facts in this one type.
/// For a backward analysis the IN fact is therefore what the node's transfer
/// produced and the OUT fact what the node's successors handed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Facts<F> {
    /// The IN fact for each node (indexed by `DenseId::index()`).
    input: Vec<F>,
    /// The OUT fact for each node (indexed by `DenseId::index()`).
    output: Vec<F>,
    stats: SolveStats,
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

    /// What the solve did. It held every IN and OUT fact throughout, so its
    /// peak is two facts per node slot.
    #[must_use]
    pub const fn stats(&self) -> SolveStats {
        self.stats
    }

    /// Creates the facts of a solve, from vectors indexed by
    /// `DenseId::index()` up to the node bound of the graph, and the facts
    /// the solve held besides them.
    pub(crate) fn new(input: Vec<F>, output: Vec<F>, steps: usize, also_held: usize) -> Self {
        let peak = input.len() + output.len() + also_held;
        Self {
            input,
            output,
            stats: SolveStats::new(steps, 0, peak),
        }
    }

    /// The IN and OUT vectors, for a solve that resumes from them.
    pub(crate) fn into_parts(self) -> (Vec<F>, Vec<F>) {
        (self.input, self.output)
    }
}

/// Meets two optional facts, treating `None` as unreachable bottom.
///
/// This is the standard reachability lifting of a lattice: `None` means "no
/// path reaches this point yet", so it is the identity of the meet and only
/// two reached facts invoke the inner `meet`. Use it as the `meet` of an
/// analysis whose `Fact` is `Option<F>` and whose `bottom` is `None`, so
/// unreachable code never fabricates a fact; [`try_meet_options`] is the
/// same lifting for a fallible problem.
#[must_use]
pub fn meet_options<F: Clone>(
    left: &Option<F>,
    right: &Option<F>,
    meet: impl FnOnce(&F, &F) -> F,
) -> Option<F> {
    let lifted = try_meet_options(left, right, |left, right| {
        Ok::<_, Infallible>(meet(left, right))
    });
    match lifted {
        Ok(merged) => merged,
        Err(never) => match never {},
    }
}

/// Meets two optional facts with a fallible `meet`, treating `None` as
/// unreachable bottom, as [`meet_options`] does.
///
/// A fallible problem over `Option` facts uses it as its `meet`, so that only
/// two reached facts can fail to merge.
///
/// # Errors
///
/// Returns what `meet` returns when both facts are reached and it rejects
/// them.
pub fn try_meet_options<F: Clone, E>(
    left: &Option<F>,
    right: &Option<F>,
    meet: impl FnOnce(&F, &F) -> Result<F, E>,
) -> Result<Option<F>, E> {
    match (left, right) {
        (None, None) => Ok(None),
        (Some(fact), None) | (None, Some(fact)) => Ok(Some(fact.clone())),
        (Some(left), Some(right)) => meet(left, right).map(Some),
    }
}
