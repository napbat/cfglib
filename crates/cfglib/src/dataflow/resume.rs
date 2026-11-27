//! Resumed solves: re-solve after an edit from the facts of the solve before
//! it, instead of from bottom.
//!
//! A seeded solve ([`Seeds::Of`](super::solve::Seeds::Of)) starts every node
//! at bottom and visits only what its seeds reach, so it computes the facts
//! of a region against bottom inputs from everything it did not visit. After
//! an edit those facts are too small wherever the changed region reads from
//! the rest of the graph. [`resume_node_problem`] and [`resume_edge_problem`]
//! take the facts of the previous solve instead, reuse them in place, and
//! re-solve only what the edit can change (J. Erhard et al., *Interactive
//! abstract interpretation: reanalyzing multithreaded C programs for cheap*,
//! STTT 26, 2024; S. Arzt, E. Bodden, *Reviser*, ICSE 2014).
//!
//! [`Resume::changed`] names every node whose transfer or boundary changed,
//! every node whose upstream edges along the analysis direction changed (an
//! edge added, removed or redirected into it), and every removed node. The
//! edit must keep node and edge identities: a compaction renumbers them, so
//! its facts cannot be resumed.
//!
//! A resume takes the [`Solve`] descriptor of the solve it continues. Its
//! config applies in full: order, step limit and narrowing passes. Its seeds
//! name the nodes the solve computes: [`Seeds::Live`](super::solve::Seeds::Live)
//! every node, [`Seeds::Reachable`](super::solve::Seeds::Reachable) what the
//! root reaches, with the root receiving the boundary fact, and
//! [`Seeds::Of`](super::solve::Seeds::Of) what the named nodes reach. A node
//! outside that domain is never re-solved: a reset leaves it at bottom, as a
//! fresh rooted solve leaves what its root does not reach, and otherwise it
//! keeps its previous fact.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::bits::DenseBits;
use super::edge_fixpoint::{self, EdgeFacts, TryEdgeProblem};
use super::node_fixpoint::{self, TryNodeProblem};
use super::solve::{Facts, Solve, SolveConfig, TrySolveError};
use super::worklist::Worklist;
use crate::block::BlockId;
use crate::graph::edge_traverse::EdgeAdjacency;
use crate::graph::edge_view::EdgeView;
use crate::graph::traverse::{Adjacency, by_axis};
use crate::graph::view::{DenseId, Direction, GraphView};

/// How a resumed solve treats the facts of the previous solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResumeMode {
    /// The edit may shrink facts. Every node that a changed node reaches
    /// along the analysis direction starts again from bottom; every other
    /// node keeps its fact, which the edit cannot change. The result equals
    /// a full solve of the edited graph, and the work is that of the region
    /// the edit reaches.
    Reset,
    /// The caller guarantees that no fact of the new least fixpoint is below
    /// the previous one, as when edges or seeds are only added to a may
    /// analysis. The changed nodes are re-solved from the previous facts and
    /// the solve climbs from there, with nothing reset.
    Grow,
}

/// What an edit changed, and how to resume from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resume<'a, N = BlockId> {
    /// Every node whose transfer, boundary or upstream edges changed, and
    /// every removed node.
    pub changed: &'a [N],
    /// How the previous facts are treated.
    pub mode: ResumeMode,
}

/// The edit, with the domain of the solve it resumes.
struct Edit<'c, N> {
    changed: &'c [N],
    mode: ResumeMode,
    /// The nodes the solve computes, `None` for every node.
    solved: Option<Vec<bool>>,
    /// The root that receives the boundary fact.
    entry: Option<N>,
}

impl<'c, N: DenseId> Edit<'c, N> {
    fn new<G: GraphView<NodeId = N>>(
        graph: &G,
        edit: Resume<'c, N>,
        solve: Solve<'_, N>,
        direction: Direction,
    ) -> Self {
        let (solved, entry) = solve.seeds.domain(graph, direction);
        Self {
            changed: edit.changed,
            mode: edit.mode,
            solved,
            entry,
        }
    }
}

/// Re-solve a node-level problem after an edit, from the facts of the solve
/// before it.
///
/// `previous` holds the facts of a full or resumed solve of the same problem
/// before the edit; they are reused in place. Slots for nodes the edit
/// added start at bottom, slots past the current bound are dropped, and a
/// removed node named in `edit.changed` gets bottom. [`ResumeMode::Reset`]
/// resets and re-solves the nodes that the live changed nodes reach along
/// the analysis direction; [`ResumeMode::Grow`] re-solves from the changed
/// nodes without a reset. Either way only nodes in the domain of
/// `solve.seeds` run (see the [module documentation](self)). The cost is one
/// pass over the node ids, as a seeded solve makes, the walk of the region,
/// and the solver work inside it. The facts outside the region are those of
/// the previous solve, so in
/// [`IterationOrder::WeakTopological`](crate::IterationOrder) with widening
/// they are the previous solve's widened facts.
///
/// # Panics
///
/// Panics when a changed node is out of range, or a seed of `solve` is.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error or
/// [`TrySolveError::Solver`] when work remains at the configured limit.
///
/// # Examples
///
/// ```
/// use cfglib::{
///     Direction, Graph, NodeId, NodeProblem, Resume, ResumeMode, Solve, resume_node_problem,
///     solve_node_problem,
/// };
///
/// // Forward taint from the nodes in `sources`.
/// struct Taint(Vec<NodeId>);
///
/// impl NodeProblem<Graph<(), ()>> for Taint {
///     type Fact = bool;
///
///     fn direction(&self) -> Direction {
///         Direction::Forward
///     }
///     fn bottom(&self, _: &Graph<(), ()>) -> bool {
///         false
///     }
///     fn boundary(&self, _: &Graph<(), ()>) -> bool {
///         false
///     }
///     fn meet(&self, a: &bool, b: &bool) -> bool {
///         *a || *b
///     }
///     fn transfer(&self, _: &Graph<(), ()>, node: NodeId, input: &bool) -> bool {
///         *input || self.0.contains(&node)
///     }
/// }
///
/// let mut graph = Graph::<(), ()>::new();
/// let [a, b, c, d] = [(); 4].map(|()| graph.add_node(()));
/// graph.add_edge(a, b, ());
/// graph.add_edge(b, c, ());
/// let before = solve_node_problem(&graph, &Taint(vec![a]), Solve::default()).unwrap();
///
/// // The edit adds `c -> d`: only `d`'s upstream edges changed.
/// graph.add_edge(c, d, ());
/// let edit = Resume { changed: &[d], mode: ResumeMode::Reset };
/// let after = resume_node_problem(&graph, &Taint(vec![a]), before, edit, Solve::default())
///     .unwrap();
/// assert!(*after.fact_out(d));
/// assert_eq!(after.stats().step_count(), 1);
/// ```
pub fn resume_node_problem<G, P>(
    graph: &G,
    problem: &P,
    previous: Facts<P::Fact>,
    edit: Resume<'_, G::NodeId>,
    solve: Solve<'_, G::NodeId>,
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
{
    let direction = problem.direction();
    let edit = Edit::new(graph, edit, solve, direction);
    by_axis!(
        direction,
        resume_nodes_along(graph, problem, previous, &edit, solve.config)
    )
}

/// Re-solve an edge-sensitive problem after an edit, from the node and edge
/// facts of the solve before it.
///
/// As [`resume_node_problem`]; in [`ResumeMode::Reset`] the edges leaving a
/// reset node along the analysis direction are reset too. An edge the edit
/// added starts at bottom, and the node that writes it — its source
/// forward, its target backward — is re-solved so the edge gets its fact.
/// The edge slots follow the edited view: a removed edge's slot is empty, as
/// in a full solve, which costs one pass over the edge ids.
///
/// # Panics
///
/// Panics when a changed node is out of range, or a seed of `solve` is.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error or
/// [`TrySolveError::Solver`] when work remains at the configured limit.
pub fn resume_edge_problem<G, P>(
    graph: &G,
    problem: &P,
    previous: EdgeFacts<P::Fact>,
    edit: Resume<'_, G::NodeId>,
    solve: Solve<'_, G::NodeId>,
) -> Result<EdgeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
{
    let direction = problem.direction();
    let edit = Edit::new(graph, edit, solve, direction);
    by_axis!(
        direction,
        resume_edges_along(graph, problem, previous, &edit, solve.config)
    )
}

/// Sizes a previous solve's facts to the edited graph's bound: added slots
/// start at `bottom`, slots past the bound go.
fn fit<F: Clone>(facts: &mut Vec<F>, bound: usize, bottom: &F) {
    facts.resize(bound, bottom.clone());
}

/// The nodes to re-solve: the live changed nodes, and in
/// [`ResumeMode::Reset`] everything they reach along `downstream`. Calls
/// `reset` on each node a reset clears, with whether it is live: the
/// region's nodes and the removed changed nodes.
fn region<G: GraphView, D: Adjacency>(
    downstream: D,
    graph: &G,
    edit: &Edit<'_, G::NodeId>,
    mut reset: impl FnMut(G::NodeId, bool),
) -> Worklist {
    let bound = graph.node_bound();
    let live = Worklist::live(graph);
    let mut changed = Vec::with_capacity(edit.changed.len());
    for &node in edit.changed {
        assert!(node.index() < bound, "changed node is out of range");
        if live.contains(node.index()) {
            changed.push(node);
        } else {
            reset(node, false);
        }
    }
    let solved = |index: usize| edit.solved.as_ref().is_none_or(|solved| solved[index]);
    let mut worklist = Worklist::new(bound);
    match edit.mode {
        ResumeMode::Reset => {
            // The walk resets each node as it reaches it, so the cost is the
            // region's. Through `reachable_in`, whose byte per node then
            // needs a pass over the whole bound to find the region, resuming
            // the 100-node region at the end of a 100 000-node chain
            // (`api_resume_node_problem`) took 1.22x main's time; with this
            // walk and `Worklist::of` filling its row directly, 0.97x.
            let mut reached = DenseBits::new(bound);
            let mut stack = changed;
            while let Some(node) = stack.pop() {
                let index = node.index();
                if reached.insert(index) {
                    reset(node, true);
                    if solved(index) {
                        worklist.insert(index);
                    }
                    stack.extend(downstream.neighbors(graph, node));
                }
            }
        }
        ResumeMode::Grow => {
            for node in changed {
                if solved(node.index()) {
                    worklist.insert(node.index());
                }
            }
        }
    }
    worklist
}

fn resume_nodes_along<G, P, D>(
    downstream: D,
    graph: &G,
    problem: &P,
    previous: Facts<P::Fact>,
    edit: &Edit<'_, G::NodeId>,
    config: SolveConfig,
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
    D: Adjacency,
{
    let direction = problem.direction();
    let bottom = problem.bottom(graph);
    let (input, output) = previous.into_parts();
    let (mut input, mut output) = direction.program_order(input, output);
    fit(&mut input, graph.node_bound(), &bottom);
    fit(&mut output, graph.node_bound(), &bottom);
    let worklist = region(downstream, graph, edit, |node, _| {
        input[node.index()] = bottom.clone();
        output[node.index()] = bottom.clone();
    });
    let facts = || node_fixpoint::Dense { input, output };
    node_fixpoint::solve_dense(
        downstream, graph, problem, worklist, edit.entry, config, facts,
    )
}

fn resume_edges_along<G, P, D>(
    downstream: D,
    graph: &G,
    problem: &P,
    previous: EdgeFacts<P::Fact>,
    edit: &Edit<'_, G::NodeId>,
    config: SolveConfig,
) -> Result<EdgeFacts<P::Fact>, TrySolveError<P::Error>>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
    D: EdgeAdjacency,
    D::Against: EdgeAdjacency,
{
    let bottom = problem.bottom(graph);
    let (mut input, mut output, mut previous_edges) = previous.into_parts();
    fit(&mut input, graph.node_bound(), &bottom);
    fit(&mut output, graph.node_bound(), &bottom);
    // An edge the edit added has no previous fact: it starts at bottom, and
    // the node that writes it along the analysis direction runs again.
    let mut edge = vec![None; graph.edge_bound()];
    let mut writers = Vec::new();
    for id in graph.edge_ids() {
        let held = previous_edges.get_mut(id.index()).and_then(Option::take);
        if held.is_none() {
            writers.push(downstream.previous(graph.edge(id)));
        }
        edge[id.index()] = Some(held.unwrap_or_else(|| bottom.clone()));
    }
    let mut worklist = region(downstream, graph, edit, |node, live| {
        input[node.index()] = bottom.clone();
        output[node.index()] = bottom.clone();
        // A removed node's edges went with it and have no slot left.
        if live {
            for id in downstream.edges(graph, node) {
                edge[id.index()] = Some(bottom.clone());
            }
        }
    });
    let solved = |index: usize| edit.solved.as_ref().is_none_or(|solved| solved[index]);
    for writer in writers {
        if solved(writer.index()) {
            worklist.insert(writer.index());
        }
    }
    let facts = || edge_fixpoint::Dense {
        bottom,
        input,
        output,
        edge,
    };
    edge_fixpoint::solve_dense(
        downstream, graph, problem, worklist, edit.entry, config, facts,
    )
}

#[cfg(test)]
mod tests;
