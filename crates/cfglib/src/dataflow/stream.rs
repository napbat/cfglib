//! Streamed solves: each node's facts go to the caller as soon as they are
//! final, and are freed as soon as nothing reads them any more.
//!
//! A solve that returns [`Facts`](super::fixpoint::Facts) holds an IN and an
//! OUT fact for every node until it returns, and an edge solve one more fact
//! per edge. A consumer that only checks a property at each node, reports a
//! diagnostic, or copies a projection into its own sparse table does not need
//! them all at once. [`stream_problem`], [`stream_node_problem`] and
//! [`stream_edge_problem`] run the same solve in
//! [`IterationOrder::WeakTopological`] and hand each node to a visitor once,
//! as soon as the solve has left the outermost element of the order that
//! holds it: nothing outside an element can queue it again, so its facts are
//! final (S. K. Kim, A. J. Venet, A. V. Thakur, *Memory-efficient fixpoint
//! computation*, FMSD 65, 2025).
//!
//! The node's IN fact is freed right after the visit. Its OUT fact is freed
//! once the outermost element of its last reader is final, its own included,
//! because the worklist solver reads a node's OUT again whenever it revisits a
//! downstream node. An edge solve frees a node's facts at its visit, since
//! only edges carry facts to other nodes, and an edge's fact once its target
//! has been visited. Peak memory is then a matter of the widest outermost
//! loops and of how far facts travel past them, not of the graph's size.
//!
//! Prefer a stream when facts are large (bit rows over the node set) and the
//! consumer keeps only a projection of them. A stream returns no facts; a
//! consumer that needs them all keeps using the solvers.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::edge_fixpoint::{self, EdgeStore, TryEdgeProblem};
use super::fixpoint::{self, BlockProblem, IterationOrder, SolveConfig, TryProblem, TrySolveError};
use super::node_fixpoint::{self, NodeStore, TryNodeProblem};
use super::worklist::Worklist;
use crate::graph::edge_traverse::EdgeAdjacency;
use crate::graph::edge_view::EdgeView;
use crate::graph::traverse::{Adjacency, by_axis};
use crate::graph::view::Direction;
use crate::graph::view::{DenseId, GraphView, RootedView};
use crate::graph::weak_topological_order::WeakTopologicalOrder;
use crate::identity::word;

/// What a streamed solve did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveStats {
    steps: usize,
    peak_live_facts: usize,
}

impl SolveStats {
    /// Worklist entries processed, narrowing passes included: the
    /// [`Facts::steps`](super::fixpoint::Facts::steps) of the same solve in
    /// [`IterationOrder::WeakTopological`] that keeps every fact.
    #[must_use]
    pub const fn steps(self) -> usize {
        self.steps
    }

    /// The most facts the solve held at once: IN and OUT facts of nodes, and
    /// facts on edges for an edge problem. A fact that still equals bottom
    /// because nothing reached it is not held. A solve that returns every
    /// fact holds two per node, and one per edge, from start to end.
    #[must_use]
    pub const fn peak_live_facts(self) -> usize {
        self.peak_live_facts
    }
}

/// Solve a fallible block-level problem and stream each block's facts to
/// `visit` once they are final.
///
/// `visit` receives every block the solve reaches exactly once — every block
/// the root reaches forward, every block backward — with its IN and OUT
/// facts in program order, as [`Facts`](super::fixpoint::Facts) reports
/// them. The solve runs in [`IterationOrder::WeakTopological`] whatever order
/// `config` names, because that order decides when a fact is final; the step
/// limit and the narrowing passes of `config` apply. An infallible problem
/// implements [`TryProblem`] with `Error = Infallible`.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error or
/// [`TrySolveError::Solver`] when work remains at the configured limit. The
/// blocks visited before the error keep their final facts.
pub fn stream_problem<V, P, F>(
    view: &V,
    problem: &P,
    config: SolveConfig,
    visit: F,
) -> Result<SolveStats, TrySolveError<P::Error>>
where
    V: RootedView,
    P: TryProblem<V>,
    F: FnMut(V::NodeId, &P::Fact, &P::Fact),
{
    let direction = problem.direction();
    let worklist = fixpoint::initial_worklist(view, direction);
    let entry = matches!(direction, Direction::Forward).then(|| view.root());
    stream_nodes(view, &BlockProblem(problem), worklist, entry, config, visit)
}

/// Solve a fallible node-level problem and stream each node's facts to
/// `visit` once they are final.
///
/// `visit` receives every live node exactly once, with its IN and OUT facts
/// in program order. The solve runs in [`IterationOrder::WeakTopological`]
/// whatever order `config` names, because that order decides when a fact is
/// final; the step limit and the narrowing passes of `config` apply. An
/// infallible problem implements [`TryNodeProblem`] with
/// `Error = Infallible`.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error or
/// [`TrySolveError::Solver`] when work remains at the configured limit. The
/// nodes visited before the error keep their final facts.
///
/// # Examples
///
/// Report the tainted nodes of a long chain of loops while holding only the
/// facts of the loop being solved and of the edge into the next one.
///
/// ```
/// use core::convert::Infallible;
///
/// use cfglib::{Direction, Graph, NodeId, SolveConfig, TryNodeProblem, stream_node_problem};
///
/// struct Taint(NodeId);
///
/// impl TryNodeProblem<Graph<(), ()>> for Taint {
///     type Fact = bool;
///     type Error = Infallible;
///
///     fn direction(&self) -> Direction {
///         Direction::Forward
///     }
///     fn bottom(&self, _: &Graph<(), ()>) -> bool {
///         false
///     }
///     fn boundary(&self, _: &Graph<(), ()>) -> Result<bool, Infallible> {
///         Ok(false)
///     }
///     fn meet(&self, _: &Graph<(), ()>, _: NodeId, a: &bool, b: &bool) -> Result<bool, Infallible> {
///         Ok(*a || *b)
///     }
///     fn transfer(&self, _: &Graph<(), ()>, node: NodeId, input: &bool) -> Result<bool, Infallible> {
///         Ok(*input || node == self.0)
///     }
/// }
///
/// // head -> latch -> head, then on to the next loop's head.
/// let mut graph = Graph::<(), ()>::new();
/// let heads: Vec<NodeId> = (0..1_000)
///     .map(|_| {
///         let head = graph.add_node(());
///         let latch = graph.add_node(());
///         graph.add_edge(head, latch, ());
///         graph.add_edge(latch, head, ());
///         head
///     })
///     .collect();
/// for pair in heads.windows(2) {
///     graph.add_edge(pair[0], pair[1], ());
/// }
///
/// let mut tainted = 0;
/// let stats = stream_node_problem(&graph, &Taint(heads[500]), SolveConfig::new(), |_, _, out| {
///     tainted += usize::from(*out);
/// })
/// .unwrap();
/// assert_eq!(tainted, 1_000);
/// assert!(stats.peak_live_facts() < 10);
/// ```
pub fn stream_node_problem<G, P, F>(
    graph: &G,
    problem: &P,
    config: SolveConfig,
    visit: F,
) -> Result<SolveStats, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
    F: FnMut(G::NodeId, &P::Fact, &P::Fact),
{
    stream_nodes(graph, problem, Worklist::live(graph), None, config, visit)
}

/// Solve a fallible edge-sensitive problem and stream each node's facts to
/// `visit` once they are final.
///
/// `visit` receives every live node exactly once, with its physical input
/// and output facts. The fact on an edge is the problem's
/// [`transfer_edge`](TryEdgeProblem::transfer_edge) of its source's facts,
/// so a visitor that needs it computes it from what it receives at the
/// source. The solve runs in [`IterationOrder::WeakTopological`] whatever
/// order `config` names; the step limit and the narrowing passes of `config`
/// apply.
///
/// # Errors
///
/// Returns [`TrySolveError::Problem`] for a consumer error or
/// [`TrySolveError::Solver`] when work remains at the configured limit. The
/// nodes visited before the error keep their final facts.
pub fn stream_edge_problem<G, P, F>(
    graph: &G,
    problem: &P,
    config: SolveConfig,
    visit: F,
) -> Result<SolveStats, TrySolveError<P::Error>>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
    F: FnMut(G::NodeId, &P::Fact, &P::Fact),
{
    by_axis!(
        problem.direction(),
        stream_edges_along(graph, problem, config, visit)
    )
}

fn stream_nodes<G, P, F>(
    graph: &G,
    problem: &P,
    worklist: Worklist,
    entry: Option<G::NodeId>,
    config: SolveConfig,
    visit: F,
) -> Result<SolveStats, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
    F: FnMut(G::NodeId, &P::Fact, &P::Fact),
{
    by_axis!(
        problem.direction(),
        stream_nodes_along(graph, problem, worklist, entry, config, visit)
    )
}

fn stream_nodes_along<G, P, D, F>(
    downstream: D,
    graph: &G,
    problem: &P,
    worklist: Worklist,
    entry: Option<G::NodeId>,
    config: SolveConfig,
    visit: F,
) -> Result<SolveStats, TrySolveError<P::Error>>
where
    G: GraphView,
    P: TryNodeProblem<G>,
    D: Adjacency,
    F: FnMut(G::NodeId, &P::Fact, &P::Fact),
{
    let config = config.with_order(IterationOrder::WeakTopological);
    let store = |order: Option<&WeakTopologicalOrder<G::NodeId>>| {
        let order = order.expect("a stream runs in the weak topological order");
        let direction = problem.direction();
        StreamedNodes::new(
            graph,
            order,
            downstream,
            problem.bottom(graph),
            direction,
            visit,
        )
    };
    let (store, steps) =
        node_fixpoint::solve_along(downstream, graph, problem, worklist, entry, config, store)?;
    Ok(SolveStats {
        steps,
        peak_live_facts: store.census.peak,
    })
}

fn stream_edges_along<G, P, D, F>(
    downstream: D,
    graph: &G,
    problem: &P,
    config: SolveConfig,
    visit: F,
) -> Result<SolveStats, TrySolveError<P::Error>>
where
    G: EdgeView,
    P: TryEdgeProblem<G>,
    D: EdgeAdjacency,
    D::Against: EdgeAdjacency,
    F: FnMut(G::NodeId, &P::Fact, &P::Fact),
{
    let config = config.with_order(IterationOrder::WeakTopological);
    let store = |_: Option<&WeakTopologicalOrder<G::NodeId>>| StreamedEdges {
        graph,
        upstream: downstream.against(),
        bottom: problem.bottom(graph),
        input: vec![None; graph.node_bound()],
        output: vec![None; graph.node_bound()],
        edge: vec![None; graph.edge_bound()],
        finished: 0,
        census: Census::default(),
        visit,
    };
    let (store, steps) =
        edge_fixpoint::solve_edges(graph, problem, Worklist::live(graph), config, store)?;
    Ok(SolveStats {
        steps,
        peak_live_facts: store.census.peak,
    })
}

/// How many facts a stream holds, and the most it held.
#[derive(Debug, Default)]
struct Census {
    live: usize,
    peak: usize,
}

impl Census {
    /// Stores `fact` in `slot`, counting a slot that held none.
    fn put<F>(&mut self, slot: &mut Option<F>, fact: F) {
        if slot.is_none() {
            self.live += 1;
            self.peak = self.peak.max(self.live);
        }
        *slot = Some(fact);
    }

    /// Frees what `slot` holds.
    fn free<F>(&mut self, slot: &mut Option<F>) {
        if slot.take().is_some() {
            self.live -= 1;
        }
    }
}

/// The facts a node stream still needs, in flow order; an empty slot is
/// bottom until written, and freed once nothing reads it.
struct StreamedNodes<F, V> {
    bottom: F,
    direction: Direction,
    input: Vec<Option<F>>,
    output: Vec<Option<F>>,
    /// Release rows by position: row `end` lists the dense indices of the
    /// nodes whose OUT fact is last read inside the outermost element that
    /// ends at `end`, between `release_start[end]` and
    /// `release_start[end + 1]`.
    release_start: Vec<u32>,
    release: Vec<u32>,
    /// Every position before this one has been visited and its rows released.
    finished: usize,
    census: Census,
    visit: V,
}

impl<F: Clone, V> StreamedNodes<F, V> {
    /// Plans when each node's OUT fact can go: once the outermost element of
    /// its last reader along `downstream`, or its own, is final.
    fn new<G: GraphView, D: Adjacency>(
        graph: &G,
        order: &WeakTopologicalOrder<G::NodeId>,
        downstream: D,
        bottom: F,
        direction: Direction,
        visit: V,
    ) -> Self {
        let ends = order.outermost_ends();
        let position = |node: G::NodeId| {
            let rank = order.rank(node);
            rank.expect("a stream orders everything its nodes reach") as usize
        };
        let count = order.order().len();
        let mut last_read = Vec::with_capacity(count);
        let mut release_start = vec![0_u32; count + 2];
        for (at, &node) in order.order().iter().enumerate() {
            let mut end = ends[at];
            for next in downstream.neighbors(graph, node) {
                end = end.max(ends[position(next)]);
            }
            last_read.push(end);
            release_start[end as usize + 1] += 1;
        }
        for row in 1..release_start.len() {
            release_start[row] += release_start[row - 1];
        }
        let mut cursor = release_start.clone();
        let mut release = vec![0_u32; count];
        for (&node, &end) in order.order().iter().zip(&last_read) {
            let slot = &mut cursor[end as usize];
            release[*slot as usize] = word(node.index());
            *slot += 1;
        }
        Self {
            bottom,
            direction,
            input: vec![None; graph.node_bound()],
            output: vec![None; graph.node_bound()],
            release_start,
            release,
            finished: 0,
            census: Census::default(),
            visit,
        }
    }
}

impl<N, F, V> NodeStore<N, F> for StreamedNodes<F, V>
where
    N: DenseId,
    V: FnMut(N, &F, &F),
{
    const FINISHES: bool = true;

    fn input(&self, index: usize) -> &F {
        self.input[index].as_ref().unwrap_or(&self.bottom)
    }

    fn output(&self, index: usize) -> &F {
        self.output[index].as_ref().unwrap_or(&self.bottom)
    }

    fn set_input(&mut self, index: usize, fact: F) {
        self.census.put(&mut self.input[index], fact);
    }

    fn set_output(&mut self, index: usize, fact: F) {
        self.census.put(&mut self.output[index], fact);
    }

    fn finish(&mut self, order: &WeakTopologicalOrder<N>, end: usize) {
        for &node in order.order().get(self.finished..end).unwrap_or_default() {
            let index = node.index();
            let flowing_in = self.input[index].take();
            if flowing_in.is_some() {
                self.census.live -= 1;
            }
            let flowing_in = flowing_in.as_ref().unwrap_or(&self.bottom);
            let flowing_out = self.output[index].as_ref().unwrap_or(&self.bottom);
            let (fact_in, fact_out) = self.direction.program_order(flowing_in, flowing_out);
            (self.visit)(node, fact_in, fact_out);
        }
        // An element ends at a position after its last one, so the rows to
        // release are the ends after what was finished, up to `end`.
        for row in (self.finished + 1)..=end {
            let released = self.release_start[row] as usize..self.release_start[row + 1] as usize;
            for &index in &self.release[released] {
                self.census.free(&mut self.output[index as usize]);
            }
        }
        self.finished = self.finished.max(end);
    }
}

/// The facts an edge stream still needs, nodes in program order; an empty
/// slot is bottom until written, and freed once nothing reads it.
struct StreamedEdges<'g, G, U, F, V> {
    graph: &'g G,
    upstream: U,
    bottom: F,
    input: Vec<Option<F>>,
    output: Vec<Option<F>>,
    edge: Vec<Option<F>>,
    /// Every position before this one has been visited.
    finished: usize,
    census: Census,
    visit: V,
}

impl<G, U, F, V> EdgeStore<G::NodeId, F> for StreamedEdges<'_, G, U, F, V>
where
    G: EdgeView,
    U: EdgeAdjacency,
    F: PartialEq,
    V: FnMut(G::NodeId, &F, &F),
{
    const FINISHES: bool = true;

    fn bottom(&self) -> &F {
        &self.bottom
    }

    fn input(&self, index: usize) -> &F {
        self.input[index].as_ref().unwrap_or(&self.bottom)
    }

    fn output(&self, index: usize) -> &F {
        self.output[index].as_ref().unwrap_or(&self.bottom)
    }

    fn set_node(&mut self, index: usize, input: F, output: F) {
        self.census.put(&mut self.input[index], input);
        self.census.put(&mut self.output[index], output);
    }

    fn edge(&self, index: usize) -> &F {
        self.edge[index].as_ref().unwrap_or(&self.bottom)
    }

    fn update_edge(&mut self, index: usize, fact: F) -> bool {
        let changed = fact != *self.edge(index);
        if changed {
            self.census.put(&mut self.edge[index], fact);
        }
        changed
    }

    fn finish(&mut self, order: &WeakTopologicalOrder<G::NodeId>, end: usize) {
        for &node in order.order().get(self.finished..end).unwrap_or_default() {
            let index = node.index();
            (self.visit)(
                node,
                self.input[index].as_ref().unwrap_or(&self.bottom),
                self.output[index].as_ref().unwrap_or(&self.bottom),
            );
            self.census.free(&mut self.input[index]);
            self.census.free(&mut self.output[index]);
            for edge in self.upstream.edges(self.graph, node) {
                self.census.free(&mut self.edge[edge.index()]);
            }
        }
        self.finished = self.finished.max(end);
    }
}

#[cfg(test)]
mod tests;
