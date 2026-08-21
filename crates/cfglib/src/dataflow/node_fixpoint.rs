//! Node-level fixpoint dataflow over any [`DirectedGraphView`].
//!
//! The instruction fixpoint in [`fixpoint`](super::fixpoint) is bound to
//! [`Cfg`](crate::Cfg) blocks; this is its graph-shaped counterpart: one
//! fact per node, meet over the in-edges (out-edges backward), transfer
//! per node. It serves analyses over non-CFG graphs — taint or
//! reachability-with-facts over a value-flow graph, closure over an import
//! or include graph — anywhere the graph is the program representation.
//!
//! [`solve_node_problem`] queues every node; [`solve_node_problem_from`]
//! queues a chosen subset, so an incremental or dirty-region analysis pays
//! for the part of the graph its change actually reaches.

extern crate alloc;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

use super::fixpoint::Direction;
use crate::graph::traverse::{Adjacency, Incoming, Outgoing};
use crate::graph::view::{DenseNodeId, DirectedGraphView};

/// A node-level dataflow problem over a graph view `G`.
///
/// Termination requires the usual contract: `meet` and `transfer` monotone
/// over a finite-height fact lattice.
pub trait NodeProblem<G: DirectedGraphView> {
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
}

/// The solved per-node facts of a [`NodeProblem`].
#[derive(Debug, Clone)]
pub struct NodeFacts<F> {
    input: Vec<F>,
    output: Vec<F>,
}

impl<F> NodeFacts<F> {
    /// The met fact entering `node`.
    #[must_use]
    pub fn fact_in<N: DenseNodeId>(&self, node: N) -> &F {
        &self.input[node.index()]
    }

    /// The transferred fact leaving `node`.
    #[must_use]
    pub fn fact_out<N: DenseNodeId>(&self, node: N) -> &F {
        &self.output[node.index()]
    }
}

/// Solve a [`NodeProblem`] to fixpoint with a worklist.
#[must_use]
pub fn solve_node_problem<G, P>(graph: &G, problem: &P) -> NodeFacts<P::Fact>
where
    G: DirectedGraphView,
    P: NodeProblem<G>,
{
    let queued = vec![true; graph.node_count()];
    let worklist: VecDeque<G::NodeId> = graph.node_ids().collect();
    solve_seeded(graph, problem, worklist, queued)
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
/// Seeding every node is exactly [`solve_node_problem`], including the
/// initial worklist order; duplicate seeds are queued once. With no seeds
/// nothing is visited and every fact stays `bottom` — an empty change set
/// costs an allocation, not a traversal.
///
/// The facts of unvisited nodes are `bottom`, not stale values from a
/// previous solve: this solves a fresh problem from a subset of entry
/// points, it does not resume one. A consumer holding previous results
/// merges them itself, which is why the seeded solve keeps the same
/// `NodeFacts` shape.
///
/// # Examples
///
/// ```
/// use cfglib::{DirectedGraph, Direction, NodeId, NodeProblem, solve_node_problem_from};
///
/// // Forward taint over a value-flow graph.
/// struct Taint(Vec<NodeId>);
/// impl NodeProblem<DirectedGraph<&'static str, ()>> for Taint {
///     type Fact = bool;
///     fn direction(&self) -> Direction {
///         Direction::Forward
///     }
///     fn bottom(&self, _: &DirectedGraph<&'static str, ()>) -> bool {
///         false
///     }
///     fn boundary(&self, _: &DirectedGraph<&'static str, ()>) -> bool {
///         false
///     }
///     fn meet(&self, a: &bool, b: &bool) -> bool {
///         *a || *b
///     }
///     fn transfer(
///         &self,
///         _: &DirectedGraph<&'static str, ()>,
///         node: NodeId,
///         input: &bool,
///     ) -> bool {
///         *input || self.0.contains(&node)
///     }
/// }
///
/// let mut graph = DirectedGraph::<&'static str, ()>::new();
/// let edited = graph.add_node("edited");
/// let downstream = graph.add_node("downstream");
/// let untouched = graph.add_node("untouched");
/// graph.add_edge(edited, downstream, ());
///
/// // Re-solving from the node that changed still reaches everything its
/// // facts flow to, and never visits the rest of the graph.
/// let facts = solve_node_problem_from(&graph, &Taint(vec![edited]), &[edited]);
/// assert!(*facts.fact_out(downstream));
/// assert!(!*facts.fact_out(untouched));
/// ```
///
/// # Panics
///
/// Panics when a seed is not a node in `graph`.
#[must_use]
pub fn solve_node_problem_from<G, P>(
    graph: &G,
    problem: &P,
    seeds: &[G::NodeId],
) -> NodeFacts<P::Fact>
where
    G: DirectedGraphView,
    P: NodeProblem<G>,
{
    let mut queued = vec![false; graph.node_count()];
    let mut worklist = VecDeque::with_capacity(seeds.len());
    for &seed in seeds {
        assert!(
            seed.index() < graph.node_count(),
            "seed node is out of range"
        );
        if !queued[seed.index()] {
            queued[seed.index()] = true;
            worklist.push_back(seed);
        }
    }
    solve_seeded(graph, problem, worklist, queued)
}

/// The worklist solver both entry points run, differing only in which nodes
/// start queued.
fn solve_seeded<G, P>(
    graph: &G,
    problem: &P,
    worklist: VecDeque<G::NodeId>,
    queued: Vec<bool>,
) -> NodeFacts<P::Fact>
where
    G: DirectedGraphView,
    P: NodeProblem<G>,
{
    let node_count = graph.node_count();
    let boundary = problem.boundary(graph);
    let input = vec![problem.bottom(graph); node_count];
    let output = vec![problem.bottom(graph); node_count];

    match problem.direction() {
        Direction::Forward => solve_seeded_by_axis(
            graph, problem, Incoming, Outgoing, worklist, queued, &boundary, input, output,
        ),
        Direction::Backward => solve_seeded_by_axis(
            graph, problem, Outgoing, Incoming, worklist, queued, &boundary, input, output,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_seeded_by_axis<G, P, U, D>(
    graph: &G,
    problem: &P,
    upstream: U,
    downstream: D,
    mut worklist: VecDeque<G::NodeId>,
    mut queued: Vec<bool>,
    boundary: &P::Fact,
    mut input: Vec<P::Fact>,
    mut output: Vec<P::Fact>,
) -> NodeFacts<P::Fact>
where
    G: DirectedGraphView,
    P: NodeProblem<G>,
    U: Adjacency,
    D: Adjacency,
{
    while let Some(node) = worklist.pop_front() {
        queued[node.index()] = false;

        let mut neighbors = upstream.neighbors(graph, node);
        let met = match neighbors.next() {
            None => boundary.clone(),
            Some(first) => match neighbors.next() {
                None => output[first.index()].clone(),
                Some(second) => {
                    let mut current = problem.meet(&output[first.index()], &output[second.index()]);
                    for from in neighbors {
                        current = problem.meet(&current, &output[from.index()]);
                    }
                    current
                }
            },
        };

        let transferred = problem.transfer(graph, node, &met);
        input[node.index()] = met;
        if transferred != output[node.index()] {
            output[node.index()] = transferred;
            for next in downstream.neighbors(graph, node) {
                if !queued[next.index()] {
                    queued[next.index()] = true;
                    worklist.push_back(next);
                }
            }
        }
    }

    NodeFacts { input, output }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::directed::{DirectedGraph, NodeId};

    /// Taint: a node is tainted when it is a source or any upstream node's
    /// output is tainted — over a value-flow-shaped graph.
    struct Taint {
        sources: alloc::vec::Vec<NodeId>,
        direction: Direction,
    }

    impl Taint {
        fn forward(sources: alloc::vec::Vec<NodeId>) -> Self {
            Self {
                sources,
                direction: Direction::Forward,
            }
        }

        fn backward(sources: alloc::vec::Vec<NodeId>) -> Self {
            Self {
                sources,
                direction: Direction::Backward,
            }
        }
    }

    impl<E> NodeProblem<DirectedGraph<&'static str, E>> for Taint {
        type Fact = bool;

        fn direction(&self) -> Direction {
            self.direction
        }

        fn bottom(&self, _graph: &DirectedGraph<&'static str, E>) -> bool {
            false
        }

        fn boundary(&self, _graph: &DirectedGraph<&'static str, E>) -> bool {
            false
        }

        fn meet(&self, a: &bool, b: &bool) -> bool {
            *a || *b
        }

        fn transfer(
            &self,
            _graph: &DirectedGraph<&'static str, E>,
            node: NodeId,
            input: &bool,
        ) -> bool {
            *input || self.sources.contains(&node)
        }
    }

    /// `source -> a <-> b`, plus `clean -> b` and a disconnected `island`.
    fn flow_fixture() -> (DirectedGraph<&'static str, ()>, [NodeId; 5]) {
        let mut graph: DirectedGraph<&'static str, ()> = DirectedGraph::new();
        let source = graph.add_node("source");
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let clean = graph.add_node("clean");
        let island = graph.add_node("island");
        graph.add_edge(source, a, ());
        graph.add_edge(a, b, ());
        graph.add_edge(b, a, ()); // cycle
        graph.add_edge(clean, b, ());
        (graph, [source, a, b, clean, island])
    }

    #[test]
    fn taint_propagates_through_cycles() {
        let (graph, [source, a, b, clean, _]) = flow_fixture();

        let facts = solve_node_problem(&graph, &Taint::forward(alloc::vec![source]));
        assert!(*facts.fact_out(source));
        assert!(*facts.fact_out(a), "reached through the source");
        assert!(*facts.fact_out(b), "reached through the cycle");
        assert!(!*facts.fact_out(clean), "no path from the source");
        assert!(*facts.fact_in(b), "met input at b includes the tainted a");
    }

    #[test]
    fn seeding_every_node_is_the_full_solve() {
        let (graph, [source, _, b, clean, island]) = flow_fixture();
        let all: alloc::vec::Vec<NodeId> = graph.node_ids().collect();
        let mut reversed = all.clone();
        reversed.reverse();

        for problem in [
            Taint::forward(alloc::vec![source]),
            Taint::forward(alloc::vec![clean, island]),
            Taint::backward(alloc::vec![b]),
            Taint::backward(alloc::vec![]),
        ] {
            let full = solve_node_problem(&graph, &problem);
            // Seed order only reorders the worklist, never the fixpoint.
            for seeds in [&all, &reversed] {
                let seeded = solve_node_problem_from(&graph, &problem, seeds);
                for node in graph.node_ids() {
                    assert_eq!(seeded.fact_in(node), full.fact_in(node));
                    assert_eq!(seeded.fact_out(node), full.fact_out(node));
                }
            }
            // Duplicated seeds are queued once and change nothing.
            let doubled: alloc::vec::Vec<NodeId> = all.iter().chain(all.iter()).copied().collect();
            let seeded = solve_node_problem_from(&graph, &problem, &doubled);
            for node in graph.node_ids() {
                assert_eq!(seeded.fact_out(node), full.fact_out(node));
            }
        }
    }

    #[test]
    fn no_seeds_leaves_every_fact_at_bottom() {
        let (graph, [source, ..]) = flow_fixture();
        let facts = solve_node_problem_from(&graph, &Taint::forward(alloc::vec![source]), &[]);
        for node in graph.node_ids() {
            assert!(!*facts.fact_in(node));
            assert!(!*facts.fact_out(node), "not even the source is visited");
        }

        // An empty graph has nothing to seed and nothing to solve.
        let empty: DirectedGraph<&'static str, ()> = DirectedGraph::new();
        let facts = solve_node_problem_from(&empty, &Taint::forward(alloc::vec![]), &[]);
        assert_eq!(empty.node_count(), 0);
        assert!(empty.node_ids().all(|node| *facts.fact_out(node)));
    }

    #[test]
    fn a_seeded_solve_propagates_only_where_transfers_carry_it() {
        let (graph, [source, a, b, clean, island]) = flow_fixture();
        // `clean` is a source too, but seeding only `source` never visits it,
        // so its own taint is never generated.
        let facts = solve_node_problem_from(
            &graph,
            &Taint::forward(alloc::vec![source, clean]),
            &[source],
        );
        assert!(*facts.fact_out(source));
        assert!(*facts.fact_out(a), "carried along the seed's out-edges");
        assert!(*facts.fact_out(b), "and around the cycle");
        assert!(!*facts.fact_out(clean), "never queued, never transferred");
        assert!(!*facts.fact_out(island), "unreachable from the seed");

        // Seeding the middle of the graph propagates forward from there only.
        let facts = solve_node_problem_from(&graph, &Taint::forward(alloc::vec![b]), &[b]);
        assert!(*facts.fact_out(b));
        assert!(*facts.fact_out(a), "b's successor");
        assert!(!*facts.fact_out(source), "upstream of the seed");
    }

    #[test]
    fn a_backward_seeded_solve_walks_the_out_edges() {
        let (graph, [source, a, b, _, _]) = flow_fixture();
        // Backward, facts travel from `b` to its predecessors' inputs.
        let facts = solve_node_problem_from(&graph, &Taint::backward(alloc::vec![b]), &[b]);
        assert!(*facts.fact_out(b));
        assert!(*facts.fact_out(a), "a's successor b is tainted");
        assert!(*facts.fact_out(source), "and so on upstream");
    }

    #[test]
    #[should_panic(expected = "seed node is out of range")]
    fn an_out_of_range_seed_panics() {
        let (graph, _) = flow_fixture();
        let beyond = NodeId::from_index(graph.node_count());
        let _ = solve_node_problem_from(&graph, &Taint::forward(alloc::vec![]), &[beyond]);
    }
}
