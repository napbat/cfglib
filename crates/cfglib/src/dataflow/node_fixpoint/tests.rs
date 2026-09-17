extern crate alloc;

use super::*;
use crate::graph::store::{Graph, NodeId};

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

impl<E> NodeProblem<Graph<&'static str, E>> for Taint {
    type Fact = bool;

    fn direction(&self) -> Direction {
        self.direction
    }

    fn bottom(&self, _graph: &Graph<&'static str, E>) -> bool {
        false
    }

    fn boundary(&self, _graph: &Graph<&'static str, E>) -> bool {
        false
    }

    fn meet(&self, a: &bool, b: &bool) -> bool {
        *a || *b
    }

    fn transfer(&self, _graph: &Graph<&'static str, E>, node: NodeId, input: &bool) -> bool {
        *input || self.sources.contains(&node)
    }
}

/// `source -> a <-> b`, plus `clean -> b` and a disconnected `island`.
fn flow_fixture() -> (Graph<&'static str, ()>, [NodeId; 5]) {
    let mut graph: Graph<&'static str, ()> = Graph::new();
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

    let facts = solve_node_problem(&graph, &Taint::forward(alloc::vec![source])).unwrap();
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
        let full = solve_node_problem(&graph, &problem).unwrap();
        // Seed order only reorders the worklist, never the fixpoint.
        for seeds in [&all, &reversed] {
            let seeded = solve_node_problem_from(&graph, &problem, seeds).unwrap();
            for node in graph.node_ids() {
                assert_eq!(seeded.fact_in(node), full.fact_in(node));
                assert_eq!(seeded.fact_out(node), full.fact_out(node));
            }
        }
        // Duplicated seeds are queued once and change nothing.
        let doubled: alloc::vec::Vec<NodeId> = all.iter().chain(all.iter()).copied().collect();
        let seeded = solve_node_problem_from(&graph, &problem, &doubled).unwrap();
        for node in graph.node_ids() {
            assert_eq!(seeded.fact_out(node), full.fact_out(node));
        }
    }
}

#[test]
fn no_seeds_leaves_every_fact_at_bottom() {
    let (graph, [source, ..]) = flow_fixture();
    let facts = solve_node_problem_from(&graph, &Taint::forward(alloc::vec![source]), &[]).unwrap();
    assert_eq!(facts.steps(), 0);
    for node in graph.node_ids() {
        assert!(!*facts.fact_in(node));
        assert!(!*facts.fact_out(node), "not even the source is visited");
    }

    // An empty graph has nothing to seed and nothing to solve.
    let empty: Graph<&'static str, ()> = Graph::new();
    let facts = solve_node_problem_from(&empty, &Taint::forward(alloc::vec![]), &[]).unwrap();
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
    )
    .unwrap();
    assert!(*facts.fact_out(source));
    assert!(*facts.fact_out(a), "carried along the seed's out-edges");
    assert!(*facts.fact_out(b), "and around the cycle");
    assert!(!*facts.fact_out(clean), "never queued, never transferred");
    assert!(!*facts.fact_out(island), "unreachable from the seed");

    // Seeding the middle of the graph propagates forward from there only.
    let facts = solve_node_problem_from(&graph, &Taint::forward(alloc::vec![b]), &[b]).unwrap();
    assert!(*facts.fact_out(b));
    assert!(*facts.fact_out(a), "b's successor");
    assert!(!*facts.fact_out(source), "upstream of the seed");
}

#[test]
fn a_backward_seeded_solve_walks_the_out_edges() {
    let (graph, [source, a, b, _, _]) = flow_fixture();
    // Backward, facts travel from `b` to its predecessors' inputs.
    let facts = solve_node_problem_from(&graph, &Taint::backward(alloc::vec![b]), &[b]).unwrap();
    assert!(*facts.fact_out(b));
    assert!(*facts.fact_out(a), "a's successor b is tainted");
    assert!(*facts.fact_out(source), "and so on upstream");
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn an_out_of_range_seed_panics() {
    let (graph, _) = flow_fixture();
    let beyond = NodeId::from_index(graph.node_bound());
    let _ = solve_node_problem_from(&graph, &Taint::forward(alloc::vec![]), &[beyond]);
}

#[test]
fn step_limit_reports_the_pending_node() {
    let (graph, [source, ..]) = flow_fixture();
    let error = solve_node_problem_with_config(
        &graph,
        &Taint::forward(alloc::vec![source]),
        SolveConfig::with_step_limit(1),
    )
    .unwrap_err();
    let SolveError::StepLimitExceeded { limit, steps, .. } = error;
    assert_eq!(limit, 1);
    assert_eq!(steps, 1);
}

/// A taint clone that rejects a designated forbidden node.
struct Guarded {
    sources: alloc::vec::Vec<NodeId>,
    forbidden: NodeId,
}

impl<E> TryNodeProblem<Graph<&'static str, E>> for Guarded {
    type Fact = bool;
    type Error = &'static str;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &Graph<&'static str, E>) -> bool {
        false
    }

    fn boundary(&self, _graph: &Graph<&'static str, E>) -> Result<bool, Self::Error> {
        Ok(false)
    }

    fn meet(&self, a: &bool, b: &bool) -> Result<bool, Self::Error> {
        Ok(*a || *b)
    }

    fn transfer(
        &self,
        _graph: &Graph<&'static str, E>,
        node: NodeId,
        input: &bool,
    ) -> Result<bool, Self::Error> {
        if node == self.forbidden {
            return Err("forbidden node visited");
        }
        Ok(*input || self.sources.contains(&node))
    }
}

#[test]
fn a_consumer_error_is_reported_as_a_problem_error() {
    let (graph, [source, a, ..]) = flow_fixture();
    let error = try_solve_node_problem(
        &graph,
        &Guarded {
            sources: alloc::vec![source],
            forbidden: a,
        },
    )
    .unwrap_err();
    assert_eq!(error, TrySolveError::Problem("forbidden node visited"));
}

#[test]
fn a_fallible_seeded_solve_avoids_unreached_nodes() {
    let (graph, [source, a, b, clean, island]) = flow_fixture();
    // The forbidden node is never reached from the seed, so the solve
    // completes.
    let facts = try_solve_node_problem_from(
        &graph,
        &Guarded {
            sources: alloc::vec![source],
            forbidden: island,
        },
        &[source],
    )
    .unwrap();
    assert!(*facts.fact_out(a));
    assert!(*facts.fact_out(b));
    assert!(!*facts.fact_out(clean));
}
