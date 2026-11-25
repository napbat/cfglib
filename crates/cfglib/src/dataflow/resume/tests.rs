//! Resumed solves against fresh solves of the edited graph, and the gap of a
//! seeded solve that they close.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::convert::Infallible;

use super::{ResumeMode, resume_node_problem, resume_problem};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::fixpoint::{Direction, Facts, SolveConfig, TryProblem, try_solve_problem};
use crate::dataflow::node_fixpoint::{
    TryNodeProblem, try_solve_node_problem, try_solve_node_problem_from,
};
use crate::edge::EdgeKind;
use crate::graph::store::{Graph, NodeId};

/// Taint as a bit row: each source sets its own bit, and a node's fact is
/// what flows in with its own bit added when it is a source.
struct Taint {
    sources: Vec<(NodeId, u64)>,
}

impl TryNodeProblem<Graph<(), ()>> for Taint {
    type Fact = u64;
    type Error = Infallible;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _: &Graph<(), ()>) -> u64 {
        0
    }

    fn boundary(&self, _: &Graph<(), ()>) -> Result<u64, Infallible> {
        Ok(0)
    }

    fn meet(&self, _: &Graph<(), ()>, _: NodeId, a: &u64, b: &u64) -> Result<u64, Infallible> {
        Ok(a | b)
    }

    fn transfer(&self, _: &Graph<(), ()>, node: NodeId, input: &u64) -> Result<u64, Infallible> {
        let own = self.sources.iter().filter(|(source, _)| *source == node);
        Ok(own.fold(*input, |bits, (_, bit)| bits | bit))
    }
}

fn chain(count: usize) -> (Graph<(), ()>, Vec<NodeId>) {
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..count).map(|_| graph.add_node(())).collect();
    for pair in nodes.windows(2) {
        graph.add_edge(pair[0], pair[1], ());
    }
    (graph, nodes)
}

fn assert_same(actual: &Facts<u64>, expected: &Facts<u64>, nodes: &[NodeId]) {
    for &node in nodes {
        assert_eq!(
            actual.fact_in(node),
            expected.fact_in(node),
            "IN of {node:?}"
        );
        assert_eq!(
            actual.fact_out(node),
            expected.fact_out(node),
            "OUT of {node:?}"
        );
    }
}

#[test]
fn a_resume_keeps_what_flows_in_from_nodes_it_does_not_visit() {
    // a -> b -> c with taint at a; the edit makes b a source too.
    let (graph, nodes) = chain(3);
    let [a, b, c] = [nodes[0], nodes[1], nodes[2]];
    let before = Taint {
        sources: vec![(a, 1)],
    };
    let after = Taint {
        sources: vec![(a, 1), (b, 2)],
    };
    let previous = try_solve_node_problem(&graph, &before).unwrap();

    // The seeded solve starts b and c at bottom and never visits a, so a's
    // taint is lost downstream of the edit.
    let seeded = try_solve_node_problem_from(&graph, &after, &[b]).unwrap();
    assert_eq!((*seeded.fact_out(b), *seeded.fact_out(c)), (2, 2));

    let fresh = try_solve_node_problem(&graph, &after).unwrap();
    for mode in [ResumeMode::Reset, ResumeMode::Grow] {
        let config = SolveConfig::new();
        let resumed = resume_node_problem(&graph, &after, previous.clone(), &[b], mode, config);
        let resumed = resumed.unwrap();
        assert_eq!((*resumed.fact_out(b), *resumed.fact_out(c)), (3, 3));
        assert_same(&resumed, &fresh, &nodes);
        assert_eq!(resumed.steps(), 2, "{mode:?} re-solves b and c only");
    }
}

#[test]
fn a_reset_costs_the_region_the_edit_reaches_not_the_graph() {
    // A hundred thousand nodes; the edit's region is the last hundred, with a
    // loop among them.
    let count = 100_000;
    let (mut graph, nodes) = chain(count);
    graph.add_edge(nodes[count - 10], nodes[count - 60], ());
    let first = nodes[count - 100];
    let before = Taint {
        sources: vec![(nodes[0], 1)],
    };
    let after = Taint {
        sources: vec![(nodes[0], 1), (first, 2)],
    };
    let previous = try_solve_node_problem(&graph, &before).unwrap();
    let config = SolveConfig::new();
    let resumed = resume_node_problem(
        &graph,
        &after,
        previous,
        &[first],
        ResumeMode::Reset,
        config,
    )
    .unwrap();
    assert!(resumed.steps() <= 4 * 100, "{} steps", resumed.steps());
    let fresh = try_solve_node_problem(&graph, &after).unwrap();
    assert_same(&resumed, &fresh, &nodes);
}

#[test]
fn slots_follow_the_edited_bound_and_a_removed_node_gets_bottom() {
    let (mut graph, nodes) = chain(3);
    let problem = Taint {
        sources: vec![(nodes[0], 1), (nodes[2], 4)],
    };
    let previous = try_solve_node_problem(&graph, &problem).unwrap();
    assert_eq!(*previous.fact_out(nodes[2]), 5);

    // Remove the last node and add two: the removed slot keeps bottom, the
    // new ones are solved.
    assert!(graph.remove_node(nodes[2]));
    let added = [graph.add_node(()), graph.add_node(())];
    graph.add_edge(nodes[1], added[0], ());
    graph.add_edge(added[0], added[1], ());
    let changed = [nodes[2], added[0]];
    let config = SolveConfig::new();
    let resumed = resume_node_problem(
        &graph,
        &problem,
        previous,
        &changed,
        ResumeMode::Reset,
        config,
    )
    .unwrap();
    let fresh = try_solve_node_problem(&graph, &problem).unwrap();
    assert_eq!(*resumed.fact_out(nodes[2]), 0);
    assert_same(
        &resumed,
        &fresh,
        &[nodes[0], nodes[1], nodes[2], added[0], added[1]],
    );
}

#[test]
#[should_panic(expected = "changed node is out of range")]
fn a_changed_node_beyond_the_bound_panics() {
    let (graph, _) = chain(2);
    let problem = Taint { sources: vec![] };
    let previous = try_solve_node_problem(&graph, &problem).unwrap();
    let beyond = [NodeId::from_index(5)];
    let config = SolveConfig::new();
    let _ = resume_node_problem(
        &graph,
        &problem,
        previous,
        &beyond,
        ResumeMode::Reset,
        config,
    );
}

/// Forward reachability from the entry over blocks.
struct Reached;

impl TryProblem<Cfg<()>> for Reached {
    type Fact = bool;
    type Error = Infallible;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _: &Cfg<()>) -> bool {
        false
    }

    fn boundary(&self, _: &Cfg<()>) -> Result<bool, Infallible> {
        Ok(true)
    }

    fn meet(&self, _: &Cfg<()>, _: BlockId, a: &bool, b: &bool) -> Result<bool, Infallible> {
        Ok(*a || *b)
    }

    fn transfer(&self, _: &Cfg<()>, _: BlockId, input: &bool) -> Result<bool, Infallible> {
        Ok(*input)
    }
}

#[test]
fn a_forward_block_resume_leaves_blocks_the_root_no_longer_reaches_at_bottom() {
    // entry -> a -> b; removing entry -> a cuts a and b off, and a, which
    // has no predecessor left, is not a boundary the way a seeded block is.
    let mut cfg = Cfg::<()>::new();
    let [a, b] = [cfg.new_block(), cfg.new_block()];
    let cut = cfg.add_edge(cfg.entry(), a, EdgeKind::Fallthrough);
    cfg.add_edge(a, b, EdgeKind::Fallthrough);
    let previous = try_solve_problem(&cfg, &Reached).unwrap();
    assert!(*previous.fact_out(b));

    assert!(cfg.remove_edge(cut));
    let config = SolveConfig::new();
    let resumed = resume_problem(&cfg, &Reached, previous, &[a], ResumeMode::Reset, config);
    let resumed = resumed.unwrap();
    let fresh = try_solve_problem(&cfg, &Reached).unwrap();
    for block in [cfg.entry(), a, b] {
        assert_eq!(resumed.fact_in(block), fresh.fact_in(block));
        assert_eq!(resumed.fact_out(block), fresh.fact_out(block));
    }
    assert!(!*resumed.fact_out(a) && !*resumed.fact_out(b));
    assert_eq!(resumed.steps(), 0);
}
