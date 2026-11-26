//! Resumed solves against fresh solves of the edited graph, and the gap of a
//! seeded solve that they close.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::convert::Infallible;

use super::{Resume, ResumeMode, resume_node_problem};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::node_fixpoint::{TryNodeProblem, solve_node_problem};
use crate::dataflow::solve::{Facts, Seeds, Solve};
use crate::edge::EdgeKind;
use crate::graph::store::{Graph, NodeId};
use crate::graph::view::{Direction, RootedView};

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
    let previous = solve_node_problem(&graph, &before, Solve::default()).unwrap();

    // The seeded solve starts b and c at bottom and never visits a, so a's
    // taint is lost downstream of the edit.
    let seeds = Solve {
        seeds: Seeds::Of(&[b]),
        ..Solve::default()
    };
    let seeded = solve_node_problem(&graph, &after, seeds).unwrap();
    assert_eq!((*seeded.fact_out(b), *seeded.fact_out(c)), (2, 2));

    let fresh = solve_node_problem(&graph, &after, Solve::default()).unwrap();
    for mode in [ResumeMode::Reset, ResumeMode::Grow] {
        let edit = Resume {
            changed: &[b],
            mode,
        };
        let resumed = resume_node_problem(&graph, &after, previous.clone(), edit, Solve::default());
        let resumed = resumed.unwrap();
        assert_eq!((*resumed.fact_out(b), *resumed.fact_out(c)), (3, 3));
        assert_same(&resumed, &fresh, &nodes);
        assert_eq!(
            resumed.stats().step_count(),
            2,
            "{mode:?} re-solves b and c only"
        );
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
    let previous = solve_node_problem(&graph, &before, Solve::default()).unwrap();
    let edit = Resume {
        changed: &[first],
        mode: ResumeMode::Reset,
    };
    let resumed = resume_node_problem(&graph, &after, previous, edit, Solve::default()).unwrap();
    let steps = resumed.stats().step_count();
    assert!(steps <= 4 * 100, "{steps} steps");
    let fresh = solve_node_problem(&graph, &after, Solve::default()).unwrap();
    assert_same(&resumed, &fresh, &nodes);
}

#[test]
fn slots_follow_the_edited_bound_and_a_removed_node_gets_bottom() {
    let (mut graph, nodes) = chain(3);
    let problem = Taint {
        sources: vec![(nodes[0], 1), (nodes[2], 4)],
    };
    let previous = solve_node_problem(&graph, &problem, Solve::default()).unwrap();
    assert_eq!(*previous.fact_out(nodes[2]), 5);

    // Remove the last node and add two: the removed slot keeps bottom, the
    // new ones are solved.
    assert!(graph.remove_node(nodes[2]));
    let added = [graph.add_node(()), graph.add_node(())];
    graph.add_edge(nodes[1], added[0], ());
    graph.add_edge(added[0], added[1], ());
    let changed = [nodes[2], added[0]];
    let edit = Resume {
        changed: &changed,
        mode: ResumeMode::Reset,
    };
    let resumed = resume_node_problem(&graph, &problem, previous, edit, Solve::default()).unwrap();
    let fresh = solve_node_problem(&graph, &problem, Solve::default()).unwrap();
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
    let previous = solve_node_problem(&graph, &problem, Solve::default()).unwrap();
    let beyond = [NodeId::from_index(5)];
    let edit = Resume {
        changed: &beyond,
        mode: ResumeMode::Reset,
    };
    let _ = resume_node_problem(&graph, &problem, previous, edit, Solve::default());
}

/// Forward reachability from the entry over blocks.
struct Reached;

impl TryNodeProblem<Cfg<()>> for Reached {
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
fn a_reachable_resume_leaves_blocks_the_root_no_longer_reaches_at_bottom() {
    // entry -> a -> b; removing entry -> a cuts a and b off, and a, which
    // has no predecessor left, is not a boundary the way a seeded block is.
    let mut cfg = Cfg::<()>::new();
    let [a, b] = [cfg.new_block(), cfg.new_block()];
    let cut = cfg.add_edge(cfg.entry(), a, EdgeKind::Fallthrough);
    cfg.add_edge(a, b, EdgeKind::Fallthrough);
    let rooted = Solve {
        seeds: Seeds::Reachable(cfg.root()),
        ..Solve::default()
    };
    let previous = solve_node_problem(&cfg, &Reached, rooted).unwrap();
    assert!(*previous.fact_out(b));

    assert!(cfg.remove_edge(cut));
    let edit = Resume {
        changed: &[a],
        mode: ResumeMode::Reset,
    };
    let resumed = resume_node_problem(&cfg, &Reached, previous, edit, rooted).unwrap();
    let fresh = solve_node_problem(&cfg, &Reached, rooted).unwrap();
    for block in [cfg.entry(), a, b] {
        assert_eq!(resumed.fact_in(block), fresh.fact_in(block));
        assert_eq!(resumed.fact_out(block), fresh.fact_out(block));
    }
    assert!(!*resumed.fact_out(a) && !*resumed.fact_out(b));
    assert_eq!(resumed.stats().step_count(), 0);
}

#[test]
fn a_resume_from_named_seeds_re_solves_only_within_their_reach() {
    // a -> b -> c and x -> c. Resuming within what `b` reaches re-solves b
    // and c after the edit at a; a is reset and, outside b's reach, stays at
    // bottom as in a fresh solve from b, while x keeps its previous fact.
    let (mut graph, nodes) = chain(3);
    let [a, b, c] = [nodes[0], nodes[1], nodes[2]];
    let x = graph.add_node(());
    graph.add_edge(x, c, ());
    let problem = Taint {
        sources: vec![(a, 1), (x, 8)],
    };
    let previous = solve_node_problem(&graph, &problem, Solve::default()).unwrap();
    let within_b = Solve {
        seeds: Seeds::Of(&[b]),
        ..Solve::default()
    };
    let edit = Resume {
        changed: &[a, b],
        mode: ResumeMode::Reset,
    };
    let resumed = resume_node_problem(&graph, &problem, previous, edit, within_b).unwrap();
    assert_eq!(
        *resumed.fact_out(a),
        0,
        "reset, outside b's reach, not re-solved"
    );
    assert_eq!(*resumed.fact_out(b), 0, "b reads a's reset fact");
    assert_eq!(*resumed.fact_out(c), 8, "c still meets x's kept fact");
    assert_eq!(*resumed.fact_out(x), 8);
    assert_eq!(resumed.stats().step_count(), 2);
}
