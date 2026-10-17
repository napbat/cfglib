//! The node-level solver over a consumer-owned graph, the way symtree-xref
//! runs taint over a symbol graph: the consumer implements `GraphView` on its
//! own storage, never converts it, and asks which nodes a value can reach.

use std::collections::BTreeSet;
use std::fmt;

use cfglib::{
    Direction, Facts, Graph, GraphView, NodeId, NodeProblem, SolveConfig, SolveError,
    TryNodeProblem, TrySolveError, solve_node_problem, solve_node_problem_from,
    solve_node_problem_from_with_config, solve_node_problem_with_config, try_solve_node_problem,
    try_solve_node_problem_from, try_solve_node_problem_from_with_config,
    try_solve_node_problem_with_config,
};

use crate::graphs::{Rng, Shape};

/// A consumer's symbol graph: dense `u32` ids, forward and reverse adjacency,
/// and a slot that was retired without renumbering its neighbors.
struct SymbolGraph {
    alive: Vec<bool>,
    edges: Vec<(u32, u32)>,
}

impl SymbolGraph {
    fn from_shape(shape: &Shape) -> Self {
        let mut alive = vec![true; shape.blocks];
        if let Some(hole) = shape.hole {
            alive[hole] = false;
        }
        let edges = shape
            .live_edges()
            .into_iter()
            .map(|(_, (source, target, _))| {
                (
                    u32::try_from(source).expect("a small graph"),
                    u32::try_from(target).expect("a small graph"),
                )
            })
            .collect();
        Self { alive, edges }
    }
}

impl GraphView for SymbolGraph {
    type NodeId = u32;

    fn node_bound(&self) -> usize {
        self.alive.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.alive.len())
            .filter(|index| self.alive[*index])
            .map(|index| u32::try_from(index).expect("a small graph"))
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.edges
            .iter()
            .filter(move |(source, _)| *source == node)
            .map(|(_, target)| *target)
    }

    fn predecessors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.edges
            .iter()
            .filter(move |(_, target)| *target == node)
            .map(|(source, _)| *source)
    }
}

/// Taint: a node is tainted when a source reaches it (forward) or when it
/// reaches a source, read as a sink (backward).
struct Taint {
    sources: Vec<u32>,
    direction: Direction,
}

impl<G: GraphView<NodeId = u32>> NodeProblem<G> for Taint {
    type Fact = bool;

    fn direction(&self) -> Direction {
        self.direction
    }

    fn bottom(&self, _graph: &G) -> bool {
        false
    }

    fn boundary(&self, _graph: &G) -> bool {
        false
    }

    fn meet(&self, a: &bool, b: &bool) -> bool {
        *a || *b
    }

    fn transfer(&self, _graph: &G, node: u32, input: &bool) -> bool {
        *input || self.sources.contains(&node)
    }
}

/// The nodes a breadth-first walk from `starts` reaches, following edges
/// forward or backward, by the shape's own edge list.
fn reach(shape: &Shape, starts: &[u32], direction: Direction) -> BTreeSet<u32> {
    let mut seen = BTreeSet::new();
    let mut queue: Vec<u32> = starts.to_vec();
    while let Some(node) = queue.pop() {
        if !seen.insert(node) {
            continue;
        }
        let index = usize::try_from(node).expect("a small graph");
        let next = match direction {
            Direction::Forward => shape.successors(index),
            Direction::Backward => shape.predecessors(index),
        };
        queue.extend(
            next.into_iter()
                .map(|n| u32::try_from(n).expect("a small graph")),
        );
    }
    seen
}

fn random_sources(rng: &mut Rng, shape: &Shape) -> Vec<u32> {
    let live = shape.live();
    (0..=rng.below(3))
        .map(|_| u32::try_from(live[rng.below(live.len())]).expect("a small graph"))
        .collect()
}

/// The nodes whose transferred fact is set. Facts are reported in program
/// order, so what a backward node transfers is its IN fact.
fn tainted(facts: &Facts<bool>, shape: &Shape, direction: Direction) -> BTreeSet<u32> {
    shape
        .live()
        .into_iter()
        .map(|node| u32::try_from(node).expect("a small graph"))
        .filter(|node| match direction {
            Direction::Forward => *facts.fact_out(*node),
            Direction::Backward => *facts.fact_in(*node),
        })
        .collect()
}

#[test]
fn taint_over_a_consumer_graph_equals_reachability_from_the_sources() {
    let mut rng = Rng::new(21);
    for direction in [Direction::Forward, Direction::Backward] {
        for _ in 0..300 {
            let shape = Shape::random(&mut rng, false, false);
            let graph = SymbolGraph::from_shape(&shape);
            let sources = random_sources(&mut rng, &shape);
            let problem = Taint {
                sources: sources.clone(),
                direction,
            };
            let facts = solve_node_problem(&graph, &problem).unwrap();
            assert_eq!(
                tainted(&facts, &shape, direction),
                reach(&shape, &sources, direction),
                "{direction:?} {shape:?}"
            );
        }
    }
}

#[test]
fn the_same_taint_over_the_crates_own_graph_store_gives_the_same_answer() {
    let mut rng = Rng::new(22);
    for _ in 0..200 {
        let shape = Shape::random(&mut rng, false, false);
        let mut store = Graph::<(), ()>::new();
        let ids: Vec<NodeId> = (0..shape.blocks).map(|_| store.add_node(())).collect();
        for (_, (source, target, _)) in shape.live_edges() {
            store.add_edge(ids[source], ids[target], ());
        }
        if let Some(hole) = shape.hole {
            assert!(store.remove_node(ids[hole]));
        }
        let sources = random_sources(&mut rng, &shape);
        let facts = solve_node_problem(&store, &StoreTaint(sources.clone())).unwrap();
        let reached = reach(&shape, &sources, Direction::Forward);
        for node in shape.live() {
            let flagged = *facts.fact_out(ids[node]);
            assert_eq!(
                flagged,
                reached.contains(&u32::try_from(node).expect("a small graph")),
                "{shape:?}"
            );
        }
    }
}

/// Forward taint over [`Graph`], whose nodes are [`NodeId`].
struct StoreTaint(Vec<u32>);

impl NodeProblem<Graph<(), ()>> for StoreTaint {
    type Fact = bool;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &Graph<(), ()>) -> bool {
        false
    }

    fn boundary(&self, _graph: &Graph<(), ()>) -> bool {
        false
    }

    fn meet(&self, a: &bool, b: &bool) -> bool {
        *a || *b
    }

    fn transfer(&self, _graph: &Graph<(), ()>, node: NodeId, input: &bool) -> bool {
        *input
            || self
                .0
                .contains(&u32::try_from(node.index()).expect("a small graph"))
    }
}

#[test]
fn node_facts_are_in_program_order_whichever_way_the_analysis_flows() {
    // a -> b -> c, with c a sink; the backward flow runs c, b, a.
    let graph = SymbolGraph {
        alive: vec![true; 3],
        edges: vec![(0, 1), (1, 2)],
    };
    let facts = solve_node_problem(
        &graph,
        &Taint {
            sources: vec![2],
            direction: Direction::Backward,
        },
    )
    .unwrap();
    // c has no successor: the boundary fact (false) is what follows it, and
    // it is tainted before it because it is the sink itself.
    assert!(!*facts.fact_out(2_u32));
    assert!(*facts.fact_in(2_u32));
    // What c's fact hands back reaches the end of b and then of a.
    assert!(*facts.fact_out(1_u32));
    assert!(*facts.fact_out(0_u32));
}

#[test]
fn seeding_a_source_reaches_exactly_what_the_source_reaches() {
    let mut rng = Rng::new(23);
    for _ in 0..200 {
        let shape = Shape::random(&mut rng, false, false);
        let graph = SymbolGraph::from_shape(&shape);
        let live = shape.live();
        let seed = u32::try_from(live[rng.below(live.len())]).expect("a small graph");
        let problem = Taint {
            sources: vec![seed],
            direction: Direction::Forward,
        };
        let seeded = solve_node_problem_from(&graph, &problem, &[seed]).unwrap();
        let full = solve_node_problem(&graph, &problem).unwrap();
        assert_eq!(
            tainted(&seeded, &shape, Direction::Forward),
            reach(&shape, &[seed], Direction::Forward)
        );
        assert_eq!(
            tainted(&seeded, &shape, Direction::Forward),
            tainted(&full, &shape, Direction::Forward)
        );
        // A duplicate seed is queued once.
        let doubled = solve_node_problem_from(&graph, &problem, &[seed, seed]).unwrap();
        assert_eq!(doubled.steps(), seeded.steps());
    }
}

#[test]
fn only_the_seeded_node_is_a_source_for_a_seeded_solve() {
    // Two sources, one seed: the other source is never visited, so its taint
    // is not part of the answer.
    let graph = SymbolGraph {
        alive: vec![true; 4],
        edges: vec![(0, 1), (2, 3)],
    };
    let problem = Taint {
        sources: vec![0, 2],
        direction: Direction::Forward,
    };
    let facts = solve_node_problem_from(&graph, &problem, &[0]).unwrap();
    assert!(*facts.fact_out(1_u32));
    assert!(!*facts.fact_out(2_u32));
    assert!(!*facts.fact_out(3_u32));
    let nothing = solve_node_problem_from(&graph, &problem, &[]).unwrap();
    assert_eq!(nothing.steps(), 0);
    assert!(!*nothing.fact_out(0_u32));
}

#[test]
fn a_removed_slot_is_never_visited_and_keeps_bottom() {
    // Node 1 was retired; a source list that still names it must not taint it.
    let graph = SymbolGraph {
        alive: vec![true, false, true],
        edges: vec![(0, 2)],
    };
    let problem = Taint {
        sources: vec![1],
        direction: Direction::Forward,
    };
    let facts = solve_node_problem(&graph, &problem).unwrap();
    assert!(
        !*facts.fact_out(1_u32),
        "a removed node is not a node of the view"
    );
    assert!(!*facts.fact_out(2_u32));
}

#[test]
#[should_panic(expected = "seed")]
fn seeding_a_removed_slot_panics() {
    let graph = SymbolGraph {
        alive: vec![true, false],
        edges: vec![],
    };
    let problem = Taint {
        sources: vec![],
        direction: Direction::Forward,
    };
    let _ = solve_node_problem_from(&graph, &problem, &[1]);
}

#[test]
fn a_cycle_converges_and_a_step_limit_below_the_work_fails_with_its_position() {
    // 0 -> 1 -> 2 -> 1: taint enters the cycle and circulates.
    let graph = SymbolGraph {
        alive: vec![true; 3],
        edges: vec![(0, 1), (1, 2), (2, 1)],
    };
    let problem = Taint {
        sources: vec![0],
        direction: Direction::Forward,
    };
    let unbounded = solve_node_problem(&graph, &problem).unwrap();
    let needed = unbounded.steps();
    let short = SolveConfig::with_step_limit(needed - 1);
    let SolveError::StepLimitExceeded {
        limit,
        steps,
        pending_node,
    } = solve_node_problem_with_config(&graph, &problem, short).unwrap_err();
    assert_eq!((limit, steps), (needed - 1, needed - 1));
    assert!(pending_node < graph.node_bound());

    let seeds = [0_u32, 1, 2];
    assert!(solve_node_problem_from_with_config(&graph, &problem, &seeds, short).is_err());
    for limit in [needed, needed + 5] {
        let config = SolveConfig::with_step_limit(limit);
        let bounded = solve_node_problem_with_config(&graph, &problem, config).unwrap();
        let seeded = solve_node_problem_from_with_config(&graph, &problem, &seeds, config).unwrap();
        for node in 0..3_u32 {
            assert_eq!(bounded.fact_out(node), unbounded.fact_out(node));
            assert_eq!(seeded.fact_out(node), unbounded.fact_out(node));
        }
    }
}

/// Taint that rejects one node, as a verifier rejects an ill-typed symbol.
struct Verifying {
    sources: Vec<u32>,
    rejected: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rejected(u32);

impl fmt::Display for Rejected {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "node {} rejected its input", self.0)
    }
}

impl std::error::Error for Rejected {}

impl<G: GraphView<NodeId = u32>> TryNodeProblem<G> for Verifying {
    type Fact = bool;
    type Error = Rejected;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &G) -> bool {
        false
    }

    fn boundary(&self, _graph: &G) -> Result<bool, Rejected> {
        Ok(false)
    }

    fn meet(&self, _graph: &G, _node: u32, a: &bool, b: &bool) -> Result<bool, Rejected> {
        Ok(*a || *b)
    }

    fn transfer(&self, _graph: &G, node: u32, input: &bool) -> Result<bool, Rejected> {
        if *input && self.rejected == Some(node) {
            return Err(Rejected(node));
        }
        Ok(*input || self.sources.contains(&node))
    }
}

#[test]
fn the_fallible_node_solvers_agree_with_the_infallible_one_and_report_refusals() {
    let graph = SymbolGraph {
        alive: vec![true; 4],
        edges: vec![(0, 1), (1, 2), (2, 3)],
    };
    let sources = vec![0];
    let plain = solve_node_problem(
        &graph,
        &Taint {
            sources: sources.clone(),
            direction: Direction::Forward,
        },
    )
    .unwrap();
    let accepting = Verifying {
        sources: sources.clone(),
        rejected: None,
    };
    let seeds = [0_u32, 1, 2, 3];
    let config = SolveConfig::new();
    let results = [
        try_solve_node_problem(&graph, &accepting),
        try_solve_node_problem_with_config(&graph, &accepting, config),
        try_solve_node_problem_from(&graph, &accepting, &seeds),
        try_solve_node_problem_from_with_config(&graph, &accepting, &seeds, config),
    ];
    for result in results {
        let facts = result.unwrap();
        for node in 0..4_u32 {
            assert_eq!(facts.fact_out(node), plain.fact_out(node));
        }
    }

    // Tainted input reaches node 2, which rejects it; node 3 is never asked.
    let rejecting = Verifying {
        sources,
        rejected: Some(2),
    };
    assert_eq!(
        try_solve_node_problem(&graph, &rejecting).unwrap_err(),
        TrySolveError::Problem(Rejected(2))
    );
    assert_eq!(
        try_solve_node_problem_from(&graph, &rejecting, &[0]).unwrap_err(),
        TrySolveError::Problem(Rejected(2))
    );
    // Without a taint source nothing reaches node 2, so nothing is rejected.
    let clean = Verifying {
        sources: vec![],
        rejected: Some(2),
    };
    assert!(try_solve_node_problem(&graph, &clean).is_ok());

    let limited = SolveConfig::with_step_limit(1);
    assert!(matches!(
        try_solve_node_problem_with_config(&graph, &accepting, limited),
        Err(TrySolveError::Solver(SolveError::StepLimitExceeded {
            limit: 1,
            steps: 1,
            ..
        }))
    ));
}

#[test]
fn a_removed_node_of_the_crates_graph_store_is_neither_counted_nor_visited() {
    let mut store = Graph::<(), ()>::new();
    let kept = store.add_node(());
    let removed = store.add_node(());
    let other = store.add_node(());
    store.add_edge(kept, other, ());
    assert!(store.remove_node(removed));
    // The only source is the removed node: it must stay at bottom, and the
    // solve visits the two live nodes once each.
    let removed_index = u32::try_from(removed.index()).expect("a small graph");
    let facts = solve_node_problem(&store, &StoreTaint(vec![removed_index])).unwrap();
    assert_eq!(facts.steps(), 2);
    assert!(!*facts.fact_out(removed));
    assert!(!*facts.fact_out(other));
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn seeding_a_removed_node_of_the_crates_graph_store_panics() {
    let mut store = Graph::<(), ()>::new();
    let kept = store.add_node(());
    let removed = store.add_node(());
    store.add_edge(kept, removed, ());
    assert!(store.remove_node(removed));
    let _ = solve_node_problem_from(&store, &StoreTaint(vec![]), &[removed]);
}
