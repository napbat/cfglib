//! The edge-sensitive solver: facts on edges as well as nodes, exceptional
//! edges that leave before a node's effect, and the `Reachable` lifting for
//! verification-style analyses whose bottom is "unreached".

use std::collections::BTreeSet;
use std::fmt;

use cfglib::{
    BlockId, Cfg, Direction, Edge, EdgeId, EdgeKind, EdgeProblem, EdgeRef, EdgeView, FilteredEdges,
    KindedEdge, Reachable, ReachableEdgeProblem, SolveConfig, SolveError, TrySolveError,
    solve_edge_problem, solve_edge_problem_from, solve_edge_problem_from_with_config,
    solve_edge_problem_with_config, try_solve_edge_problem, try_solve_edge_problem_from,
    try_solve_edge_problem_from_with_config, try_solve_edge_problem_with_config,
};

use crate::graphs::{GenKill, Rng, Shape, union};
use crate::reference::edge_round_robin;

const BOUNDARY_MARK: u8 = 6;
const TRUE_MARK: u8 = 8;

fn id(index: usize) -> BlockId {
    BlockId::from_index(index)
}

fn edge_id(index: usize) -> EdgeId {
    EdgeId::from_index(index)
}

/// Gen/kill labels with edge-specific rules: an unwind edge carries the
/// node's physical pre-state, and a taken-branch edge adds a mark.
struct Labels<'a> {
    effects: &'a GenKill,
    direction: Direction,
}

impl Labels<'_> {
    fn edge_fact(
        &self,
        kind: EdgeKind,
        input: &BTreeSet<u8>,
        output: &BTreeSet<u8>,
    ) -> BTreeSet<u8> {
        let flowing = match self.direction {
            Direction::Forward => output,
            Direction::Backward => input,
        };
        match kind {
            EdgeKind::ExceptionUnwind => match self.direction {
                Direction::Forward => input.clone(),
                Direction::Backward => output.clone(),
            },
            EdgeKind::ConditionalTrue => {
                let mut marked = flowing.clone();
                marked.insert(TRUE_MARK);
                marked
            }
            _ => flowing.clone(),
        }
    }
}

impl<G> EdgeProblem<G> for Labels<'_>
where
    G: EdgeView<NodeId = BlockId>,
    G::EdgeData: KindedEdge,
{
    type Fact = BTreeSet<u8>;

    fn direction(&self) -> Direction {
        self.direction
    }

    fn bottom(&self, _graph: &G) -> Self::Fact {
        BTreeSet::new()
    }

    fn boundary(&self, graph: &G, node: BlockId) -> Option<Self::Fact> {
        let is_boundary = match self.direction {
            Direction::Forward => node.index() == 0,
            Direction::Backward => graph.outgoing(node).next().is_none(),
        };
        is_boundary.then(|| BTreeSet::from([BOUNDARY_MARK]))
    }

    fn meet(&self, left: &Self::Fact, right: &Self::Fact) -> Self::Fact {
        union(left, right)
    }

    fn transfer_node(&self, _graph: &G, node: BlockId, flow_fact: &Self::Fact) -> Self::Fact {
        self.effects.apply(node.index(), flow_fact)
    }

    fn transfer_edge(
        &self,
        _graph: &G,
        edge: EdgeRef<'_, BlockId, G::EdgeId, G::EdgeData>,
        node_input: &Self::Fact,
        node_output: &Self::Fact,
    ) -> Self::Fact {
        self.edge_fact(edge.data().kind(), node_input, node_output)
    }
}

fn assert_matches_reference(shape: &Shape, direction: Direction, effects: &GenKill) {
    let cfg = shape.cfg();
    let problem = Labels { effects, direction };
    let facts = solve_edge_problem(&cfg, &problem).unwrap();
    let forward = direction == Direction::Forward;
    let reference = edge_round_robin(
        shape,
        forward,
        &BTreeSet::new(),
        &|node| {
            let is_boundary = if forward {
                node == 0
            } else {
                shape.successors(node).is_empty()
            };
            is_boundary.then(|| BTreeSet::from([BOUNDARY_MARK]))
        },
        &union,
        &|node, flow| effects.apply(node, flow),
        &|kind, input, output| problem.edge_fact(kind, input, output),
    );
    for node in shape.live() {
        assert_eq!(
            facts.fact_in(id(node)),
            &reference.input[node],
            "IN of {node} in {shape:?}"
        );
        assert_eq!(
            facts.fact_out(id(node)),
            &reference.output[node],
            "OUT of {node} in {shape:?}"
        );
    }
    assert_eq!(facts.edge_facts().len(), cfg.edge_bound());
    for (index, expected) in reference.edge.iter().enumerate() {
        assert_eq!(
            facts.fact_on(edge_id(index)),
            expected.as_ref(),
            "edge {index} in {shape:?}"
        );
        assert_eq!(facts.edge_facts()[index].as_ref(), expected.as_ref());
    }
}

#[test]
fn forward_edge_problems_match_the_round_robin_reference_with_unwind_edges_and_holes() {
    let mut rng = Rng::new(31);
    for _ in 0..300 {
        let shape = Shape::random(&mut rng, false, true);
        let effects = GenKill::random(&mut rng, shape.blocks);
        assert_matches_reference(&shape, Direction::Forward, &effects);
    }
}

#[test]
fn backward_edge_problems_match_the_round_robin_reference_with_unwind_edges_and_holes() {
    let mut rng = Rng::new(32);
    for _ in 0..300 {
        let shape = Shape::random(&mut rng, false, true);
        let effects = GenKill::random(&mut rng, shape.blocks);
        assert_matches_reference(&shape, Direction::Backward, &effects);
    }
}

#[test]
fn an_unwind_edge_carries_the_pre_state_while_a_normal_edge_carries_the_post_state() {
    // entry (gens 1) -> normal, and entry --unwind--> handler.
    let shape = Shape {
        blocks: 3,
        edges: vec![
            (0, 1, EdgeKind::Fallthrough),
            (0, 2, EdgeKind::ExceptionUnwind),
        ],
        hole: None,
    };
    let effects = GenKill::with_gen(3, &[(0, 1)]);
    let cfg = shape.cfg();
    let facts = solve_edge_problem(
        &cfg,
        &Labels {
            effects: &effects,
            direction: Direction::Forward,
        },
    )
    .unwrap();
    assert_eq!(facts.fact_in(id(1)), &BTreeSet::from([BOUNDARY_MARK, 1]));
    assert_eq!(facts.fact_in(id(2)), &BTreeSet::from([BOUNDARY_MARK]));
    assert_eq!(
        facts.fact_on(edge_id(0)),
        Some(&BTreeSet::from([BOUNDARY_MARK, 1]))
    );
    assert_eq!(
        facts.fact_on(edge_id(1)),
        Some(&BTreeSet::from([BOUNDARY_MARK]))
    );
}

#[test]
fn a_filtered_view_solves_like_the_graph_without_the_filtered_edges() {
    let mut rng = Rng::new(33);
    for _ in 0..150 {
        let shape = Shape::random(&mut rng, true, true);
        let effects = GenKill::random(&mut rng, shape.blocks);
        let problem = Labels {
            effects: &effects,
            direction: Direction::Forward,
        };
        let cfg = shape.cfg();
        let normal_only = |_: EdgeId, edge: &Edge<()>| edge.kind() != EdgeKind::ExceptionUnwind;
        let filtered = FilteredEdges::new(&cfg, normal_only);
        let through_view = solve_edge_problem(&filtered, &problem).unwrap();

        let mut rebuilt = shape.clone();
        rebuilt
            .edges
            .retain(|(_, _, kind)| *kind != EdgeKind::ExceptionUnwind);
        let rebuilt_cfg = rebuilt.cfg();
        let direct = solve_edge_problem(&rebuilt_cfg, &problem).unwrap();
        for node in shape.live() {
            assert_eq!(
                through_view.fact_in(id(node)),
                direct.fact_in(id(node)),
                "{shape:?}"
            );
            assert_eq!(
                through_view.fact_out(id(node)),
                direct.fact_out(id(node)),
                "{shape:?}"
            );
        }
        for (index, (_, _, kind)) in shape.edges.iter().enumerate() {
            if *kind == EdgeKind::ExceptionUnwind {
                assert_eq!(
                    through_view.fact_on(edge_id(index)),
                    None,
                    "an excluded edge has no fact"
                );
            }
        }
        assert!(
            filtered
                .edge_ids()
                .all(|edge| through_view.fact_on(edge).is_some())
        );
    }
}

#[test]
fn seeded_edge_solves_visit_only_what_the_seeds_reach() {
    // entry -> a -> b, and an island c -> b.
    let shape = Shape::new(4, &[(0, 1), (1, 2), (3, 2)]);
    let effects = GenKill::with_gen(4, &[(0, 0), (1, 1), (3, 3)]);
    let cfg = shape.cfg();
    let problem = Labels {
        effects: &effects,
        direction: Direction::Forward,
    };
    let full = solve_edge_problem(&cfg, &problem).unwrap();
    // The island is visited by the full solve and its gen reaches the join.
    assert!(full.fact_in(id(2)).contains(&3));

    let seeded = solve_edge_problem_from(&cfg, &problem, &[id(0)]).unwrap();
    assert!(
        !seeded.fact_in(id(2)).contains(&3),
        "the island was never seeded"
    );
    assert_eq!(
        seeded.fact_in(id(2)),
        &BTreeSet::from([BOUNDARY_MARK, 0, 1])
    );
    assert_eq!(
        seeded.fact_on(edge_id(2)),
        Some(&BTreeSet::new()),
        "an unvisited edge stays at bottom"
    );

    let doubled = solve_edge_problem_from(&cfg, &problem, &[id(0), id(0)]).unwrap();
    assert_eq!(doubled.steps(), seeded.steps());
    assert_eq!(
        solve_edge_problem_from(&cfg, &problem, &[])
            .unwrap()
            .steps(),
        0
    );

    let everything: Vec<BlockId> = cfg.block_ids().collect();
    let all = solve_edge_problem_from(&cfg, &problem, &everything).unwrap();
    for node in cfg.block_ids() {
        assert_eq!(all.fact_in(node), full.fact_in(node));
        assert_eq!(all.fact_out(node), full.fact_out(node));
    }
    for edge in cfg.edge_ids() {
        assert_eq!(all.fact_on(edge), full.fact_on(edge));
    }
}

#[test]
fn the_step_limit_applies_to_every_edge_solver_form_and_raising_it_converges() {
    // entry -> head -> body -> head, head -> exit
    let shape = Shape::new(4, &[(0, 1), (1, 2), (2, 1), (1, 3)]);
    let effects = GenKill::with_gen(4, &[(2, 1), (3, 2)]);
    let cfg = shape.cfg();
    let problem = Labels {
        effects: &effects,
        direction: Direction::Forward,
    };
    let unbounded = solve_edge_problem(&cfg, &problem).unwrap();
    let needed = unbounded.steps();
    let everything: Vec<BlockId> = cfg.block_ids().collect();
    let short = SolveConfig::with_step_limit(needed - 1);

    for error in [
        solve_edge_problem_with_config(&cfg, &problem, short).unwrap_err(),
        solve_edge_problem_from_with_config(&cfg, &problem, &everything, short).unwrap_err(),
    ] {
        let SolveError::StepLimitExceeded {
            limit,
            steps,
            pending_node,
        } = error;
        assert_eq!((limit, steps), (needed - 1, needed - 1));
        assert!(pending_node < cfg.block_bound());
    }
    for limit in [needed, needed + 7] {
        let config = SolveConfig::with_step_limit(limit);
        let bounded = solve_edge_problem_with_config(&cfg, &problem, config).unwrap();
        let seeded =
            solve_edge_problem_from_with_config(&cfg, &problem, &everything, config).unwrap();
        for node in cfg.block_ids() {
            assert_eq!(bounded.fact_out(node), unbounded.fact_out(node));
            assert_eq!(seeded.fact_out(node), unbounded.fact_out(node));
        }
    }
}

#[test]
fn a_removed_block_is_not_visited_even_when_the_problem_has_a_boundary_for_it() {
    /// Every node starts with a fact, as a graph with several roots would.
    struct EveryNodeIsARoot;

    impl EdgeProblem<Cfg<()>> for EveryNodeIsARoot {
        type Fact = BTreeSet<u8>;

        fn direction(&self) -> Direction {
            Direction::Forward
        }

        fn bottom(&self, _graph: &Cfg<()>) -> Self::Fact {
            BTreeSet::new()
        }

        fn boundary(&self, _graph: &Cfg<()>, _node: BlockId) -> Option<Self::Fact> {
            Some(BTreeSet::from([1]))
        }

        fn meet(&self, left: &Self::Fact, right: &Self::Fact) -> Self::Fact {
            union(left, right)
        }

        fn transfer_node(&self, _graph: &Cfg<()>, _node: BlockId, flow: &Self::Fact) -> Self::Fact {
            flow.clone()
        }
    }

    let mut shape = Shape::new(3, &[(0, 1), (1, 2)]);
    shape.hole = Some(1);
    let cfg = shape.cfg();
    let facts = solve_edge_problem(&cfg, &EveryNodeIsARoot).unwrap();
    assert_eq!(facts.fact_in(id(0)), &BTreeSet::from([1]));
    assert_eq!(facts.fact_in(id(2)), &BTreeSet::from([1]));
    assert_eq!(
        facts.fact_in(id(1)),
        &BTreeSet::new(),
        "a removed block is not a node of the view"
    );
    assert_eq!(
        facts.steps(),
        cfg.block_count(),
        "each live block is visited once"
    );
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn seeding_a_removed_block_panics() {
    let mut shape = Shape::new(3, &[(0, 1), (1, 2)]);
    shape.hole = Some(1);
    let effects = GenKill::with_gen(3, &[]);
    let cfg = shape.cfg();
    let problem = Labels {
        effects: &effects,
        direction: Direction::Forward,
    };
    let _ = solve_edge_problem_from(&cfg, &problem, &[id(1)]);
}

/// Verification-style stack depth: unreached is `None`, merges must agree.
struct Depth {
    deltas: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Mismatch {
    node: usize,
    left: i32,
    right: i32,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "depths {} and {} disagree at {}",
            self.left, self.right, self.node
        )
    }
}

impl std::error::Error for Mismatch {}

impl ReachableEdgeProblem<Cfg<()>> for Depth {
    type Fact = i32;
    type Error = Mismatch;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn entry_fact(&self, _graph: &Cfg<()>, node: BlockId) -> Result<Option<i32>, Mismatch> {
        Ok((node.index() == 0).then_some(0))
    }

    fn merge(
        &self,
        _graph: &Cfg<()>,
        node: BlockId,
        left: &i32,
        right: &i32,
    ) -> Result<i32, Mismatch> {
        if left == right {
            Ok(*left)
        } else {
            Err(Mismatch {
                node: node.index(),
                left: *left,
                right: *right,
            })
        }
    }

    fn transfer(&self, _graph: &Cfg<()>, node: BlockId, input: &i32) -> Result<i32, Mismatch> {
        Ok(input + self.deltas[node.index()])
    }

    fn edge_observes_input(
        &self,
        _graph: &Cfg<()>,
        edge: EdgeRef<'_, BlockId, EdgeId, Edge<()>>,
    ) -> bool {
        edge.kind() == EdgeKind::ExceptionUnwind
    }
}

fn depth_shape(edges: &[(usize, usize, EdgeKind)], blocks: usize) -> Cfg<()> {
    Shape {
        blocks,
        edges: edges.to_vec(),
        hole: None,
    }
    .cfg()
}

#[test]
fn reachable_lifting_keeps_unreached_nodes_at_none_and_reached_ones_at_their_depth() {
    use EdgeKind::{ConditionalFalse, ConditionalTrue, Fallthrough};
    // A diamond whose arms agree, plus an island nothing reaches.
    let cfg = depth_shape(
        &[
            (0, 1, ConditionalTrue),
            (0, 2, ConditionalFalse),
            (1, 3, Fallthrough),
            (2, 3, Fallthrough),
            (4, 3, Fallthrough),
        ],
        5,
    );
    let problem = Reachable(Depth {
        deltas: vec![1, 2, 2, 0, 9],
    });
    let facts = try_solve_edge_problem(&cfg, &problem).unwrap();
    assert_eq!(facts.fact_in(id(0)), &Some(0));
    assert_eq!(facts.fact_out(id(0)), &Some(1));
    assert_eq!(facts.fact_in(id(3)), &Some(3));
    assert_eq!(facts.fact_in(id(4)), &None);
    assert_eq!(facts.fact_out(id(4)), &None);
    // The island's edge carries nothing, so it does not disturb the merge.
    assert_eq!(facts.fact_on(edge_id(4)), Some(&None));
}

#[test]
fn reachable_lifting_reports_a_disagreeing_merge_as_a_consumer_error() {
    use EdgeKind::{ConditionalFalse, ConditionalTrue, Fallthrough};
    let cfg = depth_shape(
        &[
            (0, 1, ConditionalTrue),
            (0, 2, ConditionalFalse),
            (1, 3, Fallthrough),
            (2, 3, Fallthrough),
        ],
        4,
    );
    let problem = Reachable(Depth {
        deltas: vec![0, 1, 2, 0],
    });
    match try_solve_edge_problem(&cfg, &problem).unwrap_err() {
        TrySolveError::Problem(mismatch) => {
            assert_eq!(mismatch.node, 3);
            assert_eq!(
                BTreeSet::from([mismatch.left, mismatch.right]),
                BTreeSet::from([1, 2])
            );
        }
        TrySolveError::Solver(error) => panic!("expected a consumer error, got {error}"),
    }
}

#[test]
fn reachable_lifting_lets_an_exceptional_edge_observe_the_nodes_pre_state() {
    use EdgeKind::{ExceptionUnwind, Fallthrough};
    let cfg = depth_shape(&[(0, 1, Fallthrough), (0, 2, ExceptionUnwind)], 3);
    let problem = Reachable(Depth {
        deltas: vec![3, 0, 0],
    });
    let facts = try_solve_edge_problem(&cfg, &problem).unwrap();
    assert_eq!(
        facts.fact_in(id(1)),
        &Some(3),
        "the normal successor sees the node's effect"
    );
    assert_eq!(
        facts.fact_in(id(2)),
        &Some(0),
        "the handler sees the state the node received"
    );
}

#[test]
fn the_fallible_edge_solvers_share_the_step_limit_and_the_seed_contract() {
    use EdgeKind::Fallthrough;
    let cfg = depth_shape(&[(0, 1, Fallthrough), (1, 2, Fallthrough)], 3);
    let problem = Reachable(Depth {
        deltas: vec![1, 1, 1],
    });
    let full = try_solve_edge_problem(&cfg, &problem).unwrap();
    let needed = full.steps();
    let everything: Vec<BlockId> = cfg.block_ids().collect();

    let seeded = try_solve_edge_problem_from(&cfg, &problem, &[id(0)]).unwrap();
    assert_eq!(seeded.fact_in(id(2)), full.fact_in(id(2)));
    let nothing = try_solve_edge_problem_from(&cfg, &problem, &[]).unwrap();
    assert_eq!(
        nothing.fact_in(id(0)),
        &None,
        "nothing was visited, so nothing was reached"
    );

    let short = SolveConfig::with_step_limit(needed - 1);
    for error in [
        try_solve_edge_problem_with_config(&cfg, &problem, short).unwrap_err(),
        try_solve_edge_problem_from_with_config(&cfg, &problem, &everything, short).unwrap_err(),
    ] {
        match error {
            TrySolveError::Solver(SolveError::StepLimitExceeded { limit, steps, .. }) => {
                assert_eq!((limit, steps), (needed - 1, needed - 1));
            }
            TrySolveError::Problem(mismatch) => panic!("unexpected consumer error: {mismatch}"),
        }
    }
    let enough = SolveConfig::with_step_limit(needed);
    assert!(try_solve_edge_problem_with_config(&cfg, &problem, enough).is_ok());
    assert!(try_solve_edge_problem_from_with_config(&cfg, &problem, &everything, enough).is_ok());
}
