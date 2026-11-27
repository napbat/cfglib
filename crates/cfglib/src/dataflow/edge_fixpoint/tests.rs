use super::*;
use crate::dataflow::solve::{Seeds, SolveError, try_meet_options};
use crate::graph::edge_view::FilteredEdges;
use crate::{BlockId, Cfg, Edge, EdgeKind};

fn seeded(seeds: &[BlockId]) -> Solve<'_, BlockId> {
    Solve {
        seeds: Seeds::Of(seeds),
        ..Solve::default()
    }
}

fn limited(max_steps: usize) -> Solve<'static, BlockId> {
    Solve {
        config: SolveConfig {
            max_steps: Some(max_steps),
            ..SolveConfig::default()
        },
        ..Solve::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Normal,
    Handler(u8),
    Unwind,
}

struct PreAndPost {
    entry: BlockId,
}

impl EdgeProblem<Cfg<(), Route>> for PreAndPost {
    type Fact = u16;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &Cfg<(), Route>) -> Self::Fact {
        0
    }

    fn boundary(&self, _graph: &Cfg<(), Route>, node: BlockId) -> Option<Self::Fact> {
        (node == self.entry).then_some(1)
    }

    fn meet(&self, left: &Self::Fact, right: &Self::Fact) -> Self::Fact {
        left | right
    }

    fn transfer_node(
        &self,
        _graph: &Cfg<(), Route>,
        node: BlockId,
        input: &Self::Fact,
    ) -> Self::Fact {
        if node == self.entry {
            input | 2
        } else {
            *input
        }
    }

    fn transfer_edge(
        &self,
        _graph: &Cfg<(), Route>,
        edge: EdgeRef<'_, BlockId, crate::EdgeId, Edge<Route>>,
        node_input: &Self::Fact,
        node_output: &Self::Fact,
    ) -> Self::Fact {
        match *edge.data().payload() {
            Route::Normal => *node_output,
            Route::Handler(order) => node_input | (1 << (order + 2)),
            Route::Unwind => node_input | (1 << 8),
        }
    }
}

#[test]
fn normal_and_exceptional_edges_receive_different_physical_states() {
    let mut cfg = Cfg::<(), Route>::with_edge_payload();
    let normal_target = cfg.new_block();
    let first_handler = cfg.new_block();
    let second_handler = cfg.new_block();
    let unwind_target = cfg.new_block();
    let entry = cfg.entry();
    let normal = cfg.add_edge(
        entry,
        normal_target,
        Edge::new(EdgeKind::Fallthrough).with_payload(Route::Normal),
    );
    let handler_zero = cfg.add_edge(
        entry,
        first_handler,
        Edge::new(EdgeKind::ExceptionHandler).with_payload(Route::Handler(0)),
    );
    let handler_one = cfg.add_edge(
        entry,
        second_handler,
        Edge::new(EdgeKind::ExceptionHandler).with_payload(Route::Handler(1)),
    );
    let unwind = cfg.add_edge(
        entry,
        unwind_target,
        Edge::new(EdgeKind::ExceptionUnwind).with_payload(Route::Unwind),
    );

    let facts = solve_edge_problem(&cfg, &PreAndPost { entry }, Solve::default()).unwrap();
    assert_eq!(facts.fact_in(entry), &1);
    assert_eq!(facts.fact_out(entry), &3);
    assert_eq!(facts.fact_on(normal), Some(&3));
    assert_eq!(facts.fact_on(handler_zero), Some(&5));
    assert_eq!(facts.fact_on(handler_one), Some(&9));
    assert_eq!(facts.fact_on(unwind), Some(&257));
    assert_eq!(
        cfg.outgoing(entry).collect::<Vec<_>>(),
        [normal, handler_zero, handler_one, unwind]
    );
}

#[test]
fn step_limit_failure_is_structured_and_deterministic() {
    let cfg = Cfg::<(), Route>::with_edge_payload();
    let error =
        solve_edge_problem(&cfg, &PreAndPost { entry: cfg.entry() }, limited(0)).unwrap_err();
    assert_eq!(
        error,
        TrySolveError::Solver(SolveError::StepLimitExceeded {
            limit: 0,
            steps: 0,
            pending_node: 0,
        })
    );
}

/// A frame-verification-shaped analysis: facts exist only where reached,
/// merges take the deeper frame, exceptional edges observe the pre-state.
struct Depths {
    entry: BlockId,
}

impl TryEdgeProblem<Cfg<(), Route>> for Depths {
    type Fact = Option<u16>;
    type Error = &'static str;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &Cfg<(), Route>) -> Option<u16> {
        None
    }

    fn boundary(
        &self,
        _graph: &Cfg<(), Route>,
        node: BlockId,
    ) -> Result<Option<Option<u16>>, &'static str> {
        Ok((node == self.entry).then_some(Some(0)))
    }

    fn meet(
        &self,
        _graph: &Cfg<(), Route>,
        _node: BlockId,
        left: &Option<u16>,
        right: &Option<u16>,
    ) -> Result<Option<u16>, &'static str> {
        try_meet_options(left, right, |left, right| Ok((*left).max(*right)))
    }

    fn transfer_node(
        &self,
        _graph: &Cfg<(), Route>,
        _node: BlockId,
        input: &Option<u16>,
    ) -> Result<Option<u16>, &'static str> {
        input
            .map(|depth| depth.checked_add(1).ok_or("depth overflow"))
            .transpose()
    }

    fn transfer_edge(
        &self,
        _graph: &Cfg<(), Route>,
        edge: EdgeRef<'_, BlockId, crate::EdgeId, Edge<Route>>,
        node_input: &Option<u16>,
        node_output: &Option<u16>,
    ) -> Result<Option<u16>, &'static str> {
        Ok(match edge.data().payload() {
            Route::Unwind => *node_input,
            _ => *node_output,
        })
    }
}

#[test]
fn option_facts_short_circuit_unreached_code_and_observe_pre_states() {
    let mut cfg = Cfg::<(), Route>::with_edge_payload();
    let entry = cfg.entry();
    let body = cfg.new_block();
    let handler = cfg.new_block();
    let unreached = cfg.new_block();
    cfg.new_block(); // dead block: never reached, never transferred
    cfg.add_edge(
        entry,
        body,
        Edge::new(EdgeKind::Fallthrough).with_payload(Route::Normal),
    );
    let unwind = cfg.add_edge(
        body,
        handler,
        Edge::new(EdgeKind::ExceptionUnwind).with_payload(Route::Unwind),
    );
    let normal = cfg.add_edge(
        body,
        handler,
        Edge::new(EdgeKind::Fallthrough).with_payload(Route::Normal),
    );

    let facts =
        solve_edge_problem(&cfg, &Depths { entry }, Solve::default()).expect("no transfer fails");

    assert_eq!(facts.fact_in(entry), &Some(0));
    assert_eq!(facts.fact_out(entry), &Some(1));
    assert_eq!(facts.fact_in(body), &Some(1));
    assert_eq!(facts.fact_out(body), &Some(2));
    assert_eq!(
        facts.fact_on(unwind),
        Some(&Some(1)),
        "the exceptional edge observes the pre-state"
    );
    assert_eq!(
        facts.fact_on(normal),
        Some(&Some(2)),
        "the normal edge observes the post-state"
    );
    assert_eq!(
        facts.fact_in(handler),
        &Some(2),
        "merges take the deeper frame"
    );
    assert_eq!(
        facts.fact_in(unreached),
        &None,
        "unreached code has no fact, not an empty one"
    );
    assert_eq!(facts.fact_out(unreached), &None);
}

#[test]
fn a_failing_transfer_is_a_problem_error() {
    struct Overflowing;
    impl TryEdgeProblem<Cfg<(), Route>> for Overflowing {
        type Fact = Option<u16>;
        type Error = &'static str;
        fn direction(&self) -> Direction {
            Direction::Forward
        }
        fn bottom(&self, _graph: &Cfg<(), Route>) -> Option<u16> {
            None
        }
        fn boundary(
            &self,
            _graph: &Cfg<(), Route>,
            _node: BlockId,
        ) -> Result<Option<Option<u16>>, &'static str> {
            Ok(Some(Some(u16::MAX)))
        }
        fn meet(
            &self,
            _graph: &Cfg<(), Route>,
            _node: BlockId,
            left: &Option<u16>,
            right: &Option<u16>,
        ) -> Result<Option<u16>, &'static str> {
            try_meet_options(left, right, |left, right| Ok((*left).max(*right)))
        }
        fn transfer_node(
            &self,
            _graph: &Cfg<(), Route>,
            _node: BlockId,
            input: &Option<u16>,
        ) -> Result<Option<u16>, &'static str> {
            input
                .map(|depth| depth.checked_add(1).ok_or("depth overflow"))
                .transpose()
        }
    }
    let cfg = Cfg::<(), Route>::with_edge_payload();
    let error = solve_edge_problem(&cfg, &Overflowing, Solve::default()).unwrap_err();
    assert_eq!(error, TrySolveError::Problem("depth overflow"));
}

/// Every node is a boundary, so a visited node is marked.
struct EveryNodeIsBoundary;

impl EdgeProblem<Cfg<(), Route>> for EveryNodeIsBoundary {
    type Fact = u8;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &Cfg<(), Route>) -> Self::Fact {
        0
    }

    fn boundary(&self, _graph: &Cfg<(), Route>, _node: BlockId) -> Option<Self::Fact> {
        Some(1)
    }

    fn meet(&self, left: &Self::Fact, right: &Self::Fact) -> Self::Fact {
        left | right
    }

    fn transfer_node(
        &self,
        _graph: &Cfg<(), Route>,
        _node: BlockId,
        input: &Self::Fact,
    ) -> Self::Fact {
        *input
    }
}

#[test]
fn a_removed_block_is_never_visited() {
    let mut cfg = Cfg::<(), Route>::with_edge_payload();
    let removed = cfg.new_block();
    assert!(cfg.remove_block(removed));

    let facts = solve_edge_problem(&cfg, &EveryNodeIsBoundary, Solve::default()).unwrap();
    assert_eq!(facts.stats().step_count(), cfg.block_count());
    assert_eq!(
        facts.fact_in(removed),
        &0,
        "a removed block's slot stays at bottom"
    );
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn a_removed_seed_panics() {
    let mut cfg = Cfg::<(), Route>::with_edge_payload();
    let removed = cfg.new_block();
    assert!(cfg.remove_block(removed));
    let _ = solve_edge_problem(&cfg, &EveryNodeIsBoundary, seeded(&[removed]));
}

/// `entry` fans out to a normal target, two handlers and an unwind target.
fn routed_cfg() -> (Cfg<(), Route>, BlockId) {
    let mut cfg = Cfg::<(), Route>::with_edge_payload();
    let entry = cfg.entry();
    let routes = [
        (EdgeKind::Fallthrough, Route::Normal),
        (EdgeKind::ExceptionHandler, Route::Handler(0)),
        (EdgeKind::ExceptionHandler, Route::Handler(1)),
        (EdgeKind::ExceptionUnwind, Route::Unwind),
    ];
    for (kind, route) in routes {
        let target = cfg.new_block();
        cfg.add_edge(entry, target, Edge::new(kind).with_payload(route));
    }
    (cfg, entry)
}

#[test]
fn seeding_every_block_is_the_full_solve_and_the_limit_is_exact() {
    let (cfg, entry) = routed_cfg();
    let problem = PreAndPost { entry };
    let all: Vec<BlockId> = cfg.block_ids().collect();
    let full = solve_edge_problem(&cfg, &problem, Solve::default()).unwrap();
    assert_eq!(
        solve_edge_problem(&cfg, &problem, seeded(&all)).unwrap(),
        full
    );

    let steps = full.stats().step_count();
    let live_edges = cfg.edge_ids().count();
    assert_eq!(
        full.stats().peak_count(),
        2 * cfg.block_bound() + live_edges
    );
    for seeds in [Seeds::Live, Seeds::Of(&all)] {
        let exact = Solve {
            seeds,
            ..limited(steps)
        };
        assert!(solve_edge_problem(&cfg, &problem, exact).is_ok());
        let short = Solve {
            seeds,
            ..limited(steps - 1)
        };
        let error = SolveError::from(solve_edge_problem(&cfg, &problem, short).unwrap_err());
        let SolveError::StepLimitExceeded {
            limit, steps: done, ..
        } = error;
        assert_eq!((limit, done), (steps - 1, steps - 1));
    }
}

/// Marks the first node and passes the mark along every edge.
struct MarkFirstForward;

impl<G: EdgeView> EdgeProblem<G> for MarkFirstForward {
    type Fact = u8;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &G) -> Self::Fact {
        0
    }

    fn boundary(&self, _graph: &G, node: G::NodeId) -> Option<Self::Fact> {
        (node.index() == 0).then_some(1)
    }

    fn meet(&self, left: &Self::Fact, right: &Self::Fact) -> Self::Fact {
        left | right
    }

    fn transfer_node(&self, _graph: &G, _node: G::NodeId, input: &Self::Fact) -> Self::Fact {
        *input
    }
}

/// Marks every node without an outgoing edge and passes the mark back along
/// every edge.
struct MarkExitsBackward;

impl<G: EdgeView> EdgeProblem<G> for MarkExitsBackward {
    type Fact = u8;

    fn direction(&self) -> Direction {
        Direction::Backward
    }

    fn bottom(&self, _graph: &G) -> Self::Fact {
        0
    }

    fn boundary(&self, graph: &G, node: G::NodeId) -> Option<Self::Fact> {
        graph.outgoing(node).next().is_none().then_some(1)
    }

    fn meet(&self, left: &Self::Fact, right: &Self::Fact) -> Self::Fact {
        left | right
    }

    fn transfer_node(&self, _graph: &G, _node: G::NodeId, input: &Self::Fact) -> Self::Fact {
        *input
    }
}

#[test]
fn a_backward_solve_carries_the_exit_mark_to_every_ancestor() {
    let mut cfg = Cfg::<(), Route>::with_edge_payload();
    let middle = cfg.new_block();
    let exit = cfg.new_block();
    let island = cfg.new_block();
    let first = cfg.add_edge(
        cfg.entry(),
        middle,
        Edge::new(EdgeKind::Fallthrough).with_payload(Route::Normal),
    );
    let second = cfg.add_edge(
        middle,
        exit,
        Edge::new(EdgeKind::Fallthrough).with_payload(Route::Normal),
    );

    let full = solve_edge_problem(&cfg, &MarkExitsBackward, Solve::default()).unwrap();
    for block in [cfg.entry(), middle, exit, island] {
        assert_eq!(full.fact_in(block), &1, "{block}");
        assert_eq!(full.fact_out(block), &1, "{block}");
    }
    assert_eq!(full.fact_on(first), Some(&1));
    assert_eq!(full.fact_on(second), Some(&1));

    // Seeding the exit alone carries the change up through its ancestors
    // and never visits the island.
    let from_exit = solve_edge_problem(&cfg, &MarkExitsBackward, seeded(&[exit])).unwrap();
    assert_eq!(from_exit.fact_in(cfg.entry()), &1);
    assert_eq!(from_exit.fact_in(island), &0);
    assert_eq!(from_exit.stats().step_count(), 3);
}

#[test]
fn a_filtered_view_solves_without_its_excluded_edges() {
    let mut cfg = Cfg::<(), Route>::with_edge_payload();
    let kept_target = cfg.new_block();
    let cut_target = cfg.new_block();
    let kept = cfg.add_edge(
        cfg.entry(),
        kept_target,
        Edge::new(EdgeKind::Fallthrough).with_payload(Route::Normal),
    );
    let cut = cfg.add_edge(
        cfg.entry(),
        cut_target,
        Edge::new(EdgeKind::Fallthrough).with_payload(Route::Normal),
    );
    let view = FilteredEdges::new(&cfg, |edge: crate::EdgeId, _: &Edge<Route>| edge != cut);

    let facts = solve_edge_problem(&view, &MarkFirstForward, Solve::default()).unwrap();
    assert_eq!(facts.fact_in(kept_target), &1);
    assert_eq!(facts.fact_in(cut_target), &0, "its only edge is excluded");
    assert_eq!(facts.fact_on(kept), Some(&1));
    assert_eq!(facts.fact_on(cut), None, "an excluded edge carries no fact");
}
