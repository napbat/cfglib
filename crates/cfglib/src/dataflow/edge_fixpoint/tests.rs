use super::*;
use crate::graph::edge_view::FilteredEdges;
use crate::{BlockId, Cfg, Edge, EdgeKind};

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
    let normal =
        cfg.add_edge_with_payload(entry, normal_target, EdgeKind::Fallthrough, Route::Normal);
    let handler_zero = cfg.add_edge_with_payload(
        entry,
        first_handler,
        EdgeKind::ExceptionHandler,
        Route::Handler(0),
    );
    let handler_one = cfg.add_edge_with_payload(
        entry,
        second_handler,
        EdgeKind::ExceptionHandler,
        Route::Handler(1),
    );
    let unwind = cfg.add_edge_with_payload(
        entry,
        unwind_target,
        EdgeKind::ExceptionUnwind,
        Route::Unwind,
    );

    let facts = solve_edge_problem(&cfg, &PreAndPost { entry }).unwrap();
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
    let error = solve_edge_problem_with_config(
        &cfg,
        &PreAndPost { entry: cfg.entry() },
        SolveConfig::with_step_limit(0),
    )
    .unwrap_err();
    assert_eq!(
        error,
        SolveError::StepLimitExceeded {
            limit: 0,
            steps: 0,
            pending_node: 0,
        }
    );
}

/// A frame-verification-shaped analysis: facts exist only where reached,
/// merges add, exceptional edges observe the pre-state.
struct Depths {
    entry: BlockId,
}

impl ReachableEdgeProblem<Cfg<(), Route>> for Depths {
    type Fact = u16;
    type Error = &'static str;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn entry_fact(
        &self,
        _graph: &Cfg<(), Route>,
        node: BlockId,
    ) -> Result<Option<u16>, &'static str> {
        Ok((node == self.entry).then_some(0))
    }

    fn merge(
        &self,
        _graph: &Cfg<(), Route>,
        _node: BlockId,
        left: &u16,
        right: &u16,
    ) -> Result<u16, &'static str> {
        Ok((*left).max(*right))
    }

    fn transfer(
        &self,
        _graph: &Cfg<(), Route>,
        _node: BlockId,
        input: &u16,
    ) -> Result<u16, &'static str> {
        input.checked_add(1).ok_or("depth overflow")
    }

    fn edge_observes_input(
        &self,
        _graph: &Cfg<(), Route>,
        edge: crate::EdgeRef<'_, BlockId, crate::EdgeId, Edge<Route>>,
    ) -> bool {
        matches!(edge.data().payload(), Route::Unwind)
    }
}

#[test]
fn reachable_lifting_short_circuits_and_observes_pre_states() {
    let mut cfg = Cfg::<(), Route>::with_edge_payload();
    let entry = cfg.entry();
    let body = cfg.new_block();
    let handler = cfg.new_block();
    let unreached = cfg.new_block();
    cfg.new_block(); // dead block: never reached, never transferred
    let _ = unreached;
    cfg.add_edge_with_payload(entry, body, EdgeKind::Fallthrough, Route::Normal);
    let unwind = cfg.add_edge_with_payload(body, handler, EdgeKind::ExceptionUnwind, Route::Unwind);
    let normal = cfg.add_edge_with_payload(body, handler, EdgeKind::Fallthrough, Route::Normal);

    let facts =
        try_solve_edge_problem(&cfg, &Reachable(Depths { entry })).expect("no transfer fails");

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
        facts.fact_in(unreached),
        &None,
        "unreached code has no fact, not an empty one"
    );
    assert_eq!(facts.fact_out(unreached), &None);
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

    let facts = solve_edge_problem(&cfg, &EveryNodeIsBoundary).unwrap();
    assert_eq!(facts.steps(), cfg.block_count());
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
    let _ = solve_edge_problem_from(&cfg, &EveryNodeIsBoundary, &[removed]);
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
        cfg.add_edge_with_payload(entry, target, kind, route);
    }
    (cfg, entry)
}

fn solver_error<F>(result: Result<F, TrySolveError<Infallible>>) -> SolveError {
    match result.err() {
        Some(TrySolveError::Solver(error)) => error,
        Some(TrySolveError::Problem(never)) => match never {},
        None => panic!("the solve stayed within its step limit"),
    }
}

#[test]
fn every_entry_point_of_the_matrix_reaches_the_same_fixpoint() {
    let (cfg, entry) = routed_cfg();
    let problem = PreAndPost { entry };
    let infallible = InfallibleProblem(&problem);
    let all: Vec<BlockId> = cfg.block_ids().collect();
    let config = SolveConfig::new();
    let full = solve_edge_problem(&cfg, &problem).unwrap();

    assert_eq!(
        solve_edge_problem_with_config(&cfg, &problem, config).unwrap(),
        full
    );
    assert_eq!(solve_edge_problem_from(&cfg, &problem, &all).unwrap(), full);
    assert_eq!(
        solve_edge_problem_from_with_config(&cfg, &problem, &all, config).unwrap(),
        full
    );
    assert_eq!(try_solve_edge_problem(&cfg, &infallible).unwrap(), full);
    assert_eq!(
        try_solve_edge_problem_with_config(&cfg, &infallible, config).unwrap(),
        full
    );
    assert_eq!(
        try_solve_edge_problem_from(&cfg, &infallible, &all).unwrap(),
        full
    );
    assert_eq!(
        try_solve_edge_problem_from_with_config(&cfg, &infallible, &all, config).unwrap(),
        full
    );
}

#[test]
fn every_entry_point_of_the_matrix_enforces_the_step_limit() {
    let (cfg, entry) = routed_cfg();
    let problem = PreAndPost { entry };
    let infallible = InfallibleProblem(&problem);
    let all: Vec<BlockId> = cfg.block_ids().collect();
    let steps = solve_edge_problem(&cfg, &problem).unwrap().steps();
    let exact = SolveConfig::with_step_limit(steps);
    let short = SolveConfig::with_step_limit(steps - 1);

    assert!(solve_edge_problem_with_config(&cfg, &problem, exact).is_ok());
    assert!(solve_edge_problem_from_with_config(&cfg, &problem, &all, exact).is_ok());
    assert!(try_solve_edge_problem_with_config(&cfg, &infallible, exact).is_ok());
    assert!(try_solve_edge_problem_from_with_config(&cfg, &infallible, &all, exact).is_ok());

    let errors = [
        solve_edge_problem_with_config(&cfg, &problem, short).unwrap_err(),
        solve_edge_problem_from_with_config(&cfg, &problem, &all, short).unwrap_err(),
        solver_error(try_solve_edge_problem_with_config(&cfg, &infallible, short)),
        solver_error(try_solve_edge_problem_from_with_config(
            &cfg,
            &infallible,
            &all,
            short,
        )),
    ];
    let SolveError::StepLimitExceeded {
        limit, steps: done, ..
    } = errors[0];
    assert_eq!((limit, done), (steps - 1, steps - 1));
    for error in errors {
        assert_eq!(error, errors[0], "every path reports the same pending node");
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
    let first =
        cfg.add_edge_with_payload(cfg.entry(), middle, EdgeKind::Fallthrough, Route::Normal);
    let second = cfg.add_edge_with_payload(middle, exit, EdgeKind::Fallthrough, Route::Normal);

    let full = solve_edge_problem(&cfg, &MarkExitsBackward).unwrap();
    for block in [cfg.entry(), middle, exit, island] {
        assert_eq!(full.fact_in(block), &1, "{block}");
        assert_eq!(full.fact_out(block), &1, "{block}");
    }
    assert_eq!(full.fact_on(first), Some(&1));
    assert_eq!(full.fact_on(second), Some(&1));

    // Seeding the exit alone carries the change up through its ancestors
    // and never visits the island.
    let from_exit = solve_edge_problem_from(&cfg, &MarkExitsBackward, &[exit]).unwrap();
    assert_eq!(from_exit.fact_in(cfg.entry()), &1);
    assert_eq!(from_exit.fact_in(island), &0);
    assert_eq!(from_exit.steps(), 3);
}

#[test]
fn a_filtered_view_solves_without_its_excluded_edges() {
    let mut cfg = Cfg::<(), Route>::with_edge_payload();
    let kept_target = cfg.new_block();
    let cut_target = cfg.new_block();
    let kept = cfg.add_edge_with_payload(
        cfg.entry(),
        kept_target,
        EdgeKind::Fallthrough,
        Route::Normal,
    );
    let cut = cfg.add_edge_with_payload(
        cfg.entry(),
        cut_target,
        EdgeKind::Fallthrough,
        Route::Normal,
    );
    let view = FilteredEdges::new(&cfg, |edge: crate::EdgeId, _: &Edge<Route>| edge != cut);

    let facts = solve_edge_problem(&view, &MarkFirstForward).unwrap();
    assert_eq!(facts.fact_in(kept_target), &1);
    assert_eq!(facts.fact_in(cut_target), &0, "its only edge is excluded");
    assert_eq!(facts.fact_on(kept), Some(&1));
    assert_eq!(facts.fact_on(cut), None, "an excluded edge carries no fact");
}
