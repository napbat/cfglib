use super::*;
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
