extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::*;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::node_fixpoint::{NodeProblem, TryNodeProblem, solve_node_problem};
use crate::edge::EdgeKind;
use crate::graph::view::RootedView;
use crate::test_util::{DfInst, LiveVariables, df_def, df_use};

/// entry → a → b, with `x` defined in entry and used in b.
fn liveness_fixture() -> (Cfg<DfInst>, BlockId, BlockId) {
    let mut cfg: Cfg<DfInst> = Cfg::new();
    let a = cfg.new_block();
    let b = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(df_def("def_x", 0));
    cfg.block_mut(b).push(df_use("use_x", 0));
    cfg.add_edge(cfg.entry(), a, EdgeKind::Fallthrough);
    cfg.add_edge(a, b, EdgeKind::Fallthrough);
    (cfg, a, b)
}

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

#[test]
fn full_solve_reaches_fixpoint_and_counts_steps() {
    let (cfg, a, _) = liveness_fixture();
    let facts = solve_node_problem(&cfg, &LiveVariables, Solve::default()).unwrap();
    assert!(facts.fact_in(a).contains(&0), "x live through a");
    assert_eq!(
        facts.stats().step_count(),
        5,
        "three blocks, then b's use climbs twice"
    );
    assert_eq!(facts.stats().round_count(), 0);
    assert_eq!(facts.stats().peak_count(), 2 * cfg.block_bound());
}

#[test]
fn seeding_every_block_matches_the_full_solve() {
    let (cfg, _, _) = liveness_fixture();
    let full = solve_node_problem(&cfg, &LiveVariables, Solve::default()).unwrap();
    let all: Vec<BlockId> = cfg.block_ids().collect();
    let from_all = solve_node_problem(&cfg, &LiveVariables, seeded(&all)).unwrap();
    assert_eq!(from_all, full);
}

#[test]
fn no_seeds_leaves_every_fact_at_bottom() {
    let (cfg, a, _) = liveness_fixture();
    let facts = solve_node_problem(&cfg, &LiveVariables, seeded(&[])).unwrap();
    assert!(facts.fact_in(a).is_empty());
    assert_eq!(facts.stats().step_count(), 0);
}

#[test]
fn the_step_limit_is_exact_and_names_the_pending_block_for_every_seed_kind() {
    let (cfg, ..) = liveness_fixture();
    let all: Vec<BlockId> = cfg.block_ids().collect();
    let steps = solve_node_problem(&cfg, &LiveVariables, Solve::default())
        .unwrap()
        .stats()
        .step_count();
    for seeds in [Seeds::Live, Seeds::Of(&all)] {
        let exact = Solve {
            seeds,
            ..limited(steps)
        };
        assert!(solve_node_problem(&cfg, &LiveVariables, exact).is_ok());
        let short = Solve {
            seeds,
            ..limited(steps - 1)
        };
        let error = SolveError::from(solve_node_problem(&cfg, &LiveVariables, short).unwrap_err());
        let SolveError::StepLimitExceeded {
            limit,
            steps: done,
            pending_node,
        } = error;
        assert_eq!((limit, done), (steps - 1, steps - 1));
        assert_eq!(pending_node, cfg.entry().index(), "the last block to run");
    }
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn an_out_of_range_seed_panics() {
    let (cfg, _, _) = liveness_fixture();
    let beyond = [BlockId::from_index(cfg.block_bound())];
    let _ = solve_node_problem(&cfg, &LiveVariables, seeded(&beyond));
}

/// A fallible liveness clone that rejects any block defining variable 7.
struct Rejecting;

impl TryNodeProblem<Cfg<DfInst>> for Rejecting {
    type Fact = BTreeSet<u16>;
    type Error = &'static str;

    fn direction(&self) -> Direction {
        Direction::Backward
    }

    fn bottom(&self, _cfg: &Cfg<DfInst>) -> Self::Fact {
        BTreeSet::new()
    }

    fn boundary(&self, _cfg: &Cfg<DfInst>) -> Result<Self::Fact, Self::Error> {
        Ok(BTreeSet::new())
    }

    fn meet(
        &self,
        _cfg: &Cfg<DfInst>,
        _block: BlockId,
        a: &Self::Fact,
        b: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        Ok(a.union(b).copied().collect())
    }

    fn transfer(
        &self,
        cfg: &Cfg<DfInst>,
        block: BlockId,
        input: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        use crate::InstrInfo;
        for inst in cfg.block(block).instructions() {
            if inst.defs().contains(&7) {
                return Err("variable 7 is reserved");
            }
        }
        Ok(input.clone())
    }
}

#[test]
fn a_consumer_error_is_reported_as_a_problem_error() {
    let mut cfg: Cfg<DfInst> = Cfg::new();
    cfg.block_mut(cfg.entry()).push(df_def("bad", 7));
    let error = solve_node_problem(&cfg, &Rejecting, Solve::default()).unwrap_err();
    assert_eq!(error, TrySolveError::Problem("variable 7 is reserved"));
}

#[test]
fn a_fallible_solve_succeeds_on_clean_input() {
    let (cfg, a, _) = liveness_fixture();
    let facts = solve_node_problem(&cfg, &Rejecting, Solve::default()).unwrap();
    assert!(facts.fact_in(a).is_empty());
}

#[test]
fn backward_solver_propagates_through_disconnected_components() {
    struct BackwardReachability;

    impl NodeProblem<Cfg<()>> for BackwardReachability {
        type Fact = bool;

        fn direction(&self) -> Direction {
            Direction::Backward
        }

        fn bottom(&self, _cfg: &Cfg<()>) -> Self::Fact {
            false
        }

        fn boundary(&self, _cfg: &Cfg<()>) -> Self::Fact {
            true
        }

        fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
            *a || *b
        }

        fn transfer(&self, _cfg: &Cfg<()>, _block: BlockId, input: &Self::Fact) -> Self::Fact {
            *input
        }
    }

    let mut cfg = Cfg::new();
    let disconnected_predecessor = cfg.new_block();
    let disconnected_exit = cfg.new_block();
    cfg.add_edge(
        disconnected_predecessor,
        disconnected_exit,
        EdgeKind::Fallthrough,
    );

    let facts = solve_node_problem(&cfg, &BackwardReachability, Solve::default()).unwrap();
    assert!(*facts.fact_in(disconnected_predecessor));
    assert!(*facts.fact_out(disconnected_predecessor));
    assert!(*facts.fact_in(disconnected_exit));
    assert!(*facts.fact_out(disconnected_exit));
}

/// `true` exactly where flow from the entry fact arrives.
struct EntryFlows;

impl NodeProblem<Cfg<()>> for EntryFlows {
    type Fact = bool;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _cfg: &Cfg<()>) -> Self::Fact {
        false
    }

    fn boundary(&self, _cfg: &Cfg<()>) -> Self::Fact {
        true
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
        *a || *b
    }

    fn transfer(&self, _cfg: &Cfg<()>, _block: BlockId, input: &Self::Fact) -> Self::Fact {
        *input
    }
}

fn rooted(cfg: &Cfg<()>) -> Solve<'static, BlockId> {
    Solve {
        seeds: Seeds::Reachable(cfg.root()),
        ..Solve::default()
    }
}

#[test]
fn a_reachable_root_that_is_a_branch_target_still_receives_the_boundary_fact() {
    let mut cfg = Cfg::new();
    let body = cfg.new_block();
    cfg.add_edge(cfg.entry(), body, EdgeKind::Fallthrough);
    cfg.add_edge(body, cfg.entry(), EdgeKind::Back);

    let facts = solve_node_problem(&cfg, &EntryFlows, rooted(&cfg)).unwrap();
    assert!(*facts.fact_in(cfg.entry()));
    assert!(*facts.fact_in(body));

    // Every live node instead: the entry has an upstream neighbor, so nothing
    // receives the boundary fact and nothing flows.
    let live = solve_node_problem(&cfg, &EntryFlows, Solve::default()).unwrap();
    assert!(!*live.fact_in(cfg.entry()));
}

#[test]
fn a_reachable_solve_visits_only_what_the_root_reaches() {
    // entry → reached; orphan → fed has no path from the entry. Every live
    // node would give the orphan the boundary fact; the rooted solve never
    // queues it, so it and what it feeds stay at bottom.
    let mut cfg = Cfg::new();
    let reached = cfg.new_block();
    let orphan = cfg.new_block();
    let fed = cfg.new_block();
    cfg.add_edge(cfg.entry(), reached, EdgeKind::Fallthrough);
    cfg.add_edge(orphan, fed, EdgeKind::Fallthrough);

    let facts = solve_node_problem(&cfg, &EntryFlows, rooted(&cfg)).unwrap();
    assert!(*facts.fact_out(reached));
    assert!(!*facts.fact_in(orphan) && !*facts.fact_out(fed));
    assert_eq!(facts.stats().step_count(), 2);

    let live = solve_node_problem(&cfg, &EntryFlows, Solve::default()).unwrap();
    assert!(*live.fact_in(orphan) && *live.fact_out(fed));
}

#[test]
fn a_backward_reachable_solve_walks_against_the_edges() {
    // Liveness rooted at the exit `b` covers what reaches `b`: everything.
    let (cfg, a, b) = liveness_fixture();
    let full = solve_node_problem(&cfg, &LiveVariables, Solve::default()).unwrap();
    let from_exit = Solve {
        seeds: Seeds::Reachable(b),
        ..Solve::default()
    };
    let facts = solve_node_problem(&cfg, &LiveVariables, from_exit).unwrap();
    assert_eq!(facts.fact_in(a), full.fact_in(a));
    assert_eq!(facts.fact_in(cfg.entry()), full.fact_in(cfg.entry()));
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn a_removed_root_panics() {
    let mut cfg: Cfg<()> = Cfg::new();
    let gone = cfg.new_block();
    assert!(cfg.remove_block(gone));
    let solve = Solve {
        seeds: Seeds::Reachable(gone),
        ..Solve::default()
    };
    let _ = solve_node_problem(&cfg, &EntryFlows, solve);
}

#[test]
fn a_backward_seeded_solve_visits_only_what_its_seed_reaches() {
    let (cfg, a, b) = liveness_fixture();
    let full = solve_node_problem(&cfg, &LiveVariables, Solve::default()).unwrap();
    assert!(full.fact_in(a).contains(&0), "b reads x, so x is live in a");

    // From `b` the change flows up through every predecessor.
    let from_exit = solve_node_problem(&cfg, &LiveVariables, seeded(&[b])).unwrap();
    for block in [cfg.entry(), a, b] {
        assert_eq!(from_exit.fact_in(block), full.fact_in(block), "{block}");
        assert_eq!(from_exit.fact_out(block), full.fact_out(block), "{block}");
    }
    assert_eq!(
        from_exit.stats().step_count(),
        3,
        "the exit, then each ancestor once"
    );

    // From `a` alone nothing downstream of it is ever visited: `b` stays at
    // bottom, so `a` sees nothing live and the entry is never queued.
    let from_middle = solve_node_problem(&cfg, &LiveVariables, seeded(&[a])).unwrap();
    assert_eq!(from_middle.stats().step_count(), 1);
    assert!(from_middle.fact_in(a).is_empty());
    assert!(from_middle.fact_in(b).is_empty());
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn a_removed_seed_panics() {
    let (mut cfg, a, _) = liveness_fixture();
    assert!(cfg.remove_block(a));
    let _ = solve_node_problem(&cfg, &LiveVariables, seeded(&[a]));
}

#[test]
fn a_removed_block_is_never_visited() {
    let (mut cfg, a, b) = liveness_fixture();
    assert!(cfg.remove_block(a));

    let facts = solve_node_problem(&cfg, &LiveVariables, Solve::default()).unwrap();
    assert_eq!(facts.stats().step_count(), cfg.block_count());
    assert!(
        facts.fact_in(a).is_empty(),
        "the slot of a removed block stays at bottom"
    );
    assert!(facts.fact_in(b).contains(&0));
}

#[test]
fn program_order_puts_what_a_backward_analysis_transfers_at_the_start() {
    assert_eq!(
        Direction::Forward.program_order("receives", "transfers"),
        ("receives", "transfers")
    );
    assert_eq!(
        Direction::Backward.program_order("receives", "transfers"),
        ("transfers", "receives")
    );
}

#[test]
fn counting_steps_takes_all_or_none_and_names_the_pending_node() {
    let mut steps = 3;
    assert_eq!(count_steps(Some(5), &mut steps, 2, 9), Ok(()));
    assert_eq!(steps, 5);
    assert_eq!(
        count_steps(Some(5), &mut steps, 1, 9),
        Err(SolveError::StepLimitExceeded {
            limit: 5,
            steps: 5,
            pending_node: 9,
        })
    );
    let mut steps = 4;
    assert!(count_steps(Some(5), &mut steps, 2, 1).is_err());
    assert_eq!(steps, 4, "a refused batch counts nothing");
    assert_eq!(count_step(None, &mut steps, 0), Ok(()));
    assert_eq!(steps, 5);
}

#[test]
fn meeting_options_treats_none_as_the_identity_in_both_forms() {
    let sum = |a: &u32, b: &u32| a + b;
    for fact in [None, Some(3)] {
        assert_eq!(meet_options(&None, &fact, sum), fact);
        assert_eq!(meet_options(&fact, &None, sum), fact);
    }
    assert_eq!(meet_options(&Some(3), &Some(4), sum), Some(7));

    let checked = |a: &u32, b: &u32| a.checked_add(*b).ok_or("overflow");
    assert_eq!(
        try_meet_options(&None, &Some(u32::MAX), checked),
        Ok(Some(u32::MAX))
    );
    assert_eq!(try_meet_options(&Some(1), &Some(2), checked), Ok(Some(3)));
    assert_eq!(
        try_meet_options(&Some(u32::MAX), &Some(1), checked),
        Err("overflow"),
        "only two reached facts reach the inner meet"
    );
}

#[test]
fn an_infallible_solver_error_converts_into_a_solve_error() {
    let limit = SolveError::StepLimitExceeded {
        limit: 1,
        steps: 1,
        pending_node: 0,
    };
    assert_eq!(SolveError::from(TrySolveError::Solver(limit)), limit);
}
