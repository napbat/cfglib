extern crate alloc;

use alloc::collections::BTreeSet;

use super::*;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::liveness::LivenessProblem;
use crate::edge::EdgeKind;
use crate::test_util::{DfInst, df_def, df_use};

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

#[test]
fn full_solve_reaches_fixpoint_and_counts_steps() {
    let (cfg, a, _) = liveness_fixture();
    let facts = solve_problem(&cfg, &LivenessProblem).unwrap();
    assert!(facts.fact_in(a).contains(&0), "x live through a");
    assert!(facts.steps() > 0);
}

#[test]
fn seeding_every_block_matches_the_full_solve() {
    let (cfg, _, _) = liveness_fixture();
    let full = solve_problem(&cfg, &LivenessProblem).unwrap();
    let all: alloc::vec::Vec<BlockId> = cfg.block_ids().collect();
    let seeded = solve_problem_from(&cfg, &LivenessProblem, &all).unwrap();
    for block in &all {
        assert_eq!(seeded.fact_in(*block), full.fact_in(*block));
        assert_eq!(seeded.fact_out(*block), full.fact_out(*block));
    }
}

#[test]
fn no_seeds_leaves_every_fact_at_bottom() {
    let (cfg, a, _) = liveness_fixture();
    let facts = solve_problem_from(&cfg, &LivenessProblem, &[]).unwrap();
    assert!(facts.fact_in(a).is_empty());
    assert_eq!(facts.steps(), 0);
}

#[test]
fn step_limit_reports_the_pending_block() {
    let (cfg, _, _) = liveness_fixture();
    let error = solve_problem_with_config(&cfg, &LivenessProblem, SolveConfig::with_step_limit(1))
        .unwrap_err();
    let SolveError::StepLimitExceeded { limit, steps, .. } = error;
    assert_eq!(limit, 1);
    assert_eq!(steps, 1);
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn an_out_of_range_seed_panics() {
    let (cfg, _, _) = liveness_fixture();
    let beyond = BlockId::from_index(cfg.block_bound());
    let _ = solve_problem_from(&cfg, &LivenessProblem, &[beyond]);
}

/// A fallible liveness clone that rejects any block defining variable 7.
struct Rejecting;

impl TryProblem<Cfg<DfInst>> for Rejecting {
    type Fact = BTreeSet<u16>;
    type Error = &'static str;

    fn direction(&self) -> Direction {
        Direction::Backward
    }

    fn bottom(&self) -> Self::Fact {
        BTreeSet::new()
    }

    fn entry_fact(&self) -> Result<Self::Fact, Self::Error> {
        Ok(BTreeSet::new())
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Result<Self::Fact, Self::Error> {
        Ok(a.union(b).copied().collect())
    }

    fn transfer(
        &self,
        cfg: &Cfg<DfInst>,
        block: BlockId,
        input: &Self::Fact,
    ) -> Result<Self::Fact, Self::Error> {
        use crate::dataflow::InstrInfo;
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
    let error = try_solve_problem(&cfg, &Rejecting).unwrap_err();
    assert_eq!(error, TrySolveError::Problem("variable 7 is reserved"));
}

#[test]
fn a_fallible_solve_succeeds_on_clean_input() {
    let (cfg, a, _) = liveness_fixture();
    let facts = try_solve_problem(&cfg, &Rejecting).unwrap();
    assert!(facts.fact_in(a).is_empty());
}

#[test]
fn backward_solver_propagates_through_disconnected_components() {
    struct BackwardReachability;

    impl Problem<Cfg<()>> for BackwardReachability {
        type Fact = bool;

        fn direction(&self) -> Direction {
            Direction::Backward
        }

        fn bottom(&self) -> Self::Fact {
            false
        }

        fn entry_fact(&self) -> Self::Fact {
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

    let facts = solve_problem(&cfg, &BackwardReachability).unwrap();
    assert!(*facts.fact_in(disconnected_predecessor));
    assert!(*facts.fact_out(disconnected_predecessor));
    assert!(*facts.fact_in(disconnected_exit));
    assert!(*facts.fact_out(disconnected_exit));
}

/// `true` exactly where flow from the entry fact arrives.
struct EntryFlows;

impl Problem<Cfg<()>> for EntryFlows {
    type Fact = bool;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self) -> Self::Fact {
        false
    }

    fn entry_fact(&self) -> Self::Fact {
        true
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
        *a || *b
    }

    fn transfer(&self, _cfg: &Cfg<()>, _block: BlockId, input: &Self::Fact) -> Self::Fact {
        *input
    }
}

#[test]
fn a_forward_entry_that_is_a_branch_target_still_receives_the_entry_fact() {
    let mut cfg = Cfg::new();
    let body = cfg.new_block();
    cfg.add_edge(cfg.entry(), body, EdgeKind::Fallthrough);
    cfg.add_edge(body, cfg.entry(), EdgeKind::Back);

    let facts = solve_problem(&cfg, &EntryFlows).unwrap();
    assert!(*facts.fact_in(cfg.entry()));
    assert!(*facts.fact_in(body));
}

fn assert_same_facts(
    blocks: &[BlockId],
    expected: &Facts<BTreeSet<u16>>,
    actual: &Facts<BTreeSet<u16>>,
) {
    assert_eq!(actual.steps(), expected.steps());
    for &block in blocks {
        assert_eq!(actual.fact_in(block), expected.fact_in(block));
        assert_eq!(actual.fact_out(block), expected.fact_out(block));
    }
}

fn solver_error<F>(result: Result<F, TrySolveError<Infallible>>) -> SolveError {
    match result {
        Err(TrySolveError::Solver(error)) => error,
        Err(TrySolveError::Problem(never)) => match never {},
        Ok(_) => panic!("the solve stayed within its step limit"),
    }
}

#[test]
fn every_entry_point_of_the_matrix_reaches_the_same_fixpoint() {
    let (cfg, ..) = liveness_fixture();
    let blocks: alloc::vec::Vec<BlockId> = cfg.block_ids().collect();
    let config = SolveConfig::new();
    let infallible = InfallibleProblem(&LivenessProblem);
    let full = solve_problem(&cfg, &LivenessProblem).unwrap();

    let variants = [
        solve_problem_with_config(&cfg, &LivenessProblem, config).unwrap(),
        solve_problem_from(&cfg, &LivenessProblem, &blocks).unwrap(),
        solve_problem_from_with_config(&cfg, &LivenessProblem, &blocks, config).unwrap(),
        try_solve_problem(&cfg, &infallible).unwrap(),
        try_solve_problem_with_config(&cfg, &infallible, config).unwrap(),
        try_solve_problem_from(&cfg, &infallible, &blocks).unwrap(),
        try_solve_problem_from_with_config(&cfg, &infallible, &blocks, config).unwrap(),
    ];
    for variant in &variants {
        assert_same_facts(&blocks, &full, variant);
    }
}

#[test]
fn every_entry_point_of_the_matrix_enforces_the_step_limit() {
    let (cfg, ..) = liveness_fixture();
    let blocks: alloc::vec::Vec<BlockId> = cfg.block_ids().collect();
    let infallible = InfallibleProblem(&LivenessProblem);
    let steps = solve_problem(&cfg, &LivenessProblem).unwrap().steps();
    let exact = SolveConfig::with_step_limit(steps);
    let short = SolveConfig::with_step_limit(steps - 1);

    assert!(solve_problem_with_config(&cfg, &LivenessProblem, exact).is_ok());
    assert!(solve_problem_from_with_config(&cfg, &LivenessProblem, &blocks, exact).is_ok());
    assert!(try_solve_problem_with_config(&cfg, &infallible, exact).is_ok());
    assert!(try_solve_problem_from_with_config(&cfg, &infallible, &blocks, exact).is_ok());

    let errors = [
        solve_problem_with_config(&cfg, &LivenessProblem, short).unwrap_err(),
        solve_problem_from_with_config(&cfg, &LivenessProblem, &blocks, short).unwrap_err(),
        solver_error(try_solve_problem_with_config(&cfg, &infallible, short)),
        solver_error(try_solve_problem_from_with_config(
            &cfg,
            &infallible,
            &blocks,
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

#[test]
fn a_backward_seeded_solve_visits_only_what_its_seed_reaches() {
    let (cfg, a, b) = liveness_fixture();
    let full = solve_problem(&cfg, &LivenessProblem).unwrap();
    assert!(full.fact_in(a).contains(&0), "b reads x, so x is live in a");

    // From `b` the change flows up through every predecessor.
    let from_exit = solve_problem_from(&cfg, &LivenessProblem, &[b]).unwrap();
    for block in [cfg.entry(), a, b] {
        assert_eq!(from_exit.fact_in(block), full.fact_in(block), "{block}");
        assert_eq!(from_exit.fact_out(block), full.fact_out(block), "{block}");
    }
    assert_eq!(from_exit.steps(), 3, "the exit, then each ancestor once");

    // From `a` alone nothing downstream of it is ever visited: `b` stays at
    // bottom, so `a` sees nothing live and the entry is never queued.
    let from_middle = solve_problem_from(&cfg, &LivenessProblem, &[a]).unwrap();
    assert_eq!(from_middle.steps(), 1);
    assert!(from_middle.fact_in(a).is_empty());
    assert!(from_middle.fact_in(b).is_empty());
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn a_removed_seed_panics() {
    let (mut cfg, a, _) = liveness_fixture();
    assert!(cfg.remove_block(a));
    let _ = solve_problem_from(&cfg, &LivenessProblem, &[a]);
}

#[test]
fn a_removed_block_is_never_visited() {
    let (mut cfg, a, b) = liveness_fixture();
    assert!(cfg.remove_block(a));

    let facts = solve_problem(&cfg, &LivenessProblem).unwrap();
    assert_eq!(facts.steps(), cfg.block_count());
    assert!(
        facts.fact_in(a).is_empty(),
        "the slot of a removed block stays at bottom"
    );
    assert!(facts.fact_in(b).contains(&0));
}
