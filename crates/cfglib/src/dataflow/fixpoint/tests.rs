extern crate alloc;

use alloc::collections::BTreeSet;

use super::*;
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
#[should_panic(expected = "seed block is out of range")]
fn an_out_of_range_seed_panics() {
    let (cfg, _, _) = liveness_fixture();
    let beyond = BlockId::from_index(cfg.block_count());
    let _ = solve_problem_from(&cfg, &LivenessProblem, &[beyond]);
}

/// A fallible liveness clone that rejects any block defining variable 7.
struct Rejecting;

impl TryProblem<DfInst> for Rejecting {
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

    impl Problem<()> for BackwardReachability {
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
