//! The block-level solver (`solve_problem` and its seeded, bounded, and
//! fallible forms) against a round-robin reference, plus the documented
//! contracts of seeds, step limits, and consumer errors.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::fmt;

use cfglib::{
    BlockId, Cfg, DenseId, Direction, Problem, RootedView, SolveConfig, SolveError, TryProblem,
    TrySolveError, meet_options, solve_problem, solve_problem_from, solve_problem_from_with_config,
    solve_problem_with_config, try_solve_problem, try_solve_problem_from,
    try_solve_problem_from_with_config, try_solve_problem_with_config,
};

use crate::graphs::{GenKill, Rng, Shape, intersection, union};
use crate::reference::round_robin;

/// The fact an analysis starts a boundary block with; no block generates or
/// kills it, so it is visible exactly where the boundary reaches.
const BOUNDARY_MARK: u8 = 6;

fn boundary() -> BTreeSet<u8> {
    BTreeSet::from([BOUNDARY_MARK])
}

/// A gen/kill problem whose meet is union: reaching-definitions shaped.
struct May<'a> {
    effects: &'a GenKill,
    direction: Direction,
}

impl<V: RootedView> Problem<V> for May<'_> {
    type Fact = BTreeSet<u8>;

    fn direction(&self) -> Direction {
        self.direction
    }

    fn bottom(&self) -> Self::Fact {
        BTreeSet::new()
    }

    fn entry_fact(&self) -> Self::Fact {
        boundary()
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
        union(a, b)
    }

    fn transfer(&self, _view: &V, block: V::NodeId, input: &Self::Fact) -> Self::Fact {
        self.effects.apply(block.index(), input)
    }
}

/// A gen/kill problem whose meet is intersection, lifted over `Option` with
/// `meet_options` so that `None` is "no path has reached here yet".
struct Must<'a> {
    effects: &'a GenKill,
}

impl<V: RootedView> Problem<V> for Must<'_> {
    type Fact = Option<BTreeSet<u8>>;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self) -> Self::Fact {
        None
    }

    fn entry_fact(&self) -> Self::Fact {
        Some(boundary())
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
        meet_options(a, b, intersection)
    }

    fn transfer(&self, _view: &V, block: V::NodeId, input: &Self::Fact) -> Self::Fact {
        input
            .as_ref()
            .map(|facts| self.effects.apply(block.index(), facts))
    }
}

fn id(index: usize) -> BlockId {
    BlockId::from_index(index)
}

fn domain(shape: &Shape, direction: Direction) -> Vec<usize> {
    match direction {
        Direction::Forward => shape.reachable().into_iter().collect(),
        Direction::Backward => shape.live(),
    }
}

/// The IN and OUT facts the block solver reports, from a reference whose
/// "merged" side is the analysis direction's input.
fn assert_matches_reference(shape: &Shape, direction: Direction, effects: &GenKill, what: &str) {
    let cfg = shape.cfg();
    let facts = solve_problem(&cfg, &May { effects, direction }).unwrap();
    let forward = direction == Direction::Forward;
    let reference = round_robin(
        shape,
        &domain(shape, direction),
        forward,
        &BTreeSet::new(),
        &boundary(),
        &union,
        &|block, input| effects.apply(block, input),
    );
    for block in shape.live() {
        // The block-level facts are physical: IN is before the block's
        // instructions and OUT after, whichever way the analysis flows.
        let (physical_in, physical_out) = if forward {
            (&reference.merged[block], &reference.transferred[block])
        } else {
            (&reference.transferred[block], &reference.merged[block])
        };
        assert_eq!(
            facts.fact_in(id(block)),
            physical_in,
            "{what}: IN of {block} in {shape:?}"
        );
        assert_eq!(
            facts.fact_out(id(block)),
            physical_out,
            "{what}: OUT of {block} in {shape:?}"
        );
    }
}

#[test]
fn forward_may_analysis_matches_the_round_robin_reference_on_random_graphs() {
    let mut rng = Rng::new(11);
    for _ in 0..300 {
        let shape = Shape::random(&mut rng, true, false);
        let effects = GenKill::random(&mut rng, shape.blocks);
        assert_matches_reference(&shape, Direction::Forward, &effects, "forward");
    }
}

#[test]
fn backward_may_analysis_matches_the_round_robin_reference_on_random_graphs() {
    let mut rng = Rng::new(12);
    for _ in 0..300 {
        let shape = Shape::random(&mut rng, false, false);
        let effects = GenKill::random(&mut rng, shape.blocks);
        assert_matches_reference(&shape, Direction::Backward, &effects, "backward");
    }
}

#[test]
fn forward_must_analysis_lifted_with_meet_options_matches_the_reference() {
    let mut rng = Rng::new(13);
    for _ in 0..300 {
        let shape = Shape::random(&mut rng, true, false);
        let effects = GenKill::random(&mut rng, shape.blocks);
        let cfg = shape.cfg();
        let facts = solve_problem(&cfg, &Must { effects: &effects }).unwrap();
        let reference = round_robin(
            &shape,
            &domain(&shape, Direction::Forward),
            true,
            &None,
            &Some(boundary()),
            &|a, b| meet_options(a, b, intersection),
            &|block, input| input.as_ref().map(|facts| effects.apply(block, facts)),
        );
        for block in shape.live() {
            assert_eq!(
                facts.fact_in(id(block)),
                &reference.merged[block],
                "IN of {block} in {shape:?}"
            );
            assert_eq!(
                facts.fact_out(id(block)),
                &reference.transferred[block],
                "OUT of {block} in {shape:?}"
            );
        }
    }
}

#[test]
fn meet_options_treats_none_as_the_identity_and_meets_only_two_reached_facts() {
    let calls = Cell::new(0_usize);
    let sum = |a: &u32, b: &u32| {
        calls.set(calls.get() + 1);
        a + b
    };
    for fact in [None, Some(3_u32)] {
        assert_eq!(meet_options(&None, &fact, sum), fact);
        assert_eq!(meet_options(&fact, &None, sum), fact);
    }
    assert_eq!(calls.get(), 0, "no inner meet without two reached facts");
    assert_eq!(meet_options(&Some(3), &Some(4), sum), Some(7));
    assert_eq!(calls.get(), 1);
}

#[test]
fn blocks_unreachable_from_the_entry_keep_bottom_and_feed_nothing_forward() {
    // entry -> a -> exit, and an unreachable block that also flows into exit.
    let shape = Shape::new(4, &[(0, 1), (1, 2), (3, 2)]);
    let effects = GenKill::with_gen(4, &[(1, 1), (3, 9)]);
    let cfg = shape.cfg();
    let facts = solve_problem(
        &cfg,
        &May {
            effects: &effects,
            direction: Direction::Forward,
        },
    )
    .unwrap();
    assert_eq!(facts.fact_in(id(3)), &BTreeSet::new());
    assert_eq!(facts.fact_out(id(3)), &BTreeSet::new());
    assert_eq!(facts.fact_in(id(2)), &BTreeSet::from([BOUNDARY_MARK, 1]));
}

#[test]
fn a_backward_solve_also_computes_facts_for_blocks_the_entry_cannot_reach() {
    let shape = Shape::new(4, &[(0, 1), (1, 2), (3, 2)]);
    let effects = GenKill::with_gen(4, &[(2, 5), (3, 4)]);
    let cfg = shape.cfg();
    let facts = solve_problem(
        &cfg,
        &May {
            effects: &effects,
            direction: Direction::Backward,
        },
    )
    .unwrap();
    assert_eq!(facts.fact_in(id(3)), &BTreeSet::from([BOUNDARY_MARK, 5, 4]));
}

#[test]
fn forward_entry_fact_reaches_an_entry_that_is_also_a_branch_target() {
    // entry -> a -> entry: the entry has a predecessor but is still where
    // the analysis starts, so its boundary fact is part of its input.
    let shape = Shape::new(2, &[(0, 1), (1, 0)]);
    let effects = GenKill::with_gen(2, &[(1, 2)]);
    let cfg = shape.cfg();
    let facts = solve_problem(
        &cfg,
        &May {
            effects: &effects,
            direction: Direction::Forward,
        },
    )
    .unwrap();
    assert_eq!(facts.fact_in(id(0)), &BTreeSet::from([BOUNDARY_MARK, 2]));
    assert_eq!(facts.fact_in(id(1)), &BTreeSet::from([BOUNDARY_MARK, 2]));
}

#[test]
fn removing_a_block_does_not_change_the_facts_of_the_others_beyond_its_edges() {
    let mut rng = Rng::new(14);
    let mut holes = 0;
    while holes < 60 {
        let shape = Shape::random(&mut rng, true, false);
        if shape.hole.is_none() {
            continue;
        }
        holes += 1;
        let effects = GenKill::random(&mut rng, shape.blocks);
        assert_matches_reference(&shape, Direction::Forward, &effects, "hole");
        assert_matches_reference(&shape, Direction::Backward, &effects, "hole");
    }
}

fn chain_with_gens() -> (Cfg<()>, GenKill) {
    let shape = Shape::new(4, &[(0, 1), (1, 2), (2, 3)]);
    (
        shape.cfg(),
        GenKill::with_gen(4, &[(0, 0), (1, 1), (2, 2), (3, 3)]),
    )
}

#[test]
fn a_seeded_solve_visits_only_what_the_seeds_reach() {
    let (cfg, effects) = chain_with_gens();
    let problem = May {
        effects: &effects,
        direction: Direction::Forward,
    };
    let facts = solve_problem_from(&cfg, &problem, &[id(1)]).unwrap();
    // Block 0 was never visited, so block 1 sees its bottom output and the
    // boundary fact (given only to a block with no predecessor) is absent.
    assert_eq!(facts.fact_in(id(0)), &BTreeSet::new());
    assert_eq!(facts.fact_out(id(0)), &BTreeSet::new());
    assert_eq!(facts.fact_in(id(1)), &BTreeSet::new());
    assert_eq!(facts.fact_out(id(3)), &BTreeSet::from([1, 2, 3]));
    assert_eq!(facts.steps(), 3);
}

#[test]
fn seeding_the_entry_equals_the_full_solve_and_seeding_nothing_visits_nothing() {
    let mut rng = Rng::new(15);
    for _ in 0..100 {
        let shape = Shape::random(&mut rng, true, false);
        let effects = GenKill::random(&mut rng, shape.blocks);
        let cfg = shape.cfg();
        for direction in [Direction::Forward, Direction::Backward] {
            let problem = May {
                effects: &effects,
                direction,
            };
            let full = solve_problem(&cfg, &problem).unwrap();
            let seeds: Vec<BlockId> = domain(&shape, direction).into_iter().map(id).collect();
            let seeded = solve_problem_from(&cfg, &problem, &seeds).unwrap();
            assert_eq!(seeded.steps(), full.steps(), "{shape:?}");
            for block in shape.live() {
                assert_eq!(
                    seeded.fact_in(id(block)),
                    full.fact_in(id(block)),
                    "{shape:?}"
                );
                assert_eq!(
                    seeded.fact_out(id(block)),
                    full.fact_out(id(block)),
                    "{shape:?}"
                );
            }
            let nothing = solve_problem_from(&cfg, &problem, &[]).unwrap();
            assert_eq!(nothing.steps(), 0);
            assert!(
                shape
                    .live()
                    .iter()
                    .all(|block| nothing.fact_out(id(*block)).is_empty())
            );
        }
    }
}

#[test]
#[should_panic(expected = "seed")]
fn a_seed_that_is_a_removed_block_panics() {
    let mut shape = Shape::new(3, &[(0, 1), (1, 2)]);
    shape.hole = Some(1);
    let effects = GenKill::with_gen(3, &[]);
    let cfg = shape.cfg();
    let _ = solve_problem_from(
        &cfg,
        &May {
            effects: &effects,
            direction: Direction::Forward,
        },
        &[id(1)],
    );
}

fn loop_graph() -> (Cfg<()>, GenKill) {
    // entry -> head -> body -> head, head -> exit
    let shape = Shape::new(4, &[(0, 1), (1, 2), (2, 1), (1, 3)]);
    (shape.cfg(), GenKill::with_gen(4, &[(2, 1), (3, 2)]))
}

#[test]
fn the_step_limit_reports_where_the_unfinished_solve_stopped() {
    let (cfg, effects) = loop_graph();
    let problem = May {
        effects: &effects,
        direction: Direction::Forward,
    };
    let unbounded = solve_problem(&cfg, &problem).unwrap();
    let needed = unbounded.steps();
    assert!(
        needed > cfg.block_count(),
        "the loop forces at least one revisit"
    );

    let short = needed - 1;
    let error =
        solve_problem_with_config(&cfg, &problem, SolveConfig::with_step_limit(short)).unwrap_err();
    let SolveError::StepLimitExceeded {
        limit,
        steps,
        pending_node,
    } = error;
    assert_eq!((limit, steps), (short, short));
    assert!(
        cfg.contains_block(id(pending_node)),
        "the pending node is a live block"
    );

    let SolveError::StepLimitExceeded { steps, .. } =
        solve_problem_with_config(&cfg, &problem, SolveConfig::with_step_limit(0)).unwrap_err();
    assert_eq!(steps, 0);
}

#[test]
fn raising_the_step_limit_to_the_work_needed_converges_to_the_unbounded_facts() {
    let (cfg, effects) = loop_graph();
    let problem = May {
        effects: &effects,
        direction: Direction::Forward,
    };
    let unbounded = solve_problem(&cfg, &problem).unwrap();
    let seeds: Vec<BlockId> = cfg.block_ids().collect();
    for limit in [
        unbounded.steps(),
        unbounded.steps() + 1,
        unbounded.steps() + 100,
    ] {
        let config = SolveConfig::with_step_limit(limit);
        let bounded = solve_problem_with_config(&cfg, &problem, config).unwrap();
        let seeded = solve_problem_from_with_config(&cfg, &problem, &seeds, config).unwrap();
        for block in cfg.block_ids() {
            assert_eq!(bounded.fact_out(block), unbounded.fact_out(block));
            assert_eq!(seeded.fact_out(block), unbounded.fact_out(block));
        }
        assert_eq!(bounded.steps(), unbounded.steps());
    }
}

#[test]
fn a_solve_config_without_a_limit_is_unbounded() {
    assert_eq!(SolveConfig::new().max_steps(), None);
    assert_eq!(SolveConfig::default(), SolveConfig::new());
    assert_eq!(SolveConfig::with_step_limit(7).max_steps(), Some(7));
}

/// What a fallible problem refuses to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    Entry,
    Meet,
    Transfer(usize),
}

impl fmt::Display for Refusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "the problem refused: {self:?}")
    }
}

impl std::error::Error for Refusal {}

/// [`May`] forward, rejecting what `refusal` names.
struct Fallible<'a> {
    effects: &'a GenKill,
    refusal: Option<Refusal>,
}

impl<V: RootedView> TryProblem<V> for Fallible<'_> {
    type Fact = BTreeSet<u8>;
    type Error = Refusal;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self) -> Self::Fact {
        BTreeSet::new()
    }

    fn entry_fact(&self) -> Result<Self::Fact, Refusal> {
        match self.refusal {
            Some(Refusal::Entry) => Err(Refusal::Entry),
            _ => Ok(boundary()),
        }
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Result<Self::Fact, Refusal> {
        match self.refusal {
            Some(Refusal::Meet) => Err(Refusal::Meet),
            _ => Ok(union(a, b)),
        }
    }

    fn transfer(
        &self,
        _view: &V,
        block: V::NodeId,
        input: &Self::Fact,
    ) -> Result<Self::Fact, Refusal> {
        match self.refusal {
            Some(Refusal::Transfer(refused)) if refused == block.index() => {
                Err(Refusal::Transfer(refused))
            }
            _ => Ok(self.effects.apply(block.index(), input)),
        }
    }
}

#[test]
fn a_fallible_solve_without_refusals_equals_the_infallible_solve() {
    let mut rng = Rng::new(16);
    for _ in 0..100 {
        let shape = Shape::random(&mut rng, true, false);
        let effects = GenKill::random(&mut rng, shape.blocks);
        let cfg = shape.cfg();
        let plain = solve_problem(
            &cfg,
            &May {
                effects: &effects,
                direction: Direction::Forward,
            },
        )
        .unwrap();
        let fallible = Fallible {
            effects: &effects,
            refusal: None,
        };
        let all: Vec<BlockId> = shape.reachable().into_iter().map(id).collect();
        let results = [
            try_solve_problem(&cfg, &fallible),
            try_solve_problem_with_config(&cfg, &fallible, SolveConfig::new()),
            try_solve_problem_from(&cfg, &fallible, &all),
            try_solve_problem_from_with_config(&cfg, &fallible, &all, SolveConfig::new()),
        ];
        for result in results {
            let facts = result.unwrap();
            for block in shape.live() {
                assert_eq!(facts.fact_in(id(block)), plain.fact_in(id(block)));
                assert_eq!(facts.fact_out(id(block)), plain.fact_out(id(block)));
            }
        }
    }
}

#[test]
fn consumer_refusals_surface_as_problem_errors_naming_the_refusal() {
    // entry -> a, entry -> b, a -> join, b -> join, and an unreachable block.
    let shape = Shape::new(5, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    let effects = GenKill::with_gen(5, &[]);
    let cfg = shape.cfg();
    for refusal in [Refusal::Entry, Refusal::Meet, Refusal::Transfer(2)] {
        let problem = Fallible {
            effects: &effects,
            refusal: Some(refusal),
        };
        assert_eq!(
            try_solve_problem(&cfg, &problem).unwrap_err(),
            TrySolveError::Problem(refusal)
        );
    }
}

#[test]
fn a_refusal_in_code_the_solve_never_visits_is_not_raised() {
    let shape = Shape::new(5, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    let effects = GenKill::with_gen(5, &[]);
    let cfg = shape.cfg();
    // Block 4 is unreachable, so the forward solve never transfers it.
    let problem = Fallible {
        effects: &effects,
        refusal: Some(Refusal::Transfer(4)),
    };
    assert!(try_solve_problem(&cfg, &problem).is_ok());
    // Seeding it asks for exactly that transfer.
    assert_eq!(
        try_solve_problem_from(&cfg, &problem, &[id(4)]).unwrap_err(),
        TrySolveError::Problem(Refusal::Transfer(4))
    );
}

#[test]
fn a_fallible_problem_is_asked_for_its_entry_fact_when_the_root_is_seeded() {
    let shape = Shape::new(3, &[(0, 1), (1, 2)]);
    let effects = GenKill::with_gen(3, &[]);
    let cfg = shape.cfg();
    let problem = Fallible {
        effects: &effects,
        refusal: Some(Refusal::Entry),
    };
    assert_eq!(
        try_solve_problem_from(&cfg, &problem, &[id(0)]).unwrap_err(),
        TrySolveError::Problem(Refusal::Entry)
    );
}

#[test]
fn a_seeded_root_starts_with_the_entry_fact_and_a_seeded_inner_block_never_does() {
    // entry (gens 1) -> a (gens 2) -> b. The entry fact is the distinguishable
    // mark `BOUNDARY_MARK`, which no block generates.
    let shape = Shape::new(3, &[(0, 1), (1, 2)]);
    let effects = GenKill::with_gen(3, &[(0, 1), (1, 2)]);
    let cfg = shape.cfg();
    let problem = May {
        effects: &effects,
        direction: Direction::Forward,
    };

    let from_root = solve_problem_from(&cfg, &problem, &[id(0)]).unwrap();
    assert_eq!(from_root.fact_in(id(0)), &BTreeSet::from([BOUNDARY_MARK]));
    assert_eq!(
        from_root.fact_in(id(2)),
        &BTreeSet::from([BOUNDARY_MARK, 1, 2]),
        "the entry fact travels with what the blocks generate"
    );

    // Seeded at a block that has a predecessor, the input is the meet of the
    // predecessors' outputs. The predecessor was never visited, so that is
    // bottom: the entry fact does not appear anywhere.
    let from_inner = solve_problem_from(&cfg, &problem, &[id(1)]).unwrap();
    assert_eq!(from_inner.fact_in(id(1)), &BTreeSet::new());
    assert_eq!(from_inner.fact_out(id(1)), &BTreeSet::from([2]));
    assert_eq!(from_inner.fact_in(id(2)), &BTreeSet::from([2]));
    for block in shape.live() {
        assert!(!from_inner.fact_in(id(block)).contains(&BOUNDARY_MARK));
        assert!(!from_inner.fact_out(id(block)).contains(&BOUNDARY_MARK));
    }
    assert_eq!(
        from_inner.fact_in(id(0)),
        &BTreeSet::new(),
        "an unvisited block stays at bottom"
    );
    assert_eq!(from_inner.fact_out(id(0)), &BTreeSet::new());
}

#[test]
fn a_fallible_solve_reports_its_own_step_limit_separately_from_consumer_errors() {
    let (cfg, effects) = loop_graph();
    let fallible = Fallible {
        effects: &effects,
        refusal: None,
    };
    let needed = try_solve_problem(&cfg, &fallible).unwrap().steps();
    let seeds: Vec<BlockId> = cfg.block_ids().collect();
    let short = SolveConfig::with_step_limit(needed - 1);
    let errors = [
        try_solve_problem_with_config(&cfg, &fallible, short).unwrap_err(),
        try_solve_problem_from_with_config(&cfg, &fallible, &seeds, short).unwrap_err(),
    ];
    for error in errors {
        match error {
            TrySolveError::Solver(SolveError::StepLimitExceeded { limit, steps, .. }) => {
                assert_eq!((limit, steps), (needed - 1, needed - 1));
            }
            TrySolveError::Problem(refusal) => panic!("a step limit is not a refusal: {refusal}"),
        }
        assert_eq!(
            error,
            TrySolveError::from(SolveError::StepLimitExceeded {
                limit: needed - 1,
                steps: needed - 1,
                pending_node: match error {
                    TrySolveError::Solver(SolveError::StepLimitExceeded {
                        pending_node, ..
                    }) => pending_node,
                    TrySolveError::Problem(_) => unreachable!(),
                },
            })
        );
    }
    let enough = SolveConfig::with_step_limit(needed);
    assert!(try_solve_problem_with_config(&cfg, &fallible, enough).is_ok());
}

#[test]
fn both_error_kinds_expose_their_cause_through_the_error_trait() {
    use std::error::Error;
    let solver = TrySolveError::<Refusal>::Solver(SolveError::StepLimitExceeded {
        limit: 1,
        steps: 1,
        pending_node: 0,
    });
    let consumer = TrySolveError::Problem(Refusal::Meet);
    assert!(solver.source().is_some());
    assert!(consumer.source().is_some());
    assert!(consumer.to_string().contains("Meet"));
    assert!(solver.to_string().contains('1'));
}
