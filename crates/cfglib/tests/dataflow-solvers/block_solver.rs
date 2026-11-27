//! Per-block problems over control-flow graphs, solved by the node solver
//! the way a CFG analysis is rooted: forward from the entry
//! (`Seeds::Reachable`), backward from every live block (`Seeds::Live`).
//! They run against a round-robin reference, plus the documented contracts
//! of seeds, step limits, and consumer errors.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::fmt;

use cfglib::{
    BlockId, Cfg, Direction, EdgeKind, Facts, IterationOrder, NodeProblem, Resume, ResumeMode,
    Seeds, Solve, SolveConfig, SolveError, TryNodeProblem, TrySolveError, meet_options,
    resume_node_problem, solve_node_problem, stream_node_problem,
};

use crate::graphs::{
    Edit, GenKill, ORDERS, Rng, Shape, assert_same_facts, assert_streams_like, intersection, live,
    rooted, seeded, step_limit, union,
};
use crate::reference::round_robin;

/// The fact an analysis starts a boundary block with; no block generates or
/// kills it, so it is visible exactly where the boundary reaches.
const BOUNDARY_MARK: u8 = 6;

fn boundary() -> BTreeSet<u8> {
    BTreeSet::from([BOUNDARY_MARK])
}

/// Solves `problem` as a per-block analysis of `cfg` within `config`:
/// forward from the entry, backward from every live block.
fn solve_rooted<P: TryNodeProblem<Cfg<()>>>(
    cfg: &Cfg<()>,
    problem: &P,
    config: SolveConfig,
) -> Result<Facts<P::Fact>, TrySolveError<P::Error>> {
    solve_node_problem(cfg, problem, rooted(cfg, problem.direction(), config))
}

/// A gen/kill problem whose meet is union: reaching-definitions shaped.
struct May<'a> {
    effects: &'a GenKill,
    direction: Direction,
}

impl NodeProblem<Cfg<()>> for May<'_> {
    type Fact = BTreeSet<u8>;

    fn direction(&self) -> Direction {
        self.direction
    }

    fn bottom(&self, _cfg: &Cfg<()>) -> Self::Fact {
        BTreeSet::new()
    }

    fn boundary(&self, _cfg: &Cfg<()>) -> Self::Fact {
        boundary()
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
        union(a, b)
    }

    fn transfer(&self, _cfg: &Cfg<()>, block: BlockId, input: &Self::Fact) -> Self::Fact {
        self.effects.apply(block.index(), input)
    }
}

/// A gen/kill problem whose meet is intersection, lifted over `Option` with
/// `meet_options` so that `None` is "no path has reached here yet".
struct Must<'a> {
    effects: &'a GenKill,
}

impl NodeProblem<Cfg<()>> for Must<'_> {
    type Fact = Option<BTreeSet<u8>>;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _cfg: &Cfg<()>) -> Self::Fact {
        None
    }

    fn boundary(&self, _cfg: &Cfg<()>) -> Self::Fact {
        Some(boundary())
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
        meet_options(a, b, intersection)
    }

    fn transfer(&self, _cfg: &Cfg<()>, block: BlockId, input: &Self::Fact) -> Self::Fact {
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

/// The IN and OUT facts of a per-block analysis, from a reference whose
/// "merged" side is the analysis direction's input.
fn assert_matches_reference(shape: &Shape, direction: Direction, effects: &GenKill, what: &str) {
    let cfg = shape.cfg();
    let facts = solve_rooted(&cfg, &May { effects, direction }, SolveConfig::default()).unwrap();
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
        let must = Must { effects: &effects };
        let facts = solve_rooted(&cfg, &must, SolveConfig::default()).unwrap();
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

/// Where nothing flows back into the entry and the entry reaches every live
/// block, the rooted forward solve of a control-flow graph is the solve from
/// every live block: the whole result is equal, steps included.
#[test]
fn a_rooted_forward_solve_equals_the_live_solve_when_the_entry_reaches_every_block() {
    let mut rng = Rng::new(13);
    let mut compared = 0;
    while compared < 100 {
        let shape = Shape::random(&mut rng, true, false);
        let effects = GenKill::random(&mut rng, shape.blocks);
        if shape.reachable().len() != shape.live().len() {
            continue;
        }
        compared += 1;
        let cfg = shape.cfg();
        let problem = May {
            effects: &effects,
            direction: Direction::Forward,
        };
        let from_entry = solve_rooted(&cfg, &problem, SolveConfig::default()).unwrap();
        let from_live = solve_node_problem(&cfg, &problem, Solve::default()).unwrap();
        assert_eq!(from_entry, from_live, "{shape:?}");
    }
    // A loop, every block of which the entry reaches.
    let (cfg, effects) = loop_graph();
    let problem = May {
        effects: &effects,
        direction: Direction::Forward,
    };
    let from_entry = solve_rooted(&cfg, &problem, SolveConfig::default()).unwrap();
    let from_live = solve_node_problem(&cfg, &problem, Solve::default()).unwrap();
    assert_eq!(from_entry, from_live);
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
    let problem = May {
        effects: &effects,
        direction: Direction::Forward,
    };
    let facts = solve_rooted(&cfg, &problem, SolveConfig::default()).unwrap();
    assert_eq!(facts.fact_in(id(3)), &BTreeSet::new());
    assert_eq!(facts.fact_out(id(3)), &BTreeSet::new());
    assert_eq!(facts.fact_in(id(2)), &BTreeSet::from([BOUNDARY_MARK, 1]));
}

#[test]
fn a_backward_solve_also_computes_facts_for_blocks_the_entry_cannot_reach() {
    let shape = Shape::new(4, &[(0, 1), (1, 2), (3, 2)]);
    let effects = GenKill::with_gen(4, &[(2, 5), (3, 4)]);
    let cfg = shape.cfg();
    let problem = May {
        effects: &effects,
        direction: Direction::Backward,
    };
    let facts = solve_rooted(&cfg, &problem, SolveConfig::default()).unwrap();
    assert_eq!(facts.fact_in(id(3)), &BTreeSet::from([BOUNDARY_MARK, 5, 4]));
}

#[test]
fn forward_entry_fact_reaches_an_entry_that_is_also_a_branch_target() {
    // entry -> a -> entry: the entry has a predecessor but is still where
    // the analysis starts, so its boundary fact is part of its input.
    let shape = Shape::new(2, &[(0, 1), (1, 0)]);
    let effects = GenKill::with_gen(2, &[(1, 2)]);
    let cfg = shape.cfg();
    let problem = May {
        effects: &effects,
        direction: Direction::Forward,
    };
    let facts = solve_rooted(&cfg, &problem, SolveConfig::default()).unwrap();
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
    let facts = solve_node_problem(&cfg, &problem, seeded(&[id(1)], SolveConfig::default()));
    let facts = facts.unwrap();
    // Block 0 was never visited, so block 1 sees its bottom output and the
    // boundary fact (given only to a block with no predecessor) is absent.
    assert_eq!(facts.fact_in(id(0)), &BTreeSet::new());
    assert_eq!(facts.fact_out(id(0)), &BTreeSet::new());
    assert_eq!(facts.fact_in(id(1)), &BTreeSet::new());
    assert_eq!(facts.fact_out(id(3)), &BTreeSet::from([1, 2, 3]));
    assert_eq!(facts.stats().step_count(), 3);
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
            let full = solve_rooted(&cfg, &problem, SolveConfig::default()).unwrap();
            let seeds: Vec<BlockId> = domain(&shape, direction).into_iter().map(id).collect();
            let from_seeds = seeded(&seeds, SolveConfig::default());
            let from_seeds = solve_node_problem(&cfg, &problem, from_seeds).unwrap();
            assert_eq!(
                from_seeds.stats().step_count(),
                full.stats().step_count(),
                "{shape:?}"
            );
            for block in shape.live() {
                assert_eq!(
                    from_seeds.fact_in(id(block)),
                    full.fact_in(id(block)),
                    "{shape:?}"
                );
                assert_eq!(
                    from_seeds.fact_out(id(block)),
                    full.fact_out(id(block)),
                    "{shape:?}"
                );
            }
            let nothing = seeded(&[], SolveConfig::default());
            let nothing = solve_node_problem(&cfg, &problem, nothing).unwrap();
            assert_eq!(nothing.stats().step_count(), 0);
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
    let problem = May {
        effects: &effects,
        direction: Direction::Forward,
    };
    let _ = solve_node_problem(&cfg, &problem, seeded(&[id(1)], SolveConfig::default()));
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
    let unbounded = solve_rooted(&cfg, &problem, SolveConfig::default()).unwrap();
    let needed = unbounded.stats().step_count();
    assert!(
        needed > cfg.block_count(),
        "the loop forces at least one revisit"
    );

    let short = needed - 1;
    let error = solve_rooted(&cfg, &problem, step_limit(short)).unwrap_err();
    let SolveError::StepLimitExceeded {
        limit,
        steps,
        pending_node,
    } = SolveError::from(error);
    assert_eq!((limit, steps), (short, short));
    assert!(
        cfg.contains_block(id(pending_node)),
        "the pending node is a live block"
    );

    let error = solve_rooted(&cfg, &problem, step_limit(0)).unwrap_err();
    let SolveError::StepLimitExceeded { steps, .. } = SolveError::from(error);
    assert_eq!(steps, 0);
}

#[test]
fn raising_the_step_limit_to_the_work_needed_converges_to_the_unbounded_facts() {
    let (cfg, effects) = loop_graph();
    let problem = May {
        effects: &effects,
        direction: Direction::Forward,
    };
    let unbounded = solve_rooted(&cfg, &problem, SolveConfig::default()).unwrap();
    let needed = unbounded.stats().step_count();
    let seeds: Vec<BlockId> = cfg.block_ids().collect();
    for limit in [needed, needed + 1, needed + 100] {
        let config = step_limit(limit);
        let bounded = solve_rooted(&cfg, &problem, config).unwrap();
        let from_seeds = solve_node_problem(&cfg, &problem, seeded(&seeds, config)).unwrap();
        for block in cfg.block_ids() {
            assert_eq!(bounded.fact_out(block), unbounded.fact_out(block));
            assert_eq!(from_seeds.fact_out(block), unbounded.fact_out(block));
        }
        assert_eq!(bounded.stats().step_count(), needed);
    }
}

#[test]
fn a_default_solve_starts_from_every_live_node_without_a_limit() {
    let solve = Solve::<BlockId>::default();
    assert_eq!(solve.seeds, Seeds::Live);
    assert_eq!(solve.config, SolveConfig::default());
    assert_eq!(solve.config.max_steps, None);
    assert_eq!(solve.config.order, IterationOrder::Index);
    assert_eq!(solve.config.narrowing_passes, 0);
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

impl TryNodeProblem<Cfg<()>> for Fallible<'_> {
    type Fact = BTreeSet<u8>;
    type Error = Refusal;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _cfg: &Cfg<()>) -> Self::Fact {
        BTreeSet::new()
    }

    fn boundary(&self, _cfg: &Cfg<()>) -> Result<Self::Fact, Refusal> {
        match self.refusal {
            Some(Refusal::Entry) => Err(Refusal::Entry),
            _ => Ok(boundary()),
        }
    }

    fn meet(
        &self,
        _cfg: &Cfg<()>,
        _block: BlockId,
        a: &Self::Fact,
        b: &Self::Fact,
    ) -> Result<Self::Fact, Refusal> {
        match self.refusal {
            Some(Refusal::Meet) => Err(Refusal::Meet),
            _ => Ok(union(a, b)),
        }
    }

    fn transfer(
        &self,
        _cfg: &Cfg<()>,
        block: BlockId,
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
        let problem = May {
            effects: &effects,
            direction: Direction::Forward,
        };
        let plain = solve_rooted(&cfg, &problem, SolveConfig::default()).unwrap();
        let fallible = Fallible {
            effects: &effects,
            refusal: None,
        };
        let all: Vec<BlockId> = shape.reachable().into_iter().map(id).collect();
        let results = [
            solve_rooted(&cfg, &fallible, SolveConfig::default()),
            solve_node_problem(&cfg, &fallible, seeded(&all, SolveConfig::default())),
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
            solve_rooted(&cfg, &problem, SolveConfig::default()).unwrap_err(),
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
    assert!(solve_rooted(&cfg, &problem, SolveConfig::default()).is_ok());
    // Seeding it asks for exactly that transfer.
    assert_eq!(
        solve_node_problem(&cfg, &problem, seeded(&[id(4)], SolveConfig::default())).unwrap_err(),
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
        solve_node_problem(&cfg, &problem, seeded(&[id(0)], SolveConfig::default())).unwrap_err(),
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

    let from_root = solve_node_problem(&cfg, &problem, seeded(&[id(0)], SolveConfig::default()));
    let from_root = from_root.unwrap();
    assert_eq!(from_root.fact_in(id(0)), &BTreeSet::from([BOUNDARY_MARK]));
    assert_eq!(
        from_root.fact_in(id(2)),
        &BTreeSet::from([BOUNDARY_MARK, 1, 2]),
        "the entry fact travels with what the blocks generate"
    );

    // Seeded at a block that has a predecessor, the input is the meet of the
    // predecessors' outputs. The predecessor was never visited, so that is
    // bottom: the entry fact does not appear anywhere.
    let from_inner = solve_node_problem(&cfg, &problem, seeded(&[id(1)], SolveConfig::default()));
    let from_inner = from_inner.unwrap();
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
    let needed = solve_rooted(&cfg, &fallible, SolveConfig::default())
        .unwrap()
        .stats()
        .step_count();
    let seeds: Vec<BlockId> = cfg.block_ids().collect();
    let short = step_limit(needed - 1);
    let errors = [
        solve_rooted(&cfg, &fallible, short).unwrap_err(),
        solve_node_problem(&cfg, &fallible, seeded(&seeds, short)).unwrap_err(),
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
    assert!(solve_rooted(&cfg, &fallible, step_limit(needed)).is_ok());
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

/// The least fixpoint does not depend on the iteration order: every order,
/// with or without descending passes, reaches the facts of the index order,
/// forward and backward, for rooted, seeded, and live solves.
#[test]
fn every_iteration_order_reaches_the_facts_of_the_index_order() {
    let mut rng = Rng::new(14);
    for _ in 0..300 {
        for direction in [Direction::Forward, Direction::Backward] {
            let shape = Shape::random(&mut rng, false, false);
            let effects = GenKill::random(&mut rng, shape.blocks);
            let cfg = shape.cfg();
            let alive = shape.live();
            let blocks = || alive.iter().map(|&block| id(block));
            let seeds = [id(alive[rng.below(alive.len())])];
            let may = May {
                effects: &effects,
                direction,
            };
            let must = Must { effects: &effects };
            let full = solve_rooted(&cfg, &may, SolveConfig::default()).unwrap();
            let from_seeds = seeded(&seeds, SolveConfig::default());
            let from_seeds = solve_node_problem(&cfg, &may, from_seeds).unwrap();
            let from_live = solve_node_problem(&cfg, &may, Solve::default()).unwrap();
            let must_full = solve_rooted(&cfg, &must, SolveConfig::default()).unwrap();
            for order in ORDERS {
                for passes in [0, 2] {
                    let config = SolveConfig {
                        max_steps: None,
                        order,
                        narrowing_passes: passes,
                    };
                    let what = format!("{order:?} with {passes} passes, {direction:?}, {shape:?}");
                    let ordered = solve_rooted(&cfg, &may, config).unwrap();
                    assert_same_facts(&ordered, &full, blocks(), &what);
                    let ordered = solve_node_problem(&cfg, &may, seeded(&seeds, config));
                    assert_same_facts(&ordered.unwrap(), &from_seeds, blocks(), &what);
                    let ordered = solve_node_problem(&cfg, &may, live(config)).unwrap();
                    assert_same_facts(&ordered, &from_live, blocks(), &what);
                    let ordered = solve_rooted(&cfg, &must, config).unwrap();
                    assert_same_facts(&ordered, &must_full, blocks(), &what);
                    if direction == Direction::Forward {
                        let fallible = Fallible {
                            effects: &effects,
                            refusal: None,
                        };
                        let ordered = solve_rooted(&cfg, &fallible, config);
                        assert_same_facts(&ordered.unwrap(), &full, blocks(), &what);
                    }
                }
            }
        }
    }
}

/// A rooted stream visits every block the solve reaches exactly once, with
/// the facts and in the steps of the solve in the weak topological order
/// that keeps every fact, forward and backward, with and without narrowing
/// passes; the same problem streamed from every live block visits each of
/// them the same way.
#[test]
fn a_stream_visits_every_reached_block_once_with_the_facts_of_the_solve() {
    let mut rng = Rng::new(17);
    for _ in 0..200 {
        for direction in [Direction::Forward, Direction::Backward] {
            let shape = Shape::random(&mut rng, false, false);
            let effects = GenKill::random(&mut rng, shape.blocks);
            let cfg = shape.cfg();
            let reached = || domain(&shape, direction).into_iter().map(id);
            let may = May {
                effects: &effects,
                direction,
            };
            let must = Must { effects: &effects };
            for passes in [0, 2] {
                let config = SolveConfig {
                    max_steps: None,
                    order: IterationOrder::WeakTopological,
                    narrowing_passes: passes,
                };
                let what = format!("{direction:?} with {passes} passes, {shape:?}");
                let from_entry = rooted(&cfg, direction, config);
                let facts = solve_node_problem(&cfg, &may, from_entry).unwrap();
                let stream = |visit: &mut dyn FnMut(BlockId, &_, &_)| {
                    stream_node_problem(&cfg, &may, from_entry, visit).unwrap()
                };
                assert_streams_like(&facts, reached(), stream, &what);
                if direction == Direction::Forward {
                    let facts = solve_node_problem(&cfg, &must, from_entry).unwrap();
                    let stream = |visit: &mut dyn FnMut(BlockId, &_, &_)| {
                        stream_node_problem(&cfg, &must, from_entry, visit).unwrap()
                    };
                    assert_streams_like(&facts, reached(), stream, &what);
                }
                let facts = solve_node_problem(&cfg, &may, live(config)).unwrap();
                let every = shape.live().into_iter().map(id);
                let stream = |visit: &mut dyn FnMut(BlockId, &_, &_)| {
                    stream_node_problem(&cfg, &may, live(config), visit).unwrap()
                };
                assert_streams_like(&facts, every, stream, &what);
            }
        }
    }
}

/// After a random edit — an added or removed edge, or new effects for one
/// block — resuming from the previous facts gives the facts of a fresh solve
/// of the edited graph, forward and backward, for a may and a must problem,
/// in the index and the weak topological order. When the edit only adds an
/// edge to a forward may problem, its facts can only grow, and growing gives
/// them too; backward, an added edge takes the boundary fact from a block
/// that had no successor, so facts can shrink.
#[test]
fn a_resume_after_an_edit_equals_a_fresh_solve_of_the_edited_graph() {
    let mut rng = Rng::new(18);
    let kinds = [EdgeKind::Fallthrough, EdgeKind::ConditionalTrue];
    for round in 0..300 {
        for direction in [Direction::Forward, Direction::Backward] {
            let only_grow = round % 2 == 0;
            let shape = Shape::random(&mut rng, false, false);
            let mut effects = GenKill::random(&mut rng, shape.blocks);
            let mut cfg = shape.cfg();
            let before = solve_rooted(
                &cfg,
                &May {
                    effects: &effects,
                    direction,
                },
                SolveConfig::default(),
            )
            .unwrap();
            let must_before =
                solve_rooted(&cfg, &Must { effects: &effects }, SolveConfig::default()).unwrap();
            let edit = Edit::random(&mut rng, &mut cfg, &mut effects, &shape, &kinds, only_grow);
            let changed = [edit.changed(direction)];
            let must_changed = [edit.changed(Direction::Forward)];
            let may = May {
                effects: &effects,
                direction,
            };
            let must = Must { effects: &effects };
            let fresh = solve_rooted(&cfg, &may, SolveConfig::default()).unwrap();
            let must_fresh = solve_rooted(&cfg, &must, SolveConfig::default()).unwrap();
            let modes: &[ResumeMode] = if only_grow && direction == Direction::Forward {
                &[ResumeMode::Reset, ResumeMode::Grow]
            } else {
                &[ResumeMode::Reset]
            };
            let blocks = || shape.live().into_iter().map(id);
            for order in [IterationOrder::Index, IterationOrder::WeakTopological] {
                let config = SolveConfig {
                    order,
                    ..SolveConfig::default()
                };
                for &mode in modes {
                    let what =
                        format!("{mode:?} {order:?} {direction:?} after {edit:?}, {shape:?}");
                    let resume = Resume {
                        changed: &changed,
                        mode,
                    };
                    let solve = rooted(&cfg, direction, config);
                    let resumed = resume_node_problem(&cfg, &may, before.clone(), resume, solve);
                    assert_same_facts(&resumed.unwrap(), &fresh, blocks(), &what);
                }
                let what = format!("must {order:?} after {edit:?}, {shape:?}");
                let resume = Resume {
                    changed: &must_changed,
                    mode: ResumeMode::Reset,
                };
                let solve = rooted(&cfg, Direction::Forward, config);
                let resumed = resume_node_problem(&cfg, &must, must_before.clone(), resume, solve);
                assert_same_facts(&resumed.unwrap(), &must_fresh, blocks(), &what);
            }
        }
    }
}
