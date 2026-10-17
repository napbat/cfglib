//! Liveness, reaching definitions, and def-use chains over a toy instruction
//! type, against explicit program-point exploration, with unwind edges,
//! removed blocks, and unreachable code.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, DefUseChains, EdgeKind, Liveness, LivenessProblem, ProgramPoint, ReachingDef,
    ReachingDefs, SeededLivenessProblem, solve_problem,
};

use crate::analysis_oracles::{Ins, Program, State, VARIABLES};
use crate::graphs::{Rng, Shape};

fn id(index: usize) -> BlockId {
    BlockId::from_index(index)
}

fn point(block: usize, inst_idx: usize) -> ProgramPoint {
    ProgramPoint {
        block: id(block),
        inst_idx,
    }
}

fn no_seeds(_: usize) -> Vec<u8> {
    Vec::new()
}

fn assert_liveness_matches(
    program: &Program,
    liveness: &Liveness<u8>,
    cfg: &Cfg<Ins>,
    seeds: &dyn Fn(usize) -> Vec<u8>,
) {
    let mut everywhere = BTreeSet::new();
    for block in program.shape.live() {
        let length = program.code[block].len();
        let live_in = program.live_set(block, 0, seeds);
        let live_out = program.live_set(block, length, seeds);
        assert_eq!(liveness.live_in(id(block)), &live_in, "live-in of {block}");
        assert_eq!(
            liveness.live_out(id(block)),
            &live_out,
            "live-out of {block}"
        );
        for variable in 0..VARIABLES {
            assert_eq!(
                liveness.is_live_in(&variable, id(block)),
                live_in.contains(&variable)
            );
            assert_eq!(
                liveness.is_live_out(&variable, id(block)),
                live_out.contains(&variable)
            );
        }
        let before = liveness.live_before_instructions(cfg, id(block));
        let after = liveness.live_after_instructions(cfg, id(block));
        assert_eq!(before.len(), length);
        assert_eq!(after.len(), length);
        for index in 0..length {
            assert_eq!(
                before[index],
                program.live_set(block, index, seeds),
                "before {block}:{index}"
            );
            assert_eq!(
                after[index],
                program.live_set(block, index + 1, seeds),
                "after {block}:{index}"
            );
        }
        assert_eq!(
            liveness.live_on_unwind(cfg, id(block)),
            program.live_on_unwind(block, seeds),
            "on unwind from {block}"
        );
        everywhere.extend(live_in);
        everywhere.extend(live_out);
    }
    assert_eq!(liveness.all_live_variables(cfg), everywhere);
}

#[test]
fn liveness_matches_program_point_exploration_with_unwinds_holes_and_unreachable_code() {
    let mut rng = Rng::new(41);
    for _ in 0..250 {
        let program = Program::random(&mut rng, true);
        let cfg = program.cfg();
        assert_liveness_matches(&program, &Liveness::compute(&cfg), &cfg, &no_seeds);
    }
}

#[test]
fn liveness_with_exits_reads_the_seed_at_the_end_of_the_named_blocks() {
    let mut rng = Rng::new(42);
    for _ in 0..250 {
        let program = Program::random(&mut rng, true);
        let cfg = program.cfg();
        let seeds = |block: usize| {
            if program.shape.successors(block).is_empty() {
                vec![0, 3]
            } else {
                Vec::new()
            }
        };
        let liveness = Liveness::compute_with_exits(&cfg, |block| seeds(block.index()));
        assert_liveness_matches(&program, &liveness, &cfg, &seeds);
    }
}

#[test]
fn the_block_level_liveness_problem_equals_liveness_without_unwinds() {
    let mut rng = Rng::new(43);
    for _ in 0..250 {
        let program = Program::random(&mut rng, false);
        let cfg = program.cfg();
        let seeds = |block: usize| {
            if program.shape.successors(block).is_empty() {
                vec![1]
            } else {
                Vec::new()
            }
        };
        let precise = Liveness::compute_with_exits(&cfg, |block| seeds(block.index()));
        let raw = solve_problem(
            &cfg,
            &SeededLivenessProblem::new(|block: BlockId| seeds(block.index())),
        )
        .unwrap();
        let plain = solve_problem(&cfg, &LivenessProblem).unwrap();
        let unseeded = Liveness::compute(&cfg);
        for block in program.shape.live() {
            // The seed joins the transfer, so IN carries it already; OUT of a
            // raw solve carries only what flows back from successors.
            assert_eq!(
                raw.fact_in(id(block)),
                precise.live_in(id(block)),
                "IN of {block}"
            );
            let seeded_out: BTreeSet<u8> = raw
                .fact_out(id(block))
                .iter()
                .copied()
                .chain(seeds(block))
                .collect();
            assert_eq!(&seeded_out, precise.live_out(id(block)), "OUT of {block}");
            assert_eq!(plain.fact_in(id(block)), unseeded.live_in(id(block)));
            assert_eq!(plain.fact_out(id(block)), unseeded.live_out(id(block)));
        }
    }
}

#[test]
fn the_block_level_liveness_problem_never_misses_what_liveness_finds_with_unwinds() {
    let mut rng = Rng::new(44);
    for _ in 0..250 {
        let program = Program::random(&mut rng, true);
        let cfg = program.cfg();
        let precise = Liveness::compute(&cfg);
        let coarse = solve_problem(&cfg, &LivenessProblem).unwrap();
        for block in program.shape.live() {
            assert!(
                precise
                    .live_in(id(block))
                    .is_subset(coarse.fact_in(id(block))),
                "IN of {block}"
            );
            assert!(
                precise
                    .live_out(id(block))
                    .is_subset(coarse.fact_out(id(block))),
                "OUT of {block}"
            );
        }
    }
}

/// `entry` ends by throwing in its first instruction and defines `a` after
/// it; the normal successor reads `a`, the handler reads `b`.
fn throw_then_define() -> (Cfg<Ins>, BlockId, BlockId) {
    // Variables: a = 0, b = 1.
    let shape = Shape {
        blocks: 3,
        edges: vec![
            (0, 1, EdgeKind::Fallthrough),
            (0, 2, EdgeKind::ExceptionUnwind),
        ],
        hole: None,
    };
    let code = vec![
        vec![Ins::throwing(&[], &[]), Ins::new(&[], &[0])],
        vec![Ins::new(&[0], &[])],
        vec![Ins::new(&[1], &[])],
    ];
    (shape.build(code), id(1), id(2))
}

#[test]
fn an_unwind_reads_only_the_handlers_needs_while_the_block_level_problem_is_coarser() {
    let (cfg, _, _) = throw_then_define();
    let precise = Liveness::compute(&cfg);
    // Only `b` crosses the throwing instruction toward the handler: `a` is
    // written after it, so nothing before the block needs it.
    assert_eq!(precise.live_in(id(0)), &BTreeSet::from([1]));
    assert_eq!(precise.live_on_unwind(&cfg, id(0)), BTreeSet::from([1]));
    assert_eq!(precise.live_out(id(0)), &BTreeSet::from([0]));

    let coarse = solve_problem(&cfg, &LivenessProblem).unwrap();
    assert!(precise.live_in(id(0)).is_subset(coarse.fact_in(id(0))));
}

#[test]
fn an_unwind_out_of_a_block_that_cannot_throw_departs_after_the_block() {
    let shape = Shape {
        blocks: 3,
        edges: vec![
            (0, 1, EdgeKind::Fallthrough),
            (0, 2, EdgeKind::ExceptionUnwind),
        ],
        hole: None,
    };
    let code = vec![vec![Ins::new(&[], &[0])], vec![], vec![Ins::new(&[0], &[])]];
    let cfg = shape.build(code);
    let liveness = Liveness::compute(&cfg);
    assert!(liveness.live_on_unwind(&cfg, id(0)).is_empty());
    assert_eq!(
        liveness.live_out(id(0)),
        &BTreeSet::from([0]),
        "the handler's read is in live-out"
    );

    let reaching = ReachingDefs::compute(&cfg);
    assert_eq!(
        reaching.defs_of_at_entry(&0, id(2)),
        [point(0, 0)],
        "the definition completed before departure"
    );
}

#[test]
fn reaching_definitions_match_program_point_exploration() {
    let mut rng = Rng::new(45);
    for _ in 0..250 {
        let program = Program::random(&mut rng, true);
        let cfg = program.cfg();
        let reaching = ReachingDefs::compute(&cfg);
        let exploration = program.explore();
        for block in program.shape.live() {
            let length = program.code[block].len();
            let as_defs = |state: State| -> BTreeSet<ReachingDef<u8>> {
                state
                    .into_iter()
                    .map(|(variable, (b, i))| ReachingDef {
                        variable,
                        site: point(b, i),
                    })
                    .collect()
            };
            assert_eq!(
                reaching.reaching_in(id(block)),
                &as_defs(exploration.reaching(block, 0)),
                "reaching in {block}"
            );
            assert_eq!(
                reaching.reaching_out(id(block)),
                &as_defs(exploration.reaching(block, length)),
                "reaching out {block}"
            );
            for variable in 0..VARIABLES {
                let expected: BTreeSet<ProgramPoint> = exploration
                    .reaching(block, 0)
                    .into_iter()
                    .filter(|(written, _)| *written == variable)
                    .map(|(_, (b, i))| point(b, i))
                    .collect();
                let actual = reaching.defs_of_at_entry(&variable, id(block));
                assert_eq!(actual.iter().copied().collect::<BTreeSet<_>>(), expected);
                assert!(
                    actual.windows(2).all(|pair| pair[0] < pair[1]),
                    "ordered by site"
                );
            }
        }
    }
}

#[test]
fn a_definition_after_a_throwing_instruction_reaches_the_normal_successor_but_not_the_handler() {
    // x = 1; (throws); x = 2 -> normal reads x, handler reads x. Variable 0.
    let shape = Shape {
        blocks: 3,
        edges: vec![
            (0, 1, EdgeKind::Fallthrough),
            (0, 2, EdgeKind::ExceptionUnwind),
        ],
        hole: None,
    };
    let code = vec![
        vec![
            Ins::new(&[], &[0]),
            Ins::throwing(&[], &[]),
            Ins::new(&[], &[0]),
        ],
        vec![Ins::new(&[0], &[])],
        vec![Ins::new(&[0], &[])],
    ];
    let cfg = shape.build(code);
    let reaching = ReachingDefs::compute(&cfg);
    assert_eq!(reaching.defs_of_at_entry(&0, id(1)), [point(0, 2)]);
    assert_eq!(reaching.defs_of_at_entry(&0, id(2)), [point(0, 0)]);

    let chains = DefUseChains::compute(&cfg);
    assert_eq!(chains.defs_of(point(2, 0)), &BTreeSet::from([point(0, 0)]));
    assert_eq!(chains.defs_of(point(1, 0)), &BTreeSet::from([point(0, 2)]));
}

#[test]
fn def_use_chains_match_program_point_exploration_and_mirror_each_other() {
    let mut rng = Rng::new(46);
    for _ in 0..250 {
        let program = Program::random(&mut rng, true);
        let cfg = program.cfg();
        let chains = DefUseChains::compute(&cfg);
        let suppliers = program.explore().suppliers(&program);

        let mut used: BTreeSet<(usize, usize)> = BTreeSet::new();
        for (use_site, defs) in &suppliers {
            used.extend(defs);
            let expected: BTreeSet<ProgramPoint> =
                defs.iter().map(|(b, i)| point(*b, *i)).collect();
            assert_eq!(
                chains.defs_of(point(use_site.0, use_site.1)),
                &expected,
                "suppliers of {use_site:?}"
            );
        }
        let mut definitions = BTreeSet::new();
        let reachable = program.shape.reachable();
        for &block in &reachable {
            for (index, ins) in program.code[block].iter().enumerate() {
                if !ins.uses.is_empty() && !suppliers.contains_key(&(block, index)) {
                    assert!(
                        chains.defs_of(point(block, index)).is_empty(),
                        "unreachable use {block}:{index}"
                    );
                }
                if ins.defs.is_empty() {
                    continue;
                }
                definitions.insert((block, index));
                let readers: BTreeSet<ProgramPoint> = suppliers
                    .iter()
                    .filter(|(_, defs)| defs.contains(&(block, index)))
                    .map(|(site, _)| point(site.0, site.1))
                    .collect();
                assert_eq!(
                    chains.uses_of(point(block, index)),
                    &readers,
                    "readers of {block}:{index}"
                );
            }
        }
        let dead: BTreeSet<ProgramPoint> = definitions
            .difference(&used)
            .map(|(b, i)| point(*b, *i))
            .collect();
        // Chains inside unreachable code are not specified: only blocks the
        // entry reaches are compared.
        let reachable_dead: BTreeSet<ProgramPoint> = chains
            .dead_defs()
            .into_iter()
            .filter(|site| reachable.contains(&site.block.index()))
            .collect();
        assert_eq!(reachable_dead, dead);
        for (def, uses) in &chains.def_use {
            for user in uses {
                assert!(
                    chains.use_def[user].contains(def),
                    "use-def mirrors def-use"
                );
            }
        }
    }
}

#[test]
fn a_loop_carried_definition_reaches_its_own_use_and_a_dead_definition_is_reported() {
    // entry: x = 0 ; loop: use x; x = x + 1 (reads and rewrites x) ; exit: nothing.
    let shape = Shape::new(3, &[(0, 1), (1, 1), (1, 2)]);
    let code = vec![
        vec![Ins::new(&[], &[0]), Ins::new(&[], &[1])],
        vec![Ins::new(&[0], &[0])],
        vec![],
    ];
    let cfg = shape.build(code);
    let chains = DefUseChains::compute(&cfg);
    assert_eq!(
        chains.defs_of(point(1, 0)),
        &BTreeSet::from([point(0, 0), point(1, 0)])
    );
    assert_eq!(chains.uses_of(point(1, 0)), &BTreeSet::from([point(1, 0)]));
    // `y` (variable 1) written at (0, 1) is never read; the loop-carried
    // definition at (1, 0) is read by the next iteration.
    assert_eq!(chains.dead_defs(), [point(0, 1)]);
}
