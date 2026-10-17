//! Copy and alias propagation and dead-code removal, against the
//! interpreter and against counts derived by hand from their documented
//! guards.

use cfglib::{
    BasicBlock, BlockId, Cfg, DeadCode, EdgeKind, EffectInfo, InstrInfo, ProgramPoint,
    alias_propagation, copy_propagation, copy_propagation_with_exits, dead_code_elimination,
    dead_code_elimination_with_exits, remove_dead_code,
};

use crate::check::{assert_same_behavior, assert_well_formed};
use crate::interp::{input_vectors, run};
use crate::oracle::Reference;
use crate::programs::{corpus, program};
use crate::toy::{Inst, Op, Var};

#[test]
fn copy_and_alias_propagation_preserve_behavior() {
    for case in corpus() {
        let mut copied = case.cfg.clone();
        copy_propagation(&mut copied);
        assert_same_behavior(&case, "copy propagation", &case.cfg, &copied);
        assert_well_formed(case.name, "copy propagation", &copied);

        let mut aliased = case.cfg.clone();
        alias_propagation(&mut aliased);
        assert_same_behavior(&case, "alias propagation", &case.cfg, &aliased);

        let mut kept = case.cfg.clone();
        copy_propagation_with_exits(&mut kept, |_| Vec::new());
        assert_eq!(
            kept, copied,
            "{}: the empty exit set is the plain pass",
            case.name
        );
    }
}

#[test]
fn dead_code_elimination_removes_exactly_the_reported_instructions_and_no_effect() {
    let effectful = |cfg: &Cfg<Inst<Var>>| {
        cfg.blocks()
            .flat_map(BasicBlock::instructions)
            .filter(|inst| !inst.effects().is_empty() || inst.may_unwind())
            .count()
    };
    for case in corpus() {
        let dead = DeadCode::compute(&case.cfg);
        for point in &dead.instructions {
            let inst = &case.cfg.block(point.block).instructions()[point.inst_idx];
            assert!(
                inst.effects().is_empty() && !inst.may_unwind() && !inst.defs().is_empty(),
                "{}: DeadCode lists {inst:?}",
                case.name
            );
        }
        let reference = Reference::new(&case.cfg);
        let unreachable: Vec<BlockId> = case
            .cfg
            .block_ids()
            .filter(|block| !reference.reachable.contains(block))
            .collect();
        assert_eq!(
            dead.unreachable_blocks, unreachable,
            "{}: unreachable blocks",
            case.name
        );

        let mut eliminated = case.cfg.clone();
        assert_eq!(
            dead_code_elimination(&mut eliminated),
            dead.instructions.len(),
            "{}",
            case.name
        );
        assert_eq!(
            case.cfg
                .blocks()
                .map(|block| block.instructions().len())
                .sum::<usize>()
                - eliminated
                    .blocks()
                    .map(|block| block.instructions().len())
                    .sum::<usize>(),
            dead.instructions.len(),
            "{}: removed instruction count",
            case.name
        );
        assert_eq!(
            effectful(&case.cfg),
            effectful(&eliminated),
            "{}: effectful instructions",
            case.name
        );
        assert_same_behavior(&case, "dead code elimination", &case.cfg, &eliminated);
    }
}

#[test]
fn remove_dead_code_reaches_a_fixpoint_with_nothing_left_to_report() {
    for case in corpus() {
        let mut cleaned = case.cfg.clone();
        remove_dead_code(&mut cleaned);
        assert_same_behavior(&case, "remove_dead_code", &case.cfg, &cleaned);
        assert_well_formed(case.name, "remove_dead_code", &cleaned);
        let remaining = DeadCode::compute(&cleaned);
        assert!(remaining.is_empty(), "{}: {remaining:?}", case.name);
    }
}

#[test]
fn dead_code_reports_the_chain_feeding_only_dead_definitions() {
    let case = program("copy_chain");
    let entry = case.cfg.entry();
    let dead = DeadCode::compute(&case.cfg);
    assert_eq!(
        dead.instructions,
        [4, 5].map(|inst_idx| ProgramPoint {
            block: entry,
            inst_idx
        })
    );
    assert_eq!(dead.unreachable_blocks, Vec::<BlockId>::new());
}

#[test]
fn dead_code_keeps_a_throwing_instruction_and_what_its_handler_reads() {
    let mut cfg = Cfg::<Inst<Var>>::new();
    let entry = cfg.entry();
    let done = cfg.new_block();
    let handler = cfg.new_block();
    for inst in [
        Inst::add("t", "a", "a"),
        Inst::add("junk", "t", "t"),
        Inst::divide("unused", "a", "b"),
    ] {
        cfg.block_mut(entry).push(inst);
    }
    cfg.block_mut(done).push(Inst::ret("a"));
    cfg.block_mut(handler).push(Inst::ret("t"));
    cfg.add_edge(entry, done, EdgeKind::Fallthrough);
    cfg.add_edge(entry, handler, EdgeKind::ExceptionUnwind);

    let dead = DeadCode::compute(&cfg);
    assert_eq!(
        dead.instructions,
        [ProgramPoint {
            block: entry,
            inst_idx: 1
        }]
    );
    let mut eliminated = cfg.clone();
    assert_eq!(dead_code_elimination(&mut eliminated), 1);
    for inputs in input_vectors(&["a", "b"]) {
        assert_eq!(run(&eliminated, &inputs).0, run(&cfg, &inputs).0);
    }
}

#[test]
fn copy_propagation_collapses_a_copy_chain_to_its_source() {
    let case = program("copy_chain");
    let mut cfg = case.cfg.clone();
    let stats = copy_propagation(&mut cfg);
    assert_eq!(stats.uses_rewritten, 4);
    assert_eq!(stats.instructions_removed, 3);
    let entry = cfg.block(cfg.entry()).instructions();
    assert_eq!(entry.last().expect("the return remains").uses, [Var("t")]);
    assert!(entry.iter().all(|inst| inst.op != Op::Copy));
}

#[test]
fn copy_propagation_leaves_a_copy_whose_source_the_loop_redefines() {
    let case = program("loop_carried_copy");
    let mut cfg = case.cfg.clone();
    let stats = copy_propagation(&mut cfg);
    assert_eq!((stats.uses_rewritten, stats.instructions_removed), (0, 0));
    assert_eq!(cfg, case.cfg);
}

#[test]
fn alias_propagation_forwards_parallel_copy_pairs_that_copy_propagation_ignores() {
    let case = program("single_definition_parallel_copy");
    let mut copied = case.cfg.clone();
    let stats = copy_propagation(&mut copied);
    assert_eq!((stats.uses_rewritten, stats.instructions_removed), (0, 0));

    let mut aliased = case.cfg.clone();
    let stats = alias_propagation(&mut aliased);
    assert_eq!((stats.uses_rewritten, stats.instructions_removed), (2, 1));
    let entry = aliased.block(aliased.entry()).instructions();
    assert_eq!(entry[0].op, Op::Add);
    assert_eq!(entry[0].uses, [Var("a"), Var("b")]);
}

#[test]
fn copy_propagation_forwards_a_copy_to_both_arms_of_a_diamond_and_past_the_join() {
    let case = program("copies_across_a_diamond");
    let mut cfg = case.cfg.clone();
    let stats = copy_propagation(&mut cfg);
    // The four readers of `v` are the comparison, both arms, and the return.
    assert_eq!((stats.uses_rewritten, stats.instructions_removed), (4, 1));
    assert!(
        cfg.blocks()
            .flat_map(BasicBlock::instructions)
            .all(|inst| !inst.uses.contains(&Var("v")) && !inst.defs.contains(&Var("v")))
    );
    assert_same_behavior(&case, "copy propagation", &case.cfg, &cfg);
}

#[test]
fn copy_propagation_forwards_a_copy_of_an_input_into_a_loop() {
    let case = program("input_copy_read_in_loop");
    let mut cfg = case.cfg.clone();
    let stats = copy_propagation(&mut cfg);
    assert_eq!((stats.uses_rewritten, stats.instructions_removed), (1, 1));
    assert!(
        cfg.blocks()
            .flat_map(BasicBlock::instructions)
            .all(|inst| !inst.uses.contains(&Var("k")))
    );
    assert_same_behavior(&case, "copy propagation", &case.cfg, &cfg);
}

#[test]
fn copy_propagation_with_exits_keeps_a_copy_whose_destination_leaves_the_function() {
    let mut cfg = Cfg::<Inst<Var>>::new();
    cfg.block_mut(cfg.entry()).push(Inst::copy("result", "a"));
    let mut handed_on = cfg.clone();

    let stats = copy_propagation(&mut cfg);
    assert_eq!(
        stats.instructions_removed, 1,
        "nothing inside reads the copy"
    );

    let stats = copy_propagation_with_exits(&mut handed_on, |_| vec![Var("result")]);
    assert_eq!(
        stats.instructions_removed, 0,
        "the caller reads it after the return"
    );
    assert_eq!(handed_on.block(handed_on.entry()).instructions().len(), 1);
}

#[test]
fn dead_code_with_exits_keeps_what_the_caller_reads_and_what_feeds_it() {
    let mut cfg = Cfg::<Inst<Var>>::new();
    cfg.block_mut(cfg.entry()).push(Inst::add("x", "a", "b"));
    cfg.block_mut(cfg.entry())
        .push(Inst::add("result", "x", "x"));
    cfg.block_mut(cfg.entry()).push(Inst::add("junk", "a", "a"));
    let entry = cfg.entry();
    let point = |inst_idx| ProgramPoint {
        block: entry,
        inst_idx,
    };

    assert_eq!(
        DeadCode::compute(&cfg).instructions,
        [point(0), point(1), point(2)]
    );
    let with_exits = DeadCode::compute_with_exits(&cfg, |_| vec![Var("result")]);
    assert_eq!(with_exits.instructions, [point(2)]);
    let mut eliminated = cfg.clone();
    assert_eq!(
        dead_code_elimination_with_exits(&mut eliminated, |_| vec![Var("result")]),
        1
    );
    assert_eq!(eliminated.block(entry).instructions().len(), 2);
}

#[test]
fn dead_code_roots_retain_an_observed_instruction_and_the_chain_feeding_it() {
    let case = program("copy_chain");
    let entry = case.cfg.entry();
    let point = |inst_idx| ProgramPoint {
        block: entry,
        inst_idx,
    };
    // `junk` (4) feeds `more_junk` (5); observing the consumer keeps both.
    let observed_consumer =
        DeadCode::compute_with_roots(&case.cfg, |_| Vec::new(), |at, _| at == point(5));
    assert_eq!(observed_consumer.instructions, Vec::<ProgramPoint>::new());
    // Observing only the producer leaves the consumer dead.
    let observed_producer =
        DeadCode::compute_with_roots(&case.cfg, |_| Vec::new(), |at, _| at == point(4));
    assert_eq!(observed_producer.instructions, [point(5)]);
}
