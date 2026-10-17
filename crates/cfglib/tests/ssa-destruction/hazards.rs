//! The swap and lost-copy problems, and copies on unwind edges, checked by
//! running the lowered program.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{BasicBlock, EdgeKind, PhiCopyOrigin, copies_by_predecessor, eliminate_phis};

use crate::lowering::{
    StartCopies, Style, assert_lowering_preserves_behavior, lower, lower_with, lowered_behaviors,
    lowered_inputs,
};
use crate::machine::{Opcode, Reg, execute, input_vectors, try_execute};
use crate::programs::{
    analyze, dead_phi_diamond, departure, diamond, entry_swap_loop, handler_program,
    loop_with_early_continue, lost_copy_loop, nested_loops, swap_loop, two_phi_diamond,
};

#[test]
fn lowering_preserves_behavior_with_parallel_or_sequenced_copies() {
    let programs = [
        ("diamond", diamond()),
        ("two_phi_diamond", two_phi_diamond()),
        ("dead_phi_diamond", dead_phi_diamond()),
        ("swap_loop", swap_loop()),
        ("lost_copy_loop", lost_copy_loop()),
        ("loop_with_early_continue", loop_with_early_continue()),
        ("nested_loops", nested_loops()),
        ("entry_swap_loop", entry_swap_loop()),
    ];
    for (name, program) in &programs {
        for propagate_moves in [false, true] {
            assert_lowering_preserves_behavior(name, program, Style::Parallel, propagate_moves);
            assert_lowering_preserves_behavior(name, program, Style::Sequenced, propagate_moves);
        }
    }
}

#[test]
fn the_swap_problem_is_a_phi_cycle_that_only_a_parallel_assignment_resolves() {
    let program = swap_loop();
    let ssa = analyze(&program.cfg);
    // After forwarding the moves, the back edge copies (x, y) = (y, x) among
    // its other copies: two destinations that are each other's source.
    let lowered = lower(&program.cfg, &ssa, true, Style::Parallel);
    let swaps_a_pair = lowered
        .blocks()
        .flat_map(BasicBlock::instructions)
        .filter(|insn| insn.opcode == Opcode::ParallelMove)
        .any(|insn| {
            let pairs: Vec<_> = insn.defs.iter().zip(&insn.uses).collect();
            pairs.iter().any(|(destination, source)| {
                destination != source
                    && pairs.iter().any(|(other_destination, other_source)| {
                        other_destination == source && other_source == destination
                    })
            })
        });
    assert!(swaps_a_pair, "the back edge exchanges two registers");

    for (lowered, original) in lowered_behaviors(&program, Style::Parallel, true, true) {
        assert_eq!(lowered, original);
    }
    for (lowered, original) in lowered_behaviors(&program, Style::Sequenced, true, true) {
        assert_eq!(lowered, original);
    }
    // Plain moves in listing order lose one of the two values.
    let naive = lowered_behaviors(&program, Style::Naive, true, true);
    assert!(naive.iter().any(|(lowered, original)| lowered != original));
}

#[test]
fn the_lost_copy_problem_needs_the_critical_edge_split_before_copies_are_placed() {
    let program = lost_copy_loop();
    for (lowered, original) in lowered_behaviors(&program, Style::Parallel, true, true) {
        assert_eq!(lowered, original);
    }
    // Without the split the back edge's copy runs before the branch, so the
    // exit reads the incremented value instead of the previous one.
    let unsplit = lowered_behaviors(&program, Style::Parallel, true, false);
    assert!(
        unsplit
            .iter()
            .any(|(lowered, original)| lowered != original)
    );
}

#[test]
fn unwind_edge_copies_depart_before_each_throwing_instruction() {
    let program = handler_program();
    let ssa = analyze(&program.cfg);
    let entry = program.cfg.entry();
    let handler = program
        .cfg
        .block_ids()
        .find(|block| {
            program
                .cfg
                .incoming(*block)
                .any(|edge| program.cfg.edge(edge).kind() == EdgeKind::ExceptionUnwind)
        })
        .expect("the program has a handler");
    // `t` is written by instructions 0 and 2; the divisions are 1 and 3.
    let before_first = ssa.block(entry).instructions[0].defs()[0].clone();
    let before_second = ssa.block(entry).instructions[2].defs()[0].clone();
    let copies = eliminate_phis(&ssa);
    let phi = ssa
        .block(handler)
        .phis
        .iter()
        .find(|phi| phi.result.variable == Reg("t"))
        .expect("the handler merges t");
    let for_t: Vec<_> = copies
        .iter()
        .filter(|copy| copy.destination == phi.result)
        .collect();
    assert_eq!(for_t.len(), 2);
    assert!(for_t.iter().any(
        |copy| copy.from == PhiCopyOrigin::Departure(departure(entry, 1))
            && copy.source == before_first
    ));
    assert!(for_t.iter().any(
        |copy| copy.from == PhiCopyOrigin::Departure(departure(entry, 3))
            && copy.source == before_second
    ));

    // One parallel group per throwing instruction, none at the block end.
    let groups = copies_by_predecessor(&copies);
    let points: BTreeSet<_> = groups.iter().map(|group| group.from).collect();
    assert_eq!(
        points,
        BTreeSet::from([
            PhiCopyOrigin::Departure(departure(entry, 1)),
            PhiCopyOrigin::Departure(departure(entry, 3)),
        ])
    );

    for (lowered, original) in lowered_behaviors(&program, Style::Parallel, false, true) {
        assert_eq!(lowered, original);
    }
}

#[test]
fn a_loop_headed_by_the_entry_lowers_with_its_start_copies_in_front_of_the_entry() {
    let program = entry_swap_loop();
    let ssa = analyze(&program.cfg);
    for style in [Style::Parallel, Style::Sequenced] {
        for propagate_moves in [false, true] {
            let lowered = lower(&program.cfg, &ssa, propagate_moves, style);
            assert_ne!(
                lowered.entry(),
                program.cfg.entry(),
                "the start copies need a block of their own in front of the entry"
            );
            for inputs in input_vectors(program.inputs) {
                assert_eq!(
                    execute(&lowered, &lowered_inputs(&program.cfg, &inputs)),
                    execute(&program.cfg, &inputs),
                    "{propagate_moves}, {inputs:?}"
                );
            }
        }
    }
}

#[test]
fn start_copies_at_the_top_of_an_entry_that_heads_a_loop_run_again_on_every_back_edge() {
    // The start copies reset every carried register to its live-in, so a
    // loop that runs at all never reaches its exit, while a loop that does
    // not run sees the copies once either way.
    let program = entry_swap_loop();
    let ssa = analyze(&program.cfg);
    let lowered = lower_with(
        &program.cfg,
        &ssa,
        true,
        Style::Parallel,
        StartCopies::AtTopOfEntry,
    );
    let run = |n: i64, i: i64| {
        let inputs = BTreeMap::from([(Reg("n"), n), (Reg("i"), i), (Reg("x"), 1), (Reg("y"), 2)]);
        (
            try_execute(&lowered, &lowered_inputs(&program.cfg, &inputs)),
            execute(&program.cfg, &inputs),
        )
    };

    let (lowered_without_iterations, original) = run(0, 5);
    assert_eq!(lowered_without_iterations, Some(original));
    let (lowered_with_iterations, original) = run(3, 0);
    assert_eq!(
        lowered_with_iterations, None,
        "the loop never ends, but the program returns {original:?}"
    );
}
