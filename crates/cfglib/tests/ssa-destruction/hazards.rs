//! The swap and lost-copy problems, and copies on unwind edges, checked by
//! running the lowered program.

use std::collections::BTreeSet;

use cfglib::{BasicBlock, EdgeKind, copies_by_predecessor, eliminate_phis};

use crate::lowering::{Style, assert_lowering_preserves_behavior, lower, lowered_behaviors};
use crate::machine::{Opcode, Reg};
use crate::programs::{
    analyze, dead_phi_diamond, departure, diamond, handler_program, loop_with_early_continue,
    lost_copy_loop, nested_loops, swap_loop, two_phi_diamond,
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
    assert!(
        for_t
            .iter()
            .any(|copy| copy.from == departure(entry, 1) && copy.source == before_first)
    );
    assert!(
        for_t
            .iter()
            .any(|copy| copy.from == departure(entry, 3) && copy.source == before_second)
    );

    // One parallel group per throwing instruction, none at the block end.
    let groups = copies_by_predecessor(&copies);
    let points: BTreeSet<_> = groups.iter().map(|(point, _)| *point).collect();
    assert_eq!(
        points,
        BTreeSet::from([departure(entry, 1), departure(entry, 3)])
    );

    for (lowered, original) in lowered_behaviors(&program, Style::Parallel, false, true) {
        assert_eq!(lowered, original);
    }
}
