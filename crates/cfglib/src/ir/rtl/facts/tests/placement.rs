//! Block-parameter placement.

use super::toy::{Block, Function, Root, SLOTS, Toy, whole_places};
use crate::dataflow::ssa::DominanceFrontiers;
use crate::graph::dominator::DominatorTree;
use crate::ir::rtl::{
    BitSlice, Bits, BlockSummaries, Control, Dialect, InstructionBuilder, ParamPlacement,
    RootLiveness, RootTable, WriteTarget,
};

/// How the tests seed liveness after the exit.
#[derive(Clone, Copy)]
enum Seed {
    /// Every root is live after the exit.
    AllRoots,
    /// No root is live after the exit, so only reads in the function make a
    /// root live.
    NoRoot,
}

fn place(function: &Function, seed: Seed) -> ParamPlacement<Toy> {
    let table = RootTable::collect(function);
    let summaries = BlockSummaries::compute(function, &table);
    let liveness = match seed {
        Seed::AllRoots => RootLiveness::compute(function, &table, &summaries),
        Seed::NoRoot => RootLiveness::compute_seeded(function, &table, &summaries, |_, _| {}),
    };
    let dominators = DominatorTree::compute(function);
    let frontiers = DominanceFrontiers::compute(function, &dominators);
    ParamPlacement::compute(
        function,
        &dominators,
        &frontiers,
        &table,
        &summaries,
        &liveness,
    )
}

/// Overwrites each of `roots` with a constant at the end of `block`.
fn overwrite(function: &mut Function, block: Block, roots: &[Root]) {
    let mut record = function.record_in(block);
    for &root in roots {
        let value = record.constant_value(1, Toy::root_width(root).width());
        record.write_whole(root, value);
    }
    record.end(Control::Advance);
}

/// Reads each of `roots` at the end of `block`.
fn read(function: &mut Function, block: Block, roots: &[Root]) {
    let mut record = function.record_in(block);
    for &root in roots {
        record.read_whole(root);
    }
    record.end(Control::Advance);
}

/// The blocks of a diamond.
struct Diamond {
    entry: Block,
    left: Block,
    right: Block,
    join: Block,
}

/// Builds `entry → left, right → join`.
///
/// - `entry` overwrites `Acc`, `Tmp`, and `Flag`.
/// - `left` overwrites `Acc`, `Tmp`, and `Flag`.
/// - `right` overwrites `Tmp` and `Flag`.
/// - `join` reads `Acc` and `Flag`. No block reads `Tmp`.
fn diamond(function: &mut Function) -> Diamond {
    let entry = function.block();
    let left = function.block();
    let right = function.block();
    let join = function.block();
    function.flow(entry, &[left, right]);
    function.flow(left, &[join]);
    function.flow(right, &[join]);
    overwrite(function, entry, &[Root::Acc, Root::Tmp, Root::Flag]);
    overwrite(function, left, &[Root::Acc, Root::Tmp, Root::Flag]);
    overwrite(function, right, &[Root::Tmp, Root::Flag]);
    read(function, join, &[Root::Acc, Root::Flag]);
    Diamond {
        entry,
        left,
        right,
        join,
    }
}

#[test]
fn a_diamond_places_a_parameter_for_each_live_root_with_two_definitions() {
    let mut function = Function::new();
    let shape = diamond(&mut function);
    let placement = place(&function, Seed::NoRoot);

    assert_eq!(placement.roots_at(shape.join), [Root::Acc, Root::Flag]);
    for block in [shape.entry, shape.left, shape.right] {
        assert_eq!(placement.roots_at(block), []);
    }
    assert_eq!(placement.param_count(), 2);
}

#[test]
fn a_root_that_nothing_reads_gets_no_parameter() {
    let mut function = Function::new();
    let shape = diamond(&mut function);

    // `Tmp` has a definition in each arm, so the iterated dominance frontier
    // reaches the join. No block reads `Tmp`, so liveness prunes it.
    let pruned = place(&function, Seed::NoRoot);
    assert!(!pruned.roots_at(shape.join).contains(&Root::Tmp));

    // When the caller can read every root, `Tmp` is live at the join.
    let kept = place(&function, Seed::AllRoots);
    assert_eq!(
        kept.roots_at(shape.join),
        [Root::Acc, Root::Tmp, Root::Flag]
    );
}

#[test]
fn a_root_with_one_definition_gets_no_parameter() {
    let mut function = Function::new();
    let entry = function.block();
    let left = function.block();
    let right = function.block();
    let join = function.block();
    function.flow(entry, &[left, right]);
    function.flow(left, &[join]);
    function.flow(right, &[join]);
    overwrite(&mut function, entry, &[Root::Acc]);
    read(&mut function, join, &[Root::Acc]);

    let placement = place(&function, Seed::NoRoot);
    assert_eq!(placement.roots_at(join), []);
    assert_eq!(placement.param_count(), 0);
}

#[test]
fn a_loop_places_a_parameter_at_its_header() {
    let mut function = Function::new();
    let entry = function.block();
    let header = function.block();
    let body = function.block();
    let exit = function.block();
    function.flow(entry, &[header]);
    function.flow(header, &[body, exit]);
    function.flow(body, &[header]);
    overwrite(&mut function, entry, &[Root::Acc, Root::Tmp]);
    read(&mut function, header, &[Root::Acc]);
    overwrite(&mut function, body, &[Root::Acc, Root::Flag]);
    read(&mut function, exit, &[Root::Flag, Root::Top]);

    let placement = place(&function, Seed::NoRoot);
    // `Acc` and `Flag` have a definition in the body, and both are live at
    // the header. `Tmp` has one definition. `Top` is never written.
    assert_eq!(placement.roots_at(header), [Root::Acc, Root::Flag]);
    for block in [entry, body, exit] {
        assert_eq!(placement.roots_at(block), []);
    }
}

#[test]
fn an_unreachable_block_defines_nothing() {
    let mut function = Function::new();
    let entry = function.block();
    let left = function.block();
    let right = function.block();
    let join = function.block();
    let orphan = function.block();
    function.flow(entry, &[left, right]);
    function.flow(left, &[join]);
    function.flow(right, &[join]);
    function.flow(orphan, &[join]);
    overwrite(&mut function, entry, &[Root::Acc]);
    overwrite(&mut function, orphan, &[Root::Tmp]);
    read(&mut function, join, &[Root::Tmp]);

    let placement = place(&function, Seed::NoRoot);
    assert_eq!(placement.roots_at(join), []);
}

/// Builds `entry → left, right → join` where only `left` has a record that
/// `write_in_left` fills, and `join` reads `read_at_join`.
fn one_armed_diamond(
    write_in_left: impl FnOnce(&mut Function, Block),
    read_at_join: &[Root],
) -> (ParamPlacement<Toy>, Block) {
    let mut function = Function::new();
    let entry = function.block();
    let left = function.block();
    let right = function.block();
    let join = function.block();
    function.flow(entry, &[left, right]);
    function.flow(left, &[join]);
    function.flow(right, &[join]);
    overwrite(&mut function, entry, &[Root::Acc, Root::Slot0, Root::Slot1]);
    write_in_left(&mut function, left);
    read(&mut function, join, read_at_join);
    (place(&function, Seed::NoRoot), join)
}

#[test]
fn a_partial_write_is_a_definition_of_its_root() {
    let (placement, join) = one_armed_diamond(
        |function, left| {
            let mut record = function.record_in(left);
            let value = record.constant_value(1, 8);
            let span = BitSlice::new(8, Bits::new(8).unwrap());
            record.write_slice(Root::Acc, span, value);
            record.end(Control::Advance);
        },
        &[Root::Acc],
    );
    assert_eq!(placement.roots_at(join), [Root::Acc]);
}

#[test]
fn an_indexed_write_is_a_definition_of_every_candidate() {
    let (placement, join) = one_armed_diamond(
        |function, left| {
            let places = whole_places(&SLOTS[..2]);
            let mut record = function.record_in(left);
            let index = record.constant_value(1, 3);
            let value = record.constant_value(5, 64);
            record.write(
                WriteTarget::FiniteIndexed {
                    index,
                    candidates: &places,
                    value,
                },
                (),
            );
            record.end(Control::Advance);
        },
        &[Root::Slot0, Root::Slot1],
    );
    assert_eq!(placement.roots_at(join), [Root::Slot0, Root::Slot1]);
}

#[test]
fn a_call_in_one_arm_is_a_definition_of_every_root() {
    let (placement, join) = one_armed_diamond(
        |function, left| {
            let mut record = function.record_in(left);
            let target = record.constant_value(0x1000, 64);
            record.end(Control::Call { target });
        },
        &[Root::Acc, Root::Slot0, Root::Slot1],
    );
    assert_eq!(
        placement.roots_at(join),
        [Root::Acc, Root::Slot0, Root::Slot1]
    );
}
