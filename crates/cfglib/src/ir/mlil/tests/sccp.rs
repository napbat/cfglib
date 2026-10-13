//! Sparse conditional constant propagation.

extern crate alloc;

use alloc::vec::Vec;

use super::toy::{Block, Function, Op, Value};
use crate::ir::mlil::{Lattice, SccpFacts, UseIndex};

fn propagate(function: &Function) -> SccpFacts<i64> {
    SccpFacts::compute(function, &UseIndex::build(function))
}

/// A function whose entry block switches on a constant.
struct Branching {
    entry: Block,
    then: Block,
    other: Block,
    join: Block,
    /// The parameter of `join`.
    merged: Value,
    /// The sum of `merged` with itself.
    doubled: Value,
}

/// Builds a function whose `entry` switches on `scrutinee` to `then` (case 1)
/// and `other` (default). The arms pass 5 and 7 to the parameter of `join`.
fn branching(function: &mut Function, scrutinee: i64) -> Branching {
    let entry = function.block();
    let then = function.block();
    let other = function.block();
    let join = function.block();
    let merged = function.param(join);

    let selector = function.value(entry, Op::Const(scrutinee), &[]);
    function.terminate(
        entry,
        Op::Switch,
        &[selector],
        &[(then, &[], Some(1)), (other, &[], None)],
    );
    let five = function.value(then, Op::Const(5), &[]);
    function.terminate(then, Op::Jump, &[], &[(join, &[five], None)]);
    let seven = function.value(other, Op::Const(7), &[]);
    function.terminate(other, Op::Jump, &[], &[(join, &[seven], None)]);
    let doubled = function.value(join, Op::Add, &[merged, merged]);
    function.emit(join, Op::Return, &[doubled], 0);
    Branching {
        entry,
        then,
        other,
        join,
        merged,
        doubled,
    }
}

#[test]
fn a_decided_branch_leaves_the_other_arm_unexecuted() {
    let mut function = Function::new();
    let shape = branching(&mut function, 1);
    let facts = propagate(&function);

    assert!(facts.is_block_executable(shape.entry));
    assert!(facts.is_block_executable(shape.then));
    assert!(!facts.is_block_executable(shape.other));
    assert!(facts.is_block_executable(shape.join));
    assert_eq!(
        facts.executable_blocks::<Block>().collect::<Vec<_>>(),
        [shape.entry, shape.then, shape.join]
    );

    assert!(facts.is_edge_executable(shape.entry, 0));
    assert!(!facts.is_edge_executable(shape.entry, 1));
    assert!(facts.is_edge_executable(shape.then, 0));
    assert!(!facts.is_edge_executable(shape.other, 0));

    assert_eq!(facts.value(shape.merged), &Lattice::Constant(5));
    assert_eq!(facts.value(shape.doubled), &Lattice::Constant(10));
}

#[test]
fn the_default_edge_is_taken_when_no_case_matches() {
    let mut function = Function::new();
    let shape = branching(&mut function, 9);
    let facts = propagate(&function);

    assert!(!facts.is_block_executable(shape.then));
    assert!(facts.is_block_executable(shape.other));
    assert!(!facts.is_edge_executable(shape.entry, 0));
    assert!(facts.is_edge_executable(shape.entry, 1));
    assert_eq!(facts.value(shape.merged), &Lattice::Constant(7));
    assert!(facts.is_block_executable(shape.join));
}

#[test]
fn an_undecided_branch_merges_both_arms_into_a_varying_value() {
    let mut function = Function::new();
    let entry = function.block();
    let left = function.block();
    let right = function.block();
    let join = function.block();
    let merged = function.param(join);
    let selector = function.value(entry, Op::Load, &[]);
    function.terminate(
        entry,
        Op::Switch,
        &[selector],
        &[(left, &[], Some(1)), (right, &[], None)],
    );
    let five = function.value(left, Op::Const(5), &[]);
    function.terminate(left, Op::Jump, &[], &[(join, &[five], None)]);
    let seven = function.value(right, Op::Const(7), &[]);
    function.terminate(right, Op::Jump, &[], &[(join, &[seven], None)]);
    function.emit(join, Op::Return, &[merged], 0);

    let facts = propagate(&function);
    assert!(facts.is_edge_executable(entry, 0));
    assert!(facts.is_edge_executable(entry, 1));
    assert_eq!(facts.value(selector), &Lattice::Varying);
    assert_eq!(facts.value(merged), &Lattice::Varying);
}

#[test]
fn two_arms_that_agree_keep_the_constant() {
    let mut function = Function::new();
    let entry = function.block();
    let left = function.block();
    let right = function.block();
    let join = function.block();
    let merged = function.param(join);
    let selector = function.value(entry, Op::Load, &[]);
    function.terminate(
        entry,
        Op::Switch,
        &[selector],
        &[(left, &[], Some(1)), (right, &[], None)],
    );
    for arm in [left, right] {
        let value = function.value(arm, Op::Const(4), &[]);
        function.terminate(arm, Op::Jump, &[], &[(join, &[value], None)]);
    }
    function.emit(join, Op::Return, &[merged], 0);

    assert_eq!(propagate(&function).value(merged), &Lattice::Constant(4));
}

#[test]
fn a_loop_invariant_stays_constant_and_a_counter_becomes_varying() {
    let mut function = Function::new();
    let entry = function.block();
    let header = function.block();
    let body = function.block();
    let exit = function.block();
    let invariant = function.param(header);
    let counter = function.param(header);

    let three = function.value(entry, Op::Const(3), &[]);
    let zero = function.value(entry, Op::Const(0), &[]);
    function.terminate(entry, Op::Jump, &[], &[(header, &[three, zero], None)]);

    let more = function.value(header, Op::Load, &[]);
    function.terminate(
        header,
        Op::Switch,
        &[more],
        &[(body, &[], Some(1)), (exit, &[], None)],
    );
    let one = function.value(body, Op::Const(1), &[]);
    let next = function.value(body, Op::Add, &[counter, one]);
    function.terminate(body, Op::Jump, &[], &[(header, &[invariant, next], None)]);
    function.emit(exit, Op::Return, &[invariant], 0);

    let facts = propagate(&function);
    assert_eq!(facts.value(invariant), &Lattice::Constant(3));
    assert_eq!(facts.value(counter), &Lattice::Varying);
    assert_eq!(facts.value(next), &Lattice::Varying);
    assert!(facts.is_block_executable(exit));
}

#[test]
fn the_parameters_of_the_entry_block_vary() {
    let mut function = Function::new();
    let entry = function.block();
    let argument = function.param(entry);
    let one = function.value(entry, Op::Const(1), &[]);
    let sum = function.value(entry, Op::Add, &[argument, one]);
    function.emit(entry, Op::Return, &[sum], 0);

    let facts = propagate(&function);
    assert_eq!(facts.value(argument), &Lattice::Varying);
    assert_eq!(facts.value(one), &Lattice::Constant(1));
    assert_eq!(facts.value(sum), &Lattice::Varying);
}

#[test]
fn a_value_of_an_unreachable_block_stays_undefined() {
    let mut function = Function::new();
    let entry = function.block();
    let unreachable = function.block();
    function.emit(entry, Op::Return, &[], 0);
    let orphan = function.value(unreachable, Op::Const(9), &[]);
    function.emit(unreachable, Op::Return, &[], 0);

    let facts = propagate(&function);
    assert!(!facts.is_block_executable(unreachable));
    assert_eq!(facts.value(orphan), &Lattice::Undefined);
}
