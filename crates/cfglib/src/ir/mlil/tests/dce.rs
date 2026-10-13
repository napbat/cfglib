//! Dead-code marking.

use super::toy::{Function, Inst, Op};
use crate::ir::mlil::{DeadInstructions, UseIndex};

fn mark(function: &Function) -> DeadInstructions<Inst> {
    DeadInstructions::compute(function, &UseIndex::build(function))
}

#[test]
fn an_unused_pure_chain_is_dead_from_its_end_to_its_roots() {
    let mut function = Function::new();
    let entry = function.block();
    let (first, a) = function.emit(entry, Op::Const(1), &[], 1);
    let (second, b) = function.emit(entry, Op::Const(2), &[], 1);
    let (sum, total) = function.emit(entry, Op::Add, &[a[0], b[0]], 1);
    let (doubled, twice) = function.emit(entry, Op::Add, &[total[0], total[0]], 1);
    let (copy, _) = function.emit(entry, Op::Copy, &[twice[0]], 1);
    let (ret, _) = function.emit(entry, Op::Return, &[], 0);

    let dead = mark(&function);
    for inst in [first, second, sum, doubled, copy] {
        assert!(dead.contains(inst));
    }
    assert!(!dead.contains(ret));
    assert_eq!(dead.len(), 5);
}

#[test]
fn effectful_operations_and_their_operands_stay() {
    let mut function = Function::new();
    let entry = function.block();
    let (address_inst, address) = function.emit(entry, Op::Const(64), &[], 1);
    let (load, _) = function.emit(entry, Op::Load, &[address[0]], 1);
    let (stored_inst, stored) = function.emit(entry, Op::Const(4), &[], 1);
    let (store, _) = function.emit(entry, Op::Store, &[address[0], stored[0]], 0);
    let (ret, _) = function.emit(entry, Op::Return, &[], 0);

    let dead = mark(&function);
    for kept in [address_inst, load, stored_inst, store, ret] {
        assert!(!dead.contains(kept));
    }
    assert!(dead.is_empty());
}

#[test]
fn a_successor_argument_is_a_use() {
    let mut function = Function::new();
    let entry = function.block();
    let target = function.block();
    function.param(target);
    let (argument_inst, argument) = function.emit(entry, Op::Const(7), &[], 1);
    function.terminate(entry, Op::Jump, &[], &[(target, &[argument[0]], None)]);
    function.emit(target, Op::Return, &[], 0);

    let dead = mark(&function);
    assert!(!dead.contains(argument_inst));
    assert!(dead.is_empty());
}

#[test]
fn a_pure_instruction_without_results_is_dead() {
    let mut function = Function::new();
    let entry = function.block();
    let (nop, _) = function.emit(entry, Op::Nop, &[], 0);
    function.emit(entry, Op::Return, &[], 0);

    assert_eq!(mark(&function).as_slice(), [nop]);
}

#[test]
fn an_instruction_with_one_used_result_stays_with_its_operands() {
    let mut function = Function::new();
    let entry = function.block();
    let (source, input) = function.emit(entry, Op::Const(1), &[], 1);
    let (copy, results) = function.emit(entry, Op::Copy, &[input[0]], 2);
    let (orphan, _) = function.emit(entry, Op::Const(2), &[], 1);
    function.emit(entry, Op::Store, &[results[0], results[0]], 0);
    function.emit(entry, Op::Return, &[], 0);

    let dead = mark(&function);
    assert!(!dead.contains(source));
    assert!(!dead.contains(copy));
    assert_eq!(dead.as_slice(), [orphan]);
}

#[test]
fn a_dead_user_does_not_free_an_operand_that_another_user_needs() {
    let mut function = Function::new();
    let entry = function.block();
    let (shared_inst, shared) = function.emit(entry, Op::Const(1), &[], 1);
    let (dead_user, _) = function.emit(entry, Op::Add, &[shared[0], shared[0]], 1);
    function.emit(entry, Op::Store, &[shared[0], shared[0]], 0);
    function.emit(entry, Op::Return, &[], 0);

    let dead = mark(&function);
    assert_eq!(dead.as_slice(), [dead_user]);
    assert!(!dead.contains(shared_inst));
}
