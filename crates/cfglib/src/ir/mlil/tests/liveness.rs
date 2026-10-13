//! Value liveness.

use super::toy::{Function, Op, Value};
use crate::ir::mlil::{UseIndex, ValueLiveness};

fn sorted<const N: usize>(mut values: [Value; N]) -> [Value; N] {
    values.sort_unstable();
    values
}

#[test]
fn a_loop_keeps_its_parameters_out_of_the_live_in_sets() {
    let mut function = Function::new();
    let entry = function.block();
    let header = function.block();
    let body = function.block();
    let exit = function.block();
    let index = function.param(header);
    let limit = function.param(header);

    let zero = function.value(entry, Op::Const(0), &[]);
    let bound = function.value(entry, Op::Const(10), &[]);
    function.terminate(entry, Op::Jump, &[], &[(header, &[zero, bound], None)]);

    let below = function.value(header, Op::Lt, &[index, limit]);
    function.terminate(
        header,
        Op::Switch,
        &[below],
        &[(body, &[], Some(1)), (exit, &[], None)],
    );

    let one = function.value(body, Op::Const(1), &[]);
    let next = function.value(body, Op::Add, &[index, one]);
    function.terminate(body, Op::Jump, &[], &[(header, &[next, limit], None)]);
    function.emit(exit, Op::Return, &[index], 0);

    let uses = UseIndex::build(&function);
    let liveness = ValueLiveness::compute(&function, &uses);

    assert_eq!(liveness.live_in(entry), []);
    assert_eq!(liveness.live_out(entry), sorted([zero, bound]));
    assert_eq!(liveness.live_in(header), []);
    assert_eq!(liveness.live_out(header), sorted([index, limit]));
    assert_eq!(liveness.live_in(body), sorted([index, limit]));
    assert_eq!(liveness.live_out(body), sorted([next, limit]));
    assert_eq!(liveness.live_in(exit), [index]);
    assert_eq!(liveness.live_out(exit), []);

    assert!(liveness.is_live_in(body, limit));
    assert!(!liveness.is_live_in(header, limit));
    assert!(liveness.is_live_out(header, index));
    assert!(!liveness.is_live_out(body, index));
}

#[test]
fn a_value_lives_through_the_blocks_between_its_definition_and_its_use() {
    let mut function = Function::new();
    let entry = function.block();
    let left = function.block();
    let right = function.block();
    let join = function.block();

    let value = function.value(entry, Op::Const(7), &[]);
    let selector = function.value(entry, Op::Load, &[]);
    function.terminate(
        entry,
        Op::Switch,
        &[selector],
        &[(left, &[], Some(0)), (right, &[], None)],
    );
    function.terminate(left, Op::Jump, &[], &[(join, &[], None)]);
    function.terminate(right, Op::Jump, &[], &[(join, &[], None)]);
    function.emit(join, Op::Return, &[value], 0);

    let uses = UseIndex::build(&function);
    let liveness = ValueLiveness::compute(&function, &uses);

    assert_eq!(liveness.live_in(entry), []);
    assert_eq!(liveness.live_out(entry), [value]);
    for arm in [left, right] {
        assert_eq!(liveness.live_in(arm), [value]);
        assert_eq!(liveness.live_out(arm), [value]);
    }
    assert_eq!(liveness.live_in(join), [value]);
    assert_eq!(liveness.live_out(join), []);
}

#[test]
fn a_value_used_only_where_it_is_defined_is_never_live_at_a_block_boundary() {
    let mut function = Function::new();
    let entry = function.block();
    let local = function.value(entry, Op::Const(1), &[]);
    function.emit(entry, Op::Return, &[local], 0);

    let uses = UseIndex::build(&function);
    let liveness = ValueLiveness::compute(&function, &uses);
    assert_eq!(liveness.live_in(entry), []);
    assert_eq!(liveness.live_out(entry), []);
}

#[test]
fn an_edge_argument_is_live_out_of_its_block_and_not_live_into_the_target() {
    let mut function = Function::new();
    let entry = function.block();
    let target = function.block();
    let param = function.param(target);
    let argument = function.value(entry, Op::Const(3), &[]);
    function.terminate(entry, Op::Jump, &[], &[(target, &[argument], None)]);
    function.emit(target, Op::Return, &[param], 0);

    let uses = UseIndex::build(&function);
    let liveness = ValueLiveness::compute(&function, &uses);
    assert_eq!(liveness.live_out(entry), [argument]);
    assert_eq!(liveness.live_in(target), []);
}
