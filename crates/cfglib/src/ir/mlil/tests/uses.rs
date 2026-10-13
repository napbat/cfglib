//! The use index.

extern crate alloc;

use alloc::vec::Vec;

use super::toy::{Function, Op};
use crate::ir::mlil::{Use, UseIndex};

#[test]
fn the_index_lists_operand_and_edge_uses_in_order() {
    let mut function = Function::new();
    let entry = function.block();
    let join = function.block();
    let param = function.param(join);

    let (_, a) = function.emit(entry, Op::Const(1), &[], 1);
    let (_, b) = function.emit(entry, Op::Const(2), &[], 1);
    let (first, sum) = function.emit(entry, Op::Add, &[a[0], b[0]], 1);
    let (second, twice) = function.emit(entry, Op::Add, &[a[0], a[0]], 1);
    function.terminate(entry, Op::Jump, &[], &[(join, &[sum[0]], None)]);
    let (ret, _) = function.emit(join, Op::Return, &[param], 0);

    let index = UseIndex::build(&function);
    assert_eq!(
        index.uses(a[0]),
        [
            Use::Operand {
                inst: first,
                position: 0
            },
            Use::Operand {
                inst: second,
                position: 0
            },
            Use::Operand {
                inst: second,
                position: 1
            },
        ]
    );
    assert_eq!(
        index.uses(b[0]),
        [Use::Operand {
            inst: first,
            position: 1
        }]
    );
    assert_eq!(
        index.uses(sum[0]),
        [Use::Edge {
            block: entry,
            successor: 0,
            position: 0
        }]
    );
    assert_eq!(
        index.uses(param),
        [Use::Operand {
            inst: ret,
            position: 0
        }]
    );
    assert_eq!(index.uses(twice[0]), []);
    assert_eq!(index.use_count(a[0]), 3);
    assert_eq!(index.use_count(twice[0]), 0);
}

#[test]
fn an_edge_use_names_the_successor_and_the_argument() {
    let mut function = Function::new();
    let entry = function.block();
    let left = function.block();
    let right = function.block();
    let left_param = function.param(left);
    let right_first = function.param(right);
    let right_second = function.param(right);
    let value = function.value(entry, Op::Const(1), &[]);
    let selector = function.value(entry, Op::Load, &[]);
    function.terminate(
        entry,
        Op::Switch,
        &[selector],
        &[(left, &[value], Some(0)), (right, &[selector, value], None)],
    );
    function.emit(left, Op::Return, &[left_param], 0);
    function.emit(right, Op::Return, &[right_first, right_second], 0);

    let index = UseIndex::build(&function);
    let edge_uses: Vec<_> = index
        .uses(value)
        .iter()
        .filter(|value_use| matches!(value_use, Use::Edge { .. }))
        .copied()
        .collect();
    assert_eq!(
        edge_uses,
        [
            Use::Edge {
                block: entry,
                successor: 0,
                position: 0
            },
            Use::Edge {
                block: entry,
                successor: 1,
                position: 1
            },
        ]
    );
}
