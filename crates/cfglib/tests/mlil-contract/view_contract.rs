//! The documented contract of `ir::mlil::FunctionView`, on valid functions and
//! on deliberately broken ones.
//!
//! No MLIL verifier ships with `cfglib`, so the contract is checked by the
//! consumer-side checker of `contract.rs`. The adjacency rule is also within
//! reach of the public `verify_view`, which is asserted next to it.

use cfglib::ir::mlil::{FunctionView, ValueDef};
use cfglib::{DenseId, GraphView, RootedView, verify_view};

use crate::contract::{Violation, check};
use crate::fixtures::{self, branch, jump, ret};
use crate::glint::{Arm, Block, Function, Glint, Op, Ty, Value};
use crate::random_functions::random_function;
use crate::skewed::{Fault, Skewed};

fn violations(function: &Function) -> Vec<Violation> {
    check::<Glint, _>(function, &Op::is_terminator)
}

fn sorted<I: DenseId>(items: impl Iterator<Item = I>) -> Vec<usize> {
    let mut indices: Vec<usize> = items.map(DenseId::index).collect();
    indices.sort_unstable();
    indices
}

#[test]
fn every_fixture_satisfies_the_documented_contract() {
    for (name, function) in fixtures::all() {
        assert_eq!(violations(&function), [], "{name}");
    }
}

#[test]
fn random_functions_satisfy_the_documented_contract() {
    for seed in 0..300 {
        assert_eq!(violations(&random_function(seed)), [], "seed {seed}");
    }
}

#[test]
fn the_graph_view_of_every_valid_function_passes_verify_view() {
    for (name, function) in fixtures::all() {
        let report = verify_view(&function);
        assert!(report.is_ok(), "{name}: {report:?}");
    }
    for seed in 0..300 {
        let report = verify_view(&random_function(seed));
        assert!(report.is_ok(), "seed {seed}: {report:?}");
    }
}

#[test]
fn successor_records_give_the_adjacency_with_parallel_records_kept_parallel() {
    let function = fixtures::switch_with_parallel_records();
    let [entry, shared, other, fallback, join] = [0, 1, 2, 3, 4].map(Block::from_index);
    // Cases 1 and 2 both name `shared`, so the entry lists it twice.
    assert_eq!(
        sorted(GraphView::successors(&function, entry)),
        [1, 1, 2, 3]
    );
    assert_eq!(sorted(GraphView::predecessors(&function, shared)), [0, 0]);
    assert_eq!(sorted(GraphView::predecessors(&function, join)), [1, 2, 3]);
    assert_eq!(
        sorted(GraphView::predecessors(&function, other)),
        [entry.index()]
    );
    assert_eq!(
        sorted(GraphView::predecessors(&function, fallback)),
        [entry.index()]
    );
    assert_eq!(GraphView::predecessors(&function, entry).count(), 0);
    // The records carry the edge metadata in terminator order.
    let arms: Vec<Arm> = FunctionView::successors(&function, entry)
        .map(|record| record.edge.clone())
        .collect();
    assert_eq!(
        arms,
        [Arm::Case(1), Arm::Case(2), Arm::Case(3), Arm::Default]
    );
}

#[test]
fn a_removed_block_is_not_a_node_but_its_identities_stay_below_the_bounds() {
    let function = fixtures::with_a_removed_block();
    let live: Vec<usize> = GraphView::node_ids(&function).map(DenseId::index).collect();
    assert_eq!(live, [0, 2]);
    assert_eq!(GraphView::node_bound(&function), 3);
    // The removed block held two values' worth of instructions that no live
    // block lists: the bounds still cover them.
    let listed: usize = live
        .iter()
        .map(|&block| FunctionView::insts(&function, Block::from_index(block)).len())
        .sum();
    assert_eq!(listed, 3);
    assert_eq!(FunctionView::<Glint>::inst_bound(&function), 5);
    assert_eq!(FunctionView::<Glint>::value_bound(&function), 2);
    assert_eq!(violations(&function), []);
}

#[test]
fn every_value_is_defined_where_the_view_says() {
    let function = fixtures::counting_sum();
    let [entry, header, body, exit] = [0, 1, 2, 3].map(Block::from_index);
    let n = FunctionView::params(&function, entry)[0];
    assert_eq!(
        FunctionView::def(&function, n),
        ValueDef::Param {
            block: entry,
            index: 0
        }
    );
    let [i, acc] = [0, 1].map(|index| FunctionView::params(&function, header)[index]);
    assert_eq!(
        FunctionView::def(&function, acc),
        ValueDef::Param {
            block: header,
            index: 1
        }
    );
    let accumulate = FunctionView::insts(&function, body)[0];
    let result = FunctionView::results(&function, accumulate)[0];
    assert_eq!(FunctionView::op(&function, accumulate), &Op::Add);
    assert_eq!(FunctionView::operands(&function, accumulate), [acc, i]);
    assert_eq!(FunctionView::block_of(&function, accumulate), body);
    assert_eq!(
        FunctionView::def(&function, result),
        ValueDef::Result {
            inst: accumulate,
            index: 0
        }
    );
    assert_eq!(FunctionView::ty(&function, result), &Ty::I32);
    assert_eq!(FunctionView::params(&function, exit).len(), 1);
}

#[test]
fn a_terminator_ends_every_block_and_only_it_has_successor_records() {
    for (name, function) in fixtures::all() {
        for block in GraphView::node_ids(&function) {
            let insts = FunctionView::insts(&function, block);
            let last = *insts.last().expect("every live block has a terminator");
            assert!(FunctionView::op(&function, last).is_terminator(), "{name}");
            for &inst in &insts[..insts.len() - 1] {
                assert!(!FunctionView::op(&function, inst).is_terminator(), "{name}");
            }
        }
    }
}

#[test]
fn a_block_that_ends_before_its_terminator_is_reported() {
    let mut f = Function::new();
    let entry = f.entry();
    let value = f.add_inst(entry, Op::Const(1), &[], &[Ty::I32])[0];
    ret(&mut f, entry, value);
    f.add_inst(entry, Op::Const(2), &[], &[Ty::I32]);
    assert_eq!(
        violations(&f),
        [
            Violation::NotATerminator { block: 0 },
            Violation::TerminatorNotLast {
                block: 0,
                position: 1
            },
        ]
    );
}

#[test]
fn a_block_without_instructions_has_no_terminator() {
    let mut f = Function::new();
    let entry = f.entry();
    let empty = f.add_block();
    jump(&mut f, entry, empty, &[]);
    assert_eq!(violations(&f), [Violation::EmptyBlock { block: 1 }]);
}

#[test]
fn an_operation_that_is_no_terminator_cannot_own_successor_records() {
    let mut f = Function::new();
    let entry = f.entry();
    let exit = f.add_block();
    let x = f.add_param(entry, Ty::I32);
    f.terminate(entry, Op::Add, &[x, x], vec![(exit, vec![], Arm::Always)]);
    ret(&mut f, exit, x);
    assert_eq!(
        violations(&f),
        [
            Violation::NotATerminator { block: 0 },
            Violation::SuccessorsOfNonTerminator { block: 0 },
        ]
    );
}

#[test]
fn successor_arguments_must_match_the_target_parameters_in_count() {
    let mut f = Function::new();
    let entry = f.entry();
    let exit = f.add_block();
    let r = f.add_param(exit, Ty::I32);
    jump(&mut f, entry, exit, &[]);
    ret(&mut f, exit, r);
    assert_eq!(
        violations(&f),
        [Violation::ArgumentCount {
            block: 0,
            record: 0
        }]
    );
}

#[test]
fn successor_arguments_must_match_the_target_parameters_in_type() {
    let mut f = Function::new();
    let entry = f.entry();
    let flag = f.add_param(entry, Ty::Bool);
    let exit = f.add_block();
    let r = f.add_param(exit, Ty::I32);
    jump(&mut f, entry, exit, &[flag]);
    ret(&mut f, exit, r);
    assert_eq!(
        violations(&f),
        [Violation::ArgumentType {
            block: 0,
            record: 0,
            position: 0
        }]
    );
}

#[test]
fn a_record_that_passes_too_many_arguments_is_reported_per_record() {
    let mut f = Function::new();
    let entry = f.entry();
    let x = f.add_param(entry, Ty::I32);
    let flag = f.add_param(entry, Ty::Bool);
    let (taken, other) = (f.add_block(), f.add_block());
    let r = f.add_param(other, Ty::I32);
    branch(&mut f, entry, flag, (taken, &[x]), (other, &[x]));
    ret(&mut f, taken, x);
    ret(&mut f, other, r);
    // The first record passes one argument to a block without parameters;
    // the second is right.
    assert_eq!(
        violations(&f),
        [Violation::ArgumentCount {
            block: 0,
            record: 0
        }]
    );
}

#[test]
fn a_value_that_two_places_define_is_reported() {
    let mut f = Function::new();
    let entry = f.entry();
    let y = f.add_inst(entry, Op::Const(1), &[], &[Ty::I32])[0];
    ret(&mut f, entry, y);
    // The entry lists its own result among its parameters as well.
    f.alias_param(entry, y);
    assert_eq!(
        violations(&f),
        [
            Violation::DefinitionMismatch { value: y.index() },
            Violation::DefinedTwice { value: y.index() },
        ]
    );
}

#[test]
fn def_must_report_the_place_that_lists_the_value() {
    let mut f = Function::new();
    let entry = f.entry();
    let x = f.add_param(entry, Ty::I32);
    ret(&mut f, entry, x);
    f.override_def(
        x,
        ValueDef::Param {
            block: entry,
            index: 3,
        },
    );
    assert_eq!(
        violations(&f),
        [Violation::DefinitionMismatch { value: x.index() }]
    );
}

#[test]
fn block_of_must_name_the_block_that_lists_the_instruction() {
    let mut f = Function::new();
    let entry = f.entry();
    let other = f.add_block();
    let x = f.add_param(entry, Ty::I32);
    jump(&mut f, entry, other, &[]);
    let terminator = f.terminate(other, Op::Return, &[x], Vec::new());
    f.override_block_of(terminator, entry);
    assert_eq!(
        violations(&f),
        [Violation::BlockOfMismatch {
            inst: terminator.index()
        }]
    );
}

#[test]
fn an_operand_that_only_a_removed_block_defines_is_undefined() {
    let mut f = Function::new();
    let entry = f.entry();
    let doomed = f.add_block();
    let stale = f.add_inst(doomed, Op::Const(5), &[], &[Ty::I32])[0];
    ret(&mut f, doomed, stale);
    ret(&mut f, entry, stale);
    f.remove_block(doomed);
    assert_eq!(
        violations(&f),
        [Violation::UndefinedOperand {
            value: stale.index()
        }]
    );
}

#[test]
fn a_use_must_be_dominated_by_its_definition() {
    // The join reads a value that only one arm defines.
    let mut f = Function::new();
    let entry = f.entry();
    let flag = f.add_param(entry, Ty::Bool);
    let (left, right, join) = (f.add_block(), f.add_block(), f.add_block());
    branch(&mut f, entry, flag, (left, &[]), (right, &[]));
    let only_left = f.add_inst(left, Op::Const(1), &[], &[Ty::I32])[0];
    jump(&mut f, left, join, &[]);
    jump(&mut f, right, join, &[]);
    ret(&mut f, join, only_left);
    assert_eq!(
        violations(&f),
        [Violation::NotDominated {
            value: only_left.index(),
            block: join.index()
        }]
    );
}

#[test]
fn an_instruction_cannot_read_its_own_result() {
    let mut f = Function::new();
    let entry = f.entry();
    let zero = f.add_inst(entry, Op::Const(0), &[], &[Ty::I32])[0];
    let own = Value::from_index(FunctionView::<Glint>::value_bound(&f));
    let sum = f.add_inst(entry, Op::Add, &[own, zero], &[Ty::I32])[0];
    assert_eq!(sum, own);
    ret(&mut f, entry, sum);
    assert_eq!(
        violations(&f),
        [Violation::NotDominated {
            value: own.index(),
            block: 0
        }]
    );
}

fn skewed(function: &Function, fault: Fault) -> Skewed<'_> {
    Skewed {
        inner: function,
        fault,
    }
}

#[test]
fn verify_view_accepts_the_control_case_of_the_skewed_view() {
    let function = fixtures::switch_with_parallel_records();
    assert!(verify_view(&skewed(&function, Fault::None)).is_ok());
    assert_eq!(
        check::<Glint, _>(&skewed(&function, Fault::None), &Op::is_terminator),
        []
    );
}

#[test]
fn a_predecessor_missing_from_the_graph_view_is_rejected() {
    let function = fixtures::counting_sum();
    // The header has two predecessors, the entry and the loop body.
    let header = Block::from_index(1);
    let view = skewed(&function, Fault::DropPredecessor(header));
    assert!(verify_view(&view).error_count() >= 1);
    assert_eq!(
        check::<Glint, _>(&view, &Op::is_terminator),
        [Violation::PredecessorMismatch {
            block: header.index()
        }]
    );
}

#[test]
fn a_duplicated_successor_without_a_matching_predecessor_is_rejected() {
    let function = fixtures::absolute_value();
    let entry = function.entry();
    let view = skewed(&function, Fault::ExtraSuccessor(entry));
    assert!(verify_view(&view).error_count() >= 1);
    assert_eq!(
        check::<Glint, _>(&view, &Op::is_terminator),
        [Violation::AdjacencyMismatch {
            block: entry.index()
        }]
    );
}

#[test]
fn a_root_outside_the_node_bound_is_rejected() {
    let function = fixtures::single_block();
    let view = skewed(&function, Fault::RootOutOfBounds);
    assert_eq!(
        RootedView::root(&view).index(),
        GraphView::node_bound(&function)
    );
    assert!(verify_view(&view).error_count() >= 1);
}
