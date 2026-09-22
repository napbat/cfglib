//! Coalescing a lifter temporary into the variable it is copied to.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::ir::mlil::{Function, FunctionBuilder, Signature, TypedVariable, Variable, VariableId};
use crate::test_util::toy::Span;

use super::{Edge, Operation, ToyDialect, Type};

fn integer(variable: VariableId) -> TypedVariable<ToyDialect> {
    TypedVariable::new(variable, Type::Integer)
}

/// The lift's own variables are the ones declared with role 1 here; a
/// frontend answers from its own web table.
fn is_temporary(variable: &Variable<ToyDialect>) -> bool {
    variable.role == 1
}

/// The shape an x86 lift leaves behind: a value computed into a temporary
/// and committed to a register web by a copy, read from the temporary in
/// between, then recomputed into a second temporary and committed again.
///
/// `v8 = sub(v3, 0x28); v4 = v8; call(.., v8, ..); v9 = add(v8, 0x28);
/// v5 = v9`, with `Constant` standing for the arithmetic and `Store` for
/// the call.
fn migrated_values() -> (Function<ToyDialect>, [VariableId; 4]) {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::migrated".into());
    let body = builder.new_block("body");
    let first_temporary = builder.declare_variable(1, None).unwrap();
    let first_register = builder.declare_variable(0, Some(4)).unwrap();
    let second_temporary = builder.declare_variable(1, None).unwrap();
    let second_register = builder.declare_variable(0, Some(5)).unwrap();
    builder
        .append_instruction(
            body,
            Operation::Constant(40),
            Vec::new(),
            vec![integer(first_temporary)],
            false,
            Some(Span { start: 1, end: 2 }),
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Copy,
            vec![integer(first_temporary)],
            vec![integer(first_register)],
            false,
            Some(Span { start: 2, end: 3 }),
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Store(0),
            vec![integer(first_temporary)],
            Vec::new(),
            false,
            Some(Span { start: 3, end: 4 }),
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Load(1),
            vec![integer(first_temporary)],
            vec![integer(second_temporary)],
            false,
            Some(Span { start: 4, end: 5 }),
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Copy,
            vec![integer(second_temporary)],
            vec![integer(second_register)],
            false,
            Some(Span { start: 5, end: 6 }),
        )
        .unwrap();
    builder
        .append_instruction(body, Operation::Return, Vec::new(), Vec::new(), false, None)
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    (
        builder.finish().unwrap(),
        [
            first_temporary,
            first_register,
            second_temporary,
            second_register,
        ],
    )
}

#[test]
fn a_migrated_value_comes_back_to_the_variable_it_is_copied_to() {
    let (function, [_, first_register, _, second_register]) = migrated_values();
    let (coalesced, removed) = function.coalesce_copies(is_temporary).unwrap();

    assert_eq!(removed, 2, "both commits stop being copies");
    let report = coalesced.verify();
    assert!(report.is_ok(), "{:?}", report.issues);
    let operations: Vec<(&Operation, Vec<VariableId>, Vec<VariableId>)> = coalesced
        .instructions()
        .map(|instruction| {
            (
                instruction.operation(),
                instruction.defs().to_vec(),
                instruction.uses().to_vec(),
            )
        })
        .collect();
    assert_eq!(
        operations,
        vec![
            (&Operation::Constant(40), vec![first_register], Vec::new()),
            (&Operation::Store(0), Vec::new(), vec![first_register]),
            (
                &Operation::Load(1),
                vec![second_register],
                vec![first_register]
            ),
            (&Operation::Return, Vec::new(), Vec::new()),
        ]
    );
}

#[test]
fn the_definition_keeps_its_provenance_and_the_copy_loses_its_own() {
    let (function, _) = migrated_values();
    let (coalesced, _) = function.coalesce_copies(is_temporary).unwrap();

    assert_eq!(
        coalesced.provenance().mappings_from(1).count(),
        1,
        "the rewritten definition keeps its source"
    );
    assert_eq!(
        coalesced.provenance().mappings_from(2).count(),
        0,
        "the removed copy takes its own with it"
    );
    assert_eq!(
        coalesced.instruction_count(),
        4,
        "instruction identities are dense again"
    );
}

#[test]
fn the_coalesced_variable_keeps_its_role_and_native_provenance() {
    let (function, [first_temporary, first_register, _, _]) = migrated_values();
    let (coalesced, _) = function.coalesce_copies(is_temporary).unwrap();

    let register = coalesced
        .variable(first_register)
        .expect("the destination survives");
    assert_eq!(register.role, 0);
    assert_eq!(register.native, Some(4));
    assert!(
        coalesced
            .instructions()
            .all(|instruction| !instruction.uses().contains(&first_temporary)
                && !instruction.defs().contains(&first_temporary)),
        "the temporary stops occurring, and pruning is what drops it"
    );
}

/// `t1 = e; t2 = t1; r = t2` resolves in one pass: every link of the
/// chain is renamed to its end.
#[test]
fn a_chain_of_copies_resolves_in_one_pass() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::chain".into());
    let body = builder.new_block("body");
    let first = builder.declare_variable(1, None).unwrap();
    let second = builder.declare_variable(1, None).unwrap();
    let register = builder.declare_variable(0, Some(7)).unwrap();
    builder
        .append_instruction(
            body,
            Operation::Constant(1),
            Vec::new(),
            vec![integer(first)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Copy,
            vec![integer(first)],
            vec![integer(second)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Copy,
            vec![integer(second)],
            vec![integer(register)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Store(0),
            vec![integer(register)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(body, Operation::Return, Vec::new(), Vec::new(), false, None)
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let (coalesced, removed) = function.coalesce_copies(is_temporary).unwrap();
    assert_eq!(removed, 2, "both links of the chain go in one pass");
    let report = coalesced.verify();
    assert!(report.is_ok(), "{:?}", report.issues);
    let constant = coalesced
        .instructions()
        .find(|instruction| *instruction.operation() == Operation::Constant(1))
        .expect("the definition survives");
    assert_eq!(
        constant.defs(),
        &[register],
        "the definition names the end of the chain"
    );
}

/// The three shapes the coalescing refuses, each for its own reason.
fn refused(twice_defined_temporary: bool, twice_defined_register: bool, parameter: bool) -> usize {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::refused".into());
    let body = builder.new_block("body");
    let temporary = builder.declare_variable(1, None).unwrap();
    let register = builder.declare_variable(0, Some(3)).unwrap();
    builder
        .append_instruction(
            body,
            Operation::Constant(1),
            Vec::new(),
            vec![integer(temporary)],
            false,
            None,
        )
        .unwrap();
    if twice_defined_temporary {
        builder
            .append_instruction(
                body,
                Operation::Constant(2),
                Vec::new(),
                vec![integer(temporary)],
                false,
                None,
            )
            .unwrap();
    }
    builder
        .append_instruction(
            body,
            Operation::Copy,
            vec![integer(temporary)],
            vec![integer(register)],
            false,
            None,
        )
        .unwrap();
    if twice_defined_register {
        builder
            .append_instruction(
                body,
                Operation::Constant(3),
                Vec::new(),
                vec![integer(register)],
                false,
                None,
            )
            .unwrap();
    }
    builder
        .append_instruction(
            body,
            Operation::Store(0),
            vec![integer(register)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(body, Operation::Return, Vec::new(), Vec::new(), false, None)
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    if parameter {
        builder
            .set_signature(Signature::<ToyDialect>::new(
                vec![register],
                vec![Type::Integer],
            ))
            .unwrap();
    }
    let function = builder.finish().unwrap();
    let (coalesced, removed) = function.coalesce_copies(is_temporary).unwrap();
    if removed == 0 {
        assert_eq!(coalesced, function, "a refusal changes nothing");
    }
    removed
}

#[test]
fn a_temporary_with_two_definitions_is_left_alone() {
    assert_eq!(refused(true, false, false), 0);
}

#[test]
fn a_destination_with_two_definitions_is_left_alone() {
    assert_eq!(refused(false, true, false), 0);
}

#[test]
fn a_destination_that_is_a_parameter_is_left_alone() {
    assert_eq!(refused(false, false, true), 0);
}

#[test]
fn the_plain_shape_is_coalesced() {
    assert_eq!(
        refused(false, false, false),
        1,
        "the fixture without any of the refusing traits does coalesce"
    );
}

#[test]
fn a_variable_the_caller_does_not_call_a_temporary_is_left_alone() {
    let (function, _) = migrated_values();
    let (coalesced, removed) = function.coalesce_copies(|_| false).unwrap();
    assert_eq!(removed, 0);
    assert_eq!(coalesced, function);
}
