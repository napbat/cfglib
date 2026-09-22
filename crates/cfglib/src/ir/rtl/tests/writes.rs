//! Effect statements that declare the places they write.
//!
//! An effect states the storage an operation writes with values the
//! statement does not express — the registers a call clobbers, the flags
//! an instruction leaves behind. Those writes are the statement's
//! definitions, so each one starts a fresh web, while every operand still
//! observes the state before them.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::ir::hlil::lift_function as lift_hlil;
use crate::ir::mlil;
use crate::test_util::golden::assert_golden;

use super::super::{Expr, Function, FunctionBuilder, Place, ScalarType, Statement, lift};
use super::{Edge, Effect, EffectOp, SemanticDialect, TestDialect, constant, instructions, read};

/// An effect operation over `operands` that writes `writes` without
/// expressing a value for them.
fn emit_writing(
    operands: Vec<Expr<TestDialect>>,
    writes: &[(u8, &[u8])],
) -> Statement<TestDialect> {
    Statement::Effect {
        operation: EffectOp::Emit,
        operands,
        writes: writes
            .iter()
            .map(|&(storage, lanes)| Place {
                storage,
                lanes: lanes.to_vec(),
            })
            .collect(),
        effects: vec![Effect::Emit],
        may_throw: false,
    }
}

fn word(storage: u8) -> Expr<TestDialect> {
    read(storage, &[0], ScalarType::U32)
}

/// An effect clobbering two registers it does not read, followed by a
/// reader of each.
fn clobbering_function() -> Function<TestDialect> {
    let mut builder = FunctionBuilder::<TestDialect>::new("clobber".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    builder
        .append(
            body,
            super::assign(0, &[0], constant(1, ScalarType::U32)),
            None,
        )
        .unwrap();
    builder
        .append(
            body,
            emit_writing(vec![word(0)], &[(2, &[0]), (3, &[0])]),
            None,
        )
        .unwrap();
    builder
        .append(body, emit_writing(vec![word(2), word(3)], &[]), None)
        .unwrap();
    builder
        .append(body, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    builder.finish().unwrap()
}

/// An effect whose write lands in the very register its operand reads.
fn in_place_function() -> Function<TestDialect> {
    let mut builder = FunctionBuilder::<TestDialect>::new("in-place".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    builder
        .append(
            body,
            super::assign(0, &[0], constant(1, ScalarType::U32)),
            None,
        )
        .unwrap();
    builder
        .append(body, emit_writing(vec![word(0)], &[(0, &[0])]), None)
        .unwrap();
    builder
        .append(body, emit_writing(vec![word(0)], &[]), None)
        .unwrap();
    builder
        .append(body, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    builder.finish().unwrap()
}

fn lifted(function: &Function<TestDialect>) -> mlil::Function<SemanticDialect> {
    lift(function, &()).unwrap().builder.finish().unwrap()
}

#[test]
fn an_effect_with_two_writes_defines_two_variables() {
    let mlil = lifted(&clobbering_function());
    let list = instructions(&mlil);
    assert_eq!(list.len(), 4, "one instruction per statement");

    let produced = list[0].defs();
    assert_eq!(produced.len(), 1);
    let clobber = list[1];
    assert_eq!(
        clobber.defs().len(),
        2,
        "both writes are definitions of one instruction"
    );
    assert_ne!(clobber.defs()[0], clobber.defs()[1]);
    assert_eq!(
        clobber.uses(),
        produced,
        "the operand still reads the pre-write web"
    );
    assert_eq!(
        list[2].uses(),
        clobber.defs(),
        "each later read resolves to the write that produced it"
    );
    assert!(mlil.verify().is_ok());
}

#[test]
fn a_read_after_an_effect_write_resolves_to_the_written_web() {
    let mlil = lifted(&in_place_function());
    let list = instructions(&mlil);
    let written = list[1].defs();
    assert_eq!(written.len(), 1);
    assert_eq!(
        list[2].uses(),
        written,
        "the later read observes the version the effect defined"
    );
}

#[test]
fn an_operand_reading_a_written_place_observes_the_pre_write_web() {
    let mlil = lifted(&in_place_function());
    let list = instructions(&mlil);
    let effect = list[1];
    assert_eq!(effect.uses().len(), 1, "one read of the old version");
    assert_eq!(
        effect.uses(),
        list[0].defs(),
        "the operand reads what the transfer defined, not the write"
    );
    assert_ne!(
        effect.uses()[0],
        effect.defs()[0],
        "the write starts a web of its own"
    );
}

#[test]
fn an_effect_names_the_places_it_writes_in_pseudocode() {
    assert_golden(
        "rtl/effect-writes.pseudo",
        &clobbering_function().to_pseudocode(),
    );
}

/// HLIL has no form for an operation with several definitions, so the
/// lift says so instead of dropping the extra ones.
#[test]
fn the_hlil_lift_rejects_an_effect_that_writes_two_places() {
    let mlil = lifted(&clobbering_function());
    let error = lift_hlil(&mlil).unwrap_err();
    let message = alloc::string::ToString::to_string(&error);
    assert!(
        message.contains("defines more than one result"),
        "unexpected failure: {message}"
    );
}
