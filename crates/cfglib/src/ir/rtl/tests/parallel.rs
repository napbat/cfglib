//! Parallel-move serialization of multi-assignment transfers.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{
    Edge, Effect, Expr, Function, FunctionBuilder, Place, ScalarType, SemanticDialect, Statement,
    TestDialect, assign, constant, instructions, lift, read,
};
use crate::ir::mlil;

/// Builds `r0 = 1; r1 = 2; <transfer>; return` with `Emit` effects on the
/// transfer.
fn function(assignments: Vec<(Place<TestDialect>, Expr<TestDialect>)>) -> Function<TestDialect> {
    let mut builder = FunctionBuilder::<TestDialect>::new("parallel".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    builder
        .append(body, assign(0, &[0], constant(1, ScalarType::U32)), None)
        .unwrap();
    builder
        .append(body, assign(1, &[0], constant(2, ScalarType::U32)), None)
        .unwrap();
    builder
        .append(
            body,
            Statement::Transfer {
                assignments,
                effects: vec![Effect::Emit],
                may_throw: false,
            },
            None,
        )
        .unwrap();
    builder
        .append(body, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    builder.finish().unwrap()
}

fn place(storage: u8) -> Place<TestDialect> {
    Place {
        storage,
        lanes: vec![0],
    }
}

fn word(storage: u8) -> Expr<TestDialect> {
    read(storage, &[0], ScalarType::U32)
}

/// Lifts `function` and returns the synthetic web count and MLIL.
fn lifted(function: &Function<TestDialect>) -> (usize, mlil::Function<SemanticDialect>) {
    let lifting = lift(function, &()).unwrap();
    let synthetic = lifting
        .webs
        .iter()
        .filter(|web| web.storage.is_none())
        .count();
    (synthetic, lifting.builder.finish().unwrap())
}

/// `r0 = r1, r2 = r0`: the reader of r0 runs before the writer of r0, so
/// no copy is necessary.
#[test]
fn acyclic_overlap_emits_writer_last_without_copy() {
    let (synthetic, mlil) = lifted(&function(vec![(place(0), word(1)), (place(2), word(0))]));
    assert_eq!(synthetic, 0, "no pre-state copy");
    let list = instructions(&mlil);
    assert_eq!(list.len(), 5, "two inits, two writes, return");
    let (init0, init1) = (list[0].defs(), list[1].defs());
    assert_eq!(list[2].uses(), init0, "the reader of r0 runs first");
    assert_eq!(list[3].uses(), init1, "the writer of r0 runs last");
    assert!(!list[2].effects().is_empty(), "effects stay on the first");
    assert_eq!(list[3].effects().len(), 0, "later writes carry none");
}

/// `r0 = r1, r1 = r0`: a swap copies exactly one pre-state value.
#[test]
fn swap_emits_one_copy_and_keeps_both_values() {
    let (synthetic, mlil) = lifted(&function(vec![(place(0), word(1)), (place(1), word(0))]));
    assert_eq!(synthetic, 1, "one pre-state copy");
    let list = instructions(&mlil);
    assert_eq!(list.len(), 6, "two inits, copy, two writes, return");
    let (init0, init1) = (list[0].defs(), list[1].defs());
    let copy = list[2];
    assert_eq!(copy.uses(), init0, "the copy saves the old r0");
    assert_eq!(list[3].uses(), init1, "r0 receives the old r1");
    assert_eq!(list[4].uses(), copy.defs(), "r1 receives the saved r0");
    assert!(!copy.effects().is_empty(), "effects stay on the first");
    assert_eq!(list[3].effects().len(), 0, "later writes carry none");
    assert_eq!(list[4].effects().len(), 0, "later writes carry none");
}
