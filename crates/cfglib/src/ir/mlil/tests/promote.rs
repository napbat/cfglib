//! Memory-to-variable promotion over the toy dialect.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::ir::mlil::PromotionAccess;

use super::{
    Edge, FunctionBuilder, InstructionId, Operation, ToyDialect, Type, TypedVariable, VariableId,
};

/// Two slots: slot 0 is only loaded and stored, slot 1's address escapes.
/// Instructions, in order: `store 0`, `load 0`, `addressof 1`, `store 1`,
/// `load 1`, `return`.
fn slots_function() -> crate::ir::mlil::Function<ToyDialect> {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::slots".into());
    let body = builder.new_block("body");
    let input = builder.declare_variable(0, None).unwrap();
    let first = builder.declare_variable(0, None).unwrap();
    let pointer = builder.declare_variable(0, None).unwrap();
    let second = builder.declare_variable(0, None).unwrap();
    let typed = |variable| TypedVariable::new(variable, Type::Integer);
    // Slot 0 is only loaded and stored; slot 1's address escapes.
    builder
        .append_instruction(
            body,
            Operation::Store(0),
            vec![typed(input)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Load(0),
            Vec::new(),
            vec![typed(first)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::AddressOf(1),
            Vec::new(),
            vec![typed(pointer)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Store(1),
            vec![typed(first)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Load(1),
            Vec::new(),
            vec![typed(second)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Return,
            vec![typed(second)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    builder.finish().unwrap()
}

#[test]
fn promote_memory_rewrites_unaliased_slots() {
    let function = slots_function();
    let input = VariableId::from_raw(0);
    let first = VariableId::from_raw(1);

    let promotion = function.promote_memory().unwrap();
    assert!(promotion.function.verify().is_ok());
    assert_eq!(promotion.rewritten, 2);
    assert_eq!(promotion.promoted.len(), 1, "{:?}", promotion.promoted);
    let slot = promotion.promoted[&0];
    assert_eq!(promotion.function.variable(slot).unwrap().role, 9);
    assert_eq!(promotion.function.variable(slot).unwrap().native, Some(0));

    let instruction = |index| {
        promotion
            .function
            .instruction(InstructionId::from_raw(index))
            .unwrap()
    };
    // The unaliased slot's accesses became copies through its variable.
    assert_eq!(*instruction(0).operation(), Operation::Copy);
    assert_eq!(instruction(0).uses(), [input]);
    assert_eq!(instruction(0).defs(), [slot]);
    assert_eq!(*instruction(1).operation(), Operation::Copy);
    assert_eq!(instruction(1).uses(), [slot]);
    assert_eq!(instruction(1).defs(), [first]);
    // The escaped slot's accesses stayed memory operations.
    assert_eq!(*instruction(3).operation(), Operation::Store(1));
    assert_eq!(*instruction(4).operation(), Operation::Load(1));
    assert_eq!(*instruction(5).operation(), Operation::Return);
}

#[test]
fn promote_memory_with_takes_the_classifier_over_the_dialect() {
    let function = slots_function();
    let first = VariableId::from_raw(1);
    let second = VariableId::from_raw(3);

    // The opposite judgment to the dialect's, in both directions: slot 0
    // escapes although its accesses are plain, and slot 1 is unaliased
    // although its address is taken.
    let promotion = function
        .promote_memory_with(|instruction| match instruction.operation() {
            Operation::Load(0) | Operation::Store(0) => PromotionAccess::Escape(vec![0]),
            Operation::Load(slot) => PromotionAccess::Load(*slot),
            Operation::Store(slot) => PromotionAccess::Store(*slot),
            _ => PromotionAccess::Unrelated,
        })
        .unwrap();
    assert!(promotion.function.verify().is_ok());
    assert_eq!(promotion.rewritten, 2);
    assert_eq!(
        promotion.promoted.keys().copied().collect::<Vec<_>>(),
        [1],
        "the closure's answer wins, not the trait's"
    );
    let slot = promotion.promoted[&1];
    assert_eq!(promotion.function.variable(slot).unwrap().native, Some(1));

    let instruction = |index| {
        promotion
            .function
            .instruction(InstructionId::from_raw(index))
            .unwrap()
    };
    // Slot 0 the trait would have promoted stayed in memory.
    assert_eq!(*instruction(0).operation(), Operation::Store(0));
    assert_eq!(*instruction(1).operation(), Operation::Load(0));
    // Slot 1 the trait would have escaped became copies, address-taking
    // instruction and all.
    assert_eq!(*instruction(2).operation(), Operation::AddressOf(1));
    assert_eq!(*instruction(3).operation(), Operation::Copy);
    assert_eq!(instruction(3).uses(), [first]);
    assert_eq!(instruction(3).defs(), [slot]);
    assert_eq!(*instruction(4).operation(), Operation::Copy);
    assert_eq!(instruction(4).uses(), [slot]);
    assert_eq!(instruction(4).defs(), [second]);
}

#[test]
fn promote_memory_with_the_trait_classifier_matches_promote_memory() {
    let function = slots_function();

    let implicit = function.promote_memory().unwrap();
    let explicit = function
        .promote_memory_with(<ToyDialect as crate::ir::mlil::PromoteDialect>::promotion_access)
        .unwrap();

    assert_eq!(implicit.function, explicit.function);
    assert_eq!(implicit.promoted, explicit.promoted);
    assert_eq!(implicit.rewritten, explicit.rewritten);
}
