//! Local dead-value pruning in the MLIL-to-HLIL presentation lift.

extern crate alloc;

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use super::{Edge, MediumOperation, Toy, Type};
use crate::ir::hlil::lift_function;
use crate::ir::mlil;
use crate::test_util::golden::assert_golden;

#[test]
fn transitively_dead_pure_definitions_are_omitted() {
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::dead".into());
    let block = builder.new_block("body");
    let first = builder.declare_variable(1, None).unwrap();
    let second = builder.declare_variable(1, None).unwrap();
    let typed = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Integer);
    let literal = builder
        .append_instruction(
            block,
            MediumOperation::Constant(7),
            Vec::new(),
            vec![typed(first)],
            false,
            None,
        )
        .unwrap();
    let copy = builder
        .append_instruction(
            block,
            MediumOperation::Copy,
            vec![typed(first)],
            vec![typed(second)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Return,
            Vec::new(),
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), block, Edge::Entry, None)
        .unwrap();
    let source = builder.finish().unwrap();

    let lifted = lift_function(&source).unwrap();
    let pseudo = lifted.function.to_pseudocode();

    assert_eq!(pseudo, "return;\n");
    assert!(!lifted.instructions.contains_key(&literal));
    assert!(!lifted.instructions.contains_key(&copy));
}

#[test]
fn effectful_dead_result_is_retained() {
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::effect".into());
    let block = builder.new_block("body");
    let result = builder.declare_variable(1, None).unwrap();
    let typed = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Integer);
    builder
        .append_instruction(
            block,
            MediumOperation::Call,
            Vec::new(),
            vec![typed(result)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Return,
            Vec::new(),
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), block, Edge::Entry, None)
        .unwrap();
    let source = builder.finish().unwrap();

    // The call survives its dead result because it is observable.
    assert_golden(
        "hlil/effectful-dead-result.pseudo",
        &lift_function(&source).unwrap().function.to_pseudocode(),
    );
}

#[test]
fn dead_definition_after_a_terminator_is_still_rejected() {
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::after-return".into());
    let block = builder.new_block("body");
    builder
        .append_instruction(
            block,
            MediumOperation::Return,
            Vec::new(),
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    let result = builder.declare_variable(1, None).unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Constant(7),
            Vec::new(),
            vec![mlil::TypedVariable::new(result, Type::Integer)],
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), block, Edge::Entry, None)
        .unwrap();
    let source = builder.finish().unwrap();

    let error = lift_function(&source).unwrap_err().to_string();

    assert!(error.contains("follows its block's terminator"), "{error}");
}
