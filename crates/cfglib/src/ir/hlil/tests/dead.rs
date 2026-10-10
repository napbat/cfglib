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

/// `protected: x = 1; call (throws); x = 2`, unwinding to a handler that
/// returns `x` and falling through to a block that returns something else.
/// The handler reads the first write, so it stays; no path reads the second.
#[test]
fn a_write_a_handler_reads_before_a_throw_is_retained() {
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::unwind".into());
    let protected = builder.new_block("protected");
    let after = builder.new_block("after");
    let pad = builder.new_block("pad");
    let x = builder.declare_variable(1, None).unwrap();
    let result = builder.declare_variable(1, None).unwrap();
    let typed = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Integer);
    let mut append = |block, operation, uses, defs, may_throw| {
        builder
            .append_instruction(block, operation, uses, defs, may_throw, None)
            .unwrap()
    };
    let first = append(
        protected,
        MediumOperation::Constant(1),
        Vec::new(),
        vec![typed(x)],
        false,
    );
    append(
        protected,
        MediumOperation::Call,
        Vec::new(),
        Vec::new(),
        true,
    );
    let second = append(
        protected,
        MediumOperation::Constant(2),
        Vec::new(),
        vec![typed(x)],
        false,
    );
    append(
        after,
        MediumOperation::Constant(0),
        Vec::new(),
        vec![typed(result)],
        false,
    );
    append(
        after,
        MediumOperation::Return,
        vec![typed(result)],
        Vec::new(),
        false,
    );
    append(
        pad,
        MediumOperation::Return,
        vec![typed(x)],
        Vec::new(),
        false,
    );
    builder
        .add_edge(builder.entry(), protected, Edge::Entry, None)
        .unwrap();
    builder
        .add_edge(protected, after, Edge::Fall, None)
        .unwrap();
    builder
        .add_edge(protected, pad, Edge::Unwind, None)
        .unwrap();
    builder
        .add_region(crate::Region {
            id: crate::RegionId::from_raw(0),
            protected_blocks: [protected].into_iter().collect(),
            handlers: vec![crate::Handler {
                entry: pad,
                body: crate::HandlerBody::known([pad]),
                kind: crate::HandlerKind::CatchAll,
            }],
            parent: None,
        })
        .unwrap();

    let lifted = lift_function(&builder.finish().unwrap()).unwrap();

    assert!(
        lifted.instructions.contains_key(&first),
        "the handler reads the write before the call"
    );
    assert!(
        !lifted.instructions.contains_key(&second),
        "no path reads the write after the call"
    );
}
