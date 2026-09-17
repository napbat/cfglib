//! Effect-ordered inlining tests, split out to respect the source-size
//! policy.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{Edge, MediumOperation, Toy, Type};
use crate::ir::hlil::lift_function;
use crate::ir::mlil;
use crate::test_util::golden::assert_golden;

#[test]
fn commuting_reads_fold_into_one_expression() {
    // first_load = load(a); second_load = load(b); sum = add(t, u); return x
    // Read-read commutes in this dialect, so both loads inline.
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::reads".into());
    let block = builder.new_block("body");
    let first_address = builder.declare_variable(0, None).unwrap();
    let second_address = builder.declare_variable(0, None).unwrap();
    let first_load = builder.declare_variable(1, None).unwrap();
    let second_load = builder.declare_variable(1, None).unwrap();
    let sum = builder.declare_variable(1, None).unwrap();
    let typed = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Integer);
    builder
        .append_instruction(
            block,
            MediumOperation::Load,
            vec![typed(first_address)],
            vec![typed(first_load)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Load,
            vec![typed(second_address)],
            vec![typed(second_load)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Add,
            vec![typed(first_load), typed(second_load)],
            vec![typed(sum)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Return,
            vec![typed(sum)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), block, Edge::Entry, None)
        .unwrap();

    let lifted = lift_function(&builder.finish().unwrap()).unwrap();
    assert_golden(
        "hlil/commuting-reads.pseudo",
        &lifted.function.to_pseudocode(),
    );
}

#[test]
fn reads_do_not_cross_writes() {
    // loaded = load(a); store(b, c); result = t  — the load must stay before the store.
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::ordered".into());
    let block = builder.new_block("body");
    let address = builder.declare_variable(0, None).unwrap();
    let target = builder.declare_variable(0, None).unwrap();
    let stored = builder.declare_variable(0, None).unwrap();
    let loaded = builder.declare_variable(1, None).unwrap();
    let result = builder.declare_variable(1, None).unwrap();
    let typed = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Integer);
    builder
        .append_instruction(
            block,
            MediumOperation::Load,
            vec![typed(address)],
            vec![typed(loaded)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Store,
            vec![typed(target), typed(stored)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Copy,
            vec![typed(loaded)],
            vec![typed(result)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Return,
            vec![typed(result)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), block, Edge::Entry, None)
        .unwrap();

    let lifted = lift_function(&builder.finish().unwrap()).unwrap();
    // The load stays where it was: ahead of the store it must not cross.
    assert_golden(
        "hlil/reads-before-writes.pseudo",
        &lifted.function.to_pseudocode(),
    );
}

#[test]
fn effectful_definitions_inline_only_when_order_is_preserved() {
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::calls".into());
    let block = builder.new_block("body");
    let t_first = builder.declare_variable(1, None).unwrap();
    let x = builder.declare_variable(0, None).unwrap();
    let t_second = builder.declare_variable(1, None).unwrap();
    let y = builder.declare_variable(0, None).unwrap();
    let z = builder.declare_variable(0, None).unwrap();
    let typed = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Integer);

    // t_first = call(); x = t_first        → inlines (immediately consumed)
    builder
        .append_instruction(
            block,
            MediumOperation::Call,
            Vec::new(),
            vec![typed(t_first)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Copy,
            vec![typed(t_first)],
            vec![typed(x)],
            false,
            None,
        )
        .unwrap();
    // t_second = call(x); y = call(x); z = t_second → t_second materializes
    builder
        .append_instruction(
            block,
            MediumOperation::Call,
            vec![typed(x)],
            vec![typed(t_second)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Call,
            vec![typed(x)],
            vec![typed(y)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Copy,
            vec![typed(t_second)],
            vec![typed(z)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Return,
            vec![typed(z)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), block, Edge::Entry, None)
        .unwrap();

    let lifted = lift_function(&builder.finish().unwrap()).unwrap();
    // The first call inlines into its one use. The second pair materializes
    // a temporary so the calls keep their order, and the pure copy chain
    // still folds into the return.
    assert_golden(
        "hlil/ordered-calls.pseudo",
        &lifted.function.to_pseudocode(),
    );
}

#[test]
fn inlining_crosses_a_straight_line_block_run() {
    // Frontends that emit one block per native instruction still inline:
    // the run coalesces into one list judged by the last block's live-out,
    // so a definition feeding the run's next block stays single-use.
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::chain".into());
    let first = builder.new_block("first");
    let second = builder.new_block("second");
    let third = builder.new_block("third");
    let argument = builder.declare_variable(0, None).unwrap();
    let staged = builder.declare_variable(1, None).unwrap();
    let result = builder.declare_variable(1, None).unwrap();
    let typed = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Integer);
    builder
        .append_instruction(
            first,
            MediumOperation::Constant(7),
            Vec::new(),
            vec![typed(staged)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            second,
            MediumOperation::Add,
            vec![typed(argument), typed(staged)],
            vec![typed(result)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            third,
            MediumOperation::Return,
            vec![typed(result)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), first, Edge::Entry, None)
        .unwrap();
    builder.add_edge(first, second, Edge::Fall, None).unwrap();
    builder.add_edge(second, third, Edge::Fall, None).unwrap();
    builder
        .set_signature(mlil::Signature::<Toy>::new(
            vec![argument],
            vec![Type::Integer],
        ))
        .unwrap();
    let lifted = lift_function(&builder.finish().unwrap()).unwrap();
    assert!(lifted.function.verify().is_ok());
    // Both temporaries inline across the straight-line block run.
    assert_golden(
        "hlil/inline-across-blocks.pseudo",
        &lifted.function.to_pseudocode(),
    );
}

#[test]
fn previous_value_operands_pin_both_sides() {
    // producer = add(a, a) feeds a merge's previous-value slot, and the
    // merge's own definition feeds the return: neither may inline — the
    // producer must stay a visible variable at the merge point, and the
    // merged value is not a pure expression.
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::merge".into());
    let block = builder.new_block("body");
    let input = builder.declare_variable(0, None).unwrap();
    let producer = builder.declare_variable(1, None).unwrap();
    let merged = builder.declare_variable(1, None).unwrap();
    let typed = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Integer);
    builder
        .append_instruction(
            block,
            MediumOperation::Add,
            vec![typed(input), typed(input)],
            vec![typed(producer)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Merge,
            vec![typed(input), typed(producer)],
            vec![typed(merged)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Return,
            vec![typed(merged)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), block, Edge::Entry, None)
        .unwrap();
    builder
        .set_signature(mlil::Signature::<Toy>::new(
            vec![input],
            vec![Type::Integer],
        ))
        .unwrap();

    let lifted = lift_function(&builder.finish().unwrap()).unwrap();
    // Neither side of the previous-value operand moves across the call.
    assert_golden(
        "hlil/previous-value-operand.pseudo",
        &lifted.function.to_pseudocode(),
    );
}
