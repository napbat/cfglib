//! Parallel-copy and fused-branch translation tests, split out to respect
//! the source-size policy.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{Edge, MediumOperation, Toy, Type};
use crate::ir::hlil::lift_function;
use crate::ir::mlil;
use crate::test_util::golden::assert_golden;

fn typed(variable: mlil::VariableId) -> mlil::TypedVariable<Toy> {
    mlil::TypedVariable::new(variable, Type::Integer)
}

#[test]
fn overlapping_parallel_copy_stages_through_temporaries() {
    // Parameters x and y swap: exchange [x, y] -> [y, x]; return y.
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::swap".into());
    let block = builder.new_block("body");
    let x = builder.declare_variable(0, None).unwrap();
    let y = builder.declare_variable(0, None).unwrap();
    builder
        .set_signature(mlil::Signature::<Toy>::new(vec![x, y], vec![Type::Integer]))
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Exchange,
            vec![typed(x), typed(y)],
            vec![typed(y), typed(x)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Return,
            vec![typed(y)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), block, Edge::Entry, None)
        .unwrap();

    let lifted = lift_function(&builder.finish().unwrap()).unwrap();
    assert!(lifted.function.verify().is_ok());
    // Every source is read into a temporary before any destination is
    // written, so the swap keeps both values.
    assert_golden(
        "hlil/parallel-copy-staged.pseudo",
        &lifted.function.to_pseudocode(),
    );
}

#[test]
fn disjoint_parallel_copy_moves_directly() {
    // exchange [a, b] -> [c, d] with no overlap needs no temporaries.
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::spread".into());
    let block = builder.new_block("body");
    let a = builder.declare_variable(0, None).unwrap();
    let b = builder.declare_variable(0, None).unwrap();
    let c = builder.declare_variable(0, None).unwrap();
    let d = builder.declare_variable(0, None).unwrap();
    builder
        .set_signature(mlil::Signature::<Toy>::new(vec![a, b], vec![Type::Integer]))
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Exchange,
            vec![typed(a), typed(b)],
            vec![typed(c), typed(d)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Return,
            vec![typed(d)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), block, Edge::Entry, None)
        .unwrap();

    let lifted = lift_function(&builder.finish().unwrap()).unwrap();
    let pseudo = lifted.function.to_pseudocode();
    assert_golden("hlil/parallel-copy-direct.pseudo", &pseudo);
    assert_eq!(
        lifted.function.variables().len(),
        4,
        "no temporaries were staged: {pseudo}"
    );
}

#[test]
fn fused_compare_branch_becomes_the_loop_condition() {
    // header: compare_branch(i, n); true -> body; false -> exit.
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::fused".into());
    let header = builder.new_block("header");
    let body = builder.new_block("body");
    let exit = builder.new_block("exit");
    let i = builder.declare_variable(0, None).unwrap();
    let n = builder.declare_variable(0, None).unwrap();
    let one = builder.declare_variable(1, None).unwrap();
    builder
        .append_instruction(
            header,
            MediumOperation::CompareBranch,
            vec![typed(i), typed(n)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            MediumOperation::Constant(1),
            Vec::new(),
            vec![typed(one)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            MediumOperation::Add,
            vec![typed(i), typed(one)],
            vec![typed(i)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            exit,
            MediumOperation::Return,
            vec![typed(i)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), header, Edge::Entry, None)
        .unwrap();
    builder.add_edge(header, body, Edge::True, None).unwrap();
    builder.add_edge(header, exit, Edge::False, None).unwrap();
    builder.add_edge(body, header, Edge::Jump, None).unwrap();

    let lifted = lift_function(&builder.finish().unwrap()).unwrap();
    assert!(lifted.report.is_fully_structured(), "{:?}", lifted.report);
    assert_golden(
        "hlil/fused-loop-condition.pseudo",
        &lifted.function.to_pseudocode(),
    );
}

#[test]
fn zero_use_fused_branch_embeds_its_whole_condition() {
    // head: compare_branch with no operands — legal when the operation
    // embeds its whole condition (a constant or memory read).
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::fused".into());
    let head = builder.new_block("head");
    let then = builder.new_block("then");
    let exit = builder.new_block("exit");
    let value = builder.declare_variable(0, None).unwrap();
    builder
        .append_instruction(
            head,
            MediumOperation::CompareBranch,
            Vec::new(),
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            then,
            MediumOperation::Constant(1),
            Vec::new(),
            vec![typed(value)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            then,
            MediumOperation::Call,
            vec![typed(value)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            exit,
            MediumOperation::Return,
            Vec::new(),
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), head, Edge::Entry, None)
        .unwrap();
    builder.add_edge(head, then, Edge::True, None).unwrap();
    builder.add_edge(head, exit, Edge::False, None).unwrap();
    builder.add_edge(then, exit, Edge::Fall, None).unwrap();

    let lifted = lift_function(&builder.finish().unwrap()).unwrap();
    assert!(lifted.report.is_fully_structured(), "{:?}", lifted.report);
    assert_golden(
        "hlil/fused-embedded-condition.pseudo",
        &lifted.function.to_pseudocode(),
    );
}

#[test]
fn single_pair_parallel_copy_inlines_as_a_copy() {
    // A parallel move of one pair is a plain copy (type-refinement pairs,
    // lone phi-copy commits), so it takes part in expression inlining
    // instead of always materializing an assignment.
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::refine".into());
    let block = builder.new_block("body");
    let source = builder.declare_variable(0, None).unwrap();
    let refined = builder.declare_variable(1, None).unwrap();
    let typed = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Integer);
    builder
        .append_instruction(
            block,
            MediumOperation::Exchange,
            vec![typed(source)],
            vec![typed(refined)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            block,
            MediumOperation::Return,
            vec![typed(refined)],
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
            vec![source],
            vec![Type::Integer],
        ))
        .unwrap();
    let lifted = lift_function(&builder.finish().unwrap()).unwrap();
    assert!(lifted.function.verify().is_ok());
    // The single pair inlines as a copy: no temporary is left behind.
    assert_golden(
        "hlil/parallel-copy-single.pseudo",
        &lifted.function.to_pseudocode(),
    );
}
