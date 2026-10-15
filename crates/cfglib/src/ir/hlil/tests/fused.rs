//! Parallel-copy and fused-branch translation in the lift, over the toy
//! dialects.

use crate::ir::hlil::{FlatBuilder, FunctionView, pseudocode};
use crate::test_util::golden::assert_golden;

use super::Prim;
use super::flat::{FlatKit, ToyEdge, ToyFlatBuilder, ToyForm};
use super::flat_lift::{int, lifted, lifted_text, one_block};

over_toys!(
    overlapping_parallel_copy_stages_through_temporaries,
    disjoint_parallel_copy_moves_directly,
    a_single_pair_parallel_copy_inlines_as_a_copy,
    a_fused_compare_branch_becomes_the_loop_condition,
    a_zero_use_fused_branch_embeds_its_whole_condition,
);

fn overlapping_parallel_copy_stages_through_temporaries<D: FlatKit>() {
    // The parameters swap: exchange [x, y] -> [y, x]; return y.
    let (mut flat, block) = one_block::<D>();
    let x = flat.int_variable(0);
    let y = flat.int_variable(0);
    flat.takes(&[x, y]);
    flat.emit(
        block,
        ToyForm::Exchange,
        &[int::<D>(x), int::<D>(y)],
        &[int::<D>(y), int::<D>(x)],
    );
    flat.give_back(block, &[y]);
    let source = flat.finish().expect("the swap is well formed");

    // Every source is read into a temporary before any destination is
    // written, so the swap keeps both values.
    assert_golden("hlil/parallel-copy-staged.pseudo", &lifted_text(&source));
}

fn disjoint_parallel_copy_moves_directly<D: FlatKit>() {
    // exchange [a, b] -> [c, d] has no overlap and needs no temporary.
    let (mut flat, block) = one_block::<D>();
    let [a, b, c, d] = [0, 0, 0, 0].map(|number| flat.int_variable(number));
    flat.takes(&[a, b]);
    flat.emit(
        block,
        ToyForm::Exchange,
        &[int::<D>(a), int::<D>(b)],
        &[int::<D>(c), int::<D>(d)],
    );
    flat.give_back(block, &[d]);
    let source = flat.finish().expect("the spread is well formed");

    let lifted = lifted(&source);
    assert_golden(
        "hlil/parallel-copy-direct.pseudo",
        &pseudocode(&lifted.function),
    );
    assert_eq!(
        lifted.function.variable_count(),
        4,
        "no temporaries were staged"
    );
}

/// A parallel move of one pair is a plain copy, so it takes part in
/// expression inlining instead of always materializing an assignment.
fn a_single_pair_parallel_copy_inlines_as_a_copy<D: FlatKit>() {
    let (mut flat, block) = one_block::<D>();
    let source = flat.int_variable(0);
    let refined = flat.int_variable(1);
    flat.takes(&[source]);
    flat.emit(
        block,
        ToyForm::Exchange,
        &[int::<D>(source)],
        &[int::<D>(refined)],
    );
    flat.give_back(block, &[refined]);
    let source = flat.finish().expect("the refinement is well formed");

    // The single pair inlines as a copy: no temporary is left behind.
    assert_golden("hlil/parallel-copy-single.pseudo", &lifted_text(&source));
}

/// `header: below(i, n)` with a true edge into the body and a false edge to
/// the exit, and a back edge that is a synthesized jump.
fn a_fused_compare_branch_becomes_the_loop_condition<D: FlatKit>() {
    let mut flat = ToyFlatBuilder::<D>::new();
    let header = flat.new_block(Some("header"));
    let body = flat.new_block(Some("body"));
    let exit = flat.new_block(Some("exit"));
    let i = flat.int_variable(0);
    let n = flat.int_variable(0);
    let one = flat.int_variable(1);
    flat.emit(
        header,
        ToyForm::BranchOperation(D::op(Prim::Below)),
        &[int::<D>(i), int::<D>(n)],
        &[],
    );
    flat.literal(body, 1, one);
    flat.operate(
        body,
        Prim::Add,
        &[int::<D>(i), int::<D>(one)],
        &[int::<D>(i)],
    );
    flat.give_back(exit, &[i]);
    flat.start(header);
    flat.connect(header, body, ToyEdge::True);
    flat.connect(header, exit, ToyEdge::False);
    flat.connect(body, header, ToyEdge::Jump);
    let source = flat.finish().expect("the fused loop is well formed");

    assert!(lifted(&source).report.is_fully_structured());
    assert_golden("hlil/fused-loop-condition.pseudo", &lifted_text(&source));
}

/// A fused branch with no operands is legal when the operation embeds its
/// whole condition, such as a constant or a memory read.
fn a_zero_use_fused_branch_embeds_its_whole_condition<D: FlatKit>() {
    let mut flat = ToyFlatBuilder::<D>::new();
    let head = flat.new_block(Some("head"));
    let then = flat.new_block(Some("then"));
    let exit = flat.new_block(Some("exit"));
    let value = flat.int_variable(0);
    flat.emit(head, ToyForm::BranchOperation(D::op(Prim::Below)), &[], &[]);
    flat.literal(then, 1, value);
    flat.operate(then, Prim::Call, &[int::<D>(value)], &[]);
    flat.give_back(exit, &[]);
    flat.start(head);
    flat.connect(head, then, ToyEdge::True);
    flat.connect(head, exit, ToyEdge::False);
    flat.connect(then, exit, ToyEdge::Fall);
    let source = flat
        .finish()
        .expect("the embedded condition is well formed");

    assert!(lifted(&source).report.is_fully_structured());
    assert_golden(
        "hlil/fused-embedded-condition.pseudo",
        &lifted_text(&source),
    );
}
