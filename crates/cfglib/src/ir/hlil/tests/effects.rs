//! Effect-ordered inlining and dead-value pruning in the lift, over the toy
//! dialects.

extern crate alloc;

use alloc::vec;

use crate::dataflow::ProgramPoint;
use crate::ir::hlil::{Entity, FlatBuilder, FunctionView, StmtShape, verify};
use crate::region::{Handler as RegionHandler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::golden::assert_golden;

use super::flat::{FlatKit, ToyEdge, ToyFlatBuilder, ToyForm, ToyInst};
use super::flat_lift::{boolean, int, lifted, lifted_text, one_block};
use super::{Prim, Sym};

over_toys!(
    commuting_reads_fold_into_one_expression,
    reads_do_not_cross_writes,
    effectful_definitions_inline_only_when_order_is_preserved,
    inlining_crosses_a_straight_line_block_run,
    a_run_of_blocks_absorbs_the_branch_that_follows,
    a_long_inlined_chain_lifts_without_recursion,
    a_run_of_blocks_absorbs_the_dispatch_that_follows,
    the_dialect_can_keep_a_single_use_value_materialized,
    previous_value_operands_pin_both_sides,
    transitively_dead_pure_definitions_are_omitted,
    an_effectful_dead_result_is_retained,
);

/// `first = load(a); second = load(b); sum = add(first, second); return sum`.
/// A read passes a read in both toys, so both loads inline.
fn commuting_reads_fold_into_one_expression<D: FlatKit>() {
    let (mut flat, block) = one_block::<D>();
    let first_address = flat.int_variable(0);
    let second_address = flat.int_variable(0);
    let first_load = flat.int_variable(1);
    let second_load = flat.int_variable(1);
    let sum = flat.int_variable(1);
    flat.operate(
        block,
        Prim::Load,
        &[int::<D>(first_address)],
        &[int::<D>(first_load)],
    );
    flat.operate(
        block,
        Prim::Load,
        &[int::<D>(second_address)],
        &[int::<D>(second_load)],
    );
    flat.operate(
        block,
        Prim::Add,
        &[int::<D>(first_load), int::<D>(second_load)],
        &[int::<D>(sum)],
    );
    flat.give_back(block, &[sum]);
    let source = flat.finish().expect("the reads are well formed");

    assert_golden("hlil/commuting-reads.pseudo", &lifted_text(&source));
}

/// `loaded = load(a); store(b, c); result = loaded; return result`. The load
/// must stay ahead of the store that it must not cross.
fn reads_do_not_cross_writes<D: FlatKit>() {
    let (mut flat, block) = one_block::<D>();
    let address = flat.int_variable(0);
    let target = flat.int_variable(0);
    let stored = flat.int_variable(0);
    let loaded = flat.int_variable(1);
    let result = flat.int_variable(1);
    flat.operate(block, Prim::Load, &[int::<D>(address)], &[int::<D>(loaded)]);
    flat.emit(
        block,
        ToyForm::Store(D::op(Prim::Deref)),
        &[int::<D>(target), int::<D>(stored)],
        &[],
    );
    flat.copy(block, loaded, result);
    flat.give_back(block, &[result]);
    let source = flat
        .finish()
        .expect("the read and the write are well formed");

    assert_golden("hlil/reads-before-writes.pseudo", &lifted_text(&source));
}

fn effectful_definitions_inline_only_when_order_is_preserved<D: FlatKit>() {
    let (mut flat, block) = one_block::<D>();
    let first = flat.int_variable(1);
    let x = flat.int_variable(0);
    let second = flat.int_variable(1);
    let y = flat.int_variable(0);
    let z = flat.int_variable(0);

    // `first = call(); x = first` inlines: the call is consumed at once.
    flat.operate(block, Prim::Call, &[], &[int::<D>(first)]);
    flat.copy(block, first, x);
    // `second = call(x); y = call(x); z = second`: the first call
    // materializes, so the calls keep their order.
    flat.operate(block, Prim::Call, &[int::<D>(x)], &[int::<D>(second)]);
    flat.operate(block, Prim::Call, &[int::<D>(x)], &[int::<D>(y)]);
    flat.copy(block, second, z);
    flat.give_back(block, &[z]);
    let source = flat.finish().expect("the calls are well formed");

    // The pure copy chain still folds into the return.
    assert_golden("hlil/ordered-calls.pseudo", &lifted_text(&source));
}

/// A frontend that emits one block per native instruction still inlines: the
/// run coalesces into one list judged by the live-out of its last block, so a
/// definition that feeds a later block of the run stays single-use.
fn inlining_crosses_a_straight_line_block_run<D: FlatKit>() {
    let mut flat = ToyFlatBuilder::<D>::new();
    let first = flat.new_block(Some("first"));
    let second = flat.new_block(Some("second"));
    let third = flat.new_block(Some("third"));
    let argument = flat.int_variable(0);
    let staged = flat.int_variable(1);
    let result = flat.int_variable(1);
    flat.literal(first, 7, staged);
    flat.operate(
        second,
        Prim::Add,
        &[int::<D>(argument), int::<D>(staged)],
        &[int::<D>(result)],
    );
    flat.give_back(third, &[result]);
    flat.start(first);
    flat.connect(first, second, ToyEdge::Fall);
    flat.connect(second, third, ToyEdge::Fall);
    flat.takes(&[argument]);
    let source = flat.finish().expect("the run is well formed");

    // Both temporaries inline across the run.
    assert_golden("hlil/inline-across-blocks.pseudo", &lifted_text(&source));
}

/// `n` additions, each of which consumes the one before it, inline into one
/// expression tree that is as deep as the chain is long.
fn a_long_inlined_chain_lifts_without_recursion<D: FlatKit>() {
    const LENGTH: usize = 20_000;
    let (mut flat, block) = one_block::<D>();
    let seed = flat.int_variable(0);
    let mut previous = seed;
    for _ in 0..LENGTH {
        let next = flat.int_variable(1);
        flat.operate(
            block,
            Prim::Add,
            &[int::<D>(previous), int::<D>(seed)],
            &[int::<D>(next)],
        );
        previous = next;
    }
    flat.give_back(block, &[previous]);
    flat.takes(&[seed]);
    let source = flat.finish().expect("the chain is well formed");

    let lifted = lifted(&source);

    assert_eq!(
        lifted.function.body().len(),
        1,
        "everything inlines into the return"
    );
    assert_eq!(lifted.function.expr_count(), 2 * LENGTH + 1);
    assert!(verify(&lifted.function).is_ok());
}

/// `first: staged = 7`, then `second: if (lt(staged, n)) { call() }`, entered
/// by a fallthrough. The run of `first` absorbs the decision of `second`.
fn a_run_of_blocks_absorbs_the_branch_that_follows<D: FlatKit>() {
    let mut flat = ToyFlatBuilder::<D>::new();
    let first = flat.new_block(Some("first"));
    let second = flat.new_block(Some("second"));
    let then = flat.new_block(Some("then"));
    let exit = flat.new_block(Some("exit"));
    let n = flat.int_variable(0);
    let staged = flat.int_variable(1);
    let condition = flat.int_variable(1);
    flat.literal(first, 7, staged);
    flat.operate(
        second,
        Prim::LessThan,
        &[int::<D>(staged), int::<D>(n)],
        &[boolean::<D>(condition)],
    );
    flat.branch(second, condition);
    flat.operate(then, Prim::Call, &[], &[]);
    flat.give_back(exit, &[]);
    flat.start(first);
    flat.connect(first, second, ToyEdge::Fall);
    flat.connect(second, then, ToyEdge::True);
    flat.connect(second, exit, ToyEdge::False);
    flat.connect(then, exit, ToyEdge::Fall);
    flat.takes(&[n]);
    let source = flat.finish().expect("the branch is well formed");

    assert_eq!(
        lifted_text(&source),
        "if (lt(7, v0)) {\n    call();\n}\nreturn;\n"
    );
}

/// The same run, absorbing a dispatch: the staged value is the scrutinee.
fn a_run_of_blocks_absorbs_the_dispatch_that_follows<D: FlatKit>() {
    let mut flat = ToyFlatBuilder::<D>::new();
    let first = flat.new_block(Some("first"));
    let second = flat.new_block(Some("second"));
    let arm = flat.new_block(Some("arm"));
    let rest = flat.new_block(Some("rest"));
    let exit = flat.new_block(Some("exit"));
    let staged = flat.int_variable(1);
    flat.literal(first, 3, staged);
    flat.emit(second, ToyForm::Switch, &[int::<D>(staged)], &[]);
    flat.operate(arm, Prim::Call, &[], &[]);
    flat.operate(rest, Prim::Call, &[], &[]);
    flat.give_back(exit, &[]);
    flat.start(first);
    flat.connect(first, second, ToyEdge::Fall);
    flat.connect(second, arm, ToyEdge::Case(vec![D::constant(3)]));
    flat.connect(second, rest, ToyEdge::Default);
    flat.connect(arm, exit, ToyEdge::Fall);
    flat.connect(rest, exit, ToyEdge::Fall);
    let source = flat.finish().expect("the dispatch is well formed");

    assert_eq!(
        lifted_text(&source),
        "switch (3) {\n    case 3: {\n        call();\n    }\n    default: {\n        call();\n    }\n}\nreturn;\n"
    );
}

fn the_dialect_can_keep_a_single_use_value_materialized<D: FlatKit>() {
    let (mut flat, block) = one_block::<D>();
    let input = flat.int_variable(0);
    let staged = flat.int_variable(1);
    let result = flat.int_variable(1);
    flat.push(
        block,
        ToyInst::new(
            ToyForm::Operation(D::op(Prim::Call)),
            &[int::<D>(input)],
            &[int::<D>(staged)],
        )
        .with_effects(vec![])
        .with_may_throw(false)
        .materialized(),
    );
    flat.operate(
        block,
        Prim::Add,
        &[int::<D>(staged), int::<D>(input)],
        &[int::<D>(result)],
    );
    flat.give_back(block, &[result]);
    let source = flat.finish().expect("the value is well formed");

    let lifted = lifted(&source);
    let statements = lifted.function.body();
    assert_eq!(statements.len(), 2);
    assert!(matches!(
        lifted.function.stmt(statements[0]),
        StmtShape::Assign { .. }
    ));
}

/// `producer = add(input, input)` feeds the previous-value slot of a merge,
/// and the merge's own definition feeds the return. Neither may inline: the
/// producer must stay a visible variable at the merge point, and the merged
/// value is not a pure expression.
fn previous_value_operands_pin_both_sides<D: FlatKit>() {
    let (mut flat, block) = one_block::<D>();
    let input = flat.int_variable(0);
    let producer = flat.int_variable(1);
    let merged = flat.int_variable(1);
    flat.operate(
        block,
        Prim::Add,
        &[int::<D>(input), int::<D>(input)],
        &[int::<D>(producer)],
    );
    flat.push(
        block,
        ToyInst::new(
            ToyForm::Operation(D::op(Prim::Call)),
            &[int::<D>(input), int::<D>(producer)],
            &[int::<D>(merged)],
        )
        .with_effects(vec![])
        .with_may_throw(false)
        .reading_previous_value(1),
    );
    flat.give_back(block, &[merged]);
    flat.takes(&[input]);
    let source = flat.finish().expect("the merge is well formed");

    assert_golden("hlil/previous-value-operand.pseudo", &lifted_text(&source));
}

fn transitively_dead_pure_definitions_are_omitted<D: FlatKit>() {
    let (mut flat, block) = one_block::<D>();
    let first = flat.int_variable(1);
    let second = flat.int_variable(1);
    flat.literal(block, 7, first);
    flat.copy(block, first, second);
    flat.give_back(block, &[]);
    let source = flat.finish().expect("the dead values are well formed");

    let lifted = lifted(&source);

    assert_eq!(crate::ir::hlil::pseudocode(&lifted.function), "return;\n");
    let row = |inst_idx| {
        lifted
            .correspondence
            .instructions
            .iter()
            .any(|(point, _)| *point == ProgramPoint { block, inst_idx })
    };
    assert!(!row(0), "the literal has no row");
    assert!(!row(1), "the copy has no row");
    assert!(row(2), "the return has one");
}

/// The call survives its dead result because it is observable.
fn an_effectful_dead_result_is_retained<D: FlatKit>() {
    let (mut flat, block) = one_block::<D>();
    let result = flat.int_variable(1);
    flat.operate(block, Prim::Call, &[], &[int::<D>(result)]);
    flat.give_back(block, &[]);
    let source = flat.finish().expect("the call is well formed");

    assert_golden("hlil/effectful-dead-result.pseudo", &lifted_text(&source));
}

/// `protected: x = 1; call() [throws]; x = 2`, unwinding to a handler that
/// returns `x`, and falling through to a block that returns something else.
/// The handler reads the first write, so it stays; no path reads the second.
///
/// Only `Sym` runs this: its exceptional edge has the kind
/// `ExceptionUnwind`, which departs before the throwing instruction. The
/// `Imp` edge is an `ExceptionHandler` edge, an ordinary successor that
/// observes the completed block, so the same graph reads differently there.
#[test]
fn a_write_a_handler_reads_before_a_throw_is_retained() {
    let mut flat = ToyFlatBuilder::<Sym>::new();
    let protected = flat.new_block(Some("protected"));
    let after = flat.new_block(Some("after"));
    let pad = flat.new_block(Some("pad"));
    let x = flat.int_variable(1);
    let result = flat.int_variable(1);
    flat.literal(protected, 1, x);
    flat.operate(protected, Prim::Call, &[], &[]);
    flat.literal(protected, 2, x);
    flat.literal(after, 0, result);
    flat.give_back(after, &[result]);
    flat.give_back(pad, &[x]);
    flat.start(protected);
    flat.connect(protected, after, ToyEdge::Fall);
    flat.connect(protected, pad, ToyEdge::Except);
    flat.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [protected].into_iter().collect(),
        handlers: vec![RegionHandler {
            entry: pad,
            body: HandlerBody::known([pad]),
            kind: HandlerKind::CatchAll,
        }],
        parent: None,
    });
    let source = flat.finish().expect("the unwinding block is well formed");

    let lifted = lifted(&source);
    let row = |inst_idx| {
        lifted
            .correspondence
            .instructions
            .iter()
            .find(|(point, _)| {
                *point
                    == ProgramPoint {
                        block: protected,
                        inst_idx,
                    }
            })
            .map(|(_, entity)| *entity)
    };

    assert!(
        matches!(row(0), Some(Entity::Statement(_))),
        "the handler reads the write before the call: {:?}",
        lifted.correspondence
    );
    assert_eq!(row(2), None, "no path reads the write after the call");
}
