//! Lifting a flat function into structured HLIL over the toy dialects.
//!
//! Every body is written once over [`FlatKit`] and runs over both toys. The
//! toys disagree on the kind of a synthesized jump and of an exceptional
//! edge, so a pass over both covers both conventions.

extern crate alloc;
extern crate std;

use alloc::string::ToString;
use std::thread;

use crate::ProgramPoint;
use crate::graph::view::DenseId;
use crate::ir::hlil::{
    Entity, FlatBuilder, FunctionView, LiftError, LiftMetadata, StmtShape, lift, lift_trampolines,
    lift_with_tree, pseudocode, verify,
};
use crate::test_util::golden::assert_golden;
use crate::{BlockId, ControlTree, EdgeKind, LiftReport};

use super::flat::{FlatKit, FlatVar, ToyEdge, ToyFlat, ToyFlatBuilder, ToyForm};
use super::flat_fixtures::{
    claimed_trampoline_loop, counting_loop, dispatch_with_default, guarded_call, nested_ifs,
    nested_loops,
};
use super::flat_lift::{boolean, int, lifted, lifted_text};
use super::{Imp, Prim, ToyHlil, ToyHlilBuilder, on_small_stack};

over_toys!(
    lift_recovers_a_while_loop_with_inlined_expressions,
    lift_with_tree_reads_the_callers_tree,
    lift_can_omit_the_correspondence,
    lift_does_not_assume_that_the_builder_issues_flat_indices,
    the_correspondence_names_the_entity_that_carries_each_instruction,
    lift_recovers_a_switch_with_case_values_and_default,
    a_case_edge_can_carry_several_values,
    lift_structures_declared_exception_regions,
    exit_on_true_loop_negates_by_operation_inversion,
    a_region_claimed_jump_trampoline_resolves_as_a_break,
    trampolines_are_the_instruction_free_jump_blocks,
    a_plain_tree_degrades_a_trampoline_break_to_gotos,
    an_instruction_after_the_terminator_is_rejected,
    a_multi_result_operation_is_rejected,
    a_switch_case_without_values_is_rejected,
    a_branch_where_none_is_expected_is_rejected,
    a_copy_needs_exactly_one_use,
    a_store_needs_its_value_and_no_definition,
    deeply_nested_ifs_lift_without_recursion,
    deeply_nested_loops_lift_without_recursion,
);

fn lift_recovers_a_while_loop_with_inlined_expressions<D: FlatKit>() {
    let source = counting_loop::<D>();
    let lifted = lifted(&source);
    assert!(lifted.report.is_fully_structured(), "{:?}", lifted.report);
    assert!(verify(&lifted.function).is_ok());

    // The comparison and the constant inline, so the golden shows the loop
    // with no surviving temporaries.
    assert_golden("hlil/lift-while-loop.pseudo", &pseudocode(&lifted.function));

    // The signature and the variables carry over one to one.
    assert_eq!(lifted.function.parameters().len(), 2);
    assert_eq!(lifted.function.returns().len(), 1);
    assert_eq!(lifted.function.variable_count(), source.variables.len());
}

fn lift_with_tree_reads_the_callers_tree<D: FlatKit>() {
    let source = counting_loop::<D>();
    let trampolines = lift_trampolines::<D, _>(&source);
    assert!(trampolines.is_empty(), "{trampolines:?}");
    let (tree, report) = ControlTree::compute_with_report_treating_empty(&source.cfg, &trampolines);

    let from_tree = lift_with_tree::<D, _, _>(
        &source,
        &tree,
        &report,
        ToyHlilBuilder::<D>::new(),
        LiftMetadata::Preserve,
    )
    .expect("the function lifts");
    let direct = lifted(&source);

    assert_eq!(from_tree.report, direct.report);
    assert_eq!(from_tree.correspondence, direct.correspondence);
    assert_eq!(
        pseudocode(&from_tree.function),
        pseudocode(&direct.function)
    );
}

fn lift_can_omit_the_correspondence<D: FlatKit>() {
    let source = counting_loop::<D>();
    let omitted = lift::<D, _, _>(&source, ToyHlilBuilder::<D>::new(), LiftMetadata::Omit)
        .expect("the function lifts");

    assert_eq!(omitted.correspondence.instructions, []);
    // The variable table is not metadata: remapping needs it.
    assert_eq!(
        omitted.correspondence.variables.len(),
        source.variables.len()
    );
    assert!(verify(&omitted.function).is_ok());
    // Omitting the table changes nothing a reader sees: the same golden.
    assert_golden(
        "hlil/lift-while-loop.pseudo",
        &pseudocode(&omitted.function),
    );
}

fn lift_does_not_assume_that_the_builder_issues_flat_indices<D: FlatKit>() {
    // A builder that already holds a variable issues handles that are one
    // above the flat indices. The table states the map, and the function is
    // the same loop over the shifted handles.
    let source = counting_loop::<D>();
    let mut builder = ToyHlilBuilder::<D>::new();
    builder.int_variable(9);
    let lifted =
        lift::<D, _, _>(&source, builder, LiftMetadata::Preserve).expect("the function lifts");

    let shifted: alloc::vec::Vec<usize> = lifted
        .correspondence
        .variables
        .iter()
        .map(|variable| variable.index())
        .collect();
    assert_eq!(shifted, [1, 2, 3, 4]);
    assert_eq!(lifted.function.variable_count(), source.variables.len() + 1);
    assert_eq!(
        pseudocode(&lifted.function),
        "while (lt(v1, v2)) {\n    v1 = add(v1, 1);\n}\nreturn v1;\n"
    );
}

fn the_correspondence_names_the_entity_that_carries_each_instruction<D: FlatKit>() {
    let source = counting_loop::<D>();
    let lifted = lifted(&source);
    let block = |index: usize| BlockId::from_index(index);
    let (header, body, exit) = (block(1), block(2), block(3));
    let row = |block: BlockId, inst_idx: usize| {
        lifted
            .correspondence
            .instructions
            .iter()
            .find(|(point, _)| *point == ProgramPoint { block, inst_idx })
            .map(|(_, entity)| *entity)
    };

    // The comparison inlined into the loop condition: an expression.
    let Some(Entity::Expression(comparison)) = row(header, 0) else {
        panic!("{:?}", lifted.correspondence);
    };
    assert!(matches!(
        lifted.function.expr(comparison),
        crate::ir::hlil::ExprShape::Operation { operands, .. } if operands.len() == 2
    ));
    // The branch became the while statement that tests it.
    let Some(Entity::Statement(looped)) = row(header, 1) else {
        panic!("{:?}", lifted.correspondence);
    };
    let StmtShape::While {
        condition,
        body: loop_body,
    } = lifted.function.stmt(looped)
    else {
        panic!("the branch is the while statement");
    };
    assert_eq!(condition, comparison);
    // The constant inlined into the addition; the addition is a statement.
    assert!(matches!(row(body, 0), Some(Entity::Expression(_))));
    let Some(Entity::Statement(update)) = row(body, 1) else {
        panic!("{:?}", lifted.correspondence);
    };
    assert_eq!(loop_body, [update]);
    // A pure jump has no form in HLIL, and so has no row.
    assert_eq!(row(body, 2), None);
    // The return.
    assert!(matches!(row(exit, 0), Some(Entity::Statement(_))));

    // Rows are sorted by point with at most one per point.
    let points: alloc::vec::Vec<_> = lifted
        .correspondence
        .instructions
        .iter()
        .map(|(point, _)| *point)
        .collect();
    assert!(
        points.windows(2).all(|pair| pair[0] < pair[1]),
        "{points:?}"
    );
    // Variables map by flat index; a builder that issues handles in call
    // order gives the identity.
    for (index, variable) in lifted.correspondence.variables.iter().enumerate() {
        assert_eq!(variable.index(), index);
    }
}

fn dispatch_function<D: FlatKit>(case_edges: &[(&[i64], &str)]) -> ToyFlat<D> {
    let mut flat = ToyFlatBuilder::<D>::new();
    let dispatch = flat.new_block(Some("dispatch"));
    let merge = flat.new_block(Some("merge"));
    let selector = flat.int_variable(0);
    let result = flat.int_variable(0);

    flat.emit(dispatch, ToyForm::Switch, &[int::<D>(selector)], &[]);
    flat.give_back(merge, &[result]);
    flat.start(dispatch);
    for (index, &(values, label)) in case_edges.iter().enumerate() {
        let arm = flat.new_block(Some(label));
        flat.literal(arm, i64::try_from(10 * (index + 1)).unwrap(), result);
        flat.connect(arm, merge, ToyEdge::Fall);
        let edge = ToyEdge::Case(values.iter().map(|&value| D::constant(value)).collect());
        flat.connect(dispatch, arm, edge);
    }
    let fallback = flat.new_block(Some("fallback"));
    flat.literal(fallback, 99, result);
    flat.connect(fallback, merge, ToyEdge::Fall);
    flat.connect(dispatch, fallback, ToyEdge::Default);
    flat.finish_unchecked()
}

fn lift_recovers_a_switch_with_case_values_and_default<D: FlatKit>() {
    let source = dispatch_with_default::<D>();

    let lifted = lifted(&source);
    assert!(lifted.report.is_fully_structured());
    assert_golden("hlil/lift-switch.pseudo", &pseudocode(&lifted.function));

    // The dispatch instruction is the switch statement.
    let dispatch_row = lifted
        .correspondence
        .instructions
        .iter()
        .find(|(point, _)| point.block == BlockId::from_index(1) && point.inst_idx == 0)
        .map(|(_, entity)| *entity);
    let Some(Entity::Statement(switch)) = dispatch_row else {
        panic!("{:?}", lifted.correspondence);
    };
    assert!(matches!(
        lifted.function.stmt(switch),
        StmtShape::Switch { .. }
    ));
    assert_eq!(lifted.function.switch_arms(switch).len(), 2);
}

fn a_case_edge_can_carry_several_values<D: FlatKit>() {
    let source = dispatch_function::<D>(&[(&[1, 2, 3], "many"), (&[4], "four")]);
    assert_eq!(
        lifted_text(&source),
        "switch (v0) {\n    case 1, 2, 3: {\n        v1 = 10;\n    }\n    case 4: {\n        v1 = 20;\n    }\n    default: {\n        v1 = 99;\n    }\n}\nreturn v1;\n"
    );
}

fn lift_structures_declared_exception_regions<D: FlatKit>() {
    assert_golden(
        "hlil/lift-try-catch.pseudo",
        &lifted_text(&guarded_call::<D>()),
    );
}

fn exit_on_true_loop_negates_by_operation_inversion<D: FlatKit>() {
    let mut flat = ToyFlatBuilder::<D>::new();
    let header = flat.new_block(Some("header"));
    let body = flat.new_block(Some("body"));
    let exit = flat.new_block(Some("exit"));
    let index = flat.int_variable(0);
    let bound = flat.int_variable(0);

    flat.emit(
        header,
        ToyForm::BranchOperation(D::op(Prim::Below)),
        &[int::<D>(index), int::<D>(bound)],
        &[],
    );
    flat.literal(body, 1, index);
    flat.give_back(exit, &[index]);
    flat.start(header);
    flat.connect(header, exit, ToyEdge::True);
    flat.connect(header, body, ToyEdge::False);
    flat.connect(body, header, ToyEdge::Fall);
    flat.takes(&[index, bound]);
    let source = flat.finish().expect("the inverted loop is well formed");

    // The exit-on-true test inverts in place: `while (at-least(...))`, not a
    // wrapped negation or a degraded endless loop.
    assert_golden("hlil/recover-inverted-loop.pseudo", &lifted_text(&source));
}

/// The region claim refuses out-of-bound inlining and a non-empty block
/// refuses the trampoline hop, so structuring depends on the lift treating
/// the dialect-declared pure transfers as empty.
///
/// The structurer forwards a break only through a hop whose edge has the kind
/// `Jump`. `Imp` gives its synthesized jump that kind; `Sym` gives it
/// `Unconditional`, which the structurer reads as ordinary flow, so the same
/// graph keeps its gotos there. This is the cost of a toy convention that the
/// structurer does not resolve, and the lift reports it instead of guessing.
fn a_region_claimed_jump_trampoline_resolves_as_a_break<D: FlatKit>() {
    let source = claimed_trampoline_loop::<D>();
    let lifted = lifted(&source);
    assert!(verify(&lifted.function).is_ok());
    if D::jump_kind() == EdgeKind::Jump {
        assert!(lifted.report.gotos.is_empty(), "{:?}", lifted.report);
        // The trampoline resolved to a break; no goto survives.
        assert_golden(
            "hlil/trampoline-break.pseudo",
            &pseudocode(&lifted.function),
        );
    } else {
        assert!(!lifted.report.gotos.is_empty(), "{:?}", lifted.report);
        assert!(pseudocode(&lifted.function).contains("goto"));
    }
}

fn trampolines_are_the_instruction_free_jump_blocks<D: FlatKit>() {
    let source = claimed_trampoline_loop::<D>();
    // Only `tramp` holds nothing but a jump; `latch` defines values.
    assert_eq!(
        lift_trampolines::<D, _>(&source),
        [BlockId::from_index(4)],
        "{}",
        source.dump()
    );
}

fn a_plain_tree_degrades_a_trampoline_break_to_gotos<D: FlatKit>() {
    let source = claimed_trampoline_loop::<D>();
    let (tree, report) = ControlTree::compute_with_report(&source.cfg);
    assert!(
        !report.gotos.is_empty(),
        "the plain tree keeps the trampoline"
    );

    // The caller owns the trampoline policy: `lift_with_tree` does not clear
    // anything, and hands back the report it was given.
    let lifted = lift_with_tree::<D, _, _>(
        &source,
        &tree,
        &report,
        ToyHlilBuilder::<D>::new(),
        LiftMetadata::Preserve,
    )
    .expect("the function lifts");
    assert_eq!(lifted.report, report);
    assert!(
        pseudocode(&lifted.function).contains("goto"),
        "{}",
        pseudocode(&lifted.function)
    );
}

/// The message of the failure when lifting `source`.
fn refusal<D: FlatKit>(source: &ToyFlat<D>) -> alloc::string::String {
    match lift::<D, _, _>(source, ToyHlilBuilder::<D>::new(), LiftMetadata::Preserve) {
        Err(LiftError::Unsupported(message)) => message,
        Err(LiftError::Builder(error)) => panic!("the builder refused: {error}"),
        Ok(_) => panic!("the function lifted"),
    }
}

/// One block that runs `fill` and is entered from the root.
fn single_block<D: FlatKit>(
    fill: impl FnOnce(&mut ToyFlatBuilder<D>, BlockId, FlatVar),
) -> ToyFlat<D> {
    let mut flat = ToyFlatBuilder::<D>::new();
    let block = flat.new_block(Some("body"));
    let x = flat.int_variable(0);
    fill(&mut flat, block, x);
    flat.start(block);
    flat.finish_unchecked()
}

fn an_instruction_after_the_terminator_is_rejected<D: FlatKit>() {
    // A dead definition after the return is pruned from the output but is
    // still a malformed block.
    let source = single_block::<D>(|flat, block, x| {
        flat.give_back(block, &[]);
        flat.literal(block, 7, x);
    });
    assert!(
        refusal(&source).contains("follows its block's terminator"),
        "{}",
        refusal(&source)
    );
}

fn a_multi_result_operation_is_rejected<D: FlatKit>() {
    let source = single_block::<D>(|flat, block, x| {
        let y = flat.int_variable(0);
        flat.operate(block, Prim::Add, &[], &[int::<D>(x), int::<D>(y)]);
        flat.give_back(block, &[x, y]);
    });
    assert!(
        refusal(&source).contains("defines more than one result"),
        "{}",
        refusal(&source)
    );
}

fn a_switch_case_without_values_is_rejected<D: FlatKit>() {
    let source = dispatch_function::<D>(&[(&[1], "one"), (&[], "empty")]);
    assert!(
        refusal(&source).contains("has no case values"),
        "{}",
        refusal(&source)
    );
}

fn a_branch_where_none_is_expected_is_rejected<D: FlatKit>() {
    // The block returns, so the tree reads it as a straight list, and the
    // branch in the middle of it has nothing to decide.
    let source = single_block::<D>(|flat, block, x| {
        flat.literal(block, 1, x);
        flat.branch(block, x);
    });
    assert!(
        refusal(&source).contains("unexpected branch or dispatch"),
        "{}",
        refusal(&source)
    );
}

fn a_copy_needs_exactly_one_use<D: FlatKit>() {
    let source = single_block::<D>(|flat, block, x| {
        let y = flat.int_variable(0);
        flat.emit(
            block,
            ToyForm::Copy,
            &[int::<D>(y), int::<D>(y)],
            &[int::<D>(x)],
        );
        flat.give_back(block, &[x]);
    });
    assert!(
        refusal(&source).contains("does not have exactly one use"),
        "{}",
        refusal(&source)
    );
}

fn a_store_needs_its_value_and_no_definition<D: FlatKit>() {
    let valueless = single_block::<D>(|flat, block, _| {
        flat.emit(block, ToyForm::Store(D::op(Prim::Deref)), &[], &[]);
        flat.give_back(block, &[]);
    });
    assert!(
        refusal(&valueless).contains("has no stored value"),
        "{}",
        refusal(&valueless)
    );
    let defining = single_block::<D>(|flat, block, x| {
        flat.emit(
            block,
            ToyForm::Store(D::op(Prim::Deref)),
            &[int::<D>(x)],
            &[int::<D>(x)],
        );
        flat.give_back(block, &[]);
    });
    assert!(
        refusal(&defining).contains("also defines variables"),
        "{}",
        refusal(&defining)
    );
}

/// A builder that refuses the lifted function surfaces its own error, and no
/// function escapes. Only `Imp` states a typing rule.
#[test]
fn a_builder_that_refuses_the_function_reports_its_error() {
    let source = single_block::<Imp>(|flat, block, x| {
        let flag = flat.int_variable(0);
        // `lt` orders integers, and the toy builder verifies that.
        flat.operate(
            block,
            Prim::LessThan,
            &[boolean::<Imp>(flag), boolean::<Imp>(flag)],
            &[boolean::<Imp>(x)],
        );
        flat.give_back(block, &[x]);
    });
    let error = lift::<Imp, _, _>(
        &source,
        ToyHlilBuilder::<Imp>::new(),
        LiftMetadata::Preserve,
    )
    .expect_err("the verifying builder refuses the function");
    let LiftError::Builder(report) = error else {
        panic!("the lift itself succeeded");
    };
    assert!(report.to_string().contains("non-integer"), "{report}");
}

/// The stack that the structurer needs for a nest of 100,000 levels, which
/// recurses on each. The structurer is not the walk under test.
const STRUCTURER_STACK_BYTES: usize = 512 * 1024 * 1024;

/// The control tree of `source`, computed the way [`lift`] does, on a stack
/// that holds the structurer's own recursion.
fn structured_on_a_roomy_stack<D: FlatKit>(source: &ToyFlat<D>) -> (ControlTree, LiftReport)
where
    ToyFlat<D>: Sync,
{
    thread::scope(|scope| {
        thread::Builder::new()
            .stack_size(STRUCTURER_STACK_BYTES)
            .spawn_scoped(scope, || {
                let trampolines = lift_trampolines::<D, _>(source);
                ControlTree::compute_with_report_treating_empty(&source.cfg, &trampolines)
            })
            .expect("a thread spawns")
            .join()
            .expect("the structurer finishes")
    })
}

/// How many `if` or `while` statements wrap one another from the top of the
/// function down, each as the first statement of the one outside it.
fn nesting_depth<D: FlatKit>(function: &ToyHlil<D>) -> usize {
    let mut depth = 0;
    let mut body = function.body();
    while let Some(&first) = body.first() {
        match function.stmt(first) {
            StmtShape::If {
                then_body: inner, ..
            }
            | StmtShape::While { body: inner, .. } => {
                depth += 1;
                body = inner;
            }
            _ => break,
        }
    }
    depth
}

/// Lifts `source`, a nest of `depth` compound statements around one
/// assignment, on a thread whose stack could not hold a frame per level.
fn lifts_a_deep_nest_on_a_small_stack<D: FlatKit>(source: &ToyFlat<D>, depth: usize)
where
    ToyFlat<D>: Sync,
{
    let (tree, report) = structured_on_a_roomy_stack(source);

    let (nested, statements) = on_small_stack(|| {
        let lifted = lift_with_tree::<D, _, _>(
            source,
            &tree,
            &report,
            ToyHlilBuilder::<D>::new(),
            LiftMetadata::Preserve,
        )
        .expect("a deep nest lifts");
        (
            nesting_depth(&lifted.function),
            lifted.function.stmt_count(),
        )
    });

    assert_eq!(nested, depth);
    // The compound statements, the assignment inside them, and the return.
    assert_eq!(statements, depth + 2);
}

fn deeply_nested_ifs_lift_without_recursion<D: FlatKit>()
where
    ToyFlat<D>: Sync,
{
    const DEPTH: usize = 100_000;
    lifts_a_deep_nest_on_a_small_stack(&nested_ifs::<D>(DEPTH), DEPTH);
}

fn deeply_nested_loops_lift_without_recursion<D: FlatKit>()
where
    ToyFlat<D>: Sync,
{
    // The structurer's loop analysis grows faster than linearly with the
    // nest, which bounds the depth: this one still needs several times the
    // stack of the thread that lifts it.
    const DEPTH: usize = 1_000;
    lifts_a_deep_nest_on_a_small_stack(&nested_loops::<D>(DEPTH), DEPTH);
}
