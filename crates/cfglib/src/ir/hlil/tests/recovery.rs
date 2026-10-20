//! Structure recovery over the toy dialects.

extern crate alloc;
extern crate std;

use crate::BlockId;
use crate::ProgramPoint;
use crate::graph::view::DenseId;
use crate::ir::hlil::{
    Correspondence, Entity, FunctionBuilder, FunctionView, Handler, HandlerKind, RecoverError,
    Recovery, StmtShape, SwitchArm, has_recovery_candidates, pseudocode, recover_structure, verify,
    visit_expression_tree,
};
use crate::test_util::golden::assert_golden;
use alloc::string::ToString;
use alloc::vec;

use super::fixtures::every_statement_shape;
use super::{
    EVERY_WRAP, ExprId, Imp, Kit, Prim, StmtId, ToyHlil, ToyHlilBuilder, Ty, VarId, Wrap, nested,
    on_small_stack,
};

over_toys!(
    counted_loop_recovers_as_for,
    counted_loop_keeps_a_nested_multi_statement_update_in_the_body,
    a_continue_in_the_body_keeps_the_while,
    assigning_diamond_recovers_as_selection,
    nested_assigning_diamonds_recover_as_one_selection,
    returning_diamonds_recover_as_returned_selections,
    mismatched_diamonds_are_copied,
    paired_region_recovers_with_suppressed_exits,
    a_reassigned_view_keeps_the_try,
    constructs_inside_bodies_are_recovered,
    every_statement_shape_is_copied,
    the_source_function_is_untouched,
    signature_and_variables_carry_over,
    malformed_sources_are_reported,
    candidates_are_the_shapes_recovery_starts_from,
    remap_follows_entities_to_their_copies,
    deep_expression_trees_rebuild_without_recursion,
    deeply_nested_statements_recover_without_recursion,
    a_deep_chain_of_diamonds_recovers_as_one_selection_without_recursion,
    deeply_nested_counted_loops_recover_without_recursion,
    deeply_nested_regions_recover_without_recursion,
    nested_regions_on_one_variable_recover_without_recursion,
    many_regions_over_copied_views_recover_in_one_body,
);

fn recovered<D: Kit>(function: &ToyHlil<D>) -> Recovery<ToyHlil<D>, StmtId, ExprId, VarId> {
    recover_structure(function, ToyHlilBuilder::<D>::new()).expect("the function recovers")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UpdateForm {
    Add,
    ExpandedOperand,
}

/// The new value of the index: `index + 1`, or `expanded(index) + 1`.
fn counting_update<D: Kit>(
    builder: &mut ToyHlilBuilder<D>,
    index: VarId,
    form: UpdateForm,
) -> ExprId {
    let left = builder.read(index);
    let one = builder.number(1);
    match form {
        UpdateForm::Add => builder.call(Prim::Add, &[left, one], Ty::Integer),
        UpdateForm::ExpandedOperand => {
            let expanded = builder.call(Prim::Expanded, &[left], Ty::Integer);
            builder.call(Prim::Add, &[expanded, one], Ty::Integer)
        }
    }
}

/// `v1 = 0; while (v1 < v0) { v2 = v2 + v1; <update>; } return v2;`, with the
/// ids of the statements that recovery reasons about.
struct CountingLoop<D: Kit> {
    function: ToyHlil<D>,
    init: StmtId,
    looped: StmtId,
    accumulate: StmtId,
    update: StmtId,
}

fn counting_while<D: Kit>(form: UpdateForm, with_continue: bool) -> CountingLoop<D> {
    let mut builder = ToyHlilBuilder::<D>::new();
    let bound = builder.int_variable(0);
    let index = builder.int_variable(1);
    let total = builder.int_variable(2);
    let zero = builder.number(0);
    let init = builder.store(index, zero);
    let condition_index = builder.read(index);
    let condition_bound = builder.read(bound);
    let condition = builder.call(
        Prim::LessThan,
        &[condition_index, condition_bound],
        Ty::Boolean,
    );
    let sum_left = builder.read(total);
    let sum_right = builder.read(index);
    let sum_value = builder.call(Prim::Add, &[sum_left, sum_right], Ty::Integer);
    let accumulate = builder.store(total, sum_value);
    let update_value = counting_update(&mut builder, index, form);
    let update = builder.store(index, update_value);
    let mut body = vec![accumulate];
    if with_continue {
        let skip = builder.continue_(None);
        let guard = builder.condition();
        body.push(builder.if_(guard, &[skip], &[]));
    }
    body.push(update);
    let looped = builder.while_(condition, &body);
    let result = builder.read(total);
    let ret = builder.return_(&[result]);
    let function = builder.finish(&[init, looped, ret]).unwrap();
    CountingLoop {
        function,
        init,
        looped,
        accumulate,
        update,
    }
}

fn counted_loop_recovers_as_for<D: Kit>() {
    let counting = counting_while::<D>(UpdateForm::Add, false);

    let recovery = recovered(&counting.function);

    let text = pseudocode::<D, _>(&recovery.function);
    assert_eq!(recovery.for_loops, 1, "{text}");
    assert!(verify::<D, _>(&recovery.function).is_ok());
    assert_golden("hlil/recover-for-loop.pseudo", &text);
    // The loop maps to the recovered statement, and the init, the
    // accumulation, and the update keep their copies.
    let recovered_loop = recovery.statement(counting.looped).unwrap();
    assert!(matches!(
        recovery.function.stmt(recovered_loop),
        StmtShape::For { .. }
    ));
    for copied in [counting.init, counting.accumulate, counting.update] {
        let copy = recovery.statement(copied).unwrap();
        assert!(matches!(
            recovery.function.stmt(copy),
            StmtShape::Assign { .. }
        ));
    }
}

fn counted_loop_keeps_a_nested_multi_statement_update_in_the_body<D: Kit>() {
    let counting = counting_while::<D>(UpdateForm::ExpandedOperand, false);

    let recovery = recovered(&counting.function);

    assert_eq!(recovery.for_loops, 0);
    let recovered_loop = recovery.statement(counting.looped).unwrap();
    assert!(matches!(
        recovery.function.stmt(recovered_loop),
        StmtShape::While { .. }
    ));
}

fn a_continue_in_the_body_keeps_the_while<D: Kit>() {
    // `for` runs its update on `continue`; `while` skips the trailing
    // statement. The rewrite would change behavior.
    let counting = counting_while::<D>(UpdateForm::Add, true);

    let recovery = recovered(&counting.function);

    assert_eq!(recovery.for_loops, 0);
    let recovered_loop = recovery.statement(counting.looped).unwrap();
    assert!(matches!(
        recovery.function.stmt(recovered_loop),
        StmtShape::While { .. }
    ));
}

/// `if (v0) { v1 = 1; } else { v1 = 2; } return v1;`
fn assigning_diamond<D: Kit>() -> (ToyHlil<D>, [StmtId; 3]) {
    let mut builder = ToyHlilBuilder::<D>::new();
    let flag = builder.int_variable(0);
    let out = builder.int_variable(1);
    let condition = builder.read_variable(flag, &D::ty(Ty::Boolean));
    let one = builder.number(1);
    let then_assign = builder.store(out, one);
    let two = builder.number(2);
    let else_assign = builder.store(out, two);
    let diamond = builder.if_(condition, &[then_assign], &[else_assign]);
    let result = builder.read(out);
    let ret = builder.return_(&[result]);
    let function = builder.finish(&[diamond, ret]).unwrap();
    (function, [diamond, then_assign, else_assign])
}

fn assigning_diamond_recovers_as_selection<D: Kit>() {
    let (function, [diamond, then_assign, else_assign]) = assigning_diamond::<D>();

    let recovery = recovered(&function);

    let text = pseudocode::<D, _>(&recovery.function);
    assert_eq!(recovery.selects, 1, "{text}");
    assert!(verify::<D, _>(&recovery.function).is_ok());
    // The diamond became one selection: no `if` survives in the golden.
    assert_golden("hlil/recover-select.pseudo", &text);
    // The diamond and both arms map to the single recovered assignment.
    let assignment = recovery.statement(diamond).unwrap();
    assert_eq!(recovery.statement(then_assign), Some(assignment));
    assert_eq!(recovery.statement(else_assign), Some(assignment));
    let StmtShape::Assign { target, value } = recovery.function.stmt(assignment) else {
        panic!("the selection is an assignment");
    };
    // The selection takes the type of the target, not of its condition.
    assert_eq!(recovery.function.ty(value), &D::ty(Ty::Integer));
    assert_eq!(recovery.function.ty(target), &D::ty(Ty::Integer));
}

fn nested_assigning_diamonds_recover_as_one_selection<D: Kit>() {
    // if (v0) { v2 = 1; } else { if (v1) { v2 = 2; } else { v2 = 3; } }
    let mut builder = ToyHlilBuilder::<D>::new();
    let outer_flag = builder.int_variable(0);
    let inner_flag = builder.int_variable(1);
    let out = builder.int_variable(2);
    let outer_condition = builder.read_variable(outer_flag, &D::ty(Ty::Boolean));
    let inner_condition = builder.read_variable(inner_flag, &D::ty(Ty::Boolean));
    let one = builder.number(1);
    let set_one = builder.store(out, one);
    let two = builder.number(2);
    let set_two = builder.store(out, two);
    let three = builder.number(3);
    let set_three = builder.store(out, three);
    let inner = builder.if_(inner_condition, &[set_two], &[set_three]);
    let outer = builder.if_(outer_condition, &[set_one], &[inner]);
    let function = builder.finish(&[outer]).unwrap();

    let recovery = recovered(&function);

    assert_eq!(recovery.selects, 1);
    assert_eq!(
        pseudocode::<D, _>(&recovery.function),
        "v2 = select(v0, 1, select(v1, 2, 3));\n"
    );
    let assignment = recovery.statement(outer).unwrap();
    for consumed in [inner, set_one, set_two, set_three] {
        assert_eq!(recovery.statement(consumed), Some(assignment));
    }
}

fn returning_diamonds_recover_as_returned_selections<D: Kit>() {
    // if (v0) { return 1; } else { if (v1) { return 2; } else { return 3; } }
    let mut builder = ToyHlilBuilder::<D>::new();
    let outer_flag = builder.int_variable(0);
    let inner_flag = builder.int_variable(1);
    let outer_condition = builder.read_variable(outer_flag, &D::ty(Ty::Boolean));
    let inner_condition = builder.read_variable(inner_flag, &D::ty(Ty::Boolean));
    let one = builder.number(1);
    let return_one = builder.return_(&[one]);
    let two = builder.number(2);
    let return_two = builder.return_(&[two]);
    let three = builder.number(3);
    let return_three = builder.return_(&[three]);
    let inner = builder.if_(inner_condition, &[return_two], &[return_three]);
    let outer = builder.if_(outer_condition, &[return_one], &[inner]);
    let function = builder.finish(&[outer]).unwrap();

    let recovery = recovered(&function);

    assert_eq!(recovery.selects, 1);
    assert_eq!(
        pseudocode::<D, _>(&recovery.function),
        "return select(v0, 1, select(v1, 2, 3));\n"
    );
    let returned = recovery.statement(outer).unwrap();
    for consumed in [inner, return_one, return_two, return_three] {
        assert_eq!(recovery.statement(consumed), Some(returned));
    }
    let StmtShape::Return { values } = recovery.function.stmt(returned) else {
        panic!("the selection is a return");
    };
    assert_eq!(recovery.function.ty(values[0]), &D::ty(Ty::Integer));
}

fn mismatched_diamonds_are_copied<D: Kit>() {
    // The arms assign different variables, and one arm has two statements.
    let mut builder = ToyHlilBuilder::<D>::new();
    let flag = builder.int_variable(0);
    let left = builder.int_variable(1);
    let right = builder.int_variable(2);
    let condition = builder.read_variable(flag, &D::ty(Ty::Boolean));
    let one = builder.number(1);
    let set_left = builder.store(left, one);
    let two = builder.number(2);
    let set_right = builder.store(right, two);
    let also_right = builder.noop();
    let diamond = builder.if_(condition, &[set_left], &[set_right, also_right]);
    let function = builder.finish(&[diamond]).unwrap();

    let recovery = recovered(&function);

    assert_eq!(recovery.selects, 0);
    assert_eq!(
        pseudocode::<D, _>(&recovery.function),
        pseudocode::<D, _>(&function)
    );
}

/// The ids of the statements that a recovered region consumes, by role.
struct Guard<D: Kit> {
    function: ToyHlil<D>,
    lock: VarId,
    enter: StmtId,
    guarded: StmtId,
    normal_exit: StmtId,
    cleanup_exit: StmtId,
    cleanup_call: ExprId,
    rethrow: StmtId,
}

/// `acquire(lock); try { out = 9; release(lock) } catch-any { release(lock);
/// throw(caught) }`, with an optional reassignment of `lock` in the body.
fn guarded_function<D: Kit>(reassign_lock: bool) -> Guard<D> {
    let mut builder = ToyHlilBuilder::<D>::new();
    let lock = builder.int_variable(0);
    let out = builder.int_variable(1);
    let subject = builder.read(lock);
    let enter = builder.effect(Prim::Acquire, &[subject]);
    let nine = builder.number(9);
    let work = builder.store(out, nine);
    let mut protected = vec![work];
    if reassign_lock {
        let zero = builder.number(0);
        protected.push(builder.store(lock, zero));
    }
    let exit_subject = builder.read(lock);
    let normal_exit = builder.effect(Prim::Release, &[exit_subject]);
    protected.push(normal_exit);
    let cleanup_subject = builder.read(lock);
    let cleanup_call = builder.call(Prim::Release, &[cleanup_subject], Ty::Void);
    let cleanup_exit = builder.expression_statement(cleanup_call);
    let caught = builder.call(Prim::Caught, &[], Ty::Integer);
    let rethrow_call = builder.call(Prim::Throw, &[caught], Ty::Void);
    let rethrow = builder.expression_statement(rethrow_call);
    let handler = Handler {
        kind: HandlerKind::CatchAll,
        binding: None,
        caught_types: &[],
        body: &[cleanup_exit, rethrow],
    };
    let guarded = builder.try_(&protected, &[handler], &[]);
    let function = builder.finish(&[enter, guarded]).unwrap();
    Guard {
        function,
        lock,
        enter,
        guarded,
        normal_exit,
        cleanup_exit,
        cleanup_call,
        rethrow,
    }
}

fn paired_region_recovers_with_suppressed_exits<D: Kit>() {
    let guard = guarded_function::<D>(false);

    let recovery = recovered(&guard.function);

    let text = pseudocode::<D, _>(&recovery.function);
    assert_eq!(recovery.regions, 1, "{text}");
    assert!(verify::<D, _>(&recovery.function).is_ok());
    // The pair became one region: its release and its `try` are gone.
    assert_golden("hlil/recover-region.pseudo", &text);
    // Enter, try, cleanup, and the suppressed normal exit all map onto the
    // region.
    let region = recovery.statement(guard.enter).unwrap();
    for consumed in [
        guard.guarded,
        guard.normal_exit,
        guard.cleanup_exit,
        guard.rethrow,
    ] {
        assert_eq!(recovery.statement(consumed), Some(region));
    }
    assert!(matches!(
        recovery.function.stmt(region),
        StmtShape::Region { .. }
    ));
}

fn a_reassigned_view_keeps_the_try<D: Kit>() {
    let guard = guarded_function::<D>(true);

    let recovery = recovered(&guard.function);

    assert_eq!(recovery.regions, 0);
    let guarded = recovery.statement(guard.guarded).unwrap();
    assert!(matches!(
        recovery.function.stmt(guarded),
        StmtShape::Try { .. }
    ));
}

fn constructs_inside_bodies_are_recovered<D: Kit>() {
    // `if (c) { v1 = 0; while (v1 < v0) { v1 = v1 + 1; } }`
    let mut builder = ToyHlilBuilder::<D>::new();
    let bound = builder.int_variable(0);
    let index = builder.int_variable(1);
    let zero = builder.number(0);
    let init = builder.store(index, zero);
    let read_index = builder.read(index);
    let read_bound = builder.read(bound);
    let condition = builder.call(Prim::LessThan, &[read_index, read_bound], Ty::Boolean);
    let update_value = counting_update(&mut builder, index, UpdateForm::Add);
    let update = builder.store(index, update_value);
    let looped = builder.while_(condition, &[update]);
    let guard = builder.condition();
    let wrapper = builder.if_(guard, &[init, looped], &[]);
    let function = builder.finish(&[wrapper]).unwrap();

    let recovery = recovered(&function);

    assert_eq!(recovery.for_loops, 1);
    assert_eq!(
        pseudocode::<D, _>(&recovery.function),
        "if (1) {\n    for (\n        v1 = 0;\n        ; lt(v1, v0) ;\n        v1 = add(v1, 1);\n    ) {\n    }\n}\n"
    );
}

fn every_statement_shape_is_copied<D: Kit>() {
    let function = every_statement_shape::<D>();

    let recovery = recovered(&function);

    assert_eq!(
        (recovery.for_loops, recovery.selects, recovery.regions),
        (0, 0, 0)
    );
    assert_eq!(
        pseudocode::<D, _>(&recovery.function),
        pseudocode::<D, _>(&function)
    );
    assert!(verify::<D, _>(&recovery.function).is_ok());
}

fn the_source_function_is_untouched<D: Kit>() {
    let counting = counting_while::<D>(UpdateForm::Add, false);
    let before = pseudocode::<D, _>(&counting.function);

    let _ = recovered(&counting.function);

    assert_eq!(pseudocode::<D, _>(&counting.function), before);
}

fn signature_and_variables_carry_over<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let first = builder.int_variable(3);
    let untyped = builder.declare_variable(&D::variable(4), None);
    let second = builder.int_variable(5);
    builder.set_signature(&[second, first], &[D::ty(Ty::Integer), D::ty(Ty::Boolean)]);
    let statement = builder.use_of(first);
    let function = builder.finish(&[statement]).unwrap();

    let recovery = recovered(&function);

    let rebuilt = &recovery.function;
    assert_eq!(rebuilt.variable_count(), 3);
    for variable in [first, untyped, second] {
        let copy = recovery.variable(variable);
        assert_eq!(rebuilt.variable(copy), function.variable(variable));
        assert_eq!(
            rebuilt.declared_type(copy),
            function.declared_type(variable)
        );
    }
    assert_eq!(
        rebuilt.parameters(),
        [recovery.variable(second), recovery.variable(first)]
    );
    assert_eq!(rebuilt.returns(), function.returns());
}

fn malformed_sources_are_reported<D: Kit>() {
    let malformed = |build: fn(&mut ToyHlilBuilder<D>) -> StmtId| {
        let mut builder = ToyHlilBuilder::<D>::new();
        let statement = build(&mut builder);
        let function = builder.finish_unchecked(&[statement]);
        match recover_structure(&function, ToyHlilBuilder::<D>::new()) {
            Err(RecoverError::Malformed(message)) => message,
            other => panic!("expected a malformed source, got {other:?}"),
        }
    };

    assert_eq!(
        malformed(|builder| {
            let condition = builder.condition();
            builder.if_(condition, &[StmtId::from_index(40)], &[])
        }),
        "unresolvable statement identity"
    );
    assert_eq!(
        malformed(|builder| builder.expression_statement(ExprId::from_index(41))),
        "unresolvable expression identity"
    );
    assert_eq!(
        malformed(|builder| {
            let read = builder.read_variable(VarId::from_index(42), &D::ty(Ty::Integer));
            builder.expression_statement(read)
        }),
        "unresolvable variable identity"
    );
}

fn candidates_are_the_shapes_recovery_starts_from<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let body = builder.noop();
    let looped = builder.loop_(&[body]);
    let counted = builder.for_(&[], None, &[], &[]);
    let scrutinee = builder.number(0);
    let switch = builder.switch(
        scrutinee,
        &[SwitchArm {
            values: &[D::constant(1)],
            body: &[],
        }],
        &[],
    );
    let plain = builder.finish(&[looped, counted, switch]).unwrap();
    assert!(!has_recovery_candidates::<D, _>(&plain));

    assert!(has_recovery_candidates::<D, _>(
        &counting_while::<D>(UpdateForm::Add, false).function
    ));
    assert!(has_recovery_candidates::<D, _>(&assigning_diamond::<D>().0));
    assert!(has_recovery_candidates::<D, _>(
        &guarded_function::<D>(false).function
    ));
}

fn remap_follows_entities_to_their_copies<D: Kit>() {
    let guard = guarded_function::<D>(false);
    let point = |index| ProgramPoint {
        block: BlockId::from_index(0),
        inst_idx: index,
    };
    let table = Correspondence {
        instructions: vec![
            (point(0), Entity::Statement(guard.enter)),
            (point(1), Entity::Statement(guard.rethrow)),
            (point(2), Entity::Expression(guard.cleanup_call)),
            (point(3), Entity::Variable(guard.lock)),
        ],
        variables: vec![guard.lock],
    };
    let recovery = recovered(&guard.function);

    let remapped = table.remap(&recovery);

    let region = recovery.statement(guard.enter).unwrap();
    // The cleanup of a recovered region is regenerated, so its expression has
    // no copy and its row is dropped; every other row follows its entity.
    assert_eq!(
        remapped.instructions,
        [
            (point(0), Entity::Statement(region)),
            (point(1), Entity::Statement(region)),
            (point(3), Entity::Variable(recovery.variable(guard.lock))),
        ]
    );
    assert_eq!(remapped.variables, [recovery.variable(guard.lock)]);
}

/// The walks and the rebuild keep their own stacks, so an operand chain as
/// deep as generated code can produce does not overflow the thread's.
fn deep_expression_trees_rebuild_without_recursion<D: Kit>() {
    const DEPTH: usize = 20_000;

    let mut builder = ToyHlilBuilder::<D>::new();
    let counter = builder.int_variable(0);
    let mut sum = builder.read(counter);
    for _ in 0..DEPTH {
        let one = builder.number(1);
        sum = builder.call(Prim::Add, &[sum, one], Ty::Integer);
    }
    let statement = builder.expression_statement(sum);
    let function = builder.finish(&[statement]).unwrap();

    let recovery = recovered(&function);

    assert!(verify::<D, _>(&recovery.function).is_ok());
    assert_eq!(recovery.function.expr_count(), function.expr_count());
    let rebuilt = recovery.expression(sum).unwrap();
    let mut visited = 0;
    visit_expression_tree(&recovery.function, rebuilt, &mut |_| visited += 1);
    assert_eq!(visited, 2 * DEPTH + 1);
}

/// Recovers the function that `build` makes on a thread whose stack could
/// not hold a frame per level, and returns what `summarize` makes of the
/// function and its recovery.
fn recovered_on_a_small_stack<D: Kit, T: Send>(
    build: impl FnOnce() -> ToyHlil<D> + Send,
    summarize: impl FnOnce(&ToyHlil<D>, &Recovery<ToyHlil<D>, StmtId, ExprId, VarId>) -> T + Send,
) -> T {
    on_small_stack(|| {
        let function = build();
        let recovery = recovered(&function);
        summarize(&function, &recovery)
    })
}

/// The statement walk keeps its own stack, so a function nested as deep as
/// untrusted input can make it is rebuilt on a thread whose stack could not
/// hold a frame per level, with every statement and expression copied. The
/// checks for a selection read each statement of a nest once, however many
/// levels ask.
fn deeply_nested_statements_recover_without_recursion<D: Kit>() {
    const DEPTH: usize = 100_000;

    for wraps in [
        &[Wrap::If][..],
        &[Wrap::If, Wrap::While],
        &[Wrap::While],
        &EVERY_WRAP,
    ] {
        let (recoveries, same_size, all_copied) = recovered_on_a_small_stack::<D, _>(
            || nested::<D>(wraps, DEPTH, ToyHlilBuilder::noop),
            |function, recovery| {
                (
                    [recovery.for_loops, recovery.selects, recovery.regions],
                    recovery.function.stmt_count() == function.stmt_count()
                        && recovery.function.expr_count() == function.expr_count(),
                    recovery.statements.iter().all(Option::is_some)
                        && recovery.expressions.iter().all(Option::is_some),
                )
            },
        );

        assert_eq!(recoveries, [0, 0, 0], "{wraps:?}");
        assert!(same_size, "{wraps:?}");
        assert!(all_copied, "{wraps:?}");
    }
}

/// `depth` diamonds nested in their then-arms, each with the arm that `leaf`
/// builds as its else-arm, around one more such arm.
fn diamond_chain<D: Kit>(
    depth: usize,
    leaf: fn(&mut ToyHlilBuilder<D>, VarId) -> StmtId,
) -> ToyHlil<D> {
    let mut builder = ToyHlilBuilder::<D>::new();
    let out = builder.int_variable(0);
    let mut inner = leaf(&mut builder, out);
    for _ in 0..depth {
        let otherwise = leaf(&mut builder, out);
        let condition = builder.condition();
        inner = builder.if_(condition, &[inner], &[otherwise]);
    }
    builder.finish_unchecked(&[inner])
}

/// Every walk over the diamonds of a selection, the checks that the diamond
/// is one and the build of its expression, keeps its own stack.
fn a_deep_chain_of_diamonds_recovers_as_one_selection_without_recursion<D: Kit>() {
    const DEPTH: usize = 100_000;

    let leaves: [fn(&mut ToyHlilBuilder<D>, VarId) -> StmtId; 2] = [
        |builder, out| {
            let value = builder.number(1);
            builder.store(out, value)
        },
        |builder, _| {
            let value = builder.number(1);
            builder.return_(&[value])
        },
    ];
    for leaf in leaves {
        let (selects, statements, all_mapped) = recovered_on_a_small_stack::<D, _>(
            || diamond_chain::<D>(DEPTH, leaf),
            |_, recovery| {
                (
                    recovery.selects,
                    recovery.function.stmt_count(),
                    recovery.statements.iter().all(Option::is_some),
                )
            },
        );

        assert_eq!(selects, 1);
        assert_eq!(statements, 1, "the whole chain is one statement");
        assert!(all_mapped, "every diamond and arm maps onto it");
    }
}

/// `depth` counted loops, each in the body of the one outside it:
/// `v1 = 0; while (v1 < v0) { <inner>; v1 = v1 + 1; }`.
fn counted_nest<D: Kit>(depth: usize) -> ToyHlil<D> {
    let mut builder = ToyHlilBuilder::<D>::new();
    let bound = builder.int_variable(0);
    let index = builder.int_variable(1);
    let mut body = vec![builder.noop()];
    for _ in 0..depth {
        let zero = builder.number(0);
        let init = builder.store(index, zero);
        let read_index = builder.read(index);
        let read_bound = builder.read(bound);
        let condition = builder.call(Prim::LessThan, &[read_index, read_bound], Ty::Boolean);
        let update_value = counting_update(&mut builder, index, UpdateForm::Add);
        body.push(builder.store(index, update_value));
        let looped = builder.while_(condition, &body);
        body = vec![init, looped];
    }
    builder.finish_unchecked(&body)
}

/// `depth` guarded regions, each in the guarded body of the one outside it:
/// `acquire(l); try { <inner>; release(l) } catch-any { release(l);
/// throw(caught) }`, where `l` is one variable for every level when `shared`
/// and the level's own variable otherwise.
fn guarded_nest<D: Kit>(depth: usize, shared: bool) -> ToyHlil<D> {
    let mut builder = ToyHlilBuilder::<D>::new();
    let shared_lock = builder.int_variable(0);
    let mut body = vec![builder.noop()];
    for _ in 0..depth {
        let lock = if shared {
            shared_lock
        } else {
            builder.int_variable(0)
        };
        let subject = builder.read(lock);
        let enter = builder.effect(Prim::Acquire, &[subject]);
        let exit_subject = builder.read(lock);
        body.push(builder.effect(Prim::Release, &[exit_subject]));
        let cleanup_subject = builder.read(lock);
        let cleanup_call = builder.call(Prim::Release, &[cleanup_subject], Ty::Void);
        let cleanup_exit = builder.expression_statement(cleanup_call);
        let caught = builder.call(Prim::Caught, &[], Ty::Integer);
        let rethrow_call = builder.call(Prim::Throw, &[caught], Ty::Void);
        let rethrow = builder.expression_statement(rethrow_call);
        let handler = Handler {
            kind: HandlerKind::CatchAll,
            binding: None,
            caught_types: &[],
            body: &[cleanup_exit, rethrow],
        };
        let guarded = builder.try_(&body, &[handler], &[]);
        body = vec![enter, guarded];
    }
    builder.finish_unchecked(&body)
}

/// The recovery that rebuilds an initializer and a loop into a `for` keeps
/// its own frame as well, and decides whether to recover from what the loop
/// body contains without reading the body again at every level.
fn deeply_nested_counted_loops_recover_without_recursion<D: Kit>() {
    const DEPTH: usize = 100_000;

    let for_loops = recovered_on_a_small_stack::<D, _>(
        || counted_nest::<D>(DEPTH),
        |_, recovery| recovery.for_loops,
    );

    assert_eq!(for_loops, DEPTH);
}

/// The recovery that rebuilds an enter and its `try` into a region keeps its
/// own frame too, and decides whether to recover from what the guarded body
/// assigns without reading the body again at every level. A statement costs
/// the open regions that hold the variable that it names, so a nest with a
/// variable per region is linear.
fn deeply_nested_regions_recover_without_recursion<D: Kit>() {
    const DEPTH: usize = 100_000;

    let regions = recovered_on_a_small_stack::<D, _>(
        || guarded_nest::<D>(DEPTH, false),
        |_, recovery| recovery.regions,
    );

    assert_eq!(regions, DEPTH);
}

/// Every statement that names the shared variable asks the dialect, once for
/// each open region, whether that region releases it, so a nest on one
/// variable costs a call per pair of region and statement and is shallower.
fn nested_regions_on_one_variable_recover_without_recursion<D: Kit>() {
    const DEPTH: usize = 2_000;

    let regions = recovered_on_a_small_stack::<D, _>(
        || guarded_nest::<D>(DEPTH, true),
        |_, recovery| recovery.regions,
    );

    assert_eq!(regions, DEPTH);
}

/// `count` regions one after another in one body, each over its own lock and
/// guarded through a copy of it: `view = lock; acquire(lock); try {
/// release(view) } catch-any { release(view); throw(caught) }`.
fn copied_regions<D: Kit>(count: usize) -> ToyHlil<D> {
    let mut builder = ToyHlilBuilder::<D>::new();
    let mut body = vec![];
    for _ in 0..count {
        let lock = builder.int_variable(0);
        let view = builder.int_variable(1);
        let source = builder.read(lock);
        body.push(builder.store(view, source));
        let subject = builder.read(lock);
        body.push(builder.effect(Prim::Acquire, &[subject]));
        let exit_subject = builder.read(view);
        let exit = builder.effect(Prim::Release, &[exit_subject]);
        let cleanup_subject = builder.read(view);
        let cleanup_call = builder.call(Prim::Release, &[cleanup_subject], Ty::Void);
        let cleanup_exit = builder.expression_statement(cleanup_call);
        let caught = builder.call(Prim::Caught, &[], Ty::Integer);
        let rethrow_call = builder.call(Prim::Throw, &[caught], Ty::Void);
        let rethrow = builder.expression_statement(rethrow_call);
        let handler = Handler {
            kind: HandlerKind::CatchAll,
            binding: None,
            caught_types: &[],
            body: &[cleanup_exit, rethrow],
        };
        body.push(builder.try_(&[exit], &[handler], &[]));
    }
    builder.finish_unchecked(&body)
}

/// A region whose released view is a copy of the entered object needs the
/// copy among the statements before it, which each region of a body reads
/// from one scan of the statements before it and not from a scan of its own.
fn many_regions_over_copied_views_recover_in_one_body<D: Kit>() {
    const COUNT: usize = 100_000;

    let regions = recovered_on_a_small_stack::<D, _>(
        || copied_regions::<D>(COUNT),
        |_, recovery| recovery.regions,
    );

    assert_eq!(regions, COUNT);
}

#[test]
fn imp_does_not_recover_boolean_targets() {
    // The hook says a boolean assignment is not one expression, so the
    // diamond stays a diamond.
    let mut builder = ToyHlilBuilder::<Imp>::new();
    let flag = builder.int_variable(0);
    let out = builder.int_variable(1);
    let condition = builder.read_variable(flag, &Imp::ty(Ty::Boolean));
    let mut assign = |value| {
        let target = builder.read_variable(out, &Imp::ty(Ty::Boolean));
        let constant = builder.number(value);
        builder.assign(target, constant)
    };
    let then_assign = assign(1);
    let else_assign = assign(2);
    let diamond = builder.if_(condition, &[then_assign], &[else_assign]);
    let function = builder.finish(&[diamond]).unwrap();

    let recovery = recovered(&function);

    assert_eq!(recovery.selects, 0);
    assert_eq!(
        pseudocode::<Imp, _>(&recovery.function),
        "if (v0) {\n    v1 = 1;\n} else {\n    v1 = 2;\n}\n"
    );
}

#[test]
fn builder_errors_are_reported() {
    // The rebuild repeats the source's comparison of a boolean, which `Imp`
    // rejects when the builder seals the function.
    let mut builder = ToyHlilBuilder::<Imp>::new();
    let flag = builder.condition();
    let seven = builder.number(7);
    let compare = builder.call(Prim::LessThan, &[flag, seven], Ty::Boolean);
    let statement = builder.expression_statement(compare);
    let function = builder.finish_unchecked(&[statement]);

    let error = recover_structure(&function, ToyHlilBuilder::<Imp>::new())
        .expect_err("the rebuilt function is rejected");

    assert!(matches!(error, RecoverError::Builder(_)));
    assert_eq!(
        error.to_string(),
        "HLIL builder failed: HLIL verification failed; \
         e2 compares the non-integer operand e0"
    );
}
