//! `recover_structure` and `has_recovery_candidates`: what recovery builds
//! from the shapes a lift or a bytecode frontend produces, what it refuses
//! because it could not keep the behavior, and its tables.

use std::error::Error;
use std::panic::{AssertUnwindSafe, catch_unwind};

use cfglib::ir::hlil::{RecoverError, has_recovery_candidates, recover_structure, verify};

use crate::dialect::{Calc, Op};
use crate::pipeline::{lifted, lowered, recovered, structured, text};
use crate::programs::{self, Case};
use crate::source::{
    Program, S, add, catch_all, caught, emit, eq, ge, if_then, int, lt, raise, ret, set, sub, var,
};
use crate::structured::{Builder, ExprId, ExprNode, StmtId, StmtNode, VarId};
use crate::structured_run::run_structured;

fn has_candidates(case: &Case) -> bool {
    has_recovery_candidates::<Calc, _>(&structured(&case.program))
}

#[test]
fn only_an_if_a_while_or_a_try_is_a_recovery_candidate() {
    for (case, expected) in [
        (programs::straight_line(), false),
        (programs::counted_for(), false),
        (programs::do_while_steps(), false),
        (programs::guarded_body(), false),
        (programs::explicit_pick(), false),
        (programs::branches(), true),
        (programs::sum_while(), true),
        (programs::division_guard(), true),
        (programs::break_continue(), true),
    ] {
        assert_eq!(has_candidates(&case), expected, "{}", case.program.name);
    }
}

#[test]
fn recovery_counts_the_constructs_it_builds() {
    // Counts of recover(lift(lower(program))): for loops, selections, regions.
    for (case, expected) in [
        (programs::counting_while(), (1, 0, 0)),
        (programs::sum_while(), (1, 0, 0)),
        (programs::counted_for(), (1, 0, 0)),
        (programs::nested_loops(), (2, 0, 0)),
        (programs::memory_walk(), (2, 0, 0)),
        (programs::select_assign(), (0, 1, 0)),
        (programs::select_return(), (0, 1, 0)),
        (programs::select_guarding_division(), (0, 1, 0)),
        (programs::acquire_with_cleanup(), (0, 0, 1)),
        (programs::while_break_continue(), (0, 0, 0)),
        (programs::branches(), (0, 0, 0)),
        (programs::division_guard(), (0, 0, 0)),
    ] {
        let code = lowered(&structured(&case.program)).function;
        let recovery = recovered(&lifted(&code).function);
        assert_eq!(
            (recovery.for_loops, recovery.selects, recovery.regions),
            expected,
            "{}",
            case.program.name
        );
    }
}

#[test]
fn a_counting_loop_becomes_a_for_that_carries_its_initializer_and_update() {
    let code = lowered(&structured(&programs::counting_while().program)).function;
    let function = recovered(&lifted(&code).function).function;
    let StmtNode::For {
        initializer,
        condition,
        update,
        body,
    } = &function.stmts[function.body[0].index()]
    else {
        panic!("the loop is a for");
    };
    assert_eq!((initializer.len(), update.len(), body.len()), (1, 1, 1));
    assert!(condition.is_some());
    assert_eq!(function.body.len(), 2, "the initializer left the body");
}

#[test]
fn nested_counting_loops_recover_at_both_levels() {
    let code = lowered(&structured(&programs::nested_loops().program)).function;
    let function = recovered(&lifted(&code).function).function;
    let outer = function
        .stmts
        .iter()
        .filter_map(|stmt| match stmt {
            StmtNode::For { body, .. } => Some(body),
            _ => None,
        })
        .find(|body| {
            body.iter()
                .any(|&child| matches!(function.stmts[child.index()], StmtNode::For { .. }))
        });
    assert!(outer.is_some(), "a for holds the inner for");
}

/// Runs `program` before and after recovery and returns the recovery.
fn recovers_with_the_same_behavior(
    program: &Program,
    inputs: &[&[i64]],
) -> crate::pipeline::Recovered {
    let source = structured(program);
    let recovery = recovered(&source);
    for input in inputs {
        assert_eq!(
            run_structured(&recovery.function, input),
            run_structured(&source, input),
            "{} on {input:?}",
            program.name
        );
    }
    recovery
}

#[test]
fn a_while_with_a_continue_is_not_rebuilt_as_a_for() {
    // A `for` runs its update on `continue`, and the `while` form skips the
    // trailing statement, so rebuilding it would run the increment twice.
    let program = Program::new(
        "continues_before_the_update",
        &["n"],
        &["i"],
        vec![
            set("i", int(0)),
            S::While(
                lt(var("i"), var("n")),
                vec![
                    if_then(
                        eq(var("i"), int(2)),
                        vec![set("i", add(var("i"), int(1))), S::Continue(None)],
                    ),
                    emit(var("i")),
                    set("i", add(var("i"), int(1))),
                ],
            ),
            ret(var("i")),
        ],
    );
    let recovery = recovers_with_the_same_behavior(&program, &[&[0], &[3], &[6]]);
    assert_eq!(recovery.for_loops, 0);
}

#[test]
fn a_diamond_that_assigns_different_variables_is_not_a_selection() {
    let program = Program::new(
        "two_targets",
        &["a", "b"],
        &["x", "y"],
        vec![
            S::If(
                lt(var("a"), var("b")),
                vec![set("x", var("a"))],
                vec![set("y", var("b"))],
            ),
            ret(add(var("x"), var("y"))),
        ],
    );
    let recovery = recovers_with_the_same_behavior(&program, &[&[1, 2], &[2, 1]]);
    assert_eq!(recovery.selects, 0);
}

#[test]
fn an_arm_of_two_statements_is_not_a_selection() {
    let program = Program::new(
        "two_statement_arm",
        &["a"],
        &["x"],
        vec![
            S::If(
                lt(var("a"), int(0)),
                vec![emit(var("a")), set("x", int(1))],
                vec![set("x", int(2))],
            ),
            ret(var("x")),
        ],
    );
    let recovery = recovers_with_the_same_behavior(&program, &[&[-1], &[1]]);
    assert_eq!(recovery.selects, 0);
}

#[test]
fn nested_diamonds_recover_as_one_nested_selection() {
    let program = Program::new(
        "three_way",
        &["a", "b", "c"],
        &["x"],
        vec![
            S::If(
                lt(var("a"), int(0)),
                vec![set("x", var("a"))],
                vec![S::If(
                    lt(var("b"), int(0)),
                    vec![set("x", var("b"))],
                    vec![set("x", var("c"))],
                )],
            ),
            ret(var("x")),
        ],
    );
    let recovery =
        recovers_with_the_same_behavior(&program, &[&[-1, 0, 0], &[1, -1, 0], &[1, 1, 7]]);
    assert_eq!(recovery.selects, 1);
    let function = &recovery.function;
    let StmtNode::Assign { value, .. } = &function.stmts[function.body[0].index()] else {
        panic!("the diamond became one assignment");
    };
    let ExprNode::Op {
        op: Op::Pick,
        operands,
    } = &function.exprs[value.index()].node
    else {
        panic!("the assignment selects");
    };
    assert!(matches!(
        &function.exprs[operands[2].index()].node,
        ExprNode::Op { op: Op::Pick, .. }
    ));
}

#[test]
fn a_selection_evaluates_only_the_arm_it_takes() {
    // `a / b` throws for `b == 0`. The source never evaluates it then, and
    // the recovered `pick` must not either.
    let program = programs::select_guarding_division();
    let source = structured(&program.program);
    let recovery = recovered(&source);
    assert_eq!(recovery.selects, 1);
    let uncaught = run_structured(&recovery.function, &[9, 0]);
    assert_eq!(uncaught, run_structured(&source, &[9, 0]));
    assert!(matches!(uncaught.end, crate::machine::End::Returned(_)));
}

fn region_program(body: Vec<S>, cleanup: Vec<S>) -> Program {
    Program::new(
        "region_shape",
        &["x"],
        &[],
        vec![
            S::Do(crate::source::apply(Op::Acquire, vec![var("x")])),
            S::Try(body, vec![catch_all(None, cleanup)], vec![]),
            ret(int(0)),
        ],
    )
}

fn release() -> S {
    S::Do(crate::source::apply(Op::Release, vec![var("x")]))
}

#[test]
fn an_acquire_and_a_release_around_a_rethrowing_cleanup_become_a_region() {
    let program = region_program(
        vec![emit(var("x")), release()],
        vec![release(), raise(caught())],
    );
    let recovery = recovers_with_the_same_behavior(&program, &[&[1], &[0]]);
    assert_eq!(recovery.regions, 1);
    let function = &recovery.function;
    assert!(matches!(
        &function.stmts[function.body[0].index()],
        StmtNode::Region { op: Op::Guard, .. }
    ));
}

#[test]
fn a_region_is_not_built_when_the_body_reassigns_the_acquired_variable() {
    let program = region_program(vec![set_x(), release()], vec![release(), raise(caught())]);
    let recovery = recovers_with_the_same_behavior(&program, &[&[1], &[0]]);
    assert_eq!(recovery.regions, 0);
}

fn set_x() -> S {
    set("x", add(var("x"), int(1)))
}

#[test]
fn a_cleanup_that_swallows_the_exception_is_not_a_region() {
    let program = region_program(vec![emit(var("x")), release()], vec![release()]);
    let recovery = recovers_with_the_same_behavior(&program, &[&[1], &[0]]);
    assert_eq!(recovery.regions, 0);
}

#[test]
fn a_cleanup_that_releases_something_else_is_not_a_region() {
    let program = Program::new(
        "wrong_release",
        &["x", "y"],
        &[],
        vec![
            S::Do(crate::source::apply(Op::Acquire, vec![var("x")])),
            S::Try(
                vec![emit(var("x")), release()],
                vec![catch_all(
                    None,
                    vec![
                        S::Do(crate::source::apply(Op::Release, vec![var("y")])),
                        raise(caught()),
                    ],
                )],
                vec![],
            ),
            ret(int(0)),
        ],
    );
    let recovery = recovers_with_the_same_behavior(&program, &[&[1, 2], &[0, 2]]);
    assert_eq!(recovery.regions, 0);
}

#[test]
fn recovering_a_recovered_function_changes_nothing() {
    for case in programs::all() {
        let code = lowered(&structured(&case.program)).function;
        let once = recovered(&lifted(&code).function);
        let twice = recovered(&once.function);
        assert_eq!(
            (twice.for_loops, twice.selects, twice.regions),
            (0, 0, 0),
            "{}",
            case.program.name
        );
        // The arenas may fill in another order, so compare what they say.
        assert_eq!(
            text(&twice.function),
            text(&once.function),
            "{}",
            case.program.name
        );
    }
}

#[test]
fn recovery_keeps_a_function_that_has_nothing_to_recover() {
    for case in [
        programs::straight_line(),
        programs::counted_for(),
        programs::explicit_pick(),
    ] {
        let source = structured(&case.program);
        let recovery = recovered(&source);
        assert_eq!(
            (recovery.for_loops, recovery.selects, recovery.regions),
            (0, 0, 0)
        );
        assert_eq!(recovery.function, source, "{}", case.program.name);
    }
}

#[test]
fn the_tables_name_a_copy_of_every_source_entity() {
    for case in programs::all() {
        let source = structured(&case.program);
        let recovery = recovered(&source);
        assert_eq!(recovery.statements.len(), source.stmts.len());
        assert_eq!(recovery.expressions.len(), source.exprs.len());
        assert_eq!(recovery.variables.len(), source.vars.len());
        for index in 0..source.stmts.len() {
            let stmt = StmtId::from_index(index);
            let copy = recovery
                .statement(stmt)
                .unwrap_or_else(|| panic!("{}: s{index} has no copy", case.program.name));
            assert!(copy.index() < recovery.function.stmts.len());
            assert_eq!(recovery.statements[index], Some(copy));
        }
        for index in 0..source.vars.len() {
            let variable = VarId::from_index(index);
            assert_eq!(
                recovery.function.vars[recovery.variable(variable).index()].local,
                source.vars[index].local
            );
        }
        // An expression that the rebuild regenerates (a cleanup) has no
        // copy; every other one maps to an expression of the same kind.
        for index in 0..source.exprs.len() {
            if let Some(copy) = recovery.expression(ExprId::from_index(index)) {
                assert_eq!(
                    std::mem::discriminant(&source.exprs[index].node),
                    std::mem::discriminant(&recovery.function.exprs[copy.index()].node)
                );
            }
        }
    }
}

#[test]
fn asking_for_the_copy_of_a_variable_that_the_source_lacks_panics() {
    let source = structured(&programs::straight_line().program);
    let recovery = recovered(&source);
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        recovery.variable(VarId::from_index(99))
    }));
    assert!(outcome.is_err());
    assert_eq!(recovery.statement(StmtId::from_index(999)), None);
    assert_eq!(recovery.expression(ExprId::from_index(999)), None);
}

#[test]
fn a_function_with_a_dangling_handle_is_malformed_and_verify_reports_it() {
    let mut function = structured(&programs::sum_while().program);
    function.body.push(StmtId::from_index(999));
    let error = recover_structure::<Calc, _, _>(&function, Builder::new())
        .expect_err("a body entry past the arena");
    assert!(matches!(error, RecoverError::Malformed(_)), "{error}");
    assert!(error.source().is_none());
    assert!(!verify::<Calc, _>(&function).is_ok());

    let mut function = structured(&programs::sum_while().program);
    function.exprs[0].node = ExprNode::Var(VarId::from_index(99));
    let error = recover_structure::<Calc, _, _>(&function, Builder::new())
        .expect_err("a read of a variable past the table");
    assert!(matches!(error, RecoverError::Malformed(_)), "{error}");
}

#[test]
fn a_builder_that_refuses_the_rebuilt_function_makes_recovery_fail_with_its_error() {
    let program = Program::new(
        "ill_typed",
        &["a"],
        &[],
        vec![if_then(var("a"), vec![]), ret(int(0))],
    );
    let source = program
        .build(Builder::unchecked())
        .unwrap_or_else(|report| panic!("{report}"));
    let error = recover_structure::<Calc, _, _>(&source, Builder::new())
        .expect_err("an integer is not a condition");
    let RecoverError::Builder(report) = &error else {
        panic!("expected the builder's refusal, got {error}");
    };
    assert!(error.to_string().contains(&report.to_string()));
    assert_eq!(
        error
            .source()
            .expect("the builder error is the source")
            .to_string(),
        report.to_string()
    );
}

#[test]
fn recovery_does_not_change_the_function_it_reads() {
    for case in programs::all() {
        let source = structured(&case.program);
        let snapshot = source.clone();
        let _ = recovered(&source);
        assert_eq!(source, snapshot, "{}", case.program.name);
    }
}

#[test]
fn a_compound_condition_survives_recovery_unchanged() {
    // `sub` and `ge` appear in the program only to give the loop a condition
    // with several operations; recovery rebuilds it operand by operand.
    let program = Program::new(
        "compound_condition",
        &["n"],
        &["i"],
        vec![
            set("i", int(0)),
            S::While(
                ge(sub(var("n"), var("i")), int(1)),
                vec![emit(var("i")), set("i", add(var("i"), int(1)))],
            ),
            ret(var("i")),
        ],
    );
    let recovery = recovers_with_the_same_behavior(&program, &[&[0], &[3]]);
    assert_eq!(recovery.for_loops, 1);
}
