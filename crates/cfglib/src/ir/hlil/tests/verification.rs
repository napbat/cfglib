//! Structural and dialect verification.

extern crate alloc;

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::graph::view::DenseId;
use crate::ir::hlil::{Handler, HandlerKind, SwitchArm, VerificationReport, verify};

use super::{ExprId, Imp, Kit, Prim, StmtId, Sym, ToyHlilBuilder, Ty, VarId};

over_toys!(
    a_well_formed_function_verifies,
    shared_expressions_and_orphans_are_rejected,
    transfers_need_matching_context,
    labels_resolve_by_name,
    children_must_precede_their_parents,
    assignment_targets_are_never_constants,
    switch_arms_need_case_values,
    handler_bindings_and_parameters_must_be_declared,
);

/// The report of a function that must be rejected.
fn rejection<D: Kit>(builder: ToyHlilBuilder<D>, body: &[StmtId]) -> VerificationReport {
    builder
        .finish(body)
        .expect_err("the function breaks an invariant")
}

fn a_well_formed_function_verifies<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let counter = builder.int_variable(0);
    builder.set_signature(&[counter], &[D::ty(Ty::Integer)]);
    let read = builder.read(counter);
    let ten = builder.number(10);
    let compare = builder.call(Prim::LessThan, &[read, ten], Ty::Boolean);
    let one = builder.number(1);
    let current = builder.read(counter);
    let sum = builder.call(Prim::Add, &[current, one], Ty::Integer);
    let bump = builder.store(counter, sum);
    let escape = builder.break_(None);
    let looped = builder.while_(compare, &[bump, escape]);
    let result = builder.read(counter);
    let ret = builder.return_(&[result]);
    let function = builder.finish(&[looped, ret]).unwrap();

    let report = verify::<D, _>(&function);

    assert!(report.is_ok());
    assert_eq!(report.issue_count(), 0);
}

fn shared_expressions_and_orphans_are_rejected<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let variable = builder.int_variable(0);
    let read = builder.read(variable);
    builder.return_(&[read]);
    let second = builder.return_(&[read]);

    let report = rejection(builder, &[second]);

    let message = report.to_string();
    assert!(
        message.starts_with("HLIL verification failed; "),
        "{message}"
    );
    assert!(
        message.contains("expression e0 is referenced 2 times"),
        "{message}"
    );
    assert!(
        message.contains("statement s0 is referenced 0 times"),
        "{message}"
    );
    assert_eq!(report.issue_count(), 2);
}

fn transfers_need_matching_context<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let stray_break = builder.break_(None);
    let stray_goto = builder.goto("missing");
    let stray_continue = builder.continue_(None);
    // A switch is breakable but not a loop.
    let inner_continue = builder.continue_(None);
    let scrutinee = builder.number(0);
    let one = D::constant(1);
    let switch = builder.switch(
        scrutinee,
        &[SwitchArm {
            values: &[one],
            body: &[inner_continue],
        }],
        &[],
    );

    let report = rejection(builder, &[stray_break, stray_goto, stray_continue, switch]);

    let message = report.to_string();
    assert!(
        message.contains("s0 breaks outside any loop or switch"),
        "{message}"
    );
    assert!(
        message.contains("s1 targets undefined label missing"),
        "{message}"
    );
    assert!(
        message.contains("s2 continues outside any loop"),
        "{message}"
    );
    assert!(
        message.contains("s3 continues outside any loop"),
        "{message}"
    );
}

fn labels_resolve_by_name<D: Kit>() {
    // `goto later; outer: { break outer; } later: { return; }`: a forward
    // goto and a labeled break both find their label.
    let mut builder = ToyHlilBuilder::<D>::new();
    let forward = builder.goto("later");
    let leave = builder.break_(Some("outer"));
    let outer = builder.labeled("outer", &[leave]);
    let exit = builder.return_(&[]);
    let later = builder.labeled("later", &[exit]);
    let function = builder.finish(&[forward, outer, later]).unwrap();
    assert!(verify::<D, _>(&function).is_ok());

    let mut builder = ToyHlilBuilder::<D>::new();
    let escape = builder.break_(Some("elsewhere"));
    let repeat = builder.continue_(Some("elsewhere"));
    let first = builder.labeled("twice", &[escape]);
    let second = builder.labeled("twice", &[repeat]);

    let report = rejection(builder, &[first, second]);

    let message = report.to_string();
    assert!(
        message.contains("label twice is defined 2 times"),
        "{message}"
    );
    assert!(
        message.contains("s0 breaks unenclosing label elsewhere"),
        "{message}"
    );
    assert!(
        message.contains("s1 continues unenclosing label elsewhere"),
        "{message}"
    );
}

fn children_must_precede_their_parents<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    // e0 applies an operation to e1, which does not exist yet.
    let early = builder.call(Prim::Add, &[ExprId::from_index(1)], Ty::Integer);
    let late = builder.number(1);
    let first = builder.expression_statement(early);
    let second = builder.expression_statement(late);
    // s2 lists s3, which does not exist yet.
    let condition = builder.condition();
    let conditional = builder.if_(condition, &[StmtId::from_index(3)], &[]);
    builder.noop();

    let report = rejection(builder, &[first, second, conditional]);

    let message = report.to_string();
    assert!(
        message.contains("e0 operand e1 does not precede its parent"),
        "{message}"
    );
    assert!(
        message.contains("s2 child s3 does not precede its parent"),
        "{message}"
    );
}

fn assignment_targets_are_never_constants<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let target = builder.number(1);
    let value = builder.number(2);
    let assign = builder.assign(target, value);

    let report = rejection(builder, &[assign]);

    assert!(
        report.to_string().contains("s0 assigns into constant e0"),
        "{report}"
    );
}

fn switch_arms_need_case_values<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let scrutinee = builder.number(0);
    let switch = builder.switch(
        scrutinee,
        &[SwitchArm {
            values: &[],
            body: &[],
        }],
        &[],
    );

    let report = rejection(builder, &[switch]);

    assert!(
        report
            .to_string()
            .contains("s0 has an arm with no case values"),
        "{report}"
    );
}

fn handler_bindings_and_parameters_must_be_declared<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let declared = builder.int_variable(0);
    builder.set_signature(&[declared, declared, VarId::from_index(9)], &[]);
    let handler = Handler {
        kind: HandlerKind::Catch,
        binding: Some(VarId::from_index(4)),
        caught_types: &[D::ty(Ty::Integer)],
        body: &[],
    };
    let guarded = builder.try_(&[], &[handler], &[]);

    let report = rejection(builder, &[guarded]);

    let message = report.to_string();
    assert!(
        message.contains("signature repeats parameter v0"),
        "{message}"
    );
    assert!(
        message.contains("signature names undeclared parameter v9"),
        "{message}"
    );
    assert!(
        message.contains("s0 binds undeclared variable v4"),
        "{message}"
    );
}

/// Names handles that the function does not have.
fn dangling_function<D: Kit>() -> ToyHlilBuilder<D> {
    let mut builder = ToyHlilBuilder::<D>::new();
    builder.read_variable(VarId::from_index(3), &D::ty(Ty::Integer));
    builder.call(Prim::Add, &[ExprId::from_index(99)], Ty::Integer);
    builder.expression_statement(ExprId::from_index(98));
    builder.if_(ExprId::from_index(0), &[StmtId::from_index(77)], &[]);
    builder
}

#[test]
fn dangling_handles_are_reported_and_the_dialect_is_not_asked() {
    // `Imp` reads the type of every operand of a comparison, which would panic
    // on a handle outside the function.
    let mut builder = dangling_function::<Imp>();
    builder.call(Prim::LessThan, &[ExprId::from_index(97)], Ty::Boolean);

    let report = rejection(builder, &[StmtId::from_index(55)]);

    let message = report.to_string();
    assert!(
        message.contains("e0 reads undeclared variable v3"),
        "{message}"
    );
    assert!(
        message.contains("e1 names missing operand e99"),
        "{message}"
    );
    assert!(
        message.contains("s0 names missing expression e98"),
        "{message}"
    );
    assert!(
        message.contains("s1 names missing statement s77"),
        "{message}"
    );
    assert!(
        message.contains("body names missing statement s55"),
        "{message}"
    );
    assert!(!message.contains("compares"), "{message}");
}

#[test]
fn imp_hook_runs_after_the_structure_checks() {
    let mut builder = ToyHlilBuilder::<Imp>::new();
    let flag = builder.condition();
    let seven = builder.number(7);
    let compare = builder.call(Prim::LessThan, &[flag, seven], Ty::Boolean);
    let statement = builder.expression_statement(compare);
    builder.noop();

    let report = rejection(builder, &[statement]);

    let messages: alloc::vec::Vec<_> = report.issues.iter().map(|issue| &*issue.message).collect();
    assert_eq!(messages.len(), 2, "{messages:?}");
    assert!(messages[0].starts_with("statement s1 is referenced 0 times"));
    assert_eq!(messages[1], "e2 compares the non-integer operand e0");
}

#[test]
fn sym_hook_reports_empty_constants() {
    let mut builder = ToyHlilBuilder::<Sym>::new();
    let empty = builder.constant(&alloc::string::String::new(), &());
    let statement = builder.expression_statement(empty);

    let report = rejection(builder, &[statement]);

    let messages: Vec<&str> = report.issues.iter().map(|issue| &*issue.message).collect();
    assert_eq!(messages, ["e0 is an empty constant"]);
}
