//! `verify`: it accepts what the pipeline produces, and each documented
//! structural check rejects the function that a buggy frontend would build
//! by corrupting a sealed one.

use cfglib::ir::hlil::{VerificationIssue, VerificationReport, verify};

use crate::dialect::{Calc, Lit};
use crate::pipeline::{lifted, lowered, recovered, structured};
use crate::programs;
use crate::source::{E, Program, S, add, emit, int, lt, ret, set, var};
use crate::structured::{
    Arm, Builder, Catch, CatchKind, ExprId, ExprNode, Function, StmtId, StmtNode, VarId,
};

fn report(function: &Function) -> VerificationReport {
    verify::<Calc, _>(function)
}

/// The sealed function that every corruption starts from: an assignment, an
/// `if` with a body, and a return.
fn good() -> Function {
    structured(&Program::new(
        "good",
        &["a"],
        &["r"],
        vec![
            set("r", add(var("a"), int(1))),
            S::If(lt(var("r"), int(5)), vec![emit(var("r"))], vec![]),
            ret(var("r")),
        ],
    ))
}

fn stmt_where(function: &Function, wanted: impl Fn(&StmtNode) -> bool) -> StmtId {
    let index = function
        .stmts
        .iter()
        .position(wanted)
        .expect("the function has such a statement");
    StmtId::from_index(index)
}

fn expr_where(function: &Function, wanted: impl Fn(&ExprNode) -> bool) -> ExprId {
    let index = function
        .exprs
        .iter()
        .position(|expr| wanted(&expr.node))
        .expect("the function has such an expression");
    ExprId::from_index(index)
}

/// Appends a statement to the arena and lists it first in the body.
fn prepend(function: &mut Function, node: StmtNode) -> StmtId {
    function.stmts.push(node);
    let id = StmtId::from_index(function.stmts.len() - 1);
    function.body.insert(0, id);
    id
}

fn assert_rejected(function: &Function, what: &str) -> VerificationReport {
    let report = report(function);
    assert!(!report.is_ok(), "{what} was accepted");
    assert_eq!(report.issue_count(), report.issues.len());
    report
}

#[test]
fn the_untouched_function_verifies() {
    assert!(report(&good()).is_ok());
}

#[test]
fn everything_the_pipeline_produces_verifies() {
    for case in programs::all() {
        let source = structured(&case.program);
        let lifted = lifted(&lowered(&source).function).function;
        let recovered = recovered(&lifted).function;
        for (stage, function) in [
            ("source", &source),
            ("lifted", &lifted),
            ("recovered", &recovered),
        ] {
            let report = report(function);
            assert!(report.is_ok(), "{} {stage}: {report}", case.program.name);
        }
    }
}

#[test]
fn a_signature_that_names_an_undeclared_parameter_is_rejected() {
    let mut function = good();
    function.parameters.push(VarId::from_index(99));
    assert_rejected(&function, "an undeclared parameter");
}

#[test]
fn a_signature_that_repeats_a_parameter_is_rejected() {
    let mut function = good();
    function.parameters.push(function.parameters[0]);
    assert_rejected(&function, "a repeated parameter");
}

#[test]
fn a_read_of_an_undeclared_variable_is_rejected() {
    let mut function = good();
    let read = expr_where(&function, |node| matches!(node, ExprNode::Var(_)));
    function.exprs[read.index()].node = ExprNode::Var(VarId::from_index(99));
    assert_rejected(&function, "a read of an undeclared variable");
}

#[test]
fn an_operand_that_does_not_precede_its_operation_is_rejected() {
    let mut function = good();
    let operation = expr_where(&function, |node| matches!(node, ExprNode::Op { .. }));
    let ExprNode::Op { operands, .. } = &mut function.exprs[operation.index()].node else {
        unreachable!("the expression was found as an operation");
    };
    operands[0] = operation;
    assert_rejected(&function, "an operation that is its own operand");
}

#[test]
fn an_operand_that_names_no_expression_is_rejected() {
    let mut function = good();
    let operation = expr_where(&function, |node| matches!(node, ExprNode::Op { .. }));
    let ExprNode::Op { operands, .. } = &mut function.exprs[operation.index()].node else {
        unreachable!("the expression was found as an operation");
    };
    operands[0] = ExprId::from_index(999);
    assert_rejected(&function, "an operand past the arena");
}

#[test]
fn a_child_statement_that_does_not_precede_its_parent_is_rejected() {
    let mut function = good();
    let branch = stmt_where(&function, |node| matches!(node, StmtNode::If { .. }));
    let StmtNode::If { then_body, .. } = &mut function.stmts[branch.index()] else {
        unreachable!("the statement was found as an if");
    };
    then_body.push(branch);
    assert_rejected(&function, "an if inside itself");
}

#[test]
fn a_child_statement_that_names_no_statement_is_rejected() {
    let mut function = good();
    let branch = stmt_where(&function, |node| matches!(node, StmtNode::If { .. }));
    let StmtNode::If { then_body, .. } = &mut function.stmts[branch.index()] else {
        unreachable!("the statement was found as an if");
    };
    then_body.push(StmtId::from_index(999));
    assert_rejected(&function, "a body entry past the arena");
}

#[test]
fn an_expression_with_two_parents_is_rejected() {
    let mut function = good();
    let result = stmt_where(&function, |node| matches!(node, StmtNode::Return(_)));
    let shared = expr_where(&function, |node| matches!(node, ExprNode::Op { .. }));
    let StmtNode::Return(values) = &mut function.stmts[result.index()] else {
        unreachable!("the statement was found as a return");
    };
    values.push(shared);
    assert_rejected(&function, "an expression used twice");
}

#[test]
fn an_expression_with_no_parent_is_rejected() {
    let mut function = good();
    function.exprs.push(crate::structured::Expr {
        node: ExprNode::Const(Lit::Int(1)),
        ty: crate::dialect::Ty::Int,
    });
    assert_rejected(&function, "an unused expression");
}

#[test]
fn a_statement_listed_by_two_parents_is_rejected() {
    let mut function = good();
    function.body.push(function.body[0]);
    assert_rejected(&function, "a statement listed twice");
}

#[test]
fn a_statement_listed_by_no_parent_is_rejected() {
    let mut function = good();
    function.body.pop();
    assert_rejected(&function, "an unlisted statement");
}

#[test]
fn an_assignment_into_a_constant_is_rejected() {
    let mut function = good();
    let assignment = stmt_where(&function, |node| matches!(node, StmtNode::Assign { .. }));
    let StmtNode::Assign { target, .. } = function.stmts[assignment.index()] else {
        unreachable!("the statement was found as an assignment");
    };
    function.exprs[target.index()].node = ExprNode::Const(Lit::Int(0));
    assert_rejected(&function, "an assignment into a constant");
}

#[test]
fn a_switch_arm_without_case_values_is_rejected() {
    let function = structured(&programs::switch_dispatch().program);
    assert!(report(&function).is_ok());
    let mut corrupt = function.clone();
    let dispatch = stmt_where(&corrupt, |node| matches!(node, StmtNode::Switch { .. }));
    let StmtNode::Switch { arms, .. } = &mut corrupt.stmts[dispatch.index()] else {
        unreachable!("the statement was found as a switch");
    };
    let Arm { values, .. } = &mut arms[0];
    values.clear();
    assert_rejected(&corrupt, "an arm with no values");
}

#[test]
fn a_handler_that_binds_an_undeclared_variable_is_rejected() {
    let function = structured(&programs::division_guard().program);
    assert!(report(&function).is_ok());
    let mut corrupt = function.clone();
    let guarded = stmt_where(&corrupt, |node| matches!(node, StmtNode::Try { .. }));
    let StmtNode::Try { handlers, .. } = &mut corrupt.stmts[guarded.index()] else {
        unreachable!("the statement was found as a try");
    };
    let Catch { binding, kind, .. } = &mut handlers[0];
    assert!(matches!(kind, CatchKind::CatchAll));
    *binding = Some(VarId::from_index(99));
    assert_rejected(&corrupt, "a binding of an undeclared variable");
}

#[test]
fn a_transfer_that_has_no_enclosing_target_is_rejected() {
    for stray in [
        StmtNode::Break(None),
        StmtNode::Continue(None),
        StmtNode::Break(Some("nowhere".to_owned())),
        StmtNode::Continue(Some("nowhere".to_owned())),
        StmtNode::Goto("nowhere".to_owned()),
    ] {
        let mut function = good();
        prepend(&mut function, stray.clone());
        assert_rejected(&function, &format!("{stray:?}"));
    }
}

#[test]
fn a_label_that_two_statements_define_is_rejected_and_named() {
    let mut function = good();
    for _ in 0..2 {
        prepend(
            &mut function,
            StmtNode::Labeled {
                label: "twice".to_owned(),
                body: vec![],
            },
        );
    }
    let report = assert_rejected(&function, "a label defined twice");
    assert!(
        report
            .issues
            .iter()
            .any(|issue| issue.message.contains("twice"))
    );
}

#[test]
fn the_enclosing_targets_that_the_shapes_allow_are_accepted() {
    // A `break` leaves a switch that no loop encloses, a `continue` inside a
    // switch reaches its loop, a labeled `break` leaves a labeled block, and
    // a `goto` finds a label that comes later.
    let program = Program::new(
        "targets",
        &["x"],
        &["i"],
        vec![
            S::Switch(var("x"), vec![(vec![1], vec![S::Break(None)])], vec![]),
            S::While(
                lt(var("i"), int(3)),
                vec![
                    set("i", add(var("i"), int(1))),
                    S::Switch(var("i"), vec![(vec![1], vec![S::Continue(None)])], vec![]),
                ],
            ),
            S::Label("block", vec![S::Break(Some("block"))]),
            S::Goto("later"),
            S::Label("later", vec![emit(var("i"))]),
            ret(var("i")),
        ],
    );
    let function = program
        .build(Builder::new())
        .unwrap_or_else(|rejected| panic!("{rejected}"));
    assert!(report(&function).is_ok());
}

#[test]
fn a_continue_that_names_a_labeled_statement_that_is_not_a_loop_is_rejected() {
    // `continue` continues the enclosing labeled loop that its label names.
    // A labeled block is not a loop, so the function has no meaning here.
    let program = Program::new(
        "continue_to_block",
        &[],
        &[],
        vec![
            S::Label("block", vec![S::Continue(Some("block"))]),
            ret(int(0)),
        ],
    );
    let function = program
        .build(Builder::unchecked())
        .unwrap_or_else(|rejected| panic!("{rejected}"));
    assert_rejected(&function, "a continue that names a block");
}

#[test]
fn a_dialect_that_finds_a_type_error_makes_a_structurally_sound_function_fail() {
    let program = Program::new(
        "ill_typed",
        &["a"],
        &[],
        vec![ret(add(var("a"), E::Lit(Lit::Bool(true))))],
    );
    let rejected = program
        .build(Builder::new())
        .expect_err("the checked builder verifies");
    assert_eq!(rejected.issue_count(), 1);
    let function = program
        .build(Builder::unchecked())
        .unwrap_or_else(|rejected| panic!("{rejected}"));
    assert_eq!(report(&function), rejected);
}

#[test]
fn the_dialect_hook_does_not_run_when_a_handle_is_out_of_range() {
    let program = Program::new(
        "ill_typed",
        &["a"],
        &[],
        vec![ret(add(var("a"), E::Lit(Lit::Bool(true))))],
    );
    let mut function = program
        .build(Builder::unchecked())
        .unwrap_or_else(|rejected| panic!("{rejected}"));
    let typed_only = report(&function);
    assert!(
        typed_only
            .issues
            .iter()
            .any(|issue| issue.message.contains("applies"))
    );
    function.parameters.push(VarId::from_index(99));
    let dangling = report(&function);
    assert!(!dangling.is_ok());
    assert!(
        dangling
            .issues
            .iter()
            .all(|issue| !issue.message.contains("applies")),
        "the hook ran over a function with a dangling handle"
    );
}

#[test]
fn the_context_checks_wait_until_the_flat_checks_find_nothing() {
    let mut function = good();
    function.exprs.push(crate::structured::Expr {
        node: ExprNode::Const(Lit::Int(1)),
        ty: crate::dialect::Ty::Int,
    });
    let before = report(&function).issue_count();
    assert_eq!(before, 1, "one unused expression");
    prepend(&mut function, StmtNode::Break(None));
    assert_eq!(
        report(&function).issue_count(),
        before,
        "the stray break is not examined yet"
    );
    function.exprs.pop();
    assert_eq!(
        report(&function).issue_count(),
        1,
        "now the break is the only failure"
    );
}

#[test]
fn a_report_lists_every_independent_failure_and_is_deterministic() {
    let mut function = good();
    for _ in 0..2 {
        function.parameters.push(VarId::from_index(99));
    }
    let read = expr_where(&function, |node| matches!(node, ExprNode::Var(_)));
    function.exprs[read.index()].node = ExprNode::Var(VarId::from_index(98));
    let first = report(&function);
    assert!(first.issue_count() >= 3, "{first}");
    assert_eq!(first, report(&function));
}

#[test]
fn a_report_displays_every_issue_and_is_an_error() {
    let issues = vec![
        VerificationIssue::new("first problem"),
        VerificationIssue::new("second problem"),
    ];
    let report = VerificationReport::new(issues.clone());
    let text = report.to_string();
    for issue in &issues {
        assert!(text.contains(&issue.message), "{text}");
        assert_eq!(issue.to_string(), issue.message);
    }
    let error: &dyn std::error::Error = &report;
    assert_eq!(error.to_string(), text);
    assert!(VerificationReport::new(Vec::new()).is_ok());
}
