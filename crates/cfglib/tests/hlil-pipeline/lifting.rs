//! `lift`, `lift_with_tree`, and `lift_trampolines` over the control tree:
//! what the structurer's report says, how loop conditions come out, and the
//! hooks that a bytecode instruction uses to steer inlining.

use cfglib::ControlTree;
use cfglib::GotoReason;
use cfglib::ir::hlil::{LiftMetadata, lift_trampolines, lift_with_tree};

use crate::dialect::{Calc, Lit, Op, Ty};
use crate::flat::{Code, CodeBuilder, Form, Link};
use crate::flat_run::run_flat;
use crate::machine::End;
use crate::pipeline::{lifted, lowered, structured};
use crate::programs;
use crate::structured::{Builder, ExprNode, Function, StmtNode};
use crate::structured_run::run_structured;

/// `i = 0; while (i < n) { emit(i); i = i + 1 } return i`, written as the
/// bytecode of a compare-and-branch. `exit_on_true` tests the opposite
/// relation and sends its true edge out of the loop.
fn fused_loop(exit_on_true: bool) -> Code {
    let mut b = CodeBuilder::new();
    let n = b.register("n", Ty::Int);
    let i = b.register("i", Ty::Int);
    let one = b.register("one", Ty::Int);
    b.parameters(&[n], &[Ty::Int]);
    let (init, header, body, exit) = (b.block(), b.block(), b.block(), b.block());
    b.start_at(init);
    b.push(init, Form::Const(Lit::Int(0)), &[], &[i]);
    b.link(init, header, Link::Next);
    if exit_on_true {
        b.push(header, Form::BranchOn(Op::Ge), &[i, n], &[]);
        b.link(header, exit, Link::Yes);
        b.link(header, body, Link::No);
    } else {
        b.push(header, Form::BranchOn(Op::Lt), &[i, n], &[]);
        b.link(header, body, Link::Yes);
        b.link(header, exit, Link::No);
    }
    b.push(body, Form::Apply(Op::Emit), &[i], &[]);
    b.push(body, Form::Const(Lit::Int(1)), &[], &[one]);
    b.push(body, Form::Apply(Op::Add), &[i, one], &[i]);
    b.link(body, header, Link::Goto);
    b.push(exit, Form::Return, &[i], &[]);
    b.seal()
}

fn first_while_condition(function: &Function) -> Option<(Op, usize)> {
    let condition = function.stmts.iter().find_map(|stmt| match stmt {
        StmtNode::While { condition, .. } => Some(*condition),
        _ => None,
    })?;
    match &function.exprs[condition.index()].node {
        ExprNode::Op { op, operands } => Some((*op, operands.len())),
        _ => None,
    }
}

#[test]
fn a_fused_compare_and_branch_lifts_to_a_while_over_the_comparison() {
    let code = fused_loop(false);
    let lifted = lifted(&code);
    assert!(lifted.report.is_fully_structured());
    assert_eq!(first_while_condition(&lifted.function), Some((Op::Lt, 2)));
}

#[test]
fn a_loop_that_exits_on_true_gets_the_inverted_comparison_not_a_negation() {
    // `exit when i >= n` iterates while `i < n`: the dialect's exact
    // inversion rewrites the operation, so no `not` wraps the test.
    let code = fused_loop(true);
    let lifted = lifted(&code);
    assert!(lifted.report.is_fully_structured());
    assert_eq!(first_while_condition(&lifted.function), Some((Op::Lt, 2)));
}

#[test]
fn fused_loops_keep_their_behavior_through_the_lift() {
    for exit_on_true in [false, true] {
        let code = fused_loop(exit_on_true);
        let function = lifted(&code).function;
        for n in [0, 1, 4] {
            let expected = run_flat(&code, &[n]);
            assert!(matches!(expected.end, End::Returned(_)));
            assert_eq!(run_structured(&function, &[n]), expected, "n = {n}");
        }
    }
}

/// `i = 0; loop { if (i < n) {...} }` with a `break` that routes through a
/// block holding only a no-op, the way a compiler routes it through a jump.
fn loop_with_a_break_trampoline() -> Code {
    let mut b = CodeBuilder::new();
    let n = b.register("n", Ty::Int);
    let i = b.register("i", Ty::Int);
    let one = b.register("one", Ty::Int);
    let three = b.register("three", Ty::Int);
    let more = b.register("more", Ty::Bool);
    let stop = b.register("stop", Ty::Bool);
    b.parameters(&[n], &[Ty::Int]);
    let (init, header, body, hop, latch, exit) = (
        b.block(),
        b.block(),
        b.block(),
        b.block(),
        b.block(),
        b.block(),
    );
    b.start_at(init);
    b.push(init, Form::Const(Lit::Int(0)), &[], &[i]);
    b.link(init, header, Link::Next);
    b.push(header, Form::Apply(Op::Lt), &[i, n], &[more]);
    b.push(header, Form::Branch, &[more], &[]);
    b.link(header, body, Link::Yes);
    b.link(header, exit, Link::No);
    b.push(body, Form::Apply(Op::Emit), &[i], &[]);
    b.push(body, Form::Const(Lit::Int(3)), &[], &[three]);
    b.push(body, Form::Apply(Op::Eq), &[i, three], &[stop]);
    b.push(body, Form::Branch, &[stop], &[]);
    b.link(body, hop, Link::Yes);
    b.link(body, latch, Link::No);
    b.push(hop, Form::Nop, &[], &[]);
    b.link(hop, exit, Link::Goto);
    b.push(latch, Form::Const(Lit::Int(1)), &[], &[one]);
    b.push(latch, Form::Apply(Op::Add), &[i, one], &[i]);
    b.link(latch, header, Link::Goto);
    b.push(exit, Form::Return, &[i], &[]);
    b.seal()
}

#[test]
fn a_block_that_only_transfers_is_a_trampoline_and_the_lift_structures_through_it() {
    let code = loop_with_a_break_trampoline();
    let trampolines = lift_trampolines::<Calc, _>(&code);
    let hop = code
        .cfg
        .block_ids()
        .find(|&block| {
            code.cfg.block(block).instructions().len() == 1
                && code.cfg.block(block).instructions()[0].form == Form::Nop
        })
        .expect("the bytecode has a hop");
    assert_eq!(trampolines, [hop], "no other block only transfers");

    let lifted = lifted(&code);
    assert!(lifted.report.is_fully_structured(), "{:?}", lifted.report);
    for n in [0, 2, 3, 9] {
        let expected = run_flat(&code, &[n]);
        assert_eq!(run_structured(&lifted.function, &[n]), expected, "n = {n}");
    }
}

#[test]
fn lowered_bytecode_has_no_trampoline_because_lowering_emits_no_transfer_instruction() {
    for case in programs::all() {
        let code = lowered(&structured(&case.program)).function;
        assert!(
            lift_trampolines::<Calc, _>(&code).is_empty(),
            "{}",
            case.program.name
        );
    }
}

#[test]
fn lifting_with_the_tree_of_the_trampoline_policy_is_lifting() {
    let mut codes: Vec<(String, Code)> =
        vec![("trampoline loop".to_owned(), loop_with_a_break_trampoline())];
    for case in programs::all() {
        codes.push((
            case.program.name.to_owned(),
            lowered(&structured(&case.program)).function,
        ));
    }
    for (name, code) in codes {
        let direct = lifted(&code);
        let trampolines = lift_trampolines::<Calc, _>(&code);
        let (tree, report) =
            ControlTree::compute_with_report_treating_empty(&code.cfg, &trampolines);
        let through = lift_with_tree::<Calc, _, _>(
            &code,
            &tree,
            &report,
            Builder::new(),
            LiftMetadata::Preserve,
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(through.function, direct.function, "{name}");
        assert_eq!(through.report, direct.report, "{name}");
        assert_eq!(through.correspondence, direct.correspondence, "{name}");
    }
}

#[test]
fn the_report_of_a_lift_is_the_report_of_its_control_tree() {
    for case in programs::all() {
        let code = lowered(&structured(&case.program)).function;
        let lifted = lifted(&code);
        let trampolines = lift_trampolines::<Calc, _>(&code);
        let (_, report) = ControlTree::compute_with_report_treating_empty(&code.cfg, &trampolines);
        assert_eq!(lifted.report, report, "{}", case.program.name);
    }
}

/// The programs whose lift is known to need a `goto`: a jump out of a
/// switch arm toward a join that a sibling arm's `continue` bypasses, the
/// shared continuation of a `try` and its `finally`, and a loop with two
/// entries.
const NEEDS_A_GOTO: [&str; 3] = ["switch_in_loop", "catch_and_finally", "irreducible_loop"];

#[test]
fn reducible_programs_lift_without_a_goto() {
    for case in programs::all() {
        if NEEDS_A_GOTO.contains(&case.program.name) {
            continue;
        }
        let code = lowered(&structured(&case.program)).function;
        let report = lifted(&code).report;
        assert!(
            report.is_fully_structured(),
            "{}: {report:?}",
            case.program.name
        );
    }
}

#[test]
fn an_irreducible_loop_lifts_with_gotos_that_revisit_code_and_still_verifies() {
    let source = structured(&programs::irreducible_loop().program);
    let code = lowered(&source).function;
    let lifted = lifted(&code);
    assert!(!lifted.report.is_fully_structured());
    assert!(
        lifted
            .report
            .gotos
            .iter()
            .any(|goto| goto.reason == GotoReason::RevisitedTarget),
        "{:?}",
        lifted.report
    );
    let gotos = lifted
        .function
        .stmts
        .iter()
        .filter(|stmt| matches!(stmt, StmtNode::Goto(_)))
        .count();
    let labels = lifted
        .function
        .stmts
        .iter()
        .filter(|stmt| matches!(stmt, StmtNode::Labeled { .. }))
        .count();
    assert_eq!(
        gotos,
        lifted.report.gotos.len(),
        "one goto statement per reported goto"
    );
    assert!(labels >= 1, "a goto needs its label");
    assert_eq!(lifted.report.unresolved_labels, []);
    assert!(cfglib::ir::hlil::verify::<Calc, _>(&lifted.function).is_ok());
}

#[test]
fn switch_cases_that_share_an_arm_lift_to_one_arm_with_all_their_values() {
    let code = lowered(&structured(&programs::switch_dispatch().program)).function;
    let function = lifted(&code).function;
    let arms: Vec<Vec<Lit>> = function
        .stmts
        .iter()
        .find_map(|stmt| match stmt {
            StmtNode::Switch { arms, .. } => {
                Some(arms.iter().map(|arm| arm.values.clone()).collect())
            }
            _ => None,
        })
        .expect("the program has a switch");
    assert_eq!(arms, [vec![Lit::Int(1), Lit::Int(2)], vec![Lit::Int(3)]]);
}

#[test]
fn lifted_handlers_keep_the_kind_of_the_region_handlers() {
    use crate::structured::CatchKind;
    let kinds = |case: programs::Case| {
        let code = lowered(&structured(&case.program)).function;
        let function = lifted(&code).function;
        function
            .stmts
            .iter()
            .filter_map(|stmt| match stmt {
                StmtNode::Try {
                    handlers,
                    finally_body,
                    ..
                } => Some((
                    handlers
                        .iter()
                        .map(|h| std::mem::discriminant(&h.kind))
                        .collect::<Vec<_>>(),
                    !finally_body.is_empty(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let catch_all = std::mem::discriminant(&CatchKind::CatchAll);
    let catch = std::mem::discriminant(&CatchKind::Catch);
    assert_eq!(
        kinds(programs::division_guard()),
        [(vec![catch_all], false)]
    );
    assert_eq!(kinds(programs::typed_catch()), [(vec![catch], false)]);
    assert_eq!(
        kinds(programs::catch_and_finally()),
        [(vec![catch_all], true)]
    );
    assert_eq!(
        kinds(programs::nested_try()).len(),
        2,
        "two regions lift to two trys"
    );
}

#[test]
fn the_lift_keeps_the_signature_and_the_variable_metadata() {
    for case in programs::all() {
        let code = lowered(&structured(&case.program)).function;
        let lifted = lifted(&code);
        let parameters: Vec<_> = code
            .parameters
            .iter()
            .map(|parameter| lifted.correspondence.variables[parameter.index()])
            .collect();
        assert_eq!(
            lifted.function.parameters, parameters,
            "{}",
            case.program.name
        );
        assert_eq!(
            lifted.function.returns, code.returns,
            "{}",
            case.program.name
        );
        assert_eq!(lifted.function.vars.len(), code.locals.len());
    }
}

#[test]
fn lifting_is_deterministic() {
    for case in programs::all() {
        let code = lowered(&structured(&case.program)).function;
        let (first, second) = (lifted(&code), lifted(&code));
        assert_eq!(first.function, second.function, "{}", case.program.name);
        assert_eq!(first.correspondence, second.correspondence);
        assert_eq!(first.report, second.report);
    }
}

#[test]
fn a_dead_definition_that_can_throw_stays_a_statement() {
    // `d = x / y` is never read, but it throws for `y == 0`.
    let mut b = CodeBuilder::new();
    let x = b.register("x", Ty::Int);
    let y = b.register("y", Ty::Int);
    let d = b.register("d", Ty::Int);
    b.parameters(&[x, y], &[Ty::Int]);
    let entry = b.block();
    b.start_at(entry);
    b.push(entry, Form::Apply(Op::Div), &[x, y], &[d]);
    b.push(entry, Form::Return, &[x], &[]);
    let code = b.seal();
    let function = lifted(&code).function;
    let divisions = function
        .exprs
        .iter()
        .filter(|expr| matches!(&expr.node, ExprNode::Op { op: Op::Div, .. }))
        .count();
    assert_eq!(divisions, 1);
    for input in [[6, 3], [6, 0]] {
        assert_eq!(run_structured(&function, &input), run_flat(&code, &input));
    }
    assert!(matches!(run_flat(&code, &[6, 0]).end, End::Uncaught(-1)));
}

/// `t = 5; d = t + x; return d` with the hook of the first definition and
/// the second instruction set as `edit` says.
fn pair(edit: impl Fn(&mut CodeBuilder, cfglib::BlockId)) -> Code {
    let mut b = CodeBuilder::new();
    let x = b.register("x", Ty::Int);
    let t = b.register("t", Ty::Int);
    let d = b.register("d", Ty::Int);
    b.parameters(&[x], &[Ty::Int]);
    let entry = b.block();
    b.start_at(entry);
    b.push(entry, Form::Const(Lit::Int(5)), &[], &[t]);
    edit(&mut b, entry);
    b.push(entry, Form::Apply(Op::Add), &[t, x], &[d]);
    edit(&mut b, entry);
    b.push(entry, Form::Return, &[d], &[]);
    b.seal()
}

fn assignments(function: &Function) -> usize {
    function
        .stmts
        .iter()
        .filter(|stmt| matches!(stmt, StmtNode::Assign { .. }))
        .count()
}

#[test]
fn a_definition_marked_to_materialize_is_not_inlined_into_its_consumer() {
    let plain = lifted(&pair(|_, _| {})).function;
    assert_eq!(
        assignments(&plain),
        0,
        "both definitions fold into the return"
    );
    let marked = pair(|builder, block| {
        builder.amend_last(block, |instruction| instruction.materialize = true);
    });
    let function = lifted(&marked).function;
    assert_eq!(assignments(&function), 2);
    for input in [[0], [4]] {
        assert_eq!(run_structured(&function, &input), run_flat(&marked, &input));
    }
}

#[test]
fn an_instruction_that_reads_its_previous_value_blocks_inlining_around_it() {
    // The consumer reads its destination's previous value at use 1, so the
    // definition that feeds position 1 stays visible, and the consumer's own
    // result never folds forward into the return.
    let mut b = CodeBuilder::new();
    let x = b.register("x", Ty::Int);
    let seed = b.register("seed", Ty::Int);
    let d = b.register("d", Ty::Int);
    b.parameters(&[x], &[Ty::Int]);
    let entry = b.block();
    b.start_at(entry);
    b.push(entry, Form::Const(Lit::Int(5)), &[], &[seed]);
    b.push(entry, Form::Apply(Op::Add), &[x, seed], &[d]);
    b.amend_last(entry, |instruction| instruction.previous_value = Some(1));
    b.push(entry, Form::Return, &[d], &[]);
    let code = b.seal();
    let function = lifted(&code).function;
    assert_eq!(
        assignments(&function),
        2,
        "the seed and the merge stay statements"
    );
    for input in [[0], [7]] {
        assert_eq!(run_structured(&function, &input), run_flat(&code, &input));
    }
}
