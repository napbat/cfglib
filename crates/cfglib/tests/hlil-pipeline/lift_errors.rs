//! `lift` on bytecode that no HLIL form can state, and on a builder that
//! refuses what it is given. Each malformed function has a sibling that
//! differs by the one malformation and lifts, so a refusal is the
//! malformation's doing.

use std::error::Error;
use std::panic::{AssertUnwindSafe, catch_unwind};

use cfglib::ControlTree;
use cfglib::ir::hlil::{LiftError, LiftMetadata, VerificationReport, lift, lift_with_tree};

use crate::dialect::{Calc, Lit, Op, Ty};
use crate::flat::{Code, CodeBuilder, Form, Link, Reg};
use crate::flat_run::run_flat;
use crate::machine::{End, Value};
use crate::structured::Builder;
use crate::structured_run::run_structured;

type Outcome = Result<crate::pipeline::Lifted, LiftError<VerificationReport>>;

fn try_lift(code: &Code) -> Outcome {
    lift::<Calc, _, _>(code, Builder::new(), LiftMetadata::Preserve)
}

fn assert_unsupported(outcome: Outcome, what: &str) {
    match outcome {
        Err(LiftError::Unsupported(reason)) => {
            assert!(!reason.is_empty(), "{what} gives no reason");
        }
        Err(other) => panic!("{what}: expected Unsupported, got {other}"),
        Ok(_) => panic!("{what}: lifted"),
    }
}

/// A function of one integer `x` and one boolean `flag` with an entry block.
struct Fixture {
    builder: CodeBuilder,
    x: Reg,
    y: Reg,
    z: Reg,
    flag: Reg,
    entry: cfglib::BlockId,
}

fn fixture() -> Fixture {
    let mut builder = CodeBuilder::new();
    let x = builder.register("x", Ty::Int);
    let y = builder.register("y", Ty::Int);
    let z = builder.register("z", Ty::Int);
    let flag = builder.register("flag", Ty::Bool);
    builder.parameters(&[x, flag], &[Ty::Int]);
    let entry = builder.block();
    builder.start_at(entry);
    Fixture {
        builder,
        x,
        y,
        z,
        flag,
        entry,
    }
}

/// A function that tests `flag` and returns a different integer on each arm.
fn diamond(terminator: Form, uses: impl Fn(&Fixture) -> Vec<Reg>) -> Code {
    let mut f = fixture();
    let (then_block, else_block) = (f.builder.block(), f.builder.block());
    let operands = uses(&f);
    f.builder.push(f.entry, terminator, &operands, &[]);
    f.builder.link(f.entry, then_block, Link::Yes);
    f.builder.link(f.entry, else_block, Link::No);
    f.builder.push(then_block, Form::Return, &[f.x], &[]);
    f.builder.push(else_block, Form::Return, &[f.y], &[]);
    f.builder.seal()
}

#[test]
fn a_branch_that_the_dialect_classifies_as_a_branch_lifts() {
    let code = diamond(Form::Branch, |f| vec![f.flag]);
    let lifted = try_lift(&code).unwrap_or_else(|error| panic!("{error}"));
    assert!(lifted.report.is_fully_structured());
}

#[test]
fn a_conditional_block_whose_last_instruction_is_not_a_branch_is_unsupported() {
    let code = diamond(Form::Copy, |f| vec![f.flag]);
    assert_unsupported(try_lift(&code), "a conditional block ending in a copy");
}

#[test]
fn a_branch_with_no_condition_is_unsupported() {
    assert_unsupported(
        try_lift(&diamond(Form::Branch, |_| vec![])),
        "a branch of nothing",
    );
}

#[test]
fn a_dispatch_where_the_edges_ask_for_a_branch_is_unsupported() {
    let code = diamond(Form::Switch, |f| vec![f.x]);
    assert_unsupported(try_lift(&code), "a switch with true and false edges");
}

fn dispatch(values: Vec<Lit>) -> Code {
    let mut f = fixture();
    let (arm, other) = (f.builder.block(), f.builder.block());
    f.builder.push(f.entry, Form::Switch, &[f.x], &[]);
    f.builder.link(f.entry, arm, Link::Case(values));
    f.builder.link(f.entry, other, Link::Otherwise);
    f.builder.push(arm, Form::Return, &[f.x], &[]);
    f.builder.push(other, Form::Return, &[f.y], &[]);
    f.builder.seal()
}

#[test]
fn a_switch_case_edge_without_values_is_unsupported() {
    let lifted = try_lift(&dispatch(vec![Lit::Int(1)])).unwrap_or_else(|error| panic!("{error}"));
    assert!(lifted.report.is_fully_structured());
    assert_unsupported(try_lift(&dispatch(vec![])), "a case with no values");
}

#[test]
fn an_instruction_after_the_terminator_of_its_block_is_unsupported() {
    let build = |trailing: bool| {
        let mut f = fixture();
        f.builder.push(f.entry, Form::Return, &[f.x], &[]);
        if trailing {
            f.builder
                .push(f.entry, Form::Const(Lit::Int(1)), &[], &[f.y]);
        }
        f.builder.seal()
    };
    assert!(try_lift(&build(false)).is_ok());
    assert_unsupported(try_lift(&build(true)), "code after a return");
}

#[test]
fn an_operation_with_several_definitions_is_unsupported() {
    let build = |defs: usize| {
        let mut f = fixture();
        let defined = [f.y, f.z];
        f.builder
            .push(f.entry, Form::Apply(Op::Add), &[f.x, f.x], &defined[..defs]);
        f.builder.push(f.entry, Form::Return, &[f.y], &[]);
        f.builder.seal()
    };
    assert!(try_lift(&build(1)).is_ok());
    assert_unsupported(try_lift(&build(2)), "an operation that defines two values");
}

#[test]
fn a_copy_must_have_exactly_one_use_and_one_definition() {
    let build = |uses: &dyn Fn(&Fixture) -> Vec<Reg>| {
        let mut f = fixture();
        let operands = uses(&f);
        f.builder.push(f.entry, Form::Copy, &operands, &[f.y]);
        f.builder.push(f.entry, Form::Return, &[f.y], &[]);
        f.builder.seal()
    };
    assert!(try_lift(&build(&|f| vec![f.x])).is_ok());
    assert_unsupported(try_lift(&build(&|f| vec![f.x, f.x])), "a copy of two uses");
    assert_unsupported(try_lift(&build(&|_| vec![])), "a copy of nothing");
}

#[test]
fn a_store_must_carry_a_value_and_no_definition() {
    let build = |uses: &[usize], defs: &[usize]| {
        let mut f = fixture();
        let registers = [f.x, f.y, f.z];
        let used: Vec<Reg> = uses.iter().map(|&at| registers[at]).collect();
        let defined: Vec<Reg> = defs.iter().map(|&at| registers[at]).collect();
        f.builder
            .push(f.entry, Form::Store(Op::Cell), &used, &defined);
        f.builder.push(f.entry, Form::Return, &[f.x], &[]);
        f.builder.seal()
    };
    assert!(try_lift(&build(&[0, 1], &[])).is_ok());
    assert_unsupported(try_lift(&build(&[], &[])), "a store with no value");
    assert_unsupported(try_lift(&build(&[0, 1], &[2])), "a store that defines");
}

/// Two moves whose first definition is returned, so the lift keeps them:
/// `a, b = b, a` overlaps, `c, d = a, b` does not.
fn moves(overlapping: bool) -> Code {
    let mut f = fixture();
    let (a, b) = (f.x, f.y);
    let spare = f.builder.register("spare", Ty::Int);
    let (first, second) = if overlapping { (b, a) } else { (f.z, spare) };
    f.builder
        .push(f.entry, Form::Move, &[a, b], &[first, second]);
    f.builder.push(f.entry, Form::Return, &[first], &[]);
    f.builder.seal()
}

#[test]
fn an_overlapping_parallel_move_needs_a_temporary_the_dialect_does_not_offer() {
    let disjoint = try_lift(&moves(false)).unwrap_or_else(|error| panic!("{error}"));
    let assignments = disjoint
        .function
        .stmts
        .iter()
        .filter(|stmt| matches!(stmt, crate::structured::StmtNode::Assign { .. }))
        .count();
    assert_eq!(assignments, 2, "each pair becomes one assignment");
    assert_unsupported(try_lift(&moves(true)), "a swap with no temporary");
}

#[test]
fn a_parallel_move_that_does_not_pair_its_uses_with_its_definitions_is_unsupported() {
    let mut f = fixture();
    f.builder.push(f.entry, Form::Move, &[f.x], &[f.y, f.z]);
    f.builder.push(f.entry, Form::Return, &[f.y], &[]);
    assert_unsupported(try_lift(&f.builder.seal()), "one use for two definitions");
}

#[test]
fn a_disjoint_parallel_move_lifts_to_the_behavior_of_the_move() {
    let mut f = fixture();
    let spare = f.builder.register("d", Ty::Int);
    f.builder
        .push(f.entry, Form::Const(Lit::Int(7)), &[], &[f.y]);
    f.builder
        .push(f.entry, Form::Move, &[f.x, f.y], &[f.z, spare]);
    f.builder
        .push(f.entry, Form::Apply(Op::Sub), &[spare, f.z], &[f.x]);
    f.builder.push(f.entry, Form::Return, &[f.x], &[]);
    let code = f.builder.seal();
    let lifted = try_lift(&code).unwrap_or_else(|error| panic!("{error}"));
    for input in [0, 3, -5] {
        let expected = run_flat(&code, &[input]);
        assert_eq!(expected.end, End::Returned(vec![Value::Int(7 - input)]));
        assert_eq!(run_structured(&lifted.function, &[input]), expected);
    }
}

#[test]
fn a_builder_that_refuses_the_function_makes_the_lift_fail_with_its_error() {
    // The branch decides on an integer: the lift states it as written, and
    // the dialect check of the consumer's builder refuses the `if`.
    let mut f = fixture();
    let (then_block, else_block) = (f.builder.block(), f.builder.block());
    f.builder.push(f.entry, Form::Branch, &[f.x], &[]);
    f.builder.link(f.entry, then_block, Link::Yes);
    f.builder.link(f.entry, else_block, Link::No);
    f.builder.push(then_block, Form::Return, &[f.x], &[]);
    f.builder.push(else_block, Form::Return, &[f.y], &[]);
    let code = f.builder.seal();
    let error = try_lift(&code).expect_err("an integer is not a condition");
    let LiftError::Builder(report) = &error else {
        panic!("expected the builder's refusal, got {error}");
    };
    assert!(!report.is_ok());
    assert!(error.to_string().contains(&report.to_string()));
    let source = error.source().expect("a builder error is the source");
    assert_eq!(source.to_string(), report.to_string());
}

#[test]
fn an_unsupported_lift_has_no_source_error() {
    let error = try_lift(&diamond(Form::Copy, |f| vec![f.flag])).expect_err("unsupported");
    assert!(error.source().is_none());
    assert_ne!(error.to_string(), "");
}

#[test]
fn a_predicated_tree_has_no_hlil_translation() {
    let mut f = fixture();
    f.builder
        .push(f.entry, Form::Const(Lit::Int(1)), &[], &[f.y]);
    f.builder.push(f.entry, Form::Return, &[f.y], &[]);
    let mut code = f.builder.seal();
    let guard = code.declared.len() - 1;
    assert_eq!(code.locals[guard].name, "flag");
    let flag = Reg::from_index(guard);
    code.cfg.block_mut(code.first_block()).instructions_mut()[0].guard = Some((flag, true));

    // The same function lifts through the ordinary structurer.
    assert!(try_lift(&code).is_ok());

    let (tree, report) = ControlTree::compute_predicated_with_report(&code.cfg);
    let outcome = lift_with_tree::<Calc, _, _>(
        &code,
        &tree,
        &report,
        Builder::new(),
        LiftMetadata::Preserve,
    );
    assert_unsupported(outcome, "a guarded run");
}

/// A tree computed for `big` read over the graph `small`.
fn lift_over_the_wrong_graph(small: &Code, big: &Code) -> bool {
    let (tree, report) = ControlTree::compute_with_report(&big.cfg);
    catch_unwind(AssertUnwindSafe(|| {
        lift_with_tree::<Calc, _, _>(
            small,
            &tree,
            &report,
            Builder::new(),
            LiftMetadata::Preserve,
        )
    }))
    .is_err()
}

/// One block that defines `y` `defs` times and then returns it.
fn chain(defs: usize) -> Code {
    let mut f = fixture();
    for _ in 0..defs {
        f.builder
            .push(f.entry, Form::Apply(Op::Add), &[f.x, f.x], &[f.y]);
    }
    f.builder.push(f.entry, Form::Return, &[f.y], &[]);
    f.builder.seal()
}

#[test]
fn lifting_with_a_tree_that_names_a_block_or_an_instruction_the_graph_lacks_panics() {
    // A tree of a graph with one more block than the graph it is read over.
    let mut two_blocks = fixture();
    let next = two_blocks.builder.block();
    two_blocks.builder.push(
        two_blocks.entry,
        Form::Const(Lit::Int(1)),
        &[],
        &[two_blocks.y],
    );
    two_blocks.builder.link(two_blocks.entry, next, Link::Next);
    two_blocks
        .builder
        .push(next, Form::Return, &[two_blocks.y], &[]);
    let two_blocks = two_blocks.builder.seal();
    let mut one_block = fixture();
    one_block
        .builder
        .push(one_block.entry, Form::Return, &[one_block.x], &[]);
    let one_block = one_block.builder.seal();
    assert!(lift_over_the_wrong_graph(&one_block, &two_blocks));
    // A tree whose span reaches past the instructions that the block holds.
    assert!(lift_over_the_wrong_graph(&chain(1), &chain(3)));
    // The same tree over its own graph lifts.
    let own = chain(3);
    let (tree, report) = ControlTree::compute_with_report(&own.cfg);
    assert!(
        lift_with_tree::<Calc, _, _>(&own, &tree, &report, Builder::new(), LiftMetadata::Preserve)
            .is_ok()
    );
}
