//! The control-flow shapes that the lift translates, over the toy dialects:
//! post-tested and endless loops, loop exits that need a negation or an
//! explicit test, labeled transfers, gotos, and the handler kinds.
//!
//! A shape whose text depends on what a dialect can express (`Imp` has a
//! logical negation, `Sym` has none) states both texts and picks by the
//! hook, so the test fails if a hook is read wrongly in either direction.

extern crate alloc;

use alloc::vec;

use crate::ir::hlil::FlatBuilder;
use crate::region::{Handler as RegionHandler, HandlerBody, HandlerKind, Region, RegionId};

use super::Prim;
use super::flat::{FlatKit, ToyEdge, ToyFlat, ToyFlatBuilder, ToyForm};
use super::flat_lift::{boolean, int, lifted, lifted_text};

over_toys!(
    a_break_out_of_a_loop_through_a_switch_names_the_loop,
    a_break_out_of_two_loops_names_the_outer_one,
    a_continue_of_the_outer_loop_names_it,
    a_two_entry_cycle_keeps_its_labels_and_gotos,
    handler_kinds_carry_over_and_bodies_follow_their_handlers,
    a_finally_region_lifts_to_a_finally_body,
    a_filter_handler_keeps_its_kind_and_leaves_its_predicate_to_the_sweep,
    a_post_tested_loop_lifts_to_do_while,
    a_post_tested_loop_that_exits_on_true_states_its_negation,
    a_fused_post_tested_loop_inverts_its_operation,
    a_pre_tested_loop_that_exits_on_true_states_its_negation,
    a_loop_condition_that_needs_statements_becomes_an_explicit_test,
    a_loop_without_a_test_is_endless,
    a_dead_protected_leave_block_leaves_no_dangling_goto,
);

/// How the test of a loop reads.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Test {
    /// `lt(i, n)` computed into a boolean that a branch reads.
    Plain,
    /// A fused `below(i, n)` branch.
    Fused,
}

/// `do { i = i + 1 } while (<test>); return i;`, with the test in its own
/// block, which continues to the top on `continue_on_true`.
fn post_tested<D: FlatKit>(continue_on_true: bool, kind: Test) -> ToyFlat<D> {
    let mut flat = ToyFlatBuilder::<D>::new();
    let top = flat.new_block(Some("top"));
    let check = flat.new_block(Some("check"));
    let exit = flat.new_block(Some("exit"));
    let i = flat.int_variable(0);
    let n = flat.int_variable(0);
    let one = flat.int_variable(1);
    let condition = flat.int_variable(1);
    flat.literal(top, 1, one);
    flat.operate(
        top,
        Prim::Add,
        &[int::<D>(i), int::<D>(one)],
        &[int::<D>(i)],
    );
    match kind {
        Test::Plain => {
            flat.operate(
                check,
                Prim::LessThan,
                &[int::<D>(i), int::<D>(n)],
                &[boolean::<D>(condition)],
            );
            flat.branch(check, condition);
        }
        Test::Fused => flat.emit(
            check,
            ToyForm::BranchOperation(D::op(Prim::Below)),
            &[int::<D>(i), int::<D>(n)],
            &[],
        ),
    }
    flat.give_back(exit, &[i]);
    flat.start(top);
    flat.connect(top, check, ToyEdge::Fall);
    let (back, out) = if continue_on_true {
        (ToyEdge::True, ToyEdge::False)
    } else {
        (ToyEdge::False, ToyEdge::True)
    };
    flat.connect(check, top, back);
    flat.connect(check, exit, out);
    flat.takes(&[i, n]);
    flat.finish().expect("the post-tested loop is well formed")
}

fn a_post_tested_loop_lifts_to_do_while<D: FlatKit>() {
    assert_eq!(
        lifted_text(&post_tested::<D>(true, Test::Plain)),
        "do {\n    v0 = add(v0, 1);\n} while (lt(v0, v1));\nreturn v0;\n"
    );
}

fn a_post_tested_loop_that_exits_on_true_states_its_negation<D: FlatKit>() {
    let text = lifted_text(&post_tested::<D>(false, Test::Plain));
    if D::logical_not().is_some() {
        assert_eq!(
            text,
            "do {\n    v0 = add(v0, 1);\n} while (not(lt(v0, v1)));\nreturn v0;\n"
        );
    } else {
        // No negation exists, so the post-test becomes an exit at the bottom.
        assert_eq!(
            text,
            "loop {\n    v0 = add(v0, 1);\n    if (lt(v0, v1)) {\n        break;\n    }\n}\nreturn v0;\n"
        );
    }
}

fn a_fused_post_tested_loop_inverts_its_operation<D: FlatKit>() {
    // Both toys invert `below`, so neither needs a wrapper or an exit test.
    assert_eq!(
        lifted_text(&post_tested::<D>(false, Test::Fused)),
        "do {\n    v0 = add(v0, 1);\n} while (at-least(v0, v1));\nreturn v0;\n"
    );
}

/// `while (<test>) { i = i + 1 } return i;`, which exits on true.
fn exit_on_true_loop<D: FlatKit>() -> ToyFlat<D> {
    let mut flat = ToyFlatBuilder::<D>::new();
    let header = flat.new_block(Some("header"));
    let body = flat.new_block(Some("body"));
    let exit = flat.new_block(Some("exit"));
    let i = flat.int_variable(0);
    let n = flat.int_variable(0);
    let one = flat.int_variable(1);
    let condition = flat.int_variable(1);
    flat.operate(
        header,
        Prim::LessThan,
        &[int::<D>(i), int::<D>(n)],
        &[boolean::<D>(condition)],
    );
    flat.branch(header, condition);
    flat.literal(body, 1, one);
    flat.operate(
        body,
        Prim::Add,
        &[int::<D>(i), int::<D>(one)],
        &[int::<D>(i)],
    );
    flat.give_back(exit, &[i]);
    flat.start(header);
    flat.connect(header, exit, ToyEdge::True);
    flat.connect(header, body, ToyEdge::False);
    flat.connect(body, header, ToyEdge::Fall);
    flat.takes(&[i, n]);
    flat.finish().expect("the exit-on-true loop is well formed")
}

fn a_pre_tested_loop_that_exits_on_true_states_its_negation<D: FlatKit>() {
    let text = lifted_text(&exit_on_true_loop::<D>());
    if D::logical_not().is_some() {
        assert_eq!(
            text,
            "while (not(lt(v0, v1))) {\n    v0 = add(v0, 1);\n}\nreturn v0;\n"
        );
    } else {
        assert_eq!(
            text,
            "loop {\n    if (lt(v0, v1)) {\n        break;\n    }\n    v0 = add(v0, 1);\n}\nreturn v0;\n"
        );
    }
}

fn a_loop_condition_that_needs_statements_becomes_an_explicit_test<D: FlatKit>() {
    // The header runs `call()` before it decides, so the condition is not
    // one expression: the test moves to the top of an endless loop.
    let mut flat = ToyFlatBuilder::<D>::new();
    let header = flat.new_block(Some("header"));
    let body = flat.new_block(Some("body"));
    let exit = flat.new_block(Some("exit"));
    let i = flat.int_variable(0);
    let n = flat.int_variable(0);
    let condition = flat.int_variable(1);
    flat.operate(header, Prim::Call, &[], &[]);
    flat.operate(
        header,
        Prim::LessThan,
        &[int::<D>(i), int::<D>(n)],
        &[boolean::<D>(condition)],
    );
    flat.branch(header, condition);
    flat.operate(body, Prim::Add, &[int::<D>(i), int::<D>(n)], &[int::<D>(i)]);
    flat.give_back(exit, &[i]);
    flat.start(header);
    flat.connect(header, body, ToyEdge::True);
    flat.connect(header, exit, ToyEdge::False);
    flat.connect(body, header, ToyEdge::Fall);
    flat.takes(&[i, n]);
    let source = flat.finish().expect("the loop is well formed");

    assert_eq!(
        lifted_text(&source),
        "loop {\n    call();\n    if (lt(v0, v1)) {\n    } else {\n        break;\n    }\n    v0 = add(v0, v1);\n}\nreturn v0;\n"
    );
}

fn a_loop_without_a_test_is_endless<D: FlatKit>() {
    let mut flat = ToyFlatBuilder::<D>::new();
    let spin = flat.new_block(Some("spin"));
    flat.operate(spin, Prim::Call, &[], &[]);
    flat.start(spin);
    flat.connect(spin, spin, ToyEdge::Fall);
    let source = flat.finish().expect("the spin is well formed");

    assert_eq!(lifted_text(&source), "loop {\n    call();\n}\n");
}

fn a_break_out_of_a_loop_through_a_switch_names_the_loop<D: FlatKit>() {
    // `while (i < n) { switch (s) { case 1: <leave>; case 2: call(); default:
    // call(); } }`. An unlabeled break inside a switch would leave the switch,
    // so the break that leaves the loop names it.
    let mut flat = ToyFlatBuilder::<D>::new();
    let header = flat.new_block(Some("header"));
    let dispatch = flat.new_block(Some("dispatch"));
    let exit = flat.new_block(Some("exit"));
    let again = flat.new_block(Some("again"));
    let other = flat.new_block(Some("other"));
    let probe = flat.new_block(Some("probe"));
    let rest = flat.new_block(Some("rest"));
    let [i, n, selector] = [0, 0, 0].map(|number| flat.int_variable(number));
    let [condition, leave] = [1, 1].map(|number| flat.int_variable(number));
    flat.operate(
        probe,
        Prim::LessThan,
        &[int::<D>(n), int::<D>(i)],
        &[boolean::<D>(leave)],
    );
    flat.branch(probe, leave);
    flat.connect(probe, exit, ToyEdge::True);
    flat.connect(probe, rest, ToyEdge::False);
    flat.connect(rest, header, ToyEdge::Fall);
    flat.operate(rest, Prim::Call, &[], &[]);
    flat.operate(
        header,
        Prim::LessThan,
        &[int::<D>(i), int::<D>(n)],
        &[boolean::<D>(condition)],
    );
    flat.branch(header, condition);
    flat.emit(dispatch, ToyForm::Switch, &[int::<D>(selector)], &[]);
    flat.give_back(exit, &[i]);
    flat.operate(again, Prim::Call, &[], &[]);
    flat.operate(other, Prim::Call, &[], &[]);
    flat.start(header);
    flat.connect(header, dispatch, ToyEdge::True);
    flat.connect(header, exit, ToyEdge::False);
    flat.connect(dispatch, probe, ToyEdge::Case(vec![D::constant(1)]));
    flat.connect(dispatch, again, ToyEdge::Case(vec![D::constant(2)]));
    flat.connect(dispatch, other, ToyEdge::Default);
    flat.connect(again, header, ToyEdge::Fall);
    flat.connect(other, header, ToyEdge::Fall);
    flat.takes(&[i, n, selector]);
    let source = flat.finish().expect("the dispatching loop is well formed");

    assert_eq!(
        lifted_text(&source),
        "header: {\n    while (lt(v0, v1)) {\n        switch (v2) {\n            case 1: {\n                if (lt(v1, v0)) {\n                    break header;\n                } else {\n                    call();\n                    continue;\n                }\n            }\n            case 2: {\n                call();\n                continue;\n            }\n            default: {\n                call();\n                continue;\n            }\n        }\n    }\n}\nreturn v0;\n"
    );
}

/// `while (i < n) { while (j < n) { if (k < n) <transfer>; j = j + 1 }; i = i + 1 }`
/// where the transfer leaves for the exit, or continues the outer loop.
fn nested_loops<D: FlatKit>(to_exit: bool) -> ToyFlat<D> {
    let mut flat = ToyFlatBuilder::<D>::new();
    let outer = flat.new_block(Some("outer"));
    let inner = flat.new_block(Some("inner"));
    let probe = flat.new_block(Some("probe"));
    let step = flat.new_block(Some("step"));
    let latch = flat.new_block(Some("latch"));
    let exit = flat.new_block(Some("exit"));
    let [i, j, k, n] = [0, 0, 0, 0].map(|number| flat.int_variable(number));
    let [c1, c2, c3] = [1, 1, 1].map(|number| flat.int_variable(number));
    for (block, left, flag) in [(outer, i, c1), (inner, j, c2), (probe, k, c3)] {
        flat.operate(
            block,
            Prim::LessThan,
            &[int::<D>(left), int::<D>(n)],
            &[boolean::<D>(flag)],
        );
        flat.branch(block, flag);
    }
    flat.operate(step, Prim::Add, &[int::<D>(j), int::<D>(n)], &[int::<D>(j)]);
    flat.operate(
        latch,
        Prim::Add,
        &[int::<D>(i), int::<D>(n)],
        &[int::<D>(i)],
    );
    flat.give_back(exit, &[i]);
    flat.start(outer);
    flat.connect(outer, inner, ToyEdge::True);
    flat.connect(outer, exit, ToyEdge::False);
    flat.connect(inner, probe, ToyEdge::True);
    flat.connect(inner, latch, ToyEdge::False);
    let transfer = if to_exit { exit } else { outer };
    flat.connect(probe, transfer, ToyEdge::True);
    flat.connect(probe, step, ToyEdge::False);
    flat.connect(step, inner, ToyEdge::Fall);
    flat.connect(latch, outer, ToyEdge::Fall);
    flat.takes(&[i, n]);
    flat.finish().expect("the nested loops are well formed")
}

fn a_break_out_of_two_loops_names_the_outer_one<D: FlatKit>() {
    assert_eq!(
        lifted_text(&nested_loops::<D>(true)),
        "outer: {\n    while (lt(v0, v3)) {\n        while (lt(v1, v3)) {\n            if (lt(v2, v3)) {\n                break outer;\n            } else {\n                v1 = add(v1, v3);\n                continue;\n            }\n        }\n        v0 = add(v0, v3);\n    }\n}\nreturn v0;\n"
    );
}

fn a_continue_of_the_outer_loop_names_it<D: FlatKit>() {
    assert_eq!(
        lifted_text(&nested_loops::<D>(false)),
        "outer: {\n    while (lt(v0, v3)) {\n        while (lt(v1, v3)) {\n            if (lt(v2, v3)) {\n                continue outer;\n            } else {\n                v1 = add(v1, v3);\n                continue;\n            }\n        }\n        v0 = add(v0, v3);\n    }\n}\nreturn v0;\n"
    );
}

fn a_two_entry_cycle_keeps_its_labels_and_gotos<D: FlatKit>() {
    // `start` enters the cycle at either `left` or `right`, so no loop header
    // dominates it.
    let mut flat = ToyFlatBuilder::<D>::new();
    let start = flat.new_block(Some("start"));
    let left = flat.new_block(Some("left"));
    let right = flat.new_block(Some("right"));
    let exit = flat.new_block(Some("exit"));
    let [x, y] = [0, 0].map(|number| flat.int_variable(number));
    let [c1, c2] = [1, 1].map(|number| flat.int_variable(number));
    flat.operate(
        start,
        Prim::LessThan,
        &[int::<D>(x), int::<D>(y)],
        &[boolean::<D>(c1)],
    );
    flat.branch(start, c1);
    flat.operate(left, Prim::Call, &[], &[]);
    flat.operate(
        left,
        Prim::LessThan,
        &[int::<D>(x), int::<D>(y)],
        &[boolean::<D>(c2)],
    );
    flat.branch(left, c2);
    flat.operate(right, Prim::Call, &[], &[]);
    flat.give_back(exit, &[x]);
    flat.start(start);
    flat.connect(start, left, ToyEdge::True);
    flat.connect(start, right, ToyEdge::False);
    flat.connect(left, right, ToyEdge::True);
    flat.connect(left, exit, ToyEdge::False);
    flat.connect(right, left, ToyEdge::Fall);
    flat.takes(&[x, y]);
    let source = flat.finish().expect("the cycle is well formed");

    let lifted = lifted(&source);
    assert!(!lifted.report.is_fully_structured());
    assert_eq!(
        crate::ir::hlil::pseudocode(&lifted.function),
        "if (lt(v0, v1)) {\n} else {\n    right: {\n        call();\n    }\n}\ncall();\nif (lt(v0, v1)) {\n    goto right;\n}\nreturn v0;\n"
    );
}

fn handler_kinds_carry_over_and_bodies_follow_their_handlers<D: FlatKit>() {
    let mut flat = ToyFlatBuilder::<D>::new();
    let protected = flat.new_block(Some("protected"));
    let after = flat.new_block(Some("after"));
    let caught = flat.new_block(Some("caught"));
    let fault = flat.new_block(Some("fault"));
    let x = flat.int_variable(0);
    flat.operate(protected, Prim::Call, &[], &[int::<D>(x)]);
    flat.give_back(after, &[x]);
    flat.literal(caught, 1, x);
    flat.give_back(caught, &[x]);
    flat.literal(fault, 2, x);
    flat.give_back(fault, &[x]);
    flat.start(protected);
    flat.connect(protected, after, ToyEdge::Fall);
    flat.connect(protected, caught, ToyEdge::Except);
    flat.connect(protected, fault, ToyEdge::Except);
    flat.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [protected].into_iter().collect(),
        handlers: vec![
            RegionHandler {
                entry: caught,
                body: HandlerBody::known([caught]),
                kind: HandlerKind::Catch,
            },
            RegionHandler {
                entry: fault,
                body: HandlerBody::known([fault]),
                kind: HandlerKind::Fault,
            },
        ],
        parent: None,
    });
    let source = flat.finish().expect("the handlers are well formed");

    assert_eq!(
        lifted_text(&source),
        "try {\n    v0 = call();\n} catch () {\n    return 1;\n} fault () {\n    return 2;\n}\nreturn v0;\n"
    );
}

fn a_finally_region_lifts_to_a_finally_body<D: FlatKit>() {
    let mut flat = ToyFlatBuilder::<D>::new();
    let protected = flat.new_block(Some("protected"));
    let after = flat.new_block(Some("after"));
    let cleanup = flat.new_block(Some("cleanup"));
    let x = flat.int_variable(0);
    flat.operate(protected, Prim::Call, &[], &[int::<D>(x)]);
    flat.give_back(after, &[x]);
    flat.operate(cleanup, Prim::Call, &[], &[]);
    flat.give_back(cleanup, &[x]);
    flat.start(protected);
    flat.connect(protected, after, ToyEdge::Fall);
    flat.connect(protected, cleanup, ToyEdge::Except);
    flat.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [protected].into_iter().collect(),
        handlers: vec![RegionHandler {
            entry: cleanup,
            body: HandlerBody::known([cleanup]),
            kind: HandlerKind::Finally,
        }],
        parent: None,
    });
    let source = flat.finish().expect("the cleanup is well formed");

    assert_eq!(
        lifted_text(&source),
        "try {\n    v0 = call();\n} finally {\n    call();\n    return v0;\n}\nreturn v0;\n"
    );
}

fn a_filter_handler_keeps_its_kind_and_leaves_its_predicate_to_the_sweep<D: FlatKit>() {
    let mut flat = ToyFlatBuilder::<D>::new();
    let protected = flat.new_block(Some("protected"));
    let after = flat.new_block(Some("after"));
    let handler = flat.new_block(Some("handler"));
    let predicate = flat.new_block(Some("predicate"));
    let x = flat.int_variable(0);
    flat.operate(protected, Prim::Call, &[], &[int::<D>(x)]);
    flat.give_back(after, &[x]);
    flat.literal(handler, 1, x);
    flat.give_back(handler, &[x]);
    flat.literal(predicate, 0, x);
    flat.give_back(predicate, &[x]);
    flat.start(protected);
    flat.connect(protected, after, ToyEdge::Fall);
    flat.connect(protected, handler, ToyEdge::Except);
    flat.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [protected].into_iter().collect(),
        handlers: vec![RegionHandler {
            entry: handler,
            body: HandlerBody::known([handler]),
            kind: HandlerKind::Filter {
                filter_block: predicate,
            },
        }],
        parent: None,
    });
    let source = flat.finish_unchecked();

    // The predicate is a funclet that the structuring sweep emits after the
    // function, so the filter body of the handler stays empty.
    assert_eq!(
        lifted_text(&source),
        "try {\n    v0 = call();\n} filter () {\n    when {\n    }\n    return 1;\n}\nreturn v0;\npredicate: {\n    return 0;\n}\n"
    );
}

/// `try { return i } catch { i = i + i }`, whose catch falls into a join that
/// holds only a jump, then `return i`. A dead block that only jumps to the
/// join, protected by the try, is the leave that a body which cannot complete
/// leaves behind.
fn a_dead_protected_leave_block_leaves_no_dangling_goto<D: FlatKit>() {
    let mut flat = ToyFlatBuilder::<D>::new();
    let body = flat.new_block(Some("body"));
    let pad = flat.new_block(Some("pad"));
    let join = flat.new_block(Some("join"));
    let after = flat.new_block(Some("after"));
    let leave = flat.new_block(Some("leave"));
    let i = flat.int_variable(0);
    flat.give_back(body, &[i]);
    flat.operate(pad, Prim::Add, &[int::<D>(i), int::<D>(i)], &[int::<D>(i)]);
    flat.jump(pad);
    flat.jump(join);
    flat.give_back(after, &[i]);
    flat.jump(leave);
    flat.start(body);
    flat.connect(body, pad, ToyEdge::Except);
    flat.connect(pad, join, ToyEdge::Fall);
    flat.connect(join, after, ToyEdge::Fall);
    flat.connect(leave, join, ToyEdge::Fall);
    flat.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [body, leave].into_iter().collect(),
        handlers: vec![RegionHandler {
            entry: pad,
            body: HandlerBody::known([pad]),
            kind: HandlerKind::CatchAll,
        }],
        parent: None,
    });
    flat.takes(&[i]);
    let source = flat.finish_unchecked();

    // The dead leave block is unreachable source: lifting it would need a
    // goto to the join, which is empty and so has no label.
    let text = lifted_text(&source);
    assert!(!text.contains("goto"), "{text}");
    assert_eq!(text.matches("return").count(), 2, "{text}");
}
