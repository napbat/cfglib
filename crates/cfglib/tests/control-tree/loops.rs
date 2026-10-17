//! Loops: pre-tested, post-tested, endless, nested, and the `break` and
//! `continue` transfers out of and back into them.

use cfglib::{BlockId, ControlNode, EdgeKind, LoopKind};

use crate::builders::{Flow, labeled_break_loops, pseudocode};
use crate::golden::assert_golden;
use crate::oracles::structure_and_check;
use crate::reading::{body_nodes, leaf_blocks, top_level};
use crate::tree_shape::{assert_fully_structured, nodes_where};

/// `i = 0; while i < n { sum += i; i += 1 }; return sum`, the machine shape:
/// the body ends in a `Jump` back to the header.
struct SumLoop {
    flow: Flow,
    header: BlockId,
    body: BlockId,
    step: BlockId,
    exit: BlockId,
}

fn sum_loop() -> SumLoop {
    let mut flow = Flow::starting_with("i = 0");
    let header = flow.stmt("i < n");
    let body = flow.stmt("sum += i");
    let step = flow.stmt("i += 1");
    let exit = flow.stmt("return sum");
    flow.fall(flow.entry(), header);
    flow.branch(header, body, exit);
    flow.fall(body, step);
    flow.jump(step, header);
    SumLoop {
        flow,
        header,
        body,
        step,
        exit,
    }
}

#[test]
fn for_like_loop_is_a_pre_tested_loop_between_its_init_and_its_follow() {
    let SumLoop {
        flow,
        header,
        body,
        step,
        exit,
    } = sum_loop();

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let [
        ControlNode::Block(init),
        ControlNode::Loop {
            header: loop_header,
            kind:
                LoopKind::While {
                    condition,
                    exit_on_true,
                },
            body: loop_body,
        },
        ControlNode::Return(tail),
    ] = top_level(&tree)[..]
    else {
        panic!("expected init, loop, follow: {tree:?}");
    };
    assert_eq!(init.block(), flow.entry());
    assert_eq!(loop_header, header);
    assert_eq!(
        tree.spans(condition)
            .iter()
            .map(|span| span.block())
            .collect::<Vec<_>>(),
        [header]
    );
    assert!(
        !exit_on_true,
        "the true edge iterates and the false edge leaves"
    );
    assert_eq!(leaf_blocks(&tree, loop_body), [body, step]);
    assert_eq!(tail.block(), exit);
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "i = 0\nwhile {\n    i < n\n    break if !cond;\n    sum += i\n    i += 1\n}\nreturn sum\n"
    );
}

#[test]
fn while_loop_whose_true_edge_leaves_records_the_inverted_exit() {
    // `while !done`: the ConditionalTrue edge leaves the loop.
    let mut flow = Flow::starting_with("poll()");
    let header = flow.stmt("done");
    let body = flow.stmt("work()");
    let exit = flow.stmt("return");
    flow.fall(flow.entry(), header);
    flow.branch(header, exit, body);
    flow.jump(body, header);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let loops = nodes_where(&tree, |node| matches!(node, ControlNode::Loop { .. }));
    let [
        ControlNode::Loop {
            kind: LoopKind::While { exit_on_true, .. },
            ..
        },
    ] = loops.iter().map(|&id| *tree.node(id)).collect::<Vec<_>>()[..]
    else {
        panic!("one pre-tested loop: {tree:?}");
    };
    assert!(exit_on_true);
    assert!(pseudocode(&flow.cfg, &tree).contains("break if cond;"));
}

#[test]
fn condition_chain_of_several_blocks_is_one_pre_test() {
    // `while (load(); ok())`: the header only prepares, the second block tests.
    let mut flow = Flow::starting_with("p = head");
    let header = flow.stmt("load(p)");
    let test = flow.stmt("p != null");
    let body = flow.stmt("p = p.next");
    let exit = flow.stmt("return");
    flow.fall(flow.entry(), header);
    flow.fall(header, test);
    flow.branch(test, body, exit);
    flow.jump(body, header);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let [
        _,
        ControlNode::Loop {
            kind: LoopKind::While { condition, .. },
            body: loop_body,
            ..
        },
        _,
    ] = top_level(&tree)[..]
    else {
        panic!("expected a pre-tested loop: {tree:?}");
    };
    let chain: Vec<BlockId> = tree.spans(condition).iter().map(|s| s.block()).collect();
    assert_eq!(chain, [header, test], "header first, branch block last");
    assert_eq!(leaf_blocks(&tree, loop_body), [body]);
}

#[test]
fn do_while_loop_tests_in_its_latch_after_the_body() {
    let mut flow = Flow::starting_with("i = 0");
    let body = flow.stmt("i += 1");
    let latch = flow.stmt("i < n");
    let exit = flow.stmt("return i");
    flow.fall(flow.entry(), body);
    flow.fall(body, latch);
    flow.edge(latch, body, EdgeKind::ConditionalTrue);
    flow.edge(latch, exit, EdgeKind::ConditionalFalse);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let [
        _,
        ControlNode::Loop {
            header,
            kind:
                LoopKind::DoWhile {
                    latch: latch_span,
                    continue_on_true,
                },
            body: loop_body,
        },
        _,
    ] = top_level(&tree)[..]
    else {
        panic!("expected a post-tested loop: {tree:?}");
    };
    assert_eq!(header, body);
    assert_eq!(latch_span.block(), latch);
    assert!(continue_on_true);
    assert_eq!(leaf_blocks(&tree, loop_body), [body]);
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "i = 0\ndo {\n    i += 1\n    i < n\n} while (cond)\nreturn i\n"
    );
}

#[test]
fn do_while_loop_that_exits_on_the_true_edge_negates_its_condition() {
    let mut flow = Flow::starting_with("retry()");
    let latch = flow.stmt("ok");
    let exit = flow.stmt("return");
    flow.fall(flow.entry(), latch);
    flow.edge(latch, exit, EdgeKind::ConditionalTrue);
    flow.edge(latch, flow.entry(), EdgeKind::ConditionalFalse);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let [
        ControlNode::Loop {
            kind: LoopKind::DoWhile {
                continue_on_true, ..
            },
            ..
        },
        _,
    ] = top_level(&tree)[..]
    else {
        panic!("expected a post-tested loop first: {tree:?}");
    };
    assert!(!continue_on_true);
    assert!(pseudocode(&flow.cfg, &tree).contains("} while (!cond)"));
}

#[test]
fn single_block_spin_loop_is_its_own_latch_with_an_empty_body() {
    let mut flow = Flow::starting_with("flag == 0");
    let exit = flow.stmt("return");
    flow.edge(flow.entry(), flow.entry(), EdgeKind::ConditionalTrue);
    flow.edge(flow.entry(), exit, EdgeKind::ConditionalFalse);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let [
        ControlNode::Loop {
            header,
            kind:
                LoopKind::DoWhile {
                    latch,
                    continue_on_true,
                },
            body,
        },
        _,
    ] = top_level(&tree)[..]
    else {
        panic!("expected a spin loop: {tree:?}");
    };
    assert_eq!(header, flow.entry());
    assert_eq!(latch.block(), flow.entry());
    assert!(continue_on_true);
    assert!(body.is_empty());
}

#[test]
fn loop_with_several_latches_and_no_test_is_endless_and_leaves_through_break() {
    // header: if odd { a } else { b }; a jumps back; b tests `bad` to leave.
    let mut flow = Flow::starting_with("n = next()");
    let header = flow.stmt("odd(n)");
    let odd = flow.stmt("on_odd()");
    let even = flow.stmt("bad(n)");
    let exit = flow.stmt("return");
    flow.fall(flow.entry(), header);
    flow.branch(header, odd, even);
    flow.jump(odd, header);
    flow.edge(even, exit, EdgeKind::ConditionalTrue);
    flow.edge(even, header, EdgeKind::ConditionalFalse);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let loops = nodes_where(&tree, |node| matches!(node, ControlNode::Loop { .. }));
    let [
        ControlNode::Loop {
            header: loop_header,
            kind: LoopKind::Endless,
            ..
        },
    ] = loops.iter().map(|&id| *tree.node(id)).collect::<Vec<_>>()[..]
    else {
        panic!("one endless loop: {tree:?}");
    };
    assert_eq!(loop_header, header);
    let breaks = nodes_where(&tree, |node| {
        matches!(node, ControlNode::Break { label: None })
    });
    assert_eq!(breaks.len(), 1, "the failing test leaves the loop");
}

#[test]
fn break_and_continue_resolve_against_the_loop_instead_of_becoming_gotos() {
    // while i < n { if skip(i) { continue }; if done(i) { break }; work(i) }
    let mut flow = Flow::starting_with("i = 0");
    let header = flow.stmt("i < n");
    let skip = flow.stmt("skip(i)");
    let done = flow.stmt("done(i)");
    let work = flow.stmt("work(i)");
    let exit = flow.stmt("return");
    flow.fall(flow.entry(), header);
    flow.branch(header, skip, exit);
    flow.branch(skip, header, done);
    flow.branch(done, exit, work);
    flow.jump(work, header);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let continues = nodes_where(&tree, |node| matches!(node, ControlNode::Continue { .. }));
    let breaks = nodes_where(&tree, |node| matches!(node, ControlNode::Break { .. }));
    assert_eq!(
        continues.len(),
        2,
        "the skip arm and the end of the work arm"
    );
    assert_eq!(breaks.len(), 1);
    assert!(
        nodes_where(&tree, |node| matches!(node, ControlNode::Label { .. })).is_empty(),
        "a transfer to the innermost loop needs no label"
    );
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "i = 0\nwhile {\n    i < n\n    break if !cond;\n    if {\n        continue;\n    } else {\n        if {\n            break;\n        } else {\n            work(i)\n            continue;\n        }\n    }\n}\nreturn\n"
    );
}

#[test]
fn break_out_of_the_inner_of_two_loops_stays_unlabeled() {
    // for i { for j { if hit(i, j) { break } } }
    let mut flow = Flow::starting_with("i = 0");
    let outer = flow.stmt("i < n");
    let inner_init = flow.stmt("j = 0");
    let inner = flow.stmt("j < m");
    let hit = flow.stmt("hit(i, j)");
    let step_j = flow.stmt("j += 1");
    let step_i = flow.stmt("i += 1");
    let exit = flow.stmt("return");
    flow.fall(flow.entry(), outer);
    flow.branch(outer, inner_init, exit);
    flow.fall(inner_init, inner);
    flow.branch(inner, hit, step_i);
    flow.branch(hit, step_i, step_j);
    flow.jump(step_j, inner);
    flow.jump(step_i, outer);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let loops = nodes_where(&tree, |node| matches!(node, ControlNode::Loop { .. }));
    assert_eq!(loops.len(), 2);
    let [ControlNode::Loop { body, .. }] = [*tree.node(loops[0])] else {
        panic!("the first loop in preorder is the outer one");
    };
    let outer_body = body_nodes(&tree, body);
    assert!(
        outer_body
            .iter()
            .any(|node| matches!(node, ControlNode::Loop { .. })),
        "the inner loop sits in the outer body: {tree:?}"
    );
    assert!(matches!(
        tree.node(loops[1]),
        ControlNode::Loop {
            header,
            ..
        } if *header == inner
    ));
    let breaks = nodes_where(&tree, |node| {
        matches!(node, ControlNode::Break { label: None })
    });
    assert_eq!(breaks.len(), 1);
}

#[test]
fn multi_level_break_labels_the_outer_loop_and_names_it() {
    let (flow, outer) = labeled_break_loops();

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let labeled_breaks = nodes_where(
        &tree,
        |node| matches!(node, ControlNode::Break { label: Some(block) } if *block == outer),
    );
    assert_eq!(labeled_breaks.len(), 1);
    let labels = nodes_where(
        &tree,
        |node| matches!(node, ControlNode::Label { block, .. } if *block == outer),
    );
    assert_eq!(
        labels.len(),
        1,
        "the outer loop is wrapped in the label the break names"
    );
    let ControlNode::Label { body, .. } = *tree.node(labels[0]) else {
        unreachable!("filtered to labels");
    };
    assert!(matches!(
        body_nodes(&tree, body)[..],
        [ControlNode::Loop { header, .. }] if header == outer
    ));
    assert_golden(
        "control-tree/multi-level-break.pseudo",
        &pseudocode(&flow.cfg, &tree),
    );
}

#[test]
fn continue_of_the_outer_loop_from_inside_the_inner_one_is_labeled() {
    // for i { for j { if skip(i, j) { continue outer } } }
    let mut flow = Flow::starting_with("i = 0");
    let outer = flow.stmt("i < n");
    let inner_init = flow.stmt("j = 0");
    let inner = flow.stmt("j < m");
    let skip = flow.stmt("skip(i, j)");
    let step_j = flow.stmt("j += 1");
    let step_i = flow.stmt("i += 1");
    let exit = flow.stmt("return");
    flow.fall(flow.entry(), outer);
    flow.branch(outer, inner_init, exit);
    flow.fall(inner_init, inner);
    flow.branch(inner, skip, step_i);
    flow.branch(skip, outer, step_j);
    flow.jump(step_j, inner);
    flow.jump(step_i, outer);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let outer_continues = nodes_where(
        &tree,
        |node| matches!(node, ControlNode::Continue { label: Some(block) } if *block == outer),
    );
    assert_eq!(outer_continues.len(), 1);
    let labels = nodes_where(
        &tree,
        |node| matches!(node, ControlNode::Label { block, .. } if *block == outer),
    );
    assert_eq!(labels.len(), 1);
}

#[test]
fn transfer_to_the_follow_of_the_inner_loop_that_is_the_outer_header_breaks_the_inner_loop() {
    // The inner loop's exit is the outer header, so inner loops win: leaving
    // the inner loop is a plain `break`, with no label on either loop.
    let mut flow = Flow::starting_with("i = 0");
    let outer = flow.stmt("i < n");
    let inner_init = flow.stmt("j = 0");
    let inner = flow.stmt("j < m");
    let skip = flow.stmt("skip(i, j)");
    let step_j = flow.stmt("j += 1");
    let exit = flow.stmt("return");
    flow.fall(flow.entry(), outer);
    flow.branch(outer, inner_init, exit);
    flow.fall(inner_init, inner);
    flow.branch(inner, skip, outer);
    flow.branch(skip, outer, step_j);
    flow.jump(step_j, inner);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    assert_eq!(
        nodes_where(&tree, |node| matches!(node, ControlNode::Label { .. })),
        []
    );
    assert_ne!(
        nodes_where(&tree, |node| matches!(
            node,
            ControlNode::Break { label: None }
        )),
        []
    );
}
