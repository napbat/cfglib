//! The acyclic shapes a source `if` lowers to: branches, sequences, early
//! returns, and the blocks that hold nothing.

use cfglib::{Cfg, ControlNode, ControlTree, EdgeKind, simplify};

use crate::builders::{Flow, pseudocode};
use crate::oracles::{structure_and_check, structure_and_check_covering};
use crate::reading::{body_nodes, leaf_blocks, shape, top_level};
use crate::tree_shape::assert_fully_structured;

#[test]
fn if_else_diamond_is_one_branch_with_two_arms_followed_by_its_join() {
    let mut flow = Flow::starting_with("n = len(xs)");
    let entry = flow.entry();
    flow.push(entry, "n > 0");
    let then_arm = flow.stmt("y = 1");
    let else_arm = flow.stmt("y = 2");
    let join = flow.stmt("return y");
    flow.branch(entry, then_arm, else_arm);
    flow.fall(then_arm, join);
    flow.fall(else_arm, join);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let statements = top_level(&tree);
    let [
        ControlNode::IfThenElse {
            condition,
            then_body,
            else_body,
        },
        ControlNode::Return(tail),
    ] = statements[..]
    else {
        panic!("expected an if followed by the join: {tree:?}");
    };
    assert_eq!(condition.block(), entry);
    assert_eq!(condition.range(), 0..2, "the test is the last instruction");
    assert_eq!(leaf_blocks(&tree, then_body), [then_arm]);
    assert_eq!(leaf_blocks(&tree, else_body), [else_arm]);
    assert_eq!(tail.block(), join);
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "n = len(xs)\nif {\n    y = 1\n} else {\n    y = 2\n}\nreturn y\n"
    );
}

#[test]
fn if_without_else_leaves_the_else_arm_empty() {
    let mut flow = Flow::starting_with("x > 0");
    let entry = flow.entry();
    let then_arm = flow.stmt("log(x)");
    let join = flow.stmt("return");
    flow.branch(entry, then_arm, join);
    flow.fall(then_arm, join);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let [
        ControlNode::IfThenElse {
            then_body,
            else_body,
            ..
        },
        ControlNode::Return(_),
    ] = top_level(&tree)[..]
    else {
        panic!("expected an if and its join: {tree:?}");
    };
    assert_eq!(leaf_blocks(&tree, then_body), [then_arm]);
    assert!(else_body.is_empty());
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "if {\n    log(x)\n}\nreturn\n"
    );
}

#[test]
fn if_whose_true_arm_is_the_join_keeps_its_work_in_the_else_slot() {
    let mut flow = Flow::starting_with("x > 0");
    let entry = flow.entry();
    let else_arm = flow.stmt("log(x)");
    let join = flow.stmt("return");
    flow.branch(entry, join, else_arm);
    flow.fall(else_arm, join);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let [
        ControlNode::IfThenElse {
            then_body,
            else_body,
            ..
        },
        ControlNode::Return(_),
    ] = top_level(&tree)[..]
    else {
        panic!("expected an if and its join: {tree:?}");
    };
    assert!(then_body.is_empty());
    assert_eq!(leaf_blocks(&tree, else_body), [else_arm]);
}

#[test]
fn else_if_chain_nests_each_test_in_the_previous_else_arm() {
    let mut flow = Flow::starting_with("x < 0");
    let first = flow.entry();
    let second = flow.stmt("x == 0");
    let negative = flow.stmt("sign = -1");
    let zero = flow.stmt("sign = 0");
    let positive = flow.stmt("sign = 1");
    let join = flow.stmt("return sign");
    flow.branch(first, negative, second);
    flow.branch(second, zero, positive);
    for arm in [negative, zero, positive] {
        flow.fall(arm, join);
    }

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let [
        ControlNode::IfThenElse {
            then_body: negative_body,
            else_body: rest,
            ..
        },
        ControlNode::Return(_),
    ] = top_level(&tree)[..]
    else {
        panic!("expected the outer if and the join: {tree:?}");
    };
    assert_eq!(leaf_blocks(&tree, negative_body), [negative]);
    let [
        ControlNode::IfThenElse {
            condition,
            then_body: zero_body,
            else_body: positive_body,
        },
    ] = body_nodes(&tree, rest)[..]
    else {
        panic!("the else arm is the second test alone: {tree:?}");
    };
    assert_eq!(condition.block(), second);
    assert_eq!(leaf_blocks(&tree, zero_body), [zero]);
    assert_eq!(leaf_blocks(&tree, positive_body), [positive]);
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "if {\n    sign = -1\n} else {\n    if {\n        sign = 0\n    } else {\n        sign = 1\n    }\n}\nreturn sign\n"
    );
}

#[test]
fn early_return_arm_leaves_no_merge_so_the_rest_moves_into_the_else_arm() {
    // Two exits mean the branch has no post-dominating merge: each arm runs to
    // its own end, which is the documented shape of a guard clause.
    let mut flow = Flow::starting_with("load(x)");
    let entry = flow.entry();
    flow.push(entry, "x < 0");
    let bail = flow.stmt("return -1");
    let work = flow.stmt("y = f(x)");
    let done = flow.stmt("return y");
    flow.branch(entry, bail, work);
    flow.fall(work, done);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let [
        ControlNode::IfThenElse {
            then_body,
            else_body,
            ..
        },
    ] = top_level(&tree)[..]
    else {
        panic!("a lone branch is the root: {tree:?}");
    };
    assert!(matches!(
        *tree.node(tree.root()),
        ControlNode::IfThenElse { .. }
    ));
    assert_eq!(leaf_blocks(&tree, then_body), [bail]);
    assert_eq!(leaf_blocks(&tree, else_body), [work, done]);
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "load(x)\nif {\n    return -1\n} else {\n    y = f(x)\n    return y\n}\n"
    );
}

#[test]
fn straight_line_blocks_become_a_sequence_of_leaves_ending_in_a_return() {
    let mut flow = Flow::starting_with("a = 1");
    let middle = flow.stmts(&["b = 2", "c = 3"]);
    let last = flow.stmt("return a + b + c");
    flow.fall(flow.entry(), middle);
    flow.fall(middle, last);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let statements = top_level(&tree);
    assert!(matches!(
        statements[..],
        [
            ControlNode::Block(_),
            ControlNode::Block(_),
            ControlNode::Return(_)
        ]
    ));
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "a = 1\nb = 2\nc = 3\nreturn a + b + c\n"
    );
}

#[test]
fn a_single_block_function_is_its_own_root() {
    let flow = Flow::starting_with("return 0");

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    assert_eq!(tree.node_count(), 1);
    assert!(matches!(*tree.node(tree.root()), ControlNode::Return(_)));
    assert!(!tree.is_empty());
}

#[test]
fn a_graph_without_instructions_has_an_empty_tree() {
    let cfg = Cfg::<&'static str>::new();

    let (tree, report) = structure_and_check(&cfg);

    assert!(report.is_fully_structured());
    assert!(tree.is_empty());
    assert_eq!(
        tree.node_count(),
        1,
        "the empty root sequence is the only node"
    );
    assert_eq!(pseudocode(&cfg, &tree), "");
}

#[test]
fn an_empty_join_block_leaves_no_node_and_prints_like_a_graph_without_it() {
    let mut with_join = Flow::starting_with("ok(x)");
    let test = with_join.entry();
    let yes = with_join.stmt("accept()");
    let no = with_join.stmt("reject()");
    let join = with_join.empty();
    let tail = with_join.stmt("return");
    with_join.branch(test, yes, no);
    with_join.fall(yes, join);
    with_join.fall(no, join);
    with_join.fall(join, tail);

    let mut without_join = Flow::starting_with("ok(x)");
    let test = without_join.entry();
    let yes = without_join.stmt("accept()");
    let no = without_join.stmt("reject()");
    let tail = without_join.stmt("return");
    without_join.branch(test, yes, no);
    without_join.fall(yes, tail);
    without_join.fall(no, tail);

    let (tree, report) = structure_and_check(&with_join.cfg);
    let (plain_tree, _) = structure_and_check(&without_join.cfg);
    assert_fully_structured(&tree, &report);

    assert_eq!(
        pseudocode(&with_join.cfg, &tree),
        pseudocode(&without_join.cfg, &plain_tree)
    );
    assert_eq!(shape(&tree), shape(&plain_tree));
}

#[test]
fn unreachable_code_is_not_part_of_the_tree() {
    let mut flow = Flow::starting_with("return 1");
    let dead = flow.stmt("dead()");
    let exit = flow.stmt("return 2");
    flow.fall(dead, exit);
    flow.fall(flow.entry(), exit);

    // The entry reaches the exit; nothing reaches `dead`.
    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let text = pseudocode(&flow.cfg, &tree);
    assert!(!text.contains("dead()"), "{text}");
    assert_eq!(text, "return 1\nreturn 2\n");
}

#[test]
fn simplify_and_compact_keep_the_program_and_the_tree_covers_the_survivors() {
    // entry -> (empty) -> b -> c -> test, an if whose else arm and join hold
    // nothing, and a dead block feeding the tail.
    let mut flow = Flow::starting_with("a = 1");
    let hop = flow.empty();
    let b = flow.stmt("b = 2");
    let c = flow.stmt("c = 3");
    let test = flow.stmt("c > 2");
    let work = flow.stmt("p()");
    let no_else = flow.empty();
    let join = flow.empty();
    let tail = flow.stmt("return");
    let dead = flow.stmt("dead()");
    flow.fall(flow.entry(), hop);
    flow.fall(hop, b);
    flow.fall(b, c);
    flow.fall(c, test);
    flow.branch(test, work, no_else);
    flow.fall(work, join);
    flow.fall(no_else, join);
    flow.fall(join, tail);
    flow.fall(dead, tail);

    let (before, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&before, &report);
    let program = pseudocode(&flow.cfg, &before);
    assert_eq!(program, "a = 1\nb = 2\nc = 3\nif {\n    p()\n}\nreturn\n");

    assert!(simplify(&mut flow.cfg) > 0);
    flow.cfg.compact();

    let live = flow.cfg.block_ids().collect();
    let (after, report) = structure_and_check_covering(&flow.cfg, &live);
    assert_fully_structured(&after, &report);
    assert_eq!(pseudocode(&flow.cfg, &after), program);
    assert!(
        flow.cfg.block_count() < 9,
        "simplify removed blocks, so the tree is over fewer of them"
    );
}

#[test]
fn computing_twice_gives_equal_trees() {
    let mut flow = Flow::starting_with("x > 0");
    let entry = flow.entry();
    let then_arm = flow.stmt("a()");
    let join = flow.stmt("return");
    flow.branch(entry, then_arm, join);
    flow.fall(then_arm, join);

    assert_eq!(
        ControlTree::compute(&flow.cfg),
        ControlTree::compute(&flow.cfg)
    );
}

#[test]
fn edge_payloads_do_not_change_the_tree() {
    // A consumer whose edges carry its own payload structures the same program.
    let mut with_payload = Cfg::<&'static str, u32>::with_edge_payload();
    let entry = with_payload.entry();
    let then_arm = with_payload.new_block();
    let join = with_payload.new_block();
    for (block, statement) in [(entry, "x > 0"), (then_arm, "a()"), (join, "return")] {
        with_payload.block_mut(block).push(statement);
    }
    with_payload.add_edge_with_payload(entry, then_arm, EdgeKind::ConditionalTrue, 10);
    with_payload.add_edge_with_payload(entry, join, EdgeKind::ConditionalFalse, 11);
    with_payload.add_edge_with_payload(then_arm, join, EdgeKind::Fallthrough, 12);

    let mut plain = Flow::starting_with("x > 0");
    let plain_entry = plain.entry();
    let plain_then = plain.stmt("a()");
    let plain_join = plain.stmt("return");
    plain.branch(plain_entry, plain_then, plain_join);
    plain.fall(plain_then, plain_join);

    assert_eq!(
        ControlTree::compute_with_report(&with_payload),
        ControlTree::compute_with_report(&plain.cfg)
    );
}
