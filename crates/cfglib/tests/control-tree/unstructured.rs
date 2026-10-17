//! What the report says when flow cannot be structured: irreducible cycles,
//! loops entered from the side, and transfers that leave a structural bound.

use cfglib::{ControlNode, DominatorTree, GotoReason, detect_loops, is_reducible};

use crate::builders::{Flow, irreducible_pair, pseudocode};
use crate::golden::assert_golden;
use crate::oracles::{structure_and_check, structure_and_check_dangling};
use crate::tree_shape::{
    assert_children_precede_parents, goto_targets_in_tree, label_blocks_in_tree, nodes_where,
};

fn reasons(report: &cfglib::LiftReport) -> Vec<GotoReason> {
    report.gotos.iter().map(|goto| goto.reason).collect()
}

#[test]
fn irreducible_cycle_has_no_loop_and_every_goto_revisits_a_block() {
    let (flow, a, b) = irreducible_pair();
    let dominators = DominatorTree::compute(&flow.cfg);
    assert!(
        !is_reducible(&flow.cfg, &dominators),
        "the fixture is irreducible"
    );
    assert!(
        detect_loops(&flow.cfg, &dominators).is_empty(),
        "neither block dominates the other, so there is no natural loop"
    );

    let (tree, report) = structure_and_check(&flow.cfg);

    assert!(!report.is_fully_structured());
    assert!(
        nodes_where(&tree, |node| matches!(node, ControlNode::Loop { .. })).is_empty(),
        "{tree:?}"
    );
    assert_eq!(
        reasons(&report),
        [GotoReason::RevisitedTarget, GotoReason::RevisitedTarget]
    );
    let mut targets = goto_targets_in_tree(&tree);
    targets.sort();
    assert_eq!(targets, [a, b]);
    assert_eq!(report.unresolved_labels, []);
    assert_golden(
        "control-tree/irreducible_cycle.pseudo",
        &pseudocode(&flow.cfg, &tree),
    );
}

#[test]
fn loop_entered_in_its_body_is_irreducible_and_reported_as_gotos() {
    // start enters the loop at its header `h` or, from the side, at the body
    // block `x`: the cycle h -> x -> h has two entries.
    let mut flow = Flow::starting_with("start");
    let h = flow.stmt("i < n");
    let x = flow.stmt("work()");
    let exit = flow.stmt("return");
    flow.branch(flow.entry(), h, x);
    flow.branch(h, x, exit);
    flow.jump(x, h);
    let dominators = DominatorTree::compute(&flow.cfg);
    assert!(!is_reducible(&flow.cfg, &dominators));

    let (tree, report) = structure_and_check(&flow.cfg);

    assert!(!report.is_fully_structured());
    assert_eq!(
        nodes_where(&tree, |node| matches!(node, ControlNode::Loop { .. })),
        []
    );
    assert_eq!(reasons(&report), [GotoReason::RevisitedTarget]);
    assert_eq!(
        report.gotos[0].target, x,
        "the side-entered block is the goto target"
    );
    assert_eq!(label_blocks_in_tree(&tree), [x]);
    assert_golden(
        "control-tree/multi_entry_loop.pseudo",
        &pseudocode(&flow.cfg, &tree),
    );
}

#[test]
fn jump_out_of_a_loop_to_a_shared_tail_is_an_explicit_jump() {
    // while i < n { if bad(i) { cleanup(); goto fail } i += 1 }; done(); fail: return
    let mut flow = Flow::starting_with("i = 0");
    let header = flow.stmt("i < n");
    let bad = flow.stmt("bad(i)");
    let step = flow.stmt("i += 1");
    let after = flow.stmt("done()");
    let cleanup = flow.stmt("cleanup()");
    let tail = flow.stmt("return");
    flow.fall(flow.entry(), header);
    flow.branch(header, bad, after);
    flow.branch(bad, cleanup, step);
    flow.jump(step, header);
    flow.fall(after, tail);
    flow.jump(cleanup, tail);

    let (tree, report) = structure_and_check(&flow.cfg);

    assert!(!report.is_fully_structured());
    assert_eq!(reasons(&report), [GotoReason::ExplicitJump]);
    assert_eq!(report.gotos[0].target, tail);
    assert_eq!(label_blocks_in_tree(&tree), [tail]);
    assert!(
        report.swept_blocks.is_empty(),
        "the walk reaches the tail itself"
    );
    assert_golden(
        "control-tree/jump_to_shared_tail.pseudo",
        &pseudocode(&flow.cfg, &tree),
    );
}

#[test]
fn conditional_exit_to_a_shared_block_leaves_the_loop_bound() {
    // Two loop-body tests both exit to `fail`, which is not the loop's follow
    // and is not exclusively owned: the walk refuses both, and the sweep
    // emits the block under its label.
    let mut flow = Flow::starting_with("i = 0");
    let header = flow.stmt("i < n");
    let first = flow.stmt("bad_a(i)");
    let second = flow.stmt("bad_b(i)");
    let step = flow.stmt("i += 1");
    let after = flow.stmt("return 0");
    let fail = flow.stmt("return -1");
    flow.fall(flow.entry(), header);
    flow.branch(header, first, after);
    flow.branch(first, fail, second);
    flow.branch(second, fail, step);
    flow.jump(step, header);

    let (tree, report) = structure_and_check(&flow.cfg);

    assert!(!report.is_fully_structured());
    assert_eq!(
        reasons(&report),
        [GotoReason::BoundaryEscape, GotoReason::BoundaryEscape]
    );
    assert_eq!(goto_targets_in_tree(&tree), [fail, fail]);
    assert_eq!(report.swept_blocks, [fail]);
    assert_eq!(label_blocks_in_tree(&tree), [fail]);
    assert_eq!(report.unresolved_labels, []);
    assert_golden(
        "control-tree/loop_exit_to_shared_block.pseudo",
        &pseudocode(&flow.cfg, &tree),
    );
}

#[test]
fn jump_to_exclusively_owned_code_continues_in_sequence_without_a_goto() {
    let mut flow = Flow::starting_with("a()");
    let next = flow.stmt("b()");
    let last = flow.stmt("return");
    flow.jump(flow.entry(), next);
    flow.jump(next, last);

    let (tree, report) = structure_and_check(&flow.cfg);

    assert!(report.is_fully_structured(), "{report:?}");
    assert_eq!(pseudocode(&flow.cfg, &tree), "a()\nb()\nreturn\n");
}

#[test]
fn goto_to_an_empty_block_has_no_node_to_label_and_is_reported_unresolved() {
    // As in the shared-tail shape, but the tail begins with an empty block:
    // the label has no node to attach to, so the report says the goto dangles.
    let mut flow = Flow::starting_with("i = 0");
    let header = flow.stmt("i < n");
    let bad = flow.stmt("bad(i)");
    let step = flow.stmt("i += 1");
    let after = flow.stmt("done()");
    let cleanup = flow.stmt("cleanup()");
    let landing = flow.empty();
    let tail = flow.stmt("return");
    flow.fall(flow.entry(), header);
    flow.branch(header, bad, after);
    flow.branch(bad, cleanup, step);
    flow.jump(step, header);
    flow.fall(after, landing);
    flow.jump(cleanup, landing);
    flow.fall(landing, tail);

    let (tree, report) = structure_and_check_dangling(&flow.cfg);

    assert!(!report.is_fully_structured());
    assert_eq!(report.gotos.len(), 1);
    assert_eq!(report.gotos[0].target, landing);
    assert_eq!(report.unresolved_labels, [landing]);
    assert_eq!(label_blocks_in_tree(&tree), []);
    assert_eq!(goto_targets_in_tree(&tree), [landing]);
}

#[test]
fn children_are_allocated_before_their_parents_when_goto_labels_are_applied() {
    let (flow, _, _) = irreducible_pair();
    let (tree, report) = structure_and_check(&flow.cfg);
    assert!(!report.gotos.is_empty(), "the fixture needs labels");

    assert_children_precede_parents(&tree);
}
