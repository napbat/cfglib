//! Switch arms that transfer control: out of the enclosing loop, or into a
//! sibling arm that a goto label has to wrap.

use super::{has_node_kind, pseudocode};
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::flow::FlowEffect;
use crate::test_util::{MockInst, ff};
use crate::{ControlNode, ControlTree};

/// `i = 0; while (i < n) { switch (op[i]) { case 0: leave; case 1: work();
/// default: rest(); } }`, where case 0 jumps straight to the loop's exit.
fn switch_arm_leaving_the_loop() -> Cfg<MockInst> {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let header = cfg.new_block();
    let dispatch = cfg.new_block();
    let exit = cfg.new_block();
    let work = cfg.new_block();
    let rest = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("i = 0"));
    cfg.block_mut(header).push(ff("i < n"));
    cfg.block_mut(dispatch).push(ff("switch op[i]"));
    cfg.block_mut(exit)
        .push(MockInst(FlowEffect::Return, "return"));
    cfg.block_mut(work).push(ff("work()"));
    cfg.block_mut(rest).push(ff("rest()"));
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, dispatch, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(dispatch, exit, EdgeKind::SwitchCase);
    cfg.add_edge(dispatch, work, EdgeKind::SwitchCase);
    cfg.add_edge(dispatch, rest, EdgeKind::Fallthrough);
    cfg.add_edge(work, header, EdgeKind::Jump);
    cfg.add_edge(rest, header, EdgeKind::Jump);
    cfg
}

#[test]
fn case_arm_that_leaves_the_loop_is_a_break_like_the_same_arm_of_an_if() {
    let cfg = switch_arm_leaving_the_loop();
    let (tree, report) = ControlTree::compute_with_report(&cfg);

    assert!(report.is_fully_structured(), "{report:?}\n{tree:?}");
    assert!(!has_node_kind(&tree, |node| matches!(
        node,
        ControlNode::Goto { .. }
    )));
    let breaks = tree
        .preorder()
        .filter(|&id| matches!(tree.node(id), ControlNode::Break { label: None }))
        .count();
    assert_eq!(breaks, 1, "{}", pseudocode(&cfg, &tree));
}

/// A switch whose first case falls into the second: the second arm is reached
/// twice, so its emission is wrapped in a goto label.
fn switch_with_case_fallthrough() -> Cfg<MockInst> {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let first = cfg.new_block();
    let second = cfg.new_block();
    let default_arm = cfg.new_block();
    let join = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("switch k"));
    cfg.block_mut(first).push(ff("a()"));
    cfg.block_mut(second).push(ff("b()"));
    cfg.block_mut(default_arm).push(ff("d()"));
    cfg.block_mut(join)
        .push(MockInst(FlowEffect::Return, "return"));
    cfg.add_edge(cfg.entry(), first, EdgeKind::SwitchCase);
    cfg.add_edge(cfg.entry(), second, EdgeKind::SwitchCase);
    cfg.add_edge(cfg.entry(), default_arm, EdgeKind::Fallthrough);
    cfg.add_edge(first, second, EdgeKind::Fallthrough);
    cfg.add_edge(second, join, EdgeKind::Fallthrough);
    cfg.add_edge(default_arm, join, EdgeKind::Fallthrough);
    cfg
}

#[test]
fn children_are_allocated_before_their_parents_when_goto_labels_are_applied() {
    let cfg = switch_with_case_fallthrough();
    let tree = ControlTree::compute(&cfg);

    assert!(
        has_node_kind(&tree, |node| matches!(node, ControlNode::Label { .. })),
        "the case fallthrough needs a label: {}",
        pseudocode(&cfg, &tree)
    );
    assert_eq!(tree.root().index(), tree.node_count() - 1);
    for id in tree.node_ids() {
        for body in tree.child_bodies(id) {
            for &child in tree.body(body) {
                assert!(child < id, "{child} should precede its parent {id}");
            }
        }
    }
    assert_eq!(tree.preorder().count(), tree.node_count());
}

#[test]
fn a_case_that_leaves_an_endless_loop_is_a_break() {
    // `for (;;) switch (op) { case 0: leave; case 1: work(); default: rest(); }`:
    // no test heads the loop, so its only exit is the dispatch edge of case 0.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let dispatch = cfg.new_block();
    let exit = cfg.new_block();
    let work = cfg.new_block();
    let rest = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("start"));
    cfg.block_mut(dispatch).push(ff("switch op"));
    cfg.block_mut(exit)
        .push(MockInst(FlowEffect::Return, "return"));
    cfg.block_mut(work).push(ff("work()"));
    cfg.block_mut(rest).push(ff("rest()"));
    cfg.add_edge(cfg.entry(), dispatch, EdgeKind::Fallthrough);
    cfg.add_edge(dispatch, exit, EdgeKind::SwitchCase);
    cfg.add_edge(dispatch, work, EdgeKind::SwitchCase);
    cfg.add_edge(dispatch, rest, EdgeKind::Fallthrough);
    cfg.add_edge(work, dispatch, EdgeKind::Jump);
    cfg.add_edge(rest, dispatch, EdgeKind::Jump);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert!(
        report.is_fully_structured(),
        "{report:?}\n{}",
        pseudocode(&cfg, &tree)
    );
    assert!(has_node_kind(&tree, |node| matches!(
        node,
        ControlNode::Break { label: None }
    )));
}
