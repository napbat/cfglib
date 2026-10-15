//! The structure decisions of the control-tree structurer.
//!
//! These cases are the oracle ported from the structurer that built an
//! instruction-carrying tree: the same graphs, the same expected shapes. The
//! only changes are the tree vocabulary and the cases that document how exit-free
//! regions differ (`exit_free`).

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::{
    BlockSpan, CatchHandler, ControlBody, ControlNode, ControlNodeId, ControlRun, ControlTree,
    GotoReason, LiftReport, LoopKind,
};
use crate::block::BlockId;
use crate::builder::CfgBuilder;
use crate::cfg::Cfg;
use crate::display::DisplayInstr;
use crate::edge::EdgeKind;
use crate::flow::FlowEffect;
use crate::test_util::{MockInst, ff};

mod dead_handlers;
mod exhaustive;
mod exit_free;
mod invariants;
mod predicated;
mod regions;
mod traversal;
mod treating_empty;

/// `id` and every node below it, in preorder.
fn subtree(tree: &ControlTree, id: ControlNodeId) -> Vec<ControlNodeId> {
    let mut nodes = vec![id];
    for body in tree.child_bodies(id) {
        for &child in tree.body(body) {
            nodes.extend(subtree(tree, child));
        }
    }
    nodes
}

/// Every node below the statements of `body`, in preorder.
fn body_subtree(tree: &ControlTree, body: ControlBody) -> Vec<ControlNodeId> {
    tree.body(body)
        .iter()
        .flat_map(|&child| subtree(tree, child))
        .collect()
}

/// Whether any node of the tree matches `predicate`.
fn has_node_kind(tree: &ControlTree, predicate: impl Fn(&ControlNode) -> bool) -> bool {
    tree.preorder().any(|id| predicate(tree.node(id)))
}

fn leaf_block(node: &ControlNode) -> Option<BlockId> {
    match node {
        ControlNode::Block(span) | ControlNode::Return(span) => Some(span.block()),
        _ => None,
    }
}

/// Whether a `Block` or `Return` leaf names `block` anywhere in the tree.
fn contains_block(tree: &ControlTree, block: BlockId) -> bool {
    has_node_kind(tree, |node| leaf_block(node) == Some(block))
}

/// Whether a `Block` or `Return` leaf names `block` below `body`.
fn body_contains_block(tree: &ControlTree, body: ControlBody, block: BlockId) -> bool {
    body_subtree(tree, body)
        .into_iter()
        .any(|id| leaf_block(tree.node(id)) == Some(block))
}

fn has_goto_to(tree: &ControlTree, block: BlockId) -> bool {
    has_node_kind(
        tree,
        |node| matches!(node, ControlNode::Goto { target } if *target == block),
    )
}

/// The first try region of the tree, in preorder.
fn find_try_catch(
    tree: &ControlTree,
) -> Option<(ControlBody, ControlRun<CatchHandler>, ControlBody)> {
    tree.preorder().find_map(|id| match *tree.node(id) {
        ControlNode::TryCatch {
            try_body,
            handlers,
            finally_body,
        } => Some((try_body, handlers, finally_body)),
        _ => None,
    })
}

fn find_try_handlers(tree: &ControlTree) -> Option<Vec<CatchHandler>> {
    find_try_catch(tree).map(|(_, handlers, _)| tree.handlers(handlers).to_vec())
}

/// Every loop of the tree, in preorder.
fn find_loops(tree: &ControlTree) -> Vec<(BlockId, LoopKind, ControlBody)> {
    tree.preorder()
        .filter_map(|id| match *tree.node(id) {
            ControlNode::Loop { header, kind, body } => Some((header, kind, body)),
            _ => None,
        })
        .collect()
}

/// The pseudocode of `tree`, printing each instruction by its mnemonic.
fn pseudocode<I: DisplayInstr, E>(cfg: &Cfg<I, E>, tree: &ControlTree) -> alloc::string::String {
    tree.to_pseudocode(|sink, point| {
        cfg.block(point.block).instructions()[point.inst_idx].write_mnemonic(sink)
    })
    .expect("writing pseudocode to a String cannot fail")
}

/// A machine-shaped while loop:
/// entry → header{cond}; header --true--> body --fallthrough--> header;
/// header --false--> exit{return}.
fn while_shaped_cfg() -> (Cfg<MockInst>, BlockId, BlockId, BlockId) {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let header = cfg.new_block();
    let body = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(header).push(ff("condition"));
    cfg.block_mut(body).push(ff("body_inst"));
    cfg.block_mut(exit)
        .push(MockInst(FlowEffect::Return, "after_loop"));
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, header, EdgeKind::Jump);
    (cfg, header, body, exit)
}

#[test]
fn machine_while_loop_is_classified_pre_tested() {
    let (cfg, header, body, exit) = while_shaped_cfg();
    let (tree, report) = ControlTree::compute_with_report(&cfg);
    let loops = find_loops(&tree);
    assert_eq!(loops.len(), 1, "{tree:?}");
    let (found_header, kind, loop_body) = loops[0];
    assert_eq!(found_header, header);
    let LoopKind::While { exit_on_true, .. } = kind else {
        panic!("expected a pre-tested loop, got {kind:?}");
    };
    assert!(!exit_on_true, "the true edge iterates: {tree:?}");
    assert!(body_contains_block(&tree, loop_body, body));
    assert!(
        !body_contains_block(&tree, loop_body, exit),
        "the loop exit may not be absorbed into the body: {tree:?}"
    );
    assert!(
        contains_block(&tree, exit),
        "the continuation follows the loop: {tree:?}"
    );
    assert!(report.is_fully_structured(), "{report:?}");
}

#[test]
fn exit_on_true_while_loop_is_classified_pre_tested() {
    // entry → init → header{cond}; --true--> exit; --false--> body;
    // body --Jump--> header. The taken branch leaves the loop.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let init = cfg.new_block();
    let header = cfg.new_block();
    let body = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(init).push(ff("init"));
    cfg.block_mut(header).push(ff("condition"));
    cfg.block_mut(body).push(ff("body_inst"));
    cfg.block_mut(exit)
        .push(MockInst(FlowEffect::Return, "after_loop"));
    cfg.add_edge(cfg.entry(), init, EdgeKind::Fallthrough);
    cfg.add_edge(init, header, EdgeKind::Fallthrough);
    cfg.add_edge(header, exit, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, body, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, header, EdgeKind::Jump);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    let loops = find_loops(&tree);
    assert_eq!(loops.len(), 1, "{tree:?}");
    let (_, kind, loop_body) = loops[0];
    let LoopKind::While { exit_on_true, .. } = kind else {
        panic!("expected a pre-tested loop, got {kind:?}");
    };
    assert!(exit_on_true, "the true edge leaves the loop: {tree:?}");
    assert!(body_contains_block(&tree, loop_body, body));
    assert!(!body_contains_block(&tree, loop_body, exit), "{tree:?}");
    assert!(report.is_fully_structured(), "{report:?}");
}

#[test]
fn multi_block_condition_chain_is_classified_pre_tested() {
    // Per-instruction blocks, the shape bytecode frontends produce:
    // header[t = i] → [u = n] → cond{t < u}; --false--> exit; --true-->
    // body → latch --Jump--> header.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let header = cfg.new_block();
    let compare_right = cfg.new_block();
    let condition = cfg.new_block();
    let body = cfg.new_block();
    let latch = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(header).push(ff("load_left"));
    cfg.block_mut(compare_right).push(ff("load_right"));
    cfg.block_mut(condition).push(ff("compare"));
    cfg.block_mut(body).push(ff("body_inst"));
    cfg.block_mut(latch).push(ff("increment"));
    cfg.block_mut(exit)
        .push(MockInst(FlowEffect::Return, "after_loop"));
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, compare_right, EdgeKind::Fallthrough);
    cfg.add_edge(compare_right, condition, EdgeKind::Fallthrough);
    cfg.add_edge(condition, exit, EdgeKind::ConditionalTrue);
    cfg.add_edge(condition, body, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, latch, EdgeKind::Fallthrough);
    cfg.add_edge(latch, header, EdgeKind::Jump);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    let loops = find_loops(&tree);
    assert_eq!(loops.len(), 1, "{tree:?}");
    let (found_header, kind, loop_body) = loops[0];
    assert_eq!(found_header, header);
    let LoopKind::While {
        condition: chain,
        exit_on_true,
    } = kind
    else {
        panic!("expected a chained pre-tested loop, got {kind:?}");
    };
    assert!(exit_on_true, "{tree:?}");
    let chain = tree.spans(chain);
    let blocks: Vec<BlockId> = chain.iter().map(|span| span.block()).collect();
    assert_eq!(blocks, [header, compare_right, condition], "{tree:?}");
    let mut names = Vec::new();
    for span in chain {
        assert_eq!(span.len(), 1);
        names.extend(span.instructions(&cfg).iter().map(|inst| inst.1));
    }
    assert_eq!(names, ["load_left", "load_right", "compare"], "{tree:?}");
    assert!(body_contains_block(&tree, loop_body, body));
    assert!(body_contains_block(&tree, loop_body, latch));
    assert!(!body_contains_block(&tree, loop_body, exit), "{tree:?}");
    assert!(
        contains_block(&tree, exit),
        "the continuation follows the loop: {tree:?}"
    );
    assert!(report.is_fully_structured(), "{report:?}");
}

#[test]
fn machine_do_while_loop_is_classified_post_tested() {
    // entry → body; body → latch{cond}; latch --true--> body; --false--> exit.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let body = cfg.new_block();
    let latch = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(body).push(ff("body_inst"));
    cfg.block_mut(latch).push(ff("latch_cond"));
    cfg.block_mut(exit)
        .push(MockInst(FlowEffect::Return, "after_loop"));
    cfg.add_edge(cfg.entry(), body, EdgeKind::Fallthrough);
    cfg.add_edge(body, latch, EdgeKind::Fallthrough);
    cfg.add_edge(latch, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(latch, exit, EdgeKind::ConditionalFalse);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    let loops = find_loops(&tree);
    assert_eq!(loops.len(), 1, "{tree:?}");
    let (_, kind, loop_body) = loops[0];
    let LoopKind::DoWhile {
        latch: found_latch,
        continue_on_true,
    } = kind
    else {
        panic!("expected a post-tested loop, got {kind:?}");
    };
    assert_eq!(found_latch.block(), latch);
    assert_eq!(found_latch, BlockSpan::new(latch, 0, 1));
    assert!(continue_on_true, "{tree:?}");
    assert!(body_contains_block(&tree, loop_body, body));
    assert!(
        !body_contains_block(&tree, loop_body, latch),
        "the latch is the condition, not body: {tree:?}"
    );
    assert!(contains_block(&tree, exit), "{tree:?}");
    assert!(report.is_fully_structured(), "{report:?}");
}

#[test]
fn loop_exit_from_the_body_becomes_a_break() {
    // while-shaped loop whose body also exits conditionally (a break).
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let header = cfg.new_block();
    let body = cfg.new_block();
    let tail = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(header).push(ff("condition"));
    cfg.block_mut(body).push(ff("maybe_break"));
    cfg.block_mut(tail).push(ff("tail_inst"));
    cfg.block_mut(exit)
        .push(MockInst(FlowEffect::Return, "after_loop"));
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, exit, EdgeKind::ConditionalTrue);
    cfg.add_edge(body, tail, EdgeKind::ConditionalFalse);
    cfg.add_edge(tail, header, EdgeKind::Jump);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert!(
        has_node_kind(&tree, |node| matches!(
            node,
            ControlNode::Break { label: None }
        )),
        "the in-body exit resolves to a break: {tree:?}"
    );
    assert!(
        !has_node_kind(&tree, |node| matches!(node, ControlNode::Goto { .. })),
        "{tree:?}"
    );
    assert!(report.is_fully_structured(), "{report:?}");
}

#[test]
fn multi_level_exit_becomes_a_labeled_break() {
    // outer: header_o conditional-exits; inner loop's body exits BOTH loops.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let outer_header = cfg.new_block();
    let inner_header = cfg.new_block();
    let inner_body = cfg.new_block();
    let outer_latch = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(outer_header).push(ff("outer_cond"));
    cfg.block_mut(inner_header).push(ff("inner_cond"));
    cfg.block_mut(inner_body).push(ff("inner_inst"));
    cfg.block_mut(outer_latch).push(ff("outer_latch_inst"));
    cfg.block_mut(exit)
        .push(MockInst(FlowEffect::Return, "after"));
    cfg.add_edge(cfg.entry(), outer_header, EdgeKind::Fallthrough);
    cfg.add_edge(outer_header, inner_header, EdgeKind::ConditionalTrue);
    cfg.add_edge(outer_header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(inner_header, inner_body, EdgeKind::ConditionalTrue);
    cfg.add_edge(inner_header, outer_latch, EdgeKind::ConditionalFalse);
    // The inner body either exits everything or continues the inner loop.
    cfg.add_edge(inner_body, exit, EdgeKind::ConditionalTrue);
    cfg.add_edge(inner_body, inner_header, EdgeKind::ConditionalFalse);
    cfg.add_edge(outer_latch, outer_header, EdgeKind::Jump);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert!(
        has_node_kind(&tree, |node| matches!(
            node,
            ControlNode::Break { label: Some(label) } if *label == outer_header
        )),
        "the two-level exit is a labeled break out of the outer loop: {tree:?}"
    );
    assert!(
        has_node_kind(&tree, |node| matches!(
            node,
            ControlNode::Label { block, .. } if *block == outer_header
        )),
        "the outer loop carries the label: {tree:?}"
    );
    assert!(
        !has_node_kind(&tree, |node| matches!(node, ControlNode::Goto { .. })),
        "{tree:?}"
    );
    assert!(report.is_fully_structured(), "{report:?}");
}

#[test]
fn if_else_merge_is_not_absorbed_into_an_arm() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let then_arm = cfg.new_block();
    let else_arm = cfg.new_block();
    let merge = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("cond"));
    cfg.block_mut(then_arm).push(ff("then_inst"));
    cfg.block_mut(else_arm).push(ff("else_inst"));
    cfg.block_mut(merge)
        .push(MockInst(FlowEffect::Return, "merge_inst"));
    cfg.add_edge(cfg.entry(), then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), else_arm, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, merge, EdgeKind::Fallthrough);
    cfg.add_edge(else_arm, merge, EdgeKind::Fallthrough);

    let tree = ControlTree::compute(&cfg);
    let mut checked = false;
    for id in tree.preorder() {
        if let ControlNode::IfThenElse {
            then_body,
            else_body,
            ..
        } = *tree.node(id)
        {
            checked = true;
            assert!(body_contains_block(&tree, then_body, then_arm));
            assert!(body_contains_block(&tree, else_body, else_arm));
            assert!(
                !body_contains_block(&tree, then_body, merge)
                    && !body_contains_block(&tree, else_body, merge),
                "the merge belongs after the conditional, not in an arm: {tree:?}"
            );
        }
    }
    assert!(checked, "expected an IfThenElse: {tree:?}");
    assert!(
        contains_block(&tree, merge),
        "the merge follows the conditional: {tree:?}"
    );
}

#[test]
fn switch_records_case_edges_and_default_arm() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let case_a = cfg.new_block();
    let case_b = cfg.new_block();
    let default_arm = cfg.new_block();
    let merge = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("dispatch"));
    cfg.block_mut(case_a).push(ff("case_a_inst"));
    cfg.block_mut(case_b).push(ff("case_b_inst"));
    cfg.block_mut(default_arm).push(ff("default_inst"));
    cfg.block_mut(merge)
        .push(MockInst(FlowEffect::Return, "after"));
    let edge_a = cfg.add_edge(cfg.entry(), case_a, EdgeKind::SwitchCase);
    let edge_a2 = cfg.add_edge(cfg.entry(), case_a, EdgeKind::SwitchCase);
    let edge_b = cfg.add_edge(cfg.entry(), case_b, EdgeKind::SwitchCase);
    let default_edge = cfg.add_edge(cfg.entry(), default_arm, EdgeKind::Fallthrough);
    cfg.add_edge(case_a, merge, EdgeKind::Unconditional);
    cfg.add_edge(case_b, merge, EdgeKind::Unconditional);
    cfg.add_edge(default_arm, merge, EdgeKind::Unconditional);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    let mut checked = false;
    for id in tree.preorder() {
        if let ControlNode::Switch {
            cases,
            default_body,
            default_edge: found_default,
            ..
        } = *tree.node(id)
        {
            checked = true;
            let cases = tree.cases(cases);
            assert_eq!(cases.len(), 2, "{tree:?}");
            assert_eq!(cases[0].block, case_a);
            assert_eq!(tree.edges(cases[0].edges), [edge_a, edge_a2], "{tree:?}");
            assert_eq!(tree.edges(cases[1].edges), [edge_b], "{tree:?}");
            assert_eq!(found_default, Some(default_edge), "{tree:?}");
            assert!(
                body_contains_block(&tree, default_body, default_arm),
                "the default arm is captured: {tree:?}"
            );
        }
    }
    assert!(checked, "expected a Switch: {tree:?}");
    assert!(contains_block(&tree, merge), "{tree:?}");
    assert!(report.is_fully_structured(), "{report:?}");
}

#[test]
fn compute_accepts_caller_owned_edge_payloads() {
    let mut cfg = Cfg::<MockInst, &'static str>::with_edge_payload();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(exit)
        .push(MockInst(FlowEffect::Return, "return"));
    cfg.add_edge_with_payload(cfg.entry(), exit, EdgeKind::Fallthrough, "source identity");

    let tree = ControlTree::compute(&cfg);
    let pseudo = pseudocode(&cfg, &tree);
    assert!(pseudo.contains("entry"), "{pseudo}");
    assert!(pseudo.contains("return"), "{pseudo}");
}

#[test]
fn compute_recovers_a_payload_bearing_loop_from_ordinary_jump_edges() {
    let mut cfg = Cfg::<MockInst, &'static str>::with_edge_payload();
    let header = cfg.new_block();
    let body = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(header).push(ff("condition"));
    cfg.block_mut(body).push(ff("body"));
    cfg.block_mut(exit)
        .push(MockInst(FlowEffect::Return, "return"));
    cfg.add_edge_with_payload(cfg.entry(), header, EdgeKind::Fallthrough, "entry");
    cfg.add_edge_with_payload(header, body, EdgeKind::ConditionalTrue, "true");
    cfg.add_edge_with_payload(header, exit, EdgeKind::ConditionalFalse, "false");
    cfg.add_edge_with_payload(body, header, EdgeKind::Jump, "native goto");

    let tree = ControlTree::compute(&cfg);
    assert!(
        has_node_kind(&tree, |node| matches!(node, ControlNode::Loop { .. })),
        "dominance must recover a loop without rewriting its native jump edge: {tree:?}"
    );
    assert!(
        !has_node_kind(&tree, |node| matches!(
            node,
            ControlNode::Label { .. } | ControlNode::Goto { .. }
        )),
        "the natural loop should not degrade to labels and gotos: {tree:?}"
    );
}

fn lift_pseudo(insts: Vec<MockInst>) -> alloc::string::String {
    let cfg = CfgBuilder::build(insts).unwrap();
    let tree = ControlTree::compute(&cfg);
    pseudocode(&cfg, &tree)
}

#[test]
fn lift_linear() {
    let p = lift_pseudo(vec![
        ff("a"),
        ff("b"),
        ff("c"),
        MockInst(FlowEffect::Return, "ret"),
    ]);
    assert!(p.contains('a'), "should contain instruction a: {p}");
    assert!(p.contains("ret"), "should contain ret: {p}");
    // No control flow keywords.
    assert!(!p.contains("if"), "no if expected: {p}");
    assert!(!p.contains("loop"), "no loop expected: {p}");
}

#[test]
fn lift_if_no_else() {
    let p = lift_pseudo(vec![
        ff("a"),
        MockInst(FlowEffect::ConditionalOpen, "if"),
        ff("b"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
        ff("c"),
        MockInst(FlowEffect::Return, "ret"),
    ]);
    assert!(p.contains("if {"), "should have if: {p}");
    assert!(p.contains('b'), "then body should contain b: {p}");
    assert!(p.contains('c'), "post-merge should contain c: {p}");
}

#[test]
fn lift_if_else() {
    let p = lift_pseudo(vec![
        ff("a"),
        MockInst(FlowEffect::ConditionalOpen, "if"),
        ff("then_inst"),
        MockInst(FlowEffect::ConditionalAlternate, "else"),
        ff("else_inst"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
        ff("merge"),
        MockInst(FlowEffect::Return, "ret"),
    ]);
    assert!(p.contains("if {"), "should have if: {p}");
    assert!(p.contains("then_inst"), "then arm: {p}");
    assert!(
        p.contains("else_inst") || p.contains("} else {"),
        "else arm: {p}"
    );
}

#[test]
fn lift_simple_loop() {
    let p = lift_pseudo(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        ff("body"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Return, "ret"),
    ]);
    assert!(p.contains("body"), "loop body: {p}");
    assert!(
        p.contains("loop {") || p.contains("while {") || p.contains("do {"),
        "should have a loop: {p}"
    );
}

#[test]
fn lift_loop_with_break() {
    let p = lift_pseudo(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        ff("a"),
        MockInst(FlowEffect::ConditionalBreak, "breakc"),
        ff("b"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Return, "ret"),
    ]);
    assert!(p.contains('a'), "should contain a: {p}");
    assert!(p.contains("ret"), "the continuation survives: {p}");
}

#[test]
fn lift_switch() {
    let p = lift_pseudo(vec![
        MockInst(FlowEffect::SwitchOpen, "switch"),
        ff("dispatch"),
        MockInst(FlowEffect::SwitchCase, "case0"),
        ff("arm0"),
        MockInst(FlowEffect::SwitchCase, "case1"),
        ff("arm1"),
        MockInst(FlowEffect::SwitchClose, "endswitch"),
        ff("after"),
        MockInst(FlowEffect::Return, "ret"),
    ]);
    assert!(p.contains("switch {"), "should have switch: {p}");
    assert!(p.contains("case {"), "should have case: {p}");
    assert!(p.contains("arm0"), "{p}");
    assert!(p.contains("arm1"), "{p}");
    assert!(p.contains("after"), "{p}");
}

#[test]
fn lift_if_in_loop() {
    let p = lift_pseudo(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        MockInst(FlowEffect::ConditionalOpen, "if"),
        ff("then"),
        MockInst(FlowEffect::ConditionalAlternate, "else"),
        ff("else_body"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Return, "ret"),
    ]);
    assert!(p.contains("if {"), "should have if inside loop: {p}");
    assert!(p.contains("then"), "{p}");
}

#[test]
fn lift_loop_in_if() {
    let p = lift_pseudo(vec![
        MockInst(FlowEffect::ConditionalOpen, "if"),
        MockInst(FlowEffect::LoopOpen, "loop"),
        ff("body"),
        MockInst(FlowEffect::ConditionalBreak, "breakc"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
        MockInst(FlowEffect::Return, "ret"),
    ]);
    let has_if = p.contains("if {");
    let has_loop = p.contains("loop {") || p.contains("while {") || p.contains("do {");
    assert!(has_if || has_loop, "should have nested structure: {p}");
}

#[test]
fn compute_returns_sequence_or_single() {
    let cfg = CfgBuilder::build(vec![ff("a"), MockInst(FlowEffect::Return, "ret")]).unwrap();
    let tree = ControlTree::compute(&cfg);
    assert!(!tree.is_empty(), "should not be empty");
}

#[test]
fn lift_conditional_produces_if_node() {
    let cfg = CfgBuilder::build(vec![
        ff("a"),
        MockInst(FlowEffect::ConditionalOpen, "if"),
        ff("b"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    let tree = ControlTree::compute(&cfg);
    let found = has_node_kind(&tree, |n| matches!(n, ControlNode::IfThenElse { .. }));
    assert!(found, "should contain IfThenElse node: {tree:?}");
}

#[test]
fn lift_loop_produces_loop_node() {
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        ff("x"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    let tree = ControlTree::compute(&cfg);
    let found = has_node_kind(&tree, |n| matches!(n, ControlNode::Loop { .. }));
    assert!(found, "should contain Loop node: {tree:?}");
}

#[test]
fn lift_switch_produces_switch_node() {
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::SwitchOpen, "switch"),
        ff("d"),
        MockInst(FlowEffect::SwitchCase, "c1"),
        ff("a1"),
        MockInst(FlowEffect::SwitchCase, "c2"),
        ff("a2"),
        MockInst(FlowEffect::SwitchClose, "endswitch"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    let tree = ControlTree::compute(&cfg);
    let found = has_node_kind(&tree, |n| matches!(n, ControlNode::Switch { .. }));
    assert!(found, "should contain Switch node: {tree:?}");
}

#[test]
fn lift_jump_edge_into_an_owned_target_continues_inline() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    // entry(0) --Jump--> target(1): the jump is the target's only entry
    // and no region claims it, so the walk continues there directly.
    let target = cfg.new_block();
    cfg.block_mut(cfg.entry())
        .instructions_mut()
        .push(ff("src"));
    cfg.block_mut(target).instructions_mut().push(ff("dst"));

    cfg.add_edge(cfg.entry(), target, EdgeKind::Jump);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert!(
        !has_node_kind(&tree, |n| matches!(n, ControlNode::Goto { .. })),
        "should continue inline: {tree:?}"
    );
    assert!(report.is_fully_structured(), "{report:?}");
}

#[test]
fn lift_jump_edge_produces_goto_and_reports_it() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    // entry(0) --Jump--> target(1), but an exception region claims the
    // target: region-claimed code never inlines across a jump, so the
    // transfer stays explicit.
    let target = cfg.new_block();
    let pad = cfg.new_block();
    cfg.block_mut(cfg.entry())
        .instructions_mut()
        .push(ff("src"));
    cfg.block_mut(target).instructions_mut().push(ff("dst"));
    cfg.block_mut(pad).instructions_mut().push(ff("pad"));

    cfg.add_edge(cfg.entry(), target, EdgeKind::Jump);
    cfg.add_region(crate::Region {
        id: crate::RegionId::from_raw(0),
        protected_blocks: [target].into_iter().collect(),
        handlers: vec![crate::Handler {
            entry: pad,
            body: crate::HandlerBody::known([pad]),
            kind: crate::HandlerKind::CatchAll,
        }],
        parent: None,
    });

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert!(
        has_node_kind(&tree, |n| matches!(n, ControlNode::Goto { .. })),
        "should contain Goto node: {tree:?}"
    );
    let pseudo = pseudocode(&cfg, &tree);
    assert!(pseudo.contains("goto"), "{pseudo}");
    assert_eq!(report.gotos.len(), 1, "{report:?}");
    assert_eq!(report.gotos[0].target, target);
    assert_eq!(report.gotos[0].reason, GotoReason::ExplicitJump);
    assert!(!report.is_fully_structured());
}

#[test]
fn lift_jump_to_the_merge_converges_silently() {
    // The jump lands exactly on the conditional's merge: it is the arm
    // ending, not goto residue.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let normal = cfg.new_block(); // 1
    let target = cfg.new_block(); // 2
    let end = cfg.new_block(); // 3
    let jumper = cfg.new_block(); // 4
    cfg.block_mut(cfg.entry())
        .instructions_mut()
        .push(ff("entry"));
    cfg.block_mut(normal).instructions_mut().push(ff("normal"));
    cfg.block_mut(target).instructions_mut().push(ff("dst"));
    cfg.block_mut(end).instructions_mut().push(ff("end"));
    cfg.block_mut(jumper).instructions_mut().push(ff("jumper"));

    cfg.add_edge(cfg.entry(), normal, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), jumper, EdgeKind::ConditionalFalse);
    cfg.add_edge(normal, target, EdgeKind::Fallthrough);
    cfg.add_edge(jumper, target, EdgeKind::Jump);
    cfg.add_edge(target, end, EdgeKind::Fallthrough);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert!(
        report.is_fully_structured(),
        "the join jump is structural: {report:?}"
    );
    assert!(
        !has_node_kind(&tree, |n| matches!(n, ControlNode::Label { .. })),
        "no label is needed: {tree:?}"
    );
}

#[test]
fn lift_jump_target_gets_label() {
    // The jump crosses into a sibling arm's interior with no shared merge
    // (both tails return), so it stays an explicit labeled goto.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let jumper = cfg.new_block(); // 1
    let normal = cfg.new_block(); // 2
    let target = cfg.new_block(); // 3
    let other = cfg.new_block(); // 4
    cfg.block_mut(cfg.entry())
        .instructions_mut()
        .push(ff("entry"));
    cfg.block_mut(jumper).instructions_mut().push(ff("jumper"));
    cfg.block_mut(normal).instructions_mut().push(ff("normal"));
    cfg.block_mut(target).instructions_mut().push(ff("dst"));
    cfg.block_mut(other).instructions_mut().push(ff("other"));

    cfg.add_edge(cfg.entry(), jumper, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), normal, EdgeKind::ConditionalFalse);
    cfg.add_edge(jumper, target, EdgeKind::Jump);
    cfg.add_edge(normal, target, EdgeKind::ConditionalTrue);
    cfg.add_edge(normal, other, EdgeKind::ConditionalFalse);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert!(
        has_node_kind(&tree, |n| matches!(
            n,
            ControlNode::Label { block, .. } if *block == target
        )),
        "should contain Label node: {tree:?}"
    );
    assert!(
        report.unresolved_labels.is_empty(),
        "the goto target is anchored: {report:?}"
    );
}

#[test]
fn structured_lifts_report_no_degradation() {
    let cfg = CfgBuilder::build(vec![
        ff("a"),
        MockInst(FlowEffect::ConditionalOpen, "if"),
        ff("b"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    let (_, report) = ControlTree::compute_with_report(&cfg);
    assert_eq!(report, LiftReport::default(), "{report:?}");
}

#[test]
fn tree_does_not_borrow_or_clone_the_instruction_type() {
    #[derive(Debug)]
    struct NotClone(u32);

    let mut cfg = Cfg::<NotClone>::new();
    cfg.block_mut(cfg.entry()).push(NotClone(7));
    let entry = cfg.entry();

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    let ControlNode::Return(span) = *tree.node(tree.root()) else {
        panic!("a terminal entry block lifts as a return");
    };
    assert_eq!(span, BlockSpan::new(entry, 0, 1));
    let source = &cfg.block(entry).instructions()[0];
    let read = &span.instructions(&cfg)[0];
    assert!(core::ptr::eq(read, source));
    assert_eq!(read.0, 7);
    assert_eq!(report, LiftReport::default());
    // The tree is independent of the graph it was computed from.
    drop(cfg);
    assert_eq!(tree.node_count(), 1);
}
