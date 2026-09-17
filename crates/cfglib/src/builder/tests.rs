use super::*;
use crate::test_util::{MockInst, ff};
use alloc::vec;

#[derive(Debug, Clone)]
struct GotoInst {
    effect: FlowEffect,
    target: Option<&'static str>,
    label: Option<&'static str>,
}

fn gi(effect: FlowEffect) -> GotoInst {
    GotoInst {
        effect,
        target: None,
        label: None,
    }
}

fn goto(target: &'static str) -> GotoInst {
    GotoInst {
        effect: FlowEffect::Jump,
        target: Some(target),
        label: None,
    }
}

fn label(name: &'static str) -> GotoInst {
    GotoInst {
        effect: FlowEffect::Label,
        target: None,
        label: Some(name),
    }
}

impl FlowControl for GotoInst {
    fn flow_effect(&self) -> FlowEffect {
        self.effect
    }
}

impl JumpTargets for GotoInst {
    type Target = &'static str;

    fn jump_target(&self) -> Option<&'static str> {
        self.target
    }

    fn label(&self) -> Option<&'static str> {
        self.label
    }
}

#[test]
fn leading_label_and_conditional_continuation_stay_connected() {
    // A label as the first instruction: the (empty) entry must still
    // reach it, or the whole function body is unreachable.
    let cfg = CfgBuilder::build(vec![
        label("head"),
        gi(FlowEffect::Fallthrough),
        gi(FlowEffect::Return),
    ])
    .unwrap();
    let reachable = cfg.depth_first_preorder();
    assert_eq!(reachable.len(), cfg.block_count(), "all blocks reachable");

    // A label directly after a conditional jump IS the false-path
    // continuation; the edge must exist.
    let cfg = CfgBuilder::build(vec![
        GotoInst {
            effect: FlowEffect::ConditionalJump,
            target: Some("skip"),
            label: None,
        },
        label("skip"),
        gi(FlowEffect::Return),
    ])
    .unwrap();
    let reachable = cfg.depth_first_preorder();
    assert_eq!(reachable.len(), cfg.block_count(), "false path connected");
}

#[test]
fn resolve_wires_forward_goto() {
    let mut cfg = CfgBuilder::build(vec![
        gi(FlowEffect::Fallthrough),
        goto("exit"),
        gi(FlowEffect::Fallthrough), // dead code between jump and label
        label("exit"),
        gi(FlowEffect::Return),
    ])
    .unwrap();

    let resolution = resolve_jump_edges(&mut cfg);
    assert_eq!(resolution.resolved, 1);
    assert!(resolution.unresolved.is_empty());
    let jump_edge = cfg
        .edges()
        .find(|edge| edge.kind() == EdgeKind::Jump)
        .expect("goto edge wired");
    assert_eq!(jump_edge.source(), cfg.entry());
    let target_block = cfg.block(jump_edge.target());
    assert_eq!(target_block.instructions()[0].label, Some("exit"));
}

#[test]
fn resolve_wires_backward_goto_and_conditional_taken_edge() {
    let mut cfg = CfgBuilder::build(vec![
        label("head"),
        gi(FlowEffect::Fallthrough),
        GotoInst {
            effect: FlowEffect::ConditionalJump,
            target: Some("head"),
            label: None,
        },
        gi(FlowEffect::Return),
    ])
    .unwrap();

    let resolution = resolve_jump_edges(&mut cfg);
    assert_eq!(resolution.resolved, 1);
    let taken = cfg
        .edges()
        .find(|edge| edge.kind() == EdgeKind::ConditionalTrue)
        .expect("taken edge wired");
    assert_eq!(
        cfg.block(taken.target()).instructions()[0].label,
        Some("head")
    );
    // The builder's fallthrough continuation edge is still present.
    assert!(
        cfg.edges()
            .any(|edge| edge.kind() == EdgeKind::ConditionalFalse)
    );
}

#[test]
fn resolve_reports_unresolved_and_is_idempotent() {
    let mut cfg = CfgBuilder::build(vec![
        gi(FlowEffect::Fallthrough),
        goto("nowhere"),
        label("here"),
        GotoInst {
            effect: FlowEffect::Jump,
            target: Some("here"),
            label: None,
        },
    ])
    .unwrap();

    let first = resolve_jump_edges(&mut cfg);
    assert_eq!(first.resolved, 1);
    assert_eq!(first.unresolved.len(), 1);
    assert_eq!(first.unresolved[0].1, "nowhere");

    let second = resolve_jump_edges(&mut cfg);
    assert_eq!(second.resolved, 0, "already-wired edges are not duplicated");
    assert_eq!(second.unresolved.len(), 1);
}

#[test]
fn linear_block() {
    let cfg = CfgBuilder::build(vec![ff("a"), ff("b"), ff("c")]).unwrap();
    assert_eq!(cfg.block_count(), 1);
    assert_eq!(cfg.edge_count(), 0);
    assert_eq!(cfg.block(cfg.entry()).instructions().len(), 3);
}

#[test]
fn single_return() {
    let cfg = CfgBuilder::build(vec![ff("a"), MockInst(FlowEffect::Return, "ret")]).unwrap();
    // One block with instructions, trailing empty block trimmed.
    assert_eq!(cfg.block_count(), 1);
    assert_eq!(cfg.block(cfg.entry()).instructions().len(), 2);
}

#[test]
fn if_endif_no_else() {
    // a; if; b; endif; c
    let cfg = CfgBuilder::build(vec![
        ff("a"),
        MockInst(FlowEffect::ConditionalOpen, "if"),
        ff("b"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
        ff("c"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // bb0: [a, if]
    // bb1: [b]  (true arm)
    // bb2: []   (merge — c, ret)
    assert!(cfg.block_count() >= 3);
    // Entry has two successors: true arm + false arm (merge).
    assert_eq!(cfg.outgoing(cfg.entry()).count(), 2);
}

#[test]
fn if_else_endif() {
    // a; if; b; else; c; endif; d; ret
    let cfg = CfgBuilder::build(vec![
        ff("a"),
        MockInst(FlowEffect::ConditionalOpen, "if"),
        ff("b"),
        MockInst(FlowEffect::ConditionalAlternate, "else"),
        ff("c"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
        ff("d"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // bb0: [a, if] → true(bb1), false(bb2)
    // bb1: [b]     → merge(bb3)
    // bb2: [else, c] → merge(bb3)
    // bb3: [d, ret]
    assert!(cfg.block_count() >= 4);
    assert_eq!(cfg.outgoing(cfg.entry()).count(), 2);
}

#[test]
fn simple_loop() {
    // loop; a; endloop; ret
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        ff("a"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // bb0: [loop]       → fallthrough to bb1 (header)
    // bb1: [a]          → back to bb1 (header)
    // bb2: [ret]        (post-loop, unreachable without break)
    assert!(cfg.block_count() >= 2);
}

#[test]
fn loop_with_break() {
    // loop; a; break; endloop; ret
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        ff("a"),
        MockInst(FlowEffect::Break, "break"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // post-loop block should be reachable from the break.
    assert!(cfg.block_count() >= 3);
}

#[test]
fn loop_with_conditional_break() {
    // loop; a; breakc; b; endloop; ret
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        ff("a"),
        MockInst(FlowEffect::ConditionalBreak, "breakc"),
        ff("b"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // The breakc block should have two successors:
    // - true → break_block (which goes to post-loop)
    // - false → continue block (with b)
    assert!(cfg.block_count() >= 4);
}

#[test]
fn declarations_are_stored() {
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::Declaration, "dcl_temps"),
        ff("a"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // Declarations are included in the block.
    assert_eq!(cfg.block_count(), 1);
    assert_eq!(cfg.block(cfg.entry()).instructions().len(), 3);
}

#[test]
fn dot_output() {
    let cfg = CfgBuilder::build(vec![
        ff("add"),
        MockInst(FlowEffect::ConditionalOpen, "if"),
        ff("mul"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    let dot = cfg.to_dot();
    assert!(dot.contains("digraph cfg"));
    assert!(dot.contains("bb0"));
    assert!(dot.contains("green4")); // conditional true edge
}

#[test]
fn traversal_preorder() {
    let cfg = CfgBuilder::build(vec![
        ff("a"),
        MockInst(FlowEffect::ConditionalOpen, "if"),
        ff("b"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    let pre = cfg.depth_first_preorder();
    // Entry should be first.
    assert_eq!(pre[0], cfg.entry());
    // All reachable blocks should be visited.
    assert!(pre.len() >= 3);
}

#[test]
fn dominator_tree_linear() {
    let cfg = CfgBuilder::build(vec![
        ff("a"),
        MockInst(FlowEffect::ConditionalOpen, "if"),
        ff("b"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
        ff("c"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    let dom = crate::graph::dominator::DominatorTree::compute(&cfg);
    // Entry dominates all blocks.
    for block_id in cfg.block_ids() {
        assert!(dom.dominates(cfg.entry(), block_id));
    }
}

#[test]
fn continue_jumps_to_header() {
    // loop; a; continue; b; endloop; ret
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        ff("a"),
        MockInst(FlowEffect::Continue, "continue"),
        ff("b"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // The continue should create a back-edge to the header.
    let has_back = cfg.edges().any(|e| e.kind() == EdgeKind::Back);
    assert!(has_back);
    assert!(cfg.block_count() >= 3);
}

#[test]
fn conditional_continue() {
    // loop; a; continuec; b; endloop; ret
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        ff("a"),
        MockInst(FlowEffect::ConditionalContinue, "continuec"),
        ff("b"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // continuec block has two successors: true->header, false->continue
    let has_cond_true = cfg.edges().any(|e| e.kind() == EdgeKind::ConditionalTrue);
    let has_cond_false = cfg.edges().any(|e| e.kind() == EdgeKind::ConditionalFalse);
    assert!(has_cond_true);
    assert!(has_cond_false);
    assert!(cfg.block_count() >= 4);
}

#[test]
fn conditional_return() {
    // a; retc; b; ret
    let cfg = CfgBuilder::build(vec![
        ff("a"),
        MockInst(FlowEffect::ConditionalReturn, "retc"),
        ff("b"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // retc splits into ret_block (terminal) and cont_block (with b).
    let has_cond_true = cfg.edges().any(|e| e.kind() == EdgeKind::ConditionalTrue);
    let has_cond_false = cfg.edges().any(|e| e.kind() == EdgeKind::ConditionalFalse);
    assert!(has_cond_true);
    assert!(has_cond_false);
    assert!(cfg.block_count() >= 3);
}

#[test]
fn terminate_ends_block() {
    // a; abort; b; ret
    let cfg = CfgBuilder::build(vec![
        ff("a"),
        MockInst(FlowEffect::Terminate, "abort"),
        ff("b"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // abort terminates the block; b starts a new (unreachable) block.
    assert!(cfg.block_count() >= 2);
}

#[test]
fn label_splits_block() {
    // a; label; b; ret
    let cfg = CfgBuilder::build(vec![
        ff("a"),
        MockInst(FlowEffect::Label, "label_0"),
        ff("b"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // The label should split into two blocks with a fallthrough edge.
    assert!(cfg.block_count() >= 2);
    let has_fallthrough = cfg.edges().any(|e| e.kind() == EdgeKind::Fallthrough);
    assert!(has_fallthrough);
}

#[test]
fn switch_with_cases() {
    // switch; a; case; b; case; c; endswitch; d; ret
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::SwitchOpen, "switch"),
        ff("a"),
        MockInst(FlowEffect::SwitchCase, "case"),
        ff("b"),
        MockInst(FlowEffect::SwitchCase, "default"),
        ff("c"),
        MockInst(FlowEffect::SwitchClose, "endswitch"),
        ff("d"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // switch block dispatches to multiple case arms.
    let switch_edges: Vec<_> = cfg
        .edges()
        .filter(|e| e.kind() == EdgeKind::SwitchCase)
        .collect();
    assert!(switch_edges.len() >= 2); // at least first case + case + default
    assert!(cfg.block_count() >= 5);
}

#[test]
fn switch_break_exits_switch() {
    // switch; a; break; case; b; endswitch; c; ret
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::SwitchOpen, "switch"),
        ff("a"),
        MockInst(FlowEffect::Break, "break"),
        MockInst(FlowEffect::SwitchCase, "case"),
        ff("b"),
        MockInst(FlowEffect::SwitchClose, "endswitch"),
        ff("c"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // The break should wire to the post-switch merge block.
    let unconditional_edges: Vec<_> = cfg
        .edges()
        .filter(|e| e.kind() == EdgeKind::Unconditional)
        .collect();
    assert!(!unconditional_edges.is_empty());
    assert!(cfg.block_count() >= 4);
}

#[test]
fn nested_if_in_loop() {
    // loop; if; a; else; b; endif; endloop; ret
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        MockInst(FlowEffect::ConditionalOpen, "if"),
        ff("a"),
        MockInst(FlowEffect::ConditionalAlternate, "else"),
        ff("b"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    assert!(cfg.block_count() >= 5);
    let has_back = cfg.edges().any(|e| e.kind() == EdgeKind::Back);
    assert!(has_back);
}

#[test]
fn nested_loop_in_if() {
    // if; loop; a; breakc; endloop; endif; ret
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::ConditionalOpen, "if"),
        MockInst(FlowEffect::LoopOpen, "loop"),
        ff("a"),
        MockInst(FlowEffect::ConditionalBreak, "breakc"),
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    assert!(cfg.block_count() >= 5);
    let has_back = cfg.edges().any(|e| e.kind() == EdgeKind::Back);
    assert!(has_back);
}

#[test]
fn switch_inside_loop_break_exits_switch() {
    // loop; switch; a; break; case; b; endswitch; breakc; endloop; ret
    let cfg = CfgBuilder::build(vec![
        MockInst(FlowEffect::LoopOpen, "loop"),
        MockInst(FlowEffect::SwitchOpen, "switch"),
        ff("a"),
        MockInst(FlowEffect::Break, "break"), // exits switch, not loop
        MockInst(FlowEffect::SwitchCase, "case"),
        ff("b"),
        MockInst(FlowEffect::SwitchClose, "endswitch"),
        MockInst(FlowEffect::ConditionalBreak, "breakc"), // exits loop
        MockInst(FlowEffect::LoopClose, "endloop"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // The break inside the switch should exit the switch.
    // The breakc after endswitch should exit the loop.
    assert!(cfg.block_count() >= 6);
    let has_back = cfg.edges().any(|e| e.kind() == EdgeKind::Back);
    assert!(has_back);
}
