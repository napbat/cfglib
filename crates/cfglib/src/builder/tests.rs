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

/// Assert the built graph's blocks (by instruction mnemonic) and edges (as
/// `(source, target, kind)`), both in creation order.
fn assert_shape(cfg: &Cfg<MockInst>, blocks: &[&[&str]], edges: &[(usize, usize, EdgeKind)]) {
    let actual_blocks: Vec<Vec<&str>> = cfg
        .block_ids()
        .map(|block| {
            cfg.block(block)
                .instructions()
                .iter()
                .map(|i| i.1)
                .collect()
        })
        .collect();
    let expected_blocks: Vec<Vec<&str>> = blocks.iter().map(|block| block.to_vec()).collect();
    assert_eq!(actual_blocks, expected_blocks, "blocks");
    let actual_edges: Vec<_> = cfg
        .edges()
        .map(|edge| (edge.source().index(), edge.target().index(), edge.kind()))
        .collect();
    assert_eq!(actual_edges, edges, "edges");
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
    assert_eq!(resolution.unresolved, [] as [(crate::BlockId, &str); 0]);
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
fn resolve_wires_jumps_in_a_graph_with_an_edge_payload() {
    use crate::exception::ExceptionFlow;

    let mut cfg = Cfg::<GotoInst, ExceptionFlow<u8>>::with_edge_payload();
    let target = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(goto("end"));
    cfg.block_mut(target).push(label("end"));

    let first = resolve_jump_edges(&mut cfg);
    assert_eq!(first.resolved, 1);
    let jump = cfg
        .edges()
        .find(|edge| edge.kind() == EdgeKind::Jump)
        .expect("the goto's edge is wired");
    assert_eq!((jump.source(), jump.target()), (cfg.entry(), target));
    assert_eq!(
        jump.payload(),
        &ExceptionFlow::default(),
        "a wired edge carries the default payload"
    );
    assert_eq!(resolve_jump_edges(&mut cfg).resolved, 0);
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
    // a; if; b; endif; c; ret
    let cfg = CfgBuilder::build(vec![
        ff("a"),
        MockInst(FlowEffect::ConditionalOpen, "if"),
        ff("b"),
        MockInst(FlowEffect::ConditionalClose, "endif"),
        ff("c"),
        MockInst(FlowEffect::Return, "ret"),
    ])
    .unwrap();
    // Without an `else`, the false path goes straight to the merge.
    assert_shape(
        &cfg,
        &[&["a", "if"], &["b"], &["c", "ret"]],
        &[
            (0, 1, EdgeKind::ConditionalTrue),
            (1, 2, EdgeKind::Fallthrough),
            (0, 2, EdgeKind::ConditionalFalse),
        ],
    );
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
    assert_shape(
        &cfg,
        &[&["a", "if"], &["b"], &["else", "c"], &["d", "ret"]],
        &[
            (0, 1, EdgeKind::ConditionalTrue),
            (0, 2, EdgeKind::ConditionalFalse),
            (1, 3, EdgeKind::Fallthrough),
            (2, 3, EdgeKind::Fallthrough),
        ],
    );
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
    // The header bb1 loops onto itself; the post-loop block bb2 is
    // unreachable without a break.
    assert_shape(
        &cfg,
        &[&["loop"], &["a"], &["ret"]],
        &[(0, 1, EdgeKind::Fallthrough), (1, 1, EdgeKind::Back)],
    );
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
    // The break reaches the post-loop block bb3. The loop's back edge starts
    // in bb2, the unreachable code after the break, so the body never repeats.
    assert_shape(
        &cfg,
        &[&["loop"], &["a", "break"], &[], &["ret"]],
        &[
            (0, 1, EdgeKind::Fallthrough),
            (2, 1, EdgeKind::Back),
            (1, 3, EdgeKind::Unconditional),
        ],
    );
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
    // The taken path goes to an empty break block bb2, which leaves for the
    // post-loop block bb4; the other path continues into `b` and loops.
    assert_shape(
        &cfg,
        &[&["loop"], &["a", "breakc"], &[], &["b"], &["ret"]],
        &[
            (0, 1, EdgeKind::Fallthrough),
            (1, 2, EdgeKind::ConditionalTrue),
            (1, 3, EdgeKind::ConditionalFalse),
            (3, 1, EdgeKind::Back),
            (2, 4, EdgeKind::Unconditional),
        ],
    );
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
    // `continue` closes the header bb1 onto itself; `b` is code after it, and
    // the loop's own back edge starts there.
    assert_shape(
        &cfg,
        &[&["loop"], &["a", "continue"], &["b"], &["ret"]],
        &[
            (0, 1, EdgeKind::Fallthrough),
            (1, 1, EdgeKind::Back),
            (2, 1, EdgeKind::Back),
        ],
    );
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
    // The taken path goes back to the header bb1, the other falls into `b`.
    assert_shape(
        &cfg,
        &[&["loop"], &["a", "continuec"], &["b"], &["ret"]],
        &[
            (0, 1, EdgeKind::Fallthrough),
            (1, 1, EdgeKind::ConditionalTrue),
            (1, 2, EdgeKind::ConditionalFalse),
            (2, 1, EdgeKind::Back),
        ],
    );
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
    // retc splits into an empty terminal return block and the continuation.
    assert_shape(
        &cfg,
        &[&["a", "retc"], &[], &["b", "ret"]],
        &[
            (0, 1, EdgeKind::ConditionalTrue),
            (0, 2, EdgeKind::ConditionalFalse),
        ],
    );
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
    // abort ends its block with no successor; b starts a separate block that
    // nothing reaches.
    assert_shape(&cfg, &[&["a", "abort"], &["b", "ret"]], &[]);
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
    // The label leads a new block that the first one falls into.
    assert_shape(
        &cfg,
        &[&["a"], &["label_0", "b", "ret"]],
        &[(0, 1, EdgeKind::Fallthrough)],
    );
}

#[test]
fn switch_with_cases() {
    // switch; a; case; b; default; c; endswitch; d; ret
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
    // The statements before the first marker are the first arm; every arm
    // falls through to the merge.
    assert_shape(
        &cfg,
        &[
            &["switch"],
            &["a"],
            &["case", "b"],
            &["default", "c"],
            &["d", "ret"],
        ],
        &[
            (0, 1, EdgeKind::SwitchCase),
            (0, 2, EdgeKind::SwitchCase),
            (0, 3, EdgeKind::SwitchCase),
            (1, 4, EdgeKind::Fallthrough),
            (2, 4, EdgeKind::Fallthrough),
            (3, 4, EdgeKind::Fallthrough),
        ],
    );
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
    // The break wires to the post-switch merge, bb4. Block 2, empty and
    // without predecessors, is the code after the break.
    assert_shape(
        &cfg,
        &[
            &["switch"],
            &["a", "break"],
            &[],
            &["case", "b"],
            &["c", "ret"],
        ],
        &[
            (0, 1, EdgeKind::SwitchCase),
            (0, 3, EdgeKind::SwitchCase),
            (2, 4, EdgeKind::Fallthrough),
            (3, 4, EdgeKind::Fallthrough),
            (1, 4, EdgeKind::Unconditional),
        ],
    );
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
    // The loop header bb1 holds the `if`; the merge bb4 closes the loop.
    assert_shape(
        &cfg,
        &[&["loop"], &["if"], &["a"], &["else", "b"], &[], &["ret"]],
        &[
            (0, 1, EdgeKind::Fallthrough),
            (1, 2, EdgeKind::ConditionalTrue),
            (1, 3, EdgeKind::ConditionalFalse),
            (2, 4, EdgeKind::Fallthrough),
            (3, 4, EdgeKind::Fallthrough),
            (4, 1, EdgeKind::Back),
        ],
    );
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
    // bb2 is the loop header, bb3 the break block, bb4 the loop's
    // continuation, bb5 the post-loop block, and bb6 the merge of the `if`.
    assert_shape(
        &cfg,
        &[
            &["if"],
            &["loop"],
            &["a", "breakc"],
            &[],
            &[],
            &[],
            &["ret"],
        ],
        &[
            (0, 1, EdgeKind::ConditionalTrue),
            (1, 2, EdgeKind::Fallthrough),
            (2, 3, EdgeKind::ConditionalTrue),
            (2, 4, EdgeKind::ConditionalFalse),
            (4, 2, EdgeKind::Back),
            (3, 5, EdgeKind::Unconditional),
            (5, 6, EdgeKind::Fallthrough),
            (0, 6, EdgeKind::ConditionalFalse),
        ],
    );
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
    // The `break` in the first arm (bb2) reaches the switch merge bb5, which
    // holds the `breakc`; only that one leaves the loop, for bb8.
    assert_shape(
        &cfg,
        &[
            &["loop"],
            &["switch"],
            &["a", "break"],
            &[],
            &["case", "b"],
            &["breakc"],
            &[],
            &[],
            &["ret"],
        ],
        &[
            (0, 1, EdgeKind::Fallthrough),
            (1, 2, EdgeKind::SwitchCase),
            (1, 4, EdgeKind::SwitchCase),
            (3, 5, EdgeKind::Fallthrough),
            (4, 5, EdgeKind::Fallthrough),
            (2, 5, EdgeKind::Unconditional),
            (5, 6, EdgeKind::ConditionalTrue),
            (5, 7, EdgeKind::ConditionalFalse),
            (7, 1, EdgeKind::Back),
            (6, 8, EdgeKind::Unconditional),
        ],
    );
}
