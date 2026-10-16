//! Merge points in exit-free regions.
//!
//! The structurer reads merge points from a [`PostDominatorTree`], which cuts
//! every exit-free region at the blocks closing its loops. A branch whose arm
//! never reaches an exit therefore has no merge point. The structurer used to
//! read a post-dominator tree that ignored such regions, and these cases pin
//! down where the structure differs now.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::pseudocode;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::test_util::{MockInst, ff};
use crate::{ControlTree, GotoReason, PostDominatorTree, PostNode};
use EdgeKind::{Back, Fallthrough, Jump};

const NAMES: [&str; 6] = ["b0", "b1", "b2", "b3", "b4", "b5"];

/// A graph with one instruction per block, named after the block, whose
/// block `n` ends in `blocks[n]`'s edges.
fn graph(blocks: &[&[(usize, EdgeKind)]]) -> Cfg<MockInst> {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let mut ids = vec![cfg.entry()];
    ids.extend((1..blocks.len()).map(|_| cfg.new_block()));
    for (block, edges) in blocks.iter().enumerate() {
        cfg.block_mut(ids[block]).push(ff(NAMES[block]));
        for &(target, kind) in *edges {
            cfg.add_edge(ids[block], ids[target], kind);
        }
    }
    cfg
}

fn text(cfg: &Cfg<MockInst>) -> alloc::string::String {
    pseudocode(cfg, &ControlTree::compute(cfg))
}

#[test]
fn a_branch_with_a_spinning_arm_has_no_merge_point() {
    // b0 --true--> b1 (spins on itself); b0 --false--> b2 (returns).
    let cfg = graph(&[
        &[
            (1, EdgeKind::ConditionalTrue),
            (2, EdgeKind::ConditionalFalse),
        ],
        &[(1, Jump)],
        &[],
    ]);

    // The cut joins the spinning block to the virtual exit, so the return no
    // longer post-dominates the branch.
    let post = PostDominatorTree::compute(&cfg);
    let ids: Vec<_> = cfg.block_ids().collect();
    assert_eq!(post.ipdom(ids[0]), Some(PostNode::Exit));
    assert!(!post.reaches_exit(ids[1]));
    assert_eq!(post.cut_roots(), [ids[1]]);

    // So the return is the else arm, not code after the branch.
    assert_eq!(
        text(&cfg),
        "if {\n    loop {\n        b1\n    }\n} else {\n    b2\n}\n"
    );
}

#[test]
fn an_exit_chain_beside_a_spinning_arm_stays_in_its_arm() {
    // b0 --true--> b1 (spins); b0 --false--> b2 -> b3 (returns).
    let cfg = graph(&[
        &[
            (1, EdgeKind::ConditionalTrue),
            (2, EdgeKind::ConditionalFalse),
        ],
        &[(1, Jump)],
        &[(3, Fallthrough)],
        &[],
    ]);
    assert_eq!(
        text(&cfg),
        "if {\n    loop {\n        b1\n    }\n} else {\n    b2\n    b3\n}\n"
    );
}

#[test]
fn two_spinning_arms_have_no_merge_point() {
    let cfg = graph(&[
        &[
            (1, EdgeKind::ConditionalTrue),
            (2, EdgeKind::ConditionalFalse),
        ],
        &[(1, Jump)],
        &[(2, Jump)],
    ]);
    assert_eq!(
        text(&cfg),
        "if {\n    loop {\n        b1\n    }\n} else {\n    loop {\n        b2\n    }\n}\n"
    );
}

#[test]
fn a_diamond_inside_an_endless_loop_still_merges() {
    // b0 -> b1 { b2 | b3 } -> b4 -> b1: no exit, but the loop's latch is cut,
    // and everything between the branch and the cut still post-dominates.
    let cfg = graph(&[
        &[(1, Fallthrough)],
        &[
            (2, EdgeKind::ConditionalTrue),
            (3, EdgeKind::ConditionalFalse),
        ],
        &[(4, Fallthrough)],
        &[(4, Fallthrough)],
        &[(1, Jump)],
    ]);
    assert_eq!(
        text(&cfg),
        "b0\nloop {\n    if {\n        b2\n    } else {\n        b3\n    }\n    b4\n}\n"
    );
}

#[test]
fn a_diamond_before_an_endless_loop_still_merges() {
    let cfg = graph(&[
        &[
            (1, EdgeKind::ConditionalTrue),
            (2, EdgeKind::ConditionalFalse),
        ],
        &[(3, Fallthrough)],
        &[(3, Fallthrough)],
        &[(3, Jump)],
    ]);
    assert_eq!(
        text(&cfg),
        "if {\n    b1\n} else {\n    b2\n}\nloop {\n    b3\n}\n"
    );
}

#[test]
fn a_loop_with_an_exit_keeps_its_follow() {
    // The exit-free analysis does not touch a loop some path leaves.
    let cfg = graph(&[
        &[(1, Fallthrough)],
        &[
            (2, EdgeKind::ConditionalTrue),
            (5, EdgeKind::ConditionalFalse),
        ],
        &[(3, Fallthrough)],
        &[(1, Jump)],
        &[],
        &[],
    ]);
    assert_eq!(
        text(&cfg),
        "b0\nwhile {\n    b1\n    break if !cond;\n    b2\n    b3\n}\nb5\n"
    );
}

#[test]
fn a_function_with_no_exit_is_one_endless_loop() {
    let cfg = graph(&[&[(1, Fallthrough)], &[(0, Back)]]);
    assert_eq!(text(&cfg), "loop {\n    b0\n    b1\n}\n");
}

#[test]
fn arms_that_enter_a_shared_exit_free_loop_keep_the_transfer_explicit() {
    // b0 --true--> b1 -> b3; b0 --false--> b2 -> b3; b3 -> b1 (back edge).
    // Without an exit the structurer once took the last block as the exit and
    // let the else arm end silently, losing its transfer to b3.
    let cfg = graph(&[
        &[
            (1, EdgeKind::ConditionalTrue),
            (2, EdgeKind::ConditionalFalse),
        ],
        &[(3, Fallthrough)],
        &[(3, Fallthrough)],
        &[(1, Back)],
    ]);
    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert_eq!(report.gotos.len(), 1, "{report:?}");
    assert_eq!(report.gotos[0].reason, GotoReason::RevisitedTarget);
    assert_eq!(
        pseudocode(&cfg, &tree),
        "if {\n    loop {\n        b1\n        .bb3:\n            b3\n    }\n} else {\n    b2\n    goto .bb3;\n}\n"
    );
}
