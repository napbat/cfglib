//! Transfers the structurer once lost, or repeated after a construct that had
//! already transferred, found by running each graph against its tree.

extern crate alloc;

use super::exhaustive::{Terminator, graph, unpredicated};
use super::invariants::{check, render};
use super::pseudocode;
use super::semantics::assert_streams_alike;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::{MockInst, ff};
use crate::{ControlNode, ControlTree};

/// How many streams a hand-built graph is compared under.
const MANY_STREAMS: u64 = 64;

/// Compute the tree of `cfg`, check every invariant, and compare many runs.
fn check_equivalent<I, E>(cfg: &Cfg<I, E>, context: &str) {
    let (tree, report) = ControlTree::compute_with_report(cfg);
    check(cfg, &tree, &report, context);
    assert_streams_alike(cfg, &tree, &report, context, MANY_STREAMS);
}

#[test]
fn an_inner_branch_that_merges_at_the_loop_header_continues_the_loop() {
    // bb2 branches to bb0 on both arms, so the loop header is its merge, but
    // it sits in an arm whose own merge is bb1: the transfer to bb0 must
    // survive as a continue instead of falling through into bb1.
    let cfg = graph(
        &[
            Terminator::Branch(1, 2),
            Terminator::Branch(0, 3),
            Terminator::Branch(0, 0),
            Terminator::Exit,
        ],
        &[1; 4],
        unpredicated,
    );
    check_equivalent(&cfg, "branch merging at the header inside an arm");
}

#[test]
fn a_swept_branch_that_merges_at_emitted_code_transfers_there() {
    // The switch's second arm is swept under a label, and its branch merges at
    // bb2, which the walk already emitted after the loop. The swept code must
    // go to bb2 instead of ending.
    let cfg = graph(
        &[
            Terminator::Switch {
                cases: 0b011,
                default: Some(2),
            },
            Terminator::Branch(2, 2),
            Terminator::Exit,
        ],
        &[1; 3],
        unpredicated,
    );
    check_equivalent(&cfg, "swept branch merging at emitted code");
}

#[test]
fn a_back_edge_from_dead_code_does_not_loop_the_live_program() {
    // bb1 is dead, so its back edge to bb0 never runs: bb0 returns, and the
    // tree must not wrap it in an endless loop.
    let cfg = graph(
        &[Terminator::Exit, Terminator::One(0, EdgeKind::Back)],
        &[1; 2],
        unpredicated,
    );
    let (tree, report) = ControlTree::compute_with_report(&cfg);
    check(&cfg, &tree, &report, "dead back edge");
    assert!(
        !tree
            .preorder()
            .any(|id| matches!(tree.node(id), ControlNode::Loop { .. })),
        "{}",
        render(&tree)
    );
}

#[test]
fn a_handler_entry_that_dominates_its_protected_blocks_does_not_cut_their_flow() {
    // b0 is the loop header and the catch-all pad. b1 and b2 are protected and
    // raise back to it, so dominance calls both exception edges back edges. b1
    // still falls through into b2: the tree must run b2 after b1 instead of
    // continuing the loop at b1.
    let mut cfg = graph(
        &[
            Terminator::One(1, EdgeKind::Fallthrough),
            Terminator::One(2, EdgeKind::Fallthrough),
            Terminator::Exit,
        ],
        &[1; 3],
        unpredicated,
    );
    let [b0, b1, b2] = [0, 1, 2].map(BlockId::from_index);
    cfg.add_edge(b1, b0, EdgeKind::ExceptionHandler);
    cfg.add_edge(b2, b0, EdgeKind::ExceptionHandler);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [b1, b2].into_iter().collect(),
        handlers: alloc::vec![Handler {
            entry: b0,
            body: HandlerBody::known([b0]),
            kind: HandlerKind::CatchAll,
        }],
        parent: None,
    });
    check_equivalent(&cfg, "handler entry dominating its protected blocks");
}

#[test]
fn a_retry_loop_that_leaves_through_the_enclosing_loops_back_edge_continues_it() {
    // b0 is a while loop; its body b1 is protected, and its pad b2 retries b1,
    // so b1 heads an inner loop through the exception edge. b1's only normal
    // edge returns to b0: the inner loop's body must continue the outer loop
    // there instead of ending after b1.
    let mut cfg = graph(
        &[
            Terminator::Branch(1, 3),
            Terminator::One(0, EdgeKind::Fallthrough),
            Terminator::One(1, EdgeKind::Fallthrough),
            Terminator::Exit,
        ],
        &[1; 4],
        unpredicated,
    );
    let [b1, b2] = [1, 2].map(BlockId::from_index);
    cfg.add_edge(b1, b2, EdgeKind::ExceptionHandler);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [b1].into_iter().collect(),
        handlers: alloc::vec![Handler {
            entry: b2,
            body: HandlerBody::known([b2]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });
    check_equivalent(&cfg, "retry loop inside a while loop");
}

/// Whether a statement of some body is a branch or dispatch directly followed
/// by a `continue` or `break`.
fn has_transfer_after_a_branch(tree: &ControlTree) -> bool {
    tree.node_ids().any(|id| {
        tree.child_bodies(id).any(|body| {
            tree.body(body).windows(2).any(|pair| {
                matches!(
                    tree.node(pair[0]),
                    ControlNode::IfThenElse { .. } | ControlNode::Switch { .. }
                ) && matches!(
                    tree.node(pair[1]),
                    ControlNode::Continue { .. } | ControlNode::Break { .. }
                )
            })
        })
    })
}

#[test]
fn no_transfer_repeats_after_a_branch_whose_arms_all_transfer() {
    // while (i < n) { if (done) { jmp exit } else if (skip) { jmp top } else { work; jmp top } }
    // The inner branch's arms both end in a back edge, so the walk has already
    // transferred control when it reaches the header they merge at.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let [header, done, leave, skip, again, work, exit] = [(); 7].map(|()| cfg.new_block());
    for (block, name) in [
        (cfg.entry(), "init"),
        (header, "i < n"),
        (done, "done"),
        (leave, "jmp exit"),
        (skip, "skip"),
        (again, "jmp top"),
        (work, "work"),
        (exit, "return"),
    ] {
        cfg.block_mut(block).push(ff(name));
    }
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, done, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(done, leave, EdgeKind::ConditionalTrue);
    cfg.add_edge(done, skip, EdgeKind::ConditionalFalse);
    cfg.add_edge(leave, exit, EdgeKind::Jump);
    cfg.add_edge(skip, again, EdgeKind::ConditionalTrue);
    cfg.add_edge(skip, work, EdgeKind::ConditionalFalse);
    cfg.add_edge(again, header, EdgeKind::Jump);
    cfg.add_edge(work, header, EdgeKind::Jump);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    check(&cfg, &tree, &report, "branches that all transfer");
    assert!(report.is_fully_structured(), "{report:?}");
    assert!(
        !has_transfer_after_a_branch(&tree),
        "{}",
        pseudocode(&cfg, &tree)
    );
}

#[test]
fn no_goto_follows_a_try_whose_paths_all_return() {
    // try { if (a < 0) return 0; emit(10 / a); return 1 } catch { return 7 }:
    // the code after the throwing block is unprotected, and only the try
    // reaches it, so the try holds it. Every path through the try returns, so
    // the walk has nothing to resume at.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let [test, early, work, tail, pad] = [(); 5].map(|()| cfg.new_block());
    for (block, name) in [
        (cfg.entry(), "enter"),
        (test, "a < 0"),
        (early, "return 0"),
        (work, "emit"),
        (tail, "return 1"),
        (pad, "return 7"),
    ] {
        cfg.block_mut(block).push(ff(name));
    }
    cfg.add_edge(cfg.entry(), test, EdgeKind::Fallthrough);
    cfg.add_edge(test, early, EdgeKind::ConditionalTrue);
    cfg.add_edge(test, work, EdgeKind::ConditionalFalse);
    cfg.add_edge(work, tail, EdgeKind::Fallthrough);
    cfg.add_edge(work, pad, EdgeKind::ExceptionUnwind);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [test, early, work].into_iter().collect(),
        handlers: alloc::vec![Handler {
            entry: pad,
            body: HandlerBody::known([pad]),
            kind: HandlerKind::CatchAll,
        }],
        parent: None,
    });

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    check(&cfg, &tree, &report, "a try whose paths all return");
    assert!(report.is_fully_structured(), "{report:?}");
    assert!(
        !tree
            .preorder()
            .any(|id| matches!(tree.node(id), ControlNode::Goto { .. })),
        "{}",
        pseudocode(&cfg, &tree)
    );
}

#[test]
fn a_dead_extent_tail_does_not_make_a_catch_that_resumes_look_like_it_transfers() {
    // The try body ends in gotos on every path, and the catch pad b9 jumps to
    // b5, the region's continuation, so it ends there silently. Dead blocks of
    // the catch extent (b2, b3) follow the pad's code and end in gotos, but
    // the pad's own flow still reaches the end of the try: the walk has to
    // resume at b5, which the try body already emitted, instead of dropping
    // the transfer because the last statement of the handler is a goto.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let blocks: alloc::vec::Vec<BlockId> = core::iter::once(cfg.entry())
        .chain((1..10).map(|_| cfg.new_block()))
        .collect();
    for (index, &block) in blocks.iter().enumerate() {
        let name: &'static str = alloc::string::String::leak(alloc::format!("b{index}"));
        cfg.block_mut(block).push(ff(name));
    }
    for (from, to, kind) in [
        (0, 6, EdgeKind::ConditionalTrue),
        (0, 1, EdgeKind::ConditionalFalse),
        (0, 9, EdgeKind::ExceptionHandler),
        (1, 6, EdgeKind::ConditionalTrue),
        (1, 5, EdgeKind::ConditionalFalse),
        (2, 4, EdgeKind::ConditionalTrue),
        (2, 8, EdgeKind::ConditionalFalse),
        (3, 4, EdgeKind::ConditionalTrue),
        (3, 4, EdgeKind::ConditionalFalse),
        (5, 4, EdgeKind::Jump),
        (6, 5, EdgeKind::Jump),
        (9, 5, EdgeKind::Jump),
    ] {
        cfg.add_edge(blocks[from], blocks[to], kind);
    }
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [blocks[0]].into_iter().collect(),
        handlers: alloc::vec![Handler {
            entry: blocks[9],
            body: HandlerBody::known([blocks[2], blocks[3], blocks[9]]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });
    check_equivalent(&cfg, "catch that resumes after a dead extent tail");
}
