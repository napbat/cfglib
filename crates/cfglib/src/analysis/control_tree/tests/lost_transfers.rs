//! Transfers the structurer once lost, found by running each graph against
//! its tree.

extern crate alloc;

use super::exhaustive::{Terminator, graph, unpredicated};
use super::invariants::{check, render};
use super::semantics::assert_streams_alike;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
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
