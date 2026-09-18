//! Block-and-edge shapes shared by the reusable-scratch differential tests.
//!
//! A scratch is only proved correct by a *sequence*: the leftovers one call
//! could inherit from the last are exactly what a single call cannot show. So
//! there is one sequence here, sized up and back down again, and each
//! analysis's test fills the blocks with its own instructions rather than
//! restating the shapes.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::cfg::Cfg;
use crate::edge::EdgeKind;

/// A chain of `regions` if/else diamonds, `3 * regions + 1` blocks.
///
/// Every merge block has two predecessors, so an analysis that places merges
/// has something to place at every region.
pub(crate) fn diamond_chain<I>(regions: usize) -> Cfg<I> {
    let mut cfg = Cfg::new();
    let mut current = cfg.entry();
    for _ in 0..regions {
        let then_block = cfg.new_block();
        let else_block = cfg.new_block();
        let merge = cfg.new_block();
        cfg.add_edge(current, then_block, EdgeKind::ConditionalTrue);
        cfg.add_edge(current, else_block, EdgeKind::ConditionalFalse);
        cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);
        cfg.add_edge(else_block, merge, EdgeKind::Fallthrough);
        current = merge;
    }
    cfg
}

/// A loop with two entries, which no dominator-tree shape can make reducible.
fn irreducible<I>() -> Cfg<I> {
    let mut cfg = Cfg::new();
    let left = cfg.new_block();
    let right = cfg.new_block();
    let exit = cfg.new_block();
    cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, right, EdgeKind::Fallthrough);
    cfg.add_edge(right, left, EdgeKind::Back);
    cfg.add_edge(right, exit, EdgeKind::ConditionalFalse);
    cfg
}

/// A self-loop, a block the entry never reaches, and a disconnected pair,
/// which is the dominator *forest* case rather than the tree one.
fn disconnected<I>() -> Cfg<I> {
    let mut cfg = Cfg::new();
    let body = cfg.new_block();
    let orphan = cfg.new_block();
    let orphan_successor = cfg.new_block();
    cfg.add_edge(cfg.entry(), body, EdgeKind::Fallthrough);
    cfg.add_edge(body, body, EdgeKind::Back);
    cfg.add_edge(orphan, orphan_successor, EdgeKind::Fallthrough);
    cfg
}

/// A `length`-block chain, the shape whose dominator tree is a path.
fn chain<I>(length: usize) -> Cfg<I> {
    let mut cfg = Cfg::new();
    let mut current = cfg.entry();
    for _ in 1..length {
        let next = cfg.new_block();
        cfg.add_edge(current, next, EdgeKind::Fallthrough);
        current = next;
    }
    cfg
}

/// The shared sequence: shapes whose sizes rise and fall, so a scratch reused
/// down the sequence is exercised while it grows, after it has shrunk, and
/// against the structures that have no ordinary merge at all.
///
/// The entry block is never a branch target, which is the precondition SSA
/// construction states.
pub(crate) fn scratch_sequence<I>() -> Vec<Cfg<I>> {
    vec![
        chain(1),
        diamond_chain(4),
        diamond_chain(21),
        chain(2),
        irreducible(),
        diamond_chain(1),
        disconnected(),
        chain(40),
        diamond_chain(2),
    ]
}
