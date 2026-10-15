//! What the entry cannot reach: which blocks the tree holds, and where.
//!
//! A block is owned when the entry reaches it or a handler owns it (an
//! entry, a filter funclet, a block of a known extent). Dead protected code,
//! and code only dead code reaches, is dropped, with one exception: a goto
//! that left owned code for it finds its label.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

use super::invariants::check;
use super::pseudocode;
use crate::ControlTree;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind::{self, ExceptionLeave, Fallthrough};
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::{MockInst, ff};

/// `count` blocks, each holding one instruction named by its index.
fn blocks(count: usize) -> (Cfg<MockInst>, Vec<BlockId>) {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let ids: Vec<BlockId> = core::iter::once(cfg.entry())
        .chain((1..count).map(|_| cfg.new_block()))
        .collect();
    for (index, &block) in ids.iter().enumerate() {
        let name: &'static str = alloc::string::String::leak(alloc::format!("b{index}"));
        cfg.block_mut(block).push(ff(name));
    }
    (cfg, ids)
}

/// A catch region protecting `protected`, with a handler at `entry` whose
/// extent is `extent` (the entry included).
fn catch(
    cfg: &mut Cfg<MockInst>,
    protected: &[BlockId],
    entry: BlockId,
    extent: &[BlockId],
    parent: Option<RegionId>,
) -> RegionId {
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: protected.iter().copied().collect(),
        handlers: alloc::vec![Handler {
            entry,
            body: HandlerBody::known(extent.iter().copied()),
            kind: HandlerKind::Catch,
        }],
        parent,
    })
}

fn tree_text(cfg: &Cfg<MockInst>) -> (String, crate::LiftReport) {
    let (tree, report) = ControlTree::compute_with_report(cfg);
    check(cfg, &tree, &report, "dead code");
    (pseudocode(cfg, &tree), report)
}

#[test]
fn a_dead_protected_block_is_dropped() {
    // A try whose body ends in a throw into the handler leaves its `leave`
    // block, b4's successor b6, with no way in: dead protected code that
    // jumps to the join b4. The tree holds none of it, and so no goto to the
    // join, which has no instruction of its own to label.
    //
    // b0 -> b1 (try body, throws) ; b2 = catch pad -> b3 (join) -> b4
    // b5 = dead leave block, protected by the try, -> b3
    let (mut cfg, b) = blocks(6);
    cfg.add_edge(b[0], b[1], Fallthrough);
    cfg.add_edge(b[1], b[2], EdgeKind::ExceptionHandler);
    cfg.add_edge(b[2], b[3], Fallthrough);
    cfg.add_edge(b[3], b[4], Fallthrough);
    cfg.add_edge(b[5], b[3], ExceptionLeave);
    catch(&mut cfg, &[b[1], b[5]], b[2], &[b[2]], None);
    let (text, report) = tree_text(&cfg);
    assert!(!text.contains("b5"), "{text}");
    assert!(report.gotos.is_empty(), "{report:?}");
    assert!(report.swept_blocks.is_empty(), "{report:?}");
    assert!(report.is_fully_structured(), "{report:?}");
}

#[test]
fn a_dead_protected_block_that_leaves_through_an_emptied_join_is_dropped() {
    // The same graph as the HLIL lift sees it: the join b3 is a trampoline
    // (it holds only a pure transfer), so it has no leaf. The dead block's
    // goto to it could not be labeled, and the lift failed on the dangling
    // label.
    let (mut cfg, b) = blocks(6);
    cfg.add_edge(b[0], b[1], Fallthrough);
    cfg.add_edge(b[1], b[2], EdgeKind::ExceptionHandler);
    cfg.add_edge(b[2], b[3], Fallthrough);
    cfg.add_edge(b[3], b[4], Fallthrough);
    cfg.add_edge(b[5], b[3], ExceptionLeave);
    catch(&mut cfg, &[b[1], b[5]], b[2], &[b[2]], None);
    let (tree, report) = ControlTree::compute_with_report_treating_empty(&cfg, &[b[3], b[5]]);
    let text = pseudocode(&cfg, &tree);
    assert!(!text.contains("b5"), "{text}");
    assert!(report.gotos.is_empty(), "{report:?}");
    assert!(report.unresolved_labels.is_empty(), "{report:?}");
    assert!(report.is_fully_structured(), "{report:?}");
}

#[test]
fn a_goto_out_of_a_dead_pad_finds_the_dead_code_it_names() {
    // The pad b3 flows to b4, which a second dead block b5 also enters, so
    // the pad cannot inline it and the transfer is a goto. Nothing owns b4,
    // yet the goto must find its label: b4 is swept under it. b5 itself,
    // which only b4's graph mentions, stays dropped.
    let (mut cfg, b) = blocks(6);
    cfg.add_edge(b[0], b[1], Fallthrough);
    cfg.add_edge(b[1], b[2], Fallthrough);
    cfg.add_edge(b[3], b[4], Fallthrough);
    cfg.add_edge(b[5], b[4], Fallthrough);
    catch(&mut cfg, &[b[1]], b[3], &[b[3]], None);
    let (text, report) = tree_text(&cfg);
    assert_eq!(report.swept_blocks, [b[4]], "{report:?}");
    assert!(report.unresolved_labels.is_empty(), "{report:?}");
    assert_eq!(text.matches("b4\n").count(), 1, "{text}");
    assert!(text.contains(".bb4:"), "{text}");
    assert!(!text.contains("b5"), "{text}");
}

#[test]
fn a_dead_handler_tail_after_the_region_is_not_dropped_with_the_protected_code() {
    // Dropping dead protected code must not drop the handler's own: the pad
    // b2 and its tail b3 are owned although the try body b1 never reaches
    // them, and the dead protected block b4 beside the body is not.
    let (mut cfg, b) = blocks(5);
    cfg.add_edge(b[0], b[1], Fallthrough);
    cfg.add_edge(b[2], b[3], Fallthrough);
    catch(&mut cfg, &[b[1], b[4]], b[2], &[b[2], b[3]], None);
    let (text, report) = tree_text(&cfg);
    assert!(text.contains("b2\n") && text.contains("b3\n"), "{text}");
    assert!(!text.contains("b4"), "{text}");
    assert!(report.is_fully_structured(), "{report:?}");
}
