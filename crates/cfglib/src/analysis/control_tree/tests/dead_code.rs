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
use crate::edge::EdgeKind::{
    self, ConditionalFalse, ConditionalTrue, ExceptionLeave, Fallthrough, Unconditional,
};
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
fn a_dead_pad_split_by_a_suspension_keeps_its_tail_in_the_handler() {
    // The pad b3 leaves through two ordinary edges, an exit and a
    // fallthrough to its tail b4 (a suspension has both), so the walk cannot
    // follow a single successor. The tail is the handler's own code and
    // belongs in its body, not at the end of the function.
    let (mut cfg, b) = blocks(5);
    cfg.add_edge(b[0], b[1], Fallthrough);
    cfg.add_edge(b[1], b[2], Fallthrough);
    cfg.add_edge(b[3], b[2], Unconditional);
    cfg.add_edge(b[3], b[4], Fallthrough);
    catch(&mut cfg, &[b[1]], b[3], &[b[3], b[4]], None);
    let (text, report) = tree_text(&cfg);
    assert_eq!(
        text,
        "b0\ntry {\n    b1\n} catch {\n    b3\n    b4\n}\nb2\n"
    );
    assert!(report.swept_blocks.is_empty(), "{report:?}");
    assert!(report.is_fully_structured(), "{report:?}");
}

#[test]
fn a_dead_pad_tail_that_cannot_reach_the_loop_stays_in_the_handler() {
    // A loop b1..b3 with a pad b4 whose extent holds b5, a tail that throws
    // and so never reaches the latch: the loop's bound has the pad (it feeds
    // the latch) but not the tail, which the handler walk used to escape as a
    // goto and a sweep.
    //
    // b0 -> b1 --true--> b2 -> b3 -> b1 (back) ; b1 --false--> b6
    // b4 --true--> b5 (tail, no successor) ; b4 --false--> b3
    let (mut cfg, b) = blocks(7);
    cfg.add_edge(b[0], b[1], Fallthrough);
    cfg.add_edge(b[1], b[2], ConditionalTrue);
    cfg.add_edge(b[1], b[6], ConditionalFalse);
    cfg.add_edge(b[2], b[3], Fallthrough);
    cfg.add_edge(b[3], b[1], EdgeKind::Back);
    cfg.add_edge(b[4], b[5], ConditionalTrue);
    cfg.add_edge(b[4], b[3], ConditionalFalse);
    catch(&mut cfg, &[b[2]], b[4], &[b[4], b[5]], None);
    let (text, report) = tree_text(&cfg);
    assert_eq!(text.matches("b5\n").count(), 1, "{text}");
    assert!(
        report.gotos.iter().all(|goto| goto.target != b[5]),
        "{report:?}"
    );
    assert!(!report.swept_blocks.contains(&b[5]), "{report:?}");
    assert!(report.unresolved_labels.is_empty(), "{report:?}");
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
fn a_nested_pad_stays_with_its_own_region() {
    // The finally pad b3 is reachable and splits on a suspension, so its
    // walk ends at b3. Its extent also names b5, the pad of a region that
    // protects b4. That region lifts b5 when the sweep reaches b4; the outer
    // handler must not take it first and leave the inner catch a goto.
    //
    // b0 -> b1 -> b3 ; b3 -> b2 (unconditional) ; b3 -> b4 -> b2
    // outer region: protects b1, pad b3 with extent {b3, b4, b5}
    // inner region: protects b4, pad b5 with extent {b5}
    let (mut cfg, b) = blocks(6);
    cfg.add_edge(b[0], b[1], Fallthrough);
    cfg.add_edge(b[1], b[3], Fallthrough);
    cfg.add_edge(b[3], b[2], Unconditional);
    cfg.add_edge(b[3], b[4], Fallthrough);
    cfg.add_edge(b[4], b[2], Fallthrough);
    let outer = catch(&mut cfg, &[b[1]], b[3], &[b[3], b[4], b[5]], None);
    catch(&mut cfg, &[b[4]], b[5], &[b[5]], Some(outer));
    let (text, report) = tree_text(&cfg);
    assert!(
        report.gotos.iter().all(|goto| goto.target != b[5]),
        "{report:?}"
    );
    assert_eq!(text.matches("b5\n").count(), 1, "{text}");
    let catch_at = text.rfind("catch {").expect("the inner catch");
    assert!(text[catch_at..].contains("b5"), "{text}");
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
