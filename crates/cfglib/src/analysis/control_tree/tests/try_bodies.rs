//! Which code a `try` holds: one region's protected code, the exits it
//! resumes at, and the code of other regions it must leave alone.

extern crate alloc;
use alloc::vec;

use super::{
    body_contains_block, body_subtree, contains_block, find_try_catch, has_node_kind, pseudocode,
};
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::{MockInst, ff};
use crate::{ControlNode, ControlTree};

#[test]
fn a_try_anchored_at_a_switch_case_is_structured_inside_that_case() {
    // b0 dispatches to the cases p and q, which rejoin at m. p is protected
    // and its pad h rejoins at m too: the case arm of p is the try.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let p = cfg.new_block();
    let q = cfg.new_block();
    let handler = cfg.new_block();
    let merge = cfg.new_block();
    for (block, name) in [
        (cfg.entry(), "dispatch"),
        (p, "protected"),
        (q, "other"),
        (handler, "caught"),
        (merge, "after"),
    ] {
        cfg.block_mut(block).push(ff(name));
    }
    cfg.add_edge(cfg.entry(), p, EdgeKind::SwitchCase);
    cfg.add_edge(cfg.entry(), q, EdgeKind::SwitchCase);
    cfg.add_edge(p, merge, EdgeKind::Fallthrough);
    cfg.add_edge(q, merge, EdgeKind::Fallthrough);
    cfg.add_edge(p, handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(handler, merge, EdgeKind::Fallthrough);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [p].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::known([handler]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert_eq!(
        report.unstructured_regions,
        [],
        "the region has a known extent: {}",
        pseudocode(&cfg, &tree)
    );
    let (try_body, _, _) = find_try_catch(&tree).expect("the case of p is a try");
    assert!(body_contains_block(&tree, try_body, p));
}

#[test]
fn a_region_entered_at_two_blocks_is_not_one_try() {
    // Both arms of the branch are protected by one region, so ordinary flow
    // enters its protected code at two blocks. No single `try` holds both
    // arms: the tree would cover one and leave the other raising into a pad
    // that its try does not enclose, so the region stays plain flow.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let first = cfg.new_block();
    let second = cfg.new_block();
    let handler = cfg.new_block();
    let merge = cfg.new_block();
    for (block, name) in [
        (cfg.entry(), "test"),
        (first, "first"),
        (second, "second"),
        (handler, "caught"),
        (merge, "after"),
    ] {
        cfg.block_mut(block).push(ff(name));
    }
    cfg.add_edge(cfg.entry(), first, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), second, EdgeKind::ConditionalFalse);
    cfg.add_edge(first, merge, EdgeKind::Fallthrough);
    cfg.add_edge(second, merge, EdgeKind::Fallthrough);
    cfg.add_edge(first, handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(second, handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(handler, merge, EdgeKind::Fallthrough);
    let region = cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [first, second].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::known([handler]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert_eq!(report.unstructured_regions, [region], "{tree:?}");
    assert!(
        !has_node_kind(&tree, |node| matches!(node, ControlNode::TryCatch { .. })),
        "{}",
        pseudocode(&cfg, &tree)
    );
    assert!(contains_block(&tree, handler), "the pad is still emitted");
}

/// `try { x } catch { ... }; between; try { y } catch { ... }; end`, where the
/// first handler leaves to a block of its own, so the first region's exits
/// differ between its body and its handler. With `branching_first_try` the
/// first body also has a second exit, to a side block that rejoins at the end.
fn two_tries_in_sequence(branching_first_try: bool) -> (Cfg<MockInst>, [crate::BlockId; 2]) {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let between = cfg.new_block();
    let second = cfg.new_block();
    let end = cfg.new_block();
    let first_pad = cfg.new_block();
    let first_pad_exit = cfg.new_block();
    let second_pad = cfg.new_block();
    let side = cfg.new_block();
    for (block, name) in [
        (cfg.entry(), "x"),
        (between, "between"),
        (second, "y"),
        (end, "end"),
        (first_pad, "first_caught"),
        (first_pad_exit, "first_leaves"),
        (second_pad, "second_caught"),
        (side, "side"),
    ] {
        cfg.block_mut(block).push(ff(name));
    }
    if branching_first_try {
        cfg.add_edge(cfg.entry(), between, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), side, EdgeKind::ConditionalFalse);
        cfg.add_edge(side, end, EdgeKind::Fallthrough);
    } else {
        cfg.add_edge(cfg.entry(), between, EdgeKind::Fallthrough);
    }
    cfg.add_edge(between, second, EdgeKind::Fallthrough);
    cfg.add_edge(second, end, EdgeKind::Fallthrough);
    cfg.add_edge(cfg.entry(), first_pad, EdgeKind::ExceptionHandler);
    cfg.add_edge(first_pad, first_pad_exit, EdgeKind::Fallthrough);
    cfg.add_edge(second, second_pad, EdgeKind::ExceptionHandler);
    for (protected, pad, extent) in [
        (cfg.entry(), first_pad, vec![first_pad]),
        (second, second_pad, vec![second_pad]),
    ] {
        cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: [protected].into_iter().collect(),
            handlers: vec![Handler {
                entry: pad,
                body: HandlerBody::known(extent),
                kind: HandlerKind::Catch,
            }],
            parent: None,
        });
    }
    (cfg, [between, second])
}

fn count_tries_in_try_bodies(tree: &ControlTree) -> usize {
    tree.preorder()
        .filter_map(|id| match *tree.node(id) {
            ControlNode::TryCatch { try_body, .. } => Some(try_body),
            _ => None,
        })
        .map(|body| {
            body_subtree(tree, body)
                .into_iter()
                .filter(|&id| matches!(tree.node(id), ControlNode::TryCatch { .. }))
                .count()
        })
        .sum()
}

#[test]
fn a_try_resumes_after_its_body_when_its_handler_leaves_elsewhere() {
    // The first region's body falls into `between`, its handler into
    // `first_leaves`, so the two exits differ. The code after the try belongs
    // after it, not in it, and the second try follows the first.
    let (cfg, [between, _]) = two_tries_in_sequence(false);
    let (tree, report) = ControlTree::compute_with_report(&cfg);
    let (try_body, _, _) = find_try_catch(&tree).expect("the first region is a try");
    assert!(
        !body_contains_block(&tree, try_body, between),
        "{}",
        pseudocode(&cfg, &tree)
    );
    assert_eq!(count_tries_in_try_bodies(&tree), 0);
    assert!(report.is_fully_structured(), "{report:?}");
}

#[test]
fn a_try_is_not_swallowed_into_the_try_before_it() {
    // The first try leaves its body along two arms, so it has no single exit
    // to resume at. The unprotected code after it may be lifted into it, but
    // the second region's protected code may not.
    let (cfg, _) = two_tries_in_sequence(true);
    let (tree, _) = ControlTree::compute_with_report(&cfg);
    assert_eq!(
        count_tries_in_try_bodies(&tree),
        0,
        "{}",
        pseudocode(&cfg, &tree)
    );
}

#[test]
fn a_try_does_not_take_the_test_block_of_its_loop_from_another_region() {
    // b1 heads a do-while whose latch b2 is protected by a second region, not
    // by the try around b1. The latch is the loop's test, which the loop node
    // would own, so the loop is endless inside the try and b2 keeps its try.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let header = cfg.new_block();
    let latch = cfg.new_block();
    let exit = cfg.new_block();
    let first_pad = cfg.new_block();
    let second_pad = cfg.new_block();
    for (block, name) in [
        (cfg.entry(), "start"),
        (header, "loop_head"),
        (latch, "loop_test"),
        (exit, "end"),
        (first_pad, "first_caught"),
        (second_pad, "second_caught"),
    ] {
        cfg.block_mut(block).push(ff(name));
    }
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, latch, EdgeKind::Fallthrough);
    cfg.add_edge(latch, header, EdgeKind::ConditionalTrue);
    cfg.add_edge(latch, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(header, first_pad, EdgeKind::ExceptionHandler);
    cfg.add_edge(latch, second_pad, EdgeKind::ExceptionHandler);
    for (protected, pad) in [(header, first_pad), (latch, second_pad)] {
        cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: [protected].into_iter().collect(),
            handlers: vec![Handler {
                entry: pad,
                body: HandlerBody::known([pad]),
                kind: HandlerKind::Catch,
            }],
            parent: None,
        });
    }

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert_eq!(
        tree.preorder()
            .filter(|&id| matches!(tree.node(id), ControlNode::TryCatch { .. }))
            .count(),
        2,
        "both regions stay a try: {}",
        pseudocode(&cfg, &tree)
    );
    assert_eq!(report.unstructured_regions, [], "{report:?}");
    assert_eq!(count_tries_in_try_bodies(&tree), 0);
}
