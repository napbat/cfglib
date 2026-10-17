//! Loops inside code the entry cannot reach.
//!
//! A catch pad that no exception edge leads to is laid out although the entry
//! never reaches it, and a loop in it is as much a loop as one in live code:
//! the tree must show it as one, not as a goto to the loop's own head.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::invariants::check;
use super::pseudocode;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::{MockInst, ff};
use crate::{ControlNode, ControlTree};

const NAMES: [&str; 6] = ["b0", "b1", "b2", "b3", "b4", "b5"];

/// How the dead loop closes.
#[derive(Debug, Clone, Copy)]
enum Closing {
    /// The header branches back to itself.
    SelfEdge,
    /// A second block returns to the header by an edge tagged `Back`.
    Tagged,
    /// A second block returns to the header by an ordinary jump, so only
    /// dominance within the dead code says it is a loop.
    Untagged,
}

/// A live `b0 -> b1`, and a try over `{b0}` whose catch pad `b2` no edge leads
/// to. The pad runs `b2 -> b3`, `b3` is the loop header and leaves to `b5`
/// when the loop ends, and the loop closes as `closing` says over `b4`.
fn dead_pad_with_a_loop(closing: Closing) -> Cfg<MockInst> {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let b: Vec<BlockId> = core::iter::once(cfg.entry())
        .chain((1..NAMES.len()).map(|_| cfg.new_block()))
        .collect();
    for (index, &block) in b.iter().enumerate() {
        cfg.block_mut(block).push(ff(NAMES[index]));
    }
    cfg.add_edge(b[0], b[1], EdgeKind::Fallthrough);
    cfg.add_edge(b[2], b[3], EdgeKind::Fallthrough);
    cfg.add_edge(b[3], b[5], EdgeKind::ConditionalFalse);
    match closing {
        Closing::SelfEdge => {
            cfg.add_edge(b[3], b[3], EdgeKind::ConditionalTrue);
        }
        Closing::Tagged | Closing::Untagged => {
            cfg.add_edge(b[3], b[4], EdgeKind::ConditionalTrue);
            let back = if matches!(closing, Closing::Tagged) {
                EdgeKind::Back
            } else {
                EdgeKind::Fallthrough
            };
            cfg.add_edge(b[4], b[3], back);
        }
    }
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [b[0]].into_iter().collect(),
        handlers: vec![Handler {
            entry: b[2],
            body: HandlerBody::known([b[2], b[3], b[4], b[5]]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });
    cfg
}

/// The headers of the tree's loops, in tree order.
fn loop_headers(tree: &ControlTree) -> Vec<BlockId> {
    tree.preorder()
        .filter_map(|id| match tree.node(id) {
            ControlNode::Loop { header, .. } => Some(*header),
            _ => None,
        })
        .collect()
}

#[test]
fn a_loop_in_a_dead_catch_pad_is_a_loop_however_it_closes() {
    for closing in [Closing::SelfEdge, Closing::Tagged, Closing::Untagged] {
        let cfg = dead_pad_with_a_loop(closing);
        let header = cfg.block_ids().nth(3).expect("the loop header");
        let (tree, report) = ControlTree::compute_with_report(&cfg);
        check(&cfg, &tree, &report, "dead loop");
        assert_eq!(loop_headers(&tree), [header], "{closing:?}");
        assert!(
            report.is_fully_structured(),
            "{closing:?}: the loop must not be a goto to its own head: {report:?}"
        );
    }
}

#[test]
fn a_dead_pad_loop_nests_in_the_pad_beside_the_live_code() {
    let cfg = dead_pad_with_a_loop(Closing::Untagged);
    let tree = ControlTree::compute(&cfg);
    let text = pseudocode(&cfg, &tree);
    // The live code first, the handler holding the loop, nothing duplicated.
    assert_eq!(text.matches("b3").count(), 1, "{text}");
    assert_eq!(text.matches("b4").count(), 1, "{text}");
    assert!(
        text.find("catch").expect("a catch handler") < text.find("while").expect("a loop"),
        "{text}"
    );
    assert!(text.ends_with("b1\n"), "{text}");
}

#[test]
fn a_dead_loop_beside_a_live_loop_leaves_both_as_loops() {
    // Live: b0 -> b1 {cond} -> b2 -> b1, b1 -> b6 {return}. The dead pad of
    // `dead_pad_with_a_loop` rides along, and a dead block jumps into b2.
    let mut cfg = dead_pad_with_a_loop(Closing::Untagged);
    let live_head = cfg.new_block();
    let live_body = cfg.new_block();
    let live_exit = cfg.new_block();
    let feeder = cfg.new_block();
    for (block, name) in [
        (live_head, "head"),
        (live_body, "body"),
        (live_exit, "exit"),
        (feeder, "feeder"),
    ] {
        cfg.block_mut(block).push(ff(name));
    }
    let b1 = cfg.block_ids().nth(1).expect("b1");
    cfg.add_edge(b1, live_head, EdgeKind::Fallthrough);
    cfg.add_edge(live_head, live_body, EdgeKind::ConditionalTrue);
    cfg.add_edge(live_head, live_exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(live_body, live_head, EdgeKind::Back);
    // Dead code that enters the live loop's body does not make it a latch.
    cfg.add_edge(feeder, live_body, EdgeKind::Jump);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    check(&cfg, &tree, &report, "dead loop beside a live loop");

    let dead_header = cfg.block_ids().nth(3).expect("the dead loop's header");
    let mut headers = loop_headers(&tree);
    headers.sort_unstable();
    assert_eq!(headers, [dead_header, live_head]);
}
