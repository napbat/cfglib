//! Flows that leave the function by a tail call or a conditional return.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{inst, payload};
use crate::builder::address::{AddressCfgOptions, CallPolicy, build_address_cfg};
use crate::edge::EdgeKind;
use crate::flow::{CallSite, Flow};

#[test]
fn a_tail_call_is_a_call_edge_without_a_continuation() {
    let instructions = vec![
        inst(0, Flow::TailCall { target: 2 }),
        inst(1, Flow::Return),
        inst(2, Flow::Return),
    ];
    let graph =
        build_address_cfg(instructions, &[], AddressCfgOptions::default(), payload).unwrap();

    let edges: Vec<_> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    assert_eq!(edges, vec![(0, 2, EdgeKind::Call, "call")]);
    assert_ne!(graph.instruction_blocks[&0], graph.instruction_blocks[&1]);
}

#[test]
fn a_flattened_tail_call_ends_its_block_without_an_edge_and_is_reported() {
    let instructions = vec![
        inst(0, Flow::Next),
        inst(1, Flow::TailCall { target: 3 }),
        inst(2, Flow::Next),
        inst(3, Flow::Return),
    ];
    let options = AddressCfgOptions {
        calls: CallPolicy::Flatten,
        ..AddressCfgOptions::default()
    };
    let graph = build_address_cfg(instructions, &[], options, payload).unwrap();

    let block = |address| graph.instruction_blocks[&address];
    assert_eq!(block(0), block(1));
    assert_ne!(block(1), block(2), "control does not continue after it");
    assert_eq!(block(2), block(3), "a flattened callee is not a leader");
    assert!(graph.cfg.edges().next().is_none());
    assert_eq!(
        graph.calls,
        vec![CallSite {
            source: 1,
            target: Some(3)
        }]
    );
    assert!(
        graph.unresolved_transfers.is_empty(),
        "a flattened callee is a call site, not a missing edge"
    );
}

#[test]
fn a_conditional_return_falls_through_and_leaves_by_no_edge() {
    let instructions = vec![inst(0, Flow::ConditionalReturn), inst(1, Flow::Return)];
    let graph =
        build_address_cfg(instructions, &[], AddressCfgOptions::default(), payload).unwrap();

    let edges: Vec<_> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    assert_eq!(edges, vec![(0, 1, EdgeKind::ConditionalFalse, "fall")]);
    assert!(
        graph.unresolved_transfers.is_empty(),
        "a return leaves the function, so nothing is missing"
    );
}
