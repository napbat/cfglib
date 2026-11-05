extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::builder::address::tests::{inst, payload};
use crate::builder::address::{
    AddressBuildError, AddressCfgOptions, CallPolicy, StatedEdge, StatedEdges,
    build_address_cfg_from_edges,
};
use crate::edge::EdgeKind;
use crate::flow::Flow;

type Edge = (u32, u32, EdgeKind, &'static str);

fn options() -> AddressCfgOptions<u32> {
    AddressCfgOptions {
        entry: Some(0),
        calls: CallPolicy::Edges,
    }
}

/// A stated edge from `source` to `target`, with a payload that marks it as
/// stated.
fn stated(source: u32, target: u32, kind: EdgeKind) -> StatedEdge<u32, Edge> {
    StatedEdge {
        target,
        kind,
        payload: (source, target, kind, "stated"),
    }
}

fn build(
    edges: StatedEdges<u32, Edge>,
    flows: Vec<Flow<u32, i32>>,
) -> Result<Vec<Edge>, AddressBuildError<u32>> {
    let instructions = flows
        .into_iter()
        .zip(0..)
        .map(|(flow, address)| inst(address, flow))
        .collect();
    let graph = build_address_cfg_from_edges(instructions, edges, &[], &[], options(), payload)?;
    let mut edges: Vec<Edge> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    edges.sort_unstable_by_key(|&(source, target, _, role)| (source, target, role));
    Ok(edges)
}

#[test]
fn a_stated_continuation_that_the_flow_lacks_is_an_edge() {
    // A trap has no continuation in its flow. The caller states that control
    // comes back after it.
    let edges = BTreeMap::from([
        (0, vec![stated(0, 1, EdgeKind::Fallthrough)]),
        (1, Vec::new()),
    ]);
    assert_eq!(
        build(edges, vec![Flow::Trap, Flow::Return]),
        Ok(vec![(0, 1, EdgeKind::Fallthrough, "stated")])
    );
}

#[test]
fn an_empty_statement_leaves_out_every_normal_edge_of_the_flow() {
    // The callee of the call at 1 never returns. A branch reaches 2, so the
    // stream holds it, but no edge leaves the call.
    let edges = BTreeMap::from([
        (
            0,
            vec![
                stated(0, 2, EdgeKind::ConditionalTrue),
                stated(0, 1, EdgeKind::ConditionalFalse),
            ],
        ),
        (1, Vec::new()),
        (2, Vec::new()),
    ]);
    let flows = vec![
        Flow::Conditional { target: 2 },
        Flow::Call { target: 9 },
        Flow::Return,
    ];
    assert_eq!(
        build(edges, flows),
        Ok(vec![
            (0, 1, EdgeKind::ConditionalFalse, "stated"),
            (0, 2, EdgeKind::ConditionalTrue, "stated"),
        ])
    );
}

#[test]
fn an_instruction_without_stated_edges_continues_and_a_call_keeps_its_continuation() {
    let edges = BTreeMap::from([(2, Vec::new())]);
    let flows = vec![Flow::Next, Flow::Call { target: 9 }, Flow::Return];
    assert_eq!(
        build(edges, flows),
        Ok(vec![(1, 2, EdgeKind::CallReturn, "cont")])
    );
}

#[test]
fn an_unstated_transfer_is_an_error() {
    assert_eq!(
        build(BTreeMap::new(), vec![Flow::Jump { target: 1 }, Flow::Next]),
        Err(AddressBuildError::UnstatedTransfer { address: 0 })
    );
}

#[test]
fn an_unstated_continuation_to_no_instruction_is_an_error() {
    // The stream ends after 0, so the continuation of 0 has no instruction.
    assert_eq!(
        build(BTreeMap::new(), vec![Flow::Next]),
        Err(AddressBuildError::MissingContinuation {
            source: 0,
            target: 1
        })
    );
}

#[test]
fn a_stated_source_or_target_must_start_an_instruction() {
    let edges = BTreeMap::from([(7, Vec::new()), (1, Vec::new())]);
    assert_eq!(
        build(edges, vec![Flow::Next, Flow::Return]),
        Err(AddressBuildError::MissingStatedSource { address: 7 })
    );
    let edges = BTreeMap::from([(0, vec![stated(0, 5, EdgeKind::Jump)]), (1, Vec::new())]);
    assert_eq!(
        build(edges, vec![Flow::Jump { target: 5 }, Flow::Return]),
        Err(AddressBuildError::MissingStatedTarget {
            source: 0,
            target: 5
        })
    );
}
