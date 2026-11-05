extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::builder::address::tests::{Inst, inst, payload};
use crate::builder::address::{
    AddressBuildError, AddressCfgOptions, AddressGraph, CallPolicy, GivenEdge, GivenEdges,
    UnwindEnd, UnwindPath, UnwindRoute, UnwindStep, build_address_cfg_from_given_edges,
};
use crate::edge::EdgeKind;
use crate::flow::{CallSite, Flow};

type Edge = (u32, u32, EdgeKind, &'static str);

fn options(calls: CallPolicy) -> AddressCfgOptions<u32> {
    AddressCfgOptions {
        entry: Some(0),
        calls,
    }
}

/// A given edge from `source` to `target`, with a payload that marks it as
/// given.
fn given(source: u32, target: u32, kind: EdgeKind) -> GivenEdge<u32, Edge> {
    GivenEdge {
        target,
        kind,
        payload: (source, target, kind, "given"),
    }
}

/// One instruction for each flow, at the addresses `0..`.
fn stream(flows: Vec<Flow<u32, i32>>) -> Vec<Inst> {
    flows
        .into_iter()
        .zip(0..)
        .map(|(flow, address)| inst(address, flow))
        .collect()
}

/// The edges of `graph`, sorted.
fn sorted_edges(graph: &AddressGraph<Inst, Edge>) -> Vec<Edge> {
    let mut edges: Vec<Edge> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    edges.sort_unstable_by_key(|&(source, target, _, role)| (source, target, role));
    edges
}

fn build(
    edges: GivenEdges<u32, Edge>,
    flows: Vec<Flow<u32, i32>>,
) -> Result<Vec<Edge>, AddressBuildError<u32>> {
    let graph = build_address_cfg_from_given_edges(
        stream(flows),
        edges,
        &[],
        &[],
        options(CallPolicy::Edges),
        payload,
    )?;
    Ok(sorted_edges(&graph))
}

#[test]
fn a_given_continuation_that_the_flow_lacks_is_an_edge() {
    // A trap has no continuation in its flow. The caller gives the edge that
    // comes back after it.
    let edges = BTreeMap::from([
        (0, vec![given(0, 1, EdgeKind::Fallthrough)]),
        (1, Vec::new()),
    ]);
    assert_eq!(
        build(edges, vec![Flow::Trap, Flow::Return]),
        Ok(vec![(0, 1, EdgeKind::Fallthrough, "given")])
    );
}

#[test]
fn a_given_kind_is_the_choice_of_the_caller() {
    // The flow of 0 is a conditional branch, but the caller gives one jump.
    let edges = BTreeMap::from([(0, vec![given(0, 2, EdgeKind::Jump)]), (2, Vec::new())]);
    assert_eq!(
        build(
            edges,
            vec![Flow::Conditional { target: 2 }, Flow::Next, Flow::Return]
        ),
        Ok(vec![
            (0, 2, EdgeKind::Jump, "given"),
            (1, 2, EdgeKind::Fallthrough, "seq"),
        ])
    );
}

#[test]
fn an_empty_list_leaves_out_every_normal_edge_of_the_flow() {
    // The callee of the call at 1 never returns. A branch reaches 2, so the
    // stream holds it, but no edge leaves the call.
    let edges = BTreeMap::from([
        (
            0,
            vec![
                given(0, 2, EdgeKind::ConditionalTrue),
                given(0, 1, EdgeKind::ConditionalFalse),
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
            (0, 1, EdgeKind::ConditionalFalse, "given"),
            (0, 2, EdgeKind::ConditionalTrue, "given"),
        ])
    );
}

#[test]
fn an_instruction_without_given_edges_continues_and_a_call_keeps_its_continuation() {
    let edges = BTreeMap::from([(2, Vec::new())]);
    let flows = vec![Flow::Next, Flow::Call { target: 9 }, Flow::Return];
    assert_eq!(
        build(edges, flows),
        Ok(vec![(1, 2, EdgeKind::CallReturn, "cont")])
    );
}

#[test]
fn a_given_call_under_flatten_ends_its_block_and_is_reported() {
    // The call at 1 has no given edges, so it is an operation of its block.
    // The call at 2 has given edges, so it ends its block and only its given
    // edge leaves it. Both are call sites.
    let edges = BTreeMap::from([
        (2, vec![given(2, 3, EdgeKind::Fallthrough)]),
        (3, Vec::new()),
    ]);
    let flows = vec![
        Flow::Next,
        Flow::Call { target: 9 },
        Flow::Call { target: 8 },
        Flow::Return,
    ];
    let graph = build_address_cfg_from_given_edges(
        stream(flows),
        edges,
        &[],
        &[],
        options(CallPolicy::Flatten),
        payload,
    )
    .expect("the given edges are valid");

    assert_eq!(graph.cfg.block_count(), 2);
    assert_eq!(graph.instruction_blocks[&0], graph.instruction_blocks[&2]);
    assert_ne!(graph.instruction_blocks[&2], graph.instruction_blocks[&3]);
    assert_eq!(
        sorted_edges(&graph),
        vec![(2, 3, EdgeKind::Fallthrough, "given")]
    );
    assert_eq!(
        graph.calls,
        vec![
            CallSite {
                source: 1,
                target: Some(9),
            },
            CallSite {
                source: 2,
                target: Some(8),
            },
        ]
    );
    assert_eq!(graph.unresolved_transfers, Vec::new());
}

#[test]
fn routes_work_with_given_edges() {
    // 0 is protected. When it throws, the runtime enters the funclet at 2,
    // which returns to the runtime, and ends at the handler at 3. The handler
    // jumps back to 1.
    let edges = BTreeMap::from([
        (1, Vec::new()),
        (2, Vec::new()),
        (3, vec![given(3, 1, EdgeKind::Jump)]),
    ]);
    let flows = vec![
        Flow::Next,
        Flow::Return,
        Flow::Return,
        Flow::Jump { target: 1 },
    ];
    let routes = [UnwindRoute {
        protected: 0..1,
        paths: vec![UnwindPath {
            steps: vec![UnwindStep::Funclet(2)],
            end: UnwindEnd::Handler(3),
        }],
    }];
    let mut graph = build_address_cfg_from_given_edges(
        stream(flows),
        edges,
        &[],
        &routes,
        options(CallPolicy::Edges),
        payload,
    )
    .expect("the given edges and the routes are valid");
    graph
        .add_unwind_routes(&routes, payload)
        .expect("the routes are valid");

    assert_eq!(graph.cfg.block_count(), 4);
    assert_eq!(
        sorted_edges(&graph),
        vec![
            (0, 1, EdgeKind::Fallthrough, "seq"),
            (0, 2, EdgeKind::ExceptionUnwind, "route"),
            (2, 3, EdgeKind::ExceptionUnwind, "route"),
            (3, 1, EdgeKind::Jump, "given"),
        ]
    );
}

#[test]
fn a_transfer_without_given_edges_is_an_error() {
    assert_eq!(
        build(BTreeMap::new(), vec![Flow::Jump { target: 1 }, Flow::Next]),
        Err(AddressBuildError::TransferWithoutGivenEdges { address: 0 })
    );
}

#[test]
fn the_first_error_in_address_order_wins() {
    // 0 jumps without given edges, and the given target of 1 starts no
    // instruction.
    let edges = BTreeMap::from([(1, vec![given(1, 7, EdgeKind::Jump)])]);
    assert_eq!(
        build(edges, vec![Flow::Jump { target: 1 }, Flow::Return]),
        Err(AddressBuildError::TransferWithoutGivenEdges { address: 0 })
    );
}

#[test]
fn a_continuation_without_given_edges_to_no_instruction_is_an_error() {
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
fn a_given_source_or_target_must_start_an_instruction() {
    let edges = BTreeMap::from([(7, Vec::new()), (1, Vec::new())]);
    assert_eq!(
        build(edges, vec![Flow::Next, Flow::Return]),
        Err(AddressBuildError::MissingGivenSource { address: 7 })
    );
    let edges = BTreeMap::from([(0, vec![given(0, 5, EdgeKind::Jump)]), (1, Vec::new())]);
    assert_eq!(
        build(edges, vec![Flow::Jump { target: 5 }, Flow::Return]),
        Err(AddressBuildError::MissingGivenTarget {
            source: 0,
            target: 5
        })
    );
}
