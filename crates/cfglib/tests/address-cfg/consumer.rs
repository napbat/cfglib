//! What a consumer does with the result: choose its own edge payload, build
//! twice and expect the same graph, filter exceptional flow out of a view,
//! and print the text form.

use cfglib::{
    AddressEdgeInfo, CallPolicy, Cfg, DominatorTree, Edge, EdgeId, EdgeKind, EdgeRole,
    ExceptionDisposition, ExceptionFlow, ExceptionPhase, FilteredEdges, HandlerKind,
    build_address_cfg, verify, verify_edge_view,
};

use crate::bytecode::Isa32;
use crate::harness::{
    Built, Settings, TryRange, build, build_with, catch, handlers_of, plain_options,
};
use crate::isa::{Isa, Layout, Op};
use crate::transfers::kitchen_sink;

/// A function with every transfer kind and two protected ranges, one of them
/// holding a call and a dispatch.
fn function_with_handlers() -> (Vec<Op>, Vec<TryRange>) {
    (
        kitchen_sink(),
        vec![
            catch(1..3, 7),
            TryRange {
                protected: 3..6,
                entry: 9,
                kind: HandlerKind::Finally,
            },
        ],
    )
}

/// One protected range over two faulting instructions, a return, and a
/// handler body.
fn guarded_function() -> (Vec<Op>, Vec<TryRange>) {
    (
        vec![
            Op::Pure,
            Op::Work,
            Op::Work,
            Op::Return,
            Op::Work,
            Op::Return,
        ],
        vec![catch(1..3, 4)],
    )
}

fn assert_same_graph<I: Isa>(first: &Built<I>, second: &Built<I>) {
    assert_eq!(first.graph.cfg, second.graph.cfg);
    assert_eq!(first.graph.cfg.to_text(), second.graph.cfg.to_text());
    assert_eq!(
        first.graph.instruction_blocks,
        second.graph.instruction_blocks
    );
    assert_eq!(first.graph.handler_refs, second.graph.handler_refs);
    assert_eq!(
        first.graph.unresolved_transfers,
        second.graph.unresolved_transfers
    );
    assert_eq!(first.graph.calls, second.graph.calls);
}

fn building_the_same_stream_twice_gives_identical_graphs<I: Isa>() {
    let (ops, table) = function_with_handlers();
    for calls in [CallPolicy::Edges, CallPolicy::Flatten] {
        let settings = Settings { entry: None, calls };
        let first = build_with::<I>(&ops, table.clone(), settings);
        let second = build_with::<I>(&ops, table.clone(), settings);
        assert_same_graph(&first, &second);
    }
    let first = build::<I>(&kitchen_sink());
    assert_same_graph(&first, &build::<I>(&kitchen_sink()));
}

fn the_edge_payload_type_does_not_change_the_graph<I: Isa>() {
    let (ops, table) = function_with_handlers();
    let with_wire = build_with::<I>(&ops, table.clone(), Settings::default());

    let layout = Layout::<I>::assemble(&ops);
    let bare = build_address_cfg(
        layout.instructions().to_vec(),
        &handlers_of(&layout, &table),
        plain_options(),
        |_| (),
    )
    .unwrap();
    assert_eq!(bare.cfg.to_text(), with_wire.graph.cfg.to_text());
    assert_eq!(bare.cfg.edge_count(), with_wire.graph.cfg.edge_count());
    assert_eq!(bare.instruction_blocks, with_wire.graph.instruction_blocks);
    assert_eq!(bare.handler_refs, with_wire.graph.handler_refs);
    assert!(verify(&bare.cfg).is_ok());
}

fn each_edge_is_described_once_with_its_instruction_addresses_and_kind<I: Isa>() {
    let (ops, table) = guarded_function();
    let layout = Layout::<I>::assemble(&ops);
    let mut described = Vec::new();
    let graph = build_address_cfg(
        layout.instructions().to_vec(),
        &handlers_of(&layout, &table),
        plain_options(),
        |info: AddressEdgeInfo<'_, I::Addr, I::Key>| {
            described.push((
                layout.position_of(info.source).unwrap(),
                layout.position_of(info.target).unwrap(),
                info.kind,
                info.role.kind(),
            ));
        },
    )
    .unwrap();
    // Three sequential edges, then the two unwind edges the table adds.
    let expected = [
        (0, 1, EdgeKind::Fallthrough),
        (1, 2, EdgeKind::Fallthrough),
        (2, 3, EdgeKind::Fallthrough),
        (1, 4, EdgeKind::ExceptionUnwind),
        (2, 4, EdgeKind::ExceptionUnwind),
    ];
    assert_eq!(described.len(), expected.len());
    assert_eq!(graph.cfg.edge_count(), expected.len());
    for (source, target, kind) in expected {
        assert!(
            described.contains(&(source, target, kind, kind)),
            "{source} -> {target} {kind} is described, and its role maps to the same kind"
        );
    }
}

fn exception_flow_payloads_separate_normal_flow_from_handler_flow<I: Isa>() {
    let (ops, table) = guarded_function();
    let layout = Layout::<I>::assemble(&ops);
    // The elevator shape: the payload says which edges are exceptional, and
    // carries the handler-table index for the ones that are.
    let graph = build_address_cfg(
        layout.instructions().to_vec(),
        &handlers_of(&layout, &table),
        plain_options(),
        |info: AddressEdgeInfo<'_, I::Addr, I::Key>| match info.role {
            EdgeRole::Unwind { handler } => ExceptionFlow::exceptional(
                ExceptionPhase::Unwind,
                Some(ExceptionDisposition::ExecuteHandler),
                handler,
            ),
            _ => ExceptionFlow::normal(usize::MAX),
        },
    )
    .unwrap();
    let cfg: &Cfg<_, ExceptionFlow<usize>> = &graph.cfg;
    let handler_block = graph.instruction_blocks[&layout.address(4)];
    let after_try = graph.instruction_blocks[&layout.address(3)];

    // Payload and structural kind agree on which edges are exceptional.
    for edge in cfg.edges() {
        assert_eq!(
            edge.payload().phase().is_some(),
            edge.kind().is_exceptional()
        );
    }
    assert_eq!(cfg.edges().filter(|e| e.kind().is_exceptional()).count(), 2);
    for edge in cfg.edges().filter(|edge| edge.kind().is_exceptional()) {
        assert_eq!(
            edge.payload().disposition(),
            Some(ExceptionDisposition::ExecuteHandler)
        );
        assert_eq!(*edge.payload().metadata(), 0);
        assert_eq!(edge.target(), handler_block);
    }

    let full = DominatorTree::compute(cfg);
    assert!(full.is_reachable(handler_block));
    let normal = FilteredEdges::new(cfg, |_: EdgeId, edge: &Edge<ExceptionFlow<usize>>| {
        edge.payload().phase().is_none()
    });
    assert!(verify_edge_view(&normal).is_ok());
    let normal_only = DominatorTree::compute(&normal);
    assert!(normal_only.is_reachable(after_try));
    assert!(
        !normal_only.is_reachable(handler_block),
        "without unwind edges nothing reaches the handler"
    );
}

#[test]
fn the_text_form_lists_blocks_edges_and_regions_of_a_built_graph() {
    let (ops, table) = guarded_function();
    let built = build_with::<Isa32>(&ops, table, Settings::default());
    // Edges print in insertion order: normal edges per block, then the
    // unwind edges per table entry.
    let expected = "\
entry bb0
bb0:
    push
bb1:
    load
bb2:
    load
bb3:
    ret
bb4:
    load
    ret
bb0 -> bb1 fallthrough
bb1 -> bb2 fallthrough
bb2 -> bb3 fallthrough
bb1 -> bb4 unwind
bb2 -> bb4 unwind
region r0 protected=bb1,bb2
    handler catch entry=bb4 body=?
";
    assert_eq!(built.cfg().to_text(), expected);
}

over_both_isas!(
    building_the_same_stream_twice_gives_identical_graphs,
    the_edge_payload_type_does_not_change_the_graph,
    each_edge_is_described_once_with_its_instruction_addresses_and_kind,
    exception_flow_payloads_separate_normal_flow_from_handler_flow,
);
