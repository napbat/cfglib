extern crate alloc;

use alloc::vec;

use super::{body, build, funclets, main_code, route, stream, try_build, wire};
use crate::builder::address::{
    AddressBuildError, AddressCfgOptions, AddressHandler, UnwindEnd, UnwindStep, build_address_cfg,
    build_address_cfg_with_routes,
};
use crate::edge::EdgeKind;
use crate::flow::Flow;
use crate::region::HandlerKind;

const RET: Flow<u32, i32> = Flow::Return;
const NEXT: Flow<u32, i32> = Flow::Next;

fn error_of(
    instructions: alloc::vec::Vec<crate::builder::address::tests::Inst>,
    routes: &[crate::builder::address::UnwindRoute<u32>],
) -> Option<AddressBuildError<u32>> {
    try_build(instructions, routes).err()
}

#[test]
fn a_protected_range_must_start_and_end_on_instruction_starts() {
    let code = || stream(&[main_code(2), body(10, &[RET])]);
    let paths = || vec![(funclets(&[10]), UnwindEnd::Propagate)];
    let reversed = (2, 1);

    assert_eq!(
        error_of(code(), &[route(5..6, paths())]),
        Some(AddressBuildError::MissingRangeStart { address: 5 }),
    );
    assert_eq!(
        error_of(code(), &[route(0..5, paths())]),
        Some(AddressBuildError::MissingRangeEnd { address: 5 }),
    );
    assert_eq!(
        error_of(code(), &[route(1..1, paths())]),
        Some(AddressBuildError::EmptyProtectedRange { start: 1, end: 1 }),
    );
    assert_eq!(
        error_of(code(), &[route(reversed.0..reversed.1, paths())]),
        Some(AddressBuildError::EmptyProtectedRange { start: 2, end: 1 }),
    );
    assert!(
        error_of(code(), &[route(0..2, paths())]).is_none(),
        "a range that ends at an instruction start is valid"
    );
}

#[test]
fn a_protected_range_can_end_at_the_end_of_the_code() {
    let instructions = main_code(1);
    let routes = [route(0..2, vec![(vec![], UnwindEnd::Propagate)])];
    let built = build(instructions, &routes);

    assert_eq!(built.hops.len(), 2, "both instructions are protected");
    assert_ne!(built.block_of(0), built.block_of(1));
}

#[test]
fn a_funclet_entry_or_handler_must_be_an_instruction_start() {
    let code = || stream(&[main_code(1), body(10, &[RET]), body(30, &[RET])]);

    assert_eq!(
        error_of(
            code(),
            &[route(
                0..1,
                vec![(funclets(&[10, 15]), UnwindEnd::Propagate)]
            )]
        ),
        Some(AddressBuildError::MissingRouteEntry { address: 15 }),
    );
    assert_eq!(
        error_of(
            code(),
            &[route(0..1, vec![(funclets(&[10]), UnwindEnd::Handler(31))])]
        ),
        Some(AddressBuildError::MissingRouteEntry { address: 31 }),
    );
    assert!(
        error_of(
            code(),
            &[route(0..1, vec![(funclets(&[10]), UnwindEnd::Handler(30))])]
        )
        .is_none()
    );
}

#[test]
fn a_resume_end_after_a_funclet_is_an_error() {
    let code = || stream(&[main_code(2), body(10, &[RET])]);
    let bad = route(1..2, vec![(funclets(&[10]), UnwindEnd::Resume)]);
    let good = route(0..1, vec![(vec![UnwindStep::Call(1)], UnwindEnd::Resume)]);

    assert_eq!(
        error_of(code(), &[good.clone(), bad.clone()]),
        Some(AddressBuildError::ResumeAfterFunclet { route: 1 }),
    );
    assert!(error_of(code(), &[good]).is_none());

    // Phase 2 rejects it too, for a graph that phase 1 built without the route.
    let mut graph =
        build_address_cfg_with_routes(code(), &[], &[], AddressCfgOptions::default(), wire)
            .expect("valid");
    assert_eq!(
        graph.add_unwind_routes(&[bad], wire).err(),
        Some(AddressBuildError::ResumeAfterFunclet { route: 0 }),
    );
}

/// Plain table graph with one block for the straight line at 0..=2.
fn plain_graph()
-> crate::builder::address::AddressGraph<crate::builder::address::tests::Inst, super::Wire> {
    build_address_cfg(
        stream(&[
            body(0, &[NEXT, NEXT, RET]),
            body(10, &[NEXT, RET]),
            body(20, &[RET]),
        ]),
        &[],
        AddressCfgOptions::default(),
        wire,
    )
    .expect("valid")
}

#[test]
fn phase_two_rejects_an_address_that_does_not_start_its_block() {
    let mut graph = plain_graph();
    let blocks = graph.cfg.block_count();
    let edges = graph.cfg.edge_count();
    let propagate = |funclets_: &[u32]| vec![(funclets(funclets_), UnwindEnd::Propagate)];

    // A protected instruction in the middle of a block.
    assert_eq!(
        graph
            .add_unwind_routes(&[route(1..2, propagate(&[20]))], wire)
            .err(),
        Some(AddressBuildError::RouteAddressNotBlockStart { address: 1 }),
    );
    // A funclet entry in the middle of a block.
    assert_eq!(
        graph
            .add_unwind_routes(&[route(0..1, propagate(&[10, 11]))], wire)
            .err(),
        Some(AddressBuildError::RouteAddressNotBlockStart { address: 11 }),
    );
    // A handler in the middle of a block.
    assert_eq!(
        graph
            .add_unwind_routes(
                &[route(0..1, vec![(funclets(&[10]), UnwindEnd::Handler(1))])],
                wire
            )
            .err(),
        Some(AddressBuildError::RouteAddressNotBlockStart { address: 1 }),
    );
    // An address that is no instruction.
    assert_eq!(
        graph
            .add_unwind_routes(&[route(0..1, propagate(&[5]))], wire)
            .err(),
        Some(AddressBuildError::MissingRouteEntry { address: 5 }),
    );
    assert_eq!(
        graph
            .add_unwind_routes(&[route(100..101, propagate(&[10]))], wire)
            .err(),
        Some(AddressBuildError::MissingRangeStart { address: 100 }),
    );
    assert_eq!(
        graph
            .add_unwind_routes(&[route(0..12, propagate(&[10]))], wire)
            .err(),
        Some(AddressBuildError::MissingRangeEnd { address: 12 }),
    );
    assert_eq!(
        (graph.cfg.block_count(), graph.cfg.edge_count()),
        (blocks, edges),
        "a failed call leaves the graph as it was"
    );
}

#[test]
fn phase_two_accepts_a_graph_whose_addresses_lead_their_blocks() {
    let mut graph = plain_graph();
    let hops = graph
        .add_unwind_routes(
            &[route(0..1, vec![(funclets(&[10]), UnwindEnd::Handler(20))])],
            wire,
        )
        .expect("each address leads its block");

    assert_eq!(
        hops.first().map(|hop| hop.calls.clone()),
        Some(vec![vec![]])
    );
}

#[test]
fn the_route_builder_adds_no_edge_from_the_handler_table() {
    let instructions = || stream(&[body(0, &[NEXT, RET]), body(10, &[RET])]);
    let handlers = [AddressHandler {
        protected: 0..1,
        entry: 10,
        kind: HandlerKind::Catch,
    }];
    let table = build_address_cfg(
        instructions(),
        &handlers,
        AddressCfgOptions::default(),
        wire,
    )
    .expect("valid");
    let routed = build_address_cfg_with_routes(
        instructions(),
        &handlers,
        &[],
        AddressCfgOptions::default(),
        wire,
    )
    .expect("valid");

    let unwind_edges = |graph: &crate::builder::address::AddressGraph<_, super::Wire>| {
        graph
            .cfg
            .edges()
            .filter(|edge| edge.data().kind() == EdgeKind::ExceptionUnwind)
            .count()
    };
    assert!(
        unwind_edges(&table) > 0,
        "the plain builder keeps the table edges"
    );
    assert_eq!(
        unwind_edges(&routed),
        0,
        "the route builder leaves them out"
    );
    assert_eq!(
        routed.handler_refs.len(),
        handlers.len(),
        "the table still gives the regions"
    );
    assert_eq!(routed.cfg.regions().len(), table.cfg.regions().len());
}

#[test]
fn the_route_builder_validates_the_handler_table() {
    let handlers = [AddressHandler {
        protected: 0..1,
        entry: 7,
        kind: HandlerKind::Catch,
    }];
    let result = build_address_cfg_with_routes(
        stream(&[body(0, &[NEXT, RET])]),
        &handlers,
        &[],
        AddressCfgOptions::default(),
        wire,
    );

    assert_eq!(
        result.err(),
        Some(AddressBuildError::MissingHandlerEntry { address: 7 })
    );
}

#[test]
fn routes_make_each_protected_instruction_and_each_funclet_return_lead_a_block() {
    let instructions = || stream(&[body(0, &[NEXT, NEXT, NEXT, RET]), body(10, &[NEXT, RET])]);
    let routes = [route(0..1, vec![(funclets(&[10]), UnwindEnd::Propagate)])];
    let guarded = build_address_cfg_with_routes(
        instructions(),
        &[],
        &routes,
        AddressCfgOptions::default(),
        wire,
    )
    .expect("valid");
    let plain =
        build_address_cfg_with_routes(instructions(), &[], &[], AddressCfgOptions::default(), wire)
            .expect("valid");

    let same_block =
        |graph: &crate::builder::address::AddressGraph<_, super::Wire>, a: u32, b: u32| {
            graph.instruction_blocks[&a] == graph.instruction_blocks[&b]
        };
    assert!(
        !same_block(&guarded, 0, 1),
        "a protected instruction leads its block"
    );
    assert!(
        same_block(&guarded, 1, 2),
        "an unprotected line stays one block"
    );
    assert!(
        !same_block(&guarded, 2, 3),
        "a return leads its block when a route exists"
    );
    assert!(
        !same_block(&guarded, 10, 11),
        "a funclet entry and its return are two blocks"
    );
    assert!(
        same_block(&plain, 1, 3),
        "with no route, a return keeps its block"
    );
}

#[test]
fn a_tail_jump_leads_a_block_when_a_route_exists() {
    let instructions = stream(&[main_code(1), body(10, &[NEXT, Flow::Jump { target: 900 }])]);
    let routes = [route(0..1, vec![(funclets(&[10]), UnwindEnd::Propagate)])];
    let built = build(instructions, &routes);

    assert_ne!(built.block_of(10), built.block_of(11));
}
