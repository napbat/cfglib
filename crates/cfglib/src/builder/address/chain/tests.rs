extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::HandlerChain;
use crate::EdgeKind;
use crate::builder::address::tests::{Inst, inst, payload};
use crate::builder::address::{
    AddressBuildError, AddressCfgOptions, AddressGraph, AddressHandler, build_address_cfg,
    build_address_cfg_with_chains,
};
use crate::flow::Flow;
use crate::region::HandlerKind;

fn chain_stream() -> Vec<Inst> {
    vec![
        inst(0, Flow::Next),
        inst(1, Flow::Return),
        inst(10, Flow::Next),
        inst(11, Flow::Return),
        inst(20, Flow::Return),
        inst(30, Flow::Next),
        inst(31, Flow::Return),
    ]
}

fn chain_handlers() -> [AddressHandler<u32>; 2] {
    [
        AddressHandler {
            protected: 0..1,
            entry: 10,
            kind: HandlerKind::Finally,
        },
        AddressHandler {
            protected: 0..1,
            entry: 20,
            kind: HandlerKind::Catch,
        },
    ]
}

type Payload = (u32, u32, EdgeKind, &'static str);

fn resume_edges(graph: &AddressGraph<Inst, Payload>) -> Vec<(u32, u32)> {
    let mut edges: Vec<_> = graph
        .cfg
        .edges()
        .map(|edge| *edge.payload())
        .filter(|&(_, _, kind, role)| role == "resume" && kind == EdgeKind::ExceptionUnwind)
        .map(|(source, target, _, _)| (source, target))
        .collect();
    edges.sort_unstable();
    edges
}

#[test]
fn the_return_of_each_funclet_enters_the_next_funclet_and_the_last_enters_the_catch() {
    let chains = [HandlerChain {
        handler: 0,
        then: vec![30],
    }];

    let graph = build_address_cfg_with_chains(
        chain_stream(),
        &chain_handlers(),
        &chains,
        AddressCfgOptions::default(),
        payload,
    )
    .unwrap();

    // The protected instruction keeps its unwind edges to the first funclet
    // and to the catch handler. The returns chain the funclets, and the
    // catch follows each funclet: the second one can belong to an outer
    // scope, which the runtime does not run before the catch.
    assert_eq!(resume_edges(&graph), [(11, 20), (11, 30), (31, 20)]);
    let mut unwinds: Vec<_> = graph
        .cfg
        .edges()
        .map(|edge| *edge.payload())
        .filter(|&(_, _, _, role)| role == "unwind")
        .map(|(source, target, _, _)| (source, target))
        .collect();
    unwinds.sort_unstable();
    assert_eq!(unwinds, [(0, 10), (0, 20)]);
}

#[test]
fn every_return_of_a_funclet_with_a_branch_enters_the_next_step() {
    let instructions = vec![
        inst(0, Flow::Next),
        inst(1, Flow::Return),
        inst(10, Flow::Conditional { target: 12 }),
        inst(11, Flow::Return),
        inst(12, Flow::Return),
        inst(30, Flow::Return),
    ];
    let handlers = [AddressHandler {
        protected: 0..1,
        entry: 10,
        kind: HandlerKind::Finally,
    }];
    let chains = [HandlerChain {
        handler: 0,
        then: vec![30],
    }];

    let graph = build_address_cfg_with_chains(
        instructions,
        &handlers,
        &chains,
        AddressCfgOptions::default(),
        payload,
    )
    .unwrap();

    // Without a catch handler in the range, the last funclet only returns:
    // the exception propagates.
    assert_eq!(resume_edges(&graph), [(11, 30), (12, 30)]);
}

#[test]
fn a_graph_without_a_chain_has_no_resume_edge() {
    let graph = build_address_cfg(
        chain_stream(),
        &chain_handlers(),
        AddressCfgOptions::default(),
        payload,
    )
    .unwrap();

    assert_eq!(resume_edges(&graph), Vec::<(u32, u32)>::new());
}

#[test]
fn a_chain_with_an_unknown_handler_or_a_missing_entry_is_an_error() {
    let build = |chains: &[HandlerChain<u32>]| {
        build_address_cfg_with_chains(
            chain_stream(),
            &chain_handlers(),
            chains,
            AddressCfgOptions::default(),
            payload,
        )
        .map(|_| ())
    };

    assert_eq!(
        build(&[HandlerChain {
            handler: 5,
            then: Vec::new()
        }]),
        Err(AddressBuildError::UnknownChainHandler { handler: 5 })
    );
    assert_eq!(
        build(&[HandlerChain {
            handler: 0,
            then: vec![99]
        }]),
        Err(AddressBuildError::MissingChainEntry { address: 99 })
    );
}

#[test]
fn a_chain_makes_each_return_a_block_of_its_own() {
    let chains = [HandlerChain {
        handler: 0,
        then: vec![30],
    }];
    let with_chain = build_address_cfg_with_chains(
        chain_stream(),
        &chain_handlers(),
        &chains,
        AddressCfgOptions::default(),
        payload,
    )
    .unwrap();
    let without_chain = build_address_cfg(
        chain_stream(),
        &chain_handlers(),
        AddressCfgOptions::default(),
        payload,
    )
    .unwrap();

    let held = |graph: &AddressGraph<Inst, Payload>| {
        graph
            .cfg
            .block(graph.instruction_blocks[&11])
            .instructions()
            .len()
    };
    assert_eq!(held(&with_chain), 1);
    assert_eq!(held(&without_chain), 2);
}

#[test]
fn a_funclet_that_ends_in_a_tail_jump_enters_the_next_step() {
    // The first funclet adjusts a value and jumps to code outside the stream.
    // The jump is a tail call: the callee returns to the runtime. The second
    // funclet jumps into the stream, so it is no tail call.
    let instructions = vec![
        inst(0, Flow::Next),
        inst(1, Flow::Return),
        inst(10, Flow::Next),
        inst(11, Flow::Jump { target: 999 }),
        inst(30, Flow::Conditional { target: 32 }),
        inst(31, Flow::Return),
        inst(32, Flow::Jump { target: 31 }),
    ];
    let handlers = [AddressHandler {
        protected: 0..1,
        entry: 10,
        kind: HandlerKind::Finally,
    }];
    let chains = [HandlerChain {
        handler: 0,
        then: vec![30],
    }];

    let graph = build_address_cfg_with_chains(
        instructions,
        &handlers,
        &chains,
        AddressCfgOptions::default(),
        payload,
    )
    .unwrap();

    // The tail jump at 11 enters the funclet at 30, and the return at 31
    // leaves. The jump at 32 has an edge, so it ends no funclet.
    assert_eq!(resume_edges(&graph), [(11, 30)]);
    assert_eq!(
        graph
            .cfg
            .block(graph.instruction_blocks[&11])
            .instructions()
            .len(),
        1
    );
}

#[test]
fn two_chains_that_share_a_funclet_add_one_resume_edge_for_each_pair() {
    // Two runs of one protected range give two handlers that enter the same
    // funclets.
    let handlers = [
        AddressHandler {
            protected: 0..1,
            entry: 10,
            kind: HandlerKind::Finally,
        },
        AddressHandler {
            protected: 1..2,
            entry: 10,
            kind: HandlerKind::Finally,
        },
    ];
    let chains = [
        HandlerChain {
            handler: 0,
            then: vec![30],
        },
        HandlerChain {
            handler: 1,
            then: vec![30],
        },
    ];
    let instructions = vec![
        inst(0, Flow::Next),
        inst(1, Flow::Next),
        inst(2, Flow::Return),
        inst(10, Flow::Return),
        inst(30, Flow::Return),
    ];

    let graph = build_address_cfg_with_chains(
        instructions,
        &handlers,
        &chains,
        AddressCfgOptions::default(),
        payload,
    )
    .unwrap();

    assert_eq!(resume_edges(&graph), [(10, 30)]);
}

#[test]
fn the_last_funclet_enters_the_catch_handlers_whose_range_overlaps_its_own() {
    let instructions = vec![
        inst(0, Flow::Next),
        inst(1, Flow::Next),
        inst(2, Flow::Return),
        inst(10, Flow::Return),
        inst(20, Flow::Return),
        inst(40, Flow::Return),
    ];
    let handlers = [
        AddressHandler {
            protected: 0..1,
            entry: 10,
            kind: HandlerKind::Finally,
        },
        // Overlaps the range of the cleanup, and is wider than it.
        AddressHandler {
            protected: 0..2,
            entry: 20,
            kind: HandlerKind::Catch,
        },
        // Protects another instruction.
        AddressHandler {
            protected: 1..2,
            entry: 40,
            kind: HandlerKind::Catch,
        },
    ];
    let chains = [HandlerChain {
        handler: 0,
        then: Vec::new(),
    }];

    let graph = build_address_cfg_with_chains(
        instructions,
        &handlers,
        &chains,
        AddressCfgOptions::default(),
        payload,
    )
    .unwrap();

    assert_eq!(resume_edges(&graph), [(10, 20)]);
}
