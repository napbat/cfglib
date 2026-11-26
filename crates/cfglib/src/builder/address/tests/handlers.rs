//! The handler table: protected ranges, regions, and handler kinds.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{inst, payload, sized_inst};
use crate::HandlerBody;
use crate::builder::address::{
    AddressBuildError, AddressCfgOptions, AddressHandler, AddressHandlerKind, AddressInstruction,
    build_address_cfg,
};
use crate::flow::Flow;
use crate::region::HandlerKind;

#[test]
fn exception_tables_make_protected_instructions_leaders_with_unwind_edges() {
    let instructions = vec![
        inst(0, Flow::Next),
        inst(1, Flow::Next),
        inst(2, Flow::Return),
        inst(3, Flow::Return),
    ];
    let handlers = [AddressHandler {
        protected: 0..2,
        entry: 3,
        kind: AddressHandlerKind::CatchAll,
    }];
    let graph = build_address_cfg(
        instructions,
        &handlers,
        AddressCfgOptions::default(),
        payload,
    )
    .unwrap();

    assert_ne!(
        graph.instruction_blocks[&0], graph.instruction_blocks[&1],
        "every protected instruction leads its own block"
    );
    let unwinds: Vec<_> = graph
        .cfg
        .edges()
        .filter(|edge| edge.payload().3 == "unwind")
        .map(|edge| edge.payload().0)
        .collect();
    assert_eq!(unwinds, vec![0, 1]);

    assert_eq!(graph.handler_refs.len(), 1);
    let regions = graph.cfg.regions();
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].handlers[0].entry, graph.instruction_blocks[&3]);
    assert_eq!(regions[0].handlers[0].body, HandlerBody::Unknown);
    assert_eq!(regions[0].protected_blocks.len(), 2);
}

#[test]
fn a_non_throwing_protected_instruction_keeps_no_unwind_edge() {
    let mut pure = inst(0, Flow::Next);
    pure.throws = false;
    let instructions = vec![pure, inst(1, Flow::Return), inst(2, Flow::Return)];
    let handlers = [AddressHandler {
        protected: 0..1,
        entry: 2,
        kind: AddressHandlerKind::CatchAll,
    }];
    let graph = build_address_cfg(
        instructions,
        &handlers,
        AddressCfgOptions::default(),
        payload,
    )
    .unwrap();
    assert!(graph.cfg.edges().all(|edge| edge.payload().3 != "unwind"),);
}

#[test]
fn nested_ranges_register_enclosing_regions_first() {
    let instructions = vec![
        inst(0, Flow::Next),
        inst(1, Flow::Next),
        inst(2, Flow::Return),
        inst(3, Flow::Return),
        inst(4, Flow::Return),
    ];
    // Table lists the nested range first; construction must still register
    // the enclosing range before it and wire the parent.
    let handlers = [
        AddressHandler {
            protected: 1..2,
            entry: 3,
            kind: AddressHandlerKind::CatchAll,
        },
        AddressHandler {
            protected: 0..3,
            entry: 4,
            kind: AddressHandlerKind::CatchAll,
        },
    ];
    let graph = build_address_cfg(
        instructions,
        &handlers,
        AddressCfgOptions::default(),
        payload,
    )
    .unwrap();
    let regions = graph.cfg.regions();
    assert_eq!(regions.len(), 2);
    assert_eq!(
        regions[0].protected_blocks.len(),
        3,
        "outer registered first"
    );
    assert_eq!(regions[1].protected_blocks.len(), 1);
    assert_eq!(regions[1].parent, Some(regions[0].id));
    // Table order survives in the returned refs.
    assert_eq!(graph.handler_refs[0].region(), regions[1].id);
    assert_eq!(graph.handler_refs[1].region(), regions[0].id);
}

#[test]
fn shared_protected_ranges_share_one_region() {
    let instructions = vec![
        inst(0, Flow::Return),
        inst(1, Flow::Return),
        inst(2, Flow::Return),
    ];
    let handlers = [
        AddressHandler {
            protected: 0..1,
            entry: 1,
            kind: AddressHandlerKind::Catch,
        },
        AddressHandler {
            protected: 0..1,
            entry: 2,
            kind: AddressHandlerKind::CatchAll,
        },
    ];
    let graph = build_address_cfg(
        instructions,
        &handlers,
        AddressCfgOptions::default(),
        payload,
    )
    .unwrap();
    assert_eq!(graph.cfg.regions().len(), 1);
    assert_eq!(graph.cfg.regions()[0].handlers.len(), 2);
    assert_eq!(
        graph.handler_refs[0].region(),
        graph.handler_refs[1].region()
    );
    assert_eq!(graph.handler_refs[0].index(), 0);
    assert_eq!(graph.handler_refs[1].index(), 1);
}

#[test]
fn a_range_ending_at_code_end_needs_no_end_instruction() {
    let instructions = vec![inst(0, Flow::Next), inst(1, Flow::Return)];
    let handlers = [AddressHandler {
        protected: 0..2,
        entry: 1,
        kind: AddressHandlerKind::CatchAll,
    }];
    assert!(
        build_address_cfg(
            instructions,
            &handlers,
            AddressCfgOptions::default(),
            payload
        )
        .is_ok()
    );
}

#[test]
fn a_table_filter_resolves_to_the_block_that_starts_at_its_address() {
    // Without the filter, 2..=4 is one straight-line block.
    let instructions = vec![
        inst(0, Flow::Next),
        inst(1, Flow::Return),
        inst(2, Flow::Next),
        inst(3, Flow::Next),
        inst(4, Flow::Return),
        inst(5, Flow::Return),
    ];
    let handlers = [AddressHandler {
        protected: 0..1,
        entry: 5,
        kind: AddressHandlerKind::Filter { filter: 3 },
    }];
    let graph = build_address_cfg(
        instructions,
        &handlers,
        AddressCfgOptions::default(),
        payload,
    )
    .unwrap();

    assert_ne!(
        graph.instruction_blocks[&2], graph.instruction_blocks[&3],
        "the filter address leads a block"
    );
    let filter_block = graph.instruction_blocks[&3];
    assert_eq!(
        graph
            .cfg
            .block(filter_block)
            .instructions()
            .first()
            .map(AddressInstruction::address),
        Some(3),
        "the filter block starts at the filter address"
    );
    let handler = graph.cfg.handler(graph.handler_refs[0]).unwrap();
    assert_eq!(handler.kind, HandlerKind::Filter { filter_block });
    assert_eq!(handler.entry, graph.instruction_blocks[&5]);
}

#[test]
fn a_table_filter_off_an_instruction_start_is_a_missing_handler_entry() {
    let instructions = vec![
        sized_inst(0, 2, Flow::Next),
        inst(2, Flow::Return),
        inst(3, Flow::Return),
    ];
    let handlers = [AddressHandler {
        protected: 0..2,
        entry: 3,
        kind: AddressHandlerKind::Filter { filter: 1 },
    }];
    assert_eq!(
        build_address_cfg(
            instructions,
            &handlers,
            AddressCfgOptions::default(),
            payload
        )
        .unwrap_err(),
        AddressBuildError::MissingHandlerEntry { address: 1 }
    );
}
