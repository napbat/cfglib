//! Where instructions sit in the address space: gaps, overlapping decodes,
//! the top of the space, and a consumer-defined address type.

use cfglib::{
    AddressBuildError, AddressCfgOptions, AddressGraph, AddressHandler, AddressInstruction,
    AddressSpace, BlockId, EdgeKind, Flow, HandlerKind, HandlerRef, build_address_cfg, verify,
};

use crate::bytecode::{Bytecode, Opcode};
use crate::harness::{
    FALL, NOT_TAKEN, TAKEN, UNWIND, assert_empty, plain_options as no_options, try_build,
};
use crate::isa::{Isa, Op};
use crate::machine::{Machine, Mnemonic};
/// Encodes `op`, which names no destination, at `offset` bytes above the
/// start of the code.
fn place<I: Isa>(offset: u64, op: &Op) -> I::Insn {
    I::encode(I::at(offset), op, &|dest| {
        panic!("raw placement takes operations without destinations, not {dest:?}")
    })
}

/// Every block's outgoing edges as `(target block index, kind)`.
fn successors<I: AddressInstruction, E>(graph: &AddressGraph<I, E>) -> Vec<Vec<(usize, EdgeKind)>> {
    graph
        .cfg
        .block_ids()
        .map(|block| {
            graph
                .cfg
                .outgoing(block)
                .map(|edge| {
                    let edge = graph.cfg.edge(edge);
                    (edge.target().index(), edge.kind())
                })
                .collect()
        })
        .collect()
}

fn a_fall_through_does_not_cross_an_address_gap<I: Isa>() {
    // The work ends at offset 0x13 and the next instruction starts at 0x40:
    // nothing is at 0x13 for control to continue into.
    let stream = vec![place::<I>(0x10, &Op::Work), place::<I>(0x40, &Op::Return)];
    let graph = try_build(stream, &[], no_options()).unwrap();
    assert_eq!(graph.cfg.block_count(), 2);
    assert_eq!(graph.cfg.edge_count(), 0);
    assert_empty(&graph.unresolved_transfers);
    assert!(verify(&graph.cfg).is_ok());
    assert_eq!(graph.instruction_blocks.len(), 2);
}

fn overlapping_decodes_continue_at_each_instructions_own_end<I: Isa>() {
    // Two decodes of the same bytes: A (0..3) continues into C at 3, while
    // B (2..5) starts inside A and continues into D at 5.
    let stream = || {
        vec![
            place::<I>(0, &Op::Work),
            place::<I>(2, &Op::Work),
            place::<I>(3, &Op::Return),
            place::<I>(5, &Op::Return),
        ]
    };
    let graph = try_build(stream(), &[], no_options()).unwrap();
    let blocks: Vec<BlockId> = [0, 2, 3, 5]
        .map(|offset| graph.instruction_blocks[&I::at(offset)])
        .to_vec();
    let distinct: std::collections::BTreeSet<_> = blocks.iter().collect();
    assert_eq!(distinct.len(), 4, "every overlapping decode leads a block");
    assert_eq!(
        successors(&graph),
        vec![vec![(2, FALL)], vec![(3, FALL)], vec![], vec![]]
    );
    assert!(verify(&graph.cfg).is_ok());

    // Entering at the inner decode follows the other path through the same
    // blocks.
    let options = AddressCfgOptions {
        entry: Some(I::at(2)),
        calls: cfglib::CallPolicy::Edges,
    };
    let inner = try_build(stream(), &[], options).unwrap();
    assert_eq!(successors(&inner), successors(&graph));
    assert_eq!(inner.cfg.entry(), inner.instruction_blocks[&I::at(2)]);
}

over_both_isas!(
    a_fall_through_does_not_cross_an_address_gap,
    overlapping_decodes_continue_at_each_instructions_own_end,
);

#[test]
fn a_range_may_end_at_the_last_representable_address_of_the_u32_space() {
    let top = u32::MAX;
    let stream = vec![
        Bytecode {
            offset: top - 4,
            opcode: Opcode::Load(0),
        },
        Bytecode {
            offset: top - 1,
            opcode: Opcode::Ret,
        },
    ];
    let handler = AddressHandler {
        protected: top - 4..top,
        entry: top - 1,
        kind: HandlerKind::Catch,
    };
    let graph = try_build(stream, &[handler], no_options()).unwrap();
    assert_eq!(graph.cfg.regions().len(), 1);
    assert_eq!(graph.cfg.regions()[0].protected_blocks.len(), 2);
    // The load falls through into the handler's block and also unwinds to it.
    assert_eq!(
        successors(&graph),
        vec![vec![(1, FALL), (1, UNWIND)], vec![]]
    );
    assert!(verify(&graph.cfg).is_ok());
}

#[test]
fn a_range_may_end_at_the_last_representable_address_of_the_u64_space() {
    let top = u64::MAX;
    let stream = vec![
        Machine {
            address: top - 4,
            length: 3,
            mnemonic: Mnemonic::Mov,
            faults: true,
        },
        Machine {
            address: top - 1,
            length: 1,
            mnemonic: Mnemonic::Ret,
            faults: false,
        },
    ];
    let handler = AddressHandler {
        protected: top - 4..top,
        entry: top - 1,
        kind: HandlerKind::Catch,
    };
    let graph = try_build(stream, &[handler], no_options()).unwrap();
    assert_eq!(graph.cfg.regions().len(), 1);
    assert_eq!(graph.cfg.regions()[0].protected_blocks.len(), 2);
    assert_eq!(
        successors(&graph),
        vec![vec![(1, FALL), (1, UNWIND)], vec![]]
    );
    assert!(verify(&graph.cfg).is_ok());
}

/// A relative virtual address: a consumer type that is not a primitive
/// integer, so the builder may rely only on the `AddressSpace` contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Rva(u32);

impl AddressSpace for Rva {
    fn distance_from(self, earlier: Self) -> u64 {
        u64::from(self.0 - earlier.0)
    }
}

#[derive(Debug, Clone)]
struct Thunk {
    rva: Rva,
    size: u32,
    flow: Flow<Rva, ()>,
}

impl AddressInstruction for Thunk {
    type Address = Rva;
    type CaseKey = ();

    fn address(&self) -> Rva {
        self.rva
    }

    fn end_address(&self) -> Option<Rva> {
        self.rva.0.checked_add(self.size).map(Rva)
    }

    fn flow(&self) -> Flow<Rva, ()> {
        self.flow.clone()
    }
}

#[test]
fn a_consumer_defined_address_type_orders_nested_regions_by_its_own_distance() {
    let thunk = |rva: u32, flow: Flow<Rva, ()>| Thunk {
        rva: Rva(rva),
        size: 4,
        flow,
    };
    let stream = vec![
        thunk(0x1000, Flow::Next),
        thunk(0x1004, Flow::Next),
        thunk(
            0x1008,
            Flow::Conditional {
                target: Rva(0x1010),
            },
        ),
        thunk(0x100c, Flow::Next),
        thunk(0x1010, Flow::Return),
    ];
    // Table order is outer, inner: the inner range lies within the outer one.
    let handlers = [
        AddressHandler {
            protected: Rva(0x1004)..Rva(0x100c),
            entry: Rva(0x1010),
            kind: HandlerKind::Catch,
        },
        AddressHandler {
            protected: Rva(0x1004)..Rva(0x1008),
            entry: Rva(0x1010),
            kind: HandlerKind::CatchAll,
        },
    ];
    let graph = build_address_cfg(stream, &handlers, no_options(), |info| info.kind).unwrap();
    assert_eq!(
        successors(&graph),
        vec![
            vec![(1, FALL)],
            vec![(2, FALL), (4, UNWIND), (4, UNWIND)],
            vec![(4, TAKEN), (3, NOT_TAKEN), (4, UNWIND)],
            vec![(4, FALL)],
            vec![],
        ]
    );
    let regions = graph.cfg.regions();
    assert_eq!(regions.len(), 2);
    assert_eq!(regions[0].protected_blocks.len(), 2);
    assert_eq!(regions[1].protected_blocks.len(), 1);
    assert_eq!(regions[0].parent, None);
    assert_eq!(regions[1].parent, Some(regions[0].id));
    assert_eq!(
        graph.handler_refs,
        [
            HandlerRef::new(regions[0].id, 0),
            HandlerRef::new(regions[1].id, 0)
        ]
    );
    assert!(verify(&graph.cfg).is_ok());
}

#[test]
fn a_consumer_defined_address_type_reports_stream_errors_in_its_own_terms() {
    let stream = vec![Thunk {
        rva: Rva(0x1000),
        size: 0,
        flow: Flow::Next,
    }];
    assert_eq!(
        build_address_cfg(stream, &[], no_options(), |_| ()).err(),
        Some(AddressBuildError::ZeroSizeInstruction {
            address: Rva(0x1000)
        })
    );
}

#[test]
fn address_space_distance_is_the_non_negative_difference_for_every_primitive() {
    assert_eq!(7_u8.distance_from(2), 5);
    assert_eq!(500_u16.distance_from(500), 0);
    assert_eq!(0x1_0000_u32.distance_from(1), 0xFFFF);
    assert_eq!(u64::MAX.distance_from(0), u64::MAX);
    assert_eq!(0x40_usize.distance_from(0x10), 0x30);
}
