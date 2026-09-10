extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::region::HandlerKind;
use crate::{EdgeKind, HandlerBody};

use super::{
    AddressBuildError, AddressEdgeRole, AddressFlow, AddressHandler, AddressInstruction,
    build_address_cfg,
};

#[derive(Debug, Clone)]
struct Inst {
    address: u32,
    size: u32,
    flow: AddressFlow<u32, i32>,
    throws: bool,
}

fn inst(address: u32, flow: AddressFlow<u32, i32>) -> Inst {
    Inst {
        address,
        size: 1,
        flow,
        throws: true,
    }
}

impl AddressInstruction for Inst {
    type Address = u32;
    type CaseKey = i32;

    fn address(&self) -> u32 {
        self.address
    }

    fn end_address(&self) -> Option<u32> {
        self.address.checked_add(self.size)
    }

    fn flow(&self) -> AddressFlow<u32, i32> {
        self.flow.clone()
    }

    fn retains_exception_edge(&self) -> bool {
        self.throws
    }
}

fn sized_inst(address: u32, size: u32, flow: AddressFlow<u32, i32>) -> Inst {
    Inst {
        address,
        size,
        flow,
        throws: true,
    }
}

/// Records `(source, target, kind, role tag)` per edge.
fn payload(info: super::AddressEdgeInfo<'_, u32, i32>) -> (u32, u32, EdgeKind, &'static str) {
    let role = match info.role {
        AddressEdgeRole::Sequential => "seq",
        AddressEdgeRole::ConditionalTaken => "taken",
        AddressEdgeRole::ConditionalFallThrough => "fall",
        AddressEdgeRole::Branch => "branch",
        AddressEdgeRole::SwitchDefault => "default",
        AddressEdgeRole::SwitchCase { .. } => "case",
        AddressEdgeRole::Call => "call",
        AddressEdgeRole::CallContinuation { .. } => "cont",
        AddressEdgeRole::Unwind { .. } => "unwind",
    };
    (info.source, info.target, info.kind, role)
}

#[test]
fn leaders_split_at_targets_and_after_terminators() {
    // 0: conditional -> 3; 1: fallthrough; 2: fallthrough; 3: return
    let instructions = vec![
        inst(0, AddressFlow::Conditional { target: 3 }),
        inst(1, AddressFlow::FallThrough),
        inst(2, AddressFlow::FallThrough),
        inst(3, AddressFlow::Return),
    ];
    let graph = build_address_cfg(instructions, &[], payload).unwrap();
    // Blocks: [0], [1, 2], [3].
    assert_eq!(graph.cfg.block_count(), 3);
    assert_eq!(
        graph.instruction_blocks[&1], graph.instruction_blocks[&2],
        "no leader splits a straight line"
    );
    assert_ne!(graph.instruction_blocks[&0], graph.instruction_blocks[&1]);
    assert_ne!(graph.instruction_blocks[&2], graph.instruction_blocks[&3]);

    let mut edges: Vec<_> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    edges.sort_unstable_by_key(|&(source, target, _, role)| (source, target, role));
    assert_eq!(
        edges,
        vec![
            (0, 1, EdgeKind::ConditionalFalse, "fall"),
            (0, 3, EdgeKind::ConditionalTrue, "taken"),
            (2, 3, EdgeKind::Fallthrough, "seq"),
        ],
    );
}

#[test]
fn switches_and_calls_produce_keyed_and_continuation_edges() {
    let instructions = vec![
        inst(
            0,
            AddressFlow::Switch {
                default: 3,
                cases: vec![(7, 2)],
            },
        ),
        inst(1, AddressFlow::Call { target: 3 }),
        inst(2, AddressFlow::Return),
        inst(3, AddressFlow::Return),
    ];
    let graph = build_address_cfg(instructions, &[], payload).unwrap();
    let mut edges: Vec<_> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    edges.sort_unstable_by_key(|&(source, target, _, role)| (source, target, role));
    assert_eq!(
        edges,
        vec![
            (0, 2, EdgeKind::SwitchCase, "case"),
            (0, 3, EdgeKind::SwitchCase, "default"),
            (1, 2, EdgeKind::CallReturn, "cont"),
            (1, 3, EdgeKind::Call, "call"),
        ],
    );
}

#[test]
fn a_call_ends_its_block_so_the_return_site_gets_both_edges() {
    // The instruction after the call is a leader only because the call
    // ends its block; absorbing it would silently drop both call edges.
    let instructions = vec![
        inst(0, AddressFlow::Call { target: 2 }),
        inst(1, AddressFlow::Return),
        inst(2, AddressFlow::Return),
    ];
    let graph = build_address_cfg(instructions, &[], payload).unwrap();
    assert_ne!(graph.instruction_blocks[&0], graph.instruction_blocks[&1]);
    let mut edges: Vec<_> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    edges.sort_unstable_by_key(|&(source, target, _, role)| (source, target, role));
    assert_eq!(
        edges,
        vec![
            (0, 1, EdgeKind::CallReturn, "cont"),
            (0, 2, EdgeKind::Call, "call"),
        ],
    );
}

#[test]
fn exception_tables_make_protected_instructions_leaders_with_unwind_edges() {
    let instructions = vec![
        inst(0, AddressFlow::FallThrough),
        inst(1, AddressFlow::FallThrough),
        inst(2, AddressFlow::Return),
        inst(3, AddressFlow::Return),
    ];
    let handlers = [AddressHandler {
        protected: 0..2,
        entry: 3,
        kind: HandlerKind::CatchAll,
    }];
    let graph = build_address_cfg(instructions, &handlers, payload).unwrap();

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
    let mut pure = inst(0, AddressFlow::FallThrough);
    pure.throws = false;
    let instructions = vec![
        pure,
        inst(1, AddressFlow::Return),
        inst(2, AddressFlow::Return),
    ];
    let handlers = [AddressHandler {
        protected: 0..1,
        entry: 2,
        kind: HandlerKind::CatchAll,
    }];
    let graph = build_address_cfg(instructions, &handlers, payload).unwrap();
    assert!(graph.cfg.edges().all(|edge| edge.payload().3 != "unwind"),);
}

#[test]
fn nested_ranges_register_enclosing_regions_first() {
    let instructions = vec![
        inst(0, AddressFlow::FallThrough),
        inst(1, AddressFlow::FallThrough),
        inst(2, AddressFlow::Return),
        inst(3, AddressFlow::Return),
        inst(4, AddressFlow::Return),
    ];
    // Table lists the nested range first; construction must still register
    // the enclosing range before it and wire the parent.
    let handlers = [
        AddressHandler {
            protected: 1..2,
            entry: 3,
            kind: HandlerKind::CatchAll,
        },
        AddressHandler {
            protected: 0..3,
            entry: 4,
            kind: HandlerKind::CatchAll,
        },
    ];
    let graph = build_address_cfg(instructions, &handlers, payload).unwrap();
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
        inst(0, AddressFlow::Return),
        inst(1, AddressFlow::Return),
        inst(2, AddressFlow::Return),
    ];
    let handlers = [
        AddressHandler {
            protected: 0..1,
            entry: 1,
            kind: HandlerKind::Catch,
        },
        AddressHandler {
            protected: 0..1,
            entry: 2,
            kind: HandlerKind::CatchAll,
        },
    ];
    let graph = build_address_cfg(instructions, &handlers, payload).unwrap();
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
fn overlapping_paths_use_each_instruction_end_for_fall_through() {
    let instructions = vec![
        sized_inst(0, 2, AddressFlow::Conditional { target: 3 }),
        sized_inst(2, 5, AddressFlow::FallThrough),
        inst(3, AddressFlow::FallThrough),
        inst(4, AddressFlow::FallThrough),
        inst(5, AddressFlow::FallThrough),
        inst(6, AddressFlow::FallThrough),
        inst(7, AddressFlow::Return),
    ];

    let graph = build_address_cfg(instructions, &[], payload).unwrap();

    assert_eq!(graph.cfg.block_count(), 4);
    assert_eq!(graph.instruction_blocks[&3], graph.instruction_blocks[&6]);
    assert_ne!(graph.instruction_blocks[&2], graph.instruction_blocks[&3]);
    assert_ne!(graph.instruction_blocks[&6], graph.instruction_blocks[&7]);
    let mut edges: Vec<_> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    edges.sort_unstable_by_key(|&(source, target, _, role)| (source, target, role));
    assert_eq!(
        edges,
        vec![
            (0, 2, EdgeKind::ConditionalFalse, "fall"),
            (0, 3, EdgeKind::ConditionalTrue, "taken"),
            (2, 7, EdgeKind::Fallthrough, "seq"),
            (6, 7, EdgeKind::Fallthrough, "seq"),
        ]
    );
}

#[test]
fn a_fall_through_does_not_cross_an_address_gap() {
    let instructions = vec![
        inst(0, AddressFlow::FallThrough),
        inst(2, AddressFlow::Return),
    ];

    let graph = build_address_cfg(instructions, &[], payload).unwrap();

    assert_eq!(graph.cfg.block_count(), 2);
    assert_eq!(graph.cfg.edge_count(), 0);
}

#[test]
fn invalid_streams_are_rejected_with_exact_errors() {
    let duplicate = vec![
        inst(0, AddressFlow::FallThrough),
        inst(0, AddressFlow::Return),
    ];
    assert_eq!(
        build_address_cfg(duplicate, &[], payload).unwrap_err(),
        AddressBuildError::UnorderedInstruction {
            previous: 0,
            address: 0,
        },
    );

    let missing_target = vec![inst(0, AddressFlow::Unconditional { target: 9 })];
    assert_eq!(
        build_address_cfg(missing_target, &[], payload).unwrap_err(),
        AddressBuildError::MissingBranchTarget {
            source: 0,
            target: 9
        },
    );

    let missing_handler = vec![inst(0, AddressFlow::Return)];
    assert_eq!(
        build_address_cfg(
            missing_handler,
            &[AddressHandler {
                protected: 0..1,
                entry: 9,
                kind: HandlerKind::CatchAll,
            }],
            payload,
        )
        .unwrap_err(),
        AddressBuildError::MissingHandlerEntry { address: 9 },
    );

    let empty_range = vec![inst(0, AddressFlow::Return)];
    assert_eq!(
        build_address_cfg(
            empty_range,
            &[AddressHandler {
                protected: 0..0,
                entry: 0,
                kind: HandlerKind::CatchAll,
            }],
            payload,
        )
        .unwrap_err(),
        AddressBuildError::EmptyProtectedRange { start: 0, end: 0 },
    );
}

#[test]
fn a_range_ending_at_code_end_needs_no_end_instruction() {
    let instructions = vec![
        inst(0, AddressFlow::FallThrough),
        inst(1, AddressFlow::Return),
    ];
    let handlers = [AddressHandler {
        protected: 0..2,
        entry: 1,
        kind: HandlerKind::CatchAll,
    }];
    assert!(build_address_cfg(instructions, &handlers, payload).is_ok());
}
