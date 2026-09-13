extern crate alloc;
extern crate std;

use alloc::vec;
use alloc::vec::Vec;
use std::cell::Cell;
use std::rc::Rc;

use crate::region::HandlerKind;
use crate::{EdgeKind, HandlerBody};

use super::{
    AddressBuildError, AddressCfgOptions, AddressHandler, AddressInstruction, CallPolicy,
    build_address_cfg,
};
use crate::flow::{CallSite, EdgeRole, Flow, UnresolvedRole, UnresolvedTransfer};

#[derive(Debug, Clone)]
struct Inst {
    address: u32,
    size: u32,
    flow: Flow<u32, i32>,
    throws: bool,
}

fn inst(address: u32, flow: Flow<u32, i32>) -> Inst {
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

    fn flow(&self) -> Flow<u32, i32> {
        self.flow.clone()
    }

    fn retains_exception_edge(&self) -> bool {
        self.throws
    }
}

fn sized_inst(address: u32, size: u32, flow: Flow<u32, i32>) -> Inst {
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
        EdgeRole::Sequential => "seq",
        EdgeRole::ConditionalTaken => "taken",
        EdgeRole::ConditionalFallThrough => "fall",
        EdgeRole::Jump => "branch",
        EdgeRole::SwitchDefault => "default",
        EdgeRole::SwitchCase { .. } => "case",
        EdgeRole::Call => "call",
        EdgeRole::CallContinuation { .. } => "cont",
        EdgeRole::Exceptional => "exceptional",
        EdgeRole::Unwind { .. } => "unwind",
    };
    (info.source, info.target, info.kind, role)
}

#[test]
fn leaders_split_at_targets_and_after_terminators() {
    // 0: conditional -> 3; 1: fallthrough; 2: fallthrough; 3: return
    let instructions = vec![
        inst(0, Flow::Conditional { target: 3 }),
        inst(1, Flow::Next),
        inst(2, Flow::Next),
        inst(3, Flow::Return),
    ];
    let graph =
        build_address_cfg(instructions, &[], AddressCfgOptions::default(), payload).unwrap();
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
            Flow::Switch {
                default: 3,
                cases: vec![(7, 2)],
            },
        ),
        inst(1, Flow::Call { target: 3 }),
        inst(2, Flow::Return),
        inst(3, Flow::Return),
    ];
    let graph =
        build_address_cfg(instructions, &[], AddressCfgOptions::default(), payload).unwrap();
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
        inst(0, Flow::Call { target: 2 }),
        inst(1, Flow::Return),
        inst(2, Flow::Return),
    ];
    let graph =
        build_address_cfg(instructions, &[], AddressCfgOptions::default(), payload).unwrap();
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
        inst(0, Flow::Next),
        inst(1, Flow::Next),
        inst(2, Flow::Return),
        inst(3, Flow::Return),
    ];
    let handlers = [AddressHandler {
        protected: 0..2,
        entry: 3,
        kind: HandlerKind::CatchAll,
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
        kind: HandlerKind::CatchAll,
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
            kind: HandlerKind::CatchAll,
        },
        AddressHandler {
            protected: 0..3,
            entry: 4,
            kind: HandlerKind::CatchAll,
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
            kind: HandlerKind::Catch,
        },
        AddressHandler {
            protected: 0..1,
            entry: 2,
            kind: HandlerKind::CatchAll,
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
fn overlapping_paths_use_each_instruction_end_for_fall_through() {
    let instructions = vec![
        sized_inst(0, 2, Flow::Conditional { target: 3 }),
        sized_inst(2, 5, Flow::Next),
        inst(3, Flow::Next),
        inst(4, Flow::Next),
        inst(5, Flow::Next),
        inst(6, Flow::Next),
        inst(7, Flow::Return),
    ];

    let graph =
        build_address_cfg(instructions, &[], AddressCfgOptions::default(), payload).unwrap();

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
    let instructions = vec![inst(0, Flow::Next), inst(2, Flow::Return)];

    let graph =
        build_address_cfg(instructions, &[], AddressCfgOptions::default(), payload).unwrap();

    assert_eq!(graph.cfg.block_count(), 2);
    assert_eq!(graph.cfg.edge_count(), 0);
}

#[test]
fn instruction_flow_is_classified_once() {
    struct CountedInst {
        inner: Inst,
        calls: Rc<Cell<usize>>,
    }

    impl AddressInstruction for CountedInst {
        type Address = u32;
        type CaseKey = i32;

        fn address(&self) -> u32 {
            self.inner.address()
        }

        fn end_address(&self) -> Option<u32> {
            self.inner.end_address()
        }

        fn flow(&self) -> Flow<u32, i32> {
            self.calls.set(self.calls.get() + 1);
            self.inner.flow()
        }

        fn retains_exception_edge(&self) -> bool {
            self.inner.retains_exception_edge()
        }
    }

    let calls = Rc::new(Cell::new(0));
    let instructions = [
        inst(0, Flow::Conditional { target: 2 }),
        inst(1, Flow::Next),
        inst(2, Flow::Return),
    ]
    .into_iter()
    .map(|inner| CountedInst {
        inner,
        calls: Rc::clone(&calls),
    })
    .collect();

    build_address_cfg(instructions, &[], AddressCfgOptions::default(), |_| ()).unwrap();
    assert_eq!(calls.get(), 3, "each instruction is classified once");
}

#[test]
fn invalid_streams_are_rejected_with_exact_errors() {
    let duplicate = vec![inst(0, Flow::Next), inst(0, Flow::Return)];
    assert_eq!(
        build_address_cfg(duplicate, &[], AddressCfgOptions::default(), payload).unwrap_err(),
        AddressBuildError::UnorderedInstruction {
            previous: 0,
            address: 0,
        },
    );

    let missing_handler = vec![inst(0, Flow::Return)];
    assert_eq!(
        build_address_cfg(
            missing_handler,
            &[AddressHandler {
                protected: 0..1,
                entry: 9,
                kind: HandlerKind::CatchAll,
            }],
            AddressCfgOptions::default(),
            payload
        )
        .unwrap_err(),
        AddressBuildError::MissingHandlerEntry { address: 9 },
    );

    let empty_range = vec![inst(0, Flow::Return)];
    assert_eq!(
        build_address_cfg(
            empty_range,
            &[AddressHandler {
                protected: 0..0,
                entry: 0,
                kind: HandlerKind::CatchAll,
            }],
            AddressCfgOptions::default(),
            payload
        )
        .unwrap_err(),
        AddressBuildError::EmptyProtectedRange { start: 0, end: 0 },
    );
}

#[test]
fn missing_conditional_target_keeps_fallthrough_and_reports_the_exit() {
    let instructions = vec![
        inst(0, Flow::Conditional { target: 9 }),
        inst(1, Flow::Return),
    ];
    let graph =
        build_address_cfg(instructions, &[], AddressCfgOptions::default(), payload).unwrap();

    let edges: Vec<_> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    assert_eq!(edges, vec![(0, 1, EdgeKind::ConditionalFalse, "fall")]);
    assert_eq!(
        graph.unresolved_transfers,
        vec![UnresolvedTransfer {
            source: 0,
            target: Some(9),
            role: UnresolvedRole::ConditionalTaken,
        }]
    );
}

#[test]
fn partial_switch_keeps_known_cases_and_reports_each_missing_destination() {
    let instructions = vec![
        inst(
            0,
            Flow::Switch {
                default: 9,
                cases: vec![(7, 2), (8, 10), (9, 3)],
            },
        ),
        inst(1, Flow::Return),
        inst(2, Flow::Return),
        inst(3, Flow::Return),
    ];
    let graph =
        build_address_cfg(instructions, &[], AddressCfgOptions::default(), payload).unwrap();

    let edges: Vec<_> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    assert_eq!(
        edges,
        vec![
            (0, 2, EdgeKind::SwitchCase, "case"),
            (0, 3, EdgeKind::SwitchCase, "case"),
        ]
    );
    assert_eq!(
        graph.unresolved_transfers,
        vec![
            UnresolvedTransfer {
                source: 0,
                target: Some(9),
                role: UnresolvedRole::SwitchDefault,
            },
            UnresolvedTransfer {
                source: 0,
                target: Some(10),
                role: UnresolvedRole::SwitchCase { index: 1 },
            },
        ]
    );
}

#[test]
fn requested_entry_is_independent_of_lowest_recovered_address() {
    let instructions = vec![inst(0, Flow::Return), inst(10, Flow::Jump { target: 0 })];
    let graph = build_address_cfg(
        instructions,
        &[],
        AddressCfgOptions {
            entry: Some(10),
            ..AddressCfgOptions::default()
        },
        payload,
    )
    .unwrap();

    assert_eq!(graph.cfg.entry(), graph.instruction_blocks[&10]);
    assert_ne!(graph.cfg.entry(), graph.instruction_blocks[&0]);
    assert_eq!(
        build_address_cfg(
            vec![inst(0, Flow::Return)],
            &[],
            AddressCfgOptions {
                entry: Some(7),
                ..AddressCfgOptions::default()
            },
            payload
        )
        .unwrap_err(),
        AddressBuildError::MissingEntry { address: 7 }
    );
}

#[test]
fn exceptional_target_keeps_normal_continuation() {
    let instructions = vec![
        inst(0, Flow::Exceptional { target: 2 }),
        inst(1, Flow::Return),
        inst(2, Flow::Return),
    ];
    let graph =
        build_address_cfg(instructions, &[], AddressCfgOptions::default(), payload).unwrap();

    let edges: Vec<_> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    assert_eq!(
        edges,
        vec![
            (0, 2, EdgeKind::ExceptionUnwind, "exceptional"),
            (0, 1, EdgeKind::Fallthrough, "seq"),
        ]
    );
}

#[test]
fn indirect_and_missing_exceptional_exits_are_explicit() {
    let graph = build_address_cfg(
        vec![
            inst(0, Flow::Indirect),
            inst(1, Flow::Exceptional { target: 9 }),
            inst(2, Flow::Return),
        ],
        &[],
        AddressCfgOptions::default(),
        payload,
    )
    .unwrap();

    assert_eq!(
        graph.unresolved_transfers,
        vec![
            UnresolvedTransfer {
                source: 0,
                target: None,
                role: UnresolvedRole::Indirect,
            },
            UnresolvedTransfer {
                source: 1,
                target: Some(9),
                role: UnresolvedRole::Exceptional,
            },
        ]
    );
    assert!(
        graph
            .cfg
            .edges()
            .any(|edge| { *edge.payload() == (1, 2, EdgeKind::Fallthrough, "seq") })
    );
}

#[test]
fn a_range_ending_at_code_end_needs_no_end_instruction() {
    let instructions = vec![inst(0, Flow::Next), inst(1, Flow::Return)];
    let handlers = [AddressHandler {
        protected: 0..2,
        entry: 1,
        kind: HandlerKind::CatchAll,
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
fn an_exceptional_instruction_leads_its_own_block() {
    let instructions = vec![
        inst(0, Flow::Next),
        inst(1, Flow::Exceptional { target: 3 }),
        inst(2, Flow::Return),
        inst(3, Flow::Return),
    ];
    let graph =
        build_address_cfg(instructions, &[], AddressCfgOptions::default(), payload).unwrap();

    assert_ne!(
        graph.instruction_blocks[&0], graph.instruction_blocks[&1],
        "the exceptional instruction starts its own block"
    );
    let mut edges: Vec<_> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    edges.sort_unstable_by_key(|&(source, target, _, role)| (source, target, role));
    assert_eq!(
        edges,
        vec![
            (0, 1, EdgeKind::Fallthrough, "seq"),
            (1, 2, EdgeKind::Fallthrough, "seq"),
            (1, 3, EdgeKind::ExceptionUnwind, "exceptional"),
        ]
    );
}

#[test]
fn flattened_calls_continue_in_their_block_and_are_reported() {
    let instructions = vec![
        inst(0, Flow::Call { target: 3 }),
        inst(1, Flow::IndirectCall),
        inst(2, Flow::Return),
        inst(3, Flow::Return),
    ];
    let options = AddressCfgOptions {
        entry: None,
        calls: CallPolicy::Flatten,
    };
    let graph = build_address_cfg(instructions, &[], options, payload).unwrap();

    assert_eq!(graph.instruction_blocks[&0], graph.instruction_blocks[&2]);
    assert_ne!(
        graph.instruction_blocks[&0], graph.instruction_blocks[&3],
        "a flattened call target is not a leader"
    );
    assert!(
        graph.cfg.edges().next().is_none(),
        "no edge leaves the caller's block"
    );
    assert_eq!(
        graph.calls,
        vec![
            CallSite {
                source: 0,
                target: Some(3)
            },
            CallSite {
                source: 1,
                target: None
            },
        ]
    );
    assert!(graph.unresolved_transfers.is_empty());
}

#[test]
fn a_flattened_terminator_call_continues_sequentially() {
    let instructions = vec![
        inst(0, Flow::Conditional { target: 2 }),
        inst(1, Flow::Call { target: 9 }),
        inst(2, Flow::Return),
    ];
    let options = AddressCfgOptions {
        entry: None,
        calls: CallPolicy::Flatten,
    };
    let graph = build_address_cfg(instructions, &[], options, payload).unwrap();

    let mut edges: Vec<_> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    edges.sort_unstable_by_key(|&(source, target, _, role)| (source, target, role));
    assert_eq!(
        edges,
        vec![
            (0, 1, EdgeKind::ConditionalFalse, "fall"),
            (0, 2, EdgeKind::ConditionalTrue, "taken"),
            (1, 2, EdgeKind::Fallthrough, "seq"),
        ]
    );
    assert_eq!(
        graph.calls,
        vec![CallSite {
            source: 1,
            target: Some(9)
        }]
    );
    assert!(
        graph.unresolved_transfers.is_empty(),
        "a flattened call to an undecoded callee is a call site, not a missing edge"
    );
}

#[test]
fn an_indirect_call_kept_as_an_edge_is_an_unresolved_transfer() {
    let instructions = vec![inst(0, Flow::IndirectCall), inst(1, Flow::Return)];
    let graph =
        build_address_cfg(instructions, &[], AddressCfgOptions::default(), payload).unwrap();

    let edges: Vec<_> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    assert_eq!(edges, vec![(0, 1, EdgeKind::CallReturn, "cont")]);
    assert_eq!(
        graph.unresolved_transfers,
        vec![UnresolvedTransfer {
            source: 0,
            target: None,
            role: UnresolvedRole::IndirectCall,
        }]
    );
}
