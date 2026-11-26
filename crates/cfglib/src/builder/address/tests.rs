extern crate alloc;
extern crate std;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use std::cell::Cell;
use std::rc::Rc;

use super::{
    AddressBuildError, AddressCfgOptions, AddressHandler, AddressHandlerKind, AddressInstruction,
    CallPolicy, GivenEdge, UnwindEnd, UnwindPath, UnwindRoute, build_address_cfg,
};
use crate::EdgeKind;
use crate::flow::{CallSite, EdgeRole, Flow, UnresolvedRole, UnresolvedTransfer};

mod function_exits;
mod handlers;

#[derive(Debug, Clone)]
pub(super) struct Inst {
    address: u32,
    size: u32,
    flow: Flow<u32, i32>,
    throws: bool,
}

pub(super) fn inst(address: u32, flow: Flow<u32, i32>) -> Inst {
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

/// An instruction that cannot throw: it keeps no exception edge.
pub(super) fn quiet_inst(address: u32, flow: Flow<u32, i32>) -> Inst {
    Inst {
        address,
        size: 1,
        flow,
        throws: false,
    }
}

/// Records `(source, target, kind, role tag)` per edge.
pub(super) fn payload(
    info: super::AddressEdgeInfo<'_, u32, i32>,
) -> (u32, u32, EdgeKind, &'static str) {
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
        EdgeRole::Route { .. } => "route",
        EdgeRole::Continue { .. } => "continue",
    };
    (info.source, info.target, info.role.kind(), role)
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
fn instruction_blocks_maps_every_address_to_the_block_holding_it() {
    let instructions = vec![
        inst(0, Flow::Conditional { target: 3 }),
        inst(1, Flow::Next),
        inst(2, Flow::Next),
        inst(3, Flow::Return),
    ];
    let graph =
        build_address_cfg(instructions, &[], AddressCfgOptions::default(), payload).unwrap();

    let mut expected = Vec::new();
    for block in graph.cfg.block_ids() {
        for instruction in graph.cfg.block(block).instructions() {
            expected.push((instruction.address(), block));
        }
    }
    expected.sort_unstable();
    let mapped: Vec<_> = graph
        .instruction_blocks
        .iter()
        .map(|(&address, &block)| (address, block))
        .collect();
    assert_eq!(mapped, expected);
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
    let counted = |flows: Vec<Flow<u32, i32>>| -> Vec<CountedInst> {
        flows
            .into_iter()
            .zip(0..)
            .map(|(flow, address)| CountedInst {
                inner: inst(address, flow),
                calls: Rc::clone(&calls),
            })
            .collect()
    };

    let instructions = counted(vec![
        Flow::Conditional { target: 2 },
        Flow::Next,
        Flow::Return,
    ]);
    build_address_cfg(instructions, &[], AddressCfgOptions::default(), |_| ()).unwrap();
    assert_eq!(calls.get(), 3, "each instruction is classified once");

    // With given edges: a branch and a plain operation with given edges, a
    // call without them, and a route, whose check reads the ends of funclets
    // from the classified flows.
    calls.set(0);
    let instructions = counted(vec![
        Flow::Conditional { target: 3 },
        Flow::Call { target: 9 },
        Flow::Next,
        Flow::Return,
        Flow::Return,
    ]);
    let given = BTreeMap::from([
        (
            0,
            vec![GivenEdge {
                target: 3,
                kind: EdgeKind::Jump,
                payload: (),
            }],
        ),
        (
            2,
            vec![GivenEdge {
                target: 3,
                kind: EdgeKind::Fallthrough,
                payload: (),
            }],
        ),
        (3, Vec::new()),
        (4, Vec::new()),
    ]);
    let routes = [UnwindRoute {
        protected: 1..2,
        paths: vec![UnwindPath {
            steps: Vec::new(),
            end: UnwindEnd::Handler(4),
        }],
    }];
    build_address_cfg(
        instructions,
        &[],
        AddressCfgOptions {
            routes: Some(&routes),
            given_edges: Some(given),
            ..AddressCfgOptions::default()
        },
        |_| (),
    )
    .unwrap();
    assert_eq!(
        calls.get(),
        5,
        "with given edges, each instruction is classified once"
    );
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
                kind: AddressHandlerKind::CatchAll,
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
                kind: AddressHandlerKind::CatchAll,
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
                role: UnresolvedRole::Jump,
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
fn computed_conditional_and_trap_exits_keep_only_their_stated_continuations() {
    // 0: conditional computed jump; 1: conditional trap; 2: trap; 3: return
    let graph = build_address_cfg(
        vec![
            inst(0, Flow::ConditionalIndirect),
            inst(1, Flow::ConditionalTrap),
            inst(2, Flow::Trap),
            inst(3, Flow::Return),
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
                role: UnresolvedRole::ConditionalTaken,
            },
            UnresolvedTransfer {
                source: 1,
                target: None,
                role: UnresolvedRole::Exceptional,
            },
            UnresolvedTransfer {
                source: 2,
                target: None,
                role: UnresolvedRole::Exceptional,
            },
        ]
    );
    let edges: Vec<_> = graph.cfg.edges().map(|edge| *edge.payload()).collect();
    assert_eq!(
        edges,
        vec![
            (0, 1, EdgeKind::ConditionalFalse, "fall"),
            (1, 2, EdgeKind::Fallthrough, "seq"),
        ]
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
        calls: CallPolicy::Flatten,
        ..AddressCfgOptions::default()
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
    assert_eq!(
        graph.unresolved_transfers,
        [] as [crate::flow::UnresolvedTransfer<u32>; 0]
    );
}

#[test]
fn a_flattened_terminator_call_continues_sequentially() {
    let instructions = vec![
        inst(0, Flow::Conditional { target: 2 }),
        inst(1, Flow::Call { target: 9 }),
        inst(2, Flow::Return),
    ];
    let options = AddressCfgOptions {
        calls: CallPolicy::Flatten,
        ..AddressCfgOptions::default()
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
            role: UnresolvedRole::Call,
        }]
    );
}

#[test]
fn default_options_need_no_default_address_type() {
    use super::AddressSpace;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct Rva(u32);

    impl AddressSpace for Rva {
        fn distance_from(self, earlier: Self) -> u64 {
            u64::from(self.0 - earlier.0)
        }
    }

    struct Operation {
        at: Rva,
        flow: Flow<Rva, ()>,
    }

    impl AddressInstruction for Operation {
        type Address = Rva;
        type CaseKey = ();

        fn address(&self) -> Rva {
            self.at
        }

        fn end_address(&self) -> Option<Rva> {
            self.at.0.checked_add(1).map(Rva)
        }

        fn flow(&self) -> Flow<Rva, ()> {
            self.flow.clone()
        }
    }

    let options = AddressCfgOptions::<Rva, EdgeKind>::default();
    assert_eq!(options.entry, None);
    assert_eq!(options.calls, CallPolicy::Edges);

    let graph = build_address_cfg(
        vec![
            Operation {
                at: Rva(0),
                flow: Flow::Next,
            },
            Operation {
                at: Rva(1),
                flow: Flow::Return,
            },
        ],
        &[],
        options,
        |info| info.role.kind(),
    )
    .unwrap();
    assert_eq!((graph.cfg.block_count(), graph.cfg.edge_count()), (1, 0));
}
