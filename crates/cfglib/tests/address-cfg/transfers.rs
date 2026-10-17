//! Control transfers beyond plain branches: indirect and unresolved targets,
//! switches, alternate-path operations, and block-ending terminators.

use cfglib::{UnresolvedRole, UnresolvedTransfer};

use crate::harness::{
    CALL, CASE, FALL, JUMP, NOT_TAKEN, RETURNS, Role, TAKEN, UNWIND, assert_empty, build,
};
use crate::isa::Dest::{At, Beyond, Inside};
use crate::isa::{Isa, Op};

/// A function that uses every transfer kind once: a conditional, a direct
/// call, a table dispatch, an alternate-path probe, a backward jump, a
/// return, and a call through a register.
pub fn kitchen_sink() -> Vec<Op> {
    vec![
        Op::Work,
        Op::BranchIf(At(8)),
        Op::Call(At(9)),
        Op::Dispatch {
            default: At(7),
            cases: vec![(10, At(4)), (20, At(5))],
        },
        Op::Probe(At(7)),
        Op::Work,
        Op::Jump(At(0)),
        Op::Return,
        Op::CallIndirect,
        Op::Return,
    ]
}

fn an_indirect_jump_ends_its_block_and_is_reported_unresolved<I: Isa>() {
    let built = build::<I>(&[Op::Work, Op::JumpIndirect, Op::Return]);
    assert_eq!(built.blocks(), vec![vec![0, 1], vec![2]]);
    assert_eq!(built.cfg().edge_count(), 0);
    assert_eq!(
        built.graph.unresolved_transfers,
        [UnresolvedTransfer {
            source: built.layout.address(1),
            target: None,
            role: UnresolvedRole::Indirect,
        }]
    );
}

fn a_taken_target_outside_the_code_keeps_only_the_fallthrough<I: Isa>() {
    let built = build::<I>(&[Op::BranchIf(Beyond(0x40)), Op::Return]);
    assert_eq!(built.blocks(), vec![vec![0], vec![1]]);
    assert_eq!(built.successors(), vec![vec![(1, NOT_TAKEN)], vec![]]);
    assert_eq!(
        built.graph.unresolved_transfers,
        [UnresolvedTransfer {
            source: built.layout.address(0),
            target: Some(built.layout.beyond(0x40)),
            role: UnresolvedRole::ConditionalTaken,
        }]
    );
}

fn a_jump_into_the_middle_of_an_instruction_is_unresolved_and_leads_nothing<I: Isa>() {
    let built = build::<I>(&[Op::Jump(Inside(2)), Op::Return, Op::Work, Op::Return]);
    // The target is not an instruction start, so it neither joins the graph
    // nor splits the instruction it points into.
    assert_eq!(built.blocks(), vec![vec![0], vec![1], vec![2, 3]]);
    assert_eq!(built.cfg().edge_count(), 0);
    assert_eq!(
        built.graph.unresolved_transfers,
        [UnresolvedTransfer {
            source: built.layout.address(0),
            target: Some(built.layout.inside(2)),
            role: UnresolvedRole::Jump,
        }]
    );
}

fn a_switch_lists_the_default_then_every_case_in_dispatch_order<I: Isa>() {
    let built = build::<I>(&[
        Op::Work,
        Op::Dispatch {
            default: At(5),
            cases: vec![(10, At(2)), (20, At(3)), (30, At(3))],
        },
        Op::Jump(At(5)),
        Op::Jump(At(5)),
        Op::Return,
        Op::Return,
    ]);
    assert_eq!(
        built.blocks(),
        vec![vec![0, 1], vec![2], vec![3], vec![4], vec![5]]
    );
    // Two cases share one target: the parallel edges stay distinct.
    assert_eq!(
        built.successors(),
        vec![
            vec![(4, CASE), (1, CASE), (2, CASE), (2, CASE)],
            vec![(4, JUMP)],
            vec![(4, JUMP)],
            vec![],
            vec![],
        ]
    );
    assert_eq!(
        built.roles()[0],
        vec![
            Role::Default,
            Role::Case { index: 0, key: 10 },
            Role::Case { index: 1, key: 20 },
            Role::Case { index: 2, key: 30 },
        ]
    );
    assert_empty(&built.graph.unresolved_transfers);
}

fn a_partly_resolved_switch_reports_each_missing_destination_in_dispatch_order<I: Isa>() {
    let built = build::<I>(&[
        Op::Dispatch {
            default: Beyond(9),
            cases: vec![(1, At(1)), (2, Beyond(1)), (3, Inside(1))],
        },
        Op::Work,
        Op::Return,
    ]);
    assert_eq!(built.blocks(), vec![vec![0], vec![1, 2]]);
    assert_eq!(built.successors(), vec![vec![(1, CASE)], vec![]]);
    assert_eq!(built.roles()[0], vec![Role::Case { index: 0, key: 1 }]);
    let source = built.layout.address(0);
    assert_eq!(
        built.graph.unresolved_transfers,
        [
            UnresolvedTransfer {
                source,
                target: Some(built.layout.beyond(9)),
                role: UnresolvedRole::SwitchDefault,
            },
            UnresolvedTransfer {
                source,
                target: Some(built.layout.beyond(1)),
                role: UnresolvedRole::SwitchCase { index: 1 },
            },
            UnresolvedTransfer {
                source,
                target: Some(built.layout.inside(1)),
                role: UnresolvedRole::SwitchCase { index: 2 },
            },
        ]
    );
}

fn a_probe_leads_its_own_block_with_the_alternate_edge_before_the_sequential_one<I: Isa>() {
    let built = build::<I>(&[Op::Work, Op::Work, Op::Probe(At(4)), Op::Work, Op::Return]);
    // The probe splits the straight line before itself and after itself.
    assert_eq!(built.blocks(), vec![vec![0, 1], vec![2], vec![3], vec![4]]);
    assert_eq!(
        built.successors(),
        vec![
            vec![(1, FALL)],
            vec![(3, UNWIND), (2, FALL)],
            vec![(3, FALL)],
            vec![],
        ]
    );
    assert_eq!(built.roles()[1], vec![Role::Exceptional, Role::Sequential]);
}

fn a_probe_to_an_unrecovered_target_keeps_its_sequential_edge<I: Isa>() {
    let built = build::<I>(&[Op::Probe(Beyond(2)), Op::Return]);
    assert_eq!(built.successors(), vec![vec![(1, FALL)], vec![]]);
    assert_eq!(
        built.graph.unresolved_transfers,
        [UnresolvedTransfer {
            source: built.layout.address(0),
            target: Some(built.layout.beyond(2)),
            role: UnresolvedRole::Exceptional,
        }]
    );
}

fn returns_and_raises_end_blocks_without_edges<I: Isa>() {
    let built = build::<I>(&[Op::Return, Op::Raise, Op::Work, Op::Return]);
    assert_eq!(built.blocks(), vec![vec![0], vec![1], vec![2, 3]]);
    assert_eq!(built.cfg().edge_count(), 0);
    assert_empty(&built.graph.unresolved_transfers);
    assert_eq!(built.cfg().exit_blocks().count(), 3);
}

fn the_kitchen_sink_function_has_hand_derived_blocks_and_edges<I: Isa>() {
    let built = build::<I>(&kitchen_sink());
    assert_eq!(
        built.blocks(),
        vec![
            vec![0, 1],
            vec![2],
            vec![3],
            vec![4],
            vec![5, 6],
            vec![7],
            vec![8],
            vec![9],
        ]
    );
    assert_eq!(
        built.successors(),
        vec![
            vec![(6, TAKEN), (1, NOT_TAKEN)],
            vec![(7, CALL), (2, RETURNS)],
            vec![(5, CASE), (3, CASE), (4, CASE)],
            vec![(5, UNWIND), (4, FALL)],
            vec![(0, JUMP)],
            vec![],
            vec![(7, RETURNS)],
            vec![],
        ]
    );
    assert_eq!(
        built.roles(),
        vec![
            vec![Role::Taken, Role::FallThrough],
            vec![
                Role::Call,
                Role::Continuation {
                    call_site: built.raw(2)
                }
            ],
            vec![
                Role::Default,
                Role::Case { index: 0, key: 10 },
                Role::Case { index: 1, key: 20 }
            ],
            vec![Role::Exceptional, Role::Sequential],
            vec![Role::Jump],
            vec![],
            vec![Role::Continuation {
                call_site: built.raw(8)
            }],
            vec![],
        ]
    );
    assert_eq!(
        built.graph.unresolved_transfers,
        [UnresolvedTransfer {
            source: built.layout.address(8),
            target: None,
            role: UnresolvedRole::IndirectCall,
        }]
    );
    assert_empty(&built.graph.calls);
}

fn a_switch_without_cases_has_only_its_default_edge<I: Isa>() {
    let built = build::<I>(&[
        Op::Dispatch {
            default: At(1),
            cases: vec![],
        },
        Op::Return,
    ]);
    assert_eq!(built.blocks(), vec![vec![0], vec![1]]);
    assert_eq!(built.successors(), vec![vec![(1, CASE)], vec![]]);
    assert_eq!(built.roles()[0], vec![Role::Default]);
}

over_both_isas!(
    an_indirect_jump_ends_its_block_and_is_reported_unresolved,
    a_taken_target_outside_the_code_keeps_only_the_fallthrough,
    a_jump_into_the_middle_of_an_instruction_is_unresolved_and_leads_nothing,
    a_switch_lists_the_default_then_every_case_in_dispatch_order,
    a_partly_resolved_switch_reports_each_missing_destination_in_dispatch_order,
    a_probe_leads_its_own_block_with_the_alternate_edge_before_the_sequential_one,
    a_probe_to_an_unrecovered_target_keeps_its_sequential_edge,
    returns_and_raises_end_blocks_without_edges,
    the_kitchen_sink_function_has_hand_derived_blocks_and_edges,
    a_switch_without_cases_has_only_its_default_edge,
);
