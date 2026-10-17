//! Calls under both [`CallPolicy`] variants: call edges and return sites
//! when calls are transfers, and block-local calls with a call-site list
//! when they are flattened.

use cfglib::{
    AddressCfgOptions, CallPolicy, CallSite, EdgeKind, UnresolvedRole, UnresolvedTransfer,
};

use crate::harness::{
    Built, CALL, CASE, FALL, JUMP, NOT_TAKEN, RETURNS, Role, Settings, TAKEN, UNWIND, assert_empty,
    build, build_with,
};
use crate::isa::Dest::{At, Beyond};
use crate::isa::{Isa, Op};
use crate::transfers::kitchen_sink;

fn flattened() -> Settings {
    Settings {
        calls: CallPolicy::Flatten,
        ..Settings::default()
    }
}

fn a_call_ends_its_block_and_its_return_site_gets_a_continuation_edge<I: Isa>() {
    let built = build::<I>(&[Op::Work, Op::Call(At(3)), Op::Work, Op::Return]);
    assert_eq!(built.blocks(), vec![vec![0, 1], vec![2], vec![3]]);
    // The call edge precedes the continuation edge.
    assert_eq!(
        built.successors(),
        vec![vec![(2, CALL), (1, RETURNS)], vec![(2, FALL)], vec![]]
    );
    assert_eq!(
        built.roles()[0],
        vec![
            Role::Call,
            Role::Continuation {
                call_site: built.raw(1)
            }
        ]
    );
    assert!(built.graph.calls.is_empty(), "calls are edges, not sites");
}

fn edges_is_the_default_call_policy<I: Isa>() {
    assert_eq!(CallPolicy::default(), CallPolicy::Edges);
    let options = AddressCfgOptions::<u32>::default();
    assert_eq!(options.entry, None);
    assert_eq!(options.calls, CallPolicy::Edges);

    let ops = [Op::Call(At(2)), Op::Work, Op::Return];
    let implicit = build::<I>(&ops);
    let explicit = build_with::<I>(
        &ops,
        Vec::new(),
        Settings {
            calls: CallPolicy::Edges,
            ..Settings::default()
        },
    );
    assert_eq!(implicit.blocks(), explicit.blocks());
    assert_eq!(implicit.successors(), explicit.successors());
    assert_eq!(implicit.cfg().to_text(), explicit.cfg().to_text());
}

fn a_call_outside_the_code_is_unresolved_and_keeps_its_continuation<I: Isa>() {
    let built = build::<I>(&[Op::Call(Beyond(0x100)), Op::Return]);
    assert_eq!(built.successors(), vec![vec![(1, RETURNS)], vec![]]);
    assert_eq!(
        built.graph.unresolved_transfers,
        [UnresolvedTransfer {
            source: built.layout.address(0),
            target: Some(built.layout.beyond(0x100)),
            role: UnresolvedRole::Call,
        }]
    );
}

fn an_indirect_call_kept_as_an_edge_is_unresolved_and_keeps_its_continuation<I: Isa>() {
    let built = build::<I>(&[Op::CallIndirect, Op::Return]);
    assert_eq!(built.successors(), vec![vec![(1, RETURNS)], vec![]]);
    assert_eq!(
        built.graph.unresolved_transfers,
        [UnresolvedTransfer {
            source: built.layout.address(0),
            target: None,
            role: UnresolvedRole::IndirectCall,
        }]
    );
}

fn a_recursive_call_edge_returns_to_the_callers_own_block<I: Isa>() {
    let built = build::<I>(&[Op::Work, Op::Call(At(0)), Op::Return]);
    assert_eq!(built.blocks(), vec![vec![0, 1], vec![2]]);
    assert_eq!(
        built.successors(),
        vec![vec![(0, CALL), (1, RETURNS)], vec![]]
    );
}

fn an_edges_call_with_nothing_after_it_has_no_continuation_edge<I: Isa>() {
    let built = build::<I>(&[Op::Work, Op::Call(Beyond(0))]);
    assert_eq!(built.blocks(), vec![vec![0, 1]]);
    assert_eq!(built.cfg().edge_count(), 0);
    assert_eq!(
        built.graph.unresolved_transfers,
        [UnresolvedTransfer {
            source: built.layout.address(1),
            target: Some(built.layout.beyond(0)),
            role: UnresolvedRole::Call,
        }]
    );
}

fn a_flattened_call_is_an_operation_of_its_block<I: Isa>() {
    let built = build_with::<I>(
        &[Op::Work, Op::Call(At(3)), Op::Work, Op::Return],
        Vec::new(),
        flattened(),
    );
    // Neither the callee nor the return site leads a block.
    assert_eq!(built.blocks(), vec![vec![0, 1, 2, 3]]);
    assert_eq!(built.cfg().edge_count(), 0);
    assert_eq!(
        built.graph.calls,
        [CallSite {
            source: built.layout.address(1),
            target: Some(built.layout.address(3)),
        }]
    );
    assert_empty(&built.graph.unresolved_transfers);
}

fn flattened_calls_are_listed_in_address_order_with_their_targets<I: Isa>() {
    let built = build_with::<I>(&kitchen_sink(), Vec::new(), flattened());
    assert_eq!(
        built.blocks(),
        vec![
            vec![0, 1],
            vec![2, 3],
            vec![4],
            vec![5, 6],
            vec![7],
            vec![8, 9],
        ]
    );
    assert_eq!(
        built.successors(),
        vec![
            vec![(5, TAKEN), (1, NOT_TAKEN)],
            vec![(4, CASE), (2, CASE), (3, CASE)],
            vec![(4, UNWIND), (3, FALL)],
            vec![(0, JUMP)],
            vec![],
            vec![],
        ]
    );
    // A direct callee and a computed one: the call list keeps both, in
    // address order, and neither is also an unresolved transfer.
    assert_eq!(
        built.graph.calls,
        [
            CallSite {
                source: built.layout.address(2),
                target: Some(built.layout.address(9)),
            },
            CallSite {
                source: built.layout.address(8),
                target: None,
            },
        ]
    );
    assert_empty(&built.graph.unresolved_transfers);
}

fn a_flattened_call_that_is_also_a_branch_target_still_leads_a_block<I: Isa>() {
    let built = build_with::<I>(
        &[
            Op::BranchIf(At(2)),
            Op::Work,
            Op::Call(Beyond(5)),
            Op::Work,
            Op::Return,
        ],
        Vec::new(),
        flattened(),
    );
    assert_eq!(built.blocks(), vec![vec![0], vec![1], vec![2, 3, 4]]);
    assert_eq!(
        built.successors(),
        vec![vec![(2, TAKEN), (1, NOT_TAKEN)], vec![(2, FALL)], vec![]]
    );
    assert_eq!(
        built.graph.calls,
        [CallSite {
            source: built.layout.address(2),
            target: Some(built.layout.beyond(5)),
        }]
    );
    assert_empty(&built.graph.unresolved_transfers);
}

fn a_flattened_call_with_nothing_after_it_adds_no_edge<I: Isa>() {
    let built = build_with::<I>(&[Op::Work, Op::Call(Beyond(0))], Vec::new(), flattened());
    assert_eq!(built.blocks(), vec![vec![0, 1]]);
    assert_eq!(built.cfg().edge_count(), 0);
    assert_eq!(built.graph.calls.len(), 1);
    assert_empty(&built.graph.unresolved_transfers);
}

fn a_flattened_call_does_not_hide_a_raise_or_a_return<I: Isa>() {
    let built = build_with::<I>(
        &[
            Op::Call(At(1)),
            Op::CallIndirect,
            Op::Raise,
            Op::Work,
            Op::Return,
        ],
        Vec::new(),
        flattened(),
    );
    // Both calls stay in the first block; the raise still ends it.
    assert_eq!(built.blocks(), vec![vec![0, 1, 2], vec![3, 4]]);
    assert_eq!(built.cfg().edge_count(), 0);
    assert_eq!(built.graph.calls.len(), 2);
}

fn programs_without_calls_ignore_the_call_policy<I: Isa>() {
    let programs = [
        vec![Op::Work, Op::BranchIf(At(3)), Op::Work, Op::Return],
        vec![
            Op::Dispatch {
                default: At(2),
                cases: vec![(1, At(1))],
            },
            Op::Work,
            Op::Jump(At(0)),
        ],
        vec![Op::Probe(At(2)), Op::Work, Op::Return],
    ];
    for ops in programs {
        let edges = build::<I>(&ops);
        let flatten = build_with::<I>(&ops, Vec::new(), flattened());
        assert_eq!(edges.blocks(), flatten.blocks());
        assert_eq!(edges.successors(), flatten.successors());
        assert_eq!(edges.cfg().to_text(), flatten.cfg().to_text());
        assert_empty(&flatten.graph.calls);
    }
}

/// Every edge as `(source instruction, target block start, kind)`, without
/// the sequential, call, and return edges the call policy legitimately
/// changes.
fn branch_edges<I: Isa>(built: &Built<I>) -> Vec<(usize, usize, EdgeKind)> {
    let cfg = built.cfg();
    let mut edges: Vec<_> = cfg
        .edges()
        .filter(|edge| !matches!(edge.kind(), FALL | CALL | RETURNS))
        .map(|edge| {
            let source = cfg.block(edge.source()).instructions().last().unwrap();
            let target = cfg.block(edge.target()).instructions().first().unwrap();
            (
                built.position_of(cfglib::AddressInstruction::address(source)),
                built.position_of(cfglib::AddressInstruction::address(target)),
                edge.kind(),
            )
        })
        .collect();
    edges.sort_by_key(|&(source, target, kind)| (source, target, kind.name()));
    edges
}

fn flattening_removes_call_and_return_edges_and_changes_no_branch<I: Isa>() {
    let programs = [
        kitchen_sink(),
        vec![
            Op::Work,
            Op::Work,
            Op::Call(At(1)),
            Op::BranchIf(At(5)),
            Op::Call(At(0)),
            Op::Jump(At(1)),
        ],
    ];
    for ops in programs {
        let with_edges = build::<I>(&ops);
        let flatten = build_with::<I>(&ops, Vec::new(), flattened());
        assert_eq!(branch_edges(&with_edges), branch_edges(&flatten));
        let call_edges = with_edges
            .cfg()
            .edges()
            .filter(|edge| matches!(edge.kind(), CALL | RETURNS))
            .count();
        assert!(call_edges > 0);
        assert!(
            flatten
                .cfg()
                .edges()
                .all(|e| !matches!(e.kind(), CALL | RETURNS))
        );
    }
}

over_both_isas!(
    a_call_ends_its_block_and_its_return_site_gets_a_continuation_edge,
    edges_is_the_default_call_policy,
    a_call_outside_the_code_is_unresolved_and_keeps_its_continuation,
    an_indirect_call_kept_as_an_edge_is_unresolved_and_keeps_its_continuation,
    a_recursive_call_edge_returns_to_the_callers_own_block,
    an_edges_call_with_nothing_after_it_has_no_continuation_edge,
    a_flattened_call_is_an_operation_of_its_block,
    flattened_calls_are_listed_in_address_order_with_their_targets,
    a_flattened_call_that_is_also_a_branch_target_still_leads_a_block,
    a_flattened_call_with_nothing_after_it_adds_no_edge,
    a_flattened_call_does_not_hide_a_raise_or_a_return,
    programs_without_calls_ignore_the_call_policy,
    flattening_removes_call_and_return_edges_and_changes_no_branch,
);
