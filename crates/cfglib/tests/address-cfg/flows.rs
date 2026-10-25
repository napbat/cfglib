//! The flow vocabulary on its own: how every [`Flow`] variant ends blocks,
//! continues, names targets, and expands into transfers with edge roles.
//!
//! Expectations are the table the `Flow` and `EdgeRole` documentation
//! states, written out once per variant.

use cfglib::{EdgeKind, EdgeRole, Flow, Transfer, UnresolvedRole};

type TestFlow = Flow<u32, char>;

/// `(target, structural kind, unresolved role)` of every transfer `flow`
/// makes from address 10 when the next position is `next`.
fn transfers(
    flow: &TestFlow,
    next: Option<u32>,
) -> Vec<(Option<u32>, EdgeKind, Option<UnresolvedRole>)> {
    flow.transfers(10, next)
        .map(|transfer| {
            (
                transfer.target,
                transfer.role.kind(),
                transfer.unresolved_role(),
            )
        })
        .collect()
}

struct Row {
    flow: TestFlow,
    ends_block: bool,
    continues: bool,
    is_call: bool,
    direct_target: Option<u32>,
    transfers: Vec<(Option<u32>, EdgeKind, Option<UnresolvedRole>)>,
}

/// The rows of the flows that name no direct target, in table order.
fn targetless_rows() -> Vec<Row> {
    use EdgeKind::{
        Call, CallReturn, ConditionalFalse, ConditionalTrue, ExceptionUnwind, Fallthrough, Jump,
    };
    vec![
        Row {
            flow: Flow::Next,
            ends_block: false,
            continues: true,
            is_call: false,
            direct_target: None,
            transfers: vec![(Some(11), Fallthrough, None)],
        },
        Row {
            flow: Flow::Return,
            ends_block: true,
            continues: false,
            is_call: false,
            direct_target: None,
            transfers: vec![],
        },
        Row {
            flow: Flow::Throw,
            ends_block: true,
            continues: false,
            is_call: false,
            direct_target: None,
            transfers: vec![],
        },
        Row {
            flow: Flow::Indirect,
            ends_block: true,
            continues: false,
            is_call: false,
            direct_target: None,
            transfers: vec![(None, Jump, Some(UnresolvedRole::Indirect))],
        },
        Row {
            flow: Flow::ConditionalIndirect,
            ends_block: true,
            continues: true,
            is_call: false,
            direct_target: None,
            transfers: vec![
                (
                    None,
                    ConditionalTrue,
                    Some(UnresolvedRole::ConditionalIndirect),
                ),
                (Some(11), ConditionalFalse, None),
            ],
        },
        Row {
            flow: Flow::IndirectCall,
            ends_block: true,
            continues: true,
            is_call: true,
            direct_target: None,
            transfers: vec![
                (None, Call, Some(UnresolvedRole::IndirectCall)),
                (Some(11), CallReturn, None),
            ],
        },
        Row {
            flow: Flow::Trap,
            ends_block: true,
            continues: false,
            is_call: false,
            direct_target: None,
            transfers: vec![(None, ExceptionUnwind, Some(UnresolvedRole::Trap))],
        },
        Row {
            flow: Flow::ConditionalTrap,
            ends_block: true,
            continues: true,
            is_call: false,
            direct_target: None,
            transfers: vec![
                (None, ExceptionUnwind, Some(UnresolvedRole::Trap)),
                (Some(11), Fallthrough, None),
            ],
        },
    ]
}

/// The rows of the flows that name one direct target, in table order.
fn single_target_rows() -> Vec<Row> {
    use EdgeKind::{
        Call, CallReturn, ConditionalFalse, ConditionalTrue, ExceptionUnwind, Fallthrough, Jump,
    };
    vec![
        Row {
            flow: Flow::Jump { target: 40 },
            ends_block: true,
            continues: false,
            is_call: false,
            direct_target: Some(40),
            transfers: vec![(Some(40), Jump, Some(UnresolvedRole::Jump))],
        },
        Row {
            flow: Flow::Conditional { target: 40 },
            ends_block: true,
            continues: true,
            is_call: false,
            direct_target: Some(40),
            transfers: vec![
                (
                    Some(40),
                    ConditionalTrue,
                    Some(UnresolvedRole::ConditionalTaken),
                ),
                (Some(11), ConditionalFalse, None),
            ],
        },
        Row {
            flow: Flow::Call { target: 40 },
            ends_block: true,
            continues: true,
            is_call: true,
            direct_target: Some(40),
            transfers: vec![
                (Some(40), Call, Some(UnresolvedRole::Call)),
                (Some(11), CallReturn, None),
            ],
        },
        Row {
            flow: Flow::Exceptional { target: 40 },
            ends_block: true,
            continues: true,
            is_call: false,
            direct_target: Some(40),
            transfers: vec![
                (Some(40), ExceptionUnwind, Some(UnresolvedRole::Exceptional)),
                (Some(11), Fallthrough, None),
            ],
        },
    ]
}

/// The row of the multi-way switch: a default and two keyed cases.
fn switch_row() -> Row {
    use EdgeKind::SwitchCase;
    Row {
        flow: Flow::Switch {
            default: 50,
            cases: vec![('a', 60), ('b', 70)],
        },
        ends_block: true,
        continues: false,
        is_call: false,
        direct_target: None,
        transfers: vec![
            (Some(50), SwitchCase, Some(UnresolvedRole::SwitchDefault)),
            (
                Some(60),
                SwitchCase,
                Some(UnresolvedRole::SwitchCase { index: 0 }),
            ),
            (
                Some(70),
                SwitchCase,
                Some(UnresolvedRole::SwitchCase { index: 1 }),
            ),
        ],
    }
}

fn rows() -> Vec<Row> {
    let mut rows = targetless_rows();
    rows.extend(single_target_rows());
    rows.push(switch_row());
    rows
}

#[test]
fn every_flow_variant_has_the_documented_block_and_transfer_shape() {
    for row in rows() {
        let flow = &row.flow;
        assert_eq!(flow.ends_block(), row.ends_block, "{flow:?} ends_block");
        assert_eq!(flow.continues(), row.continues, "{flow:?} continues");
        assert_eq!(flow.is_call(), row.is_call, "{flow:?} is_call");
        assert_eq!(
            flow.direct_target(),
            row.direct_target,
            "{flow:?} direct_target"
        );
        assert_eq!(
            transfers(flow, Some(11)),
            row.transfers,
            "{flow:?} transfers"
        );
    }
}

#[test]
fn a_flow_without_a_next_position_skips_only_its_continuation() {
    for row in rows() {
        // A flow that continues lists its continuation last, so losing the
        // next position drops exactly that one transfer.
        let expected = if row.continues {
            row.transfers[..row.transfers.len() - 1].to_vec()
        } else {
            row.transfers.clone()
        };
        assert_eq!(transfers(&row.flow, None), expected, "{:?}", row.flow);
    }
}

#[test]
fn a_call_continuation_names_the_calling_operation() {
    for flow in [TestFlow::Call { target: 40 }, TestFlow::IndirectCall] {
        let continuation: Transfer<'_, u32, char> = flow
            .transfers(10, Some(11))
            .nth(1)
            .expect("a call continues after it returns");
        assert_eq!(
            continuation.role,
            EdgeRole::CallContinuation { call_site: 10 }
        );
        assert_eq!(continuation.target, Some(11));
    }
}

#[test]
fn switch_cases_carry_their_dispatch_index_and_key() {
    let flow: TestFlow = Flow::Switch {
        default: 50,
        cases: vec![('x', 60), ('y', 70)],
    };
    let roles: Vec<_> = flow.transfers(10, Some(11)).map(|t| t.role).collect();
    assert_eq!(
        roles,
        [
            EdgeRole::SwitchDefault,
            EdgeRole::SwitchCase {
                index: 0,
                key: &'x'
            },
            EdgeRole::SwitchCase {
                index: 1,
                key: &'y'
            },
        ]
    );
}

#[test]
fn every_edge_role_maps_to_one_structural_kind_and_only_a_call_role_is_a_call() {
    let key = 'k';
    let table: [(EdgeRole<'_, u32, char>, EdgeKind, bool); 10] = [
        (EdgeRole::Sequential, EdgeKind::Fallthrough, false),
        (EdgeRole::ConditionalTaken, EdgeKind::ConditionalTrue, false),
        (
            EdgeRole::ConditionalFallThrough,
            EdgeKind::ConditionalFalse,
            false,
        ),
        (EdgeRole::Jump, EdgeKind::Jump, false),
        (EdgeRole::SwitchDefault, EdgeKind::SwitchCase, false),
        (
            EdgeRole::SwitchCase {
                index: 3,
                key: &key,
            },
            EdgeKind::SwitchCase,
            false,
        ),
        (EdgeRole::Call, EdgeKind::Call, true),
        (
            EdgeRole::CallContinuation { call_site: 7 },
            EdgeKind::CallReturn,
            false,
        ),
        (EdgeRole::Exceptional, EdgeKind::ExceptionUnwind, false),
        (
            EdgeRole::Unwind { handler: 2 },
            EdgeKind::ExceptionUnwind,
            false,
        ),
    ];
    for (role, kind, is_call) in table {
        assert_eq!(role.kind(), kind, "{role:?}");
        assert_eq!(role.is_call(), is_call, "{role:?}");
    }
}

#[test]
fn a_continuation_and_an_unwind_have_no_unresolved_form() {
    let target_less = |role| Transfer::<u32, char> { target: None, role };
    let named = |role| Transfer::<u32, char> {
        target: Some(5),
        role,
    };
    for role in [
        EdgeRole::Sequential,
        EdgeRole::ConditionalFallThrough,
        EdgeRole::CallContinuation { call_site: 1 },
        EdgeRole::Unwind { handler: 0 },
    ] {
        assert_eq!(named(role).unresolved_role(), None, "{role:?}");
        assert_eq!(target_less(role).unresolved_role(), None, "{role:?}");
    }
    // A computed jump and a computed call are the only transfers whose
    // missing target is unresolved because it was never named.
    assert_eq!(
        target_less(EdgeRole::Jump).unresolved_role(),
        Some(UnresolvedRole::Indirect)
    );
    assert_eq!(
        target_less(EdgeRole::Call).unresolved_role(),
        Some(UnresolvedRole::IndirectCall)
    );
}
