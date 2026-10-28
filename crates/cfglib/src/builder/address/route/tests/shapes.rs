extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::{Built, body, build, call, funclets, main_code, route, stream};
use crate::builder::address::{HopEnd, UnwindEnd, UnwindHop, UnwindStep};
use crate::edge::EdgeKind;
use crate::flow::Flow;

const RET: Flow<u32, i32> = Flow::Return;
const NEXT: Flow<u32, i32> = Flow::Next;

/// Two protected instructions that enter the funclet at 10 with different
/// remaining paths, so the funclet has two instances.
fn two_contexts() -> [crate::builder::address::UnwindRoute<u32>; 2] {
    [
        route(0..1, vec![(funclets(&[10]), UnwindEnd::Handler(30))]),
        route(1..2, vec![(funclets(&[10, 20]), UnwindEnd::Propagate)]),
    ]
}

fn assert_instances_isolated(built: &Built, address: u32) {
    let instances = built.instances_of(address);
    assert_eq!(instances.len(), 2, "two contexts, two instances");
    let (first, second) = (built.reach(instances[0]), built.reach(instances[1]));
    assert!(first.is_disjoint(&second));
    assert_eq!(
        first.len(),
        second.len(),
        "the copy has the shape of the original"
    );
}

#[test]
fn handlers_at_several_positions_are_exact() {
    let instructions = stream(&[
        main_code(2),
        body(10, &[RET]),
        body(20, &[RET]),
        body(30, &[RET]),
        body(40, &[RET]),
        body(50, &[RET]),
        body(60, &[RET]),
    ]);
    let routes = [route(
        0..1,
        vec![
            (vec![], UnwindEnd::Handler(60)),
            (funclets(&[10]), UnwindEnd::Handler(40)),
            (funclets(&[10, 20]), UnwindEnd::Handler(50)),
            (funclets(&[10, 20, 30]), UnwindEnd::Propagate),
        ],
    )];
    let built = build(instructions, &routes);

    built.assert_exact(&routes, &[40, 50, 60]);
    assert_eq!(
        built.route_edges(),
        BTreeSet::from([(0, 10), (0, 60), (10, 20), (10, 40), (20, 30), (20, 50)]),
    );
}

#[test]
fn trailing_calls_of_a_path_belong_to_its_end() {
    let instructions = stream(&[main_code(2), body(10, &[RET])]);
    let routes = [route(
        0..1,
        vec![
            (vec![UnwindStep::Funclet(10), call(4)], UnwindEnd::Propagate),
            (vec![UnwindStep::Funclet(10), call(3)], UnwindEnd::Propagate),
        ],
    )];
    let built = build(instructions, &routes);

    assert_eq!(
        built.hops_of(built.block_of(10)),
        [&UnwindHop {
            source: built.block_of(10),
            end: HopEnd::Leave,
            calls: vec![vec![3], vec![4]],
        }]
    );
    assert_eq!(
        built.hops_of(built.block_of(0)),
        [&UnwindHop {
            source: built.block_of(0),
            end: HopEnd::Block(built.block_of(10)),
            calls: vec![vec![]],
        }],
        "the trailing calls run after the funclet, not before it"
    );
}

#[test]
fn a_funclet_with_two_returns_enters_the_next_step_from_both() {
    let instructions = stream(&[
        main_code(2),
        body(10, &[Flow::Conditional { target: 13 }, RET]),
        body(13, &[RET]),
        body(20, &[RET]),
        body(30, &[RET]),
    ]);
    let routes = two_contexts();
    let built = build(instructions, &routes);

    built.assert_exact(&routes, &[30]);
    let handler_edges: BTreeSet<_> = built
        .route_edges()
        .into_iter()
        .filter(|&(_, target)| target == 30)
        .collect();
    assert_eq!(handler_edges, BTreeSet::from([(11, 30), (13, 30)]));
    assert_eq!(built.instances_of(13).len(), 2);
    assert_eq!(
        built.route_edges().iter().filter(|(_, t)| *t == 20).count(),
        2,
        "each return of the other instance enters the next funclet"
    );
}

#[test]
fn a_funclet_that_ends_in_a_tail_jump_returns_through_the_jump() {
    let instructions = stream(&[
        main_code(2),
        body(10, &[NEXT, Flow::Jump { target: 900 }]),
        body(20, &[RET]),
        body(30, &[RET]),
    ]);
    let routes = [route(
        0..1,
        vec![
            (funclets(&[10]), UnwindEnd::Handler(30)),
            (funclets(&[10, 20]), UnwindEnd::Propagate),
        ],
    )];
    let built = build(instructions, &routes);

    built.assert_exact(&routes, &[30]);
    assert_eq!(
        built.route_edges(),
        BTreeSet::from([(0, 10), (11, 20), (11, 30)]),
        "the tail jump leaves the block, and the block has the unwind edges"
    );
}

#[test]
fn a_jump_inside_the_stream_is_no_return() {
    let instructions = stream(&[
        main_code(2),
        body(10, &[Flow::Jump { target: 12 }]),
        body(12, &[RET]),
        body(30, &[RET]),
    ]);
    let routes = [route(0..1, vec![(funclets(&[10]), UnwindEnd::Handler(30))])];
    let built = build(instructions, &routes);

    built.assert_exact(&routes, &[30]);
    assert_eq!(
        built.route_edges(),
        BTreeSet::from([(0, 10), (12, 30)]),
        "only the return at 12 enters the handler"
    );
}

#[test]
fn a_funclet_that_never_returns_has_no_way_on() {
    let instructions = stream(&[
        main_code(2),
        body(10, &[Flow::Jump { target: 10 }]),
        body(20, &[RET]),
    ]);
    let routes = [route(
        0..1,
        vec![
            (funclets(&[10]), UnwindEnd::Propagate),
            (funclets(&[20]), UnwindEnd::Propagate),
        ],
    )];
    let built = build(instructions, &routes);

    assert_eq!(
        built.paths_from(built.block_of(0), &[]),
        BTreeSet::from([(vec![20], UnwindEnd::Propagate)]),
        "the way through the funclet that never returns ends in the funclet"
    );
    assert_eq!(built.hops_of(built.block_of(10)), Vec::<&UnwindHop>::new());
    assert_eq!(built.route_edges(), BTreeSet::from([(0, 10), (0, 20)]));
}

#[test]
fn a_funclet_with_an_inner_branch_is_copied_with_all_its_blocks() {
    let instructions = stream(&[
        main_code(2),
        body(
            10,
            &[
                Flow::Conditional { target: 13 },
                NEXT,
                Flow::Jump { target: 14 },
                NEXT,
                RET,
            ],
        ),
        body(20, &[RET]),
        body(30, &[RET]),
    ]);
    let routes = two_contexts();
    let built = build(instructions, &routes);

    built.assert_exact(&routes, &[30]);
    assert_instances_isolated(&built, 10);
    assert_eq!(built.instances_of(14).len(), 2);
    assert_eq!(built.instances_of(13).len(), 2);
    assert_eq!(built.instances_of(11).len(), 2);
}

#[test]
fn a_funclet_that_loops_is_copied_with_a_loop_in_the_copy() {
    let instructions = stream(&[
        main_code(2),
        body(10, &[NEXT, Flow::Conditional { target: 10 }, RET]),
        body(20, &[RET]),
        body(30, &[RET]),
    ]);
    let routes = two_contexts();
    let built = build(instructions, &routes);

    built.assert_exact(&routes, &[30]);
    assert_instances_isolated(&built, 10);
    for entry in built.instances_of(10) {
        assert!(
            built.graph.cfg.successors(entry).any(|next| next == entry),
            "the loop of each instance goes back to the entry of that instance"
        );
    }
}

#[test]
fn a_call_in_a_funclet_keeps_its_callee_and_is_not_copied() {
    let instructions = stream(&[
        main_code(2),
        body(10, &[Flow::Call { target: 70 }, RET]),
        body(20, &[RET]),
        body(30, &[RET]),
        body(70, &[RET]),
    ]);
    let routes = two_contexts();
    let built = build(instructions, &routes);

    built.assert_exact(&routes, &[30]);
    assert_eq!(
        built.instances_of(70).len(),
        1,
        "the callee is no funclet block"
    );
    let callee = built.block_of(70);
    for entry in built.instances_of(10) {
        let calls = built
            .graph
            .cfg
            .outgoing(entry)
            .filter(|&edge| built.graph.cfg.edge(edge).data().kind() == EdgeKind::Call)
            .map(|edge| built.graph.cfg.edge(edge).target())
            .collect::<Vec<_>>();
        assert_eq!(calls, [callee], "each instance calls the callee");
    }
    assert_eq!(built.instances_of(11).len(), 2);
}

#[test]
fn a_funclet_does_not_copy_the_entry_of_another_funclet() {
    // The funclet at 10 falls through into the entry of the funclet at 11,
    // which the routes name as well.
    let instructions = stream(&[
        main_code(2),
        body(10, &[NEXT]),
        body(11, &[RET]),
        body(30, &[RET]),
    ]);
    let routes = [
        route(0..1, vec![(funclets(&[10]), UnwindEnd::Handler(30))]),
        route(1..2, vec![(funclets(&[10, 11]), UnwindEnd::Propagate)]),
    ];
    let built = build(instructions, &routes);

    assert_eq!(built.instances_of(10).len(), 2);
    assert_eq!(
        built.instances_of(11).len(),
        1,
        "the funclet at 11 is no part of the region of the funclet at 10"
    );
    for entry in built.instances_of(10) {
        assert_eq!(
            built.graph.cfg.successors(entry).collect::<Vec<_>>(),
            [built.block_of(11)],
            "each instance of 10 falls into the one block at 11"
        );
    }
}
