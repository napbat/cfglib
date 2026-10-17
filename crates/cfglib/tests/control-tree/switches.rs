//! `switch`: dispatch edges grouped into case arms, the default arm and its
//! edge, fallthrough, and arms that leave a loop.

use std::collections::BTreeSet;

use cfglib::{BlockId, ControlNode, EdgeKind, GotoReason};

use crate::builders::{Flow, pseudocode};
use crate::implied_flow::{implied_flow, normal_edges};
use crate::oracles::structure_and_check;
use crate::reading::{leaf_blocks, top_level};
use crate::tree_shape::{
    assert_fully_structured, goto_targets_in_tree, label_blocks_in_tree, nodes_where,
};

/// `switch (k) { case 0: a(); case 1: b(); case 2: c(); default: d() }` with
/// every arm breaking to the join.
struct ThreeWay {
    flow: Flow,
    dispatch: BlockId,
    arms: [BlockId; 3],
    default_arm: BlockId,
    join: BlockId,
}

fn dispatch_with_default() -> ThreeWay {
    let mut flow = Flow::starting_with("k = x % 3");
    let dispatch = flow.entry();
    flow.push(dispatch, "switch k");
    let arms = [flow.stmt("a()"), flow.stmt("b()"), flow.stmt("c()")];
    let default_arm = flow.stmt("d()");
    let join = flow.stmt("return");
    for arm in arms {
        flow.case(dispatch, arm);
        flow.fall(arm, join);
    }
    flow.fall(dispatch, default_arm);
    flow.fall(default_arm, join);
    ThreeWay {
        flow,
        dispatch,
        arms,
        default_arm,
        join,
    }
}

#[test]
fn switch_with_a_default_is_one_dispatch_with_an_arm_per_target_and_its_join() {
    let ThreeWay {
        flow,
        dispatch,
        arms,
        default_arm,
        join,
    } = dispatch_with_default();

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let [
        ControlNode::Switch {
            condition,
            cases,
            default_body,
            default_edge,
        },
        ControlNode::Return(tail),
    ] = top_level(&tree)[..]
    else {
        panic!("expected a switch and its join: {tree:?}");
    };
    assert_eq!(condition.block(), dispatch);
    assert_eq!(condition.range(), 0..2);
    assert_eq!(tail.block(), join);

    let cases = tree.cases(cases);
    assert_eq!(
        cases.iter().map(|case| case.block).collect::<Vec<_>>(),
        arms
    );
    for case in cases {
        assert_eq!(leaf_blocks(&tree, case.body), [case.block]);
        let [edge] = tree.edges(case.edges) else {
            panic!("one dispatch edge selects each arm");
        };
        let edge = flow.cfg.edge(*edge);
        assert_eq!(
            (edge.source(), edge.target(), edge.kind()),
            (dispatch, case.block, EdgeKind::SwitchCase)
        );
    }
    assert_eq!(leaf_blocks(&tree, default_body), [default_arm]);
    let default_edge = flow
        .cfg
        .edge(default_edge.expect("the default has an edge"));
    assert_eq!(
        (default_edge.source(), default_edge.target()),
        (dispatch, default_arm)
    );
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "k = x % 3\nswitch (switch k) {\n    case {\n        a()\n    }\n    case {\n        b()\n    }\n    case {\n        c()\n    }\n    default {\n        d()\n    }\n}\nreturn\n"
    );
}

#[test]
fn case_labels_sharing_a_target_are_one_arm_selected_by_several_edges() {
    // case 1: case 2: a();   case 3: b();   default: c()
    let mut flow = Flow::starting_with("switch k");
    let dispatch = flow.entry();
    let shared = flow.stmt("a()");
    let single = flow.stmt("b()");
    let default_arm = flow.stmt("c()");
    let join = flow.stmt("return");
    let first = flow.case(dispatch, shared);
    let second = flow.case(dispatch, shared);
    let third = flow.case(dispatch, single);
    flow.fall(dispatch, default_arm);
    for arm in [shared, single, default_arm] {
        flow.fall(arm, join);
    }

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let [ControlNode::Switch { cases, .. }, _] = top_level(&tree)[..] else {
        panic!("expected a switch: {tree:?}");
    };
    let cases = tree.cases(cases);
    assert_eq!(cases.len(), 2, "two arms, not three");
    assert_eq!(cases[0].block, shared);
    assert_eq!(
        tree.edges(cases[0].edges),
        [first, second],
        "in dispatch order"
    );
    assert_eq!(cases[1].block, single);
    assert_eq!(tree.edges(cases[1].edges), [third]);
}

#[test]
fn default_that_goes_straight_to_the_join_has_an_edge_and_no_body() {
    let mut flow = Flow::starting_with("switch k");
    let dispatch = flow.entry();
    let first = flow.stmt("a()");
    let second = flow.stmt("b()");
    let join = flow.stmt("return");
    flow.case(dispatch, first);
    flow.case(dispatch, second);
    let default_edge = flow.edge(dispatch, join, EdgeKind::Fallthrough);
    flow.fall(first, join);
    flow.fall(second, join);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let [
        ControlNode::Switch {
            default_body,
            default_edge: recorded,
            ..
        },
        _,
    ] = top_level(&tree)[..]
    else {
        panic!("expected a switch: {tree:?}");
    };
    assert!(default_body.is_empty());
    assert_eq!(recorded, Some(default_edge));
}

#[test]
fn switch_without_a_default_edge_has_no_default() {
    let mut flow = Flow::starting_with("switch k");
    let dispatch = flow.entry();
    let first = flow.stmt("a()");
    let second = flow.stmt("b()");
    let join = flow.stmt("return");
    flow.case(dispatch, first);
    flow.case(dispatch, second);
    flow.fall(first, join);
    flow.fall(second, join);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let [
        ControlNode::Switch {
            default_body,
            default_edge,
            ..
        },
        _,
    ] = top_level(&tree)[..]
    else {
        panic!("expected a switch: {tree:?}");
    };
    assert!(default_body.is_empty());
    assert_eq!(default_edge, None);
}

#[test]
fn arms_that_return_have_no_merge_and_end_where_they_return() {
    let mut flow = Flow::starting_with("switch k");
    let dispatch = flow.entry();
    let first = flow.stmt("return 1");
    let second = flow.stmt("return 2");
    let default_arm = flow.stmt("return 0");
    flow.case(dispatch, first);
    flow.case(dispatch, second);
    flow.fall(dispatch, default_arm);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    assert!(matches!(
        *tree.node(tree.root()),
        ControlNode::Switch { .. }
    ));
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "switch (switch k) {\n    case {\n        return 1\n    }\n    case {\n        return 2\n    }\n    default {\n        return 0\n    }\n}\n"
    );
}

#[test]
fn fallthrough_between_cases_is_a_reported_goto_to_the_next_arm() {
    // case 0: a();  /* falls through */  case 1: b(); break;  default: d()
    let mut flow = Flow::starting_with("switch k");
    let dispatch = flow.entry();
    let first = flow.stmt("a()");
    let second = flow.stmt("b()");
    let default_arm = flow.stmt("d()");
    let join = flow.stmt("return");
    flow.case(dispatch, first);
    flow.case(dispatch, second);
    flow.fall(dispatch, default_arm);
    flow.fall(first, second);
    flow.fall(second, join);
    flow.fall(default_arm, join);

    let (tree, report) = structure_and_check(&flow.cfg);

    assert!(!report.is_fully_structured());
    assert_eq!(report.gotos.len(), 1, "{report:?}");
    assert_eq!(report.gotos[0].target, second);
    assert_eq!(report.gotos[0].reason, GotoReason::RevisitedTarget);
    assert_eq!(goto_targets_in_tree(&tree), [second]);
    assert_eq!(label_blocks_in_tree(&tree), [second]);
    assert_eq!(report.unresolved_labels, []);

    // The goto carries exactly the one edge the structure cannot express.
    let implied = implied_flow(&tree, |block| flow.cfg.outgoing(block).next().is_none());
    assert_eq!(implied.through_gotos, BTreeSet::from([(first, second)]));
    let mut structural_edges = normal_edges(&flow.cfg);
    structural_edges.remove(&(first, second));
    assert_eq!(implied.structural, structural_edges);
}

#[test]
fn case_that_continues_the_loop_is_an_arm_that_ends_at_the_loop_header() {
    // while i < n { switch op[i] { case 0: continue; case 1: work(); default: rest() } }
    // Every arm ends by going back to the header, so the header is the merge.
    let mut flow = Flow::starting_with("i = 0");
    let header = flow.stmt("i < n");
    let dispatch = flow.stmt("switch op[i]");
    let work = flow.stmt("work()");
    let rest = flow.stmt("rest()");
    let exit = flow.stmt("return");
    flow.fall(flow.entry(), header);
    flow.branch(header, dispatch, exit);
    flow.case(dispatch, header);
    flow.case(dispatch, work);
    flow.fall(dispatch, rest);
    flow.jump(work, header);
    flow.jump(rest, header);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let switches = nodes_where(&tree, |node| matches!(node, ControlNode::Switch { .. }));
    let [ControlNode::Switch { cases, .. }] = switches
        .iter()
        .map(|&id| *tree.node(id))
        .collect::<Vec<_>>()[..]
    else {
        panic!("one switch: {tree:?}");
    };
    let cases = tree.cases(cases);
    assert_eq!(cases[0].block, header, "the case names its own target");
    assert!(
        cases[0].body.is_empty(),
        "an arm that goes to the merge has nothing to run"
    );
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "i = 0\nwhile {\n    i < n\n    break if !cond;\n    switch (switch op[i]) {\n        case {\n        }\n        case {\n            work()\n            continue;\n        }\n        default {\n            rest()\n            continue;\n        }\n    }\n}\nreturn\n"
    );
}

#[test]
fn case_arm_that_leaves_the_loop_is_a_break_like_the_same_arm_of_an_if() {
    // while i < n { switch op[i] { case 0: break; case 1: work(); continue; default: rest(); continue } }
    let mut flow = Flow::starting_with("i = 0");
    let header = flow.stmt("i < n");
    let dispatch = flow.stmt("switch op[i]");
    let work = flow.stmt("work()");
    let rest = flow.stmt("rest()");
    let exit = flow.stmt("return");
    flow.fall(flow.entry(), header);
    flow.branch(header, dispatch, exit);
    flow.case(dispatch, exit);
    flow.case(dispatch, work);
    flow.fall(dispatch, rest);
    flow.jump(work, header);
    flow.jump(rest, header);

    let (tree, report) = structure_and_check(&flow.cfg);

    assert_fully_structured(&tree, &report);
    assert_eq!(
        nodes_where(&tree, |node| matches!(
            node,
            ControlNode::Break { label: None }
        ))
        .len(),
        1
    );
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "i = 0\nwhile {\n    i < n\n    break if !cond;\n    switch (switch op[i]) {\n        case {\n            break;\n        }\n        case {\n            work()\n            continue;\n        }\n        default {\n            rest()\n            continue;\n        }\n    }\n}\nreturn\n"
    );
}
