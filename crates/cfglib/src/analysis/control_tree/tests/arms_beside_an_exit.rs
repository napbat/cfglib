//! Arms that rejoin beside an arm that leaves the function, in a walk with no
//! stop: the body of a function outside any `try`, and a loop body.

use super::rejoining_arms::{blocks, structured};
use crate::ControlTree;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::flow::FlowEffect;
use crate::test_util::{MockInst, ff};

/// A new block that leaves the function through `effect`.
fn exit(cfg: &mut Cfg<MockInst>, effect: FlowEffect, name: &'static str) -> BlockId {
    let block = cfg.new_block();
    cfg.block_mut(block).push(MockInst(effect, name));
    block
}

/// `switch (k) { case: p; case: q; default: throw }; z; return`, outside any
/// `try`: the completing cases rejoin at `z`, and the throwing default ends
/// its arm, as the C# `switch` with `default: throw new X();` lowers.
#[test]
fn switch_arms_beside_a_throwing_default_rejoin_outside_a_try() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let [dispatch, first_case, second_case, join] = blocks(&mut cfg, ["k", "p", "q", "z"]);
    let throw = exit(&mut cfg, FlowEffect::Terminate, "throw");
    let leave = exit(&mut cfg, FlowEffect::Return, "ret");
    cfg.add_edge(cfg.entry(), dispatch, EdgeKind::Fallthrough);
    cfg.add_edge(dispatch, first_case, EdgeKind::SwitchCase);
    cfg.add_edge(dispatch, second_case, EdgeKind::SwitchCase);
    cfg.add_edge(dispatch, throw, EdgeKind::Fallthrough);
    cfg.add_edge(first_case, join, EdgeKind::Unconditional);
    cfg.add_edge(second_case, join, EdgeKind::Unconditional);
    cfg.add_edge(join, leave, EdgeKind::Fallthrough);

    let text = structured(&cfg);
    assert_eq!(text.matches('z').count(), 1, "{text}");
    assert!(
        text.ends_with("    default {\n        throw\n    }\n}\nz\nret\n"),
        "{text}"
    );
}

/// `if (a) { x } else if (b) { y } else { return }; z; return` outside any
/// `try`: the arms rejoin at `z`, and the early return ends its arm.
#[test]
fn else_if_arms_beside_a_return_rejoin_outside_a_try() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let [test, then_arm, second_test, second_arm, join] =
        blocks(&mut cfg, ["a", "x", "b", "y", "z"]);
    let early = exit(&mut cfg, FlowEffect::Return, "early");
    let leave = exit(&mut cfg, FlowEffect::Return, "ret");
    cfg.add_edge(cfg.entry(), test, EdgeKind::Fallthrough);
    cfg.add_edge(test, then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(test, second_test, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, join, EdgeKind::Fallthrough);
    cfg.add_edge(second_test, second_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(second_test, early, EdgeKind::ConditionalFalse);
    cfg.add_edge(second_arm, join, EdgeKind::Unconditional);
    cfg.add_edge(join, leave, EdgeKind::Fallthrough);

    assert_eq!(
        structured(&cfg),
        "if (a) {\n    x\n} else {\n    if (b) {\n        y\n    } else {\n        early\n    }\n}\
         \nz\nret\n"
    );
}

/// The code after the join may leave by several exits that the completing
/// arms all reach: `switch (k) { case: p; case: q; default: throw }; if (c)
/// { return 1 } else { return 2 }`.
#[test]
fn arms_rejoin_before_code_that_leaves_by_several_exits() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let [dispatch, first_case, second_case, join] = blocks(&mut cfg, ["k", "p", "q", "c"]);
    let throw = exit(&mut cfg, FlowEffect::Terminate, "throw");
    let one = exit(&mut cfg, FlowEffect::Return, "ret1");
    let two = exit(&mut cfg, FlowEffect::Return, "ret2");
    cfg.add_edge(cfg.entry(), dispatch, EdgeKind::Fallthrough);
    cfg.add_edge(dispatch, first_case, EdgeKind::SwitchCase);
    cfg.add_edge(dispatch, second_case, EdgeKind::SwitchCase);
    cfg.add_edge(dispatch, throw, EdgeKind::Fallthrough);
    cfg.add_edge(first_case, join, EdgeKind::Unconditional);
    cfg.add_edge(second_case, join, EdgeKind::Unconditional);
    cfg.add_edge(join, one, EdgeKind::ConditionalTrue);
    cfg.add_edge(join, two, EdgeKind::ConditionalFalse);

    let text = structured(&cfg);
    assert!(
        text.ends_with(
            "    default {\n        throw\n    }\n}\nif (c) {\n    ret1\n} else {\n    ret2\n}\n"
        ),
        "{text}"
    );
}

/// `if (a) { x; if (c) { return } } else { y }; z; return`: an arm that may
/// leave early still rejoins, and its early exit ends only the inner arm.
#[test]
fn an_arm_that_may_return_early_still_rejoins() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let [test, then_arm, early_test, second_arm, join] =
        blocks(&mut cfg, ["a", "x", "c", "y", "z"]);
    let early = exit(&mut cfg, FlowEffect::Return, "early");
    let leave = exit(&mut cfg, FlowEffect::Return, "ret");
    cfg.add_edge(cfg.entry(), test, EdgeKind::Fallthrough);
    cfg.add_edge(test, then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(test, second_arm, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, early_test, EdgeKind::Fallthrough);
    cfg.add_edge(early_test, early, EdgeKind::ConditionalTrue);
    cfg.add_edge(early_test, join, EdgeKind::ConditionalFalse);
    cfg.add_edge(second_arm, join, EdgeKind::Fallthrough);
    cfg.add_edge(join, leave, EdgeKind::Fallthrough);

    let printed = structured(&cfg);
    assert_eq!(printed.matches('z').count(), 1, "{printed}");
    assert!(
        printed.ends_with("} else {\n    y\n}\nz\nret\n"),
        "{printed}"
    );
}

/// `while (l) { if (a) { x } else if (b) { y } else { throw }; z }`: in a
/// loop body, the arms rejoin at the latch, which ends the code of the body.
#[test]
fn arms_beside_a_throw_rejoin_in_a_loop_body() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let [
        header,
        test,
        then_arm,
        second_test,
        second_arm,
        latch,
        after,
    ] = blocks(&mut cfg, ["l", "a", "x", "b", "y", "z", "after"]);
    let throw = exit(&mut cfg, FlowEffect::Terminate, "throw");
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, test, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, after, EdgeKind::ConditionalFalse);
    cfg.add_edge(test, then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(test, second_test, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, latch, EdgeKind::Fallthrough);
    cfg.add_edge(second_test, second_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(second_test, throw, EdgeKind::ConditionalFalse);
    cfg.add_edge(second_arm, latch, EdgeKind::Unconditional);
    cfg.add_edge(latch, header, EdgeKind::Unconditional);

    let printed = structured(&cfg);
    assert_eq!(printed.matches('z').count(), 1, "{printed}");
    assert!(printed.contains("    }\n    z\n"), "{printed}");
}

/// `if (a) { x } else { return }; z; return`: with one arm that completes,
/// nothing rejoins, and that arm runs on to the end of the function inside
/// the `if` as before.
#[test]
fn a_lone_completing_arm_still_runs_to_the_end_of_the_function() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let [test, then_arm, join] = blocks(&mut cfg, ["a", "x", "z"]);
    let early = exit(&mut cfg, FlowEffect::Return, "early");
    let leave = exit(&mut cfg, FlowEffect::Return, "ret");
    cfg.add_edge(cfg.entry(), test, EdgeKind::Fallthrough);
    cfg.add_edge(test, then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(test, early, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, join, EdgeKind::Fallthrough);
    cfg.add_edge(join, leave, EdgeKind::Fallthrough);

    assert_eq!(
        structured(&cfg),
        "if (a) {\n    x\n    z\n    ret\n} else {\n    early\n}\n"
    );
}

/// `switch (k) { case: p; case: q; default: throw }` whose cases leave by
/// their own returns: no exit is shared, so nothing rejoins and the switch
/// keeps its form.
#[test]
fn arms_that_leave_by_their_own_exits_do_not_rejoin() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let [dispatch] = blocks(&mut cfg, ["k"]);
    let first_case = exit(&mut cfg, FlowEffect::Return, "p");
    let second_case = exit(&mut cfg, FlowEffect::Return, "q");
    let throw = exit(&mut cfg, FlowEffect::Terminate, "throw");
    cfg.add_edge(cfg.entry(), dispatch, EdgeKind::Fallthrough);
    cfg.add_edge(dispatch, first_case, EdgeKind::SwitchCase);
    cfg.add_edge(dispatch, second_case, EdgeKind::SwitchCase);
    cfg.add_edge(dispatch, throw, EdgeKind::Fallthrough);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert!(report.is_fully_structured(), "{report:?}");
    let text = super::pseudocode(&cfg, &tree);
    assert!(
        text.ends_with("    default {\n        throw\n    }\n}\n"),
        "{text}"
    );
}

/// `if (e) { jumper; goto dst } else if (normal) { dst } else { return }`:
/// the jump into the other arm lands where the arms rejoin beside the arm
/// that returns, so it is no goto.
#[test]
fn a_jump_into_the_other_arm_rejoins_beside_an_arm_that_returns() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    let [jumper, normal] = blocks(&mut cfg, ["jumper", "normal"]);
    let target = exit(&mut cfg, FlowEffect::Return, "dst");
    let other = exit(&mut cfg, FlowEffect::Return, "other");
    cfg.add_edge(cfg.entry(), jumper, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), normal, EdgeKind::ConditionalFalse);
    cfg.add_edge(jumper, target, EdgeKind::Jump);
    cfg.add_edge(normal, target, EdgeKind::ConditionalTrue);
    cfg.add_edge(normal, other, EdgeKind::ConditionalFalse);

    assert_eq!(
        structured(&cfg),
        "if (entry) {\n    jumper\n} else {\n    if (normal) {\n    } else {\n        other\n    }\n}\
         \ndst\n"
    );
}
