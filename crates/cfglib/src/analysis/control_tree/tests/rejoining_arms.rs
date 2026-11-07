//! Arms that rejoin beside an arm that diverges, and a `try` body or catch
//! arm that completes into the finally that ordinary flow enters.

extern crate alloc;

use super::pseudocode;
use crate::ControlTree;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::flow::FlowEffect;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::{MockInst, ff};

/// A region over `protected` with one handler of `kind` at `entry` that owns
/// `body`.
fn region(
    id: u32,
    protected: &[BlockId],
    handlers: &[(BlockId, HandlerKind, &[BlockId])],
) -> Region {
    Region {
        id: RegionId::from_raw(id),
        protected_blocks: protected.iter().copied().collect(),
        handlers: handlers
            .iter()
            .map(|&(entry, kind, body)| Handler {
                entry,
                body: HandlerBody::known(body.iter().copied()),
                kind,
            })
            .collect(),
        parent: None,
    }
}

/// New blocks, one per mnemonic, each holding that one instruction.
fn blocks<const N: usize>(cfg: &mut Cfg<MockInst>, names: [&'static str; N]) -> [BlockId; N] {
    names.map(|name| {
        let block = cfg.new_block();
        cfg.block_mut(block).push(ff(name));
        block
    })
}

/// The structured text of `cfg`, which must structure without a goto.
fn structured(cfg: &Cfg<MockInst>) -> alloc::string::String {
    let (tree, report) = ControlTree::compute_with_report(cfg);
    let text = pseudocode(cfg, &tree);
    assert!(report.is_fully_structured(), "{report:?}\n{text}");
    text
}

/// `try { t } catch { if (a) { x } else if (b) { y } else { rethrow }; z }
/// after`: two arms rejoin at `z`, which runs before the continuation.
#[test]
fn handler_arms_beside_a_rethrow_rejoin_at_code_before_the_continuation() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    let [
        protected,
        test,
        then_arm,
        second_test,
        second_arm,
        rethrow,
        join,
        after,
    ] = blocks(&mut cfg, ["t", "a", "x", "b", "y", "rethrow", "z", "after"]);
    let exit = cfg.new_block();
    cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
    cfg.add_edge(protected, after, EdgeKind::ExceptionLeave);
    cfg.add_edge(protected, test, EdgeKind::ExceptionHandler);
    cfg.add_edge(test, then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(test, second_test, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, join, EdgeKind::Fallthrough);
    cfg.add_edge(second_test, second_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(second_test, rethrow, EdgeKind::ConditionalFalse);
    cfg.add_edge(second_arm, join, EdgeKind::Unconditional);
    cfg.add_edge(join, after, EdgeKind::Fallthrough);
    cfg.add_edge(rethrow, exit, EdgeKind::ExceptionUnwind);
    cfg.add_region(region(
        0,
        &[protected],
        &[(
            test,
            HandlerKind::Catch,
            &[test, then_arm, second_test, second_arm, rethrow, join],
        )],
    ));

    assert_eq!(
        structured(&cfg),
        "entry\ntry {\n    t\n} catch {\n    if (a) {\n        x\n    } else {\n        if (b) {\n            \
         y\n        } else {\n            rethrow\n        }\n    }\n    z\n}\nafter\n"
    );
}

/// `try { t } catch { switch (k) { case: p; case: q; default: rethrow }; z }
/// after`: the completing cases rejoin at `z`, before the continuation.
#[test]
fn switch_arms_beside_a_throwing_default_rejoin_at_code_before_the_continuation() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    let [
        protected,
        dispatch,
        first_case,
        second_case,
        rethrow,
        join,
        after,
    ] = blocks(&mut cfg, ["t", "k", "p", "q", "rethrow", "z", "after"]);
    let exit = cfg.new_block();
    cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
    cfg.add_edge(protected, after, EdgeKind::ExceptionLeave);
    cfg.add_edge(protected, dispatch, EdgeKind::ExceptionHandler);
    cfg.add_edge(dispatch, first_case, EdgeKind::SwitchCase);
    cfg.add_edge(dispatch, second_case, EdgeKind::SwitchCase);
    cfg.add_edge(dispatch, rethrow, EdgeKind::Fallthrough);
    cfg.add_edge(first_case, join, EdgeKind::Unconditional);
    cfg.add_edge(second_case, join, EdgeKind::Unconditional);
    cfg.add_edge(join, after, EdgeKind::Fallthrough);
    cfg.add_edge(rethrow, exit, EdgeKind::ExceptionUnwind);
    cfg.add_region(region(
        0,
        &[protected],
        &[(
            dispatch,
            HandlerKind::Catch,
            &[dispatch, first_case, second_case, rethrow, join],
        )],
    ));

    let text = structured(&cfg);
    assert_eq!(text.matches('z').count(), 1, "{text}");
    assert!(
        text.ends_with(
            "        default {\n            rethrow\n        }\n    }\n    z\n}\nafter\n"
        ),
        "{text}"
    );
}

/// `try { if (a) { x } else if (b) { y } else { return }; z } catch { h }
/// after`: the rejoin is not tied to a handler. In a try body, the arms that
/// reach the continuation still meet at `z` first.
#[test]
fn try_body_arms_beside_a_return_rejoin_at_code_before_the_continuation() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    let [test, then_arm, second_test, second_arm, join, pad, after] =
        blocks(&mut cfg, ["a", "x", "b", "y", "z", "h", "after"]);
    let leave = cfg.new_block();
    cfg.block_mut(leave)
        .push(MockInst(FlowEffect::Return, "leave"));
    cfg.add_edge(cfg.entry(), test, EdgeKind::Fallthrough);
    cfg.add_edge(test, then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(test, second_test, EdgeKind::ConditionalFalse);
    cfg.add_edge(test, pad, EdgeKind::ExceptionHandler);
    cfg.add_edge(then_arm, join, EdgeKind::Fallthrough);
    cfg.add_edge(second_test, second_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(second_test, leave, EdgeKind::ConditionalFalse);
    cfg.add_edge(second_arm, join, EdgeKind::Fallthrough);
    cfg.add_edge(join, after, EdgeKind::ExceptionLeave);
    cfg.add_edge(pad, after, EdgeKind::Fallthrough);
    cfg.add_region(region(
        0,
        &[test, then_arm, second_test, second_arm, leave, join],
        &[(pad, HandlerKind::Catch, &[pad])],
    ));

    assert_eq!(
        structured(&cfg),
        "entry\ntry {\n    if (a) {\n        x\n    } else {\n        if (b) {\n            y\n        \
         } else {\n            leave\n        }\n    }\n    z\n} catch {\n    h\n}\nafter\n"
    );
}

/// `try { t } catch { if (a) { x } else { rethrow } } after`: with one arm
/// that completes, nothing rejoins, and the arm runs on to the continuation
/// inside the `if` as before.
#[test]
fn a_lone_completing_arm_still_runs_to_the_continuation() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    let [protected, test, then_arm, join, rethrow, after] =
        blocks(&mut cfg, ["t", "a", "x", "z", "rethrow", "after"]);
    let exit = cfg.new_block();
    cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
    cfg.add_edge(protected, after, EdgeKind::ExceptionLeave);
    cfg.add_edge(protected, test, EdgeKind::ExceptionHandler);
    cfg.add_edge(test, then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(test, rethrow, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, join, EdgeKind::Fallthrough);
    cfg.add_edge(join, after, EdgeKind::Fallthrough);
    cfg.add_edge(rethrow, exit, EdgeKind::ExceptionUnwind);
    cfg.add_region(region(
        0,
        &[protected],
        &[(test, HandlerKind::Catch, &[test, then_arm, join, rethrow])],
    ));

    assert_eq!(
        structured(&cfg),
        "entry\ntry {\n    t\n} catch {\n    if (a) {\n        x\n        z\n    } else {\n        \
         rethrow\n    }\n}\nafter\n"
    );
}

/// `try { if (a) { x } else { y } } finally { f } after`, lowered with one
/// copy of the finally that the try body runs into and that goes on to
/// `after`.
#[test]
fn a_try_body_completes_into_the_finally_it_runs_into() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    let [test, then_arm, second_arm, cleanup, after] =
        blocks(&mut cfg, ["a", "x", "y", "f", "after"]);
    let leave = cfg.new_block();
    cfg.add_edge(cfg.entry(), test, EdgeKind::Fallthrough);
    cfg.add_edge(test, then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(test, second_arm, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, leave, EdgeKind::Fallthrough);
    cfg.add_edge(second_arm, leave, EdgeKind::Fallthrough);
    cfg.add_edge(leave, cleanup, EdgeKind::ExceptionLeave);
    for block in [test, then_arm, second_arm] {
        cfg.add_edge(block, cleanup, EdgeKind::ExceptionUnwind);
    }
    cfg.add_edge(cleanup, after, EdgeKind::ExceptionLeave);
    cfg.add_region(region(
        0,
        &[test, then_arm, second_arm, leave],
        &[(cleanup, HandlerKind::Finally, &[cleanup])],
    ));

    assert_eq!(
        structured(&cfg),
        "entry\ntry {\n    if (a) {\n        x\n    } else {\n        y\n    }\n} finally {\n    f\n}\
         \nafter\n"
    );
}

/// `try { try { t } catch { c } } finally { f } after`: the inner try goes on
/// at the outer finally's entry, which ends the outer body.
#[test]
fn a_catch_nested_in_a_try_finally_completes_into_the_finally() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    let [protected, inner_pad, cleanup, after] = blocks(&mut cfg, ["t", "c", "f", "after"]);
    cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
    cfg.add_edge(protected, cleanup, EdgeKind::ExceptionLeave);
    cfg.add_edge(protected, inner_pad, EdgeKind::ExceptionHandler);
    cfg.add_edge(inner_pad, cleanup, EdgeKind::ExceptionLeave);
    cfg.add_edge(inner_pad, cleanup, EdgeKind::ExceptionUnwind);
    cfg.add_edge(cleanup, after, EdgeKind::ExceptionLeave);
    cfg.add_region(region(
        0,
        &[protected],
        &[(inner_pad, HandlerKind::Catch, &[inner_pad])],
    ));
    cfg.add_region(region(
        1,
        &[protected, inner_pad],
        &[(cleanup, HandlerKind::Finally, &[cleanup])],
    ));

    assert_eq!(
        structured(&cfg),
        "entry\ntry {\n    try {\n        t\n    } catch {\n        c\n    }\n} finally {\n    f\n}\
         \nafter\n"
    );
}

/// A finally that only exception edges enter keeps the region's continuation
/// as the end of the try body: the body that falls through to `after` is not
/// cut short at the finally's entry.
#[test]
fn a_finally_entered_only_by_exceptions_leaves_the_body_to_its_continuation() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    let [protected, cleanup, after] = blocks(&mut cfg, ["t", "f", "after"]);
    cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
    cfg.add_edge(protected, after, EdgeKind::Fallthrough);
    cfg.add_edge(protected, cleanup, EdgeKind::ExceptionUnwind);
    cfg.add_region(region(
        0,
        &[protected],
        &[(cleanup, HandlerKind::Finally, &[cleanup])],
    ));

    assert_eq!(
        structured(&cfg),
        "entry\ntry {\n    t\n} finally {\n    f\n}\nafter\n"
    );
}
