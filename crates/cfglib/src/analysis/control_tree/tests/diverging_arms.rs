//! Branches with an arm that never completes, such as a handler arm that
//! rethrows, and transfers into blocks without code.

extern crate alloc;
use alloc::vec;

use super::{body_contains_block, find_try_catch, find_try_handlers, has_goto_to, pseudocode};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::flow::FlowEffect;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::{MockInst, ff};
use crate::{ControlNode, ControlTree, GotoDiagnostic, GotoReason};

/// A catch region over `protected` whose handler at `entry` owns `body`.
fn catch_region(id: u32, protected: &[BlockId], entry: BlockId, body: &[BlockId]) -> Region {
    Region {
        id: RegionId::from_raw(id),
        protected_blocks: protected.iter().copied().collect(),
        handlers: vec![Handler {
            entry,
            body: HandlerBody::known(body.iter().copied()),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    }
}

#[test]
fn handler_branch_with_a_diverging_arm_converges_at_the_continuation() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let try_body = cfg.new_block();
    let handler = cfg.new_block();
    let then_arm = cfg.new_block();
    let rethrow = cfg.new_block();
    let after = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(try_body).push(ff("try_inst"));
    cfg.block_mut(handler).push(ff("filter_test"));
    cfg.block_mut(then_arm).push(ff("handler_body"));
    cfg.block_mut(rethrow).push(ff("rethrow"));
    cfg.block_mut(after).push(ff("after"));
    cfg.add_edge(cfg.entry(), try_body, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, after, EdgeKind::ExceptionLeave);
    cfg.add_edge(try_body, handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(handler, then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(handler, rethrow, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, after, EdgeKind::Fallthrough);
    cfg.add_edge(rethrow, exit, EdgeKind::ExceptionUnwind);
    cfg.add_region(catch_region(
        0,
        &[try_body],
        handler,
        &[handler, then_arm, rethrow],
    ));

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    let text = pseudocode(&cfg, &tree);
    assert!(report.is_fully_structured(), "{report:?}\n{text}");
    let handlers = find_try_handlers(&tree).expect("the region is a try");
    assert!(
        body_contains_block(&tree, handlers[0].body, then_arm),
        "{text}"
    );
    assert!(
        body_contains_block(&tree, handlers[0].body, rethrow),
        "{text}"
    );
    assert!(
        !body_contains_block(&tree, handlers[0].body, after),
        "{text}"
    );
    assert_eq!(text.matches("after").count(), 1, "{text}");
}

#[test]
fn handler_dispatch_with_a_diverging_default_converges_at_the_continuation() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let try_body = cfg.new_block();
    let handler = cfg.new_block();
    let first = cfg.new_block();
    let second = cfg.new_block();
    let rethrow = cfg.new_block();
    let after = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(try_body).push(ff("try_inst"));
    cfg.block_mut(handler).push(ff("switch_code"));
    cfg.block_mut(first).push(ff("first_case"));
    cfg.block_mut(second).push(ff("second_case"));
    cfg.block_mut(rethrow).push(ff("rethrow"));
    cfg.block_mut(after).push(ff("after"));
    cfg.add_edge(cfg.entry(), try_body, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, after, EdgeKind::ExceptionLeave);
    cfg.add_edge(try_body, handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(handler, first, EdgeKind::SwitchCase);
    cfg.add_edge(handler, second, EdgeKind::SwitchCase);
    cfg.add_edge(handler, rethrow, EdgeKind::Fallthrough);
    cfg.add_edge(first, after, EdgeKind::Fallthrough);
    cfg.add_edge(second, after, EdgeKind::Fallthrough);
    cfg.add_edge(rethrow, exit, EdgeKind::ExceptionUnwind);
    cfg.add_region(catch_region(
        0,
        &[try_body],
        handler,
        &[handler, first, second, rethrow],
    ));

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    let text = pseudocode(&cfg, &tree);
    assert!(report.is_fully_structured(), "{report:?}\n{text}");
    let handlers = find_try_handlers(&tree).expect("the region is a try");
    assert!(
        body_contains_block(&tree, handlers[0].body, first),
        "{text}"
    );
    assert!(
        body_contains_block(&tree, handlers[0].body, second),
        "{text}"
    );
    assert!(
        !body_contains_block(&tree, handlers[0].body, after),
        "{text}"
    );
    assert_eq!(text.matches("after").count(), 1, "{text}");
}

/// `try { try { a } catch { if (test) { body } else { rethrow } } } catch
/// { outer } after`, where the inner try leaves through a block without code
/// and the handler arm joins it through another, as a source lowering emits
/// them; the rethrow raises into the outer handler.
#[test]
fn rethrowing_handler_of_a_try_nested_last_in_a_try_is_structured() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let inner_try = cfg.new_block();
    let inner_pad = cfg.new_block();
    let then_arm = cfg.new_block();
    let join = cfg.new_block();
    let rethrow = cfg.new_block();
    let leave = cfg.new_block();
    let outer_pad = cfg.new_block();
    let after = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(inner_try).push(ff("inner_try"));
    cfg.block_mut(inner_pad).push(ff("filter_test"));
    cfg.block_mut(then_arm).push(ff("handler_body"));
    cfg.block_mut(rethrow).push(ff("rethrow"));
    cfg.block_mut(outer_pad).push(ff("outer_catch"));
    cfg.block_mut(after)
        .push(MockInst(FlowEffect::Return, "after"));
    cfg.add_edge(cfg.entry(), inner_try, EdgeKind::Fallthrough);
    cfg.add_edge(inner_try, leave, EdgeKind::ExceptionLeave);
    cfg.add_edge(inner_try, inner_pad, EdgeKind::ExceptionHandler);
    cfg.add_edge(inner_pad, then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(inner_pad, rethrow, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, join, EdgeKind::Fallthrough);
    cfg.add_edge(join, leave, EdgeKind::Fallthrough);
    cfg.add_edge(rethrow, outer_pad, EdgeKind::ExceptionHandler);
    cfg.add_edge(leave, after, EdgeKind::ExceptionLeave);
    cfg.add_edge(outer_pad, after, EdgeKind::Fallthrough);
    let outer = cfg.add_region(catch_region(
        0,
        &[inner_try, inner_pad, then_arm, join, rethrow, leave],
        outer_pad,
        &[outer_pad],
    ));
    let mut inner = catch_region(
        1,
        &[inner_try],
        inner_pad,
        &[inner_pad, then_arm, join, rethrow],
    );
    inner.parent = Some(outer);
    cfg.add_region(inner);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    let text = pseudocode(&cfg, &tree);
    assert!(report.is_fully_structured(), "{report:?}\n{text}");
    let (outer_body, _, _) = find_try_catch(&tree).expect("the outer region is a try");
    assert!(body_contains_block(&tree, outer_body, rethrow), "{text}");
    assert!(!body_contains_block(&tree, outer_body, after), "{text}");
}

/// A walk that reaches a block without code that another arm already passed
/// goes on where that block leads: `if (a) { x } else if (b) { y } else {
/// rethrow }` in a handler, whose arms join through blocks without code.
#[test]
fn an_arm_converges_through_a_block_without_code_another_arm_passed() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let try_body = cfg.new_block();
    let handler = cfg.new_block();
    let x = cfg.new_block();
    let join = cfg.new_block();
    let second_test = cfg.new_block();
    let y = cfg.new_block();
    let inner_join = cfg.new_block();
    let rethrow = cfg.new_block();
    let after = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(try_body).push(ff("try_inst"));
    cfg.block_mut(handler).push(ff("a"));
    cfg.block_mut(x).push(ff("x"));
    cfg.block_mut(second_test).push(ff("b"));
    cfg.block_mut(y).push(ff("y"));
    cfg.block_mut(rethrow).push(ff("rethrow"));
    cfg.block_mut(after).push(ff("after"));
    cfg.add_edge(cfg.entry(), try_body, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, after, EdgeKind::ExceptionLeave);
    cfg.add_edge(try_body, handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(handler, x, EdgeKind::ConditionalTrue);
    cfg.add_edge(handler, second_test, EdgeKind::ConditionalFalse);
    cfg.add_edge(x, join, EdgeKind::Fallthrough);
    cfg.add_edge(join, after, EdgeKind::Fallthrough);
    cfg.add_edge(second_test, y, EdgeKind::ConditionalTrue);
    cfg.add_edge(second_test, rethrow, EdgeKind::ConditionalFalse);
    cfg.add_edge(y, inner_join, EdgeKind::Fallthrough);
    cfg.add_edge(inner_join, join, EdgeKind::Fallthrough);
    cfg.add_edge(rethrow, exit, EdgeKind::ExceptionUnwind);
    cfg.add_region(catch_region(
        0,
        &[try_body],
        handler,
        &[handler, x, join, second_test, y, inner_join, rethrow],
    ));

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    let text = pseudocode(&cfg, &tree);
    assert!(report.is_fully_structured(), "{report:?}\n{text}");
    assert_eq!(text.matches("after").count(), 1, "{text}");
}

/// A goto that a block without code would take names the block that it
/// leads to, which carries a label, never the block itself. The arms cross:
/// both reach `tail` through the block without code and both reach `c`, so
/// neither is where they rejoin, and the second arm reaches each by a goto.
#[test]
fn a_goto_through_a_block_without_code_names_the_block_it_leads_to() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let a = cfg.new_block();
    let empty = cfg.new_block();
    let tail = cfg.new_block();
    let second_test = cfg.new_block();
    let b = cfg.new_block();
    let c = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("first_test"));
    cfg.block_mut(a).push(ff("a"));
    cfg.block_mut(tail)
        .push(MockInst(FlowEffect::Return, "tail"));
    cfg.block_mut(second_test).push(ff("second_test"));
    cfg.block_mut(b).push(ff("b"));
    cfg.block_mut(c).push(MockInst(FlowEffect::Return, "c"));
    cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), second_test, EdgeKind::ConditionalFalse);
    cfg.add_edge(a, empty, EdgeKind::ConditionalTrue);
    cfg.add_edge(a, c, EdgeKind::ConditionalFalse);
    cfg.add_edge(empty, tail, EdgeKind::Fallthrough);
    cfg.add_edge(second_test, b, EdgeKind::ConditionalTrue);
    cfg.add_edge(second_test, c, EdgeKind::ConditionalFalse);
    cfg.add_edge(b, empty, EdgeKind::Fallthrough);

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    let text = pseudocode(&cfg, &tree);
    assert_eq!(
        report.gotos,
        [
            GotoDiagnostic {
                target: tail,
                reason: GotoReason::RevisitedTarget,
            },
            GotoDiagnostic {
                target: c,
                reason: GotoReason::RevisitedTarget,
            }
        ],
        "{text}"
    );
    assert!(report.unresolved_labels.is_empty(), "{report:?}\n{text}");
    assert!(has_goto_to(&tree, tail), "{text}");
    assert!(
        tree.preorder()
            .any(|id| matches!(*tree.node(id), ControlNode::Label { block, .. } if block == tail)),
        "{text}"
    );
}
