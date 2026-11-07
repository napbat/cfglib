//! Branches in a loop body whose merge is the loop's latch while another
//! transfer leaves the loop: `for (…; …; ++i) { if (a) continue; if (b)
//! break; work }`.

use super::rejoining_arms::{blocks, structured};
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::test_util::MockInst;

/// The `for` loop of napbat/cfglib#35, with the transfers it keeps:
/// `continue` leads its test to the latch, `break` leads its test out of
/// the loop, and a dropped one falls into the next test instead.
fn for_loop(with_continue: bool, with_break: bool, back: EdgeKind) -> Cfg<MockInst> {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let [
        header,
        continue_test,
        skip,
        break_test,
        leave,
        work,
        latch,
        after,
    ] = blocks(
        &mut cfg,
        ["i < n", "a", "skip", "b", "leave", "work", "++i", "after"],
    );
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, continue_test, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, after, EdgeKind::ConditionalFalse);
    cfg.add_edge(continue_test, skip, EdgeKind::ConditionalTrue);
    cfg.add_edge(continue_test, break_test, EdgeKind::ConditionalFalse);
    let skipped = if with_continue { latch } else { break_test };
    cfg.add_edge(skip, skipped, EdgeKind::Unconditional);
    cfg.add_edge(break_test, leave, EdgeKind::ConditionalTrue);
    cfg.add_edge(break_test, work, EdgeKind::ConditionalFalse);
    let left = if with_break { after } else { work };
    cfg.add_edge(leave, left, EdgeKind::Unconditional);
    cfg.add_edge(work, latch, EdgeKind::Fallthrough);
    cfg.add_edge(latch, header, back);
    cfg
}

/// `if (a) continue` and `if (b) break` both: the `break` leaves the loop,
/// so the latch does not post-dominate `if (a)` in the whole graph, yet the
/// arms that stay in the loop rejoin there, and the latch runs once after
/// the branch, whatever kind the back edge has.
#[test]
fn a_continue_beside_a_break_merges_at_the_latch() {
    for back in [EdgeKind::Jump, EdgeKind::Back] {
        assert_eq!(
            structured(&for_loop(true, true, back)),
            "while {\n    i < n\n    break if !cond;\n    if (a) {\n        skip\n    } else {\n        \
             if (b) {\n            leave\n            break;\n        } else {\n            work\n        \
             }\n    }\n    ++i\n}\nafter\n",
            "{back:?}"
        );
    }
}

/// Either transfer alone leaves the latch the post-dominator merge.
#[test]
fn a_continue_alone_or_a_break_alone_merges_at_the_latch() {
    for (with_continue, with_break) in [(true, false), (false, true)] {
        let text = structured(&for_loop(with_continue, with_break, EdgeKind::Jump));
        assert_eq!(text.matches("++i").count(), 1, "{text}");
    }
}

/// `outer: for (…; …; ++i) { for (…; …; ++j) { if (a) continue outer; if (b)
/// break outer; } }`: the outer latch is reached by the `continue outer` and
/// by the end of the inner loop, while the `break outer` leaves both loops.
/// When the inner loop ends through a block without code, as symtree's
/// TypeScript lowering emits it, that block is the inner loop's follow, and
/// the `continue outer` that reaches the outer latch past it is a `break`
/// all the same.
#[test]
fn a_labeled_continue_beside_a_labeled_break_reaches_the_outer_latch_once() {
    for through_empty in [false, true] {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let [
            outer,
            start,
            inner,
            continue_test,
            skip,
            break_test,
            leave,
            inner_latch,
            outer_latch,
            after,
        ] = blocks(
            &mut cfg,
            [
                "i < n", "j = 0", "j < m", "a", "skip", "b", "leave", "++j", "++i", "after",
            ],
        );
        let inner_exit = if through_empty {
            let empty = cfg.new_block();
            cfg.add_edge(empty, outer_latch, EdgeKind::Fallthrough);
            empty
        } else {
            outer_latch
        };
        cfg.add_edge(cfg.entry(), outer, EdgeKind::Fallthrough);
        cfg.add_edge(outer, start, EdgeKind::ConditionalTrue);
        cfg.add_edge(outer, after, EdgeKind::ConditionalFalse);
        cfg.add_edge(start, inner, EdgeKind::Fallthrough);
        cfg.add_edge(inner, continue_test, EdgeKind::ConditionalTrue);
        cfg.add_edge(inner, inner_exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(continue_test, skip, EdgeKind::ConditionalTrue);
        cfg.add_edge(continue_test, break_test, EdgeKind::ConditionalFalse);
        cfg.add_edge(skip, outer_latch, EdgeKind::Unconditional);
        cfg.add_edge(break_test, leave, EdgeKind::ConditionalTrue);
        cfg.add_edge(break_test, inner_latch, EdgeKind::ConditionalFalse);
        cfg.add_edge(leave, after, EdgeKind::Unconditional);
        cfg.add_edge(inner_latch, inner, EdgeKind::Back);
        cfg.add_edge(outer_latch, outer, EdgeKind::Back);

        assert_eq!(
            structured(&cfg),
            ".bb1:\n    while {\n        i < n\n        break if !cond;\n        j = 0\n        while {\n\
             \x20           j < m\n            break if !cond;\n            if (a) {\n                \
             skip\n                break;\n            } else {\n                if (b) {\n                \
             \x20   leave\n                    break .bb1;\n                } else {\n                    \
             ++j\n                    continue;\n                }\n            }\n        }\n        \
             ++i\n    }\nafter\n",
            "{through_empty}"
        );
    }
}
