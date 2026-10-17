//! Exception-region structuring: try/catch, handler extents, and the code
//! around them.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::{
    body_contains_block, body_subtree, contains_block, find_try_catch, find_try_handlers,
    has_goto_to, has_node_kind, leaf_block, pseudocode,
};
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::{MockInst, ff};
use crate::{ControlNode, ControlTree};

#[test]
fn compute_try_catch_produces_try_node() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    // entry(0) → try_body(1) → after(3)
    //            try_body(1) --Exception--> handler(2) → after(3)
    let try_body = cfg.new_block(); // 1
    let handler_block = cfg.new_block(); // 2
    let after = cfg.new_block(); // 3

    cfg.block_mut(cfg.entry())
        .instructions_mut()
        .push(ff("entry"));
    cfg.block_mut(try_body)
        .instructions_mut()
        .push(ff("try_inst"));
    cfg.block_mut(handler_block)
        .instructions_mut()
        .push(ff("catch_inst"));
    cfg.block_mut(after).instructions_mut().push(ff("after"));

    cfg.add_edge(cfg.entry(), try_body, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, after, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, handler_block, EdgeKind::ExceptionHandler);
    cfg.add_edge(handler_block, after, EdgeKind::Fallthrough);

    let mut protected = BTreeSet::new();
    protected.insert(try_body);
    let region = cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: protected,
        handlers: vec![Handler {
            entry: handler_block,
            body: HandlerBody::known([handler_block]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });

    let tree = ControlTree::compute(&cfg);
    let found = has_node_kind(&tree, |n| matches!(n, ControlNode::TryCatch { .. }));
    assert!(found, "should contain TryCatch node: {tree:?}");
    let pseudo = pseudocode(&cfg, &tree);
    assert!(
        pseudo.contains("try"),
        "pseudocode should contain try: {pseudo}"
    );
    let handlers = find_try_handlers(&tree).expect("known body produces structured handlers");
    assert_eq!(handlers.len(), 1);
    assert_eq!(handlers[0].handler, crate::HandlerRef::new(region, 0));
    assert_eq!(handlers[0].entry, handler_block);
    assert_eq!(handlers[0].kind, HandlerKind::Catch);
    assert!(body_contains_block(&tree, handlers[0].body, handler_block));
    assert!(
        !body_contains_block(&tree, handlers[0].body, after),
        "handler lifting must stop at the declared complete extent: {tree:?}"
    );
}

#[test]
fn compute_does_not_invent_structure_for_unknown_handler_body() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let try_body = cfg.new_block();
    let handler = cfg.new_block();
    let after = cfg.new_block();
    let entry = cfg.entry();

    cfg.block_mut(entry).instructions_mut().push(ff("entry"));
    cfg.block_mut(try_body)
        .instructions_mut()
        .push(ff("try_inst"));
    cfg.block_mut(handler)
        .instructions_mut()
        .push(ff("catch_inst"));
    cfg.block_mut(after).instructions_mut().push(ff("after"));
    cfg.add_edge(entry, try_body, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, after, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(handler, after, EdgeKind::Fallthrough);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [try_body].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::unknown(),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert!(
        !has_node_kind(&tree, |node| matches!(node, ControlNode::TryCatch { .. })),
        "unknown extents must remain unstructured: {tree:?}"
    );
    assert!(
        contains_block(&tree, try_body),
        "ordinary CFG lifting still retains the protected block: {tree:?}"
    );
    assert!(
        contains_block(&tree, after),
        "the normal continuation survives an unknown extent: {tree:?}"
    );
    assert!(
        contains_block(&tree, handler),
        "the handler code survives an unknown extent: {tree:?}"
    );
    assert_eq!(report.unstructured_regions, [RegionId::from_raw(0)]);
}

#[test]
fn compute_preserves_fault_and_filter_handler_kinds() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let try_body = cfg.new_block();
    let fault = cfg.new_block();
    let filter = cfg.new_block();
    let filtered_handler = cfg.new_block();
    let after = cfg.new_block();
    let entry = cfg.entry();

    cfg.block_mut(try_body)
        .instructions_mut()
        .push(ff("try_inst"));
    cfg.block_mut(fault)
        .instructions_mut()
        .push(ff("fault_inst"));
    cfg.block_mut(filtered_handler)
        .instructions_mut()
        .push(ff("filtered_inst"));
    cfg.add_edge(entry, try_body, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, after, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, fault, EdgeKind::ExceptionUnwind);
    cfg.add_edge(try_body, filtered_handler, EdgeKind::ExceptionHandler);

    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [try_body].into_iter().collect::<BTreeSet<_>>(),
        handlers: vec![
            Handler {
                entry: fault,
                body: HandlerBody::known([fault]),
                kind: HandlerKind::Fault,
            },
            Handler {
                entry: filtered_handler,
                body: HandlerBody::known([filtered_handler]),
                kind: HandlerKind::Filter {
                    filter_block: filter,
                },
            },
        ],
        parent: None,
    });

    let tree = ControlTree::compute(&cfg);
    let kinds: Vec<_> = find_try_handlers(&tree)
        .expect("region structures")
        .iter()
        .map(|handler| handler.kind)
        .collect();
    assert_eq!(
        kinds,
        [
            HandlerKind::Fault,
            HandlerKind::Filter {
                filter_block: filter
            }
        ]
    );
    let pseudo = pseudocode(&cfg, &tree);
    assert!(pseudo.contains("} fault {"), "{pseudo}");
    assert!(
        pseudo.contains(&alloc::format!("}} filter (.bb{}) {{", filter.index())),
        "{pseudo}"
    );
}

#[test]
fn inner_unknown_region_does_not_suppress_an_outer_known_one() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let b0 = cfg.new_block();
    let b1 = cfg.new_block();
    let known_handler = cfg.new_block();
    let unknown_handler = cfg.new_block();
    let after = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(b0).push(ff("b0_inst"));
    cfg.block_mut(b1).push(ff("b1_inst"));
    cfg.block_mut(known_handler).push(ff("known_catch"));
    cfg.block_mut(unknown_handler).push(ff("unknown_catch"));
    cfg.block_mut(after).push(ff("after"));
    cfg.add_edge(cfg.entry(), b0, EdgeKind::Fallthrough);
    cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
    cfg.add_edge(b1, after, EdgeKind::Fallthrough);
    cfg.add_edge(b1, known_handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(known_handler, after, EdgeKind::Fallthrough);
    cfg.add_edge(b0, unknown_handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(unknown_handler, after, EdgeKind::Fallthrough);

    let outer_id = cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [b0, b1].into_iter().collect(),
        handlers: vec![Handler {
            entry: known_handler,
            body: HandlerBody::known([known_handler]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [b0].into_iter().collect(),
        handlers: vec![Handler {
            entry: unknown_handler,
            body: HandlerBody::unknown(),
            kind: HandlerKind::Catch,
        }],
        parent: Some(outer_id),
    });

    let tree = ControlTree::compute(&cfg);
    assert!(
        has_node_kind(&tree, |node| matches!(node, ControlNode::TryCatch { .. })),
        "the complete outer region still structures: {tree:?}"
    );
    for block in [b0, b1, known_handler, unknown_handler, after] {
        assert!(
            contains_block(&tree, block),
            "no block may be dropped: {block:?} in {tree:?}"
        );
    }
}

#[test]
fn conditional_leaving_a_region_records_the_exit() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let t0 = cfg.new_block();
    let t1 = cfg.new_block();
    let exit_true = cfg.new_block();
    let exit_false = cfg.new_block();
    let handler = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(t0).push(ff("t0_inst"));
    cfg.block_mut(t1).push(ff("t1_inst"));
    cfg.block_mut(exit_true).push(ff("early_return"));
    cfg.block_mut(exit_false).push(ff("late_return"));
    cfg.block_mut(handler).push(ff("catch_inst"));
    cfg.add_edge(cfg.entry(), t0, EdgeKind::Fallthrough);
    cfg.add_edge(t0, t1, EdgeKind::Fallthrough);
    cfg.add_edge(t1, exit_true, EdgeKind::ConditionalTrue);
    cfg.add_edge(t1, exit_false, EdgeKind::ConditionalFalse);
    cfg.add_edge(t0, handler, EdgeKind::ExceptionHandler);

    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [t0, t1].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::known([handler]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });

    let tree = ControlTree::compute(&cfg);
    for block in [exit_true, exit_false] {
        assert!(
            contains_block(&tree, block),
            "code on a region-leaving arm may not be dropped: {block:?} in {tree:?}"
        );
    }
}

#[test]
fn switch_case_leaving_a_region_stays_as_a_goto() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let t0 = cfg.new_block();
    let selector = cfg.new_block();
    let case_in = cfg.new_block();
    let case_out = cfg.new_block();
    let handler = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(t0).push(ff("t0_inst"));
    cfg.block_mut(selector).push(ff("selector"));
    cfg.block_mut(case_in).push(ff("case_in_inst"));
    cfg.block_mut(case_out).push(ff("case_out_inst"));
    cfg.block_mut(handler).push(ff("catch_inst"));
    cfg.add_edge(cfg.entry(), t0, EdgeKind::Fallthrough);
    cfg.add_edge(t0, selector, EdgeKind::Fallthrough);
    cfg.add_edge(selector, case_in, EdgeKind::SwitchCase);
    cfg.add_edge(selector, case_out, EdgeKind::SwitchCase);
    cfg.add_edge(t0, handler, EdgeKind::ExceptionHandler);

    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [t0, selector, case_in].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::known([handler]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });

    let tree = ControlTree::compute(&cfg);
    let case_count = tree.preorder().find_map(|id| match *tree.node(id) {
        ControlNode::Switch { cases, .. } => Some(tree.cases(cases).len()),
        _ => None,
    });
    assert_eq!(case_count, Some(2), "both cases survive: {tree:?}");
    assert!(
        has_goto_to(&tree, case_out),
        "the out-of-region case is an explicit jump: {tree:?}"
    );
    assert!(
        contains_block(&tree, case_out),
        "the out-of-region case body may not be dropped: {tree:?}"
    );
}

#[test]
fn handler_tail_beyond_the_declared_extent_is_preserved() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let try_body = cfg.new_block();
    let handler = cfg.new_block();
    let handler_tail = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(try_body).push(ff("try_inst"));
    cfg.block_mut(handler).push(ff("catch_inst"));
    cfg.block_mut(handler_tail).push(ff("ht_handler_tail"));
    cfg.add_edge(cfg.entry(), try_body, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(handler, handler_tail, EdgeKind::Fallthrough);

    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [try_body].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::known([handler]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    let handlers = find_try_handlers(&tree).expect("region structures");
    assert!(
        !body_contains_block(&tree, handlers[0].body, handler_tail),
        "handler lifting stops at the declared extent: {tree:?}"
    );
    // The tail past the extent is the region's unique normal exit, so it
    // becomes the try/catch continuation instead of goto residue.
    assert!(
        contains_block(&tree, handler_tail),
        "the tail beyond the extent may not be dropped: {tree:?}"
    );
    assert!(
        report.gotos.is_empty(),
        "the extent exit converges structurally: {report:?}"
    );
}

#[test]
fn linear_chain_leaving_a_region_records_the_exit() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let t0 = cfg.new_block();
    let t1 = cfg.new_block();
    let continuation = cfg.new_block();
    let handler = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(t0).push(ff("t0_inst"));
    cfg.block_mut(t1).push(ff("t1_inst"));
    cfg.block_mut(continuation).push(ff("post_try"));
    cfg.block_mut(handler).push(ff("catch_inst"));
    cfg.add_edge(cfg.entry(), t0, EdgeKind::Fallthrough);
    cfg.add_edge(t0, t1, EdgeKind::Fallthrough);
    cfg.add_edge(t1, continuation, EdgeKind::Fallthrough);
    cfg.add_edge(t0, handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(handler, continuation, EdgeKind::Fallthrough);

    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [t0, t1].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::known([handler]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });

    let tree = ControlTree::compute(&cfg);
    assert!(
        contains_block(&tree, continuation),
        "the post-try continuation may not be dropped: {tree:?}"
    );
}

#[test]
fn nested_region_respects_the_enclosing_extent() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let try_body = cfg.new_block();
    let h0 = cfg.new_block();
    let h1 = cfg.new_block();
    let escape = cfg.new_block();
    let nested_handler = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(try_body).push(ff("try_inst"));
    cfg.block_mut(h0).push(ff("h0_inst"));
    cfg.block_mut(h1).push(ff("h1_inst"));
    cfg.block_mut(escape).push(ff("x_escape_inst"));
    cfg.block_mut(nested_handler).push(ff("nested_catch"));
    cfg.add_edge(cfg.entry(), try_body, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, h0, EdgeKind::ExceptionHandler);
    cfg.add_edge(h0, h1, EdgeKind::Fallthrough);
    cfg.add_edge(h1, escape, EdgeKind::Fallthrough);
    cfg.add_edge(h1, nested_handler, EdgeKind::ExceptionHandler);

    let outer_id = cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [try_body].into_iter().collect(),
        handlers: vec![Handler {
            entry: h0,
            body: HandlerBody::known([h0, h1]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [h1, escape].into_iter().collect(),
        handlers: vec![Handler {
            entry: nested_handler,
            body: HandlerBody::known([nested_handler]),
            kind: HandlerKind::Catch,
        }],
        parent: Some(outer_id),
    });

    let tree = ControlTree::compute(&cfg);
    let handlers = find_try_handlers(&tree).expect("outer region structures");
    assert!(
        !body_contains_block(&tree, handlers[0].body, escape),
        "a nested region may not escape the enclosing extent: {tree:?}"
    );
    assert!(
        contains_block(&tree, escape),
        "the escaping block may not be dropped: {tree:?}"
    );
}

#[test]
fn region_anchor_follows_flow_order_not_block_ids() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    // Allocate the interior block first so its id is smaller than the
    // region's flow entry.
    let interior = cfg.new_block();
    let flow_entry = cfg.new_block();
    let handler = cfg.new_block();
    let after = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(interior).push(ff("interior_inst"));
    cfg.block_mut(flow_entry).push(ff("entry_inst"));
    cfg.block_mut(handler).push(ff("catch_inst"));
    cfg.block_mut(after).push(ff("after"));
    cfg.add_edge(cfg.entry(), flow_entry, EdgeKind::Fallthrough);
    cfg.add_edge(flow_entry, interior, EdgeKind::Fallthrough);
    cfg.add_edge(interior, after, EdgeKind::Fallthrough);
    cfg.add_edge(flow_entry, handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(handler, after, EdgeKind::Fallthrough);

    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [interior, flow_entry].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::known([handler]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });

    let tree = ControlTree::compute(&cfg);
    assert!(
        has_node_kind(&tree, |node| matches!(node, ControlNode::TryCatch { .. })),
        "the region structures from its flow entry: {tree:?}"
    );
    for block in [flow_entry, interior, handler, after] {
        assert!(
            contains_block(&tree, block),
            "no block may be dropped: {block:?} in {tree:?}"
        );
    }
}

#[test]
fn multiple_finally_handlers_concatenate() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let try_body = cfg.new_block();
    let first_finally = cfg.new_block();
    let second_finally = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(try_body).push(ff("try_inst"));
    cfg.block_mut(first_finally).push(ff("first_cleanup"));
    cfg.block_mut(second_finally).push(ff("second_cleanup"));
    cfg.add_edge(cfg.entry(), try_body, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, first_finally, EdgeKind::ExceptionUnwind);
    cfg.add_edge(try_body, second_finally, EdgeKind::ExceptionUnwind);

    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [try_body].into_iter().collect(),
        handlers: vec![
            Handler {
                entry: first_finally,
                body: HandlerBody::known([first_finally]),
                kind: HandlerKind::Finally,
            },
            Handler {
                entry: second_finally,
                body: HandlerBody::known([second_finally]),
                kind: HandlerKind::Finally,
            },
        ],
        parent: None,
    });

    let tree = ControlTree::compute(&cfg);
    let (_, handlers, finally) = find_try_catch(&tree).expect("expected a TryCatch");
    assert!(
        tree.handlers(handlers).is_empty(),
        "finally is not a handler arm"
    );
    assert!(
        body_contains_block(&tree, finally, first_finally),
        "the first cleanup may not be discarded: {tree:?}"
    );
    assert!(
        body_contains_block(&tree, finally, second_finally),
        "the second cleanup follows in declaration order: {tree:?}"
    );
    let order: Vec<_> = body_subtree(&tree, finally)
        .into_iter()
        .filter_map(|id| leaf_block(tree.node(id)))
        .collect();
    assert_eq!(order, [first_finally, second_finally]);
}

#[test]
fn filter_funclet_code_is_emitted() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let try_body = cfg.new_block();
    let handler = cfg.new_block();
    // The funclet is invoked by the runtime: no incoming CFG edge at all.
    let filter_block = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(try_body).push(ff("try_inst"));
    cfg.block_mut(handler).push(ff("except_inst"));
    cfg.block_mut(filter_block).push(ff("filter_predicate"));
    cfg.add_edge(cfg.entry(), try_body, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, handler, EdgeKind::ExceptionHandler);

    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [try_body].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::known([handler]),
            kind: HandlerKind::Filter { filter_block },
        }],
        parent: None,
    });

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert!(
        contains_block(&tree, filter_block),
        "the filter funclet's predicate code may not be dropped: {tree:?}"
    );
    assert_eq!(report.swept_blocks, [filter_block]);
}

#[test]
fn a_branch_inside_a_filter_funclet_merges_like_reachable_code() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let try_body = cfg.new_block();
    let handler = cfg.new_block();
    // The funclet has no incoming CFG edge, so the entry cannot reach it, yet
    // its diamond still has a merge point.
    let filter = cfg.new_block();
    let arm_a = cfg.new_block();
    let arm_b = cfg.new_block();
    let join = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(try_body).push(ff("try_inst"));
    cfg.block_mut(handler).push(ff("except_inst"));
    cfg.block_mut(filter).push(ff("filter_test"));
    cfg.block_mut(arm_a).push(ff("arm_a"));
    cfg.block_mut(arm_b).push(ff("arm_b"));
    cfg.block_mut(join).push(ff("join"));
    cfg.add_edge(cfg.entry(), try_body, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(filter, arm_a, EdgeKind::ConditionalTrue);
    cfg.add_edge(filter, arm_b, EdgeKind::ConditionalFalse);
    cfg.add_edge(arm_a, join, EdgeKind::Fallthrough);
    cfg.add_edge(arm_b, join, EdgeKind::Fallthrough);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [try_body].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::known([handler]),
            kind: HandlerKind::Filter {
                filter_block: filter,
            },
        }],
        parent: None,
    });

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert_eq!(report.swept_blocks, [filter]);
    assert!(
        report.gotos.is_empty(),
        "the funclet's branch merges at its join: {report:?}"
    );
    let text = pseudocode(&cfg, &tree);
    assert!(
        text.contains(
            ".bb3:\n    if {\n        arm_a\n    } else {\n        arm_b\n    }\n    join\n"
        ),
        "{text}"
    );
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "HandlerBody::Known must contain its own handler entry")]
fn a_known_body_omitting_its_entry_is_a_frontend_bug() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let try_body = cfg.new_block();
    let handler = cfg.new_block();
    let elsewhere = cfg.new_block();
    cfg.block_mut(try_body).push(ff("try_inst"));
    cfg.add_edge(cfg.entry(), try_body, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, handler, EdgeKind::ExceptionHandler);

    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [try_body].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::known([elsewhere]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });

    let _ = ControlTree::compute(&cfg);
}

#[test]
fn try_anchored_at_a_conditional_keeps_the_branch_structure() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let branch = cfg.new_block();
    let then_arm = cfg.new_block();
    let else_arm = cfg.new_block();
    let merge = cfg.new_block();
    let handler = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(branch).push(ff("branch_inst"));
    cfg.block_mut(then_arm).push(ff("then_inst"));
    cfg.block_mut(else_arm).push(ff("else_inst"));
    cfg.block_mut(merge).push(ff("merge_inst"));
    cfg.block_mut(handler).push(ff("catch_inst"));
    cfg.add_edge(cfg.entry(), branch, EdgeKind::Fallthrough);
    cfg.add_edge(branch, then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(branch, else_arm, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, merge, EdgeKind::Fallthrough);
    cfg.add_edge(else_arm, merge, EdgeKind::Fallthrough);
    cfg.add_edge(branch, handler, EdgeKind::ExceptionHandler);

    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [branch, then_arm, else_arm, merge].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::known([handler]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });

    let tree = ControlTree::compute(&cfg);
    let (try_body, _, _) = find_try_catch(&tree).expect("expected a TryCatch");
    let inside = body_subtree(&tree, try_body);
    assert!(
        inside
            .iter()
            .any(|&id| matches!(tree.node(id), ControlNode::IfThenElse { .. })),
        "a branch anchoring a region keeps its structure inside the try: {tree:?}"
    );
    for block in [then_arm, else_arm, merge] {
        assert!(
            body_contains_block(&tree, try_body, block),
            "the try body carries the whole branch: {block:?} in {tree:?}"
        );
    }
}

#[test]
fn try_anchored_at_a_switch_selector_keeps_the_dispatch() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let selector = cfg.new_block();
    let case_a = cfg.new_block();
    let case_b = cfg.new_block();
    let handler = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(selector).push(ff("selector"));
    cfg.block_mut(case_a).push(ff("case_a_inst"));
    cfg.block_mut(case_b).push(ff("case_b_inst"));
    cfg.block_mut(handler).push(ff("catch_inst"));
    cfg.add_edge(cfg.entry(), selector, EdgeKind::Fallthrough);
    cfg.add_edge(selector, case_a, EdgeKind::SwitchCase);
    cfg.add_edge(selector, case_b, EdgeKind::SwitchCase);
    cfg.add_edge(selector, handler, EdgeKind::ExceptionHandler);

    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [selector, case_a, case_b].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::known([handler]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });

    let tree = ControlTree::compute(&cfg);
    let (try_body, _, _) = find_try_catch(&tree).expect("expected a TryCatch");
    let inside = body_subtree(&tree, try_body);
    assert!(
        inside
            .iter()
            .any(|&id| matches!(tree.node(id), ControlNode::Switch { .. })),
        "a dispatch anchoring a region keeps its structure inside the try: {tree:?}"
    );
    let mut case_ids = Vec::new();
    for id in inside {
        if let ControlNode::Switch { cases, .. } = *tree.node(id) {
            case_ids.extend(tree.cases(cases).iter().map(|case| case.block));
        }
    }
    assert_eq!(
        case_ids,
        [case_a, case_b],
        "the try body carries every case: {tree:?}"
    );
}

#[test]
fn try_anchored_at_a_loop_header_keeps_the_loop() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let header = cfg.new_block();
    let latch = cfg.new_block();
    let handler = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(header).push(ff("header_inst"));
    cfg.block_mut(latch).push(ff("latch_inst"));
    cfg.block_mut(handler).push(ff("catch_inst"));
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, latch, EdgeKind::Fallthrough);
    cfg.add_edge(latch, header, EdgeKind::Back);
    cfg.add_edge(header, handler, EdgeKind::ExceptionHandler);

    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [header, latch].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::known([handler]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });

    let tree = ControlTree::compute(&cfg);
    let (try_body, _, _) = find_try_catch(&tree).expect("expected a TryCatch");
    let inside = body_subtree(&tree, try_body);
    assert!(
        inside
            .iter()
            .any(|&id| matches!(tree.node(id), ControlNode::Loop { .. })),
        "a loop header anchoring a region keeps its structure inside the try: {tree:?}"
    );
    assert!(
        body_contains_block(&tree, try_body, latch),
        "the loop body stays inside the try: {tree:?}"
    );
}

#[test]
fn a_try_anchored_at_a_switch_case_is_structured_inside_that_case() {
    // b0 dispatches to the cases p and q, which rejoin at m. p is protected
    // and its pad h rejoins at m too: the case arm of p is the try.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let p = cfg.new_block();
    let q = cfg.new_block();
    let handler = cfg.new_block();
    let merge = cfg.new_block();
    for (block, name) in [
        (cfg.entry(), "dispatch"),
        (p, "protected"),
        (q, "other"),
        (handler, "caught"),
        (merge, "after"),
    ] {
        cfg.block_mut(block).push(ff(name));
    }
    cfg.add_edge(cfg.entry(), p, EdgeKind::SwitchCase);
    cfg.add_edge(cfg.entry(), q, EdgeKind::SwitchCase);
    cfg.add_edge(p, merge, EdgeKind::Fallthrough);
    cfg.add_edge(q, merge, EdgeKind::Fallthrough);
    cfg.add_edge(p, handler, EdgeKind::ExceptionHandler);
    cfg.add_edge(handler, merge, EdgeKind::Fallthrough);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [p].into_iter().collect(),
        handlers: vec![Handler {
            entry: handler,
            body: HandlerBody::known([handler]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });

    let (tree, report) = ControlTree::compute_with_report(&cfg);
    assert_eq!(
        report.unstructured_regions,
        [],
        "the region has a known extent: {}",
        pseudocode(&cfg, &tree)
    );
    let (try_body, _, _) = find_try_catch(&tree).expect("the case of p is a try");
    assert!(body_contains_block(&tree, try_body, p));
}
