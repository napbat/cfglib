//! `try / catch / finally` through exception regions: how a region becomes a
//! `TryCatch` node, what owns the handler code, and what stays ordinary flow.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, ControlNode, EdgeKind, GotoReason, HandlerKind, HandlerRef, promote_handler_extents,
};

use crate::builders::{Flow, handler, handler_over, pseudocode, try_catch_finally};
use crate::coverage::{blocks_under, reachable_blocks};
use crate::golden::assert_golden;
use crate::oracles::{
    structure_and_check, structure_and_check_covering, structure_and_check_shape_only,
};
use crate::reading::{body_nodes, leaf_blocks, top_level};
use crate::tree_shape::{assert_fully_structured, nodes_where};

/// The first `TryCatch` node of a tree, in preorder.
fn first_try(tree: &cfglib::ControlTree) -> ControlNode {
    let tries = nodes_where(tree, |node| matches!(node, ControlNode::TryCatch { .. }));
    *tree.node(*tries.first().expect("the tree holds a try/catch"))
}

/// `setup(); try { risky() } catch { log(e) }; return`
fn try_catch() -> (Flow, [BlockId; 4], cfglib::RegionId) {
    let mut flow = Flow::starting_with("setup()");
    let body = flow.stmt("risky()");
    let pad = flow.stmt("log(e)");
    let after = flow.stmt("return");
    flow.fall(flow.entry(), body);
    flow.fall(body, after);
    flow.edge(body, pad, EdgeKind::ExceptionHandler);
    flow.fall(pad, after);
    let region = flow.region(&[body], vec![handler(pad, HandlerKind::Catch)]);
    let entry = flow.entry();
    (flow, [entry, body, pad, after], region)
}

#[test]
fn region_with_a_catch_handler_becomes_a_try_catch_node() {
    let (flow, [entry, body, pad, after], region) = try_catch();

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);
    assert!(
        report.swept_blocks.is_empty(),
        "the pad is lifted inside its handler"
    );

    let [
        ControlNode::Block(setup),
        ControlNode::TryCatch {
            try_body,
            handlers,
            finally_body,
        },
        ControlNode::Return(tail),
    ] = top_level(&tree)[..]
    else {
        panic!("expected setup, try/catch, tail: {tree:?}");
    };
    assert_eq!(setup.block(), entry);
    assert_eq!(leaf_blocks(&tree, try_body), [body]);
    assert!(finally_body.is_empty());
    let [arm] = tree.handlers(handlers) else {
        panic!("one catch arm");
    };
    assert_eq!(arm.handler, HandlerRef::new(region, 0));
    assert_eq!(arm.entry, pad);
    assert_eq!(arm.kind, HandlerKind::Catch);
    assert_eq!(leaf_blocks(&tree, arm.body), [pad]);
    assert_eq!(tail.block(), after);
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "setup()\ntry {\n    risky()\n} catch {\n    log(e)\n}\nreturn\n"
    );
}

#[test]
fn catch_and_finally_share_one_region_and_the_cleanup_runs_after_both() {
    // Both handlers are entered through exception edges only, and every
    // path ends at `after`.
    let flow = try_catch_finally();
    let region = flow.cfg.regions()[0].id;
    let body = *flow.cfg.regions()[0]
        .protected_blocks
        .first()
        .expect("the region protects the body");
    let cleanup = flow.cfg.regions()[0].handlers[1].entry;

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let ControlNode::TryCatch {
        try_body,
        handlers,
        finally_body,
    } = first_try(&tree)
    else {
        unreachable!("first_try returns a try/catch");
    };
    assert_eq!(leaf_blocks(&tree, try_body), [body]);
    let arms = tree.handlers(handlers);
    assert_eq!(arms.len(), 1, "a finally handler has no arm of its own");
    assert_eq!(arms[0].handler, HandlerRef::new(region, 0));
    assert_eq!(leaf_blocks(&tree, finally_body), [cleanup]);
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "open()\ntry {\n    work()\n} catch {\n    recover()\n} finally {\n    close()\n}\nreturn\n"
    );
}

#[test]
fn handler_kinds_keep_their_declaration_order_and_their_text() {
    let mut flow = Flow::starting_with("risky()");
    let after = flow.stmt("return");
    let catch = flow.stmt("on_error(e)");
    let filtered = flow.stmt("on_filtered(e)");
    let filter = flow.stmt("is_recoverable(e)");
    let catch_all = flow.stmt("on_any()");
    let fault = flow.stmt("rollback()");
    let entry = flow.entry();
    flow.fall(entry, after);
    for pad in [catch, filtered, catch_all, fault] {
        flow.edge(entry, pad, EdgeKind::ExceptionHandler);
        flow.fall(pad, after);
    }
    let region = flow.region(
        &[entry],
        vec![
            handler(catch, HandlerKind::Catch),
            handler(
                filtered,
                HandlerKind::Filter {
                    filter_block: filter,
                },
            ),
            handler(catch_all, HandlerKind::CatchAll),
            handler(fault, HandlerKind::Fault),
        ],
    );

    // The filter funclet is code only its handler owns, so the entry does not
    // reach it; it is covered all the same.
    let mut covered = reachable_blocks(&flow.cfg);
    covered.insert(filter);
    let (tree, report) = structure_and_check_covering(&flow.cfg, &covered);

    let ControlNode::TryCatch { handlers, .. } = first_try(&tree) else {
        unreachable!("first_try returns a try/catch");
    };
    let arms = tree.handlers(handlers);
    let kinds: Vec<HandlerKind> = arms.iter().map(|arm| arm.kind).collect();
    assert_eq!(
        kinds,
        [
            HandlerKind::Catch,
            HandlerKind::Filter {
                filter_block: filter
            },
            HandlerKind::CatchAll,
            HandlerKind::Fault,
        ]
    );
    let refs: Vec<HandlerRef> = arms.iter().map(|arm| arm.handler).collect();
    assert_eq!(
        refs,
        [0, 1, 2, 3].map(|index| HandlerRef::new(region, index))
    );
    assert_eq!(
        report.swept_blocks,
        [filter],
        "the funclet is swept: {report:?}"
    );
    assert_golden(
        "control-tree/handler-kinds.pseudo",
        &pseudocode(&flow.cfg, &tree),
    );
}

#[test]
fn several_finally_handlers_concatenate_in_declaration_order() {
    let mut flow = Flow::starting_with("setup()");
    let body = flow.stmt("work()");
    let first = flow.stmt("release_a()");
    let second = flow.stmt("release_b()");
    let after = flow.stmt("return");
    flow.fall(flow.entry(), body);
    flow.fall(body, after);
    flow.edge(body, first, EdgeKind::ExceptionUnwind);
    flow.edge(body, second, EdgeKind::ExceptionUnwind);
    flow.fall(first, after);
    flow.fall(second, after);
    flow.region(
        &[body],
        vec![
            handler(first, HandlerKind::Finally),
            handler(second, HandlerKind::Finally),
        ],
    );

    let (tree, report) = structure_and_check_shape_only(&flow.cfg, &reachable_blocks(&flow.cfg));
    assert_fully_structured(&tree, &report);

    let ControlNode::TryCatch {
        handlers,
        finally_body,
        ..
    } = first_try(&tree)
    else {
        unreachable!("first_try returns a try/catch");
    };
    assert_eq!(tree.handlers(handlers), []);
    assert_eq!(leaf_blocks(&tree, finally_body), [first, second]);
}

#[test]
fn nested_regions_nest_their_try_catch_nodes_and_keep_exception_edges_inside() {
    let mut flow = Flow::starting_with("begin()");
    let outer_body = flow.stmt("outer_work()");
    let inner_body = flow.stmt("inner_work()");
    let inner_pad = flow.stmt("inner_catch()");
    let rest = flow.stmt("after_inner()");
    let outer_pad = flow.stmt("outer_catch()");
    let end = flow.stmt("return");
    flow.fall(flow.entry(), outer_body);
    flow.fall(outer_body, inner_body);
    flow.fall(inner_body, rest);
    flow.fall(inner_pad, rest);
    flow.fall(rest, end);
    flow.fall(outer_pad, end);
    flow.edge(inner_body, inner_pad, EdgeKind::ExceptionHandler);
    flow.edge(outer_body, outer_pad, EdgeKind::ExceptionHandler);
    flow.edge(rest, outer_pad, EdgeKind::ExceptionHandler);
    let outer = flow.region(
        &[outer_body, inner_body, inner_pad, rest],
        vec![handler(outer_pad, HandlerKind::Catch)],
    );
    let inner = flow.region(&[inner_body], vec![handler(inner_pad, HandlerKind::Catch)]);
    flow.cfg.region_mut(inner).expect("just added").parent = Some(outer);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let tries = nodes_where(&tree, |node| matches!(node, ControlNode::TryCatch { .. }));
    assert_eq!(tries.len(), 2);
    let ControlNode::TryCatch {
        try_body: outer_try,
        handlers: outer_handlers,
        ..
    } = *tree.node(tries[0])
    else {
        unreachable!("filtered to try/catch");
    };
    assert_eq!(tree.handlers(outer_handlers)[0].entry, outer_pad);
    assert_eq!(
        body_nodes(&tree, outer_try)
            .iter()
            .map(crate::reading::kind_name)
            .collect::<Vec<_>>(),
        ["block", "try", "block"],
        "outer_work, the inner try, after_inner"
    );
    let ControlNode::TryCatch {
        try_body: inner_try,
        handlers: inner_handlers,
        ..
    } = *tree.node(tries[1])
    else {
        unreachable!("filtered to try/catch");
    };
    assert_eq!(leaf_blocks(&tree, inner_try), [inner_body]);
    assert_eq!(tree.handlers(inner_handlers)[0].entry, inner_pad);
    assert!(
        blocks_under(&tree, outer_try).contains(&inner_pad),
        "the inner catch body is inside the outer try"
    );
    assert_golden(
        "control-tree/nested-try.pseudo",
        &pseudocode(&flow.cfg, &tree),
    );
}

#[test]
fn protected_block_that_returns_is_a_block_leaf_since_its_exception_edge_is_a_successor() {
    // `Return` is a block without successors; this block returns yet keeps its
    // exception edge to the pad, so the tree holds it as a plain `Block` leaf.
    let mut flow = Flow::starting_with("open()");
    let body = flow.stmt("return read()");
    let pad = flow.stmt("log(e)");
    let after = flow.stmt("return 0");
    flow.fall(flow.entry(), body);
    flow.edge(body, pad, EdgeKind::ExceptionHandler);
    flow.fall(pad, after);
    flow.region(&[body], vec![handler(pad, HandlerKind::Catch)]);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let ControlNode::TryCatch {
        try_body, handlers, ..
    } = first_try(&tree)
    else {
        unreachable!("first_try returns a try/catch");
    };
    assert!(matches!(
        body_nodes(&tree, try_body)[..],
        [ControlNode::Block(span)] if span.block() == body
    ));
    let [arm] = tree.handlers(handlers) else {
        panic!("one catch arm");
    };
    assert_eq!(blocks_under(&tree, arm.body), BTreeSet::from([pad]));
}

#[test]
fn catch_arm_that_returns_ends_in_a_return_leaf() {
    let mut flow = Flow::starting_with("risky()");
    let pad = flow.stmt("return -1");
    let after = flow.stmt("return 0");
    let entry = flow.entry();
    flow.fall(entry, after);
    flow.edge(entry, pad, EdgeKind::ExceptionHandler);
    flow.region(&[entry], vec![handler(pad, HandlerKind::Catch)]);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let ControlNode::TryCatch { handlers, .. } = first_try(&tree) else {
        unreachable!("first_try returns a try/catch");
    };
    let [arm] = tree.handlers(handlers) else {
        panic!("one catch arm");
    };
    assert!(matches!(
        body_nodes(&tree, arm.body)[..],
        [ControlNode::Return(span)] if span.block() == pad
    ));
}

#[test]
fn shared_cleanup_entered_by_ordinary_flow_leaves_the_handler_extent_and_is_reported() {
    // The lowering symtree emits for try/catch/finally: one cleanup block that
    // the protected body and the catch pad both enter by ordinary flow. The
    // structurer documents that a region is well formed only when ordinary flow
    // enters a handler from its own extent, so each of those transfers is a
    // goto out of the structural bound, and the report says so.
    let mut flow = Flow::starting_with("open()");
    let body = flow.stmt("work()");
    let pad = flow.stmt("recover()");
    let cleanup = flow.stmt("close()");
    let after = flow.stmt("return");
    flow.fall(flow.entry(), body);
    flow.fall(body, cleanup);
    flow.edge(body, pad, EdgeKind::ExceptionHandler);
    flow.edge(body, cleanup, EdgeKind::ExceptionUnwind);
    flow.fall(pad, cleanup);
    flow.fall(cleanup, after);
    flow.region(
        &[body],
        vec![
            handler(pad, HandlerKind::Catch),
            handler(cleanup, HandlerKind::Finally),
        ],
    );

    // The partition and the implied-flow oracles still hold: the tree stays
    // faithful to the graph, it only cannot express the flow structurally.
    let (tree, report) = structure_and_check(&flow.cfg);

    assert!(!report.is_fully_structured());
    assert_eq!(
        report
            .gotos
            .iter()
            .map(|goto| goto.reason)
            .collect::<Vec<_>>(),
        [GotoReason::BoundaryEscape, GotoReason::BoundaryEscape]
    );
    assert!(report.gotos.iter().all(|goto| goto.target == cleanup));
    assert_eq!(report.unresolved_labels, []);
    assert_eq!(
        nodes_where(&tree, |node| matches!(node, ControlNode::TryCatch { .. })).len(),
        1
    );
}

#[test]
fn try_inside_a_loop_keeps_the_loop_around_the_try_catch() {
    // while i < n { try { risky(i) } catch { skip(i) }; i += 1 }
    let mut flow = Flow::starting_with("i = 0");
    let header = flow.stmt("i < n");
    let body = flow.stmt("risky(i)");
    let pad = flow.stmt("skip(i)");
    let step = flow.stmt("i += 1");
    let exit = flow.stmt("return");
    flow.fall(flow.entry(), header);
    flow.branch(header, body, exit);
    flow.fall(body, step);
    flow.edge(body, pad, EdgeKind::ExceptionHandler);
    flow.fall(pad, step);
    flow.jump(step, header);
    flow.region(&[body], vec![handler(pad, HandlerKind::Catch)]);

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "i = 0\nwhile {\n    i < n\n    break if !cond;\n    try {\n        risky(i)\n    } catch {\n        skip(i)\n    }\n    i += 1\n}\nreturn\n"
    );
}

#[test]
fn catch_pad_that_no_exception_edge_enters_is_still_emitted_in_its_handler() {
    // No call unwinds into the pad, yet its body is the source of a catch
    // clause: the tree holds it exactly once.
    let (mut flow, [_, body, pad, _], region) = try_catch();
    let doomed: Vec<_> = flow
        .cfg
        .edge_ids()
        .filter(|&edge| flow.cfg.edge(edge).kind() == EdgeKind::ExceptionHandler)
        .collect();
    for edge in doomed {
        flow.cfg.remove_edge(edge);
    }
    assert!(!reachable_blocks(&flow.cfg).contains(&pad));
    let mut covered = reachable_blocks(&flow.cfg);
    covered.insert(pad);

    let (tree, report) = structure_and_check_covering(&flow.cfg, &covered);

    assert_fully_structured(&tree, &report);
    let ControlNode::TryCatch {
        try_body, handlers, ..
    } = first_try(&tree)
    else {
        unreachable!("first_try returns a try/catch");
    };
    assert_eq!(leaf_blocks(&tree, try_body), [body]);
    let [arm] = tree.handlers(handlers) else {
        panic!("one catch arm");
    };
    assert_eq!(arm.handler, HandlerRef::new(region, 0));
    assert_eq!(leaf_blocks(&tree, arm.body), [pad]);
}

#[test]
fn protected_block_the_entry_cannot_reach_is_dropped_as_dead_code() {
    let mut flow = Flow::starting_with("return 0");
    let dead = flow.stmt("never()");
    let pad = flow.stmt("never_caught()");
    let after = flow.stmt("return 1");
    flow.fall(dead, after);
    flow.edge(dead, pad, EdgeKind::ExceptionHandler);
    flow.fall(pad, after);
    flow.region(&[dead], vec![handler(pad, HandlerKind::Catch)]);

    let (tree, _) = cfglib::ControlTree::compute_with_report(&flow.cfg);

    let text = pseudocode(&flow.cfg, &tree);
    assert!(!text.contains("never()"), "{text}");
    assert!(text.contains("return 0"), "{text}");
    assert!(
        text.contains("never_caught()"),
        "a handler's own code is never dropped: {text}"
    );
}

#[test]
fn unknown_handler_extent_stays_ordinary_flow_and_is_reported() {
    let (mut flow, [_, body, pad, after], region) = try_catch();
    flow.cfg
        .handler_mut(HandlerRef::new(region, 0))
        .expect("handler")
        .body = cfglib::HandlerBody::unknown();

    let (tree, report) = structure_and_check(&flow.cfg);

    assert!(!report.is_fully_structured());
    assert_eq!(report.unstructured_regions, [region]);
    assert!(
        nodes_where(&tree, |node| matches!(node, ControlNode::TryCatch { .. })).is_empty(),
        "no handler extent is guessed"
    );
    // Every block is still there, as ordinary flow.
    let text = pseudocode(&flow.cfg, &tree);
    for block in [body, pad, after] {
        assert!(
            text.contains(flow.cfg.block(block).instructions()[0]),
            "{text}"
        );
    }
}

#[test]
fn promoting_unknown_extents_on_a_derived_graph_lets_the_structurer_emit_the_try() {
    let (mut flow, _, region) = try_catch();
    flow.cfg
        .handler_mut(HandlerRef::new(region, 0))
        .expect("handler")
        .body = cfglib::HandlerBody::unknown();
    let (_, unstructured) = cfglib::ControlTree::compute_with_report(&flow.cfg);
    assert_eq!(unstructured.unstructured_regions, [region]);

    // The canonical graph keeps saying "unknown"; a clone lifted for
    // presentation gets the sound dominated extent.
    let mut derived = flow.cfg.clone();
    assert_eq!(promote_handler_extents(&mut derived), 1);
    let (tree, report) = structure_and_check(&derived);

    assert_fully_structured(&tree, &report);
    assert_eq!(
        nodes_where(&tree, |node| matches!(node, ControlNode::TryCatch { .. })).len(),
        1
    );
    assert!(
        !flow.cfg.regions()[region.index()].handlers[0]
            .body
            .is_known()
    );
}

#[test]
fn two_handlers_with_disjoint_extents_each_own_their_blocks() {
    // catch a { a(); a_more() } catch b { b() }
    let mut flow = Flow::starting_with("risky()");
    let after = flow.stmt("return");
    let a_pad = flow.stmt("a()");
    let a_more = flow.stmt("a_more()");
    let b_pad = flow.stmt("b()");
    let entry = flow.entry();
    flow.fall(entry, after);
    flow.edge(entry, a_pad, EdgeKind::ExceptionHandler);
    flow.edge(entry, b_pad, EdgeKind::ExceptionHandler);
    flow.fall(a_pad, a_more);
    flow.fall(a_more, after);
    flow.fall(b_pad, after);
    flow.region(
        &[entry],
        vec![
            handler_over(a_pad, HandlerKind::Catch, &[a_pad, a_more]),
            handler(b_pad, HandlerKind::Catch),
        ],
    );

    let (tree, report) = structure_and_check(&flow.cfg);
    assert_fully_structured(&tree, &report);

    let ControlNode::TryCatch { handlers, .. } = first_try(&tree) else {
        unreachable!("first_try returns a try/catch");
    };
    let [first, second] = tree.handlers(handlers) else {
        panic!("two catch arms");
    };
    assert_eq!(leaf_blocks(&tree, first.body), [a_pad, a_more]);
    assert_eq!(leaf_blocks(&tree, second.body), [b_pad]);
}
