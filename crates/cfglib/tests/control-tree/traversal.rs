//! Reading a tree: preorder, the visitor walk, child bodies, spans, and the
//! program-order instruction walk, each checked against a plain recursion over
//! the tree or a hand-derived order.

use std::collections::BTreeSet;

use cfglib::{
    Cfg, ControlDescent, ControlNode, ControlNodeId, ControlTree, ControlVisitor, EdgeKind,
};

use crate::builders::{Flow, dispatch, irreducible_pair, labeled_break_loops, try_catch_finally};
use crate::coverage::leaf_spans;
use crate::oracles::structure_and_check;
use crate::tree_shape::{
    EventLog, nodes_where, reference_depth, reference_postorder, reference_preorder,
};

/// Trees that between them hold every node kind a plain `compute` produces.
fn sample_trees() -> Vec<(&'static str, Cfg<&'static str>, ControlTree)> {
    let mut graphs = vec![
        ("labeled loops", labeled_break_loops().0),
        ("try catch finally", try_catch_finally()),
        ("dispatch", dispatch()),
        ("irreducible", irreducible_pair().0),
    ];
    let mut spin = Flow::starting_with("poll()");
    let done = spin.stmt("return");
    spin.edge(spin.entry(), spin.entry(), EdgeKind::ConditionalTrue);
    spin.edge(spin.entry(), done, EdgeKind::ConditionalFalse);
    graphs.push(("spin", spin));
    graphs
        .into_iter()
        .map(|(name, flow)| {
            let (tree, _) = structure_and_check(&flow.cfg);
            (name, flow.cfg, tree)
        })
        .collect()
}

#[test]
fn the_samples_cover_every_node_kind_a_plain_tree_has() {
    let mut kinds = BTreeSet::new();
    for (_, _, tree) in sample_trees() {
        kinds.extend(
            tree.preorder()
                .map(|id| crate::reading::kind_name(tree.node(id))),
        );
    }
    for kind in [
        "block", "sequence", "if", "loop", "switch", "break", "continue", "return", "label",
        "goto", "try",
    ] {
        assert!(kinds.contains(kind), "no sample holds a {kind}: {kinds:?}");
    }
}

#[test]
fn preorder_matches_a_plain_recursion_over_the_child_bodies() {
    for (name, _, tree) in sample_trees() {
        assert_eq!(
            tree.preorder().collect::<Vec<_>>(),
            reference_preorder(&tree),
            "{name}"
        );
    }
}

#[test]
fn a_cloned_preorder_resumes_where_the_original_was() {
    let (_, _, tree) = sample_trees().remove(0);
    let mut walk = tree.preorder();
    let first_three: Vec<_> = walk.by_ref().take(3).collect();

    let fork = walk.clone();
    let rest: Vec<_> = walk.collect();

    assert_eq!(fork.collect::<Vec<_>>(), rest);
    let all: Vec<_> = first_three.into_iter().chain(rest).collect();
    assert_eq!(all, reference_preorder(&tree));
}

#[test]
fn walk_enters_in_preorder_and_leaves_in_postorder_with_every_pair_nested() {
    for (name, _, tree) in sample_trees() {
        let mut log = EventLog::default();
        tree.walk(&mut log);

        let entered: Vec<ControlNodeId> = log
            .events
            .iter()
            .filter(|(enter, _)| *enter)
            .map(|&(_, id)| id)
            .collect();
        let left: Vec<ControlNodeId> = log
            .events
            .iter()
            .filter(|(enter, _)| !*enter)
            .map(|&(_, id)| id)
            .collect();
        assert_eq!(entered, reference_preorder(&tree), "{name}");
        assert_eq!(left, reference_postorder(&tree), "{name}");

        let mut open = Vec::new();
        for &(enter, id) in &log.events {
            if enter {
                open.push(id);
            } else {
                assert_eq!(open.pop(), Some(id), "{name}: leave pairs with its enter");
            }
        }
        assert_eq!(open, []);
    }
}

#[test]
fn skipping_a_node_hides_its_children_but_still_leaves_it() {
    let (flow, _) = labeled_break_loops();
    let (tree, _) = structure_and_check(&flow.cfg);
    let loops = nodes_where(&tree, |node| matches!(node, ControlNode::Loop { .. }));
    let outer_loop = loops[0];

    let mut log = EventLog::default();
    log.skip.insert(outer_loop);
    tree.walk(&mut log);

    let descendants: BTreeSet<ControlNodeId> = tree
        .node_ids()
        .filter(|&id| is_descendant(&tree, outer_loop, id))
        .collect();
    assert!(!descendants.is_empty(), "the loop has children to hide");
    for id in &descendants {
        assert!(
            !log.events.iter().any(|&(_, seen)| seen == *id),
            "{id:?} is under the skipped loop"
        );
    }
    assert!(log.events.contains(&(true, outer_loop)));
    assert!(
        log.events.contains(&(false, outer_loop)),
        "leave still follows enter"
    );
    let visited: BTreeSet<ControlNodeId> = log.events.iter().map(|&(_, id)| id).collect();
    let everyone: BTreeSet<ControlNodeId> = tree.node_ids().collect();
    assert_eq!(
        visited,
        everyone.difference(&descendants).copied().collect()
    );
}

/// Whether `id` sits anywhere under `ancestor`, by plain recursion.
fn is_descendant(tree: &ControlTree, ancestor: ControlNodeId, id: ControlNodeId) -> bool {
    tree.child_bodies(ancestor)
        .flat_map(|body| tree.body(body).to_vec())
        .any(|child| child == id || is_descendant(tree, child, id))
}

#[test]
fn a_visitor_that_tracks_depth_agrees_with_a_recursive_depth() {
    #[derive(Default)]
    struct Depth {
        current: usize,
        deepest: usize,
    }
    impl ControlVisitor for Depth {
        fn enter(&mut self, _tree: &ControlTree, _node: ControlNodeId) -> ControlDescent {
            self.current += 1;
            self.deepest = self.deepest.max(self.current);
            ControlDescent::Into
        }

        fn leave(&mut self, _tree: &ControlTree, _node: ControlNodeId) {
            self.current -= 1;
        }
    }

    for (name, _, tree) in sample_trees() {
        let mut depth = Depth::default();
        tree.walk(&mut depth);
        assert_eq!(depth.current, 0, "{name}");
        assert_eq!(depth.deepest, reference_depth(&tree, tree.root()), "{name}");
    }
}

#[test]
fn child_bodies_follow_the_documented_order_for_every_node_kind() {
    for (name, _, tree) in sample_trees() {
        for id in tree.node_ids() {
            let expected = match *tree.node(id) {
                ControlNode::Block(_)
                | ControlNode::Return(_)
                | ControlNode::Break { .. }
                | ControlNode::Continue { .. }
                | ControlNode::Goto { .. } => vec![],
                ControlNode::Sequence(body)
                | ControlNode::Loop { body, .. }
                | ControlNode::Label { body, .. }
                | ControlNode::Guarded { body, .. } => vec![body],
                ControlNode::IfThenElse {
                    then_body,
                    else_body,
                    ..
                } => vec![then_body, else_body],
                ControlNode::Switch {
                    cases,
                    default_body,
                    ..
                } => tree
                    .cases(cases)
                    .iter()
                    .map(|case| case.body)
                    .chain([default_body])
                    .collect(),
                ControlNode::TryCatch {
                    try_body,
                    handlers,
                    finally_body,
                } => [try_body]
                    .into_iter()
                    .chain(tree.handlers(handlers).iter().map(|handler| handler.body))
                    .chain([finally_body])
                    .collect(),
            };
            let bodies = tree.child_bodies(id);
            assert_eq!(bodies.len(), expected.len(), "{name}: {:?}", tree.node(id));
            assert_eq!(bodies.size_hint(), (expected.len(), Some(expected.len())));
            assert_eq!(bodies.clone().collect::<Vec<_>>(), expected, "{name}");
            let mut backwards = expected.clone();
            backwards.reverse();
            assert_eq!(bodies.rev().collect::<Vec<_>>(), backwards, "{name}");
        }
    }
}

#[test]
fn child_bodies_can_be_consumed_from_both_ends() {
    let flow = dispatch();
    let (tree, _) = structure_and_check(&flow.cfg);
    let switch = nodes_where(&tree, |node| matches!(node, ControlNode::Switch { .. }))[0];
    let all: Vec<_> = tree.child_bodies(switch).collect();
    assert_eq!(all.len(), 3, "two cases and the default");

    let mut bodies = tree.child_bodies(switch);
    assert_eq!(bodies.next(), Some(all[0]));
    assert_eq!(bodies.next_back(), Some(all[2]));
    assert_eq!(bodies.len(), 1);
    assert_eq!(bodies.next(), Some(all[1]));
    assert_eq!(bodies.next_back(), None);
}

#[test]
fn spans_name_whole_blocks_and_read_back_their_instructions() {
    for (name, cfg, tree) in sample_trees() {
        for span in leaf_spans(&tree) {
            let instructions = cfg.block(span.block()).instructions();
            assert_eq!(
                (span.start(), span.end()),
                (0, instructions.len()),
                "{name}"
            );
            assert_eq!(span.range(), 0..instructions.len());
            assert_eq!(span.len(), instructions.len());
            assert_eq!(span.is_empty(), instructions.is_empty());
            assert_eq!(span.instructions(&cfg), instructions);
            let points = span.points();
            assert_eq!(points.len(), instructions.len());
            for (index, point) in points.enumerate() {
                assert_eq!((point.block, point.inst_idx), (span.block(), index));
            }
        }
    }
}

#[test]
#[should_panic(expected = "out of range")]
fn reading_a_span_from_a_graph_the_tree_was_not_computed_from_panics() {
    let flow = Flow::starting_with("return");
    let (tree, _) = structure_and_check(&flow.cfg);
    let ControlNode::Return(span) = *tree.node(tree.root()) else {
        panic!("a lone return");
    };
    let other = Cfg::<&'static str>::new();

    let _ = span.instructions(&other);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn asking_for_a_node_the_tree_does_not_have_panics() {
    let flow = Flow::starting_with("return");
    let (tree, _) = structure_and_check(&flow.cfg);

    let _ = tree.node(ControlNodeId::from_raw(9));
}

#[test]
fn instructions_come_in_program_structure_order() {
    // while i < n { sum += i; i += 1 }: the condition runs before the body.
    let mut pre_tested = Flow::starting_with("i = 0");
    let header = pre_tested.stmt("i < n");
    let body = pre_tested.stmt("sum += i");
    let step = pre_tested.stmt("i += 1");
    let exit = pre_tested.stmt("return sum");
    pre_tested.fall(pre_tested.entry(), header);
    pre_tested.branch(header, body, exit);
    pre_tested.fall(body, step);
    pre_tested.jump(step, header);

    // do { i += 1 } while (i < n): the condition runs after the body.
    let mut post_tested = Flow::starting_with("i = 0");
    let body = post_tested.stmt("i += 1");
    let latch = post_tested.stmt("i < n");
    let exit = post_tested.stmt("return i");
    post_tested.fall(post_tested.entry(), body);
    post_tested.fall(body, latch);
    post_tested.edge(latch, body, EdgeKind::ConditionalTrue);
    post_tested.edge(latch, exit, EdgeKind::ConditionalFalse);

    // A branch block's instructions come before both arms.
    let mut branch = Flow::starting_with("n = len(xs)");
    let test = branch.entry();
    branch.push(test, "n > 0");
    let yes = branch.stmt("y = 1");
    let no = branch.stmt("y = 2");
    let join = branch.stmt("return y");
    branch.branch(test, yes, no);
    branch.fall(yes, join);
    branch.fall(no, join);

    for (flow, expected) in [
        (
            pre_tested,
            vec!["i = 0", "i < n", "sum += i", "i += 1", "return sum"],
        ),
        (post_tested, vec!["i = 0", "i += 1", "i < n", "return i"]),
        (
            branch,
            vec!["n = len(xs)", "n > 0", "y = 1", "y = 2", "return y"],
        ),
    ] {
        let (tree, _) = structure_and_check(&flow.cfg);
        let mut seen = Vec::new();
        tree.for_each_instruction(&flow.cfg, |_, instruction| seen.push(*instruction));
        assert_eq!(seen, expected);
    }
}

#[test]
fn a_tree_is_a_value_that_clones_and_compares_equal() {
    let (_, _, tree) = sample_trees().remove(1);

    let copy = tree.clone();

    assert_eq!(copy, tree);
    assert_eq!(copy.root(), tree.root());
    assert_eq!(copy.node_count(), tree.node_count());
}

#[test]
fn runs_report_the_length_of_the_slice_they_name() {
    for (name, _, tree) in sample_trees() {
        for id in tree.node_ids() {
            for body in tree.child_bodies(id) {
                assert_eq!(body.len(), tree.body(body).len(), "{name}");
                assert_eq!(body.is_empty(), tree.body(body).is_empty());
            }
            match *tree.node(id) {
                ControlNode::Switch { cases, .. } => {
                    assert_eq!(cases.len(), tree.cases(cases).len());
                    for case in tree.cases(cases) {
                        assert_eq!(case.edges.len(), tree.edges(case.edges).len());
                        assert!(!case.edges.is_empty(), "an arm is selected by an edge");
                    }
                }
                ControlNode::TryCatch { handlers, .. } => {
                    assert_eq!(handlers.len(), tree.handlers(handlers).len());
                }
                ControlNode::Loop {
                    kind: cfglib::LoopKind::While { condition, .. },
                    ..
                } => {
                    assert_eq!(condition.len(), tree.spans(condition).len());
                    assert!(!condition.is_empty(), "the header is in the chain");
                }
                _ => {}
            }
        }
    }
}
