//! Traversal, child-body order, instruction order, and pseudocode.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::{pseudocode, subtree};
use crate::ProgramPoint;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::flow::FlowEffect;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::{MockInst, ff};
use crate::{ControlDescent, ControlNode, ControlNodeId, ControlTree, ControlVisitor};

/// `entry; if c { a } else { b }; loop-free merge`.
fn diamond() -> Cfg<MockInst> {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let then_arm = cfg.new_block();
    let else_arm = cfg.new_block();
    let merge = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("prefix"));
    cfg.block_mut(cfg.entry()).push(ff("test"));
    cfg.block_mut(then_arm).push(ff("then"));
    cfg.block_mut(else_arm).push(ff("else"));
    cfg.block_mut(merge)
        .push(MockInst(FlowEffect::Return, "merge"));
    cfg.add_edge(cfg.entry(), then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), else_arm, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, merge, EdgeKind::Fallthrough);
    cfg.add_edge(else_arm, merge, EdgeKind::Fallthrough);
    cfg
}

#[test]
fn the_empty_function_is_an_empty_tree() {
    let cfg: Cfg<MockInst> = Cfg::new();
    let tree = ControlTree::compute(&cfg);
    assert!(tree.is_empty());
    assert_eq!(tree.node_count(), 1);
    assert_eq!(pseudocode(&cfg, &tree), "");
}

#[test]
fn preorder_visits_every_node_once_in_execution_order() {
    let cfg = diamond();
    let tree = ControlTree::compute(&cfg);
    let order: Vec<ControlNodeId> = tree.preorder().collect();
    assert_eq!(order, subtree(&tree, tree.root()));
    assert_eq!(order.len(), tree.node_count());
    let mut sorted = order.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), tree.node_count(), "every node exactly once");
    assert_eq!(order[0], tree.root());
}

#[test]
fn children_have_lower_handles_than_their_parent() {
    let cfg = diamond();
    let tree = ControlTree::compute(&cfg);
    assert_eq!(tree.root().index(), tree.node_count() - 1);
    for id in tree.node_ids() {
        for body in tree.child_bodies(id) {
            for &child in tree.body(body) {
                assert!(child < id, "{child} should precede {id}");
            }
        }
    }
    assert_eq!(tree.node_ids().len(), tree.node_count());
}

#[test]
fn child_bodies_follow_execution_order() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let case_a = cfg.new_block();
    let case_b = cfg.new_block();
    let default_arm = cfg.new_block();
    let merge = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("dispatch"));
    cfg.block_mut(case_a).push(ff("a"));
    cfg.block_mut(case_b).push(ff("b"));
    cfg.block_mut(default_arm).push(ff("d"));
    cfg.block_mut(merge)
        .push(MockInst(FlowEffect::Return, "after"));
    cfg.add_edge(cfg.entry(), case_a, EdgeKind::SwitchCase);
    cfg.add_edge(cfg.entry(), case_b, EdgeKind::SwitchCase);
    cfg.add_edge(cfg.entry(), default_arm, EdgeKind::Fallthrough);
    for arm in [case_a, case_b, default_arm] {
        cfg.add_edge(arm, merge, EdgeKind::Unconditional);
    }

    let tree = ControlTree::compute(&cfg);
    let switch = tree
        .preorder()
        .find(|&id| matches!(tree.node(id), ControlNode::Switch { .. }))
        .expect("a switch");
    let ControlNode::Switch {
        cases,
        default_body,
        ..
    } = *tree.node(switch)
    else {
        unreachable!()
    };
    let expected: Vec<_> = tree
        .cases(cases)
        .iter()
        .map(|case| case.body)
        .chain([default_body])
        .collect();
    let bodies = tree.child_bodies(switch);
    assert_eq!(bodies.len(), 3);
    assert_eq!(bodies.clone().collect::<Vec<_>>(), expected);
    let mut backwards: Vec<_> = bodies.rev().collect();
    backwards.reverse();
    assert_eq!(backwards, expected);
}

#[test]
fn try_child_bodies_are_try_handlers_then_finally() {
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let try_body = cfg.new_block();
    let catch = cfg.new_block();
    let cleanup = cfg.new_block();
    let after = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(try_body).push(ff("try"));
    cfg.block_mut(catch).push(ff("catch"));
    cfg.block_mut(cleanup).push(ff("cleanup"));
    cfg.block_mut(after).push(ff("after"));
    cfg.add_edge(cfg.entry(), try_body, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, after, EdgeKind::Fallthrough);
    cfg.add_edge(try_body, catch, EdgeKind::ExceptionHandler);
    cfg.add_edge(try_body, cleanup, EdgeKind::ExceptionUnwind);
    cfg.add_edge(catch, after, EdgeKind::Fallthrough);
    cfg.add_edge(cleanup, after, EdgeKind::Fallthrough);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [try_body].into_iter().collect(),
        handlers: vec![
            Handler {
                entry: catch,
                body: HandlerBody::known([catch]),
                kind: HandlerKind::Catch,
            },
            Handler {
                entry: cleanup,
                body: HandlerBody::known([cleanup]),
                kind: HandlerKind::Finally,
            },
        ],
        parent: None,
    });

    let tree = ControlTree::compute(&cfg);
    let try_node = tree
        .preorder()
        .find(|&id| matches!(tree.node(id), ControlNode::TryCatch { .. }))
        .expect("a try region");
    let ControlNode::TryCatch {
        try_body,
        handlers,
        finally_body,
    } = *tree.node(try_node)
    else {
        unreachable!()
    };
    let expected: Vec<_> = [try_body]
        .into_iter()
        .chain(tree.handlers(handlers).iter().map(|handler| handler.body))
        .chain([finally_body])
        .collect();
    assert_eq!(tree.child_bodies(try_node).collect::<Vec<_>>(), expected);
    assert_eq!(expected.len(), 3);
}

#[test]
fn leaf_nodes_have_no_child_bodies() {
    let cfg = diamond();
    let tree = ControlTree::compute(&cfg);
    for id in tree.preorder() {
        let leaf = matches!(
            tree.node(id),
            ControlNode::Block(_)
                | ControlNode::Return(_)
                | ControlNode::Break { .. }
                | ControlNode::Continue { .. }
                | ControlNode::Goto { .. }
        );
        assert_eq!(
            tree.child_bodies(id).len() == 0,
            leaf,
            "{:?}",
            tree.node(id)
        );
    }
}

/// Records enter and leave events, optionally skipping loops.
struct Recorder {
    events: Vec<(bool, ControlNodeId)>,
    skip_loops: bool,
}

impl ControlVisitor for Recorder {
    fn enter(&mut self, tree: &ControlTree, node: ControlNodeId) -> ControlDescent {
        self.events.push((true, node));
        if self.skip_loops && matches!(tree.node(node), ControlNode::Loop { .. }) {
            ControlDescent::Skip
        } else {
            ControlDescent::Into
        }
    }

    fn leave(&mut self, _: &ControlTree, node: ControlNodeId) {
        self.events.push((false, node));
    }
}

fn nested_loop_cfg() -> Cfg<MockInst> {
    // entry → outer{cond} → inner{cond} → body → inner; inner --false--> latch → outer
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let outer = cfg.new_block();
    let inner = cfg.new_block();
    let body = cfg.new_block();
    let latch = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(outer).push(ff("outer_cond"));
    cfg.block_mut(inner).push(ff("inner_cond"));
    cfg.block_mut(body).push(ff("inner_body"));
    cfg.block_mut(latch).push(ff("latch"));
    cfg.block_mut(exit)
        .push(MockInst(FlowEffect::Return, "exit"));
    cfg.add_edge(cfg.entry(), outer, EdgeKind::Fallthrough);
    cfg.add_edge(outer, inner, EdgeKind::ConditionalTrue);
    cfg.add_edge(outer, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(inner, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(inner, latch, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, inner, EdgeKind::Jump);
    cfg.add_edge(latch, outer, EdgeKind::Jump);
    cfg
}

#[test]
fn walk_pairs_enter_and_leave_around_the_children() {
    let cfg = nested_loop_cfg();
    let tree = ControlTree::compute(&cfg);
    let mut recorder = Recorder {
        events: Vec::new(),
        skip_loops: false,
    };
    tree.walk(&mut recorder);

    let entered: Vec<_> = recorder
        .events
        .iter()
        .filter(|event| event.0)
        .map(|event| event.1)
        .collect();
    assert_eq!(entered, tree.preorder().collect::<Vec<_>>());

    // Leaves come in reverse nesting order: a stack discipline.
    let mut open = Vec::new();
    for &(is_enter, id) in &recorder.events {
        if is_enter {
            open.push(id);
        } else {
            assert_eq!(open.pop(), Some(id));
        }
    }
    assert_eq!(open, []);
    assert_eq!(recorder.events.len(), 2 * tree.node_count());
}

#[test]
fn skipping_a_node_still_leaves_it_but_not_its_children() {
    let cfg = nested_loop_cfg();
    let tree = ControlTree::compute(&cfg);
    let mut recorder = Recorder {
        events: Vec::new(),
        skip_loops: true,
    };
    tree.walk(&mut recorder);

    let loops: Vec<_> = tree
        .preorder()
        .filter(|&id| matches!(tree.node(id), ControlNode::Loop { .. }))
        .collect();
    assert_eq!(loops.len(), 2, "{tree:?}");
    let outer = loops[0];
    let nested = loops[1];
    let entered = |id| recorder.events.contains(&(true, id));
    assert!(entered(outer));
    assert!(
        !entered(nested),
        "the children of a skipped loop are not entered"
    );
    assert!(recorder.events.contains(&(false, outer)));
    let position = |event| recorder.events.iter().position(|e| *e == event).unwrap();
    assert_eq!(
        position((false, outer)),
        position((true, outer)) + 1,
        "a skipped node is left immediately"
    );
}

#[test]
fn for_each_instruction_follows_program_structure_order() {
    // Pre-tested loop: the condition comes before the body.
    let (cfg, ..) = super::while_shaped_cfg();
    let tree = ControlTree::compute(&cfg);
    let mut names = Vec::new();
    tree.for_each_instruction(&cfg, |_, inst: &MockInst| names.push(inst.1));
    assert_eq!(names, ["entry", "condition", "body_inst", "after_loop"]);

    // Post-tested loop: the latch's condition comes after the body.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let body = cfg.new_block();
    let latch = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(body).push(ff("body_inst"));
    cfg.block_mut(latch).push(ff("latch_cond"));
    cfg.block_mut(exit)
        .push(MockInst(FlowEffect::Return, "after_loop"));
    cfg.add_edge(cfg.entry(), body, EdgeKind::Fallthrough);
    cfg.add_edge(body, latch, EdgeKind::Fallthrough);
    cfg.add_edge(latch, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(latch, exit, EdgeKind::ConditionalFalse);
    let tree = ControlTree::compute(&cfg);
    let mut names = Vec::new();
    tree.for_each_instruction(&cfg, |_, inst: &MockInst| names.push(inst.1));
    assert_eq!(names, ["entry", "body_inst", "latch_cond", "after_loop"]);
}

#[test]
fn for_each_instruction_reports_positions() {
    let cfg = diamond();
    let tree = ControlTree::compute(&cfg);
    let mut points: Vec<ProgramPoint> = Vec::new();
    tree.for_each_instruction(&cfg, |point, inst: &MockInst| {
        assert_eq!(
            cfg.block(point.block).instructions()[point.inst_idx].1,
            inst.1
        );
        points.push(point);
    });
    let blocks: Vec<BlockId> = points.iter().map(|point| point.block).collect();
    assert_eq!(blocks[0], cfg.entry());
    assert_eq!(blocks[1], cfg.entry());
    assert_eq!(points[0].inst_idx, 0);
    assert_eq!(points[1].inst_idx, 1);
    assert_eq!(points.len(), 5);
}

#[test]
fn pseudocode_prints_structure_with_the_callers_instruction_text() {
    let cfg = diamond();
    let tree = ControlTree::compute(&cfg);
    // The branch's own instruction ("test") belongs to the header; the lines
    // before it print ahead of the construct.
    assert_eq!(
        pseudocode(&cfg, &tree),
        "prefix\nif {\n    then\n} else {\n    else\n}\nmerge\n"
    );
}

#[test]
fn pseudocode_prints_each_loop_shape() {
    let (cfg, ..) = super::while_shaped_cfg();
    let tree = ControlTree::compute(&cfg);
    assert_eq!(
        pseudocode(&cfg, &tree),
        "entry\nwhile {\n    condition\n    break if !cond;\n    body_inst\n}\nafter_loop\n"
    );

    let mut cfg: Cfg<MockInst> = Cfg::new();
    let body = cfg.new_block();
    let latch = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(body).push(ff("body_inst"));
    cfg.block_mut(latch).push(ff("latch_cond"));
    cfg.block_mut(exit)
        .push(MockInst(FlowEffect::Return, "after_loop"));
    cfg.add_edge(cfg.entry(), body, EdgeKind::Fallthrough);
    cfg.add_edge(body, latch, EdgeKind::Fallthrough);
    cfg.add_edge(latch, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(latch, exit, EdgeKind::ConditionalFalse);
    let tree = ControlTree::compute(&cfg);
    assert_eq!(
        pseudocode(&cfg, &tree),
        "do {\n    body_inst\n    latch_cond\n} while (cond)\nafter_loop\n"
    );
}

#[test]
fn pseudocode_names_labels_and_gotos_by_block() {
    // A jump into a sibling arm's interior stays an explicit labeled goto.
    let mut cfg: Cfg<MockInst> = Cfg::new();
    let jumper = cfg.new_block();
    let normal = cfg.new_block();
    let target = cfg.new_block();
    let other = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(jumper).push(ff("jumper"));
    cfg.block_mut(normal).push(ff("normal"));
    cfg.block_mut(target).push(ff("dst"));
    cfg.block_mut(other).push(ff("other"));
    cfg.add_edge(cfg.entry(), jumper, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), normal, EdgeKind::ConditionalFalse);
    cfg.add_edge(jumper, target, EdgeKind::Jump);
    cfg.add_edge(normal, target, EdgeKind::ConditionalTrue);
    cfg.add_edge(normal, other, EdgeKind::ConditionalFalse);

    let tree = ControlTree::compute(&cfg);
    let text = pseudocode(&cfg, &tree);
    assert!(text.contains(&alloc::format!("goto .{target};")), "{text}");
    assert!(text.contains(&alloc::format!(".{target}:\n")), "{text}");
}

#[test]
fn a_failing_instruction_printer_fails_the_rendering() {
    let cfg = diamond();
    let tree = ControlTree::compute(&cfg);
    let result = tree.to_pseudocode(|_, _| Err(core::fmt::Error));
    assert!(result.is_err());
}

#[test]
fn a_span_names_a_run_of_one_blocks_instructions() {
    let cfg = diamond();
    let tree = ControlTree::compute(&cfg);
    let condition = tree
        .preorder()
        .find_map(|id| match *tree.node(id) {
            ControlNode::IfThenElse { condition, .. } => Some(condition),
            _ => None,
        })
        .expect("the diamond has a branch");
    assert_eq!(condition.block(), cfg.entry());
    assert_eq!((condition.start(), condition.end()), (0, 2));
    assert_eq!(condition.range(), 0..2);
    assert_eq!(condition.len(), 2);
    assert!(!condition.is_empty());
    let names: Vec<&str> = condition
        .instructions(&cfg)
        .iter()
        .map(|inst| inst.1)
        .collect();
    assert_eq!(names, ["prefix", "test"]);
    let points: Vec<ProgramPoint> = condition.points().collect();
    assert_eq!(
        points,
        [
            ProgramPoint {
                block: cfg.entry(),
                inst_idx: 0
            },
            ProgramPoint {
                block: cfg.entry(),
                inst_idx: 1
            }
        ]
    );
}

#[test]
fn node_handles_are_dense_identities() {
    use crate::DenseId;

    let id = ControlNodeId::from_index(3);
    assert_eq!(id.index(), 3);
    assert_eq!(<ControlNodeId as DenseId>::from_index(3), id);
    assert_eq!(<ControlNodeId as DenseId>::index(id), 3);
    assert_eq!(ControlNodeId::from_raw(3), id);
    assert_eq!(alloc::format!("{id}"), "node3");
}

#[test]
fn runs_know_their_length_and_compare_by_position() {
    let cfg = diamond();
    let tree = ControlTree::compute(&cfg);
    let ControlNode::Sequence(body) = *tree.node(tree.root()) else {
        panic!("a branch followed by its merge is a sequence: {tree:?}");
    };
    assert_eq!(body.len(), tree.body(body).len());
    assert!(!body.is_empty());
    let copy = body;
    assert_eq!(copy, body);
    assert_eq!(copy.len(), body.len());
}
