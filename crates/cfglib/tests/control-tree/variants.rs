//! The two structuring variants: blocks structured as if they were empty, and
//! predicated instructions regionized into guarded nodes.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, ControlNode, ControlTree, EdgeKind, InstrInfo, Predicated, ProgramPoint,
};

use crate::builders::{Flow, pseudocode};
use crate::coverage::{
    assert_instruction_walk_is_a_partition, assert_spans_tile_each_block, reachable_blocks,
};
use crate::golden::assert_golden;
use crate::implied_flow::assert_flow_matches_graph_with;
use crate::reading::{kind_name, shape, top_level};
use crate::tree_shape::{assert_well_formed, nodes_where};

/// A loop whose `break` and `continue` each compile to a block holding one
/// `jmp`: the trampolines a source-level lifter wants structured away.
struct Trampolines {
    flow: Flow,
    break_hop: BlockId,
    continue_hop: BlockId,
}

fn loop_with_trampolines() -> Trampolines {
    // while i < n { if done(i) { break }; if skip(i) { continue }; work(i); i += 1 }
    let mut flow = Flow::starting_with("i = 0");
    let header = flow.stmt("i < n");
    let done = flow.stmt("done(i)");
    let break_hop = flow.stmt("jmp after_loop");
    let skip = flow.stmt("skip(i)");
    let continue_hop = flow.stmt("jmp loop_top");
    let work = flow.stmt("work(i)");
    let step = flow.stmt("i += 1");
    let exit = flow.stmt("return");
    flow.fall(flow.entry(), header);
    flow.branch(header, done, exit);
    flow.branch(done, break_hop, skip);
    flow.jump(break_hop, exit);
    flow.branch(skip, continue_hop, work);
    flow.jump(continue_hop, header);
    flow.fall(work, step);
    flow.jump(step, header);
    Trampolines {
        flow,
        break_hop,
        continue_hop,
    }
}

#[test]
fn treating_a_jump_trampoline_as_empty_structures_it_as_a_plain_break_or_continue() {
    let Trampolines {
        flow,
        break_hop,
        continue_hop,
    } = loop_with_trampolines();
    let empty = [break_hop, continue_hop];

    let (as_is, as_is_report) = ControlTree::compute_with_report(&flow.cfg);
    let (tree, report) = ControlTree::compute_with_report_treating_empty(&flow.cfg, &empty);

    assert!(as_is_report.is_fully_structured(), "{as_is_report:?}");
    assert!(report.is_fully_structured(), "{report:?}");
    assert_eq!(
        pseudocode(&flow.cfg, &as_is),
        "i = 0\nwhile {\n    i < n\n    break if !cond;\n    if (done(i)) {\n        jmp after_loop\n        break;\n    } else {\n        if (skip(i)) {\n            jmp loop_top\n            continue;\n        } else {\n            work(i)\n            i += 1\n            continue;\n        }\n    }\n}\nreturn\n"
    );
    assert_eq!(
        pseudocode(&flow.cfg, &tree),
        "i = 0\nwhile {\n    i < n\n    break if !cond;\n    if (done(i)) {\n        break;\n    } else {\n        if (skip(i)) {\n            continue;\n        } else {\n            work(i)\n            i += 1\n            continue;\n        }\n    }\n}\nreturn\n"
    );
}

#[test]
fn treating_blocks_as_empty_equals_structuring_a_copy_whose_blocks_were_cleared() {
    let Trampolines {
        flow,
        break_hop,
        continue_hop,
    } = loop_with_trampolines();
    let empty = [break_hop, continue_hop];

    let mut cleared = flow.cfg.clone();
    for block in empty {
        cleared.block_mut(block).instructions_mut().clear();
    }
    let (expected_tree, expected_report) = ControlTree::compute_with_report(&cleared);
    let (tree, report) = ControlTree::compute_with_report_treating_empty(&flow.cfg, &empty);

    assert_eq!(tree, expected_tree);
    assert_eq!(report, expected_report);
    assert!(
        flow.cfg
            .block(break_hop)
            .instructions()
            .contains(&"jmp after_loop"),
        "the graph is not changed"
    );
}

#[test]
fn treating_no_block_as_empty_is_compute_with_report() {
    let Trampolines { flow, .. } = loop_with_trampolines();

    assert_eq!(
        ControlTree::compute_with_report_treating_empty(&flow.cfg, &[]),
        ControlTree::compute_with_report(&flow.cfg)
    );
}

#[test]
fn trees_treating_blocks_as_empty_imply_the_flow_through_those_blocks() {
    let Trampolines {
        flow,
        break_hop,
        continue_hop,
    } = loop_with_trampolines();
    let empty: BTreeSet<BlockId> = [break_hop, continue_hop].into_iter().collect();

    let (tree, report) =
        ControlTree::compute_with_report_treating_empty(&flow.cfg, &[break_hop, continue_hop]);

    assert_well_formed(&tree);
    let covered = reachable_blocks(&flow.cfg);
    assert_flow_matches_graph_with(&flow.cfg, &tree, &report, &covered, &|block| {
        empty.contains(&block) || flow.cfg.block(block).is_empty()
    });
    let mut kept = covered;
    for block in &empty {
        kept.remove(block);
    }
    assert_spans_tile_each_block(&flow.cfg, &tree, &kept);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn treating_a_block_the_graph_lacks_as_empty_panics() {
    let flow = Flow::starting_with("only");

    let _ = ControlTree::compute_with_report_treating_empty(&flow.cfg, &[BlockId::from_raw(9)]);
}

/// A shader-like instruction: wave-predicated, with the guard among its reads,
/// as the `Predicated` contract requires.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Op {
    text: &'static str,
    guard: Option<(&'static str, bool)>,
    reads: Vec<&'static str>,
}

impl Op {
    fn plain(text: &'static str) -> Self {
        Self {
            text,
            guard: None,
            reads: Vec::new(),
        }
    }

    fn when(text: &'static str, predicate: &'static str, polarity: bool) -> Self {
        Self {
            text,
            guard: Some((predicate, polarity)),
            reads: vec![predicate],
        }
    }
}

impl InstrInfo for Op {
    type Variable = &'static str;

    fn uses(&self) -> &[&'static str] {
        &self.reads
    }

    fn defs(&self) -> &[&'static str] {
        &[]
    }
}

impl Predicated for Op {
    fn predicate(&self) -> Option<(&'static str, bool)> {
        self.guard
    }
}

fn render(cfg: &Cfg<Op>, tree: &ControlTree) -> String {
    tree.to_pseudocode(|sink, point| {
        sink.write_str(cfg.block(point.block).instructions()[point.inst_idx].text)
    })
    .expect("writing to a String cannot fail")
}

fn block_of(cfg: &mut Cfg<Op>, ops: Vec<Op>) -> BlockId {
    let block = cfg.new_block();
    cfg.block_mut(block).instructions_mut().extend(ops);
    block
}

/// One block: `p0 = lane < 8; (p0) y = a; (!p0) y = b; store y`.
fn wave_select() -> Cfg<Op> {
    let mut cfg = Cfg::<Op>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).instructions_mut().extend([
        Op::plain("p0 = lane < 8"),
        Op::when("y = a", "p0", true),
        Op::when("y = b", "p0", false),
        Op::plain("store y"),
    ]);
    cfg
}

#[test]
fn guarded_runs_wrap_each_predicated_run_and_keep_the_final_run_a_return() {
    let cfg = wave_select();
    let entry = cfg.entry();

    let tree = ControlTree::compute_predicated(&cfg);

    assert_well_formed(&tree);
    let statements = top_level(&tree);
    assert_eq!(
        statements.iter().map(kind_name).collect::<Vec<_>>(),
        ["block", "guarded", "guarded", "return"],
        "{tree:?}"
    );
    let ControlNode::Guarded {
        predicate,
        when_true,
        body,
    } = statements[1]
    else {
        unreachable!("checked above");
    };
    assert_eq!(
        predicate,
        ProgramPoint {
            block: entry,
            inst_idx: 1
        }
    );
    assert!(when_true);
    assert!(matches!(
        tree.body(body).iter().map(|&id| *tree.node(id)).collect::<Vec<_>>()[..],
        [ControlNode::Block(span)] if span.block() == entry && span.range() == (1..2)
    ));
    let ControlNode::Guarded {
        predicate,
        when_true,
        ..
    } = statements[2]
    else {
        unreachable!("checked above");
    };
    assert_eq!(
        predicate,
        ProgramPoint {
            block: entry,
            inst_idx: 2
        }
    );
    assert!(!when_true);
    assert!(
        matches!(statements[3], ControlNode::Return(span) if span.range() == (3..4)),
        "the final unpredicated run keeps its Return"
    );
    assert_golden("control-tree/wave-select.pseudo", &render(&cfg, &tree));
}

#[test]
fn a_guarded_node_holds_its_run_as_its_only_child_body() {
    let cfg = wave_select();
    let tree = ControlTree::compute_predicated(&cfg);

    let guarded = nodes_where(&tree, |node| matches!(node, ControlNode::Guarded { .. }));
    assert_eq!(guarded.len(), 2);
    for id in guarded {
        let ControlNode::Guarded { body, .. } = *tree.node(id) else {
            unreachable!("filtered to guarded nodes");
        };
        assert_eq!(tree.child_bodies(id).collect::<Vec<_>>(), [body]);
    }
}

#[test]
fn spans_of_a_split_block_read_back_their_slice_of_the_instructions() {
    let cfg = wave_select();
    let entry = cfg.entry();
    let tree = ControlTree::compute_predicated(&cfg);

    let mut spans = crate::coverage::leaf_spans(&tree);
    spans.sort();

    assert_eq!(
        spans.iter().map(|span| span.range()).collect::<Vec<_>>(),
        [0..1, 1..2, 2..3, 3..4]
    );
    for span in spans {
        assert_eq!(span.block(), entry);
        assert_eq!(span.len(), span.end() - span.start());
        assert_eq!(
            span.instructions(&cfg),
            &cfg.block(entry).instructions()[span.range()]
        );
        let points: Vec<ProgramPoint> = span.points().collect();
        assert_eq!(
            points,
            (span.start()..span.end())
                .map(|inst_idx| ProgramPoint {
                    block: entry,
                    inst_idx
                })
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn a_block_splits_into_spans_that_tile_it_and_the_walk_order_is_unchanged() {
    let cfg = wave_select();
    let covered: BTreeSet<BlockId> = cfg.block_ids().collect();
    let plain = ControlTree::compute(&cfg);

    let tree = ControlTree::compute_predicated(&cfg);

    assert_spans_tile_each_block(&cfg, &tree, &covered);
    assert_instruction_walk_is_a_partition(&cfg, &tree, &covered);
    let order = |tree: &ControlTree| {
        let mut points = Vec::new();
        tree.for_each_instruction(&cfg, |point, _| points.push(point));
        points
    };
    assert_eq!(order(&tree), order(&plain));
    assert_eq!(
        crate::coverage::block_appearances(&tree)[&cfg.entry()],
        4,
        "the one block is named by four spans"
    );
}

#[test]
fn predicated_prefix_of_a_branch_header_is_hoisted_in_front_of_the_branch() {
    let mut cfg = Cfg::<Op>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).instructions_mut().extend([
        Op::plain("p0 = a < b"),
        Op::when("c = c + 1", "p0", true),
        Op::plain("branch c > 3"),
    ]);
    let then_arm = block_of(&mut cfg, vec![Op::plain("hit()")]);
    let join = block_of(&mut cfg, vec![Op::plain("return")]);
    cfg.add_edge(entry, then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, join, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, join, EdgeKind::Fallthrough);

    let tree = ControlTree::compute_predicated(&cfg);

    assert_well_formed(&tree);
    let statements = top_level(&tree);
    assert_eq!(
        statements.iter().map(kind_name).collect::<Vec<_>>(),
        ["sequence", "return"],
        "the hoisted prefix and the branch form one statement: {tree:?}"
    );
    let ControlNode::Sequence(body) = statements[0] else {
        unreachable!("checked above");
    };
    let hoisted: Vec<ControlNode> = tree.body(body).iter().map(|&id| *tree.node(id)).collect();
    assert_eq!(
        hoisted.iter().map(kind_name).collect::<Vec<_>>(),
        ["block", "guarded", "if"]
    );
    let ControlNode::IfThenElse { condition, .. } = hoisted[2] else {
        unreachable!("checked above");
    };
    assert_eq!(condition.range(), 2..3, "only the branch instruction stays");
    assert_eq!(
        render(&cfg, &tree),
        "p0 = a < b\n@guarded(c = c + 1) {\n    c = c + 1\n}\nif (branch c > 3) {\n    hit()\n}\nreturn\n"
    );
}

#[test]
fn a_guard_on_the_branch_instruction_itself_stays_inline() {
    let mut cfg = Cfg::<Op>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry)
        .instructions_mut()
        .extend([Op::plain("p0 = a < b"), Op::when("branch c", "p0", true)]);
    let then_arm = block_of(&mut cfg, vec![Op::plain("hit()")]);
    let join = block_of(&mut cfg, vec![Op::plain("return")]);
    cfg.add_edge(entry, then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, join, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, join, EdgeKind::Fallthrough);

    let tree = ControlTree::compute_predicated(&cfg);

    assert!(
        nodes_where(&tree, |node| matches!(node, ControlNode::Guarded { .. })).is_empty(),
        "a predicated branch is unrepresentable as a region: {tree:?}"
    );
    let ControlNode::IfThenElse { condition, .. } = top_level(&tree)[0] else {
        panic!("the branch stays a branch");
    };
    assert_eq!(condition.range(), 0..2);
}

#[test]
fn predicated_instructions_in_a_loop_condition_are_not_regionized() {
    // A condition re-evaluates every iteration, so hoisting a guarded
    // instruction out of it would change what runs.
    let mut cfg = Cfg::<Op>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry)
        .instructions_mut()
        .push(Op::plain("i = 0"));
    let header = block_of(
        &mut cfg,
        vec![
            Op::plain("p0 = i < n"),
            Op::when("touch(i)", "p0", true),
            Op::plain("test p0"),
        ],
    );
    let body = block_of(&mut cfg, vec![Op::plain("i += 1")]);
    let exit = block_of(&mut cfg, vec![Op::plain("return")]);
    cfg.add_edge(entry, header, EdgeKind::Fallthrough);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, header, EdgeKind::Jump);

    let tree = ControlTree::compute_predicated(&cfg);

    assert!(
        nodes_where(&tree, |node| matches!(node, ControlNode::Guarded { .. })).is_empty(),
        "{tree:?}"
    );
    assert!(render(&cfg, &tree).contains("touch(i)"));
    let covered: BTreeSet<BlockId> = cfg.block_ids().collect();
    assert_spans_tile_each_block(&cfg, &tree, &covered);
}

#[test]
fn a_graph_without_predicated_instructions_structures_like_compute() {
    // while i < n { if odd(i) { a() } else { b() }; i += 1 }
    let mut cfg = Cfg::<Op>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry)
        .instructions_mut()
        .push(Op::plain("i = 0"));
    let header = block_of(&mut cfg, vec![Op::plain("i < n")]);
    let odd = block_of(&mut cfg, vec![Op::plain("odd(i)")]);
    let then_arm = block_of(&mut cfg, vec![Op::plain("a()")]);
    let else_arm = block_of(&mut cfg, vec![Op::plain("b()")]);
    let step = block_of(&mut cfg, vec![Op::plain("i += 1")]);
    let exit = block_of(&mut cfg, vec![Op::plain("return")]);
    cfg.add_edge(entry, header, EdgeKind::Fallthrough);
    cfg.add_edge(header, odd, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(odd, then_arm, EdgeKind::ConditionalTrue);
    cfg.add_edge(odd, else_arm, EdgeKind::ConditionalFalse);
    cfg.add_edge(then_arm, step, EdgeKind::Fallthrough);
    cfg.add_edge(else_arm, step, EdgeKind::Fallthrough);
    cfg.add_edge(step, header, EdgeKind::Jump);

    let plain = ControlTree::compute(&cfg);
    let tree = ControlTree::compute_predicated(&cfg);

    assert_eq!(shape(&tree), shape(&plain));
    assert_eq!(render(&cfg, &tree), render(&cfg, &plain));
}

#[test]
fn the_predicated_report_is_the_report_of_the_unpredicated_structuring() {
    // An irreducible pair with predicated code in it still reports its gotos.
    let mut cfg = Cfg::<Op>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry)
        .instructions_mut()
        .push(Op::plain("start"));
    let a = block_of(
        &mut cfg,
        vec![Op::when("A", "p0", true), Op::plain("test A")],
    );
    let b = block_of(&mut cfg, vec![Op::plain("test B")]);
    let exit = block_of(&mut cfg, vec![Op::plain("return")]);
    cfg.add_edge(entry, a, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, b, EdgeKind::ConditionalFalse);
    cfg.add_edge(a, b, EdgeKind::ConditionalTrue);
    cfg.add_edge(a, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(b, a, EdgeKind::ConditionalTrue);
    cfg.add_edge(b, exit, EdgeKind::ConditionalFalse);

    let (_, expected) = ControlTree::compute_with_report(&cfg);
    let (tree, report) = ControlTree::compute_predicated_with_report(&cfg);

    assert_eq!(report, expected);
    assert!(!report.is_fully_structured());
    assert_well_formed(&tree);
    let covered = reachable_blocks(&cfg);
    assert_spans_tile_each_block(&cfg, &tree, &covered);
}
