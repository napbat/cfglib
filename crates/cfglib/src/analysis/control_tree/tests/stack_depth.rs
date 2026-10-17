//! Structuring and reading a deeply nested tree on a small stack.
//!
//! A decompiler feeds the structurer untrusted binaries, so the nesting depth
//! of the input must not be bounded by the call stack. Every test builds its
//! graph and runs the structurer and the readers of the tree (the regionizer,
//! the walk, the printer) on a thread with a 256 KiB stack, far less than the
//! frames of one recursive call per level take at these depths: an overflow
//! aborts the test binary.
//!
//! The depths differ by what the input size allows. A straight chain, the
//! nested `if`s, `switch`es, and goto labels are linear in their depth, so
//! they run at 100,000 levels. The protected sets of nested `try` regions and
//! the bodies of nested loops are sets of blocks, so a graph of that depth is
//! quadratic in its depth before the structurer reads it; those tests use the
//! largest depth that finishes in about a second. The printer's output is
//! quadratic in any nesting, since each level indents every line: the linear
//! shapes are printed at 3,000 levels, and the loops and tries as they are.

extern crate alloc;
extern crate std;
use alloc::vec::Vec;
use core::fmt;
use std::thread;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::display::DisplayInstr;
use crate::edge::EdgeKind;
use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};
use crate::test_util::{DfInst, df_ff, df_pred};
use crate::{
    ControlDescent, ControlNode, ControlNodeId, ControlTree, ControlVisitor, LiftReport, LoopKind,
};

const SMALL_STACK: usize = 256 * 1024;

/// Levels of a nesting whose graph and tree are linear in their depth.
const LINEAR_DEPTH: usize = 100_000;
/// Levels of nested loops: every loop holds its own set of blocks, so the
/// natural loops of the graph are quadratic in the depth, and the loop
/// detection alone takes seconds at a few thousand levels.
const LOOP_DEPTH: usize = 600;
/// Levels of nested `try` regions: every region holds its own set of protected
/// blocks.
const TRY_DEPTH: usize = 1_000;
/// Levels of a linear nesting that is printed: the text is quadratic in the
/// depth, since each line is indented by its level.
const PRINTED_DEPTH: usize = 3_000;

/// Run `work` on a thread whose stack is [`SMALL_STACK`], and rethrow its
/// panic.
fn on_small_stack(work: impl FnOnce() + Send + 'static) {
    let worker = thread::Builder::new()
        .stack_size(SMALL_STACK)
        .spawn(work)
        .expect("a thread with a small stack can be spawned");
    if let Err(panic) = worker.join() {
        std::panic::resume_unwind(panic);
    }
}

/// A block that holds one plain and one predicated instruction, so the
/// regionizer has a guard to wrap at every level.
fn new_block(cfg: &mut Cfg<DfInst>) -> BlockId {
    let block = cfg.new_block();
    fill(cfg, block);
    block
}

fn fill(cfg: &mut Cfg<DfInst>, block: BlockId) {
    cfg.block_mut(block).push(df_ff("work"));
    cfg.block_mut(block).push(df_pred("guarded", 7, true));
}

/// `count` blocks in a row, the entry first, each held by [`fill`].
fn new_blocks(cfg: &mut Cfg<DfInst>, count: usize) -> Vec<BlockId> {
    let mut blocks = Vec::with_capacity(count);
    blocks.push(cfg.entry());
    fill(cfg, cfg.entry());
    while blocks.len() < count {
        blocks.push(new_block(cfg));
    }
    blocks
}

/// `if (t0) { if (t1) { ... leaf } }; merge`: every test branches to the next
/// one on true, and to the merge on false.
fn nested_then_arms(depth: usize) -> Cfg<DfInst> {
    let mut cfg = Cfg::new();
    let tests = new_blocks(&mut cfg, depth);
    let leaf = new_block(&mut cfg);
    let merge = new_block(&mut cfg);
    for (index, &test) in tests.iter().enumerate() {
        let next = tests.get(index + 1).copied().unwrap_or(leaf);
        cfg.add_edge(test, next, EdgeKind::ConditionalTrue);
        cfg.add_edge(test, merge, EdgeKind::ConditionalFalse);
    }
    cfg.add_edge(leaf, merge, EdgeKind::Fallthrough);
    cfg
}

/// `if (t0) { a0 } else if (t1) { a1 } else ... else { last }; merge`.
fn else_if_chain(depth: usize) -> Cfg<DfInst> {
    let mut cfg = Cfg::new();
    let tests = new_blocks(&mut cfg, depth);
    let last = new_block(&mut cfg);
    let merge = new_block(&mut cfg);
    for (index, &test) in tests.iter().enumerate() {
        let arm = new_block(&mut cfg);
        let next = tests.get(index + 1).copied().unwrap_or(last);
        cfg.add_edge(test, arm, EdgeKind::ConditionalTrue);
        cfg.add_edge(test, next, EdgeKind::ConditionalFalse);
        cfg.add_edge(arm, merge, EdgeKind::Fallthrough);
    }
    cfg.add_edge(last, merge, EdgeKind::Fallthrough);
    cfg
}

/// `while (c) { if (t0) { if (t1) { ... work } } }`: a test that fails
/// continues the loop, so the merge of every `if` is the loop header, which
/// the walk has already emitted.
fn ifs_closed_by_a_loop(depth: usize) -> Cfg<DfInst> {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    fill(&mut cfg, entry);
    let header = new_block(&mut cfg);
    let exit = new_block(&mut cfg);
    let work = new_block(&mut cfg);
    let tests: Vec<BlockId> = (0..depth).map(|_| new_block(&mut cfg)).collect();
    cfg.add_edge(entry, header, EdgeKind::Fallthrough);
    cfg.add_edge(header, tests[0], EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    for (index, &test) in tests.iter().enumerate() {
        let next = tests.get(index + 1).copied().unwrap_or(work);
        cfg.add_edge(test, next, EdgeKind::ConditionalTrue);
        cfg.add_edge(test, header, EdgeKind::ConditionalFalse);
    }
    cfg.add_edge(work, header, EdgeKind::Jump);
    cfg
}

/// `b0; b1; ...; b(n-1)`: single-successor blocks in a row.
fn straight_chain(length: usize) -> Cfg<DfInst> {
    let mut cfg = Cfg::new();
    let blocks = new_blocks(&mut cfg, length);
    for pair in blocks.windows(2) {
        cfg.add_edge(pair[0], pair[1], EdgeKind::Fallthrough);
    }
    cfg
}

/// `while (c0; c1; ...; c(n-1)) { work }`: the test of a loop is a chain of
/// blocks, one flow successor each, that ends in the branch out of the loop.
fn loop_test_chain(length: usize) -> Cfg<DfInst> {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    fill(&mut cfg, entry);
    let chain: Vec<BlockId> = (0..length).map(|_| new_block(&mut cfg)).collect();
    let work = new_block(&mut cfg);
    let exit = new_block(&mut cfg);
    cfg.add_edge(entry, chain[0], EdgeKind::Fallthrough);
    for pair in chain.windows(2) {
        cfg.add_edge(pair[0], pair[1], EdgeKind::Fallthrough);
    }
    let last = chain[length - 1];
    cfg.add_edge(last, work, EdgeKind::ConditionalTrue);
    cfg.add_edge(last, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(work, chain[0], EdgeKind::Jump);
    cfg
}

/// `while (c0) { while (c1) { ... work } x1 } x0`: loop `i` tests in its
/// header and runs the loop `i + 1` and the block `x(i + 1)` after it.
fn nested_loops(depth: usize) -> Cfg<DfInst> {
    let mut cfg = Cfg::new();
    let start = new_block(&mut cfg);
    let headers: Vec<BlockId> = (0..depth).map(|_| new_block(&mut cfg)).collect();
    let work = new_block(&mut cfg);
    let exits: Vec<BlockId> = (0..depth).map(|_| new_block(&mut cfg)).collect();
    cfg.add_edge(cfg.entry(), start, EdgeKind::Fallthrough);
    cfg.add_edge(start, headers[0], EdgeKind::Fallthrough);
    for (index, &header) in headers.iter().enumerate() {
        let body = headers.get(index + 1).copied().unwrap_or(work);
        cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exits[index], EdgeKind::ConditionalFalse);
        if let Some(&next_exit) = exits.get(index + 1) {
            cfg.add_edge(next_exit, header, EdgeKind::Jump);
        }
    }
    cfg.add_edge(work, headers[depth - 1], EdgeKind::Jump);
    cfg
}

/// `switch (s0) { case 0: switch (s1) { ... } case 1: a0 default: }; merge`.
fn nested_switches(depth: usize) -> Cfg<DfInst> {
    let mut cfg = Cfg::new();
    let dispatches = new_blocks(&mut cfg, depth);
    let leaf = new_block(&mut cfg);
    let merge = new_block(&mut cfg);
    for (index, &dispatch) in dispatches.iter().enumerate() {
        let inner = dispatches.get(index + 1).copied().unwrap_or(leaf);
        let arm = new_block(&mut cfg);
        cfg.add_edge(dispatch, inner, EdgeKind::SwitchCase);
        cfg.add_edge(dispatch, arm, EdgeKind::SwitchCase);
        cfg.add_edge(dispatch, merge, EdgeKind::Fallthrough);
        cfg.add_edge(arm, merge, EdgeKind::Fallthrough);
    }
    cfg.add_edge(leaf, merge, EdgeKind::Fallthrough);
    cfg
}

/// A catch pad for the protected blocks `protected`, which reaches `after`.
fn add_try(cfg: &mut Cfg<DfInst>, protected: &[BlockId], after: BlockId) {
    let pad = new_block(cfg);
    cfg.add_edge(protected[0], pad, EdgeKind::ExceptionHandler);
    cfg.add_edge(pad, after, EdgeKind::Fallthrough);
    cfg.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: protected.iter().copied().collect(),
        handlers: alloc::vec![Handler {
            entry: pad,
            body: HandlerBody::known([pad]),
            kind: HandlerKind::Catch,
        }],
        parent: None,
    });
}

/// `try { p0; try { p1; try { ... } } }`: region `i` protects `p(i)` and
/// every block after it, so each region starts at its own block.
fn nested_tries(depth: usize) -> Cfg<DfInst> {
    let mut cfg = Cfg::new();
    let protected = new_blocks(&mut cfg, depth);
    let after = new_block(&mut cfg);
    for pair in protected.windows(2) {
        cfg.add_edge(pair[0], pair[1], EdgeKind::Fallthrough);
    }
    cfg.add_edge(protected[depth - 1], after, EdgeKind::Fallthrough);
    for index in 0..depth {
        add_try(&mut cfg, &protected[index..], after);
    }
    cfg
}

/// The same `try` nesting with every region anchored at one block: region `i`
/// protects `p0` through `p(n - 1 - i)`.
fn nested_tries_at_one_anchor(depth: usize) -> Cfg<DfInst> {
    let mut cfg = Cfg::new();
    let protected = new_blocks(&mut cfg, depth);
    let after = new_block(&mut cfg);
    for pair in protected.windows(2) {
        cfg.add_edge(pair[0], pair[1], EdgeKind::Fallthrough);
    }
    cfg.add_edge(protected[depth - 1], after, EdgeKind::Fallthrough);
    for index in 0..depth {
        add_try(&mut cfg, &protected[..depth - index], after);
    }
    cfg
}

/// A switch with one `goto` arm per level of `if (c0) { if (c1) { ... } }`:
/// the arm for level `i` jumps to the test `ci`, which the nesting also
/// reaches, so each test is emitted under its own label.
fn nested_goto_labels(depth: usize) -> Cfg<DfInst> {
    let mut cfg = Cfg::new();
    let dispatch = cfg.entry();
    fill(&mut cfg, dispatch);
    let tests: Vec<BlockId> = (0..depth).map(|_| new_block(&mut cfg)).collect();
    let leaf = new_block(&mut cfg);
    let merge = new_block(&mut cfg);
    for (index, &test) in tests.iter().enumerate() {
        let next = tests.get(index + 1).copied().unwrap_or(leaf);
        cfg.add_edge(test, next, EdgeKind::ConditionalTrue);
        cfg.add_edge(test, merge, EdgeKind::ConditionalFalse);
        let hop = new_block(&mut cfg);
        cfg.add_edge(dispatch, hop, EdgeKind::SwitchCase);
        cfg.add_edge(hop, test, EdgeKind::Jump);
    }
    cfg.add_edge(dispatch, tests[0], EdgeKind::Fallthrough);
    cfg.add_edge(leaf, merge, EdgeKind::Fallthrough);
    cfg
}

/// What a run over the tree counts.
struct Census {
    nodes: usize,
    deepest: usize,
}

/// Counts nodes and the deepest nesting of the walk.
struct Depth {
    open: usize,
    census: Census,
}

impl ControlVisitor for Depth {
    fn enter(&mut self, _: &ControlTree, _: ControlNodeId) -> ControlDescent {
        self.open += 1;
        self.census.nodes += 1;
        self.census.deepest = self.census.deepest.max(self.open);
        ControlDescent::Into
    }

    fn leave(&mut self, _: &ControlTree, _: ControlNodeId) {
        self.open -= 1;
    }
}

fn census(tree: &ControlTree) -> Census {
    let mut depth = Depth {
        open: 0,
        census: Census {
            nodes: 0,
            deepest: 0,
        },
    };
    tree.walk(&mut depth);
    assert_eq!(depth.open, 0, "every enter has its leave");
    depth.census
}

fn count_nodes(tree: &ControlTree, wanted: impl Fn(&ControlNode) -> bool) -> usize {
    tree.preorder().filter(|&id| wanted(tree.node(id))).count()
}

/// A sink that keeps only how many lines were written.
struct Lines(usize);

impl fmt::Write for Lines {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.0 += text.matches('\n').count();
        Ok(())
    }
}

/// The number of lines of the pseudocode of `tree`, the tree of `cfg`.
fn printed_lines(cfg: &Cfg<DfInst>, tree: &ControlTree) -> usize {
    let mut lines = Lines(0);
    tree.write_pseudocode(&mut lines, |sink, point| {
        cfg.block(point.block).instructions()[point.inst_idx].write_mnemonic(sink)
    })
    .expect("a line counter does not fail");
    lines.0
}

/// Print a tree of `levels` nested levels: at least a line per level.
fn assert_prints_a_line_per_level(cfg: &Cfg<DfInst>, tree: &ControlTree, levels: usize) {
    assert!(printed_lines(cfg, tree) >= levels);
}

/// Print the tree of the same shape at [`PRINTED_DEPTH`] levels.
fn print_shape(build: fn(usize) -> Cfg<DfInst>) {
    let cfg = build(PRINTED_DEPTH);
    let tree = ControlTree::compute(&cfg);
    assert_prints_a_line_per_level(&cfg, &tree, PRINTED_DEPTH);
}

/// Structure `cfg` and read the tree back the ways a consumer does. Returns
/// the tree, its report, and what the walk counted. The regionized tree of the
/// same graph is built and read as well.
fn structure_and_read(cfg: &Cfg<DfInst>) -> (ControlTree, LiftReport, Census) {
    let (tree, report) = ControlTree::compute_with_report(cfg);
    let census = census(&tree);
    assert_eq!(
        census.nodes,
        tree.node_count(),
        "the walk reaches every node"
    );
    assert_eq!(tree.preorder().count(), tree.node_count());

    let mut instructions = 0;
    tree.for_each_instruction(cfg, |_, _| instructions += 1);
    let total: usize = cfg
        .block_ids()
        .map(|block| cfg.block(block).instructions().len())
        .sum();
    assert_eq!(instructions, total, "every instruction once");

    // The same graph regionized: every block holds a predicated instruction,
    // so a guard wraps its run at every level of the nesting.
    let (guarded, guarded_report) = ControlTree::compute_predicated_with_report(cfg);
    assert_eq!(guarded_report, report);
    assert!(count_nodes(&guarded, |node| matches!(node, ControlNode::Guarded { .. })) > 0);
    let mut guarded_instructions = 0;
    guarded.for_each_instruction(cfg, |_, _| guarded_instructions += 1);
    assert_eq!(guarded_instructions, total, "every instruction once");
    (tree, report, census)
}

#[test]
fn a_hundred_thousand_nested_then_arms_structure_on_a_small_stack() {
    on_small_stack(|| {
        let cfg = nested_then_arms(LINEAR_DEPTH);
        let (tree, report, census) = structure_and_read(&cfg);
        assert!(report.is_fully_structured(), "{report:?}");
        let ifs = count_nodes(&tree, |node| matches!(node, ControlNode::IfThenElse { .. }));
        assert_eq!(ifs, LINEAR_DEPTH);
        assert!(census.deepest > LINEAR_DEPTH);
        print_shape(nested_then_arms);
    });
}

#[test]
fn a_hundred_thousand_else_ifs_structure_on_a_small_stack() {
    on_small_stack(|| {
        let cfg = else_if_chain(LINEAR_DEPTH);
        let (tree, report, census) = structure_and_read(&cfg);
        assert!(report.is_fully_structured(), "{report:?}");
        let ifs = count_nodes(&tree, |node| matches!(node, ControlNode::IfThenElse { .. }));
        assert_eq!(ifs, LINEAR_DEPTH);
        assert!(census.deepest > LINEAR_DEPTH);
        print_shape(else_if_chain);
    });
}

#[test]
fn a_hundred_thousand_ifs_closed_by_a_loop_structure_on_a_small_stack() {
    on_small_stack(|| {
        let cfg = ifs_closed_by_a_loop(LINEAR_DEPTH);
        let (tree, report, census) = structure_and_read(&cfg);
        assert!(report.is_fully_structured(), "{report:?}");
        let ifs = count_nodes(&tree, |node| matches!(node, ControlNode::IfThenElse { .. }));
        assert_eq!(ifs, LINEAR_DEPTH);
        let loops = count_nodes(&tree, |node| matches!(node, ControlNode::Loop { .. }));
        assert_eq!(loops, 1);
        assert!(census.deepest > LINEAR_DEPTH);
        print_shape(ifs_closed_by_a_loop);
    });
}

#[test]
fn a_hundred_thousand_blocks_in_a_row_structure_on_a_small_stack() {
    on_small_stack(|| {
        let cfg = straight_chain(LINEAR_DEPTH);
        let (tree, report, _) = structure_and_read(&cfg);
        assert!(report.is_fully_structured(), "{report:?}");
        let leaves = count_nodes(&tree, |node| {
            matches!(node, ControlNode::Block(_) | ControlNode::Return(_))
        });
        assert_eq!(leaves, LINEAR_DEPTH);
        print_shape(straight_chain);
    });
}

#[test]
fn a_hundred_thousand_blocks_in_a_loop_test_structure_on_a_small_stack() {
    on_small_stack(|| {
        let cfg = loop_test_chain(LINEAR_DEPTH);
        let (tree, report, _) = structure_and_read(&cfg);
        assert!(report.is_fully_structured(), "{report:?}");
        let tests: Vec<usize> = tree
            .preorder()
            .filter_map(|id| match *tree.node(id) {
                ControlNode::Loop {
                    kind: LoopKind::While { condition, .. },
                    ..
                } => Some(tree.spans(condition).len()),
                _ => None,
            })
            .collect();
        assert_eq!(tests, [LINEAR_DEPTH], "one pre-tested loop holds the chain");
        print_shape(loop_test_chain);
    });
}

#[test]
fn deeply_nested_loops_structure_on_a_small_stack() {
    on_small_stack(|| {
        let cfg = nested_loops(LOOP_DEPTH);
        let (tree, report, census) = structure_and_read(&cfg);
        assert!(report.is_fully_structured(), "{report:?}");
        let whiles = count_nodes(&tree, |node| {
            matches!(
                node,
                ControlNode::Loop {
                    kind: LoopKind::While { .. },
                    ..
                }
            )
        });
        assert_eq!(whiles, LOOP_DEPTH);
        assert!(census.deepest > LOOP_DEPTH);
        assert_prints_a_line_per_level(&cfg, &tree, LOOP_DEPTH);
    });
}

#[test]
fn deeply_nested_switches_structure_on_a_small_stack() {
    on_small_stack(|| {
        let cfg = nested_switches(LINEAR_DEPTH);
        let (tree, report, census) = structure_and_read(&cfg);
        assert!(report.is_fully_structured(), "{report:?}");
        let switches = count_nodes(&tree, |node| matches!(node, ControlNode::Switch { .. }));
        assert_eq!(switches, LINEAR_DEPTH);
        assert!(census.deepest > LINEAR_DEPTH);
        print_shape(nested_switches);
    });
}

#[test]
fn deeply_nested_tries_structure_on_a_small_stack() {
    on_small_stack(|| {
        let cfg = nested_tries(TRY_DEPTH);
        let (tree, report, census) = structure_and_read(&cfg);
        assert!(report.is_fully_structured(), "{report:?}");
        let tries = count_nodes(&tree, |node| matches!(node, ControlNode::TryCatch { .. }));
        assert_eq!(tries, TRY_DEPTH);
        assert!(census.deepest > TRY_DEPTH);
        assert_prints_a_line_per_level(&cfg, &tree, TRY_DEPTH);
    });
}

#[test]
fn deeply_nested_tries_at_one_anchor_structure_on_a_small_stack() {
    on_small_stack(|| {
        let cfg = nested_tries_at_one_anchor(TRY_DEPTH);
        let (tree, report, census) = structure_and_read(&cfg);
        // Every region but the outermost resumes at the block after the
        // region inside it, so its pad leaves the tries around it by a goto to
        // the block they all end at.
        assert_eq!(report.unstructured_regions, []);
        assert_eq!(report.unresolved_labels, []);
        assert_eq!(report.gotos.len(), TRY_DEPTH - 1);
        let tries = count_nodes(&tree, |node| matches!(node, ControlNode::TryCatch { .. }));
        assert_eq!(tries, TRY_DEPTH);
        assert!(census.deepest > TRY_DEPTH);
        assert_prints_a_line_per_level(&cfg, &tree, TRY_DEPTH);
    });
}

#[test]
fn deeply_nested_goto_labels_structure_on_a_small_stack() {
    on_small_stack(|| {
        let cfg = nested_goto_labels(LINEAR_DEPTH);
        let (tree, report, census) = structure_and_read(&cfg);
        assert!(report.unresolved_labels.is_empty(), "{report:?}");
        let labels = count_nodes(&tree, |node| matches!(node, ControlNode::Label { .. }));
        assert_eq!(labels, LINEAR_DEPTH);
        assert!(census.deepest > LINEAR_DEPTH);
        print_shape(nested_goto_labels);
    });
}
