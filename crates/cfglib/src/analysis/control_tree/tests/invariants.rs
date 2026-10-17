//! The invariants every control tree keeps, whatever the graph.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeId;
use crate::{
    ControlDescent, ControlNode, ControlNodeId, ControlTree, ControlVisitor, LiftReport, LoopKind,
};

/// The blocks whose instructions a tree must hold: the blocks the entry
/// reaches, and the blocks the handlers own — handler entries, filter
/// funclets, and the blocks of known handler extents. Sorted.
///
/// A protected block the entry cannot reach is dead code and is not owned.
pub(super) fn owned_blocks<I, E>(cfg: &Cfg<I, E>) -> Vec<BlockId> {
    let mut owned = reach(cfg, Vec::new());
    owned.extend(handler_owned(cfg));
    owned.sort_unstable();
    owned.dedup();
    owned
}

/// The blocks a tree may hold beyond the owned ones: everything the owned
/// blocks reach over edges of every kind, which a walk can inline;
/// everything the goto targets of `report` reach, which the sweep emits so
/// that the label resolves; and the code a region continues at, since the
/// walk resumes there even when only a dead protected block leads to it.
/// Sorted.
fn allowed_blocks<I, E>(cfg: &Cfg<I, E>, report: &LiftReport) -> Vec<BlockId> {
    let mut roots = handler_owned(cfg);
    roots.extend(report.gotos.iter().map(|goto| goto.target));
    for region in cfg.regions() {
        let inside: BTreeSet<BlockId> = region
            .protected_blocks
            .iter()
            .copied()
            .chain(
                region
                    .handlers
                    .iter()
                    .flat_map(|handler| handler.body.blocks().into_iter().flatten().copied()),
            )
            .chain(region.handlers.iter().map(|handler| handler.entry))
            .collect();
        for &block in &inside {
            roots.extend(
                cfg.outgoing(block)
                    .map(|edge| cfg.edge(edge))
                    .filter(|edge| !edge.kind().is_exceptional())
                    .map(|edge| edge.target())
                    .filter(|target| !inside.contains(target)),
            );
        }
    }
    reach(cfg, roots)
}

/// What the entry and `roots` reach over edges of every kind. Sorted.
pub(super) fn reach<I, E>(cfg: &Cfg<I, E>, mut stack: Vec<BlockId>) -> Vec<BlockId> {
    stack.push(cfg.entry());
    let mut reached = alloc::vec![false; cfg.block_bound()];
    while let Some(block) = stack.pop() {
        if !core::mem::replace(&mut reached[block.index()], true) {
            stack.extend(cfg.successors(block));
        }
    }
    cfg.block_ids()
        .filter(|block| reached[block.index()])
        .collect()
}

/// The handler entries, filter funclets, and known extent blocks.
fn handler_owned<I, E>(cfg: &Cfg<I, E>) -> Vec<BlockId> {
    let mut owned = Vec::new();
    for handler in cfg.regions().iter().flat_map(|region| &region.handlers) {
        owned.push(handler.entry);
        if let crate::HandlerKind::Filter { filter_block } = handler.kind {
            owned.push(filter_block);
        }
        owned.extend(handler.body.blocks().into_iter().flatten().copied());
    }
    owned
}

/// The block sets a handler owns: its entry with its known extent, and its
/// filter funclet by itself.
pub(super) fn handler_sets<I, E>(cfg: &Cfg<I, E>) -> Vec<BTreeSet<BlockId>> {
    let mut sets = Vec::new();
    for handler in cfg.regions().iter().flat_map(|region| &region.handlers) {
        let mut owned: BTreeSet<BlockId> = [handler.entry].into_iter().collect();
        owned.extend(handler.body.blocks().into_iter().flatten().copied());
        sets.push(owned);
        if let crate::HandlerKind::Filter { filter_block } = handler.kind {
            sets.push([filter_block].into_iter().collect());
        }
    }
    sets
}

/// Remove every ordinary edge that enters a handler's blocks from outside
/// them, so a pad is reached only by an exception edge or by its own extent.
///
/// A pad that ordinary flow also reaches is not a well-formed handler: the
/// same block would be both a loop's latch and a handler's entry.
pub(super) fn isolate_handlers<I, E>(cfg: &mut Cfg<I, E>) {
    for owned in handler_sets(cfg) {
        let offending: Vec<EdgeId> = owned
            .iter()
            .flat_map(|&block| cfg.incoming(block))
            .filter(|&edge| {
                let edge = cfg.edge(edge);
                !edge.kind().is_exceptional() && !owned.contains(&edge.source())
            })
            .collect();
        for edge in offending {
            cfg.remove_edge(edge);
        }
    }
}

/// A compact description of a graph for a failure message: its edges and
/// regions.
pub(super) fn describe<I, E>(cfg: &Cfg<I, E>) -> alloc::string::String {
    use core::fmt::Write as _;
    let mut text = alloc::string::String::new();
    for block in cfg.block_ids() {
        for edge in cfg.outgoing(block) {
            let edge = cfg.edge(edge);
            let _ = write!(
                text,
                "{}->{} {:?}; ",
                block.index(),
                edge.target().index(),
                edge.kind()
            );
        }
    }
    for region in cfg.regions() {
        let _ = write!(text, "region guarding {:?}: ", region.protected_blocks);
        for handler in &region.handlers {
            let _ = write!(
                text,
                "{:?} {:?} {:?}; ",
                handler.kind,
                handler.entry.index(),
                handler.body
            );
        }
    }
    text
}

/// The tree as pseudocode with each instruction named by its block, for a
/// failure message.
pub(super) fn render(tree: &ControlTree) -> alloc::string::String {
    tree.to_pseudocode(|sink, point| write!(sink, "b{}.{}", point.block.index(), point.inst_idx))
        .unwrap_or_default()
}

/// Records the nodes a walk enters, and checks that every `leave` closes the
/// innermost open node.
#[derive(Default)]
struct Balance {
    open: Vec<ControlNodeId>,
    entered: Vec<ControlNodeId>,
}

impl ControlVisitor for Balance {
    fn enter(&mut self, _: &ControlTree, node: ControlNodeId) -> ControlDescent {
        self.open.push(node);
        self.entered.push(node);
        ControlDescent::Into
    }

    fn leave(&mut self, _: &ControlTree, node: ControlNodeId) {
        assert_eq!(
            self.open.pop(),
            Some(node),
            "leave closes the innermost node"
        );
    }
}

/// The arena holds a tree: every node is reachable from the root exactly
/// once, each child has a lower handle than its parent, the root has the
/// highest, and a walk enters and leaves every node in step with the
/// preorder.
fn assert_arena_is_a_tree(tree: &ControlTree, context: &str) {
    let preorder: Vec<ControlNodeId> = tree.preorder().collect();
    let mut reached: Vec<usize> = preorder.iter().map(|id| id.index()).collect();
    reached.sort_unstable();
    assert_eq!(
        reached,
        (0..tree.node_count()).collect::<Vec<_>>(),
        "a node is unreachable or reached twice in {context}\n{}",
        render(tree)
    );
    assert_eq!(
        tree.root().index() + 1,
        tree.node_count(),
        "the root holds the highest handle in {context}"
    );
    for id in tree.node_ids() {
        for body in tree.child_bodies(id) {
            for &child in tree.body(body) {
                assert!(
                    child.index() < id.index(),
                    "child {child} of {id} is allocated after its parent in {context}\n{}",
                    render(tree)
                );
            }
        }
    }
    let mut balance = Balance::default();
    tree.walk(&mut balance);
    assert!(
        balance.open.is_empty(),
        "a walk left a node open in {context}"
    );
    assert_eq!(balance.entered, preorder, "walk order in {context}");
}

/// Check the invariants of one graph's tree and report.
///
/// - The arena is a tree (see [`assert_arena_is_a_tree`]).
/// - No instruction appears twice, and the instructions of every owned block
///   appear.
/// - No instruction appears that neither an owned block nor a goto reaches:
///   dead code stays dead.
/// - The gotos in the tree are exactly the reported ones.
/// - Every goto target has one label or is reported as a dangling label, never
///   both, and no block is labeled twice.
/// - Every swept block carries its label.
/// - The tree executes the instructions in the order the graph does.
pub(super) fn check<I, E>(cfg: &Cfg<I, E>, tree: &ControlTree, report: &LiftReport, context: &str) {
    assert_arena_is_a_tree(tree, context);

    let instructions = |blocks: Vec<BlockId>| -> Vec<(BlockId, usize)> {
        blocks
            .into_iter()
            .flat_map(|block| {
                (0..cfg.block(block).instructions().len()).map(move |index| (block, index))
            })
            .collect()
    };
    let mut seen = Vec::new();
    tree.for_each_instruction(cfg, |point, _| seen.push((point.block, point.inst_idx)));
    seen.sort_unstable();
    let mut once = seen.clone();
    once.dedup();
    assert_eq!(
        once,
        seen,
        "an instruction twice in {context}\n{}\n{report:?}",
        render(tree)
    );
    for instruction in instructions(owned_blocks(cfg)) {
        assert!(
            once.contains(&instruction),
            "instruction {instruction:?} lost in {context}\n{}\n{report:?}",
            render(tree)
        );
    }
    let allowed = instructions(allowed_blocks(cfg, report));
    for instruction in &once {
        assert!(
            allowed.contains(instruction),
            "dead instruction {instruction:?} emitted in {context}\n{}\n{report:?}",
            render(tree)
        );
    }

    let mut gotos: Vec<BlockId> = tree
        .preorder()
        .filter_map(|id| match *tree.node(id) {
            ControlNode::Goto { target } => Some(target),
            _ => None,
        })
        .collect();
    gotos.sort_unstable();
    let mut reported: Vec<BlockId> = report.gotos.iter().map(|goto| goto.target).collect();
    reported.sort_unstable();
    assert_eq!(gotos, reported, "gotos of {context}");

    let mut labels: Vec<BlockId> = tree
        .preorder()
        .filter_map(|id| match *tree.node(id) {
            ControlNode::Label { block, .. } => Some(block),
            _ => None,
        })
        .collect();
    labels.sort_unstable();
    let count = labels.len();
    labels.dedup();
    assert_eq!(
        labels.len(),
        count,
        "a block labeled twice in {context}\n{}\n{report:?}",
        render(tree)
    );
    for target in &reported {
        assert!(
            labels.contains(target) != report.unresolved_labels.contains(target),
            "label of goto {target} in {context}"
        );
    }
    // A goto to a block with instructions finds its label unless the tree
    // holds that block only as a loop's test. No node stands for an empty
    // block or for a test, so the report lists such a target and the consumer
    // falls back.
    let mut tests: Vec<BlockId> = Vec::new();
    for id in tree.preorder() {
        if let ControlNode::Loop { kind, .. } = *tree.node(id) {
            match kind {
                LoopKind::Endless => {}
                LoopKind::While { condition, .. } => {
                    tests.extend(tree.spans(condition).iter().map(|span| span.block()));
                }
                LoopKind::DoWhile { latch, .. } => tests.push(latch.block()),
            }
        }
    }
    for block in &report.unresolved_labels {
        assert!(
            reported.contains(block),
            "dangling label {block} without a goto in {context}"
        );
        assert!(
            cfg.block(*block).instructions().is_empty() || tests.contains(block),
            "goto to block {block} with instructions dangles in {context}\n{}\n{report:?}",
            render(tree)
        );
    }
    for block in &report.swept_blocks {
        assert!(labels.contains(block), "swept {block} in {context}");
    }
    // A region the report calls unstructured has no handler arm in the tree.
    for id in tree.preorder() {
        if let ControlNode::TryCatch { handlers, .. } = *tree.node(id) {
            for handler in tree.handlers(handlers) {
                assert!(
                    !report
                        .unstructured_regions
                        .contains(&handler.handler.region()),
                    "region {} is structured and reported unstructured in {context}\n{}",
                    handler.handler.region(),
                    render(tree)
                );
            }
        }
    }
    super::semantics::assert_runs_alike(cfg, tree, report, context);
    assert_eq!(
        report.is_fully_structured(),
        report.gotos.is_empty()
            && report.unstructured_regions.is_empty()
            && report.unresolved_labels.is_empty()
    );
}
