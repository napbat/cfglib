//! The invariants every control tree keeps, whatever the graph.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeId;
use crate::{ControlNode, ControlTree, LiftReport};

/// The blocks whose instructions a tree must hold: the blocks the entry
/// reaches, and the blocks a region claims — protected blocks, handler
/// entries, filter funclets, and the blocks of known handler extents — with
/// everything those reach, over edges of every kind. Sorted.
pub(super) fn owned_blocks<I, E>(cfg: &Cfg<I, E>) -> Vec<BlockId> {
    let mut owned = alloc::vec![false; cfg.block_bound()];
    let mut stack = alloc::vec![cfg.entry()];
    for region in cfg.regions() {
        stack.extend(region.protected_blocks.iter().copied());
        for handler in &region.handlers {
            stack.push(handler.entry);
            if let crate::HandlerKind::Filter { filter_block } = handler.kind {
                stack.push(filter_block);
            }
            stack.extend(handler.body.blocks().into_iter().flatten().copied());
        }
    }
    while let Some(block) = stack.pop() {
        if !core::mem::replace(&mut owned[block.index()], true) {
            stack.extend(cfg.successors(block));
        }
    }
    cfg.block_ids()
        .filter(|block| owned[block.index()])
        .collect()
}

/// The block sets a handler owns: its entry with its known extent, and its
/// filter funclet by itself.
fn handler_sets<I, E>(cfg: &Cfg<I, E>) -> Vec<BTreeSet<BlockId>> {
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
fn render(tree: &ControlTree) -> alloc::string::String {
    tree.to_pseudocode(|sink, point| write!(sink, "b{}.{}", point.block.index(), point.inst_idx))
        .unwrap_or_default()
}

/// Check the invariants of one graph's tree and report.
///
/// - Every node is reachable from the root exactly once.
/// - The instructions of every owned block appear exactly once, and no other
///   instruction appears.
/// - The gotos in the tree are exactly the reported ones.
/// - Every goto target has one label or is reported as a dangling label, never
///   both, and no block is labeled twice.
/// - Every swept block carries its label.
pub(super) fn check<I, E>(cfg: &Cfg<I, E>, tree: &ControlTree, report: &LiftReport, context: &str) {
    assert_eq!(tree.preorder().count(), tree.node_count(), "{context}");

    let mut seen = Vec::new();
    tree.for_each_instruction(cfg, |point, _| seen.push((point.block, point.inst_idx)));
    seen.sort_unstable();
    let mut expected: Vec<(BlockId, usize)> = owned_blocks(cfg)
        .into_iter()
        .flat_map(|block| {
            (0..cfg.block(block).instructions().len()).map(move |index| (block, index))
        })
        .collect();
    expected.sort_unstable();
    assert_eq!(
        seen,
        expected,
        "coverage of {context}\n{}\n{report:?}",
        render(tree)
    );

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
    for block in &report.unresolved_labels {
        assert!(
            reported.contains(block),
            "dangling label {block} without a goto in {context}"
        );
    }
    for block in &report.swept_blocks {
        assert!(labels.contains(block), "swept {block} in {context}");
    }
    assert_eq!(
        report.is_fully_structured(),
        report.gotos.is_empty()
            && report.unstructured_regions.is_empty()
            && report.unresolved_labels.is_empty()
    );
}
