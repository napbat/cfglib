//! What a tree covers: every instruction of every owned block appears under
//! exactly one node, however the structurer split the blocks.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use cfglib::{
    BlockId, BlockSpan, Cfg, ControlBody, ControlNode, ControlNodeId, ControlTree, LoopKind,
    ProgramPoint,
};

/// The blocks the entry reaches over every edge, by an explicit depth-first
/// search.
pub fn reachable_blocks<I, E>(cfg: &Cfg<I, E>) -> BTreeSet<BlockId> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![cfg.entry()];
    while let Some(block) = stack.pop() {
        if seen.insert(block) {
            stack.extend(cfg.successors(block));
        }
    }
    seen
}

/// The spans one node stands on itself, not counting its children: a leaf's
/// instructions, the condition of a branch or dispatch, a pre-tested loop's
/// condition chain, and a post-tested loop's latch.
pub fn spans_of(tree: &ControlTree, id: ControlNodeId) -> Vec<BlockSpan> {
    match *tree.node(id) {
        ControlNode::Block(span) | ControlNode::Return(span) => vec![span],
        ControlNode::IfThenElse { condition, .. } | ControlNode::Switch { condition, .. } => {
            vec![condition]
        }
        ControlNode::Loop {
            kind: LoopKind::While { condition, .. },
            ..
        } => tree.spans(condition).to_vec(),
        ControlNode::Loop {
            kind: LoopKind::DoWhile { latch, .. },
            ..
        } => vec![latch],
        _ => Vec::new(),
    }
}

/// Every span any node of the tree stands on.
pub fn leaf_spans(tree: &ControlTree) -> Vec<BlockSpan> {
    tree.preorder().flat_map(|id| spans_of(tree, id)).collect()
}

/// Every block a body stands on, descending into the bodies it holds.
pub fn blocks_under(tree: &ControlTree, body: ControlBody) -> BTreeSet<BlockId> {
    let mut blocks = BTreeSet::new();
    let mut pending = vec![body];
    while let Some(body) = pending.pop() {
        for &id in tree.body(body) {
            blocks.extend(spans_of(tree, id).into_iter().map(BlockSpan::block));
            pending.extend(tree.child_bodies(id));
        }
    }
    blocks
}

/// How many leaf spans name each block, counting only spans that hold an
/// instruction.
pub fn block_appearances(tree: &ControlTree) -> BTreeMap<BlockId, usize> {
    let mut appearances = BTreeMap::new();
    for span in leaf_spans(tree) {
        if !span.is_empty() {
            *appearances.entry(span.block()).or_insert(0) += 1;
        }
    }
    appearances
}

/// The partition property for a tree computed by `compute`: every block of
/// `covered` that holds an instruction appears in exactly one leaf, as a
/// whole-block span, and no other block appears.
pub fn assert_each_block_in_exactly_one_leaf<I, E>(
    cfg: &Cfg<I, E>,
    tree: &ControlTree,
    covered: &BTreeSet<BlockId>,
) {
    let expected: BTreeMap<BlockId, usize> = covered
        .iter()
        .filter(|&&block| !cfg.block(block).is_empty())
        .map(|&block| (block, 1))
        .collect();
    assert_eq!(block_appearances(tree), expected, "{tree:?}");
    for span in leaf_spans(tree) {
        if !span.is_empty() {
            assert_eq!(
                span.range(),
                0..cfg.block(span.block()).instructions().len(),
                "a plain tree names whole blocks"
            );
        }
    }
}

/// The instruction-level partition: the spans naming a block are disjoint and
/// tile `0..len` of its instructions exactly. A predicated tree splits a block
/// into several spans, so this is the property that survives it.
pub fn assert_spans_tile_each_block<I, E>(
    cfg: &Cfg<I, E>,
    tree: &ControlTree,
    covered: &BTreeSet<BlockId>,
) {
    let mut ranges: BTreeMap<BlockId, Vec<Range<usize>>> = BTreeMap::new();
    for span in leaf_spans(tree) {
        if !span.is_empty() {
            ranges.entry(span.block()).or_default().push(span.range());
        }
    }
    let named: BTreeSet<BlockId> = ranges.keys().copied().collect();
    let expected: BTreeSet<BlockId> = covered
        .iter()
        .copied()
        .filter(|&block| !cfg.block(block).is_empty())
        .collect();
    assert_eq!(named, expected, "{tree:?}");
    for (block, mut ranges) in ranges {
        ranges.sort_by_key(|range| range.start);
        let mut next = 0;
        for range in ranges {
            assert_eq!(range.start, next, "spans of {block} leave a gap or overlap");
            next = range.end;
        }
        assert_eq!(
            next,
            cfg.block(block).instructions().len(),
            "spans of {block} stop short of its last instruction"
        );
    }
}

/// `for_each_instruction` is a second, independent route over the same tree:
/// it must visit every instruction of `covered` exactly once, each paired with
/// the very instruction the graph holds at that point.
pub fn assert_instruction_walk_is_a_partition<I, E>(
    cfg: &Cfg<I, E>,
    tree: &ControlTree,
    covered: &BTreeSet<BlockId>,
) {
    let mut visited: Vec<ProgramPoint> = Vec::new();
    tree.for_each_instruction(cfg, |point, instruction| {
        let held: &I = &cfg.block(point.block).instructions()[point.inst_idx];
        assert!(
            std::ptr::eq(instruction, held),
            "{point} is paired with another instruction"
        );
        visited.push(point);
    });
    let unique: BTreeSet<ProgramPoint> = visited.iter().copied().collect();
    assert_eq!(
        unique.len(),
        visited.len(),
        "an instruction was visited twice"
    );
    let expected: BTreeSet<ProgramPoint> = covered
        .iter()
        .flat_map(|&block| {
            (0..cfg.block(block).instructions().len())
                .map(move |inst_idx| ProgramPoint { block, inst_idx })
        })
        .collect();
    assert_eq!(unique, expected);
}
