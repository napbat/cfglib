//! Re-linearization — serialize a CFG back to a flat instruction stream.
//!
//! The [`linearize`] function sorts blocks according to a chosen
//! [`BlockOrder`], then emits block-start markers, instructions, and
//! explicit jumps/branches so that the resulting instruction sequence is
//! semantically equivalent to the graph.
//!
//! Because cfglib is target-agnostic, the caller must provide an
//! [`Emitter`] that knows how to create jump/branch/marker instructions
//! for the target form.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;

/// How to order blocks in the output stream.
#[derive(Debug, Clone)]
pub enum BlockOrder {
    /// Reverse-postorder (good for structured code).
    ReversePostorder,
    /// Allocation order (block ids in ascending order).
    AllocationOrder,
    /// Caller-specified order.
    Custom(Vec<BlockId>),
}

/// A single instruction in the linearized output.
#[derive(Debug, Clone)]
pub struct LinearInst<I> {
    /// The instruction.
    pub inst: I,
    /// Which block this instruction came from.
    pub block: BlockId,
    /// Index within the block (label/jump synthetics use `usize::MAX`).
    pub index: usize,
}

/// IR or language adapter for emitting jump, branch, and block-start
/// instructions.
///
/// cfglib does not know how to create consumer instructions, so the frontend
/// implements this trait. Targets are [`BlockId`]s — the emitter derives its
/// own naming (assembly labels, source constructs, nothing at all); padding
/// and alignment are applied by the consumer to the returned stream.
pub trait Emitter<I> {
    /// Emit an unconditional branch to `target`.
    fn emit_jump(&self, target: BlockId) -> I;

    /// Emit a conditional branch to `target`.
    ///
    /// `condition` is the last instruction of the source block (the
    /// terminating branch instruction). The emitter can inspect it to
    /// determine the condition encoding.
    fn emit_conditional_branch(&self, condition: &I, target: BlockId) -> I;

    /// Emit a marker naming `block`, or `None` when the target form needs
    /// no explicit labels (source code, structured IRs).
    fn emit_block_start(&self, block: BlockId) -> Option<I>;
}

/// Linearize a CFG into a flat instruction stream.
///
/// Blocks are laid out in the specified [`BlockOrder`]. For each
/// block the function:
///
/// 1. Emits a block-start marker (via [`Emitter::emit_block_start`]),
///    when the emitter produces one.
/// 2. Emits the block's instructions in order.
/// 3. If the block's layout successor is not its fallthrough target,
///    emits an explicit jump or branch.
///
/// Returns the instruction stream as a `Vec<LinearInst<I>>`.
pub fn linearize<I: Clone>(
    cfg: &Cfg<I>,
    order: BlockOrder,
    emitter: &dyn Emitter<I>,
) -> Vec<LinearInst<I>> {
    let sorted: Vec<BlockId> = match order {
        BlockOrder::ReversePostorder => cfg.reverse_postorder(),
        BlockOrder::AllocationOrder => cfg
            .blocks()
            .iter()
            .map(super::super::block::BasicBlock::id)
            .collect(),
        BlockOrder::Custom(ids) => ids,
    };

    let mut out: Vec<LinearInst<I>> = Vec::new();

    for (pos, &id) in sorted.iter().enumerate() {
        let block = cfg.block(id);

        // 1. Optional block-start marker.
        if let Some(marker) = emitter.emit_block_start(id) {
            out.push(LinearInst {
                inst: marker,
                block: id,
                index: usize::MAX,
            });
        }

        // 2. Block instructions.
        for (idx, inst) in block.instructions().iter().enumerate() {
            out.push(LinearInst {
                inst: inst.clone(),
                block: id,
                index: idx,
            });
        }

        // 3. Determine whether we need an explicit jump.
        let next_in_layout = if pos + 1 < sorted.len() {
            Some(sorted[pos + 1])
        } else {
            None
        };

        let succ_edges: Vec<_> = cfg
            .successor_edges(id)
            .iter()
            .map(|&eid| cfg.edge(eid))
            .collect();

        emit_tail_jump(cfg, id, &succ_edges, next_in_layout, emitter, &mut out);
    }

    out
}

/// Returns `true` if `kind` represents a fallthrough-like edge that
/// can be satisfied by layout adjacency (no explicit jump needed).
fn is_fallthrough_kind(kind: EdgeKind) -> bool {
    matches!(
        kind,
        EdgeKind::Fallthrough | EdgeKind::ConditionalFalse | EdgeKind::CallReturn
    )
}

/// Emit trailing jump/branch if the fallthrough doesn't reach the
/// intended successor.
fn emit_tail_jump<I: Clone>(
    cfg: &Cfg<I>,
    id: BlockId,
    succ_edges: &[&crate::edge::Edge],
    next_in_layout: Option<BlockId>,
    emitter: &dyn Emitter<I>,
    out: &mut Vec<LinearInst<I>>,
) {
    if succ_edges.is_empty() {
        return; // No successors — return/terminate block.
    }

    // Partition edges into the fallthrough candidate and everything else.
    // At most one edge can be a fallthrough (satisfied by layout adjacency).
    let fallthrough = succ_edges.iter().find(|e| is_fallthrough_kind(e.kind()));
    let branches: Vec<_> = succ_edges
        .iter()
        .filter(|e| !is_fallthrough_kind(e.kind()))
        .collect();

    // Emit explicit jumps/branches for all non-fallthrough edges.
    let last_inst = cfg.block(id).instructions().last();

    for edge in &branches {
        match edge.kind() {
            // Conditional edges → emit a conditional branch.
            EdgeKind::ConditionalTrue => {
                if let Some(cond) = last_inst {
                    out.push(LinearInst {
                        inst: emitter.emit_conditional_branch(cond, edge.target()),
                        block: id,
                        index: usize::MAX,
                    });
                }
            }
            // Everything else (Jump, SwitchCase, ExceptionHandler, etc.)
            // → emit an unconditional jump.
            _ => {
                out.push(LinearInst {
                    inst: emitter.emit_jump(edge.target()),
                    block: id,
                    index: usize::MAX,
                });
            }
        }
    }

    // Handle the fallthrough edge: emit a jump only if the layout
    // successor is not the fallthrough target.
    if let Some(ft) = fallthrough.filter(|ft| next_in_layout != Some(ft.target())) {
        out.push(LinearInst {
            inst: emitter.emit_jump(ft.target()),
            block: id,
            index: usize::MAX,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::{MockInst, ff};
    use alloc::vec;

    /// A trivial emitter that produces string-based mock instructions.
    struct TestEmitter;

    impl Emitter<MockInst> for TestEmitter {
        fn emit_jump(&self, _target: BlockId) -> MockInst {
            MockInst(crate::flow::FlowEffect::Fallthrough, "jump")
        }
        fn emit_conditional_branch(&self, _cond: &MockInst, _target: BlockId) -> MockInst {
            MockInst(crate::flow::FlowEffect::Fallthrough, "branch")
        }
        fn emit_block_start(&self, _block: BlockId) -> Option<MockInst> {
            Some(MockInst(crate::flow::FlowEffect::Fallthrough, "label"))
        }
    }

    /// Collect all mnemonic names from the linearized output.
    fn mnemonics(out: &[LinearInst<MockInst>]) -> Vec<&'static str> {
        out.iter().map(|li| li.inst.1).collect()
    }

    #[test]
    fn linearize_single_block() {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("a"));
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("b"));
        let out = linearize(&cfg, BlockOrder::AllocationOrder, &TestEmitter);
        let names = mnemonics(&out);
        // Should be: label, a, b
        assert_eq!(names, vec!["label", "a", "b"]);
    }

    #[test]
    fn linearize_two_blocks_with_fallthrough() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("a"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let out = linearize(&cfg, BlockOrder::AllocationOrder, &TestEmitter);
        let names = mnemonics(&out);
        // Should be: label, a, label, b — no jump needed (fallthrough).
        assert_eq!(names, vec!["label", "a", "label", "b"]);
    }

    #[test]
    fn linearize_non_fallthrough_emits_jump() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        let c = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("a"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.block_mut(c).instructions_vec_mut().push(ff("c"));
        // entry → c (Jump), layout order is entry, b, c.
        cfg.add_edge(cfg.entry(), c, EdgeKind::Jump);
        let out = linearize(&cfg, BlockOrder::AllocationOrder, &TestEmitter);
        let names = mnemonics(&out);
        // entry's successor is c but layout next is b → needs jump.
        assert!(names.contains(&"jump"), "should emit jump: {names:?}");
    }

    #[test]
    fn linearize_conditional_branch() {
        let mut cfg = Cfg::new();
        let t = cfg.new_block();
        let f = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("cmp"));
        cfg.block_mut(t).instructions_vec_mut().push(ff("then"));
        cfg.block_mut(f).instructions_vec_mut().push(ff("else"));
        cfg.add_edge(cfg.entry(), t, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), f, EdgeKind::ConditionalFalse);
        let out = linearize(&cfg, BlockOrder::AllocationOrder, &TestEmitter);
        let names = mnemonics(&out);
        // Should have a conditional branch for the true edge.
        assert!(names.contains(&"branch"), "should emit branch: {names:?}");
    }

    #[test]
    fn linearize_rpo_order() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("entry"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let out = linearize(&cfg, BlockOrder::ReversePostorder, &TestEmitter);
        // In RPO for a linear chain, entry comes first.
        assert_eq!(out[0].block, cfg.entry());
    }

    #[test]
    fn linearize_custom_order() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("entry"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        // Reverse order: b first, then entry.
        let out = linearize(
            &cfg,
            BlockOrder::Custom(alloc::vec![b, cfg.entry()]),
            &TestEmitter,
        );
        assert_eq!(out[0].block, b);
    }
}
