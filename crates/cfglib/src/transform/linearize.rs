//! Re-linearization — serialize a CFG back to a flat instruction stream.
//!
//! The [`linearize`] function sorts blocks according to a chosen
//! [`BlockOrder`], then emits block-start markers, instructions, and the
//! jumps and branches that the block order cannot satisfy by adjacency, so
//! that the resulting instruction sequence follows the graph's normal
//! control flow.
//!
//! Because cfglib is target-agnostic, the caller must provide an
//! [`Emitter`] that knows how to create jump/branch/marker instructions
//! for the target form.
//!
//! # What an edge asks of the stream
//!
//! - [`Fallthrough`](EdgeKind::Fallthrough),
//!   [`ConditionalFalse`](EdgeKind::ConditionalFalse), and
//!   [`CallReturn`](EdgeKind::CallReturn) are satisfied by layout
//!   adjacency and cost a jump only when the target is not next.
//! - [`ConditionalTrue`](EdgeKind::ConditionalTrue) is a conditional branch
//!   on the block's last instruction.
//! - [`Unconditional`](EdgeKind::Unconditional),
//!   [`Back`](EdgeKind::Back), [`Jump`](EdgeKind::Jump), and
//!   [`ExceptionLeave`](EdgeKind::ExceptionLeave) are jumps.
//! - [`SwitchCase`](EdgeKind::SwitchCase) and
//!   [`IndirectJump`](EdgeKind::IndirectJump) name the targets of the
//!   block's own dispatching instruction, which the stream already holds, so
//!   they emit nothing.
//! - [`Call`](EdgeKind::Call) and [`IndirectCall`](EdgeKind::IndirectCall)
//!   are performed by the call instruction itself, and the exceptional
//!   kinds (see [`EdgeKind::is_exceptional`]) by the unwinder, never by a
//!   jump, so they emit nothing either.

extern crate alloc;
use alloc::vec::Vec;
use core::fmt;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;

/// How to order blocks in the output stream.
#[derive(Debug, Clone)]
pub enum BlockOrder {
    /// Reverse-postorder from the entry block (good for structured code).
    /// Blocks unreachable from the entry are omitted.
    ReversePostorder,
    /// Allocation order (block ids in ascending order).
    AllocationOrder,
    /// Caller-specified order. Every id must name a live block, and blocks
    /// left out of the list are omitted from the output.
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

impl<I> LinearInst<I> {
    /// An instruction the emitter created on behalf of `block`.
    const fn synthetic(inst: I, block: BlockId) -> Self {
        Self {
            inst,
            block,
            index: usize::MAX,
        }
    }
}

/// Why a CFG cannot be linearized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearizeError {
    /// A block has a [`ConditionalTrue`](EdgeKind::ConditionalTrue) edge but
    /// no instruction, so the emitter has no condition to branch on.
    MissingCondition {
        /// The empty block.
        block: BlockId,
    },
}

impl fmt::Display for LinearizeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCondition { block } => write!(
                formatter,
                "block {block} has a conditional edge but no instruction to branch on"
            ),
        }
    }
}

impl core::error::Error for LinearizeError {}

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
/// 3. Emits the explicit jumps and branches its outgoing edges ask for (see
///    the [module documentation](self)), including a jump for a fallthrough
///    edge whose target is not the next block in the layout.
///
/// Returns the instruction stream as a `Vec<LinearInst<I>>`.
///
/// # Errors
///
/// Returns [`LinearizeError::MissingCondition`] when a block has a
/// [`ConditionalTrue`](EdgeKind::ConditionalTrue) edge but no instruction.
pub fn linearize<I: Clone, E>(
    cfg: &Cfg<I, E>,
    order: BlockOrder,
    emitter: &dyn Emitter<I>,
) -> Result<Vec<LinearInst<I>>, LinearizeError> {
    let sorted: Vec<BlockId> = match order {
        BlockOrder::ReversePostorder => cfg.reverse_postorder(),
        BlockOrder::AllocationOrder => cfg.block_ids().collect(),
        BlockOrder::Custom(ids) => ids,
    };

    let mut out: Vec<LinearInst<I>> = Vec::new();
    for (pos, &id) in sorted.iter().enumerate() {
        if let Some(marker) = emitter.emit_block_start(id) {
            out.push(LinearInst::synthetic(marker, id));
        }
        for (index, inst) in cfg.block(id).instructions().iter().enumerate() {
            out.push(LinearInst {
                inst: inst.clone(),
                block: id,
                index,
            });
        }
        emit_transfers(cfg, id, sorted.get(pos + 1).copied(), emitter, &mut out)?;
    }
    Ok(out)
}

/// What one outgoing edge requires of the instruction stream.
#[derive(Clone, Copy)]
enum Transfer {
    /// Layout adjacency, or a jump when the target is not next.
    Layout,
    /// A conditional branch on the block's last instruction.
    Branch,
    /// An unconditional jump.
    Jump,
    /// Nothing: the block's own instruction or the unwinder performs it.
    Implicit,
}

const fn transfer(kind: EdgeKind) -> Transfer {
    match kind {
        EdgeKind::Fallthrough | EdgeKind::ConditionalFalse | EdgeKind::CallReturn => {
            Transfer::Layout
        }
        EdgeKind::ConditionalTrue => Transfer::Branch,
        EdgeKind::Unconditional | EdgeKind::Back | EdgeKind::Jump | EdgeKind::ExceptionLeave => {
            Transfer::Jump
        }
        EdgeKind::SwitchCase
        | EdgeKind::IndirectJump
        | EdgeKind::Call
        | EdgeKind::IndirectCall
        | EdgeKind::ExceptionHandler
        | EdgeKind::ExceptionUnwind
        | EdgeKind::ExceptionResume
        | EdgeKind::ExceptionContinue => Transfer::Implicit,
    }
}

/// Emit the branches and jumps that `id`'s outgoing edges ask for.
fn emit_transfers<I: Clone, E>(
    cfg: &Cfg<I, E>,
    id: BlockId,
    next_in_layout: Option<BlockId>,
    emitter: &dyn Emitter<I>,
    out: &mut Vec<LinearInst<I>>,
) -> Result<(), LinearizeError> {
    let mut layout_target = None;
    for edge in cfg.outgoing(id).map(|edge| cfg.edge(edge)) {
        match transfer(edge.kind()) {
            Transfer::Layout => {
                layout_target.get_or_insert(edge.target());
            }
            Transfer::Branch => {
                let condition = cfg
                    .block(id)
                    .instructions()
                    .last()
                    .ok_or(LinearizeError::MissingCondition { block: id })?;
                let branch = emitter.emit_conditional_branch(condition, edge.target());
                out.push(LinearInst::synthetic(branch, id));
            }
            Transfer::Jump => {
                out.push(LinearInst::synthetic(emitter.emit_jump(edge.target()), id));
            }
            Transfer::Implicit => {}
        }
    }

    // A jump only when the layout successor is not the fallthrough target.
    if let Some(target) = layout_target
        && next_in_layout != Some(target)
    {
        out.push(LinearInst::synthetic(emitter.emit_jump(target), id));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::{MockInst, ff};
    use alloc::string::ToString;
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

    fn linear<E>(cfg: &Cfg<MockInst, E>, order: BlockOrder) -> Vec<LinearInst<MockInst>> {
        linearize(cfg, order, &TestEmitter).expect("every conditional block holds an instruction")
    }

    /// The mnemonics of the stream laid out in allocation order.
    fn allocation_order<E>(cfg: &Cfg<MockInst, E>) -> Vec<&'static str> {
        mnemonics(&linear(cfg, BlockOrder::AllocationOrder))
    }

    /// Collect all mnemonic names from the linearized output.
    fn mnemonics(out: &[LinearInst<MockInst>]) -> Vec<&'static str> {
        out.iter().map(|li| li.inst.1).collect()
    }

    #[test]
    fn linearize_single_block() {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("a"));
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("b"));
        // Should be: label, a, b
        assert_eq!(allocation_order(&cfg), vec!["label", "a", "b"]);
    }

    #[test]
    fn linearize_two_blocks_with_fallthrough() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("a"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        // Should be: label, a, label, b — no jump needed (fallthrough).
        assert_eq!(allocation_order(&cfg), vec!["label", "a", "label", "b"]);
    }

    #[test]
    fn linearize_non_fallthrough_emits_jump() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        let c = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("a"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.block_mut(c).instructions_mut().push(ff("c"));
        // entry → c (Jump), layout order is entry, b, c.
        cfg.add_edge(cfg.entry(), c, EdgeKind::Jump);
        let names = allocation_order(&cfg);
        // entry's successor is c but layout next is b → needs jump.
        assert!(names.contains(&"jump"), "should emit jump: {names:?}");
    }

    #[test]
    fn linearize_conditional_branch() {
        let mut cfg = Cfg::new();
        let t = cfg.new_block();
        let f = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(ff("cmp"));
        cfg.block_mut(t).instructions_mut().push(ff("then"));
        cfg.block_mut(f).instructions_mut().push(ff("else"));
        cfg.add_edge(cfg.entry(), t, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), f, EdgeKind::ConditionalFalse);
        let names = allocation_order(&cfg);
        // Should have a conditional branch for the true edge.
        assert!(names.contains(&"branch"), "should emit branch: {names:?}");
    }

    #[test]
    fn linearize_rpo_order() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(ff("entry"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let out = linear(&cfg, BlockOrder::ReversePostorder);
        // In RPO for a linear chain, entry comes first.
        assert_eq!(out[0].block, cfg.entry());
    }

    #[test]
    fn linearize_custom_order() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(ff("entry"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        // Reverse order: b first, then entry.
        let out = linear(&cfg, BlockOrder::Custom(alloc::vec![b, cfg.entry()]));
        assert_eq!(out[0].block, b);
    }

    #[test]
    fn exceptional_and_call_edges_emit_no_transfer() {
        let silent = EdgeKind::ALL
            .into_iter()
            .filter(|kind| kind.is_exceptional())
            .chain([EdgeKind::Call, EdgeKind::IndirectCall]);
        for kind in silent {
            let mut cfg = Cfg::new();
            let next = cfg.new_block();
            let target = cfg.new_block();
            cfg.block_mut(cfg.entry()).push(ff("a"));
            cfg.add_edge(cfg.entry(), next, EdgeKind::Fallthrough);
            cfg.add_edge(cfg.entry(), target, kind);
            assert_eq!(
                allocation_order(&cfg),
                ["label", "a", "label", "label"],
                "a `{kind}` edge is not a branch of the normal flow"
            );
        }
    }

    #[test]
    fn a_dispatch_edge_is_carried_by_the_block_not_re_encoded_as_jumps() {
        for kind in [EdgeKind::SwitchCase, EdgeKind::IndirectJump] {
            let mut cfg = Cfg::new();
            let arms = [cfg.new_block(), cfg.new_block(), cfg.new_block()];
            cfg.block_mut(cfg.entry()).push(ff("dispatch"));
            for arm in arms {
                cfg.add_edge(cfg.entry(), arm, kind);
            }
            assert_eq!(
                allocation_order(&cfg),
                ["label", "dispatch", "label", "label", "label"],
                "`{kind}` edges name the targets of the block's own dispatch"
            );
        }
    }

    #[test]
    fn a_conditional_edge_of_an_empty_block_has_no_condition_to_branch_on() {
        let mut cfg = Cfg::new();
        let taken = cfg.new_block();
        let other = cfg.new_block();
        cfg.block_mut(taken).push(ff("taken"));
        cfg.add_edge(cfg.entry(), taken, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), other, EdgeKind::ConditionalFalse);

        let error = linearize(&cfg, BlockOrder::AllocationOrder, &TestEmitter)
            .expect_err("the taken edge cannot be dropped silently");
        assert_eq!(
            error,
            LinearizeError::MissingCondition { block: cfg.entry() }
        );
        assert!(error.to_string().contains("no instruction to branch on"));
    }

    #[test]
    fn linearization_follows_consumer_edge_payload_graphs() {
        let mut cfg = Cfg::<MockInst, &str>::with_edge_payload();
        let far = cfg.new_block();
        let near = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("cmp"));
        cfg.add_edge_with_payload(cfg.entry(), far, EdgeKind::ConditionalTrue, "taken");
        cfg.add_edge_with_payload(cfg.entry(), near, EdgeKind::ConditionalFalse, "not taken");
        cfg.add_edge_with_payload(far, near, EdgeKind::Unconditional, "merge");

        // `near` is not next after the entry, so its fallthrough needs a jump.
        assert_eq!(
            allocation_order(&cfg),
            ["label", "cmp", "branch", "jump", "label", "jump", "label"]
        );
    }
}
