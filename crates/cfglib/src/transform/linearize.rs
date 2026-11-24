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
//!   on the block's last instruction. When its target is the next block and
//!   the layout target is not, an emitter that
//!   [inverts the condition](Emitter::emit_inverted_conditional_branch)
//!   replaces the branch and the jump to the layout target with one inverted
//!   branch to the layout target.
//! - [`Unconditional`](EdgeKind::Unconditional),
//!   [`Back`](EdgeKind::Back), [`Jump`](EdgeKind::Jump), and
//!   [`ExceptionLeave`](EdgeKind::ExceptionLeave) are jumps. An emitter that
//!   [elides jumps to the next block](Emitter::elides_jump_to_next) omits
//!   the jump whose target is next.
//! - [`SwitchCase`](EdgeKind::SwitchCase) and
//!   [`IndirectJump`](EdgeKind::IndirectJump) name the targets of the
//!   block's own dispatching instruction, which the stream already holds, so
//!   they emit nothing.
//! - [`Call`](EdgeKind::Call) and [`IndirectCall`](EdgeKind::IndirectCall)
//!   are performed by the call instruction itself, and the exceptional
//!   kinds (see [`EdgeKind::is_exceptional`]) by the unwinder, never by a
//!   jump, so they emit nothing either.
//!
//! # Hot and cold sections
//!
//! [`hot_cold_order`](crate::hot_cold_order) stable-partitions a base order
//! by coldness class into a hot part and the cold sections. Linearizing each
//! part as its own [`BlockOrder::Custom`] gives one stream per section. The
//! last block of a part has no next block, so a layout edge that leaves the
//! part gets an explicit jump, and each stream can be placed anywhere.
//!
//! # Profile-guided order
//!
//! [`ext_tsp_order`](crate::ext_tsp_order) places blocks to maximize the
//! Ext-TSP score of a profile: an edge earns its full count when its target
//! follows its source and it can fall through, and a part of it when the
//! target is a short jump away. An edge can fall through when the stream
//! needs no transfer for it while its target is next: the layout kinds
//! always, `ConditionalTrue` when the emitter inverts the block's condition,
//! and the jump kinds when the emitter elides jumps to the next block.
//! [`cfg_ext_tsp_order`](crate::cfg_ext_tsp_order) derives that rule from
//! the emitter, and its result goes to [`linearize`] as a
//! [`BlockOrder::Custom`]. To split the function as well, pass the order to
//! [`hot_cold_order`](crate::hot_cold_order) as its base.

extern crate alloc;
use alloc::vec::Vec;
use core::fmt;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::cfg_view::CfgView;
use crate::edge::{EdgeId, EdgeKind, KindedEdge};

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

    /// Emit a conditional branch to `target` that is taken when the
    /// condition of `condition` is false, or `None` when the target form
    /// cannot invert that condition.
    ///
    /// [`linearize`] asks for it when the
    /// [`ConditionalTrue`](EdgeKind::ConditionalTrue) target of a block is
    /// the next block and the layout target is not. One inverted branch to
    /// the layout target then replaces the branch to the next block and the
    /// jump to the layout target. `condition` is the last instruction of the
    /// block, as for [`emit_conditional_branch`](Self::emit_conditional_branch).
    ///
    /// The default answers `None`, which keeps the branch and the jump.
    fn emit_inverted_conditional_branch(&self, _condition: &I, _target: BlockId) -> Option<I> {
        None
    }

    /// Whether a jump to the block placed right after its source may be
    /// omitted.
    ///
    /// When it answers `true`, [`linearize`] emits nothing for an edge of a
    /// jump kind ([`Unconditional`](EdgeKind::Unconditional),
    /// [`Back`](EdgeKind::Back), [`Jump`](EdgeKind::Jump),
    /// [`ExceptionLeave`](EdgeKind::ExceptionLeave)) whose target is the
    /// next block. The default answers `false`, which emits every such jump.
    fn elides_jump_to_next(&self) -> bool {
        false
    }
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
///    edge whose target is not the next block in the layout. The optional
///    [`Emitter`] hooks can invert a branch or omit a jump to the next
///    block; an emitter that keeps their defaults gets every branch and jump.
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
    let edges = || cfg.outgoing(id).map(|edge| cfg.edge(edge));
    // The first layout edge names the fallthrough target. It needs a jump
    // unless it is next, or an inverted branch reaches it.
    let mut layout_jump = edges()
        .find(|edge| matches!(transfer(edge.kind()), Transfer::Layout))
        .map(|edge| edge.target())
        .filter(|&target| next_in_layout != Some(target));
    for edge in edges() {
        match transfer(edge.kind()) {
            Transfer::Layout | Transfer::Implicit => {}
            Transfer::Branch => {
                let condition = cfg
                    .block(id)
                    .instructions()
                    .last()
                    .ok_or(LinearizeError::MissingCondition { block: id })?;
                let inverted = layout_jump
                    .filter(|_| next_in_layout == Some(edge.target()))
                    .and_then(|layout| emitter.emit_inverted_conditional_branch(condition, layout));
                let branch = if let Some(inverted) = inverted {
                    layout_jump = None;
                    inverted
                } else {
                    emitter.emit_conditional_branch(condition, edge.target())
                };
                out.push(LinearInst::synthetic(branch, id));
            }
            Transfer::Jump => {
                if !(emitter.elides_jump_to_next() && next_in_layout == Some(edge.target())) {
                    out.push(LinearInst::synthetic(emitter.emit_jump(edge.target()), id));
                }
            }
        }
    }

    if let Some(target) = layout_jump {
        out.push(LinearInst::synthetic(emitter.emit_jump(target), id));
    }
    Ok(())
}

/// Whether [`linearize`] under `emitter` emits no transfer for `edge` when
/// the target of the edge is the next block: the rule that a profile-guided
/// order scores fallthroughs by.
pub(crate) fn falls_through<G: CfgView>(
    cfg: &G,
    edge: EdgeId,
    emitter: &dyn Emitter<G::Inst>,
) -> bool {
    let edge = cfg.edge(edge);
    match transfer(edge.data().kind()) {
        Transfer::Layout => true,
        Transfer::Jump => emitter.elides_jump_to_next(),
        Transfer::Implicit => false,
        Transfer::Branch => {
            let source = edge.source();
            let layout = cfg
                .outgoing(source)
                .map(|other| cfg.edge(other))
                .find(|other| matches!(transfer(other.data().kind()), Transfer::Layout))
                .map(|other| other.target());
            match (cfg.instructions(source).last(), layout) {
                (Some(condition), Some(layout)) => {
                    layout != edge.target()
                        && emitter
                            .emit_inverted_conditional_branch(condition, layout)
                            .is_some()
                }
                _ => false,
            }
        }
    }
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

    /// An emitter that inverts every condition and elides jumps to the next
    /// block, and otherwise emits what [`TestEmitter`] does.
    struct FoldingEmitter;

    impl Emitter<MockInst> for FoldingEmitter {
        fn emit_jump(&self, target: BlockId) -> MockInst {
            TestEmitter.emit_jump(target)
        }
        fn emit_conditional_branch(&self, condition: &MockInst, target: BlockId) -> MockInst {
            TestEmitter.emit_conditional_branch(condition, target)
        }
        fn emit_block_start(&self, block: BlockId) -> Option<MockInst> {
            TestEmitter.emit_block_start(block)
        }
        fn emit_inverted_conditional_branch(
            &self,
            _condition: &MockInst,
            _target: BlockId,
        ) -> Option<MockInst> {
            Some(MockInst(crate::flow::FlowEffect::Fallthrough, "branch-not"))
        }
        fn elides_jump_to_next(&self) -> bool {
            true
        }
    }

    /// The entry branches to `[then, else]` on its true and false edges.
    fn two_way() -> (Cfg<MockInst>, [BlockId; 2]) {
        let mut cfg = Cfg::new();
        let arms = [cfg.new_block(), cfg.new_block()];
        cfg.block_mut(cfg.entry()).push(ff("cmp"));
        cfg.block_mut(arms[0]).push(ff("then"));
        cfg.block_mut(arms[1]).push(ff("else"));
        cfg.add_edge(cfg.entry(), arms[0], EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), arms[1], EdgeKind::ConditionalFalse);
        (cfg, arms)
    }

    fn folded(cfg: &Cfg<MockInst>, order: Vec<BlockId>) -> Vec<&'static str> {
        mnemonics(&linearize(cfg, BlockOrder::Custom(order), &FoldingEmitter).unwrap())
    }

    #[test]
    fn an_inverted_branch_replaces_the_branch_and_the_jump_when_the_true_target_is_next() {
        let (cfg, [then_arm, else_arm]) = two_way();
        let order = vec![cfg.entry(), then_arm, else_arm];
        assert_eq!(
            mnemonics(&linear(&cfg, BlockOrder::Custom(order.clone()))),
            [
                "label", "cmp", "branch", "jump", "label", "then", "label", "else"
            ],
            "the default emitter keeps today's two transfers"
        );
        assert_eq!(
            folded(&cfg, order),
            [
                "label",
                "cmp",
                "branch-not",
                "label",
                "then",
                "label",
                "else"
            ]
        );
    }

    #[test]
    fn a_branch_is_not_inverted_unless_only_the_true_target_is_next() {
        let (cfg, [then_arm, else_arm]) = two_way();
        assert_eq!(
            folded(&cfg, vec![cfg.entry(), else_arm, then_arm]),
            ["label", "cmp", "branch", "label", "else", "label", "then"],
            "the false target is next"
        );
        assert_eq!(
            folded(&cfg, vec![cfg.entry()]),
            ["label", "cmp", "branch", "jump"],
            "neither target is next"
        );
    }

    #[test]
    fn an_eliding_emitter_omits_only_the_jump_to_the_next_block() {
        let mut cfg = Cfg::new();
        let [next, far] = [cfg.new_block(), cfg.new_block()];
        cfg.add_edge(cfg.entry(), next, EdgeKind::Unconditional);
        cfg.add_edge(next, far, EdgeKind::Back);
        let order = vec![cfg.entry(), next];
        assert_eq!(
            mnemonics(&linear(&cfg, BlockOrder::Custom(order.clone()))),
            ["label", "jump", "label", "jump"]
        );
        assert_eq!(folded(&cfg, order), ["label", "label", "jump"]);
    }

    #[test]
    fn an_edge_falls_through_when_its_jump_to_the_next_block_is_not_emitted() {
        let (mut cfg, [then_arm, else_arm]) = two_way();
        let edge = |cfg: &Cfg<MockInst>, source, kind| {
            cfg.outgoing(source)
                .find(|&edge| cfg.edge(edge).kind() == kind)
                .expect("the block has an edge of that kind")
        };
        let true_edge = edge(&cfg, cfg.entry(), EdgeKind::ConditionalTrue);
        let false_edge = edge(&cfg, cfg.entry(), EdgeKind::ConditionalFalse);
        let jump = cfg.add_edge(then_arm, else_arm, EdgeKind::Jump);
        let case = cfg.add_edge(else_arm, then_arm, EdgeKind::SwitchCase);
        let lone = cfg.new_block();
        cfg.block_mut(lone).push(ff("cmp"));
        cfg.add_edge(lone, then_arm, EdgeKind::ConditionalTrue);
        let lone_true = edge(&cfg, lone, EdgeKind::ConditionalTrue);

        for emitter in [&TestEmitter as &dyn Emitter<MockInst>, &FoldingEmitter] {
            assert!(falls_through(&cfg, false_edge, emitter));
            assert!(!falls_through(&cfg, case, emitter), "a dispatch is taken");
            assert!(
                !falls_through(&cfg, lone_true, emitter),
                "no layout target to invert to"
            );
        }
        assert!(!falls_through(&cfg, true_edge, &TestEmitter));
        assert!(!falls_through(&cfg, jump, &TestEmitter));
        assert!(falls_through(&cfg, true_edge, &FoldingEmitter));
        assert!(falls_through(&cfg, jump, &FoldingEmitter));
    }
}
