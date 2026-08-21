//! CFG builder — converts a flat instruction stream into a [`Cfg`]
//! using a scope-stack approach for structured control flow.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::flow::{FlowControl, FlowEffect, JumpTargets};

/// Error returned when the instruction stream contains mismatched or
/// unexpected structured control-flow markers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// An `else` was encountered without a matching `if` scope.
    UnmatchedElse,
    /// An `endif` was encountered without a matching `if` scope.
    UnmatchedEndIf,
    /// An `endswitch` was encountered without a matching `switch` scope.
    UnmatchedEndSwitch,
    /// A `case` / `default` was encountered outside a `switch` scope.
    UnmatchedSwitchCase,
    /// An `endloop` was encountered without a matching `loop` scope.
    UnmatchedEndLoop,
    /// A `break` was encountered outside any breakable scope.
    BreakOutsideScope,
    /// A `continue` was encountered outside any loop scope.
    ContinueOutsideLoop,
    /// Scopes were still open at the end of the instruction stream.
    UnclosedScopes {
        /// Number of scopes remaining.
        remaining: usize,
    },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnmatchedElse => write!(f, "encountered `else` without matching `if`"),
            Self::UnmatchedEndIf => write!(f, "encountered `endif` without matching `if`"),
            Self::UnmatchedEndSwitch => {
                write!(f, "encountered `endswitch` without matching `switch`")
            }
            Self::UnmatchedSwitchCase => write!(f, "encountered `case` outside a `switch`"),
            Self::UnmatchedEndLoop => write!(f, "encountered `endloop` without matching `loop`"),
            Self::BreakOutsideScope => write!(f, "`break` outside any breakable scope"),
            Self::ContinueOutsideLoop => write!(f, "`continue` outside any loop scope"),
            Self::UnclosedScopes { remaining } => {
                write!(f, "{remaining} unclosed scope(s) at end of input")
            }
        }
    }
}

/// Scope frames pushed onto the builder's stack to track structured regions.
enum Scope {
    /// An `if` / `else` / `endif` region.
    If {
        /// Block before the `if` (ends with the conditional branch).
        pre_block: BlockId,
        /// Blocks that need an edge to the merge point (end of each arm).
        arm_exits: Vec<BlockId>,
    },
    /// A `loop` / `endloop` region.
    Loop {
        /// The loop header block.
        header: BlockId,
        /// Blocks that break out of this loop (need edges to post-loop).
        break_exits: Vec<BlockId>,
    },
    /// A `switch` / `case` / `endswitch` region.
    ///
    /// `break` inside a switch exits the switch (jumps to post-merge),
    /// unlike `break` in a loop which exits the loop.
    Switch {
        /// Block before the `switch` (ends with the dispatch).
        pre_block: BlockId,
        /// Blocks that need an edge to the merge point (end of each case arm).
        arm_exits: Vec<BlockId>,
        /// Blocks that explicitly break out of this switch.
        break_exits: Vec<BlockId>,
    },
}

struct BuildState<I> {
    cfg: Cfg<I>,
    current: BlockId,
    scopes: Vec<Scope>,
}

impl<I: FlowControl> BuildState<I> {
    fn new() -> Self {
        let mut cfg = Cfg {
            blocks: Vec::new(),
            edges: Vec::new(),
            succs: Vec::new(),
            preds: Vec::new(),
            entry: BlockId(0),
            regions: Vec::new(),
            cleanups: Vec::new(),
        };
        let current = cfg.new_block();
        Self {
            cfg,
            current,
            scopes: Vec::new(),
        }
    }

    fn process(&mut self, instruction: I) -> Result<(), BuildError> {
        match instruction.flow_effect() {
            FlowEffect::Declaration
            | FlowEffect::Fallthrough
            | FlowEffect::Call
            | FlowEffect::ConditionalCall
            | FlowEffect::IndirectCall
            | FlowEffect::MayThrow => {
                self.push(instruction);
                Ok(())
            }
            FlowEffect::ConditionalOpen => {
                self.open_conditional(instruction);
                Ok(())
            }
            FlowEffect::ConditionalAlternate => self.alternate_conditional(instruction),
            FlowEffect::ConditionalClose => self.close_conditional(),
            FlowEffect::SwitchOpen => {
                self.open_switch(instruction);
                Ok(())
            }
            FlowEffect::SwitchCase => self.add_switch_case(instruction),
            FlowEffect::SwitchClose => self.close_switch(),
            FlowEffect::LoopOpen => {
                self.open_loop(instruction);
                Ok(())
            }
            FlowEffect::LoopClose => self.close_loop(),
            FlowEffect::Break => self.add_break(instruction),
            FlowEffect::ConditionalBreak => self.add_conditional_break(instruction),
            FlowEffect::Continue => self.add_continue(instruction),
            FlowEffect::ConditionalContinue => self.add_conditional_continue(instruction),
            FlowEffect::Return
            | FlowEffect::Terminate
            | FlowEffect::Jump
            | FlowEffect::IndirectJump => {
                self.end_block(instruction);
                Ok(())
            }
            FlowEffect::ConditionalReturn => {
                self.add_conditional_return(instruction);
                Ok(())
            }
            FlowEffect::Label => {
                self.add_label(instruction);
                Ok(())
            }
            FlowEffect::ConditionalJump => {
                self.add_conditional_jump(instruction);
                Ok(())
            }
        }
    }

    fn push(&mut self, instruction: I) {
        self.cfg
            .block_mut(self.current)
            .instructions
            .push(instruction);
    }

    fn end_block(&mut self, instruction: I) {
        self.push(instruction);
        self.current = self.cfg.new_block();
    }

    fn open_conditional(&mut self, instruction: I) {
        self.push(instruction);
        let true_block = self.cfg.new_block();
        self.cfg
            .add_edge(self.current, true_block, EdgeKind::ConditionalTrue);
        self.scopes.push(Scope::If {
            pre_block: self.current,
            arm_exits: Vec::new(),
        });
        self.current = true_block;
    }

    fn alternate_conditional(&mut self, instruction: I) -> Result<(), BuildError> {
        let Some(Scope::If {
            pre_block,
            arm_exits,
        }) = self.scopes.last_mut()
        else {
            return Err(BuildError::UnmatchedElse);
        };
        arm_exits.push(self.current);
        let alternate = self.cfg.new_block();
        self.cfg
            .add_edge(*pre_block, alternate, EdgeKind::ConditionalFalse);
        self.cfg.block_mut(alternate).instructions.push(instruction);
        self.current = alternate;
        Ok(())
    }

    fn close_conditional(&mut self) -> Result<(), BuildError> {
        let Some(Scope::If {
            pre_block,
            mut arm_exits,
        }) = self.scopes.pop()
        else {
            return Err(BuildError::UnmatchedEndIf);
        };
        let merge = self.cfg.new_block();
        arm_exits.push(self.current);
        for exit in arm_exits {
            self.cfg.add_edge(exit, merge, EdgeKind::Fallthrough);
        }
        let has_alternate = self
            .cfg
            .successor_edges(pre_block)
            .iter()
            .any(|&edge| self.cfg.edge(edge).kind() == EdgeKind::ConditionalFalse);
        if !has_alternate {
            self.cfg
                .add_edge(pre_block, merge, EdgeKind::ConditionalFalse);
        }
        self.current = merge;
        Ok(())
    }

    fn open_switch(&mut self, instruction: I) {
        self.push(instruction);
        let first_case = self.cfg.new_block();
        self.cfg
            .add_edge(self.current, first_case, EdgeKind::SwitchCase);
        self.scopes.push(Scope::Switch {
            pre_block: self.current,
            arm_exits: Vec::new(),
            break_exits: Vec::new(),
        });
        self.current = first_case;
    }

    fn add_switch_case(&mut self, instruction: I) -> Result<(), BuildError> {
        let Some(Scope::Switch {
            pre_block,
            arm_exits,
            ..
        }) = self.scopes.last_mut()
        else {
            return Err(BuildError::UnmatchedSwitchCase);
        };
        arm_exits.push(self.current);
        let case_block = self.cfg.new_block();
        self.cfg
            .add_edge(*pre_block, case_block, EdgeKind::SwitchCase);
        self.cfg
            .block_mut(case_block)
            .instructions
            .push(instruction);
        self.current = case_block;
        Ok(())
    }

    fn close_switch(&mut self) -> Result<(), BuildError> {
        let Some(Scope::Switch {
            mut arm_exits,
            break_exits,
            ..
        }) = self.scopes.pop()
        else {
            return Err(BuildError::UnmatchedEndSwitch);
        };
        let merge = self.cfg.new_block();
        arm_exits.push(self.current);
        for exit in arm_exits {
            self.cfg.add_edge(exit, merge, EdgeKind::Fallthrough);
        }
        for break_exit in break_exits {
            self.cfg
                .add_edge(break_exit, merge, EdgeKind::Unconditional);
        }
        self.current = merge;
        Ok(())
    }

    fn open_loop(&mut self, instruction: I) {
        self.push(instruction);
        let header = self.cfg.new_block();
        self.cfg
            .add_edge(self.current, header, EdgeKind::Fallthrough);
        self.scopes.push(Scope::Loop {
            header,
            break_exits: Vec::new(),
        });
        self.current = header;
    }

    fn close_loop(&mut self) -> Result<(), BuildError> {
        let Some(Scope::Loop {
            header,
            break_exits,
        }) = self.scopes.pop()
        else {
            return Err(BuildError::UnmatchedEndLoop);
        };
        self.cfg.add_edge(self.current, header, EdgeKind::Back);
        let post_loop = self.cfg.new_block();
        for break_exit in break_exits {
            self.cfg
                .add_edge(break_exit, post_loop, EdgeKind::Unconditional);
        }
        self.current = post_loop;
        Ok(())
    }

    fn record_break(&mut self, block: BlockId) -> bool {
        self.scopes.iter_mut().rev().any(|scope| match scope {
            Scope::Loop { break_exits, .. } | Scope::Switch { break_exits, .. } => {
                break_exits.push(block);
                true
            }
            Scope::If { .. } => false,
        })
    }

    fn add_break(&mut self, instruction: I) -> Result<(), BuildError> {
        self.push(instruction);
        if !self.record_break(self.current) {
            return Err(BuildError::BreakOutsideScope);
        }
        self.current = self.cfg.new_block();
        Ok(())
    }

    fn add_conditional_break(&mut self, instruction: I) -> Result<(), BuildError> {
        self.push(instruction);
        let break_block = self.cfg.new_block();
        let continuation = self.cfg.new_block();
        self.cfg
            .add_edge(self.current, break_block, EdgeKind::ConditionalTrue);
        self.cfg
            .add_edge(self.current, continuation, EdgeKind::ConditionalFalse);
        if !self.record_break(break_block) {
            return Err(BuildError::BreakOutsideScope);
        }
        self.current = continuation;
        Ok(())
    }

    fn loop_header(&self) -> Option<BlockId> {
        self.scopes.iter().rev().find_map(|scope| match scope {
            Scope::Loop { header, .. } => Some(*header),
            Scope::If { .. } | Scope::Switch { .. } => None,
        })
    }

    fn add_continue(&mut self, instruction: I) -> Result<(), BuildError> {
        self.push(instruction);
        let Some(header) = self.loop_header() else {
            return Err(BuildError::ContinueOutsideLoop);
        };
        self.cfg.add_edge(self.current, header, EdgeKind::Back);
        self.current = self.cfg.new_block();
        Ok(())
    }

    fn add_conditional_continue(&mut self, instruction: I) -> Result<(), BuildError> {
        self.push(instruction);
        let continuation = self.cfg.new_block();
        let Some(header) = self.loop_header() else {
            return Err(BuildError::ContinueOutsideLoop);
        };
        self.cfg
            .add_edge(self.current, header, EdgeKind::ConditionalTrue);
        self.cfg
            .add_edge(self.current, continuation, EdgeKind::ConditionalFalse);
        self.current = continuation;
        Ok(())
    }

    fn add_conditional_return(&mut self, instruction: I) {
        self.push(instruction);
        let return_block = self.cfg.new_block();
        let continuation = self.cfg.new_block();
        self.cfg
            .add_edge(self.current, return_block, EdgeKind::ConditionalTrue);
        self.cfg
            .add_edge(self.current, continuation, EdgeKind::ConditionalFalse);
        self.current = continuation;
    }

    fn add_label(&mut self, instruction: I) {
        // The fallthrough edge is unconditional: a label at program start
        // must stay connected to the entry, and a label after a conditional
        // jump IS the false-path continuation. When the current block is a
        // dead-code stub (fresh after a jump/return), the edge just leaves
        // it a predecessor-less empty trampoline, which is legal.
        let label_block = self.cfg.new_block();
        self.cfg
            .add_edge(self.current, label_block, EdgeKind::Fallthrough);
        self.cfg
            .block_mut(label_block)
            .instructions
            .push(instruction);
        self.current = label_block;
    }

    fn add_conditional_jump(&mut self, instruction: I) {
        self.push(instruction);
        let continuation = self.cfg.new_block();
        self.cfg
            .add_edge(self.current, continuation, EdgeKind::ConditionalFalse);
        self.current = continuation;
    }

    fn finish(mut self) -> Result<Cfg<I>, BuildError> {
        if !self.scopes.is_empty() {
            return Err(BuildError::UnclosedScopes {
                remaining: self.scopes.len(),
            });
        }
        CfgBuilder::trim_trailing_empty(&mut self.cfg);
        Ok(self.cfg)
    }
}

/// Builds a [`Cfg<I>`] from an iterator of instructions that implement
/// [`FlowControl`].
pub struct CfgBuilder;

impl CfgBuilder {
    /// Build a CFG from a flat instruction stream.
    ///
    /// Declarations ([`FlowEffect::Declaration`]) are stored in the
    /// current block but do not affect control flow.
    ///
    /// # Errors
    ///
    /// Returns an error if the instruction stream contains mismatched
    /// structured control-flow markers (e.g. `else` without `if`).
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{CfgBuilder, FlowControl, FlowEffect};
    ///
    /// #[derive(Debug, Clone)]
    /// struct Inst(FlowEffect);
    ///
    /// impl FlowControl for Inst {
    ///     fn flow_effect(&self) -> FlowEffect { self.0 }
    /// }
    ///
    /// let cfg = CfgBuilder::build(vec![
    ///     Inst(FlowEffect::Fallthrough),
    ///     Inst(FlowEffect::ConditionalOpen),
    ///     Inst(FlowEffect::Fallthrough),
    ///     Inst(FlowEffect::ConditionalAlternate),
    ///     Inst(FlowEffect::Fallthrough),
    ///     Inst(FlowEffect::ConditionalClose),
    ///     Inst(FlowEffect::Return),
    /// ]).unwrap();
    ///
    /// assert!(cfg.num_blocks() >= 4); // entry, then, else, merge
    /// ```
    pub fn build<I: FlowControl>(
        instructions: impl IntoIterator<Item = I>,
    ) -> Result<Cfg<I>, BuildError> {
        let mut state = BuildState::new();
        for instruction in instructions {
            state.process(instruction)?;
        }
        state.finish()
    }

    /// Remove empty blocks at the end that have no predecessors (dead code
    /// artefacts from the builder).
    fn trim_trailing_empty<I>(cfg: &mut Cfg<I>) {
        while cfg.blocks.len() > 1 {
            let last = BlockId::from_index(cfg.blocks.len() - 1);
            if cfg.block(last).is_empty() && cfg.predecessor_edges(last).is_empty() {
                cfg.blocks.pop();
                cfg.succs.pop();
                cfg.preds.pop();
            } else {
                break;
            }
        }
    }
}

/// Outcome of [`resolve_jump_edges`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JumpResolution<T> {
    /// Number of jump edges wired.
    pub resolved: usize,
    /// Jumps whose target token matched no label, as `(source block, target)`.
    pub unresolved: Vec<(BlockId, T)>,
}

/// Wire the edges described by explicit jump targets after
/// [`CfgBuilder::build`].
///
/// [`CfgBuilder::build`] ends a block at a [`FlowEffect::Jump`] /
/// [`FlowEffect::ConditionalJump`] instruction but cannot know where the jump
/// lands. This pass maps every block-leading [`FlowEffect::Label`]
/// instruction's [`JumpTargets::label`] token to its block, then adds an
/// [`EdgeKind::Jump`] edge for each unconditional jump and an
/// [`EdgeKind::ConditionalTrue`] edge for each conditional jump's taken path.
///
/// The pass is idempotent: an edge that already exists with the same
/// endpoints and kind is not duplicated, so re-running it is safe.
/// Duplicate label tokens resolve to the LAST block declaring them (later
/// labels shadow earlier ones); frontends with label scoping enforce
/// uniqueness before building.
pub fn resolve_jump_edges<I: JumpTargets>(cfg: &mut Cfg<I>) -> JumpResolution<I::Target> {
    let mut labels: BTreeMap<I::Target, BlockId> = BTreeMap::new();
    for block in cfg.blocks() {
        let Some(first) = block.instructions().first() else {
            continue;
        };
        if first.flow_effect() != FlowEffect::Label {
            continue;
        }
        if let Some(token) = first.label() {
            labels.insert(token, block.id());
        }
    }

    let mut pending: Vec<(BlockId, BlockId, EdgeKind)> = Vec::new();
    let mut resolution = JumpResolution {
        resolved: 0,
        unresolved: Vec::new(),
    };
    for block in cfg.blocks() {
        let Some(last) = block.instructions().last() else {
            continue;
        };
        let kind = match last.flow_effect() {
            FlowEffect::Jump => EdgeKind::Jump,
            FlowEffect::ConditionalJump => EdgeKind::ConditionalTrue,
            _ => continue,
        };
        let Some(token) = last.jump_target() else {
            continue;
        };
        if let Some(&target) = labels.get(&token) {
            pending.push((block.id(), target, kind));
        } else {
            resolution.unresolved.push((block.id(), token));
        }
    }

    for (source, target, kind) in pending {
        let already_wired = cfg
            .successor_edges(source)
            .iter()
            .any(|&edge| cfg.edge(edge).target() == target && cfg.edge(edge).kind() == kind);
        if !already_wired {
            cfg.add_edge(source, target, kind);
            resolution.resolved += 1;
        }
    }
    resolution
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{MockInst, ff};
    use alloc::vec;

    #[derive(Debug, Clone)]
    struct GotoInst {
        effect: FlowEffect,
        target: Option<&'static str>,
        label: Option<&'static str>,
    }

    fn gi(effect: FlowEffect) -> GotoInst {
        GotoInst {
            effect,
            target: None,
            label: None,
        }
    }

    fn goto(target: &'static str) -> GotoInst {
        GotoInst {
            effect: FlowEffect::Jump,
            target: Some(target),
            label: None,
        }
    }

    fn label(name: &'static str) -> GotoInst {
        GotoInst {
            effect: FlowEffect::Label,
            target: None,
            label: Some(name),
        }
    }

    impl FlowControl for GotoInst {
        fn flow_effect(&self) -> FlowEffect {
            self.effect
        }
    }

    impl JumpTargets for GotoInst {
        type Target = &'static str;

        fn jump_target(&self) -> Option<&'static str> {
            self.target
        }

        fn label(&self) -> Option<&'static str> {
            self.label
        }
    }

    #[test]
    fn leading_label_and_conditional_continuation_stay_connected() {
        // A label as the first instruction: the (empty) entry must still
        // reach it, or the whole function body is unreachable.
        let cfg = CfgBuilder::build(vec![
            label("head"),
            gi(FlowEffect::Fallthrough),
            gi(FlowEffect::Return),
        ])
        .unwrap();
        let reachable = cfg.dfs_preorder();
        assert_eq!(reachable.len(), cfg.num_blocks(), "all blocks reachable");

        // A label directly after a conditional jump IS the false-path
        // continuation; the edge must exist.
        let cfg = CfgBuilder::build(vec![
            GotoInst {
                effect: FlowEffect::ConditionalJump,
                target: Some("skip"),
                label: None,
            },
            label("skip"),
            gi(FlowEffect::Return),
        ])
        .unwrap();
        let reachable = cfg.dfs_preorder();
        assert_eq!(reachable.len(), cfg.num_blocks(), "false path connected");
    }

    #[test]
    fn resolve_wires_forward_goto() {
        let mut cfg = CfgBuilder::build(vec![
            gi(FlowEffect::Fallthrough),
            goto("exit"),
            gi(FlowEffect::Fallthrough), // dead code between jump and label
            label("exit"),
            gi(FlowEffect::Return),
        ])
        .unwrap();

        let resolution = resolve_jump_edges(&mut cfg);
        assert_eq!(resolution.resolved, 1);
        assert_eq!(resolution.unresolved.len(), 0);
        let jump_edge = cfg
            .edges()
            .find(|edge| edge.kind() == EdgeKind::Jump)
            .expect("goto edge wired");
        assert_eq!(jump_edge.source(), cfg.entry());
        let target_block = cfg.block(jump_edge.target());
        assert_eq!(target_block.instructions()[0].label, Some("exit"));
    }

    #[test]
    fn resolve_wires_backward_goto_and_conditional_taken_edge() {
        let mut cfg = CfgBuilder::build(vec![
            label("head"),
            gi(FlowEffect::Fallthrough),
            GotoInst {
                effect: FlowEffect::ConditionalJump,
                target: Some("head"),
                label: None,
            },
            gi(FlowEffect::Return),
        ])
        .unwrap();

        let resolution = resolve_jump_edges(&mut cfg);
        assert_eq!(resolution.resolved, 1);
        let taken = cfg
            .edges()
            .find(|edge| edge.kind() == EdgeKind::ConditionalTrue)
            .expect("taken edge wired");
        assert_eq!(
            cfg.block(taken.target()).instructions()[0].label,
            Some("head")
        );
        // The builder's fallthrough continuation edge is still present.
        assert!(
            cfg.edges()
                .any(|edge| edge.kind() == EdgeKind::ConditionalFalse)
        );
    }

    #[test]
    fn resolve_reports_unresolved_and_is_idempotent() {
        let mut cfg = CfgBuilder::build(vec![
            gi(FlowEffect::Fallthrough),
            goto("nowhere"),
            label("here"),
            GotoInst {
                effect: FlowEffect::Jump,
                target: Some("here"),
                label: None,
            },
        ])
        .unwrap();

        let first = resolve_jump_edges(&mut cfg);
        assert_eq!(first.resolved, 1);
        assert_eq!(first.unresolved.len(), 1);
        assert_eq!(first.unresolved[0].1, "nowhere");

        let second = resolve_jump_edges(&mut cfg);
        assert_eq!(second.resolved, 0, "already-wired edges are not duplicated");
        assert_eq!(second.unresolved.len(), 1);
    }

    #[test]
    fn linear_block() {
        let cfg = CfgBuilder::build(vec![ff("a"), ff("b"), ff("c")]).unwrap();
        assert_eq!(cfg.num_blocks(), 1);
        assert_eq!(cfg.num_edges(), 0);
        assert_eq!(cfg.block(cfg.entry()).instructions().len(), 3);
    }

    #[test]
    fn single_return() {
        let cfg = CfgBuilder::build(vec![ff("a"), MockInst(FlowEffect::Return, "ret")]).unwrap();
        // One block with instructions, trailing empty block trimmed.
        assert_eq!(cfg.num_blocks(), 1);
        assert_eq!(cfg.block(cfg.entry()).instructions().len(), 2);
    }

    #[test]
    fn if_endif_no_else() {
        // a; if; b; endif; c
        let cfg = CfgBuilder::build(vec![
            ff("a"),
            MockInst(FlowEffect::ConditionalOpen, "if"),
            ff("b"),
            MockInst(FlowEffect::ConditionalClose, "endif"),
            ff("c"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        // bb0: [a, if]
        // bb1: [b]  (true arm)
        // bb2: []   (merge — c, ret)
        assert!(cfg.num_blocks() >= 3);
        // Entry has two successors: true arm + false arm (merge).
        assert_eq!(cfg.successor_edges(cfg.entry()).len(), 2);
    }

    #[test]
    fn if_else_endif() {
        // a; if; b; else; c; endif; d; ret
        let cfg = CfgBuilder::build(vec![
            ff("a"),
            MockInst(FlowEffect::ConditionalOpen, "if"),
            ff("b"),
            MockInst(FlowEffect::ConditionalAlternate, "else"),
            ff("c"),
            MockInst(FlowEffect::ConditionalClose, "endif"),
            ff("d"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        // bb0: [a, if] → true(bb1), false(bb2)
        // bb1: [b]     → merge(bb3)
        // bb2: [else, c] → merge(bb3)
        // bb3: [d, ret]
        assert!(cfg.num_blocks() >= 4);
        assert_eq!(cfg.successor_edges(cfg.entry()).len(), 2);
    }

    #[test]
    fn simple_loop() {
        // loop; a; endloop; ret
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            ff("a"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        // bb0: [loop]       → fallthrough to bb1 (header)
        // bb1: [a]          → back to bb1 (header)
        // bb2: [ret]        (post-loop, unreachable without break)
        assert!(cfg.num_blocks() >= 2);
    }

    #[test]
    fn loop_with_break() {
        // loop; a; break; endloop; ret
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            ff("a"),
            MockInst(FlowEffect::Break, "break"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        // post-loop block should be reachable from the break.
        assert!(cfg.num_blocks() >= 3);
    }

    #[test]
    fn loop_with_conditional_break() {
        // loop; a; breakc; b; endloop; ret
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            ff("a"),
            MockInst(FlowEffect::ConditionalBreak, "breakc"),
            ff("b"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        // The breakc block should have two successors:
        // - true → break_block (which goes to post-loop)
        // - false → continue block (with b)
        assert!(cfg.num_blocks() >= 4);
    }

    #[test]
    fn declarations_are_stored() {
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::Declaration, "dcl_temps"),
            ff("a"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        // Declarations are included in the block.
        assert_eq!(cfg.num_blocks(), 1);
        assert_eq!(cfg.block(cfg.entry()).instructions().len(), 3);
    }

    #[test]
    fn dot_output() {
        let cfg = CfgBuilder::build(vec![
            ff("add"),
            MockInst(FlowEffect::ConditionalOpen, "if"),
            ff("mul"),
            MockInst(FlowEffect::ConditionalClose, "endif"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        let dot = cfg.to_dot();
        assert!(dot.contains("digraph cfg"));
        assert!(dot.contains("bb0"));
        assert!(dot.contains("green4")); // conditional true edge
    }

    #[test]
    fn traversal_preorder() {
        let cfg = CfgBuilder::build(vec![
            ff("a"),
            MockInst(FlowEffect::ConditionalOpen, "if"),
            ff("b"),
            MockInst(FlowEffect::ConditionalClose, "endif"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        let pre = cfg.dfs_preorder();
        // Entry should be first.
        assert_eq!(pre[0], cfg.entry());
        // All reachable blocks should be visited.
        assert!(pre.len() >= 3);
    }

    #[test]
    fn dominator_tree_linear() {
        let cfg = CfgBuilder::build(vec![
            ff("a"),
            MockInst(FlowEffect::ConditionalOpen, "if"),
            ff("b"),
            MockInst(FlowEffect::ConditionalClose, "endif"),
            ff("c"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        let dom = crate::graph::dominator::DominatorTree::compute(&cfg);
        // Entry dominates all blocks.
        for b in cfg.blocks() {
            assert!(dom.dominates(cfg.entry(), b.id()));
        }
    }

    #[test]
    fn continue_jumps_to_header() {
        // loop; a; continue; b; endloop; ret
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            ff("a"),
            MockInst(FlowEffect::Continue, "continue"),
            ff("b"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        // The continue should create a back-edge to the header.
        let has_back = cfg.edges().any(|e| e.kind() == EdgeKind::Back);
        assert!(has_back);
        assert!(cfg.num_blocks() >= 3);
    }

    #[test]
    fn conditional_continue() {
        // loop; a; continuec; b; endloop; ret
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            ff("a"),
            MockInst(FlowEffect::ConditionalContinue, "continuec"),
            ff("b"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        // continuec block has two successors: true->header, false->continue
        let has_cond_true = cfg.edges().any(|e| e.kind() == EdgeKind::ConditionalTrue);
        let has_cond_false = cfg.edges().any(|e| e.kind() == EdgeKind::ConditionalFalse);
        assert!(has_cond_true);
        assert!(has_cond_false);
        assert!(cfg.num_blocks() >= 4);
    }

    #[test]
    fn conditional_return() {
        // a; retc; b; ret
        let cfg = CfgBuilder::build(vec![
            ff("a"),
            MockInst(FlowEffect::ConditionalReturn, "retc"),
            ff("b"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        // retc splits into ret_block (terminal) and cont_block (with b).
        let has_cond_true = cfg.edges().any(|e| e.kind() == EdgeKind::ConditionalTrue);
        let has_cond_false = cfg.edges().any(|e| e.kind() == EdgeKind::ConditionalFalse);
        assert!(has_cond_true);
        assert!(has_cond_false);
        assert!(cfg.num_blocks() >= 3);
    }

    #[test]
    fn terminate_ends_block() {
        // a; abort; b; ret
        let cfg = CfgBuilder::build(vec![
            ff("a"),
            MockInst(FlowEffect::Terminate, "abort"),
            ff("b"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        // abort terminates the block; b starts a new (unreachable) block.
        assert!(cfg.num_blocks() >= 2);
    }

    #[test]
    fn label_splits_block() {
        // a; label; b; ret
        let cfg = CfgBuilder::build(vec![
            ff("a"),
            MockInst(FlowEffect::Label, "label_0"),
            ff("b"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        // The label should split into two blocks with a fallthrough edge.
        assert!(cfg.num_blocks() >= 2);
        let has_fallthrough = cfg.edges().any(|e| e.kind() == EdgeKind::Fallthrough);
        assert!(has_fallthrough);
    }

    #[test]
    fn switch_with_cases() {
        // switch; a; case; b; case; c; endswitch; d; ret
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::SwitchOpen, "switch"),
            ff("a"),
            MockInst(FlowEffect::SwitchCase, "case"),
            ff("b"),
            MockInst(FlowEffect::SwitchCase, "default"),
            ff("c"),
            MockInst(FlowEffect::SwitchClose, "endswitch"),
            ff("d"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        // switch block dispatches to multiple case arms.
        let switch_edges: Vec<_> = cfg
            .edges()
            .filter(|e| e.kind() == EdgeKind::SwitchCase)
            .collect();
        assert!(switch_edges.len() >= 2); // at least first case + case + default
        assert!(cfg.num_blocks() >= 5);
    }

    #[test]
    fn switch_break_exits_switch() {
        // switch; a; break; case; b; endswitch; c; ret
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::SwitchOpen, "switch"),
            ff("a"),
            MockInst(FlowEffect::Break, "break"),
            MockInst(FlowEffect::SwitchCase, "case"),
            ff("b"),
            MockInst(FlowEffect::SwitchClose, "endswitch"),
            ff("c"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        // The break should wire to the post-switch merge block.
        let unconditional_edges: Vec<_> = cfg
            .edges()
            .filter(|e| e.kind() == EdgeKind::Unconditional)
            .collect();
        assert_ne!(unconditional_edges.len(), 0);
        assert!(cfg.num_blocks() >= 4);
    }

    #[test]
    fn nested_if_in_loop() {
        // loop; if; a; else; b; endif; endloop; ret
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            MockInst(FlowEffect::ConditionalOpen, "if"),
            ff("a"),
            MockInst(FlowEffect::ConditionalAlternate, "else"),
            ff("b"),
            MockInst(FlowEffect::ConditionalClose, "endif"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        assert!(cfg.num_blocks() >= 5);
        let has_back = cfg.edges().any(|e| e.kind() == EdgeKind::Back);
        assert!(has_back);
    }

    #[test]
    fn nested_loop_in_if() {
        // if; loop; a; breakc; endloop; endif; ret
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::ConditionalOpen, "if"),
            MockInst(FlowEffect::LoopOpen, "loop"),
            ff("a"),
            MockInst(FlowEffect::ConditionalBreak, "breakc"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::ConditionalClose, "endif"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        assert!(cfg.num_blocks() >= 5);
        let has_back = cfg.edges().any(|e| e.kind() == EdgeKind::Back);
        assert!(has_back);
    }

    #[test]
    fn switch_inside_loop_break_exits_switch() {
        // loop; switch; a; break; case; b; endswitch; breakc; endloop; ret
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            MockInst(FlowEffect::SwitchOpen, "switch"),
            ff("a"),
            MockInst(FlowEffect::Break, "break"), // exits switch, not loop
            MockInst(FlowEffect::SwitchCase, "case"),
            ff("b"),
            MockInst(FlowEffect::SwitchClose, "endswitch"),
            MockInst(FlowEffect::ConditionalBreak, "breakc"), // exits loop
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        // The break inside the switch should exit the switch.
        // The breakc after endswitch should exit the loop.
        assert!(cfg.num_blocks() >= 6);
        let has_back = cfg.edges().any(|e| e.kind() == EdgeKind::Back);
        assert!(has_back);
    }
}
