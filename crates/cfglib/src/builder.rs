//! CFG builder — converts a flat instruction stream into a [`Cfg`]
//! using a scope-stack approach for structured control flow.
//!
//! Address-shaped streams (machine code, bytecode with branch targets and
//! exception tables in one address space) build through
//! [`address::build_address_cfg`] instead, which discovers leaders rather
//! than consuming structured markers.

pub mod address;

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

impl core::error::Error for BuildError {}

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
        let cfg = Cfg::new();
        let current = cfg.entry();
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
            .outgoing(pre_block)
            .any(|edge| self.cfg.edge(edge).kind() == EdgeKind::ConditionalFalse);
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
    /// assert!(cfg.block_count() >= 4); // entry, then, else, merge
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
        while let Some(last) = cfg.block_ids().last() {
            if last == cfg.entry()
                || !cfg.block(last).is_empty()
                || cfg.incoming(last).next().is_some()
            {
                break;
            }
            cfg.remove_block(last);
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
    for block_id in cfg.block_ids() {
        let block = cfg.block(block_id);
        if let Some(first) = block.instructions().first() {
            if first.flow_effect() == FlowEffect::Label {
                if let Some(token) = first.label() {
                    labels.insert(token, block_id);
                }
            }
        }
    }

    let mut pending: Vec<(BlockId, BlockId, EdgeKind)> = Vec::new();
    let mut resolution = JumpResolution {
        resolved: 0,
        unresolved: Vec::new(),
    };
    for block_id in cfg.block_ids() {
        let block = cfg.block(block_id);
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
            pending.push((block_id, target, kind));
        } else {
            resolution.unresolved.push((block_id, token));
        }
    }

    for (source, target, kind) in pending {
        let already_wired = cfg
            .outgoing(source)
            .any(|edge| cfg.edge(edge).target() == target && cfg.edge(edge).kind() == kind);
        if !already_wired {
            cfg.add_edge(source, target, kind);
            resolution.resolved += 1;
        }
    }
    resolution
}

#[cfg(test)]
mod tests;
