//! CFG builder — converts a flat instruction stream into a [`Cfg`]
//! using a scope-stack approach for structured control flow.
//!
//! Address-shaped streams (machine code, bytecode with branch targets and
//! exception tables in one address space) build through
//! [`build_address_cfg`] instead, which discovers leaders rather than
//! consuming structured markers.

mod address;

pub use address::{
    AddressBuildError, AddressCfgOptions, AddressEdgeInfo, AddressGraph, AddressHandler,
    AddressHandlerKind, AddressInstruction, AddressSpace, CallPolicy, GivenEdge, GivenEdges,
    HopEnd, UnwindEnd, UnwindHop, UnwindPath, UnwindRoute, UnwindStep, build_address_cfg,
};

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::flow::{Flow, FlowControl, FlowEffect, UnresolvedTransfer};

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
            FlowEffect::Flow(flow) => {
                self.transfer(instruction, &flow);
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
            FlowEffect::Label(_) => {
                self.add_label(instruction);
                Ok(())
            }
        }
    }

    /// Stores an instruction that leaves by `flow` and shapes the blocks
    /// around it, as [`FlowEffect::Flow`] documents.
    fn transfer(&mut self, instruction: I, flow: &Flow<I::Target, I::CaseKey>) {
        match flow {
            Flow::Next | Flow::Call { .. } | Flow::IndirectCall => self.push(instruction),
            Flow::Return
            | Flow::Throw
            | Flow::Trap
            | Flow::Indirect
            | Flow::Jump { .. }
            | Flow::TailCall { .. }
            | Flow::Switch { .. } => self.end_block(instruction),
            Flow::Conditional { .. } | Flow::ConditionalIndirect => {
                self.continue_after(instruction, EdgeKind::ConditionalFalse);
            }
            Flow::Exceptional { .. } | Flow::ConditionalTrap => {
                self.continue_after(instruction, EdgeKind::Fallthrough);
            }
            Flow::ConditionalReturn => self.add_conditional_return(instruction),
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

    /// Ends the current block after `instruction` and continues in a new
    /// block that the edge of `kind` reaches.
    fn continue_after(&mut self, instruction: I, kind: EdgeKind) {
        self.push(instruction);
        let continuation = self.cfg.new_block();
        self.cfg.add_edge(self.current, continuation, kind);
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
    /// Each instruction's [`FlowEffect`] is a structured marker, a label, or
    /// a [`Flow`]; [`FlowEffect`] documents the blocks and edges each one
    /// makes. The named targets of the flows are wired afterwards by
    /// [`resolve_jump_edges`].
    ///
    /// # Errors
    ///
    /// Returns an error if the instruction stream contains mismatched
    /// structured control-flow markers (e.g. `else` without `if`).
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{CfgBuilder, Flow, FlowControl, FlowEffect};
    ///
    /// #[derive(Debug, Clone)]
    /// struct Inst(FlowEffect<u32, ()>);
    ///
    /// impl FlowControl for Inst {
    ///     type Target = u32;
    ///     type CaseKey = ();
    ///
    ///     fn flow_effect(&self) -> FlowEffect<u32, ()> {
    ///         self.0.clone()
    ///     }
    /// }
    ///
    /// let plain = || Inst(FlowEffect::Flow(Flow::Next));
    /// let cfg = CfgBuilder::build(vec![
    ///     plain(),
    ///     Inst(FlowEffect::ConditionalOpen),
    ///     plain(),
    ///     Inst(FlowEffect::ConditionalAlternate),
    ///     plain(),
    ///     Inst(FlowEffect::ConditionalClose),
    ///     Inst(FlowEffect::Flow(Flow::Return)),
    /// ]).unwrap();
    ///
    /// assert_eq!(cfg.block_count(), 4); // entry, then, else, merge
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
    /// artifacts from the builder).
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
    /// The transfers the pass could not wire, each from its block: a named
    /// target that matched no label, or a computed one (`target: None`).
    pub unresolved: Vec<UnresolvedTransfer<T, BlockId>>,
}

/// Wire the edges described by explicit targets after
/// [`CfgBuilder::build`].
///
/// [`CfgBuilder::build`] ends a block at an instruction whose [`Flow`]
/// names a target but cannot know where the target lands. This pass maps
/// the token of every block-leading [`FlowEffect::Label`] instruction to
/// its block. Then, for the [`Flow`] of each block's last instruction, it
/// adds an edge for every transfer other than the continuation and a call,
/// of the kind [`EdgeRole::kind`](crate::EdgeRole::kind) gives: an
/// [`EdgeKind::Jump`] for a jump, an [`EdgeKind::ConditionalTrue`] for a
/// conditional branch's taken path, an [`EdgeKind::SwitchCase`] for each
/// switch target, and an [`EdgeKind::ExceptionUnwind`] for an exceptional
/// target. A call names a function rather than a label, so the pass
/// leaves it alone. The graph can carry any edge payload `E`; each edge
/// the pass adds carries `E::default()`.
///
/// The pass is idempotent: an edge that already exists with the same
/// endpoints and kind is not duplicated, so re-running it is safe.
/// Duplicate label tokens resolve to the LAST block declaring them (later
/// labels shadow earlier ones); frontends with label scoping enforce
/// uniqueness before building.
pub fn resolve_jump_edges<I: FlowControl, E: Default>(
    cfg: &mut Cfg<I, E>,
) -> JumpResolution<I::Target> {
    let mut labels: BTreeMap<I::Target, BlockId> = BTreeMap::new();
    for block_id in cfg.block_ids() {
        if let Some(first) = cfg.block(block_id).instructions().first()
            && let FlowEffect::Label(token) = first.flow_effect()
        {
            labels.insert(token, block_id);
        }
    }

    let mut pending: Vec<(BlockId, BlockId, EdgeKind)> = Vec::new();
    let mut resolution = JumpResolution {
        resolved: 0,
        unresolved: Vec::new(),
    };
    for block_id in cfg.block_ids() {
        let Some(last) = cfg.block(block_id).instructions().last() else {
            continue;
        };
        let FlowEffect::Flow(flow) = last.flow_effect() else {
            continue;
        };
        for transfer in flow.departures() {
            let Some(role) = transfer.role.unresolved_role() else {
                continue;
            };
            if transfer.role.is_call() {
                continue;
            }
            match transfer.target.as_ref().and_then(|token| labels.get(token)) {
                Some(&target) => pending.push((block_id, target, transfer.role.kind())),
                None => resolution.unresolved.push(UnresolvedTransfer {
                    source: block_id,
                    target: transfer.target,
                    role,
                }),
            }
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
