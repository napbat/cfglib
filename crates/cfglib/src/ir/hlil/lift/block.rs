//! Translation of one flat instruction list, with effect-ordered inlining.
//!
//! The plan of [`plan`](super::plan) decides which definitions inline. This
//! module walks the list once, builds each expression in the scratch arena,
//! and emits statements to the builder, which consumes the expression trees
//! that they use.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use crate::block::BlockId;
use crate::dataflow::{InstrInfo, ProgramPoint};
use crate::graph::view::DenseId;

use super::super::builder::FunctionBuilder;
use super::super::dialect::{Dialect, LiftDialect};
use super::super::flat::{FlatFunction, InstructionShape, LiftInstruction};
use super::super::view::Entity;
use super::plan::{Shape, UnwindReads, classify, plan_inlining, retained_positions};
use super::scratch::{PendingKind, Scratch, Slot};
use super::{Entry, Fallible, LiftMetadata, Lifter};

/// What a translated instruction list may end with.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Expect {
    /// Ordinary statements. A branch or dispatch terminator is an error.
    Statements,
    /// The list must end in a conditional branch.
    Branch,
    /// The list must end in a switch dispatch.
    Switch,
}

/// The branch or dispatch value that a condition list ends with.
pub(super) struct ListEnd<H: Dialect, V> {
    /// The pending expressions of the list, which still hold the value.
    pub(super) scratch: Scratch<H, V>,
    /// The condition or scrutinee, a node of `scratch`.
    pub(super) value: Slot,
    /// The terminator instruction, which the caller maps to the structural
    /// statement that it becomes.
    pub(super) point: ProgramPoint,
}

/// The statements of one list, and its terminator value when it has one.
pub(super) type Translated<H, X, B> = Fallible<(
    Vec<<B as FunctionBuilder<H>>::Stmt>,
    Option<ListEnd<H, <X as FlatFunction<H>>::Var>>,
)>;

/// The mutable state that the emitters of one list share.
struct ListState<H: Dialect, V, S> {
    /// Per definition, the consumer position that it inlines into.
    inline_at: Vec<Option<usize>>,
    /// Consumer position, then the consumed variable, to its inlined
    /// definition.
    by_consumer: BTreeMap<usize, BTreeMap<V, usize>>,
    /// The expressions that wait for their inlined consumer.
    built: Vec<Option<Slot>>,
    /// The statements emitted so far, in program order.
    statements: Vec<S>,
    scratch: Scratch<H, V>,
}

impl<H: LiftDialect, V: DenseId, S> ListState<H, V, S> {
    /// The expression of use position `operand` of `instruction`: the tree of
    /// the inlined definition, or a fresh typed read of the variable.
    fn operand<I>(&mut self, position: usize, operand: usize, instruction: &I) -> Slot
    where
        I: LiftInstruction<H> + InstrInfo<Variable = V>,
    {
        let variable = instruction.uses()[operand];
        if let Some(&candidate) = self
            .by_consumer
            .get(&position)
            .and_then(|consumed| consumed.get(&variable))
            && let Some(slot) = self.built[candidate].take()
        {
            return slot;
        }
        let ty = instruction
            .use_types()
            .get(operand)
            .cloned()
            .unwrap_or_else(H::void_type);
        self.scratch.variable(variable, ty)
    }

    fn operands<I>(&mut self, position: usize, instruction: &I) -> Vec<Slot>
    where
        I: LiftInstruction<H> + InstrInfo<Variable = V>,
    {
        (0..instruction.uses().len())
            .map(|operand| self.operand(position, operand, instruction))
            .collect()
    }
}

fn require_single_definition<H, I>(point: ProgramPoint, instruction: &I) -> Fallible<()>
where
    H: LiftDialect,
    I: LiftInstruction<H>,
{
    if instruction.defs().len() == 1 {
        Ok(())
    } else {
        Err(format!("instruction {point} defines more than one result"))
    }
}

impl<'a, H, X, B> Lifter<'a, H, X, B>
where
    H: LiftDialect,
    X: FlatFunction<H>,
    B: FunctionBuilder<H>,
{
    /// What an unwind out of each throwing instruction of `entries` reads,
    /// from the liveness of the flat function.
    fn unwind_reads(&self, entries: &[Entry<'a, X::Inst>]) -> UnwindReads<X::Var> {
        let mut live = BTreeMap::new();
        let blocks = entries
            .iter()
            .map(|&(point, instruction)| {
                if !instruction.may_throw() {
                    return None;
                }
                live.entry(point.block)
                    .or_insert_with(|| self.liveness.live_on_unwind(self.cfg, point.block));
                Some(point.block)
            })
            .collect();
        UnwindReads { blocks, live }
    }

    /// Translates one instruction list into statements, with its terminator
    /// value when the caller expects one.
    ///
    /// `block` is the block whose live-out judges the list. A list that joins
    /// several blocks passes the last of them.
    pub(super) fn translate_list(
        &mut self,
        block: BlockId,
        entries: &[Entry<'a, X::Inst>],
        expect: Expect,
    ) -> Translated<H, X, B> {
        let instructions: Vec<&X::Inst> = entries
            .iter()
            .map(|&(_, instruction)| instruction)
            .collect();
        let shapes: Vec<Shape<H>> = instructions
            .iter()
            .map(|&instruction| classify::<H, _>(instruction))
            .collect();
        let unwind = self.unwind_reads(entries);
        let live_out = self.liveness.live_out(block);
        let retained = retained_positions::<H, _>(&instructions, live_out, &unwind);
        let inline_at = plan_inlining::<H, _>(&instructions, &shapes, live_out, &unwind, &retained);
        let mut by_consumer: BTreeMap<usize, BTreeMap<X::Var, usize>> = BTreeMap::new();
        for (candidate, consumer) in inline_at.iter().enumerate() {
            if let Some(consumer) = consumer {
                by_consumer
                    .entry(*consumer)
                    .or_default()
                    .insert(instructions[candidate].defs()[0], candidate);
            }
        }
        let mut state = ListState {
            inline_at,
            by_consumer,
            built: vec![None; entries.len()],
            statements: Vec::new(),
            scratch: Scratch::new(),
        };

        let mut end = None;
        let mut finished = false;
        for (position, &(point, instruction)) in entries.iter().enumerate() {
            if finished || end.is_some() {
                return Err(format!(
                    "instruction {point} follows its block's terminator"
                ));
            }
            if !retained[position] {
                continue;
            }
            match &shapes[position] {
                InstructionShape::ControlFlow => {}
                InstructionShape::Constant(constant) => {
                    if instruction.defs().is_empty() {
                        continue;
                    }
                    require_single_definition::<H, _>(point, instruction)?;
                    let ty = instruction.def_types()[0].clone();
                    let slot = state.scratch.constant(constant.clone(), ty);
                    self.finish_value(position, point, instruction, slot, &mut state);
                }
                InstructionShape::Copy => {
                    require_single_definition::<H, _>(point, instruction)?;
                    if instruction.uses().len() != 1 {
                        return Err(format!("copy {point} does not have exactly one use"));
                    }
                    let value = state.operand(position, 0, instruction);
                    self.finish_value(position, point, instruction, value, &mut state);
                }
                InstructionShape::Operation(op) => {
                    self.emit_operation(position, point, instruction, op, &mut state)?;
                }
                InstructionShape::Store { location } => {
                    self.emit_store(position, point, instruction, location, &mut state)?;
                }
                InstructionShape::ParallelCopy => {
                    self.emit_parallel_copy(position, point, instruction, &mut state)?;
                }
                InstructionShape::Branch
                | InstructionShape::BranchOperation(_)
                | InstructionShape::Switch => {
                    end = Some(Self::terminator_value(
                        position,
                        point,
                        instruction,
                        &shapes[position],
                        expect,
                        &mut state,
                    )?);
                }
                InstructionShape::Return => {
                    let slots = state.operands(position, instruction);
                    let values: Vec<B::Expr> = slots
                        .into_iter()
                        .map(|slot| self.emit(&state.scratch, slot))
                        .collect();
                    let statement = self.builder.return_(&values);
                    self.record(point, Entity::Statement(statement));
                    state.statements.push(statement);
                    finished = true;
                }
            }
        }
        if end.is_none() && expect != Expect::Statements {
            return Err(format!("block {block} does not end in its expected branch"));
        }
        Ok((state.statements, end))
    }

    /// Emits one dialect operation: a value, inlined or assigned, when it has
    /// one definition, and an effect statement when it has none. An operation
    /// with several definitions has no HLIL form.
    fn emit_operation(
        &mut self,
        position: usize,
        point: ProgramPoint,
        instruction: &X::Inst,
        op: &H::Op,
        state: &mut ListState<H, X::Var, B::Stmt>,
    ) -> Fallible<()> {
        let operands = state.operands(position, instruction);
        match instruction.defs().len() {
            0 => {
                let slot = state
                    .scratch
                    .operation(op.clone(), operands, H::void_type());
                let expression = self.emit(&state.scratch, slot);
                let statement = self.builder.expression_statement(expression);
                self.record(point, Entity::Statement(statement));
                state.statements.push(statement);
                Ok(())
            }
            1 => {
                let ty = instruction.def_types()[0].clone();
                let slot = state.scratch.operation(op.clone(), operands, ty);
                self.finish_value(position, point, instruction, slot, state);
                Ok(())
            }
            _ => Err(format!("instruction {point} defines more than one result")),
        }
    }

    /// Builds the condition or scrutinee value that ends a branch or dispatch
    /// list, and takes the scratch arena that holds it.
    fn terminator_value(
        position: usize,
        point: ProgramPoint,
        instruction: &X::Inst,
        shape: &Shape<H>,
        expect: Expect,
        state: &mut ListState<H, X::Var, B::Stmt>,
    ) -> Fallible<ListEnd<H, X::Var>> {
        let expected = matches!(
            (shape, expect),
            (
                InstructionShape::Branch | InstructionShape::BranchOperation(_),
                Expect::Branch
            ) | (InstructionShape::Switch, Expect::Switch)
        );
        if !expected {
            return Err(format!("unexpected branch or dispatch instruction {point}"));
        }
        let value = if let InstructionShape::BranchOperation(op) = shape {
            // The fused condition applies the operation to every use. An
            // empty use list is legal when the operation embeds its whole
            // condition (a constant or a memory read).
            let operands = state.operands(position, instruction);
            state
                .scratch
                .operation(op.clone(), operands, H::void_type())
        } else {
            if instruction.uses().is_empty() {
                return Err(format!("branch {point} has no condition operand"));
            }
            state.operand(position, 0, instruction)
        };
        Ok(ListEnd {
            scratch: mem::replace(&mut state.scratch, Scratch::new()),
            value,
            point,
        })
    }

    /// Emits one pairwise parallel move: every use into the definition at the
    /// same position, all reads before any write. Moves whose reads never
    /// observe an earlier write emit directly. Anything else stages every
    /// value through a fresh temporary first.
    fn emit_parallel_copy(
        &mut self,
        position: usize,
        point: ProgramPoint,
        instruction: &X::Inst,
        state: &mut ListState<H, X::Var, B::Stmt>,
    ) -> Fallible<()> {
        if instruction.uses().len() != instruction.defs().len() {
            return Err(format!(
                "parallel copy {point} does not pair uses with definitions"
            ));
        }
        let slots = state.operands(position, instruction);
        let written = instruction.defs();
        let direct = slots.iter().enumerate().all(|(index, &slot)| {
            let mut reads = BTreeSet::new();
            state.scratch.collect_reads(slot, &mut reads);
            written[..index]
                .iter()
                .all(|earlier| !reads.contains(earlier))
        });
        let values: Vec<B::Expr> = if direct {
            slots
                .into_iter()
                .map(|slot| self.emit(&state.scratch, slot))
                .collect()
        } else {
            let mut staged = Vec::with_capacity(slots.len());
            for (slot, ty) in slots.into_iter().zip(instruction.use_types()) {
                let Some(metadata) = H::temporary_variable(ty) else {
                    return Err(format!(
                        "overlapping parallel copy {point} needs a temporary variable"
                    ));
                };
                let temporary = self.builder.declare_variable(&metadata, Some(ty));
                let value = self.emit(&state.scratch, slot);
                let target = self.builder.read_variable(temporary, ty);
                state.statements.push(self.builder.assign(target, value));
                staged.push(self.builder.read_variable(temporary, ty));
            }
            staged
        };
        let mut first_statement = None;
        for ((value, &variable), ty) in values.into_iter().zip(written).zip(instruction.def_types())
        {
            let target = self
                .builder
                .read_variable(self.variables[variable.index()], ty);
            let statement = self.builder.assign(target, value);
            first_statement.get_or_insert(statement);
            state.statements.push(statement);
        }
        if let Some(statement) = first_statement {
            self.record(point, Entity::Statement(statement));
        }
        Ok(())
    }

    /// Emits one store: the last use assigned into the place that the
    /// location operation forms over the preceding uses.
    fn emit_store(
        &mut self,
        position: usize,
        point: ProgramPoint,
        instruction: &X::Inst,
        location: &H::Op,
        state: &mut ListState<H, X::Var, B::Stmt>,
    ) -> Fallible<()> {
        if !instruction.defs().is_empty() {
            return Err(format!("store {point} also defines variables"));
        }
        if instruction.uses().is_empty() {
            return Err(format!("store {point} has no stored value"));
        }
        let mut operands = state.operands(position, instruction);
        let value = operands.pop().expect("a store carries its value last");
        let ty = state.scratch.node(value).ty.clone();
        let place = state.scratch.operation(location.clone(), operands, ty);
        let target = self.emit(&state.scratch, place);
        let value = self.emit(&state.scratch, value);
        let statement = self.builder.assign(target, value);
        self.record(point, Entity::Statement(statement));
        state.statements.push(statement);
        Ok(())
    }

    /// Completes a value-producing instruction: parks the expression for its
    /// inlined consumer, or materializes the assignment here.
    fn finish_value(
        &mut self,
        position: usize,
        point: ProgramPoint,
        instruction: &X::Inst,
        slot: Slot,
        state: &mut ListState<H, X::Var, B::Stmt>,
    ) {
        if state.inline_at[position].is_some() {
            if self.metadata == LiftMetadata::Preserve {
                state.scratch.attach(slot, point);
            }
            state.built[position] = Some(slot);
            return;
        }
        let variable = instruction.defs()[0];
        let ty = &instruction.def_types()[0];
        let value = self.emit(&state.scratch, slot);
        let target = self
            .builder
            .read_variable(self.variables[variable.index()], ty);
        let statement = self.builder.assign(target, value);
        self.record(point, Entity::Statement(statement));
        state.statements.push(statement);
    }

    /// Builds the tree under `root` in the builder, children first, and
    /// records the expression that carries each instruction it stands for.
    pub(super) fn emit(&mut self, scratch: &Scratch<H, X::Var>, root: Slot) -> B::Expr {
        self.values.clear();
        self.work.clear();
        self.work.push((root, false));
        while let Some((slot, operands_done)) = self.work.pop() {
            let node = scratch.node(slot);
            let expression = match &node.kind {
                PendingKind::Variable(variable) => self
                    .builder
                    .read_variable(self.variables[variable.index()], &node.ty),
                PendingKind::Constant(constant) => self.builder.constant(constant, &node.ty),
                PendingKind::Operation { op, operands } => {
                    if !operands_done {
                        self.work.push((slot, true));
                        self.work
                            .extend(operands.iter().rev().map(|&operand| (operand, false)));
                        continue;
                    }
                    let mark = self.values.len() - operands.len();
                    let expression = self.builder.apply(op, &self.values[mark..], &node.ty);
                    self.values.truncate(mark);
                    expression
                }
            };
            for &point in &node.points {
                self.record(point, Entity::Expression(expression));
            }
            self.values.push(expression);
        }
        self.values
            .pop()
            .expect("the emission of a tree leaves its root")
    }
}
