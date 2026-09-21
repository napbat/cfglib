//! Flat-instruction-list translation with effect-ordered inlining.
//!
//! A definition inlines into its consumer only when doing so provably
//! preserves evaluation order: the definition has exactly one local use
//! (and is dead beyond it), pure computations may cross anything that does
//! not redefine their reads, and effectful or throwing computations may
//! cross only instructions the dialect declares them to commute with
//! ([`LiftDialect::evaluation_commutes`] — refused by default). Effectful
//! definitions inlined into one consumer additionally keep their original
//! relative order across its operands. Everything else materializes as an
//! assignment at its original position.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;

use crate::block::BlockId;
use crate::ir::mlil;
use crate::{FlowControl, FlowEffect};

use super::super::{
    Dialect, EntityId, Error, ExpressionId, ExpressionKind, Result, StatementId, StatementKind,
    VariableId, VerifyDialect,
};
use super::{LiftDialect, Lifted, Lifter};

/// What a translated instruction list is allowed to end with.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Expect {
    /// Ordinary statements; a branch or dispatch terminator is an error.
    Statements,
    /// The list must end in a conditional branch.
    Branch,
    /// The list must end in a switch dispatch.
    Switch,
}

/// The branch or dispatch value a condition list ends with.
pub(super) struct ListEnd {
    /// The condition or scrutinee expression.
    pub value: ExpressionId,
    /// The terminator instruction, mapped by the caller to the structural
    /// statement it becomes.
    pub instruction: mlil::InstructionId,
}

/// The emission shape of one MLIL instruction.
enum Shape<D: LiftDialect> {
    Constant(<D as Dialect>::Constant),
    Copy,
    Operation(<D as Dialect>::Operation),
    Store(<D as Dialect>::Operation),
    ParallelCopy,
    Branch,
    BranchOperation(<D as Dialect>::Operation),
    Dispatch,
    Return,
    ControlFlow,
}

fn classify<D: LiftDialect>(instruction: &mlil::Instruction<D>) -> Shape<D> {
    let operation = instruction.operation();
    if let Some(constant) = <D as mlil::AnalysisDialect>::constant(operation) {
        return Shape::Constant(constant);
    }
    if <D as mlil::AnalysisDialect>::is_copy(operation) {
        return Shape::Copy;
    }
    match D::lift_operation(operation) {
        Lifted::Operation(operation) => Shape::Operation(operation),
        Lifted::Store { location } => Shape::Store(location),
        // A parallel move of exactly one pair is a plain copy, so it takes
        // part in expression inlining (type-refinement pairs, lone
        // phi-copy commits).
        Lifted::ParallelCopy if instruction.uses().len() == 1 && instruction.defs().len() == 1 => {
            Shape::Copy
        }
        Lifted::ParallelCopy => Shape::ParallelCopy,
        Lifted::Branch => Shape::Branch,
        Lifted::BranchOperation(operation) => Shape::BranchOperation(operation),
        Lifted::Switch => Shape::Dispatch,
        Lifted::Return => Shape::Return,
        Lifted::ControlFlow => Shape::ControlFlow,
    }
}

/// Whether emission builds an operand expression for use position `k`.
///
/// Plain branch and dispatch terminators evaluate only their first use
/// (the condition or scrutinee); their remaining uses never become
/// expressions, so definitions feeding them must stay materialized.
fn use_is_consumed<D: LiftDialect>(shape: &Shape<D>, position: usize) -> bool {
    match shape {
        Shape::Copy
        | Shape::Operation(_)
        | Shape::Store(_)
        | Shape::ParallelCopy
        | Shape::BranchOperation(_)
        | Shape::Return => true,
        Shape::Branch | Shape::Dispatch => position == 0,
        Shape::Constant(_) | Shape::ControlFlow => false,
    }
}

/// Whether the instruction can inline into a consumer as a value.
fn is_value_shape<D: LiftDialect>(shape: &Shape<D>) -> bool {
    matches!(
        shape,
        Shape::Constant(_) | Shape::Copy | Shape::Operation(_)
    )
}

/// Instructions retained by the presentation lift after local dead-value
/// pruning. A backward liveness walk removes pure fallthrough definitions
/// whose results cannot reach a retained instruction or another block.
fn retained_positions<D: LiftDialect, P: Borrow<mlil::Instruction<D>>>(
    instructions: &[P],
    live_out: &BTreeSet<mlil::VariableId>,
) -> Vec<bool> {
    let mut live = live_out.clone();
    let mut retained = vec![true; instructions.len()];
    for (position, instruction) in instructions.iter().enumerate().rev() {
        let instruction = instruction.borrow();
        let dead_value = !instruction.defs().is_empty()
            && instruction
                .defs()
                .iter()
                .all(|variable| !live.contains(variable))
            && instruction.effects().is_empty()
            && !instruction.may_throw()
            && instruction.flow_effect() == FlowEffect::Fallthrough;
        if dead_value {
            retained[position] = false;
            continue;
        }
        for definition in instruction.defs() {
            live.remove(definition);
        }
        live.extend(instruction.uses().iter().copied());
    }
    retained
}

struct CandidateFacts<D: LiftDialect> {
    /// Every variable the candidate's expression tree reads.
    reads: BTreeSet<mlil::VariableId>,
    /// Sorted, deduplicated effects of the whole tree.
    effects: Vec<D::Effect>,
    /// Whether any node of the tree may throw.
    may_throw: bool,
}

impl<D: LiftDialect> CandidateFacts<D> {
    /// Whether reordering this tree is observable at all.
    fn is_effect_relevant(&self) -> bool {
        !self.effects.is_empty() || self.may_throw
    }

    /// Whether this tree's evaluation may move across an instruction with
    /// the given effect profile.
    fn commutes_with(&self, effects: &[D::Effect], may_throw: bool) -> bool {
        if !self.is_effect_relevant() || (effects.is_empty() && !may_throw) {
            return true;
        }
        D::evaluation_commutes(&self.effects, self.may_throw, effects, may_throw)
    }
}

/// Exact local single-use positions: one use between the definition and
/// the variable's next redefinition, and dead beyond the list unless
/// redefined inside it.
fn single_use_positions<D: LiftDialect, P: Borrow<mlil::Instruction<D>>>(
    instructions: &[P],
    shapes: &[Shape<D>],
    live_out: &BTreeSet<mlil::VariableId>,
    retained: &[bool],
) -> Vec<Option<usize>> {
    struct PendingUse {
        definition: usize,
        occurrences: usize,
        use_position: Option<usize>,
    }

    let mut viable: Vec<Option<usize>> = vec![None; instructions.len()];
    let mut active: BTreeMap<mlil::VariableId, PendingUse> = BTreeMap::new();
    for (position, instruction) in instructions.iter().enumerate() {
        let instruction = instruction.borrow();
        if !retained[position] {
            continue;
        }

        // Reads precede writes. This preserves the old segment semantics for
        // read-modify-write instructions while visiting each operand once.
        for &variable in instruction.uses() {
            if let Some(pending) = active.get_mut(&variable) {
                pending.occurrences = pending.occurrences.saturating_add(1);
                pending.use_position.get_or_insert(position);
            }
        }
        for &variable in instruction.defs() {
            if let Some(pending) = active.remove(&variable)
                && pending.occurrences == 1
            {
                viable[pending.definition] = pending.use_position;
            }
        }

        if is_value_shape(&shapes[position])
            && instruction.defs().len() == 1
            && D::previous_value_operand(instruction.operation()).is_none()
        {
            // A read-modify-write definition merges with prior state: it is
            // not a pure value and never inlines forward.
            active.insert(
                instruction.defs()[0],
                PendingUse {
                    definition: position,
                    occurrences: 0,
                    use_position: None,
                },
            );
        }
    }
    for (variable, pending) in active {
        if pending.occurrences == 1 && !live_out.contains(&variable) {
            viable[pending.definition] = pending.use_position;
        }
    }
    viable
}

/// Decide, per definition, the consumer it inlines into (if any).
fn plan_inlining<D: LiftDialect, P: Borrow<mlil::Instruction<D>>>(
    instructions: &[P],
    shapes: &[Shape<D>],
    live_out: &BTreeSet<mlil::VariableId>,
    retained: &[bool],
) -> Vec<Option<usize>> {
    let length = instructions.len();
    let viable = single_use_positions(instructions, shapes, live_out, retained);

    // Order-safety walk over the movable candidates.
    let mut inline_at: Vec<Option<usize>> = vec![None; length];
    let mut facts: Vec<Option<CandidateFacts<D>>> = (0..length).map(|_| None).collect();
    let mut active: BTreeMap<mlil::VariableId, usize> = BTreeMap::new();
    for (position, instruction) in instructions.iter().enumerate() {
        let instruction = instruction.borrow();
        if !retained[position] {
            continue;
        }
        // Consumption first: feeding this instruction is not a crossing.
        // Effect-relevant candidates consumed by one instruction must keep
        // their original relative order across its operands, since operands
        // evaluate left to right.
        let mut consumed: Vec<usize> = Vec::new();
        let mut last_effectful: Option<usize> = None;
        for (operand, &variable) in instruction.uses().iter().enumerate() {
            if !use_is_consumed(&shapes[position], operand)
                || D::previous_value_operand(instruction.operation()) == Some(operand)
            {
                // A previous-value read keeps its producer materialized:
                // the merge point must stay visible as a variable.
                continue;
            }
            if let Some(&candidate) = active.get(&variable) {
                if viable[candidate] == Some(position) {
                    let relevant = facts[candidate]
                        .as_ref()
                        .is_some_and(CandidateFacts::is_effect_relevant);
                    if relevant && last_effectful.is_some_and(|latest| candidate < latest) {
                        // Inlining here would evaluate this tree after a
                        // later-defined effectful tree: keep it materialized.
                        continue;
                    }
                    inline_at[candidate] = Some(position);
                    consumed.push(candidate);
                    active.remove(&variable);
                    if relevant {
                        last_effectful = Some(candidate);
                    }
                }
            }
        }
        // An effectful or throwing instruction bars every candidate the
        // dialect does not declare to commute with it.
        if !instruction.effects().is_empty() || instruction.may_throw() {
            active.retain(|_, &mut candidate| {
                facts[candidate].as_ref().is_some_and(|f| {
                    f.commutes_with(instruction.effects(), instruction.may_throw())
                })
            });
        }
        for &defined in instruction.defs() {
            active.remove(&defined);
            // A candidate reading the redefined variable can no longer move
            // past this point.
            active.retain(|_, &mut candidate| {
                facts[candidate]
                    .as_ref()
                    .is_some_and(|f| !f.reads.contains(&defined))
            });
        }
        if viable[position].is_some() && !D::materialize_value(instruction.operation()) {
            let mut reads = BTreeSet::new();
            let mut effects = instruction.effects().to_vec();
            let mut may_throw = instruction.may_throw();
            for &variable in instruction.uses() {
                let inlined = consumed
                    .iter()
                    .copied()
                    .find(|&candidate| instructions[candidate].borrow().defs()[0] == variable);
                if let Some(candidate) = inlined {
                    if let Some(f) = facts[candidate].as_ref() {
                        reads.extend(f.reads.iter().copied());
                        effects.extend(f.effects.iter().cloned());
                        may_throw |= f.may_throw;
                    }
                } else {
                    reads.insert(variable);
                }
            }
            effects.sort_unstable();
            effects.dedup();
            facts[position] = Some(CandidateFacts {
                reads,
                effects,
                may_throw,
            });
            active.insert(instruction.defs()[0], position);
        }
    }
    inline_at
}

/// The mutable per-list emission state shared by the shape emitters.
struct ListState {
    /// Per definition, the consumer position it inlines into.
    inline_at: Vec<Option<usize>>,
    /// Consumer position → (consumed variable → its inlined definition).
    by_consumer: BTreeMap<usize, BTreeMap<mlil::VariableId, usize>>,
    /// Expressions parked for their inlined consumer.
    built: Vec<Option<ExpressionId>>,
    /// Statements emitted so far, in program order.
    statements: Vec<StatementId>,
}

impl<D: LiftDialect + VerifyDialect> Lifter<'_, D> {
    /// Translates one block-shaped instruction list into statements, with
    /// its terminator value when the caller expects one.
    pub(super) fn translate_list<P>(
        &mut self,
        block: BlockId,
        instructions: &[P],
        expect: Expect,
    ) -> Result<(Vec<StatementId>, Option<ListEnd>)>
    where
        P: Borrow<mlil::Instruction<D>>,
    {
        let shapes: Vec<Shape<D>> = instructions
            .iter()
            .map(|instruction| classify(instruction.borrow()))
            .collect();
        let retained = retained_positions(instructions, self.liveness.live_out(block));
        let inline_at = plan_inlining(
            instructions,
            &shapes,
            self.liveness.live_out(block),
            &retained,
        );
        let mut by_consumer: BTreeMap<usize, BTreeMap<mlil::VariableId, usize>> = BTreeMap::new();
        for (candidate, consumer) in inline_at.iter().enumerate() {
            if let Some(consumer) = consumer {
                by_consumer
                    .entry(*consumer)
                    .or_default()
                    .insert(instructions[candidate].borrow().defs()[0], candidate);
            }
        }
        let mut state = ListState {
            inline_at,
            by_consumer,
            built: vec![None; instructions.len()],
            statements: Vec::new(),
        };

        let mut end = None;
        let mut finished = false;
        for (position, instruction) in instructions.iter().enumerate() {
            let instruction = instruction.borrow();
            if finished || end.is_some() {
                return Err(Error::UnsupportedLift(format!(
                    "instruction {} follows its block's terminator",
                    instruction.id()
                )));
            }
            if !retained[position] {
                continue;
            }
            match &shapes[position] {
                Shape::ControlFlow => {}
                Shape::Constant(constant) => {
                    if instruction.defs().is_empty() {
                        continue;
                    }
                    require_single_definition(instruction)?;
                    let value_type = instruction.def_types()[0].clone();
                    let expression = self
                        .builder
                        .add_expression(ExpressionKind::Constant(constant.clone()), value_type)?;
                    self.finish_value(position, instruction, expression, &mut state)?;
                }
                Shape::Copy => {
                    require_single_definition(instruction)?;
                    if instruction.uses().len() != 1 {
                        return Err(Error::UnsupportedLift(format!(
                            "copy {} does not have exactly one use",
                            instruction.id()
                        )));
                    }
                    let value = self.operand(position, 0, instruction, &mut state)?;
                    self.finish_value(position, instruction, value, &mut state)?;
                }
                Shape::Operation(operation) => {
                    self.emit_operation(position, instruction, operation.clone(), &mut state)?;
                }
                Shape::Store(location) => {
                    self.emit_store(position, instruction, location.clone(), &mut state)?;
                }
                Shape::ParallelCopy => {
                    self.emit_parallel_copy(position, instruction, &mut state)?;
                }
                Shape::Branch | Shape::BranchOperation(_) | Shape::Dispatch => {
                    end = Some(self.emit_terminator_value(
                        position,
                        instruction,
                        &shapes[position],
                        expect,
                        &mut state,
                    )?);
                }
                Shape::Return => {
                    let values = self.operands(position, instruction, &mut state)?;
                    let statement = self
                        .builder
                        .add_statement(StatementKind::Return { values }, None)?;
                    self.map_instruction(instruction.id(), EntityId::Statement(statement))?;
                    state.statements.push(statement);
                    finished = true;
                }
            }
        }
        if end.is_none() && !matches!(expect, Expect::Statements) {
            return Err(Error::UnsupportedLift(format!(
                "block {block} does not end in its expected branch"
            )));
        }
        Ok((state.statements, end))
    }

    /// Emits one dialect operation: a value (inlined or assigned) with one
    /// definition, an effect statement with none.
    fn emit_operation(
        &mut self,
        position: usize,
        instruction: &mlil::Instruction<D>,
        operation: <D as Dialect>::Operation,
        state: &mut ListState,
    ) -> Result<()> {
        let operands = self.operands(position, instruction, state)?;
        match instruction.defs().len() {
            0 => {
                let expression = self.builder.add_expression(
                    ExpressionKind::Operation {
                        operation,
                        operands,
                    },
                    D::void_type(),
                )?;
                let statement = self
                    .builder
                    .add_statement(StatementKind::Expression(expression), None)?;
                self.map_instruction(instruction.id(), EntityId::Statement(statement))?;
                state.statements.push(statement);
                Ok(())
            }
            1 => {
                let value_type = instruction.def_types()[0].clone();
                let expression = self.builder.add_expression(
                    ExpressionKind::Operation {
                        operation,
                        operands,
                    },
                    value_type,
                )?;
                self.finish_value(position, instruction, expression, state)
            }
            _ => Err(Error::UnsupportedLift(format!(
                "instruction {} defines more than one result",
                instruction.id()
            ))),
        }
    }

    /// Builds the condition or scrutinee value ending a branch or dispatch
    /// list.
    fn emit_terminator_value(
        &mut self,
        position: usize,
        instruction: &mlil::Instruction<D>,
        shape: &Shape<D>,
        expect: Expect,
        state: &mut ListState,
    ) -> Result<ListEnd> {
        let expected = matches!(
            (shape, expect),
            (Shape::Branch | Shape::BranchOperation(_), Expect::Branch)
                | (Shape::Dispatch, Expect::Switch)
        );
        if !expected {
            return Err(Error::UnsupportedLift(format!(
                "unexpected branch or dispatch instruction {}",
                instruction.id()
            )));
        }
        let value = if let Shape::BranchOperation(operation) = shape {
            // The fused condition applies the operation to every use; an
            // empty use list is legal when the operation embeds its whole
            // condition (a constant or memory read).
            let operands = self.operands(position, instruction, state)?;
            self.builder.add_expression(
                ExpressionKind::Operation {
                    operation: operation.clone(),
                    operands,
                },
                D::void_type(),
            )?
        } else {
            if instruction.uses().is_empty() {
                return Err(Error::UnsupportedLift(format!(
                    "branch {} has no condition operand",
                    instruction.id()
                )));
            }
            self.operand(position, 0, instruction, state)?
        };
        Ok(ListEnd {
            value,
            instruction: instruction.id(),
        })
    }

    /// Emits one pairwise parallel move: every use's value into the
    /// definition at the same position, all reads before any write. Moves
    /// whose reads never observe an earlier write emit directly; anything
    /// else stages every value through a fresh temporary first.
    fn emit_parallel_copy(
        &mut self,
        position: usize,
        instruction: &mlil::Instruction<D>,
        state: &mut ListState,
    ) -> Result<()> {
        let id = instruction.id();
        if instruction.uses().len() != instruction.defs().len() {
            return Err(Error::UnsupportedLift(format!(
                "parallel copy {id} does not pair uses with definitions"
            )));
        }
        let values = self.operands(position, instruction, state)?;
        let written: Vec<VariableId> = instruction
            .defs()
            .iter()
            .map(|variable| VariableId::from_raw(variable.raw()))
            .collect();
        let direct = values.iter().enumerate().all(|(index, &value)| {
            let mut reads = BTreeSet::new();
            self.expression_reads(value, &mut reads);
            written[..index]
                .iter()
                .all(|earlier| !reads.contains(earlier))
        });
        let staged: Vec<ExpressionId> = if direct {
            values
        } else {
            let Some(role) = D::temporary_role() else {
                return Err(Error::UnsupportedLift(format!(
                    "overlapping parallel copy {id} needs a temporary role"
                )));
            };
            let mut staged = Vec::with_capacity(values.len());
            for (&value, value_type) in values.iter().zip(instruction.use_types()) {
                let temporary =
                    self.builder
                        .declare_variable(role.clone(), None, Some(value_type.clone()))?;
                let target = self
                    .builder
                    .add_expression(ExpressionKind::Variable(temporary), value_type.clone())?;
                let assign = self
                    .builder
                    .add_statement(StatementKind::Assign { target, value }, None)?;
                state.statements.push(assign);
                staged.push(
                    self.builder
                        .add_expression(ExpressionKind::Variable(temporary), value_type.clone())?,
                );
            }
            staged
        };
        let mut first_statement = None;
        for ((value, &variable), value_type) in staged
            .into_iter()
            .zip(&written)
            .zip(instruction.def_types())
        {
            let target = self
                .builder
                .add_expression(ExpressionKind::Variable(variable), value_type.clone())?;
            let statement = self
                .builder
                .add_statement(StatementKind::Assign { target, value }, None)?;
            first_statement.get_or_insert(statement);
            state.statements.push(statement);
        }
        if let Some(statement) = first_statement {
            self.map_instruction(id, EntityId::Statement(statement))?;
        }
        Ok(())
    }

    /// Collects every variable read anywhere in one expression tree.
    fn expression_reads(&self, expression: ExpressionId, reads: &mut BTreeSet<VariableId>) {
        let Some(node) = self.builder.expression(expression) else {
            return;
        };
        match node.kind() {
            ExpressionKind::Variable(variable) => {
                reads.insert(*variable);
            }
            ExpressionKind::Constant(_) => {}
            ExpressionKind::Operation { operands, .. } => {
                for &operand in operands {
                    self.expression_reads(operand, reads);
                }
            }
        }
    }

    /// Emits one store: the last use assigned into the place formed by the
    /// location operation over the preceding uses.
    fn emit_store(
        &mut self,
        position: usize,
        instruction: &mlil::Instruction<D>,
        location: <D as Dialect>::Operation,
        state: &mut ListState,
    ) -> Result<()> {
        let id = instruction.id();
        if !instruction.defs().is_empty() {
            return Err(Error::UnsupportedLift(format!(
                "store {id} also defines variables"
            )));
        }
        if instruction.uses().is_empty() {
            return Err(Error::UnsupportedLift(format!(
                "store {id} has no stored value"
            )));
        }
        let mut operands = self.operands(position, instruction, state)?;
        let value = operands.pop().expect("a store carries its value last");
        let value_type = self
            .builder
            .expression(value)
            .map_or_else(D::void_type, |expression| expression.value_type().clone());
        let target = self.builder.add_expression(
            ExpressionKind::Operation {
                operation: location,
                operands,
            },
            value_type,
        )?;
        let statement = self
            .builder
            .add_statement(StatementKind::Assign { target, value }, None)?;
        self.map_instruction(id, EntityId::Statement(statement))?;
        state.statements.push(statement);
        Ok(())
    }

    /// The expression for use position `operand` of `instruction`: the
    /// inlined definition's tree, or a fresh typed variable read.
    fn operand(
        &mut self,
        position: usize,
        operand: usize,
        instruction: &mlil::Instruction<D>,
        state: &mut ListState,
    ) -> Result<ExpressionId> {
        let variable = instruction.uses()[operand];
        if let Some(&candidate) = state
            .by_consumer
            .get(&position)
            .and_then(|consumed| consumed.get(&variable))
        {
            if let Some(expression) = state.built[candidate].take() {
                return Ok(expression);
            }
        }
        let value_type = instruction
            .use_types()
            .get(operand)
            .cloned()
            .unwrap_or_else(D::void_type);
        self.builder.add_expression(
            ExpressionKind::Variable(VariableId::from_raw(variable.raw())),
            value_type,
        )
    }

    fn operands(
        &mut self,
        position: usize,
        instruction: &mlil::Instruction<D>,
        state: &mut ListState,
    ) -> Result<Vec<ExpressionId>> {
        (0..instruction.uses().len())
            .map(|operand| self.operand(position, operand, instruction, state))
            .collect()
    }

    /// Completes a value-producing instruction: park the expression for its
    /// inlined consumer, or materialize the assignment here.
    fn finish_value(
        &mut self,
        position: usize,
        instruction: &mlil::Instruction<D>,
        expression: ExpressionId,
        state: &mut ListState,
    ) -> Result<()> {
        let id = instruction.id();
        if state.inline_at[position].is_some() {
            state.built[position] = Some(expression);
            self.map_instruction(id, EntityId::Expression(expression))?;
            return Ok(());
        }
        let variable = instruction.defs()[0];
        let value_type = instruction.def_types()[0].clone();
        let target = self.builder.add_expression(
            ExpressionKind::Variable(VariableId::from_raw(variable.raw())),
            value_type,
        )?;
        let statement = self.builder.add_statement(
            StatementKind::Assign {
                target,
                value: expression,
            },
            None,
        )?;
        self.map_instruction(id, EntityId::Statement(statement))?;
        state.statements.push(statement);
        Ok(())
    }
}

fn require_single_definition<D: LiftDialect>(instruction: &mlil::Instruction<D>) -> Result<()> {
    if instruction.defs().len() == 1 {
        Ok(())
    } else {
        Err(Error::UnsupportedLift(format!(
            "instruction {} defines more than one result",
            instruction.id()
        )))
    }
}
