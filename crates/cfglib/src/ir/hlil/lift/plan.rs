//! The plan that decides which definitions inline into which consumers.
//!
//! A definition inlines into its consumer only when that provably preserves
//! the evaluation order: the definition has exactly one local use and is dead
//! beyond it, a pure computation may cross anything that does not redefine
//! its reads, and an effectful or throwing computation may cross only the
//! instructions that the dialect declares it to commute with
//! ([`LiftDialect::evaluation_commutes`], refused by default). Effectful
//! definitions that inline into one consumer also keep their original
//! relative order across its operands. A value that an unwind successor reads
//! stays materialized before each throwing instruction that can leave toward
//! it. Everything else materializes as an assignment at its original
//! position.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::flow::FlowEffect;

use super::super::dialect::{Dialect, LiftDialect};
use super::super::flat::{InstructionShape, LiftInstruction, LiftOperands};

/// The shape of one instruction as the plan and the emitters read it.
pub(super) type Shape<H> = InstructionShape<<H as Dialect>::Op, <H as Dialect>::Constant>;

/// The shape that the lift acts on. A parallel move of exactly one pair is a
/// plain copy, so it takes part in expression inlining (type-refinement
/// pairs, the lone commit of a phi copy).
pub(super) fn classify<H, I, O>(operands: &O, instruction: &I) -> Shape<H>
where
    H: LiftDialect,
    I: LiftInstruction<H>,
    O: LiftOperands<H, I>,
{
    match instruction.lift() {
        InstructionShape::ParallelCopy
            if operands.uses(instruction).len() == 1 && operands.defs(instruction).len() == 1 =>
        {
            InstructionShape::Copy
        }
        shape => shape,
    }
}

/// Whether emission builds an operand expression for use position
/// `position`.
///
/// A plain branch and a dispatch evaluate only their first use, the condition
/// or the scrutinee. Their remaining uses never become expressions, so the
/// definitions that feed them stay materialized.
fn use_is_consumed<H: LiftDialect>(shape: &Shape<H>, position: usize) -> bool {
    match shape {
        InstructionShape::Copy
        | InstructionShape::Operation(_)
        | InstructionShape::Store { .. }
        | InstructionShape::ParallelCopy
        | InstructionShape::BranchOperation(_)
        | InstructionShape::Return => true,
        InstructionShape::Branch | InstructionShape::Switch => position == 0,
        InstructionShape::Constant(_) | InstructionShape::ControlFlow => false,
    }
}

/// Whether the instruction can inline into a consumer as a value.
fn is_value_shape<H: LiftDialect>(shape: &Shape<H>) -> bool {
    matches!(
        shape,
        InstructionShape::Constant(_) | InstructionShape::Copy | InstructionShape::Operation(_)
    )
}

/// What an unwind out of each throwing instruction of a list reads.
///
/// A list can join several blocks, so each throwing instruction answers for
/// the unwind successors of its own block.
pub(super) struct UnwindReads<V> {
    /// Per list position, the block that a throwing instruction there
    /// unwinds out of.
    pub(super) blocks: Vec<Option<BlockId>>,
    /// Per such block, the variables live into its unwind successors.
    pub(super) live: BTreeMap<BlockId, BTreeSet<V>>,
}

impl<V: Ord> UnwindReads<V> {
    /// The variables that an unwind out of the instruction at `position`
    /// reads.
    fn at(&self, position: usize) -> Option<&BTreeSet<V>> {
        self.live.get(&self.blocks[position]?)
    }
}

/// The instructions that the presentation lift retains after local
/// dead-value pruning. A backward liveness walk removes the pure fallthrough
/// definitions whose results cannot reach a retained instruction, an unwind
/// successor, or another block.
pub(super) fn retained_positions<H, I, O>(
    operands: &O,
    instructions: &[&I],
    live_out: &BTreeSet<O::Variable>,
    unwind: &UnwindReads<O::Variable>,
) -> Vec<bool>
where
    H: LiftDialect,
    I: LiftInstruction<H>,
    O: LiftOperands<H, I>,
{
    let mut live = live_out.clone();
    let mut retained = vec![true; instructions.len()];
    for (position, &instruction) in instructions.iter().enumerate().rev() {
        let defs = operands.defs(instruction);
        let dead_value = !defs.is_empty()
            && defs.iter().all(|variable| !live.contains(variable))
            && operands.effects(instruction).is_empty()
            && !operands.may_throw(instruction)
            && instruction.flow_effect() == FlowEffect::Fallthrough;
        if dead_value {
            retained[position] = false;
            continue;
        }
        for definition in defs {
            live.remove(definition);
        }
        live.extend(operands.uses(instruction).iter().copied());
        if let Some(reads) = unwind.at(position) {
            live.extend(reads.iter().copied());
        }
    }
    retained
}

/// What the plan knows about one candidate expression tree.
struct CandidateFacts<H: LiftDialect, V> {
    /// Every variable that the tree reads.
    reads: BTreeSet<V>,
    /// The sorted, deduplicated effects of the whole tree.
    effects: Vec<H::Effect>,
    /// Whether any node of the tree may throw.
    may_throw: bool,
}

impl<H: LiftDialect, V> CandidateFacts<H, V> {
    /// Whether reordering this tree is observable at all.
    fn is_effect_relevant(&self) -> bool {
        !self.effects.is_empty() || self.may_throw
    }

    /// Whether the evaluation of this tree may move across an instruction
    /// with the given effect profile.
    fn commutes_with(&self, effects: &[H::Effect], may_throw: bool) -> bool {
        if !self.is_effect_relevant() || (effects.is_empty() && !may_throw) {
            return true;
        }
        H::evaluation_commutes(&self.effects, self.may_throw, effects, may_throw)
    }
}

/// The exact local single-use positions: one use between the definition and
/// the next redefinition of the variable, and dead beyond the list unless it
/// is redefined inside it. An unwind successor that reads the value before a
/// throwing instruction counts as a use that cannot inline.
fn single_use_positions<H, I, O>(
    operands: &O,
    instructions: &[&I],
    shapes: &[Shape<H>],
    live_out: &BTreeSet<O::Variable>,
    unwind: &UnwindReads<O::Variable>,
    retained: &[bool],
) -> Vec<Option<usize>>
where
    H: LiftDialect,
    I: LiftInstruction<H>,
    O: LiftOperands<H, I>,
{
    struct PendingUse {
        definition: usize,
        occurrences: usize,
        use_position: Option<usize>,
    }

    let mut viable: Vec<Option<usize>> = vec![None; instructions.len()];
    let mut active: BTreeMap<O::Variable, PendingUse> = BTreeMap::new();
    for (position, &instruction) in instructions.iter().enumerate() {
        if !retained[position] {
            continue;
        }

        // Reads precede writes. This keeps the segment semantics of a
        // read-modify-write instruction while it visits each operand once.
        for variable in operands.uses(instruction) {
            if let Some(pending) = active.get_mut(variable) {
                pending.occurrences = pending.occurrences.saturating_add(1);
                pending.use_position.get_or_insert(position);
            }
        }
        // The unwind leaves before the writes of this instruction.
        for variable in unwind.at(position).into_iter().flatten() {
            if let Some(pending) = active.get_mut(variable) {
                pending.occurrences = pending.occurrences.saturating_add(1);
            }
        }
        let defs = operands.defs(instruction);
        for variable in defs {
            if let Some(pending) = active.remove(variable)
                && pending.occurrences == 1
            {
                viable[pending.definition] = pending.use_position;
            }
        }

        if is_value_shape::<H>(&shapes[position])
            && defs.len() == 1
            && instruction.previous_value_operand().is_none()
        {
            // A read-modify-write definition merges with the prior state: it
            // is not a pure value and never inlines forward.
            active.insert(
                defs[0],
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

/// Decides, per definition, the consumer that it inlines into, if any.
pub(super) fn plan_inlining<H, I, O>(
    operands: &O,
    instructions: &[&I],
    shapes: &[Shape<H>],
    live_out: &BTreeSet<O::Variable>,
    unwind: &UnwindReads<O::Variable>,
    retained: &[bool],
) -> Vec<Option<usize>>
where
    H: LiftDialect,
    I: LiftInstruction<H>,
    O: LiftOperands<H, I>,
{
    let length = instructions.len();
    let viable =
        single_use_positions::<H, I, O>(operands, instructions, shapes, live_out, unwind, retained);

    // The order-safety walk over the movable candidates.
    let mut inline_at: Vec<Option<usize>> = vec![None; length];
    let mut facts: Vec<Option<CandidateFacts<H, O::Variable>>> =
        (0..length).map(|_| None).collect();
    let mut active: BTreeMap<O::Variable, usize> = BTreeMap::new();
    for (position, &instruction) in instructions.iter().enumerate() {
        if !retained[position] {
            continue;
        }
        // Consumption comes first: feeding this instruction is not a
        // crossing. Effect-relevant candidates that one instruction consumes
        // must keep their original relative order across its operands, since
        // operands evaluate left to right.
        let mut consumed: Vec<usize> = Vec::new();
        let mut last_effectful: Option<usize> = None;
        for (operand, variable) in operands.uses(instruction).iter().enumerate() {
            if !use_is_consumed::<H>(&shapes[position], operand)
                || instruction.previous_value_operand() == Some(operand)
            {
                // A previous-value read keeps its producer materialized: the
                // merge point must stay visible as a variable.
                continue;
            }
            if let Some(&candidate) = active.get(variable)
                && viable[candidate] == Some(position)
            {
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
                active.remove(variable);
                if relevant {
                    last_effectful = Some(candidate);
                }
            }
        }
        // An effectful or throwing instruction bars every candidate that the
        // dialect does not declare to commute with it.
        let effects = operands.effects(instruction);
        let throws = operands.may_throw(instruction);
        if !effects.is_empty() || throws {
            active.retain(|_, &mut candidate| {
                facts[candidate]
                    .as_ref()
                    .is_some_and(|f| f.commutes_with(effects, throws))
            });
        }
        let defs = operands.defs(instruction);
        for defined in defs {
            active.remove(defined);
            // A candidate that reads the redefined variable can no longer
            // move past this point.
            active.retain(|_, &mut candidate| {
                facts[candidate]
                    .as_ref()
                    .is_some_and(|f| !f.reads.contains(defined))
            });
        }
        if viable[position].is_some() && !instruction.materialize_value() {
            let mut reads = BTreeSet::new();
            let mut effects = effects.to_vec();
            let mut may_throw = throws;
            for variable in operands.uses(instruction) {
                let inlined = consumed
                    .iter()
                    .copied()
                    .find(|&candidate| operands.defs(instructions[candidate])[0] == *variable);
                if let Some(candidate) = inlined {
                    if let Some(f) = facts[candidate].as_ref() {
                        reads.extend(f.reads.iter().copied());
                        effects.extend(f.effects.iter().cloned());
                        may_throw |= f.may_throw;
                    }
                } else {
                    reads.insert(*variable);
                }
            }
            effects.sort_unstable();
            effects.dedup();
            facts[position] = Some(CandidateFacts {
                reads,
                effects,
                may_throw,
            });
            active.insert(defs[0], position);
        }
    }
    inline_at
}
