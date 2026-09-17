//! Memory-to-variable promotion over canonical MLIL functions.
//!
//! Machine frontends lift storage the ISA forces through memory — stack
//! frame slots, spill slots — as load/store operations even when nothing
//! ever aliases the location. [`Function::promote_memory`] rewrites the
//! accesses of every unaliased location into reads and writes of one fresh
//! mutable variable, moving the value into the dataflow-visible world where
//! SSA, expression inlining, and [`Function::split_variables`] apply.
//!
//! The consumer's [`PromoteDialect`] owns the memory judgment — which
//! instructions access which candidate locations, and which instructions
//! disqualify a location by taking its address or reaching it opaquely.
//! Reporting a location asserts the aliasing contract for it: nothing the
//! classification does not name can read or write it. The library owns the
//! sound part: shape validation, conservative disqualification, and an
//! identity-preserving rewrite.
//!
//! Each rewritten access becomes a dialect copy at the same instruction
//! identity — nothing is deleted, so blocks, edges, instructions, regions,
//! and provenance all keep their ids, and the ordinary cleanup passes
//! (copy propagation, dead-code elimination) tidy the residue afterwards.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt::Debug;

use super::{
    Dialect, Function, FunctionBuilder, Instruction, InstructionId, Result, TypedVariable,
    VariableId, VerifyDialect,
};

/// One instruction's relationship to promotable memory locations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromotionAccess<L> {
    /// The instruction reads `location` into its single definition.
    Load(L),
    /// The instruction writes its **last** use into `location` (the same
    /// convention as [`Lifted::Store`](crate::ir::hlil::Lifted::Store)).
    Store(L),
    /// The instruction cannot read or write any reported location.
    Unrelated,
    /// These locations may be aliased here (address taken, escaped):
    /// they are disqualified from promotion.
    Escape(Vec<L>),
    /// The instruction may reach any location opaquely: every location is
    /// disqualified.
    EscapeAll,
}

/// The consumer memory judgment driving [`Function::promote_memory`].
pub trait PromoteDialect: Dialect {
    /// Consumer identity of one promotable memory location (a frame slot,
    /// a spill index) whose address is statically fixed.
    type Location: Clone + Debug + Ord;

    /// Classifies one instruction against the candidate locations.
    ///
    /// Reporting [`Load`](PromotionAccess::Load) or
    /// [`Store`](PromotionAccess::Store) for a location asserts that every
    /// access to it is classified; [`Unrelated`](PromotionAccess::Unrelated)
    /// asserts the instruction cannot touch any reported location. When
    /// that cannot be proven, return [`Escape`](PromotionAccess::Escape) or
    /// [`EscapeAll`](PromotionAccess::EscapeAll).
    fn promotion_access(instruction: &Instruction<Self>) -> PromotionAccess<Self::Location>;

    /// The dialect copy operation rewritten accesses become (one use, one
    /// definition).
    fn copy_operation() -> Self::Operation;

    /// The role and native provenance of the variable standing in for one
    /// promoted location.
    fn promoted_variable(
        location: &Self::Location,
    ) -> (Self::VariableRole, Option<Self::NativeVariable>);
}

/// The result of promoting one function's unaliased memory locations.
#[derive(Debug, Clone)]
pub struct MemoryPromotion<D: PromoteDialect> {
    /// The rebuilt function. Every non-variable identity is unchanged;
    /// promoted-location variables are appended after the existing table.
    pub function: Function<D>,
    /// Promoted location → the variable now carrying it.
    pub promoted: BTreeMap<D::Location, VariableId>,
    /// How many load/store instructions were rewritten into copies.
    pub rewritten: usize,
}

/// The locations that survived classification: accessed somewhere, never
/// escaped, and well-shaped at every access.
fn promotable_locations<D: PromoteDialect>(source: &Function<D>) -> BTreeSet<D::Location> {
    let mut seen: BTreeSet<D::Location> = BTreeSet::new();
    let mut escaped: BTreeSet<D::Location> = BTreeSet::new();
    for instruction in source.instructions() {
        match D::promotion_access(instruction) {
            PromotionAccess::Load(location) => {
                if instruction.defs().len() != 1 {
                    escaped.insert(location.clone());
                }
                seen.insert(location);
            }
            PromotionAccess::Store(location) => {
                if instruction.uses().is_empty() {
                    escaped.insert(location.clone());
                }
                seen.insert(location);
            }
            PromotionAccess::Unrelated => {}
            PromotionAccess::Escape(locations) => escaped.extend(locations),
            PromotionAccess::EscapeAll => return BTreeSet::new(),
        }
    }
    seen.difference(&escaped).cloned().collect()
}

pub(super) fn promote_memory<D>(source: &Function<D>) -> Result<MemoryPromotion<D>>
where
    D: PromoteDialect + VerifyDialect,
{
    let report = source.verify();
    if !report.is_ok() {
        return Err(report.into());
    }
    let locations = promotable_locations(source);
    if locations.is_empty() {
        return Ok(MemoryPromotion {
            function: source.clone(),
            promoted: BTreeMap::new(),
            rewritten: 0,
        });
    }

    let mut builder = FunctionBuilder::<D>::new(source.source().clone());
    for variable in source.variables() {
        builder.declare_variable(variable.role.clone(), variable.native.clone())?;
    }
    let mut promoted: BTreeMap<D::Location, VariableId> = BTreeMap::new();
    for location in &locations {
        let (role, native) = D::promoted_variable(location);
        promoted.insert(location.clone(), builder.declare_variable(role, native)?);
    }

    builder.copy_blocks(&source.cfg);
    let mut rewritten = 0usize;
    for index in 0..source.instruction_count() {
        let id = InstructionId::from_raw(
            u32::try_from(index).expect("existing identities fit their own space"),
        );
        let point = source
            .instruction_point(id)
            .expect("dense identities cover the table");
        let instruction = source
            .instruction(id)
            .expect("a verified function stores every indexed instruction");
        let rewrite = match D::promotion_access(instruction) {
            PromotionAccess::Load(location) => promoted.get(&location).map(|&slot| {
                let value_type = instruction.def_types()[0].clone();
                (
                    alloc::vec![TypedVariable::new(slot, value_type.clone())],
                    alloc::vec![TypedVariable::new(instruction.defs()[0], value_type)],
                )
            }),
            PromotionAccess::Store(location) => promoted.get(&location).map(|&slot| {
                let position = instruction.uses().len() - 1;
                let value = instruction.uses()[position];
                let value_type = instruction.use_types()[position].clone();
                (
                    alloc::vec![TypedVariable::new(value, value_type.clone())],
                    alloc::vec![TypedVariable::new(slot, value_type)],
                )
            }),
            _ => None,
        };
        let rebuilt = if let Some((uses, defs)) = rewrite {
            rewritten += 1;
            // A classified access to an unaliased location neither faults
            // nor keeps its memory effects: it is a plain copy now.
            builder.append_instruction(point.block, D::copy_operation(), uses, defs, false, None)?
        } else {
            let uses = instruction
                .uses()
                .iter()
                .zip(instruction.use_types())
                .map(|(&variable, value_type)| TypedVariable::new(variable, value_type.clone()))
                .collect();
            let defs = instruction
                .defs()
                .iter()
                .zip(instruction.def_types())
                .map(|(&variable, value_type)| TypedVariable::new(variable, value_type.clone()))
                .collect();
            builder.append_instruction(
                point.block,
                instruction.operation().clone(),
                uses,
                defs,
                instruction.may_throw(),
                None,
            )?
        };
        debug_assert_eq!(rebuilt, id);
    }
    builder.copy_structure(&source.cfg)?;
    builder.copy_metadata(source.signature.clone(), &source.provenance)?;

    Ok(MemoryPromotion {
        function: builder.finish()?,
        promoted,
        rewritten,
    })
}
