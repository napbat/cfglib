//! Dropping the variables of one MLIL function that nothing names.
//!
//! A pass that rewrites operands leaves declarations behind: an
//! eliminated definition was the last reader of its source, a propagated
//! copy was the only occurrence of its target. Such a declaration stays
//! legal — an unoccurring variable verifies — but it outlives its
//! meaning, and a consumer that renders or allocates one variable per
//! declaration pays for every one of them.
//!
//! [`Function::prune_variables`] keeps the variables something still
//! names: an instruction use or definition, a signature parameter, or a
//! provenance entry. The rest go, and the survivors renumber densely in
//! declaration order. Block, edge, instruction, and region identities are
//! untouched, so [`VariablePruning`] is the only map a caller consults
//! afterward.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use super::variable::typed;
use super::{
    EntityId, Function, FunctionBuilder, InstructionId, Result, Signature, TypedVariable,
    VariableId, VerifyDialect,
};

/// The result of pruning one function's unreferenced variables.
///
/// This is the **variable** axis of identity rewriting, exactly as
/// [`VariableSplit`](super::VariableSplit) is: a variable is not a graph
/// entity, so neither a [`Rewrite`](crate::Rewrite) nor a
/// [`Renumbering`](crate::Renumbering) has anything to say about it. The
/// naming is the same on both axes: an old identity maps to the identity
/// that replaced it, and the reverse index answers where a new identity
/// came from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VariablePruning {
    /// New variable → the original variable it was renumbered from, dense
    /// by new identity index.
    pub origins: Vec<VariableId>,
    /// Original variable → the identity it kept. A pruned variable is
    /// absent.
    pub kept: BTreeMap<VariableId, VariableId>,
}

impl VariablePruning {
    /// The identity one surviving variable kept.
    ///
    /// Every variable an instruction or the signature names survives by
    /// construction, so this is total over the occurrences a rebuild
    /// reads.
    fn renumbered(&self, variable: VariableId) -> VariableId {
        *self
            .kept
            .get(&variable)
            .expect("a named variable is never pruned")
    }
}

impl<D: VerifyDialect> Function<D> {
    /// Returns the function without the variables nothing names, and the
    /// identity map both ways.
    ///
    /// A variable survives when an instruction uses or defines it, the
    /// signature names it as a parameter, or a provenance entry names it;
    /// the survivors renumber densely in declaration order. Blocks,
    /// edges, instructions, and regions keep their identities, order, and
    /// payloads. A function whose every variable is named is returned
    /// unchanged, with an identity map.
    ///
    /// # Errors
    ///
    /// Returns an error when the rebuilt function fails structural or
    /// dialect verification.
    pub fn prune_variables(&self) -> Result<(Self, VariablePruning)> {
        prune(self)
    }
}

/// Rebuilds `source` without the variables nothing names.
fn prune<D: VerifyDialect>(source: &Function<D>) -> Result<(Function<D>, VariablePruning)> {
    let referenced = referenced_variables(source);
    if referenced.len() == source.variables().len() {
        return Ok((source.clone(), identity_pruning(source)));
    }

    let mut builder = FunctionBuilder::<D>::new(source.source().clone());
    let mut pruning = VariablePruning::default();
    for variable in source.variables() {
        if !referenced.contains(&variable.id) {
            continue;
        }
        let renumbered =
            builder.declare_variable(variable.role.clone(), variable.native.clone())?;
        pruning.kept.insert(variable.id, renumbered);
        pruning.origins.push(variable.id);
    }

    // Instructions rebuild in identity order at their own points, which
    // is what keeps every instruction identity and position.
    builder.copy_blocks(&source.cfg);
    for index in 0..source.instruction_count() {
        let raw = u32::try_from(index).expect("existing identities fit their own space");
        let id = InstructionId::from_raw(raw);
        let Some(point) = source.instruction_point(id) else {
            continue;
        };
        let instruction = source
            .instruction(id)
            .expect("a placed identity names a stored instruction");
        let rebuilt = builder.append_instruction(
            point.block,
            instruction.operation().clone(),
            renumber(
                &pruning,
                typed::<D>(instruction.uses(), instruction.use_types()),
            ),
            renumber(
                &pruning,
                typed::<D>(instruction.defs(), instruction.def_types()),
            ),
            instruction.may_throw(),
            None,
        )?;
        debug_assert_eq!(rebuilt, id);
    }
    builder.copy_structure(&source.cfg)?;

    let parameters = source
        .signature()
        .parameters
        .iter()
        .map(|&parameter| pruning.renumbered(parameter))
        .collect();
    builder.set_signature(Signature::<D>::new(
        parameters,
        source.signature().returns.clone(),
    ))?;
    for entry in source.provenance().entries() {
        let entity = match entry.entity {
            EntityId::Variable(variable) => match pruning.kept.get(&variable) {
                Some(&renumbered) => EntityId::Variable(renumbered),
                None => continue,
            },
            other => other,
        };
        builder.map_entity(entry.source.clone(), entity)?;
    }

    Ok((builder.finish()?, pruning))
}

/// Every variable some instruction, the signature, or the provenance
/// names.
fn referenced_variables<D: VerifyDialect>(function: &Function<D>) -> BTreeSet<VariableId> {
    let mut referenced = BTreeSet::new();
    for block in function.cfg.block_ids() {
        for instruction in function.cfg.block(block).instructions() {
            referenced.extend(instruction.uses().iter().chain(instruction.defs()).copied());
        }
    }
    referenced.extend(function.signature().parameters.iter().copied());
    for entry in function.provenance().entries() {
        if let EntityId::Variable(variable) = entry.entity {
            referenced.insert(variable);
        }
    }
    referenced
}

/// The map of a function that pruned nothing.
fn identity_pruning<D: VerifyDialect>(function: &Function<D>) -> VariablePruning {
    VariablePruning {
        origins: function
            .variables()
            .iter()
            .map(|variable| variable.id)
            .collect(),
        kept: function
            .variables()
            .iter()
            .map(|variable| (variable.id, variable.id))
            .collect(),
    }
}

fn renumber<D: VerifyDialect>(
    pruning: &VariablePruning,
    occurrences: Vec<TypedVariable<D>>,
) -> Vec<TypedVariable<D>> {
    occurrences
        .into_iter()
        .map(|occurrence| {
            TypedVariable::new(
                pruning.renumbered(occurrence.variable),
                occurrence.value_type,
            )
        })
        .collect()
}
