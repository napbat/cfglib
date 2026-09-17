//! SSA-web variable splitting over canonical MLIL functions.
//!
//! Frontends commonly reuse one storage-derived variable for several
//! unrelated lifetimes (a stack slot holding an integer at one point and a
//! reference at another). [`Function::split_variables`] separates those
//! lifetimes: it computes the function's own SSA view, partitions the SSA
//! values into phi congruence classes ([`PhiWebs`]), and rebuilds the
//! canonical function with one fresh variable per class.
//!
//! The partition is sound because the SSA view is computed directly from
//! the canonical function (conventional SSA, no SSA-level rewriting), so
//! phi congruence classes are interference-free: every use's reaching
//! definitions lie in its own class, and classes of the same original
//! variable never overlap on any path.
//!
//! The rebuild preserves every non-variable identity — blocks, edges,
//! instructions, and regions keep their ids, order, payloads, and
//! provenance — so downstream side tables keyed by those identities remain
//! valid. Only variable identities change, and [`VariableSplit`] maps both
//! directions.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::dataflow::phi_web::PhiWebs;
use crate::dataflow::ssa::{SsaValue, SsaVersion};

use super::{
    EntityId, Function, FunctionBuilder, InstructionId, Result, Signature, TypedVariable,
    VariableId, VerifyDialect,
};

/// The result of splitting one function's variables by SSA phi-webs.
///
/// This is the **variable** axis of identity rewriting, which is deliberately
/// separate from the graph axis: a variable is not a graph entity, so a
/// [`Rewrite`](crate::Rewrite) (blocks and edges) and a
/// [`Renumbering`](crate::Renumbering) (a compacted store) have nothing to
/// say about it, and folding the three together would only hide which
/// identities a given pass can actually move. The naming is the same on both
/// axes: an old identity maps to the identities that replaced it, and the
/// reverse index answers where a new identity came from.
#[derive(Debug, Clone)]
pub struct VariableSplit<D: super::Dialect> {
    /// The rebuilt function. Block, edge, instruction, and region
    /// identities are unchanged; only variables were renumbered.
    pub function: Function<D>,
    /// New variable → the original variable it was split from, dense by
    /// new identity index.
    pub origins: Vec<VariableId>,
    /// Original variable → the variables split from it, in deterministic
    /// SSA-version order.
    pub splits: BTreeMap<VariableId, Vec<VariableId>>,
}

/// One congruence class of a single original variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ClassKey {
    /// A phi web, by web index.
    Web(usize),
    /// A standalone SSA version with no phi connection.
    Version(SsaVersion),
}

fn class_key(webs: &PhiWebs<VariableId>, value: &SsaValue<VariableId>) -> ClassKey {
    webs.web_of
        .get(value)
        .map_or(ClassKey::Version(value.version), |&web| ClassKey::Web(web))
}

fn remap(
    webs: &PhiWebs<VariableId>,
    assigned: &BTreeMap<(VariableId, ClassKey), VariableId>,
    value: &SsaValue<VariableId>,
) -> VariableId {
    *assigned
        .get(&(value.variable, class_key(webs, value)))
        .expect("every occurring SSA value was assigned a split class")
}

/// Every occurring (variable, version) pair: instruction uses and
/// definitions, values carried by live phi webs, and signature parameters'
/// live-in versions (the entry defines them even when unread). Dead-phi
/// versions never occur, so they allocate nothing.
fn collect_occurrences<D: VerifyDialect>(
    source: &Function<D>,
    ssa: &crate::SsaForm<VariableId>,
    webs: &PhiWebs<VariableId>,
) -> BTreeMap<VariableId, BTreeSet<SsaVersion>> {
    let mut occurrences: BTreeMap<VariableId, BTreeSet<SsaVersion>> = BTreeMap::new();
    let mut note = |value: &SsaValue<VariableId>| {
        occurrences
            .entry(value.variable)
            .or_default()
            .insert(value.version);
    };
    for value in webs.web_of.keys() {
        note(value);
    }
    for block in ssa.blocks() {
        for instruction in &block.instructions {
            for value in instruction.uses.iter().chain(&instruction.defs) {
                note(value);
            }
        }
    }
    for &parameter in &source.signature.parameters {
        occurrences
            .entry(parameter)
            .or_default()
            .insert(SsaValue::live_in(parameter).version);
    }
    occurrences
}

struct Allocation<D: super::Dialect> {
    builder: FunctionBuilder<D>,
    assigned: BTreeMap<(VariableId, ClassKey), VariableId>,
    origins: Vec<VariableId>,
    splits: BTreeMap<VariableId, Vec<VariableId>>,
}

/// Allocate one fresh variable per class, grouped by original variable in
/// identity order, then by first occurring version — deterministic.
fn allocate<D: VerifyDialect>(
    source: &Function<D>,
    webs: &PhiWebs<VariableId>,
    occurrences: &BTreeMap<VariableId, BTreeSet<SsaVersion>>,
) -> Result<Allocation<D>> {
    let mut builder = FunctionBuilder::<D>::new(source.source().clone());
    let mut assigned: BTreeMap<(VariableId, ClassKey), VariableId> = BTreeMap::new();
    let mut origins = Vec::new();
    let mut splits: BTreeMap<VariableId, Vec<VariableId>> = BTreeMap::new();
    for variable in source.variables() {
        if let Some(versions) = occurrences.get(&variable.id) {
            for &version in versions {
                let key = class_key(webs, &SsaValue::new(variable.id, version));
                if assigned.contains_key(&(variable.id, key)) {
                    continue;
                }
                let split =
                    builder.declare_variable(variable.role.clone(), variable.native.clone())?;
                assigned.insert((variable.id, key), split);
                origins.push(variable.id);
                splits.entry(variable.id).or_default().push(split);
            }
        }
        if let alloc::collections::btree_map::Entry::Vacant(vacant) = splits.entry(variable.id) {
            // Declared but never occurring: keep an identity variable so
            // provenance and consumer side tables stay resolvable.
            let split = builder.declare_variable(variable.role.clone(), variable.native.clone())?;
            origins.push(variable.id);
            vacant.insert(alloc::vec![split]);
        }
    }
    Ok(Allocation {
        builder,
        assigned,
        origins,
        splits,
    })
}

pub(super) fn split_variables<D: VerifyDialect>(source: &Function<D>) -> Result<VariableSplit<D>> {
    let ssa = source.ssa()?;
    // Live webs only: an unread phi must not unite unrelated lifetimes.
    let webs = PhiWebs::compute_live(&ssa);
    let occurrences = collect_occurrences(source, &ssa, &webs);
    let Allocation {
        mut builder,
        assigned,
        origins,
        splits,
    } = allocate(source, &webs, &occurrences)?;

    // Identical graph structure: blocks in identity order, instructions in
    // stable identity order appended at their original points, edges and
    // regions in insertion order — every rebuilt identity matches.
    builder.copy_blocks(&source.cfg);
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
        let renamed = ssa
            .instruction(point)
            .expect("the SSA view covers every program point of its function");
        let uses = renamed
            .uses
            .iter()
            .zip(instruction.use_types())
            .map(|(value, value_type)| {
                TypedVariable::new(remap(&webs, &assigned, value), value_type.clone())
            })
            .collect();
        let defs = renamed
            .defs
            .iter()
            .zip(instruction.def_types())
            .map(|(value, value_type)| {
                TypedVariable::new(remap(&webs, &assigned, value), value_type.clone())
            })
            .collect();
        let rebuilt = builder.append_instruction(
            point.block,
            instruction.operation().clone(),
            uses,
            defs,
            instruction.may_throw(),
            None,
        )?;
        debug_assert_eq!(rebuilt, id);
    }
    builder.copy_structure(&source.cfg)?;

    let parameters = source
        .signature
        .parameters
        .iter()
        .map(|&parameter| remap(&webs, &assigned, &SsaValue::live_in(parameter)))
        .collect();
    builder.set_signature(Signature::<D>::new(
        parameters,
        source.signature.returns.clone(),
    ))?;

    for entry in source.provenance.entries() {
        match entry.entity {
            EntityId::Variable(variable) => {
                for &split in splits.get(&variable).map(Vec::as_slice).unwrap_or_default() {
                    builder.map_entity(entry.source.clone(), EntityId::Variable(split))?;
                }
            }
            other => {
                builder.map_entity(entry.source.clone(), other)?;
            }
        }
    }

    Ok(VariableSplit {
        function: builder.finish()?,
        origins,
        splits,
    })
}
