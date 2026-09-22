//! Canonical rebuilds of one MLIL function after a graph-level decision.
//!
//! [`Function::dead_code_eliminated_cfg`] and
//! [`Function::copy_propagated_cfg`] answer with a *derived* graph: the
//! canonical function is untouched and its provenance still names the
//! instructions the transform dropped. That is the right answer for
//! presentation and the wrong one for storage, because such a function no
//! longer verifies and every door that verifies first — SSA, sparse
//! constants, the HLIL bridge — then refuses it.
//!
//! The doors here take the same decision and rebuild from it. Blocks,
//! edges, exception regions, cleanup routes, variables, and the signature
//! are the ones the source held; the surviving instructions keep their
//! graph order, and the provenance of a dropped instruction goes with it.
//! Instruction identities become dense again, so they are not the
//! source's — a caller holding an identity-keyed side table rebuilds it.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::{BlockId, Cfg, CopyPropagationStats, DeadCode};

use super::variable::{typed, unchanged};
use super::{
    AnalysisDialect, EntityId, Function, FunctionBuilder, Instruction, InstructionId, Result,
    VariableId, VerifyDialect,
};

impl<D: VerifyDialect> Function<D> {
    /// Returns the function without the definitions nothing reads, and
    /// how many instructions that removed.
    ///
    /// `live_out` states what leaves each block, which a graph read from
    /// the inside cannot know: a returned value, the storage a caller
    /// reads again, the state something reached through a non-returning
    /// exit observes. A caller normally answers for the blocks with no
    /// successors and hands back an empty vector everywhere else, as in
    /// [`DeadCode::compute_with_exits`].
    ///
    /// An instruction with declared effects is never removed, and neither
    /// is one that may throw: it is the throw site the exceptional edges
    /// of its block leave from, so it stays whatever liveness says about
    /// what it defines. A function with nothing to remove is returned
    /// unchanged, identities and all.
    ///
    /// # Errors
    ///
    /// Returns an error when the rebuilt function fails structural or
    /// dialect verification.
    pub fn eliminate_dead_code(
        &self,
        live_out: impl Fn(BlockId) -> Vec<VariableId>,
    ) -> Result<(Self, usize)> {
        let dead = DeadCode::compute_with_exits(&self.cfg, live_out);
        let dropped: BTreeSet<InstructionId> = dead
            .instructions
            .iter()
            .filter_map(|point| {
                let instruction = self
                    .cfg
                    .block(point.block)
                    .instructions()
                    .get(point.inst_idx)?;
                (instruction.effects().is_empty() && !instruction.may_throw())
                    .then(|| instruction.id())
            })
            .collect();
        if dropped.is_empty() {
            return Ok((self.clone(), 0));
        }
        let function = rebuild(
            self,
            &self.cfg,
            |instruction| !dropped.contains(&instruction.id()),
            unchanged,
        )?;
        Ok((function, dropped.len()))
    }
}

impl<D: AnalysisDialect + VerifyDialect> Function<D> {
    /// Returns the function with every provably value-preserving copy
    /// propagated into its readers and then removed.
    ///
    /// A removed copy loses its provenance with its identity; a function
    /// with no copy to propagate is returned unchanged. A function whose
    /// result is only ever a copy of one of its inputs loses the
    /// definition of that result, because nothing inside the function
    /// reads it — state what leaves through
    /// [`Self::propagate_copies_with_exits`].
    ///
    /// # Errors
    ///
    /// Returns an error when the rebuilt function fails structural or
    /// dialect verification.
    pub fn propagate_copies(&self) -> Result<(Self, CopyPropagationStats)> {
        self.propagate_copies_with_exits(|_| Vec::new())
    }

    /// [`Self::propagate_copies`] told what leaves each block.
    ///
    /// A copy whose definition is live at an exit under `live_out` is
    /// never removed: something outside the function reads it, and
    /// removing it would leave that place undefined where the caller
    /// looks. Its uses inside the function are rewritten to the source
    /// all the same. A caller normally answers for the blocks with no
    /// successors and hands back an empty vector everywhere else.
    ///
    /// # Errors
    ///
    /// Returns an error when the rebuilt function fails structural or
    /// dialect verification.
    pub fn propagate_copies_with_exits(
        &self,
        live_out: impl Fn(BlockId) -> Vec<VariableId>,
    ) -> Result<(Self, CopyPropagationStats)> {
        let mut cfg = self.cfg.clone();
        let statistics = crate::copy_propagation_with_exits(&mut cfg, live_out);
        if statistics.copies_removed == 0 && statistics.uses_rewritten == 0 {
            return Ok((self.clone(), statistics));
        }
        Ok((rebuild(self, &cfg, |_| true, unchanged)?, statistics))
    }
}

/// Rebuilds one canonical function from `cfg`, keeping the instructions
/// `keep` accepts and renaming every variable occurrence through
/// `rename`.
///
/// `cfg` holds the source's blocks, edges, exception regions, and cleanup
/// routes; its instructions are the source's, possibly fewer and with
/// rewritten operands. Every non-instruction identity survives, the kept
/// instructions renumber densely in graph order, and a provenance entry
/// naming an instruction that did not survive is dropped with it. The
/// variable table is copied whole, so a renaming leaves the variables it
/// renamed away declared and unoccurring — [`Function::prune_variables`]
/// is what drops them.
pub(super) fn rebuild<D: VerifyDialect>(
    source: &Function<D>,
    cfg: &Cfg<Instruction<D>, D::Edge>,
    keep: impl Fn(&Instruction<D>) -> bool,
    rename: impl Fn(VariableId) -> VariableId,
) -> Result<Function<D>> {
    let mut builder = FunctionBuilder::<D>::new(source.source().clone());
    for variable in source.variables() {
        let rebuilt = builder.declare_variable(variable.role.clone(), variable.native.clone())?;
        debug_assert_eq!(rebuilt, variable.id);
    }
    builder.copy_blocks(cfg);

    let mut rebuilt_instructions: BTreeMap<InstructionId, InstructionId> = BTreeMap::new();
    for block in cfg.block_ids() {
        for instruction in cfg.block(block).instructions() {
            if !keep(instruction) {
                continue;
            }
            let rebuilt = builder.append_instruction(
                block,
                instruction.operation().clone(),
                typed::<D>(instruction.uses(), instruction.use_types(), &rename),
                typed::<D>(instruction.defs(), instruction.def_types(), &rename),
                instruction.may_throw(),
                None,
            )?;
            rebuilt_instructions.insert(instruction.id(), rebuilt);
        }
    }

    builder.copy_structure(cfg)?;
    builder.set_signature(source.signature().clone())?;
    for entry in source.provenance().entries() {
        let entity = match entry.entity {
            EntityId::Instruction(instruction) => match rebuilt_instructions.get(&instruction) {
                Some(&rebuilt) => EntityId::Instruction(rebuilt),
                None => continue,
            },
            other => other,
        };
        builder.map_entity(entry.source.clone(), entity)?;
    }
    builder.finish()
}
