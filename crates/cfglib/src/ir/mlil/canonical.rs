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
use alloc::format;
use alloc::vec::Vec;

use crate::{BlockId, Cfg, CopyPropagationStats, DeadCode, FlowControl, FlowEffect};

use super::variable::{typed, unchanged};
use super::{
    AnalysisDialect, EntityId, Function, FunctionBuilder, Instruction, InstructionId,
    ProvenanceMap, Result, VariableId, VerifyDialect,
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
        let dropped = self.dead_instructions(live_out);
        if dropped.is_empty() {
            return Ok((self.clone(), 0));
        }
        let function = self.without(&dropped)?;
        Ok((function, dropped.len()))
    }

    /// [`Self::eliminate_dead_code`] on this function: removes the dead
    /// instructions and returns how many it removed.
    ///
    /// The kept instructions are numbered again densely in block order, as
    /// the rebuild of [`Self::eliminate_dead_code`] numbers them. A function
    /// with nothing to remove does no other work.
    pub fn eliminate_dead_code_in_place(
        &mut self,
        live_out: impl Fn(BlockId) -> Vec<VariableId>,
    ) -> usize {
        let dropped = self.dead_instructions(live_out);
        if !dropped.is_empty() {
            self.remove(&dropped);
        }
        dropped.len()
    }

    /// Returns the function without the instructions that `remove`
    /// selects, and how many instructions that removed.
    ///
    /// A consumer removes an instruction when another statement already
    /// states what it does: a call that states the push of its own return
    /// address, a return that states the pop of it, or a copy of a variable
    /// into itself. The removal is checked. A removed instruction must not
    /// transfer control, and no kept instruction may read what a removed one
    /// defines, unless the removed one reads that variable too: the readers
    /// then see the definition that reached the removed instruction. What
    /// leaves the function is outside the graph, so the caller answers for a
    /// definition that only an exit observes. A function with nothing to
    /// remove is returned unchanged, identities and all.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConstruction`](super::Error::InvalidConstruction)
    /// when a selected instruction transfers control or a kept instruction
    /// reads one of its definitions that it does not read itself, and an
    /// error when the rebuilt function fails structural or dialect
    /// verification.
    pub fn remove_instructions(
        &self,
        remove: impl Fn(&Instruction<D>) -> bool,
    ) -> Result<(Self, usize)> {
        let removed = self.removable(remove)?;
        if removed.is_empty() {
            return Ok((self.clone(), 0));
        }
        let function = self.without(&removed)?;
        Ok((function, removed.len()))
    }

    /// [`Self::remove_instructions`] on this function: removes the
    /// selected instructions and returns how many it removed.
    ///
    /// The kept instructions are numbered again densely in block order, as
    /// the rebuild of [`Self::remove_instructions`] numbers them. A function
    /// with nothing to remove does no other work.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConstruction`](super::Error::InvalidConstruction)
    /// when a selected instruction transfers control or a kept instruction
    /// reads one of its definitions that it does not read itself. The
    /// function then stays as it was.
    pub fn remove_instructions_in_place(
        &mut self,
        remove: impl Fn(&Instruction<D>) -> bool,
    ) -> Result<usize> {
        let removed = self.removable(remove)?;
        if !removed.is_empty() {
            self.remove(&removed);
        }
        Ok(removed.len())
    }

    /// Returns the instructions without effects that define only what no
    /// reader and no exit under `live_out` observes.
    fn dead_instructions(
        &self,
        live_out: impl Fn(BlockId) -> Vec<VariableId>,
    ) -> BTreeSet<InstructionId> {
        let dead = DeadCode::compute_with_exits(&self.cfg, live_out);
        dead.instructions
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
            .collect()
    }

    /// Returns the instructions that `remove` selects, after the checks of
    /// [`Self::remove_instructions`].
    fn removable(
        &self,
        remove: impl Fn(&Instruction<D>) -> bool,
    ) -> Result<BTreeSet<InstructionId>> {
        let mut removed = BTreeSet::new();
        let mut read = BTreeSet::new();
        for block in self.cfg.blocks() {
            for instruction in block.instructions() {
                if remove(instruction) {
                    if !matches!(
                        instruction.flow_effect(),
                        FlowEffect::Fallthrough | FlowEffect::MayThrow
                    ) {
                        return Err(super::Error::InvalidConstruction(format!(
                            "instruction {} transfers control and cannot be removed",
                            instruction.id()
                        )));
                    }
                    removed.insert(instruction.id());
                } else {
                    read.extend(instruction.uses().iter().copied());
                }
            }
        }
        if removed.is_empty() {
            return Ok(removed);
        }
        for block in self.cfg.blocks() {
            for instruction in block.instructions() {
                if removed.contains(&instruction.id())
                    && let Some(defined) = instruction
                        .defs()
                        .iter()
                        .find(|id| read.contains(id) && !instruction.uses().contains(id))
                {
                    return Err(super::Error::InvalidConstruction(format!(
                        "instruction {} defines {defined}, which a kept instruction reads",
                        instruction.id()
                    )));
                }
            }
        }
        Ok(removed)
    }

    /// Rebuilds the function without the instructions of `removed`.
    fn without(&self, removed: &BTreeSet<InstructionId>) -> Result<Self> {
        rebuild(
            self,
            &self.cfg,
            |instruction| !removed.contains(&instruction.id()),
            unchanged,
        )
    }

    /// Removes the instructions of `removed` and numbers the rest densely in
    /// block order, as a rebuild of the function numbers them.
    ///
    /// The provenance of a removed instruction goes with it. The graph, the
    /// variables, the signature, and every other correspondence stay. A
    /// caller removes only an instruction that transfers no control and
    /// whose definitions no kept instruction reads, so the function still
    /// verifies.
    fn remove(&mut self, removed: &BTreeSet<InstructionId>) {
        let mut renumbered: Vec<Option<InstructionId>> =
            alloc::vec![None; self.instruction_count()];
        let mut next = 0usize;
        let blocks: Vec<BlockId> = self.cfg.block_ids().collect();
        for block in blocks {
            let instructions = self.cfg.block_mut(block).instructions_mut();
            instructions.retain(|instruction| !removed.contains(&instruction.id()));
            for instruction in instructions.iter_mut() {
                // The kept identities number fewer than the old ones, which
                // fit the identity space.
                let id = InstructionId::from_raw(u32::try_from(next).unwrap_or(u32::MAX));
                next += 1;
                if let Some(slot) = renumbered.get_mut(instruction.id().index()) {
                    *slot = Some(id);
                }
                instruction.set_id(id);
            }
        }
        self.instruction_points = alloc::vec![None; next];
        self.reindex_instructions();
        let mut provenance = ProvenanceMap::new(self.provenance.source().clone());
        for entry in self.provenance.entries() {
            let entity = match entry.entity {
                EntityId::Instruction(id) => match renumbered.get(id.index()).copied().flatten() {
                    Some(id) => EntityId::Instruction(id),
                    None => continue,
                },
                other => other,
            };
            // A stored entry has a valid span, so the insert cannot fail.
            let _ = provenance.insert(entry.source.clone(), entity);
        }
        self.provenance = provenance;
        debug_assert!(self.verify().is_ok(), "a removal keeps the function valid");
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
