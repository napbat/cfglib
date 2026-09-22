//! Coalescing a lifter temporary into the variable it is copied to.
//!
//! A lift that computes into a temporary and then commits the result to a
//! native location leaves the value living in the temporary and the named
//! variable surviving only as a copy of it — `v8 = sub(v3, 0x28); v4 =
//! v8; call(.., v8, ..)`, where `v4` is the register web and `v8` is the
//! lifter's. Copy propagation makes that worse rather than better: it
//! rewrites the readers to the temporary, which is the direction away
//! from the name.
//!
//! [`Function::coalesce_copies`] runs it the other way. Where a temporary
//! is defined once, read only from that definition, and copied into a
//! variable that is defined only by that copy, the two are one value with
//! two names; the definition is rewritten to define the named variable,
//! the readers follow, and the copy goes. The caller says which variables
//! are the lift's own through `is_temporary`, because only the frontend
//! knows.
//!
//! The coalesced variable is the copy's destination, so it keeps that
//! variable's role and native provenance; each occurrence keeps the value
//! type its own instruction carried. The temporary stays declared and
//! stops occurring — [`Function::prune_variables`] is what drops it.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::dataflow::ssa::{SsaForm, SsaVersion};
use crate::{CopySource, ProgramPoint};

use super::canonical::rebuild;
use super::{
    AnalysisDialect, Function, Instruction, InstructionId, Result, Variable, VariableId,
    VerifyDialect,
};

/// What one function's variables do, read once for the whole pass.
struct Occurrences {
    /// Every instruction that defines a variable.
    definers: BTreeMap<VariableId, Vec<ProgramPoint>>,
    /// Every SSA version of a variable that an instruction reads. A phi
    /// operand is not a read: a live phi is observed through the
    /// instruction that reads its result, whose version is the phi's and
    /// not a definition's, and a dead phi is observed by nothing.
    read_versions: BTreeMap<VariableId, BTreeSet<SsaVersion>>,
}

impl Occurrences {
    fn collect<D: super::Dialect>(function: &Function<D>, ssa: &SsaForm<VariableId>) -> Self {
        let mut definers: BTreeMap<VariableId, Vec<ProgramPoint>> = BTreeMap::new();
        let mut read_versions: BTreeMap<VariableId, BTreeSet<SsaVersion>> = BTreeMap::new();
        for block in function.cfg().block_ids() {
            for (inst_idx, instruction) in function
                .cfg()
                .block(block)
                .instructions()
                .iter()
                .enumerate()
            {
                let point = ProgramPoint { block, inst_idx };
                for &defined in instruction.defs() {
                    definers.entry(defined).or_default().push(point);
                }
                let Some(renamed) = ssa.instruction(point) else {
                    continue;
                };
                for value in &renamed.uses {
                    read_versions
                        .entry(value.variable)
                        .or_default()
                        .insert(value.version);
                }
            }
        }
        Self {
            definers,
            read_versions,
        }
    }

    /// The sole instruction defining `variable`, or `None` when several or
    /// none do.
    fn sole_definer(&self, variable: VariableId) -> Option<ProgramPoint> {
        match self.definers.get(&variable)?.as_slice() {
            [point] => Some(*point),
            _ => None,
        }
    }

    /// Whether every read of `variable` observes `version`.
    ///
    /// This is the interference test in SSA terms. A read of any other
    /// version means some other value of the variable is observable — the
    /// live-in, or a phi merging a loop-carried one — and the two
    /// variables then do not hold one value over the range the coalescing
    /// would merge.
    fn reads_only(&self, variable: VariableId, version: SsaVersion) -> bool {
        self.read_versions
            .get(&variable)
            .is_none_or(|versions| versions.iter().all(|&read| read == version))
    }
}

/// The SSA version one instruction defines for `variable`.
fn defined_version(
    ssa: &SsaForm<VariableId>,
    point: ProgramPoint,
    variable: VariableId,
) -> Option<SsaVersion> {
    ssa.instruction(point)?
        .defs
        .iter()
        .find(|value| value.variable == variable)
        .map(|value| value.version)
}

impl<D: AnalysisDialect + VerifyDialect> Function<D> {
    /// Returns the function with every lifter temporary coalesced into the
    /// variable it is copied to, and how many copies that removed.
    ///
    /// A copy `d = s` is coalesced when `is_temporary` accepts `s`, `s` is
    /// defined by exactly one instruction and read only from that
    /// definition, and `d` is defined by exactly one instruction — the
    /// copy — read only from it, and is not a signature parameter. The
    /// defining instruction is then rewritten to define `d`, every read of
    /// `s` becomes a read of `d`, and the copy goes.
    ///
    /// Chains resolve in one pass: the admitted copies form a forest —
    /// each destination is defined once, so it has at most one incoming
    /// link — and each temporary is renamed to the end of its chain, so
    /// `t1 = e; t2 = t1; r = t2` becomes `r = e` without a second round.
    ///
    /// The coalesced variable keeps `d`'s role and native provenance, and
    /// every occurrence keeps the value type its instruction carried. The
    /// temporary stays declared and stops occurring; pair this with
    /// [`Self::prune_variables`] to drop it. Instruction identities become
    /// dense again, a removed copy takes its provenance with it, and the
    /// rewritten definition keeps its own.
    ///
    /// # Errors
    ///
    /// Returns a verification report when the stored function is invalid,
    /// and an error when the rebuilt function fails verification.
    pub fn coalesce_copies(
        &self,
        is_temporary: impl Fn(&Variable<D>) -> bool,
    ) -> Result<(Self, usize)> {
        let ssa = self.ssa()?;
        let occurrences = Occurrences::collect(self, &ssa);
        let mut coalesced: BTreeMap<VariableId, VariableId> = BTreeMap::new();
        let mut removed: BTreeSet<InstructionId> = BTreeSet::new();
        for block in self.cfg().block_ids() {
            for (inst_idx, instruction) in self.cfg().block(block).instructions().iter().enumerate()
            {
                let point = ProgramPoint { block, inst_idx };
                let Some((destination, temporary)) =
                    self.admits(instruction, point, &ssa, &occurrences, &is_temporary)
                else {
                    continue;
                };
                // One coalescing per temporary: two copies of the same
                // temporary name two destinations, and it can only become
                // one of them. The second stays an ordinary copy.
                if coalesced.contains_key(&temporary) {
                    continue;
                }
                coalesced.insert(temporary, destination);
                removed.insert(instruction.id());
            }
        }
        if removed.is_empty() {
            return Ok((self.clone(), 0));
        }
        let renames = resolve(&coalesced);
        let function = rebuild(
            self,
            &self.cfg,
            |instruction| !removed.contains(&instruction.id()),
            |variable| renames.get(&variable).copied().unwrap_or(variable),
        )?;
        Ok((function, removed.len()))
    }

    /// The `(destination, temporary)` one instruction coalesces, when it
    /// is a copy that may be.
    fn admits(
        &self,
        instruction: &Instruction<D>,
        point: ProgramPoint,
        ssa: &SsaForm<VariableId>,
        occurrences: &Occurrences,
        is_temporary: &impl Fn(&Variable<D>) -> bool,
    ) -> Option<(VariableId, VariableId)> {
        // `as_copy` is the whole shape: one definition, one use, no
        // declared effect, no exceptional transfer.
        let (destination, temporary) = instruction.as_copy()?;
        if destination == temporary
            || !self.variable(temporary).is_some_and(is_temporary)
            || self.signature().parameters.contains(&destination)
        {
            return None;
        }
        let definition = occurrences.sole_definer(temporary)?;
        if definition == point {
            return None;
        }
        let defined = defined_version(ssa, definition, temporary)?;
        if !occurrences.reads_only(temporary, defined) {
            return None;
        }
        // The destination must hold nothing but what this copy gives it,
        // or the earlier definition would overwrite a value something
        // still reads.
        if occurrences.sole_definer(destination)? != point {
            return None;
        }
        let copied = defined_version(ssa, point, destination)?;
        occurrences
            .reads_only(destination, copied)
            .then_some((destination, temporary))
    }
}

/// Resolves each temporary to the end of its chain.
///
/// Every destination is defined by exactly one instruction, so no two
/// chains can end at the same variable and the links form a forest; the
/// walk still stops at a repeat rather than looping forever.
fn resolve(coalesced: &BTreeMap<VariableId, VariableId>) -> BTreeMap<VariableId, VariableId> {
    let mut renames: BTreeMap<VariableId, VariableId> = BTreeMap::new();
    let mut path: Vec<VariableId> = Vec::new();
    let mut on_path: BTreeSet<VariableId> = BTreeSet::new();
    for (&temporary, &destination) in coalesced {
        if renames.contains_key(&temporary) {
            continue;
        }
        path.clear();
        on_path.clear();
        path.push(temporary);
        on_path.insert(temporary);
        let mut current = destination;
        let end = loop {
            if let Some(&found) = renames.get(&current) {
                break found;
            }
            let Some(&next) = coalesced.get(&current) else {
                break current;
            };
            if !on_path.insert(current) {
                break current;
            }
            path.push(current);
            current = next;
        };
        for node in path.drain(..) {
            renames.insert(node, end);
        }
    }
    renames
}
