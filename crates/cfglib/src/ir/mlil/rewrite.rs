//! Identity-preserving rewrites of generic MLIL instructions.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::BlockId;
use crate::dataflow::liveness::Liveness;

use super::variable::{typed, unchanged};
use super::{
    Dialect, Error, Function, Instruction, InstructionId, Result, TypedVariable, VariableId,
    VerifyDialect,
};

/// The replacement of each replaced instruction, in identity order.
type Replacements<D> = Vec<(InstructionId, InstructionReplacement<D>)>;

/// A complete replacement for one existing instruction.
///
/// The containing function keeps the instruction's identity and graph
/// position. The replacement may refer only to variables already declared by
/// that function; rebuilding verifies the resulting operation, operands, and
/// types before returning it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstructionReplacement<D: Dialect> {
    operation: D::Operation,
    uses: Vec<TypedVariable<D>>,
    defs: Vec<TypedVariable<D>>,
    may_throw: bool,
}

impl<D: Dialect> InstructionReplacement<D> {
    /// Creates one typed replacement instruction.
    #[must_use]
    pub const fn new(
        operation: D::Operation,
        uses: Vec<TypedVariable<D>>,
        defs: Vec<TypedVariable<D>>,
        may_throw: bool,
    ) -> Self {
        Self {
            operation,
            uses,
            defs,
            may_throw,
        }
    }

    /// Returns the checked operation and its typed instruction operands.
    pub(crate) fn into_parts(
        self,
    ) -> (
        D::Operation,
        Vec<TypedVariable<D>>,
        Vec<TypedVariable<D>>,
        bool,
    ) {
        (self.operation, self.uses, self.defs, self.may_throw)
    }
}

/// The result of an identity-preserving instruction rewrite.
#[derive(Debug, Clone)]
pub struct InstructionRewrite<D: Dialect> {
    /// The rebuilt and verified function.
    pub function: Function<D>,
    /// Number of instructions for which the callback supplied a replacement.
    pub rewritten: usize,
    /// Identities of variables declared after the original variables, in
    /// requested order.
    pub added_variables: Vec<VariableId>,
}

impl<D: VerifyDialect> Function<D> {
    /// Returns the function with selected instructions replaced in place.
    ///
    /// Blocks, edges, exception regions, variables, signatures, instruction
    /// identities, graph positions, and provenance are preserved exactly.
    /// Replacement metadata is recomputed from its operation through the
    /// dialect contract.
    ///
    /// # Errors
    ///
    /// Returns an error when a replacement refers to an undeclared variable
    /// or the changed function fails structural or dialect verification.
    pub fn rewrite_instructions(
        &self,
        mut replacement: impl FnMut(&Instruction<D>) -> Option<InstructionReplacement<D>>,
    ) -> Result<InstructionRewrite<D>> {
        self.rewrite_instructions_with_variables(core::iter::empty(), |instruction, _| {
            replacement(instruction)
        })
    }

    /// Replaces selected instructions of this function and returns how many
    /// it replaced.
    ///
    /// `replacement` sees the function and then each instruction in identity
    /// order. Every identity, the graph, the signature, and the provenance
    /// stay. When `replacement` replaces nothing, the function does no other
    /// work. Replacement metadata is recomputed from its operation through
    /// the dialect contract.
    ///
    /// # Errors
    ///
    /// Returns an error when a replacement refers to an undeclared variable
    /// or the changed function fails structural or dialect verification. The
    /// function then keeps its old instructions.
    pub fn replace_instructions(
        &mut self,
        mut replacement: impl FnMut(&Self, &Instruction<D>) -> Option<InstructionReplacement<D>>,
    ) -> Result<usize> {
        let replacements = self.replacements(|instruction| replacement(self, instruction))?;
        self.apply_replacements(replacements)
    }

    /// Rebuilds selected instructions after adding declared variables.
    ///
    /// Existing variable and instruction identities, graph structure,
    /// signature, and provenance remain unchanged. The new variable IDs are
    /// passed to `replacement` in declaration order and returned in
    /// [`InstructionRewrite::added_variables`]. This permits a checked ABI
    /// refinement to add an argument or clobber location that the original
    /// machine lift never read, without referring to an undeclared operand.
    ///
    /// `replacement` sees the instructions in identity order. When it
    /// replaces nothing and no variable is added, the function comes back
    /// unchanged without a check.
    ///
    /// # Errors
    /// Returns an error when the added variables exceed the identity space,
    /// a replacement uses an undeclared variable, or the changed function
    /// fails structural or dialect verification.
    pub fn rewrite_instructions_with_variables(
        &self,
        additional: impl IntoIterator<Item = (D::VariableRole, Option<D::NativeVariable>)>,
        mut replacement: impl FnMut(&Instruction<D>, &[VariableId]) -> Option<InstructionReplacement<D>>,
    ) -> Result<InstructionRewrite<D>> {
        let additional: Vec<_> = additional.into_iter().collect();
        // The added variables follow the existing ones, so their identities
        // are known before any replacement names them.
        let added_variables = (self.variables().len()..self.variables().len() + additional.len())
            .map(|index| {
                u32::try_from(index).map(VariableId::from_raw).map_err(|_| {
                    Error::InvalidConstruction("variable identity exceeds u32::MAX".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let replacements =
            self.replacements(|instruction| replacement(instruction, &added_variables))?;
        let mut function = self.clone();
        for ((role, native), &id) in additional.into_iter().zip(&added_variables) {
            function
                .variables
                .push(super::Variable { id, role, native });
        }
        let rewritten = if replacements.is_empty() && added_variables.is_empty() {
            0
        } else if replacements.is_empty() {
            let report = function.verify();
            if !report.is_ok() {
                return Err(report.into());
            }
            0
        } else {
            function.apply_replacements(replacements)?
        };
        Ok(InstructionRewrite {
            function,
            rewritten,
            added_variables,
        })
    }

    /// Returns the replacement of each instruction that `replacement`
    /// replaces, in identity order.
    fn replacements(
        &self,
        mut replacement: impl FnMut(&Instruction<D>) -> Option<InstructionReplacement<D>>,
    ) -> Result<Replacements<D>> {
        let mut replacements = Vec::new();
        for index in 0..self.instruction_count() {
            let raw = u32::try_from(index).map_err(|_| {
                Error::InvalidConstruction("instruction identity exceeds u32::MAX".into())
            })?;
            let id = InstructionId::from_raw(raw);
            let instruction = self
                .instruction(id)
                .ok_or_else(|| Error::InvalidConstruction("missing indexed instruction".into()))?;
            if let Some(replaced) = replacement(instruction) {
                replacements.push((id, replaced));
            }
        }
        Ok(replacements)
    }

    /// Puts each replacement at the position of its instruction and
    /// verifies the result.
    ///
    /// When the result does not verify, every replaced instruction returns.
    fn apply_replacements(&mut self, replacements: Replacements<D>) -> Result<usize> {
        if replacements.is_empty() {
            return Ok(0);
        }
        let count = replacements.len();
        let mut replaced = Vec::with_capacity(count);
        let mut failure = None;
        for (id, replacement) in replacements {
            let Some(point) = self.instruction_point(id) else {
                failure = Some(Error::InvalidConstruction(
                    "missing instruction point".into(),
                ));
                break;
            };
            let Some(slot) = self
                .cfg
                .block_mut(point.block)
                .instructions_mut()
                .get_mut(point.inst_idx)
            else {
                failure = Some(Error::InvalidConstruction(
                    "missing indexed instruction".into(),
                ));
                break;
            };
            let (operation, uses, defs, may_throw) = replacement.into_parts();
            let fresh = Instruction::new(id, operation, uses, defs, may_throw);
            replaced.push((point, core::mem::replace(slot, fresh)));
        }
        let failure = failure.or_else(|| {
            let report = self.verify();
            (!report.is_ok()).then(|| report.into())
        });
        let Some(failure) = failure else {
            return Ok(count);
        };
        for (point, old) in replaced {
            if let Some(slot) = self
                .cfg
                .block_mut(point.block)
                .instructions_mut()
                .get_mut(point.inst_idx)
            {
                *slot = old;
            }
        }
        Err(failure)
    }

    /// Returns the function with the unread definitions of the selected
    /// instructions dropped, and how many definitions that dropped.
    ///
    /// An instruction with declared effects survives dead-code
    /// elimination whole, definitions and all, which is right for the
    /// instruction and wrong for the definitions: a call under a calling
    /// convention states that it writes every register the convention
    /// does not preserve, and most of those writes are never read. This
    /// drops exactly those definitions and leaves the operation, the
    /// uses, and the exceptional behavior alone.
    ///
    /// `of` selects the instructions to consider — the caller knows which
    /// of its operations state writes they do not compute. A variable any
    /// instruction reads anywhere stays, so a definition a later read
    /// observes through a merge is never dropped. What the function hands
    /// back to its caller is not in the graph at all:
    /// [`Self::drop_unread_definitions_with_exits`] is the door that takes
    /// it.
    ///
    /// Nothing is removed, so every block, edge, instruction, variable,
    /// and provenance identity survives; a function with nothing to drop
    /// comes back unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when the rebuilt function fails structural or
    /// dialect verification.
    pub fn drop_unread_definitions(
        &self,
        of: impl Fn(&Instruction<D>) -> bool,
    ) -> Result<(Self, usize)> {
        self.drop_unread_definitions_with_exits(of, |_| Vec::new())
    }

    /// [`Self::drop_unread_definitions`] told what leaves each block.
    ///
    /// A definition that is live at an exit under `live_out` counts as
    /// read and stays: something outside the function observes it, which
    /// is exactly what a call's result register is when the function
    /// returns it. A caller normally answers for the blocks with no
    /// successors and hands back an empty vector everywhere else, and the
    /// empty seed is [`Self::drop_unread_definitions`].
    ///
    /// # Errors
    ///
    /// Returns an error when the rebuilt function fails structural or
    /// dialect verification.
    pub fn drop_unread_definitions_with_exits(
        &self,
        of: impl Fn(&Instruction<D>) -> bool,
        live_out: impl Fn(BlockId) -> Vec<VariableId>,
    ) -> Result<(Self, usize)> {
        let (replacements, dropped) = self.unread_definitions(of, live_out)?;
        let mut function = self.clone();
        function.apply_replacements(replacements)?;
        Ok((function, dropped))
    }

    /// [`Self::drop_unread_definitions_with_exits`] on this function:
    /// drops the definitions and returns how many it dropped.
    ///
    /// A function with nothing to drop does no other work.
    ///
    /// # Errors
    ///
    /// Returns an error when the changed function fails structural or
    /// dialect verification. The function then stays as it was.
    pub fn drop_unread_definitions_with_exits_in_place(
        &mut self,
        of: impl Fn(&Instruction<D>) -> bool,
        live_out: impl Fn(BlockId) -> Vec<VariableId>,
    ) -> Result<usize> {
        let (replacements, dropped) = self.unread_definitions(of, live_out)?;
        self.apply_replacements(replacements)?;
        Ok(dropped)
    }

    /// Returns the replacement of each instruction that `of` selects and
    /// that defines something that no reader and no exit observes, and the
    /// number of definitions that the replacements drop.
    ///
    /// The liveness runs only when some selected definition has no reader.
    fn unread_definitions(
        &self,
        of: impl Fn(&Instruction<D>) -> bool,
        live_out: impl Fn(BlockId) -> Vec<VariableId>,
    ) -> Result<(Replacements<D>, usize)> {
        let read = read_variables(self);
        let candidate = |instruction: &Instruction<D>| {
            of(instruction)
                && instruction
                    .defs()
                    .iter()
                    .any(|defined| !read.contains(defined))
        };
        if !self.instructions().any(candidate) {
            return Ok((Vec::new(), 0));
        }
        let observed = live_definitions(self, live_out);
        let mut dropped = 0usize;
        let replacements = self.replacements(|instruction| {
            if !candidate(instruction) {
                return None;
            }
            let kept: Vec<TypedVariable<D>> = instruction
                .defs()
                .iter()
                .zip(instruction.def_types())
                .filter(|(defined, _)| {
                    read.contains(defined) || observed.contains(&(instruction.id(), **defined))
                })
                .map(|(&defined, value_type)| TypedVariable::new(defined, value_type.clone()))
                .collect();
            if kept.len() == instruction.defs().len() {
                return None;
            }
            dropped += instruction.defs().len() - kept.len();
            Some(InstructionReplacement::new(
                instruction.operation().clone(),
                typed::<D>(instruction.uses(), instruction.use_types(), unchanged),
                kept,
                instruction.may_throw(),
            ))
        })?;
        Ok((replacements, dropped))
    }
}

/// Every definition whose value is still live where its instruction
/// leaves it, the caller's exit seed included.
///
/// With an empty seed this is a subset of what
/// [`read_variables`] answers — a definition is live only because
/// something reads the variable — so the unseeded door keeps exactly what
/// it always kept.
fn live_definitions<D: Dialect>(
    function: &Function<D>,
    live_out: impl Fn(BlockId) -> Vec<VariableId>,
) -> BTreeSet<(InstructionId, VariableId)> {
    let liveness: Liveness<VariableId> = Liveness::compute_with_exits(function.cfg(), live_out);
    let mut observed = BTreeSet::new();
    for block in function.cfg().block_ids() {
        let after = liveness.live_after_instructions(function.cfg(), block);
        for (index, instruction) in function
            .cfg()
            .block(block)
            .instructions()
            .iter()
            .enumerate()
        {
            for &defined in instruction.defs() {
                if after[index].contains(&defined) {
                    observed.insert((instruction.id(), defined));
                }
            }
        }
    }
    observed
}

/// Every variable some instruction of the function reads.
fn read_variables<D: Dialect>(function: &Function<D>) -> BTreeSet<VariableId> {
    let mut read = BTreeSet::new();
    for block in function.cfg().block_ids() {
        for instruction in function.cfg().block(block).instructions() {
            read.extend(instruction.uses().iter().copied());
        }
    }
    read
}
