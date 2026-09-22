//! Identity-preserving rewrites of generic MLIL instructions.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::variable::{typed, unchanged};
use super::{
    Dialect, Error, Function, FunctionBuilder, Instruction, InstructionId, Result, TypedVariable,
    VariableId, VerifyDialect,
};

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
}

/// The result of an identity-preserving instruction rewrite.
#[derive(Debug, Clone)]
pub struct InstructionRewrite<D: Dialect> {
    /// The rebuilt and verified function.
    pub function: Function<D>,
    /// Number of instructions for which the callback supplied a replacement.
    pub rewritten: usize,
}

impl<D: VerifyDialect> Function<D> {
    /// Rebuilds the function with selected instructions replaced in place.
    ///
    /// Blocks, edges, exception regions, variables, signatures, instruction
    /// identities, graph positions, and provenance are preserved exactly.
    /// Replacement metadata is recomputed from its operation through the
    /// dialect contract.
    ///
    /// # Errors
    ///
    /// Returns an error when a replacement refers to an undeclared variable
    /// or the rebuilt function fails structural or dialect verification.
    pub fn rewrite_instructions(
        &self,
        mut replacement: impl FnMut(&Instruction<D>) -> Option<InstructionReplacement<D>>,
    ) -> Result<InstructionRewrite<D>> {
        let mut builder = FunctionBuilder::<D>::new(self.source().clone());
        for variable in self.variables() {
            let rebuilt =
                builder.declare_variable(variable.role.clone(), variable.native.clone())?;
            debug_assert_eq!(rebuilt, variable.id);
        }
        builder.copy_blocks(self.cfg());

        let mut rewritten = 0usize;
        for index in 0..self.instruction_count() {
            let raw = u32::try_from(index).map_err(|_| {
                Error::InvalidConstruction("instruction identity exceeds u32::MAX".into())
            })?;
            let id = InstructionId::from_raw(raw);
            let point = self
                .instruction_point(id)
                .ok_or_else(|| Error::InvalidConstruction("missing instruction point".into()))?;
            let instruction = self
                .instruction(id)
                .ok_or_else(|| Error::InvalidConstruction("missing indexed instruction".into()))?;
            let (operation, uses, defs, may_throw) =
                if let Some(replacement) = replacement(instruction) {
                    rewritten += 1;
                    (
                        replacement.operation,
                        replacement.uses,
                        replacement.defs,
                        replacement.may_throw,
                    )
                } else {
                    (
                        instruction.operation().clone(),
                        typed::<D>(instruction.uses(), instruction.use_types(), unchanged),
                        typed::<D>(instruction.defs(), instruction.def_types(), unchanged),
                        instruction.may_throw(),
                    )
                };
            let rebuilt =
                builder.append_instruction(point.block, operation, uses, defs, may_throw, None)?;
            debug_assert_eq!(rebuilt, id);
        }

        builder.copy_structure(self.cfg())?;
        builder.copy_metadata(self.signature().clone(), self.provenance())?;

        Ok(InstructionRewrite {
            function: builder.finish()?,
            rewritten,
        })
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
    /// observes through a merge is never dropped.
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
        let read = read_variables(self);
        let mut dropped = 0usize;
        let rewrite = self.rewrite_instructions(|instruction| {
            if !of(instruction) {
                return None;
            }
            let kept: Vec<TypedVariable<D>> = instruction
                .defs()
                .iter()
                .zip(instruction.def_types())
                .filter(|(defined, _)| read.contains(defined))
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
        Ok((rewrite.function, dropped))
    }
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
