//! Checked insertion around existing generic MLIL instructions.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::variable::{typed, unchanged};
use super::{
    Dialect, EntityId, Error, Function, FunctionBuilder, Instruction, InstructionId,
    InstructionReplacement, Result, Signature, VariableId, VerifyDialect,
};

/// Ordered instructions inserted around one original instruction.
///
/// An absent replacement retains the original operation and typed operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstructionEdit<D: Dialect> {
    before: Vec<InstructionReplacement<D>>,
    replacement: Option<InstructionReplacement<D>>,
    after: Vec<InstructionReplacement<D>>,
}

impl<D: Dialect> InstructionEdit<D> {
    /// Creates an ordered checked edit.
    #[must_use]
    pub const fn new(
        before: Vec<InstructionReplacement<D>>,
        replacement: Option<InstructionReplacement<D>>,
        after: Vec<InstructionReplacement<D>>,
    ) -> Self {
        Self {
            before,
            replacement,
            after,
        }
    }
}

/// A rebuilt function and the identities needed by its consumers.
#[derive(Debug, Clone)]
pub struct InstructionSplice<D: Dialect> {
    /// The rebuilt and verified function.
    pub function: Function<D>,
    /// Variables declared after the original variables, in request order.
    pub added_variables: Vec<VariableId>,
    /// The new identity of each original instruction, indexed by old ID.
    pub original_instructions: Vec<InstructionId>,
    /// Inserted instructions before each original instruction, indexed by old ID.
    pub before: Vec<Vec<InstructionId>>,
    /// Inserted instructions after each original instruction, indexed by old ID.
    pub after: Vec<Vec<InstructionId>>,
}

impl<D: VerifyDialect> Function<D> {
    /// Inserts checked instructions before and after selected instructions.
    ///
    /// The callback sees all newly declared variable IDs. Existing blocks,
    /// edges, variables, and signatures keep their identities. Instruction
    /// IDs may move; the returned maps identify every original and inserted
    /// instruction. All three parts of an edit inherit the original
    /// instruction's source correspondences. Unedited provenance is retained.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid variable, invalid operation, identity
    /// exhaustion, or any failed structural or dialect verification.
    pub fn splice_instructions_with_variables(
        &self,
        additional: impl IntoIterator<Item = (D::VariableRole, Option<D::NativeVariable>)>,
        edit: impl FnMut(&Instruction<D>, &[VariableId]) -> Option<InstructionEdit<D>>,
    ) -> Result<InstructionSplice<D>> {
        self.splice_instructions_with_signature(additional, |signature, _| signature.clone(), edit)
    }

    /// Inserts checked instructions while refining the function signature.
    ///
    /// The signature callback sees the original signature and all newly
    /// declared variables. The builder verifies every parameter identity;
    /// the dialect verifies the completed return statements and types.
    ///
    /// # Errors
    /// Returns an error for any invalid signature, instruction, or graph.
    pub fn splice_instructions_with_signature(
        &self,
        additional: impl IntoIterator<Item = (D::VariableRole, Option<D::NativeVariable>)>,
        signature: impl FnOnce(&Signature<D>, &[VariableId]) -> Signature<D>,
        mut edit: impl FnMut(&Instruction<D>, &[VariableId]) -> Option<InstructionEdit<D>>,
    ) -> Result<InstructionSplice<D>> {
        let mut builder = FunctionBuilder::<D>::new(self.source().clone());
        for variable in self.variables() {
            let rebuilt =
                builder.declare_variable(variable.role.clone(), variable.native.clone())?;
            debug_assert_eq!(rebuilt, variable.id);
        }
        let added_variables = additional
            .into_iter()
            .map(|(role, native)| builder.declare_variable(role, native))
            .collect::<Result<Vec<_>>>()?;
        builder.copy_blocks(self.cfg());

        let mut originals = vec![None; self.instruction_count()];
        let mut before = vec![Vec::new(); self.instruction_count()];
        let mut after = vec![Vec::new(); self.instruction_count()];
        for block in self.cfg().block_ids() {
            for instruction in self.cfg().block(block).instructions() {
                let index = instruction.id().index();
                if index >= originals.len() {
                    return Err(Error::InvalidConstruction(
                        "instruction identity exceeds indexed table".into(),
                    ));
                }
                if let Some(selected) = edit(instruction, &added_variables) {
                    for operation in selected.before {
                        before[index].push(append(&mut builder, block, operation)?);
                    }
                    originals[index] = Some(append(
                        &mut builder,
                        block,
                        selected
                            .replacement
                            .unwrap_or_else(|| unchanged_instruction(instruction)),
                    )?);
                    for operation in selected.after {
                        after[index].push(append(&mut builder, block, operation)?);
                    }
                } else {
                    originals[index] = Some(append(
                        &mut builder,
                        block,
                        unchanged_instruction(instruction),
                    )?);
                }
            }
        }
        let originals = originals
            .into_iter()
            .map(|id| {
                id.ok_or_else(|| Error::InvalidConstruction("missing indexed instruction".into()))
            })
            .collect::<Result<Vec<_>>>()?;

        builder.copy_structure(self.cfg())?;
        builder.set_signature(signature(self.signature(), &added_variables))?;
        for entry in self.provenance().entries() {
            if let EntityId::Instruction(old) = entry.entity {
                let index = old.index();
                for &inserted in before[index]
                    .iter()
                    .chain(core::iter::once(&originals[index]))
                    .chain(after[index].iter())
                {
                    builder.map_entity(entry.source.clone(), EntityId::Instruction(inserted))?;
                }
            } else {
                builder.map_entity(entry.source.clone(), entry.entity)?;
            }
        }

        Ok(InstructionSplice {
            function: builder.finish()?,
            added_variables,
            original_instructions: originals,
            before,
            after,
        })
    }
}

fn unchanged_instruction<D: Dialect>(instruction: &Instruction<D>) -> InstructionReplacement<D> {
    InstructionReplacement::new(
        instruction.operation().clone(),
        typed::<D>(instruction.uses(), instruction.use_types(), unchanged),
        typed::<D>(instruction.defs(), instruction.def_types(), unchanged),
        instruction.may_throw(),
    )
}

fn append<D: Dialect>(
    builder: &mut FunctionBuilder<D>,
    block: crate::BlockId,
    instruction: InstructionReplacement<D>,
) -> Result<InstructionId> {
    let (operation, uses, defs, may_throw) = instruction.into_parts();
    builder.append_instruction(block, operation, uses, defs, may_throw, None)
}
