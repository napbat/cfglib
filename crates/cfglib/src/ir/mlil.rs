//! Generic medium-level intermediate-language storage and analysis contracts.
//!
//! MLIL functions use [`Cfg`](crate::Cfg) for control-flow storage while a
//! caller-defined [`Dialect`] supplies the semantic operation, type, effect,
//! edge, source, and native-variable vocabularies. This keeps the shared
//! representation useful across managed runtimes, native instruction sets,
//! shaders, and source-language compilers without flattening their semantics
//! into strings or a closed library-owned opcode enum.
//!
//! A transform reaches a function one of two ways, and the difference is the
//! module's central contract.
//!
//! A **derived view** — [`Function::copy_propagated_cfg`],
//! [`Function::dead_code_eliminated_cfg`],
//! [`Function::with_promoted_handler_extents`],
//! [`Function::with_duplicated_structuring_tails`],
//! [`Function::with_derived_cfg`] — answers with a graph and leaves the
//! function it came from alone. It is for presentation and analysis: the
//! canonical function keeps describing the original program, identities and
//! provenance included. A derived graph that dropped instructions is not a
//! function anymore — the provenance still names what went — so every door that
//! verifies first refuses it.
//!
//! A **canonical rebuild** — [`Function::eliminate_dead_code`],
//! [`Function::propagate_copies`], [`Function::coalesce_copies`],
//! [`Function::prune_variables`],
//! [`Function::split_variables`], [`Function::promote_memory`],
//! [`Function::rewrite_instructions`] — takes the same decision and rebuilds a
//! function that verifies, which is what a lift stores. Each states exactly
//! which identities it keeps; what a rebuild drops takes its provenance with
//! it.

mod builder;
mod canonical;
mod coalesce;
mod constant;
mod coverage;
mod dialect;
mod error;
mod function;
mod identity;
mod instruction;
mod promote;
mod provenance;
mod prune;
mod rewrite;
mod split;
mod variable;
mod verify;

pub use builder::FunctionBuilder;
pub use constant::{ConstantMaterialization, ConstantMaterializationDialect};
pub use coverage::extend_equivalent_coverage;
pub use dialect::{AnalysisDialect, Dialect, InstructionMetadata, MemoryDialect, VerifyDialect};
pub use error::{Error, Result, VerificationIssue, VerificationReport};
pub use function::Function;
pub use identity::{EntityId, InstructionId, VariableId};
pub use instruction::Instruction;
pub use promote::{MemoryPromotion, PromoteDialect, PromotionAccess};
pub use provenance::{ProvenanceEntry, ProvenanceMap};
pub use prune::VariablePruning;
pub use rewrite::{InstructionReplacement, InstructionRewrite};
pub use split::VariableSplit;
pub use variable::{TypedVariable, Variable};

/// Ordered parameter and return signature of one MLIL function.
pub type Signature<D> =
    crate::ir::signature::Signature<VariableId, <D as crate::ir::dialect::Vocabulary>::ValueType>;

#[cfg(test)]
mod tests;
