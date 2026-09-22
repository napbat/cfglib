//! Generic medium-level intermediate-language storage and analysis contracts.
//!
//! MLIL functions use [`Cfg`](crate::Cfg) for control-flow storage while a
//! caller-defined [`Dialect`] supplies the semantic operation, type, effect,
//! edge, source, and native-variable vocabularies. This keeps the shared
//! representation useful across managed runtimes, native instruction sets,
//! shaders, and source-language compilers without flattening their semantics
//! into strings or a closed library-owned opcode enum.

mod builder;
mod canonical;
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
