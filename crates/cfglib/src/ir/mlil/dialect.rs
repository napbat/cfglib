//! Consumer contracts that specialize generic MLIL storage.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt::Debug;

use crate::ir::dialect::Vocabulary;
use crate::memory::MemoryEvent;
use crate::{EdgeKind, FlowEffect};

use super::{Function, Instruction, VariableId, VerificationIssue};

/// Cached control-flow and side-effect facts for one semantic operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstructionMetadata<E> {
    /// Observable effects beyond variable definitions.
    pub effects: Vec<E>,
    /// Intraprocedural control-flow behavior.
    pub flow: FlowEffect,
    /// Whether the instruction can transfer through an exceptional edge.
    pub may_throw: bool,
}

impl<E> InstructionMetadata<E> {
    /// Creates explicit metadata for one instruction.
    #[must_use]
    pub const fn new(effects: Vec<E>, flow: FlowEffect, may_throw: bool) -> Self {
        Self {
            effects,
            flow,
            may_throw,
        }
    }
}

/// Complete storage contract for one MLIL semantic dialect.
///
/// A dialect is a semantic vocabulary, not a source ISA. Multiple frontends
/// can therefore converge on the same dialect while retaining different
/// source identities and native-variable provenance. The level-independent
/// types — value types, effects, source coordinates, variable roles — come
/// from the [`Vocabulary`] supertrait shared with
/// [`ir::hlil::Dialect`](crate::ir::hlil::Dialect); this trait adds the
/// flat-control-flow contract MLIL storage needs.
pub trait Dialect: Vocabulary {
    /// Semantic operation stored by an instruction.
    type Operation: Clone + Debug + Eq;
    /// Exact caller-owned metadata stored on control-flow edges.
    type Edge: Clone + Debug + Eq;

    /// Classifies and caches the effects of one operation.
    fn instruction_metadata(
        operation: &Self::Operation,
        may_throw: bool,
    ) -> InstructionMetadata<Self::Effect>;

    /// Returns a compact stable mnemonic for an operation.
    fn mnemonic(operation: &Self::Operation) -> &str;

    /// Maps caller edge metadata to cfglib's structural edge kind.
    fn edge_kind(edge: &Self::Edge) -> EdgeKind;

    /// Returns whether an edge is the synthetic root's unique entry edge.
    fn is_entry_edge(edge: &Self::Edge) -> bool;
}

/// Optional memory-event contract for an MLIL dialect.
///
/// Implementing this trait automatically exposes every
/// [`Instruction<Self>`] through
/// [`MemoryEventInfo`](crate::MemoryEventInfo), so generic memory analyses do
/// not need a dialect-specific instruction wrapper.
pub trait MemoryDialect: Dialect {
    /// Consumer-defined location or conservative alias region.
    type MemoryLocation: Clone + Ord;
    /// Consumer-defined fence ordering and visibility details.
    type MemoryFence: Clone + Eq;

    /// Returns one instruction's memory events in semantic order.
    fn memory_events(
        instruction: &Instruction<Self>,
    ) -> impl Iterator<Item = MemoryEvent<Self::MemoryLocation, VariableId, Self::MemoryFence>>;
}

/// Optional semantic hooks used by reusable MLIL analyses.
pub trait AnalysisDialect: Dialect {
    /// Constant domain used by propagation and recovered expressions.
    type Constant: Clone + Debug + Eq;
    /// Pure operator identity retained in recovered expression trees.
    type ExpressionOperator: Clone + Debug + Eq;
    /// Statically known call-target identity.
    type Callee: Clone + Debug + Ord;

    /// Returns whether an operation is a pure one-use, one-definition copy.
    fn is_copy(operation: &Self::Operation) -> bool;

    /// Returns whether an operation pairwise aliases definitions to uses.
    ///
    /// Each definition must retain the corresponding use's exact runtime
    /// value, though its value type or other analysis metadata may change.
    /// Reads are simultaneous and precede writes. The default admits ordinary
    /// copies; dialects may additionally admit type refinements or parallel
    /// alias commits for explicit alias propagation.
    fn is_value_alias(operation: &Self::Operation) -> bool {
        Self::is_copy(operation)
    }

    /// Returns the pure expression operator represented by an operation.
    fn expression_operator(operation: &Self::Operation) -> Option<Self::ExpressionOperator>;

    /// Returns a constant directly materialized by an operation.
    fn constant(operation: &Self::Operation) -> Option<Self::Constant>;

    /// Folds an instruction using constants known at its input.
    fn fold_constant(
        instruction: &Instruction<Self>,
        known: &BTreeMap<VariableId, Self::Constant>,
    ) -> Option<(VariableId, Self::Constant)>;

    /// Decides the conditional transfer an instruction ends its block with.
    ///
    /// Returns `Some(true)` when `known` proves that the
    /// [`ConditionalTrue`](EdgeKind::ConditionalTrue) edge is taken, and
    /// `Some(false)` when it proves that the
    /// [`ConditionalFalse`](EdgeKind::ConditionalFalse) edge is taken.
    /// Returns `None` when the instruction is not a conditional terminator,
    /// or when the condition is not decided.
    ///
    /// `known` holds the same entries [`fold_constant`](Self::fold_constant)
    /// receives. The default answers `None`, which leaves every outgoing
    /// edge of the block executable.
    fn fold_branch(
        instruction: &Instruction<Self>,
        known: &BTreeMap<VariableId, Self::Constant>,
    ) -> Option<bool> {
        let _ = (instruction, known);
        None
    }

    /// Meets two constants of this dialect's domain.
    ///
    /// `Some(c)` is the greatest constant below both `a` and `b`. `None` is
    /// the lattice bottom: no constant of the domain is below both.
    ///
    /// A domain of exact values answers `Some` only for equal arguments,
    /// which is the default. A domain that carries partial knowledge, such
    /// as known bits, answers with the part on which `a` and `b` agree.
    ///
    /// The result must be below both arguments, so meeting it again with
    /// either argument must return the result itself.
    fn meet_constants(a: &Self::Constant, b: &Self::Constant) -> Option<Self::Constant> {
        (a == b).then(|| a.clone())
    }

    /// Returns the statically known target of a call operation.
    fn callee(operation: &Self::Operation) -> Option<Self::Callee>;
}

/// Dialect-specific semantic and typing verification.
pub trait VerifyDialect: Dialect {
    /// Appends every dialect invariant violation in deterministic order.
    fn verify(function: &Function<Self>, issues: &mut Vec<VerificationIssue>);
}
