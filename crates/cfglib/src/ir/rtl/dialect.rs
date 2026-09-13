//! Consumer contracts that specialize generic RTL storage.

use core::fmt::Debug;

use crate::EdgeKind;
use crate::ir::dialect::Vocabulary;

use super::emission::{EdgeContext, Emission};
use super::error::Result;
use super::template::LiftedStatement;
use super::types::{Constraint, Shape};

/// The canonical edge vocabulary for dialects with plain two-way branching
/// and exception unwind flow.
///
/// A dialect whose control flow is fallthrough plus conditional
/// true/false plus unwind edges — no fused dispatch or typed handlers — can use this
/// as its [`Dialect::Edge`] (and its
/// [`mlil::Dialect::Edge`](crate::ir::mlil::Dialect::Edge)) and delegate
/// the trait's edge hooks to [`kind`](Self::kind) and
/// [`is_entry`](Self::is_entry). Dialects with switches, typed handler
/// metadata, or legacy continuations define their own edge type instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Edge {
    /// The synthetic root's unique entry edge.
    Entry,
    /// Sequential flow.
    Fall,
    /// Taken branch of a conditional.
    True,
    /// Not-taken branch of a conditional.
    False,
    /// Exceptional transfer from a faulting operation to a handler.
    Unwind,
}

impl Edge {
    /// The structural classification of the edge.
    #[must_use]
    pub const fn kind(self) -> EdgeKind {
        match self {
            Self::Entry | Self::Fall => EdgeKind::Fallthrough,
            Self::True => EdgeKind::ConditionalTrue,
            Self::False => EdgeKind::ConditionalFalse,
            Self::Unwind => EdgeKind::ExceptionUnwind,
        }
    }

    /// Whether the edge is the synthetic root's entry edge.
    #[must_use]
    pub const fn is_entry(self) -> bool {
        matches!(self, Self::Entry)
    }
}

/// Storage contract for one RTL semantic dialect.
///
/// The level-independent types — value types, effects, source
/// coordinates, variable roles, native storage — come from the
/// [`Vocabulary`] supertrait shared with the MLIL and HLIL dialects.
/// Storage locations are the vocabulary's `NativeVariable`: RTL operates
/// on raw language storage — shader registers, JVM locals and stack
/// slots, machine registers and flags, wasm value slots — before any
/// variable recovery.
pub trait Dialect: Vocabulary {
    /// The lane constraint domain web typing folds over.
    ///
    /// Numeric machine dialects use the provided
    /// [`ScalarType`](super::ScalarType); managed dialects supply their
    /// own domain covering references, null, uninitialized objects, and
    /// hierarchy-dependent merges.
    type Constraint: Constraint;
    /// Pure typed operator applied by expressions.
    type Operator: Clone + Debug + Eq;
    /// Effect-bearing operation retained as a statement.
    type EffectOp: Clone + Debug + Eq;
    /// Exact caller-owned metadata stored on control-flow edges — branch
    /// polarity, switch case values, handler catch types and order,
    /// legacy continuations.
    type Edge: Clone + Debug + Eq;

    /// Returns a compact stable mnemonic for an operator.
    fn mnemonic(operator: &Self::Operator) -> &str;

    /// Returns a compact stable mnemonic for an effect operation.
    fn effect_mnemonic(operation: &Self::EffectOp) -> &str;

    /// Maps caller edge metadata to cfglib's structural edge kind.
    fn edge_kind(edge: &Self::Edge) -> EdgeKind;

    /// Returns whether an edge is the synthetic root's unique entry edge.
    fn is_entry_edge(edge: &Self::Edge) -> bool;
}

/// Associates one storage-level dialect with its semantic MLIL dialect.
///
/// The markers remain distinct so several source or target machines can
/// converge on one semantic dialect. Value types, effects, source
/// coordinates, and variable roles are shared by construction. Native
/// storage is deliberately not: a stack machine, register machine, and
/// semantic variable model can each use a different location type.
pub trait MlilBridge: Dialect {
    /// The semantic MLIL dialect this RTL raises into and lowers from.
    type Mlil: crate::ir::mlil::Dialect<
            ValueType = Self::ValueType,
            Effect = Self::Effect,
            Source = Self::Source,
            SourceSpan = Self::SourceSpan,
            SourcePoint = Self::SourcePoint,
            VariableRole = Self::VariableRole,
        >;
}

/// The lifting contract from RTL into a dialect's MLIL.
///
/// A consumer implements this on its RTL marker and selects the semantic
/// destination through [`MlilBridge::Mlil`]. Multiple RTL dialects can
/// therefore converge on one MLIL dialect while retaining independent
/// edge identity domains.
pub trait Lift: MlilBridge {
    /// The MLIL value type of one lifted web shape.
    fn value_type(shape: Shape<Self::Constraint>) -> <Self as Vocabulary>::ValueType;

    /// The variable role of one lifted web.
    ///
    /// `storage` is the native location the web lives in, or `None` for
    /// a synthetic serialization temporary.
    fn web_role(
        storage: Option<&<Self as Vocabulary>::NativeVariable>,
    ) -> <Self as Vocabulary>::VariableRole;

    /// Returns the role of one ordered parameter web.
    ///
    /// The default preserves dialects that do not distinguish parameter
    /// roles from other native storage. Dialects with ordinal parameter
    /// roles override it.
    fn parameter_role(
        _ordinal: u16,
        storage: &<Self as Vocabulary>::NativeVariable,
    ) -> <Self as Vocabulary>::VariableRole {
        Self::web_role(Some(storage))
    }

    /// Chooses the native provenance retained on one semantic variable.
    ///
    /// This is an explicit translation because native RTL storage and
    /// semantic provenance need not share a type. Target-only allocation
    /// and synthetic temporaries normally return `None`; source-native
    /// locations return the semantic dialect's corresponding identity.
    fn native_variable(
        storage: &<Self as Vocabulary>::NativeVariable,
        _source: &<Self as Vocabulary>::Source,
    ) -> Option<<<Self as MlilBridge>::Mlil as Vocabulary>::NativeVariable>;

    /// Translates one lifted statement into MLIL instructions.
    ///
    /// The simple, storage-flavored form is one line —
    /// [`Emission::single`] appends one instruction whose uses,
    /// definitions, throw site, and span all derive from the statement.
    /// Semantic dialects instead call [`Emission::append`] one or more
    /// times, staging through [`Emission::temporary`] where an expansion
    /// needs intermediate values, and the context validates read
    /// alignment and exceptional placement.
    ///
    /// # Errors
    ///
    /// Returns an error when the statement has no legal translation.
    fn emit(context: &mut Emission<'_, '_, Self>, statement: LiftedStatement<Self>) -> Result<()>;

    /// Translates one RTL edge into the lifted function's MLIL edge
    /// metadata.
    ///
    /// Runs after every instruction is emitted, so the context resolves
    /// statements to emitted MLIL instruction identities — an
    /// exceptional edge's payload can carry its exact throw site in the
    /// MLIL identity domain. A dialect sharing one edge type across
    /// both levels clones the metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when edge metadata names an entity that was not
    /// emitted or otherwise has no exact MLIL representation.
    fn lift_edge(
        edge: &<Self as Dialect>::Edge,
        context: &EdgeContext<'_>,
    ) -> Result<<<Self as MlilBridge>::Mlil as crate::ir::mlil::Dialect>::Edge>;
}
