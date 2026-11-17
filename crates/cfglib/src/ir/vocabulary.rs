//! The vocabulary that every IR level of one consumer shares.

use core::fmt::Debug;
use core::hash::Hash;

/// The semantic vocabulary that every IR level of one consumer shares.
///
/// A consumer describes one semantic world. The types here (value types,
/// observable effects, source coordinates, and the metadata of a variable)
/// mean the same thing at every level. A consumer that implements the dialects
/// of several levels on one marker states these types once. Then the passes
/// between levels share value types, effects, variables, and source
/// coordinates without a conversion.
pub trait Vocabulary: Clone + Debug + Eq + 'static {
    /// The value type of one typed occurrence. A variable holds differently
    /// typed values at different points, so each occurrence has its own type.
    type ValueType: Clone + Debug + Eq + Hash + Ord;
    /// One observable effect, as purity and dead-code analyses read it.
    type Effect: Clone + Debug + Eq + Ord;
    /// The identity and coordinate system of the source function.
    type Source: Clone + Debug + Eq;
    /// One ordered source span. Provenance relates many spans to many
    /// entities.
    type SourceSpan: Clone + Debug + Eq + Ord;
    /// One source point that a query compares with provenance spans.
    type SourcePoint: Clone + Debug + Eq;
    /// The semantic role of a mutable variable.
    type VariableRole: Clone + Debug + Eq + Hash;
    /// The native storage identity of a variable in the source, when the
    /// consumer keeps one.
    type NativeVariable: Clone + Debug + Eq + Hash + Ord;

    /// Returns whether a source span is empty or reversed.
    fn span_is_empty(span: &Self::SourceSpan) -> bool;

    /// Returns whether a source span contains a point.
    fn span_contains(span: &Self::SourceSpan, point: &Self::SourcePoint) -> bool;
}

/// The metadata of one declared mutable variable.
///
/// The function that declares the variable gives it its identity, usually its
/// position in the variable table. This record holds only the metadata, so it
/// can also be the variable vocabulary of a structured dialect
/// ([`hlil::Dialect::Variable`](crate::ir::hlil::Dialect::Variable)). A
/// lifting that copies it into structured code and a lowering that declares it
/// back keep the same record.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable<D: Vocabulary> {
    /// The semantic role that analyses and presentation read.
    pub role: D::VariableRole,
    /// The storage that the source associates with the value, when the
    /// consumer keeps it.
    ///
    /// This is a hint. A lowering can choose another physical home, and it
    /// can also choose machine storage for a variable without this hint.
    pub native: Option<D::NativeVariable>,
}
