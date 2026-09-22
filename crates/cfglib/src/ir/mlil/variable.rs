//! Generic mutable variables and point-specific typed occurrences.

extern crate alloc;

use alloc::vec::Vec;

use super::{Dialect, VariableId};

/// One declared MLIL variable before SSA renaming.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable<D: Dialect> {
    /// Stable dense identity.
    pub id: VariableId,
    /// Semantic role used by analyses and presentation.
    pub role: D::VariableRole,
    /// Optional source-native storage provenance.
    pub native: Option<D::NativeVariable>,
}

/// One variable occurrence paired with its type at that program point.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypedVariable<D: Dialect> {
    /// Mutable variable identity.
    pub variable: VariableId,
    /// Value type required or produced at this occurrence.
    pub value_type: D::ValueType,
}

impl<D: Dialect> TypedVariable<D> {
    /// Creates a typed variable occurrence.
    #[must_use]
    pub const fn new(variable: VariableId, value_type: D::ValueType) -> Self {
        Self {
            variable,
            value_type,
        }
    }
}

/// Pairs each variable of one occurrence list with its type, under a
/// renaming of the identities.
///
/// The two lists come from one instruction and a verified instruction
/// keeps them the same length, so a shorter type list truncates rather
/// than inventing a type. An occurrence keeps the value type the
/// instruction carried for it whatever `rename` does with its identity.
pub(super) fn typed<D: Dialect>(
    variables: &[VariableId],
    value_types: &[D::ValueType],
    rename: impl Fn(VariableId) -> VariableId,
) -> Vec<TypedVariable<D>> {
    variables
        .iter()
        .zip(value_types)
        .map(|(&variable, value_type)| TypedVariable::new(rename(variable), value_type.clone()))
        .collect()
}

/// The renaming that changes nothing.
pub(super) const fn unchanged(variable: VariableId) -> VariableId {
    variable
}
