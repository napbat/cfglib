//! Reading a scope graph from its serialized form without trusting it.
//!
//! Resolution walks the edges of the graph underneath and then reads the data
//! of each scope it reaches through a table indexed by that scope, so a
//! document whose tables disagree with its scopes would panic in the middle of
//! a query. A document is therefore decoded into [`Stored`], the same columns
//! with no behavior, and becomes a [`ScopeGraph`] only once its tables have
//! been checked against the scopes, data, and references they index.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{
    ScopeDatum, ScopeDatumId, ScopeGraph, ScopeId, ScopeReference, ScopeReferenceId, ScopeStore,
};

/// The columns of a [`ScopeGraph`], exactly as they are written.
///
/// A field here without its counterpart in `ScopeGraph` (or the reverse) does
/// not compile, because [`TryFrom`] below names every one.
#[derive(serde::Deserialize)]
#[serde(bound(
    deserialize = "S: serde::Deserialize<'de>, L: serde::Deserialize<'de>, \
                   R: serde::Deserialize<'de>, D: serde::Deserialize<'de>, \
                   Q: serde::Deserialize<'de>"
))]
pub(super) struct Stored<S, L, R, D, Q> {
    graph: ScopeStore<S, L>,
    data: Vec<ScopeDatum<R, D>>,
    references: Vec<ScopeReference<Q>>,
    scope_data: Vec<Vec<ScopeDatumId>>,
    scope_references: Vec<Vec<ScopeReferenceId>>,
}

impl<S, L, R, D, Q> TryFrom<Stored<S, L, R, D, Q>> for ScopeGraph<S, L, R, D, Q> {
    type Error = &'static str;

    /// Accepts the tables `add_scope`, `add_datum`, and `add_reference` could
    /// have produced: every scope present, one data row and one reference row
    /// per scope, and each row listing, in identity order, exactly the data or
    /// references that name its scope.
    fn try_from(stored: Stored<S, L, R, D, Q>) -> Result<Self, Self::Error> {
        let Stored {
            graph,
            data,
            references,
            scope_data,
            scope_references,
        } = stored;
        let scopes = graph.node_bound();
        if graph.node_count() != scopes {
            return Err("a scope graph never removes a scope, but one slot is vacant");
        }
        if grouped_by_scope(
            data.iter().map(ScopeDatum::scope),
            scopes,
            ScopeDatumId::from_index,
        )
        .is_none_or(|rows| rows != scope_data)
        {
            return Err("the per-scope data lists are not the data grouped by their scope");
        }
        if grouped_by_scope(
            references.iter().map(ScopeReference::scope),
            scopes,
            ScopeReferenceId::from_index,
        )
        .is_none_or(|rows| rows != scope_references)
        {
            return Err("the per-scope reference lists are not the references grouped by scope");
        }
        Ok(Self {
            graph,
            data,
            references,
            scope_data,
            scope_references,
        })
    }
}

/// The table a graph keeps beside its entries: one row per scope listing, in
/// identity order, the entries whose owner is that scope.
///
/// `None` when an owner is not one of the `scopes`.
fn grouped_by_scope<I: Clone>(
    owners: impl Iterator<Item = ScopeId>,
    scopes: usize,
    identity: impl Fn(usize) -> I,
) -> Option<Vec<Vec<I>>> {
    let mut rows = vec![Vec::new(); scopes];
    for (index, owner) in owners.enumerate() {
        rows.get_mut(owner.index())?.push(identity(index));
    }
    Some(rows)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::Scope;
    use super::*;

    #[test]
    fn a_vacant_scope_slot_is_rejected() {
        let encoded = |removed: Option<usize>| {
            let mut store = ScopeStore::<(), ()>::tagged();
            let scopes = [
                store.add_node(Scope { payload: () }),
                store.add_node(Scope { payload: () }),
            ];
            if let Some(index) = removed {
                assert!(store.remove_node(scopes[index]));
            }
            json!({
                "graph": store,
                "data": [],
                "references": [],
                "scope_data": [[], []],
                "scope_references": [[], []],
            })
        };

        assert!(serde_json::from_value::<ScopeGraph>(encoded(None)).is_ok());
        assert!(serde_json::from_value::<ScopeGraph>(encoded(Some(1))).is_err());
    }
}
