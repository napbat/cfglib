//! Language-parametric scope graphs and name-resolution queries.
//!
//! A [`ScopeGraph`] stores scopes, labeled reachability edges, relation-tagged
//! data, and references without assigning meaning to any of those payloads.
//! [`ScopeResolution`] interprets that storage through a consumer-defined
//! [`ScopeGraphQuery`]: the query owns path-language state, declaration
//! matching, label specificity, and data ordering. This is the scope-graph
//! resolution calculus without language knowledge in the graph substrate.
//!
//! [`ScopeGraphQuery`] hooks see the stored path of each step. A second query
//! form has hooks that see no path:
//!
//! - A [`ScopeCursorQuery`] sees only a [`ScopeCursor`]: the start, the end
//!   and the length of a path.
//! - [`ScopeFirstResolution`] follows the one path that a cursor query
//!   admits and stops at the first accepted datum. It stores no path, so a
//!   walk of `D` scopes costs `O(D)` hook calls and allocates nothing.
//! - [`ScopeCursor::of_path`] gives the cursor of a stored path, so a policy
//!   type can also implement `ScopeGraphQuery` through its cursor hooks and
//!   run under every resolver.

mod cursor;
mod first;
mod query;
mod storage;

#[cfg(test)]
mod tests;

pub use cursor::{ScopeCursor, ScopeCursorQuery};
pub use first::{ScopeFirstResolution, ScopeFirstResolutionError};
pub use query::{
    ScopeGraphPathLabel, ScopeGraphQuery, ScopeLinearResolutionError, ScopePath, ScopeQuery,
    ScopeResolution, ScopeResolutionCandidate, ScopeResolutionConfig, ScopeResolutionIndex,
    ScopeResolutionStats,
};
pub use storage::{
    Scope, ScopeDatum, ScopeDatumId, ScopeEdgeId, ScopeEdgeTag, ScopeGraph, ScopeId,
    ScopeReference, ScopeReferenceId, ScopeTag,
};
