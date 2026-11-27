//! Language-parametric scope graphs and name-resolution queries.
//!
//! A [`ScopeGraph`] stores scopes, labeled reachability edges, relation-tagged
//! data, and references without assigning meaning to any of those payloads.
//! [`ScopeResolution`] interprets that storage through a consumer-defined
//! [`ScopeGraphQuery`]: the query owns path-language state, declaration
//! matching, label specificity, and data ordering. This is the scope-graph
//! resolution calculus without language knowledge in the graph substrate.
//!
//! Every resolver takes its input as a [`ScopeQuery`] (a stored reference or
//! an ephemeral query) and its bounds as a [`PathSearchConfig`](crate::PathSearchConfig),
//! and reports [`PathSearchStats`](crate::PathSearchStats). The three query
//! traits share one vocabulary of hooks and differ in what a hook sees of
//! the path:
//!
//! - A [`ScopeGraphQuery`] hook sees the stored path of each step, and the
//!   query orders paths for shadowing.
//! - A [`ScopeCursorQuery`] hook sees only a [`ScopeCursor`]: the start, the
//!   end and the length of a path. [`ScopeFirstResolution`] follows the one
//!   path that such a query admits and stops at the first accepted datum. It
//!   stores no path, so a walk of `D` scopes costs `O(D)` hook calls and
//!   allocates nothing. [`ScopeCursorPolicy`] runs the same policy under
//!   [`ScopeResolution`].
//! - A [`ScopeEnvironmentQuery`] hook sees neither the path nor the
//!   reference: it gets the scope where a path ends and the path-language
//!   state there, and a key projects each datum to what shadowing and lookup
//!   compare. The visible data of a scope in a state is then the same for
//!   every reference that reaches it, so a [`ScopeEnvironmentCache`] computes
//!   each such environment once and shares it across references.

mod cursor;
mod environment;
mod first;
mod index;
mod query;
mod storage;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;

pub use cursor::{ScopeCursor, ScopeCursorPolicy, ScopeCursorQuery};
pub use environment::{ScopeEnvironmentCache, ScopeEnvironmentQuery};
pub use first::{ScopeFirstResolution, ScopeFirstResolutionError};
pub use index::{ScopeGraphChanges, ScopeIndexConfig, ScopeIndexUpdate, ScopeResolutionIndex};
pub use query::{
    ScopeGraphPathLabel, ScopeGraphQuery, ScopePath, ScopeQuery, ScopeResolution,
    ScopeResolutionCandidate,
};
pub use storage::{
    Scope, ScopeDatum, ScopeDatumId, ScopeEdgeId, ScopeEdgeTag, ScopeGraph, ScopeId,
    ScopeReference, ScopeReferenceId, ScopeTag,
};
