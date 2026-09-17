//! Language-parametric scope graphs and name-resolution queries.
//!
//! A [`ScopeGraph`] stores scopes, labeled reachability edges, relation-tagged
//! data, and references without assigning meaning to any of those payloads.
//! [`ScopeResolution`] interprets that storage through a consumer-defined
//! [`ScopeGraphQuery`]: the query owns path-language state, declaration
//! matching, label specificity, and data ordering. This is the scope-graph
//! resolution calculus without language knowledge in the graph substrate.

mod query;
mod storage;

#[cfg(test)]
mod tests;

pub use query::{
    ScopeGraphPathLabel, ScopeGraphQuery, ScopeLinearResolutionError, ScopePath, ScopeQuery,
    ScopeResolution, ScopeResolutionCandidate, ScopeResolutionConfig, ScopeResolutionIndex,
    ScopeResolutionStats,
};
pub use storage::{
    Scope, ScopeDatum, ScopeDatumId, ScopeEdgeId, ScopeEdgeTag, ScopeGraph, ScopeId,
    ScopeReference, ScopeReferenceId, ScopeTag,
};
