//! Query hooks without a path: the policy sees where a path starts and ends
//! and how long it is, but not its scopes or edges.

use core::cmp::Ordering;

use super::query::{ScopeGraphPathLabel, ScopePath, ScopeQuery};
use super::storage::{ScopeDatumId, ScopeEdgeId, ScopeGraph, ScopeId};
use crate::graph::edge_view::EdgeRef;

/// The endpoints and the length of a scope path, without its scopes or
/// edges.
///
/// A [`ScopeCursorQuery`] sees only this summary, so a resolver can follow a
/// path without storing it. [`ScopeCursor::of_path`] gives the cursor of a
/// stored path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeCursor {
    start: ScopeId,
    end: ScopeId,
    len: usize,
}

impl ScopeCursor {
    pub(super) const fn new(start: ScopeId, end: ScopeId, len: usize) -> Self {
        Self { start, end, len }
    }

    /// Returns the cursor one hop further on, at `target`.
    pub(super) const fn advanced(self, target: ScopeId) -> Self {
        Self::new(self.start, target, self.len + 1)
    }

    /// Returns the endpoints and the length of the stored `path`.
    #[must_use]
    pub fn of_path(path: &ScopePath) -> Self {
        Self::new(path.start(), path.end(), path.len())
    }

    /// Returns the scope where the path starts.
    #[must_use]
    pub const fn start(&self) -> ScopeId {
        self.start
    }

    /// Returns the scope where the path ends now.
    #[must_use]
    pub const fn end(&self) -> ScopeId {
        self.end
    }

    /// Returns the number of scope-edge hops in the path.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether this is the path without edges at the start scope of
    /// the query.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A scope-graph query whose hooks never read the scopes or edges of a path.
///
/// Each hook matches the [`ScopeGraphQuery`](super::ScopeGraphQuery) hook
/// of the same name, with a [`ScopeCursor`] in place of the
/// [`ScopePath`]. [`ScopeFirstResolution`](super::ScopeFirstResolution) runs
/// a cursor query along one path without storing it.
///
/// To run the same policy under [`ScopeResolution`](super::ScopeResolution), implement
/// [`ScopeGraphQuery`](super::ScopeGraphQuery) for the policy type: give
/// [`ScopeCursor::of_path`] of each path to the hook of the same name, and
/// keep the default shadowing order.
pub trait ScopeCursorQuery<S, L, R, D, Q> {
    /// The path-language state that the consumer owns.
    type State: Clone;

    /// Makes the state for the path without edges at `input.scope()`.
    fn initial_state(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
    ) -> Self::State;

    /// Advances the path-language state across `edge`, or rejects the edge.
    ///
    /// `cursor` still ends at `edge.source()`.
    fn step(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        cursor: ScopeCursor,
        state: &Self::State,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, L>,
    ) -> Option<Self::State>;

    /// Visits the data that can match at the end of `cursor`.
    ///
    /// The default visits every datum of the current scope. An override must
    /// visit every datum for which [`accepts`](Self::accepts) can return
    /// `true`.
    fn visit_candidate_data(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        _input: &ScopeQuery<'_, Q>,
        cursor: ScopeCursor,
        _state: &Self::State,
        visit: &mut impl FnMut(ScopeDatumId),
    ) {
        for &datum in graph.scope_data(cursor.end()) {
            visit(datum);
        }
    }

    /// Returns whether `datum` is a reachable candidate at the end of
    /// `cursor`.
    fn accepts(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        cursor: ScopeCursor,
        state: &Self::State,
        datum: ScopeDatumId,
    ) -> bool;

    /// Visits the data that the query accepts at the end of `cursor`.
    ///
    /// The default filters
    /// [`visit_candidate_data`](Self::visit_candidate_data) through
    /// [`accepts`](Self::accepts). An override must visit exactly the data
    /// that the query policy accepts.
    fn visit_accepted_data(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        cursor: ScopeCursor,
        state: &Self::State,
        visit: &mut impl FnMut(ScopeDatumId),
    ) {
        self.visit_candidate_data(graph, input, cursor, state, &mut |datum| {
            if self.accepts(graph, input, cursor, state, datum) {
                visit(datum);
            }
        });
    }

    /// Returns whether to explore the outgoing edges after the test of the
    /// current scope.
    ///
    /// `accepted_candidate_count` counts the candidates that the query
    /// accepted at the end of `cursor`. The default continues the
    /// exploration.
    fn should_extend(
        &self,
        _graph: &ScopeGraph<S, L, R, D, Q>,
        _input: &ScopeQuery<'_, Q>,
        _cursor: ScopeCursor,
        _state: &Self::State,
        _accepted_candidate_count: usize,
    ) -> bool {
        true
    }

    /// Compares two extended labels for path specificity, as
    /// [`ScopeGraphQuery::compare_labels`](super::ScopeGraphQuery::compare_labels) does.
    fn compare_labels(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        left: ScopeGraphPathLabel<'_, L>,
        right: ScopeGraphPathLabel<'_, L>,
    ) -> Option<Ordering>;

    /// Returns whether `left` is at least as data-specific as `right`, as
    /// [`ScopeGraphQuery::data_leq`](super::ScopeGraphQuery::data_leq) decides. The default
    /// holds for every pair.
    fn data_leq(
        &self,
        _graph: &ScopeGraph<S, L, R, D, Q>,
        _input: &ScopeQuery<'_, Q>,
        _left: ScopeDatumId,
        _right: ScopeDatumId,
    ) -> bool {
        true
    }
}
