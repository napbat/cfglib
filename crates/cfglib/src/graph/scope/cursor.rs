//! Query hooks without a path: the policy sees where a path starts and ends
//! and how many hops it has, but not its scopes or edges.

use core::cmp::Ordering;

use super::query::{ScopeGraphPathLabel, ScopeGraphQuery, ScopePath, ScopeQuery};
use super::storage::{ScopeDatumId, ScopeEdgeId, ScopeGraph, ScopeId};
use crate::graph::edge_view::EdgeRef;

/// The endpoints and the hop count of a scope path, without its scopes or
/// edges.
///
/// A [`ScopeCursorQuery`] sees only this summary, so a resolver can follow a
/// path without storing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeCursor {
    start: ScopeId,
    end: ScopeId,
    hop_count: usize,
}

impl ScopeCursor {
    pub(super) const fn new(start: ScopeId, end: ScopeId, hop_count: usize) -> Self {
        Self {
            start,
            end,
            hop_count,
        }
    }

    /// Returns the cursor one hop further on, at `target`.
    pub(super) const fn advanced(self, target: ScopeId) -> Self {
        Self::new(self.start, target, self.hop_count + 1)
    }

    /// Returns the endpoints and the hop count of the stored `path`.
    fn of_path(path: &ScopePath) -> Self {
        Self::new(path.start(), path.end(), path.hop_count())
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
    pub const fn hop_count(&self) -> usize {
        self.hop_count
    }

    /// Returns whether this is the path without edges at the start scope of
    /// the query.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.hop_count == 0
    }
}

/// A scope-graph query whose hooks never read the scopes or edges of a path.
///
/// Each hook is the [`ScopeGraphQuery`] hook of the same name, with a
/// [`ScopeCursor`] in place of the [`ScopePath`]. A cursor query has no
/// shadowing order: [`ScopeFirstResolution`](super::ScopeFirstResolution)
/// runs it along one path without storing it and stops at the first datum,
/// and [`ScopeCursorPolicy`] runs it under every other resolver.
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
}

/// A [`ScopeCursorQuery`] run as a [`ScopeGraphQuery`], so that one policy
/// serves every resolver.
///
/// Each hook gives the cursor of the stored path to the cursor hook of the
/// same name. A cursor query has no shadowing order, so no candidate shadows
/// another: [`ScopeResolution::compute`](super::ScopeResolution::compute)
/// keeps every candidate the policy accepts. On a path that the policy
/// admits one edge at a time, the first of those candidates is the datum
/// that [`ScopeFirstResolution`](super::ScopeFirstResolution) finds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ScopeCursorPolicy<P>(pub P);

impl<S, L, R, D, Q, P> ScopeGraphQuery<S, L, R, D, Q> for ScopeCursorPolicy<P>
where
    P: ScopeCursorQuery<S, L, R, D, Q>,
{
    type State = P::State;

    fn initial_state(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
    ) -> Self::State {
        self.0.initial_state(graph, input)
    }

    fn step(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        path: &ScopePath,
        state: &Self::State,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, L>,
    ) -> Option<Self::State> {
        self.0
            .step(graph, input, ScopeCursor::of_path(path), state, edge)
    }

    fn visit_candidate_data(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        path: &ScopePath,
        state: &Self::State,
        visit: &mut impl FnMut(ScopeDatumId),
    ) {
        self.0
            .visit_candidate_data(graph, input, ScopeCursor::of_path(path), state, visit);
    }

    fn accepts(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        path: &ScopePath,
        state: &Self::State,
        datum: ScopeDatumId,
    ) -> bool {
        self.0
            .accepts(graph, input, ScopeCursor::of_path(path), state, datum)
    }

    fn visit_accepted_data(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        path: &ScopePath,
        state: &Self::State,
        visit: &mut impl FnMut(ScopeDatumId),
    ) {
        self.0
            .visit_accepted_data(graph, input, ScopeCursor::of_path(path), state, visit);
    }

    fn should_extend(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        path: &ScopePath,
        state: &Self::State,
        accepted_candidate_count: usize,
    ) -> bool {
        self.0.should_extend(
            graph,
            input,
            ScopeCursor::of_path(path),
            state,
            accepted_candidate_count,
        )
    }

    /// No two labels are comparable.
    fn compare_labels(
        &self,
        _graph: &ScopeGraph<S, L, R, D, Q>,
        _input: &ScopeQuery<'_, Q>,
        _left: ScopeGraphPathLabel<'_, L>,
        _right: ScopeGraphPathLabel<'_, L>,
    ) -> Option<Ordering> {
        None
    }

    /// No candidate shadows another.
    fn shadows(
        &self,
        _graph: &ScopeGraph<S, L, R, D, Q>,
        _input: &ScopeQuery<'_, Q>,
        _left_path: &ScopePath,
        _left_datum: ScopeDatumId,
        _right_path: &ScopePath,
        _right_datum: ScopeDatumId,
    ) -> bool {
        false
    }
}
