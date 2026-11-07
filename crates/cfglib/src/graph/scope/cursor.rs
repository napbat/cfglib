//! Path-free query hooks: the policy sees where a path starts and ends and how
//! long it is, never its scopes or edges.

use core::cmp::Ordering;

use crate::graph::edge_view::EdgeRef;

use super::query::{ScopeGraphPathLabel, ScopeGraphQuery, ScopePath, ScopeQuery};
use super::storage::{ScopeDatumId, ScopeEdgeId, ScopeGraph, ScopeId};

/// The endpoints and length of a scope path, without its scopes or edges.
///
/// A [`ScopeCursorQuery`] sees only this summary, so a resolver can follow a
/// path without storing it. [`ScopePath::cursor`] gives the cursor of a stored
/// path.
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

    /// The cursor one hop further on, at `target`.
    pub(super) const fn advanced(self, target: ScopeId) -> Self {
        Self::new(self.start, target, self.len + 1)
    }

    /// The scope where the path starts.
    #[must_use]
    pub const fn start(&self) -> ScopeId {
        self.start
    }

    /// The scope where the path currently ends.
    #[must_use]
    pub const fn end(&self) -> ScopeId {
        self.end
    }

    /// The number of scope-edge hops in the path.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether this is the zero-edge path at the query's starting scope.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A [`ScopeGraphQuery`] whose hooks never read the scopes or edges of a path.
///
/// Each hook matches the [`ScopeGraphQuery`] hook of the same name, with a
/// [`ScopeCursor`] in place of the [`ScopePath`]. Every implementation is also
/// a [`ScopeGraphQuery`] that shadows through the default label and data orders,
/// so one policy runs under every resolver. Only a cursor query runs under
/// [`ScopeFirstResolution`](super::ScopeFirstResolution), which follows one path
/// without storing it.
pub trait ScopeCursorQuery<S, L, R, D, Q> {
    /// Consumer-owned path-language state.
    type State: Clone;

    /// Construct the state for the zero-edge path at `input.scope()`.
    fn initial_state(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
    ) -> Self::State;

    /// Advance path-language state across `edge`, or reject the edge.
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

    /// Visit the data that can possibly match at the end of `cursor`.
    ///
    /// The default visits every datum owned by the current scope. An override
    /// must visit every datum that [`accepts`](Self::accepts) could return
    /// `true` for.
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

    /// Whether `datum` is a reachable candidate at the end of `cursor`.
    fn accepts(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        cursor: ScopeCursor,
        state: &Self::State,
        datum: ScopeDatumId,
    ) -> bool;

    /// Visit data accepted at the end of `cursor`.
    ///
    /// The default filters [`visit_candidate_data`](Self::visit_candidate_data)
    /// through [`accepts`](Self::accepts). An override must visit exactly the
    /// data accepted by the query policy.
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

    /// Whether to explore outgoing edges after testing the current scope.
    ///
    /// `accepted_candidate_count` counts candidates accepted at the end of
    /// `cursor`. The default keeps exploring.
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

    /// Compare two extended labels for path specificity, as
    /// [`ScopeGraphQuery::compare_labels`] does.
    fn compare_labels(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        left: ScopeGraphPathLabel<'_, L>,
        right: ScopeGraphPathLabel<'_, L>,
    ) -> Option<Ordering>;

    /// Whether `left` is at least as data-specific as `right`, as
    /// [`ScopeGraphQuery::data_leq`] decides. The default holds for every pair.
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

impl<S, L, R, D, Q, P> ScopeGraphQuery<S, L, R, D, Q> for P
where
    P: ScopeCursorQuery<S, L, R, D, Q>,
{
    type State = P::State;

    fn initial_state(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
    ) -> Self::State {
        ScopeCursorQuery::initial_state(self, graph, input)
    }

    fn step(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        path: &ScopePath,
        state: &Self::State,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, L>,
    ) -> Option<Self::State> {
        ScopeCursorQuery::step(self, graph, input, path.cursor(), state, edge)
    }

    fn visit_candidate_data(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        path: &ScopePath,
        state: &Self::State,
        visit: &mut impl FnMut(ScopeDatumId),
    ) {
        ScopeCursorQuery::visit_candidate_data(self, graph, input, path.cursor(), state, visit);
    }

    fn accepts(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        path: &ScopePath,
        state: &Self::State,
        datum: ScopeDatumId,
    ) -> bool {
        ScopeCursorQuery::accepts(self, graph, input, path.cursor(), state, datum)
    }

    fn visit_accepted_data(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        path: &ScopePath,
        state: &Self::State,
        visit: &mut impl FnMut(ScopeDatumId),
    ) {
        ScopeCursorQuery::visit_accepted_data(self, graph, input, path.cursor(), state, visit);
    }

    fn should_extend(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        path: &ScopePath,
        state: &Self::State,
        accepted_candidate_count: usize,
    ) -> bool {
        ScopeCursorQuery::should_extend(
            self,
            graph,
            input,
            path.cursor(),
            state,
            accepted_candidate_count,
        )
    }

    fn compare_labels(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        left: ScopeGraphPathLabel<'_, L>,
        right: ScopeGraphPathLabel<'_, L>,
    ) -> Option<Ordering> {
        ScopeCursorQuery::compare_labels(self, graph, input, left, right)
    }

    fn data_leq(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        left: ScopeDatumId,
        right: ScopeDatumId,
    ) -> bool {
        ScopeCursorQuery::data_leq(self, graph, input, left, right)
    }
}
