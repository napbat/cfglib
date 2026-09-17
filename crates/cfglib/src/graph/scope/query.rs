//! Generic scope-graph resolution and reverse indexing.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
use smallvec::SmallVec;

use crate::graph::edge_view::EdgeRef;

use super::storage::{ScopeDatumId, ScopeEdgeId, ScopeGraph, ScopeId, ScopeReferenceId};

/// Borrowed input for one stored or ephemeral scope-graph query.
///
/// Stored references expose their stable identity through [`reference`](Self::reference).
/// Ephemeral queries support completion, speculative rename checks, and other
/// editor requests without mutating the graph merely to allocate a reference.
#[derive(Debug, Clone, Copy)]
pub struct ScopeQuery<'a, Q> {
    scope: ScopeId,
    data: &'a Q,
    reference: Option<ScopeReferenceId>,
}

impl<'a, Q> ScopeQuery<'a, Q> {
    /// Create an ephemeral query beginning at `scope`.
    #[must_use]
    pub const fn new(scope: ScopeId, data: &'a Q) -> Self {
        Self {
            scope,
            data,
            reference: None,
        }
    }

    /// Borrow one stored reference as a query input.
    ///
    /// # Panics
    ///
    /// Panics when `reference` does not belong to `graph`.
    #[must_use]
    pub fn from_reference<S, L, R, D>(
        graph: &'a ScopeGraph<S, L, R, D, Q>,
        reference: ScopeReferenceId,
    ) -> Self {
        let stored = graph.reference(reference);
        Self {
            scope: stored.scope(),
            data: stored.data(),
            reference: Some(reference),
        }
    }

    /// Scope where lookup starts.
    #[must_use]
    pub const fn scope(&self) -> ScopeId {
        self.scope
    }

    /// Borrow the consumer-defined query data.
    #[must_use]
    pub const fn data(&self) -> &'a Q {
        self.data
    }

    /// Stable stored-reference identity, if this query came from the graph.
    #[must_use]
    pub const fn reference(&self) -> Option<ScopeReferenceId> {
        self.reference
    }
}

/// One element in the extended label alphabet used to order resolution paths.
///
/// [`Declaration`](Self::Declaration) is the formal end-of-path marker. A
/// query compares it with edge labels to say, for example, that a declaration
/// in the current scope shadows any result reached through a lexical-parent
/// edge.
#[derive(Debug, Clone, Copy)]
pub enum ScopeGraphPathLabel<'a, L> {
    /// A consumer-defined scope-edge label.
    Edge(&'a L),
    /// The declaration marker at the end of a resolution path.
    Declaration,
}

/// A cycle-free path through a [`ScopeGraph`].
///
/// `scopes().len()` is always `edges().len() + 1`. The first scope is the
/// query origin and the final scope owns any candidate associated with this
/// path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopePath {
    scopes: SmallVec<[ScopeId; 4]>,
    edges: SmallVec<[ScopeEdgeId; 4]>,
}

impl ScopePath {
    fn new(start: ScopeId) -> Self {
        let mut scopes = SmallVec::new();
        scopes.push(start);
        Self {
            scopes,
            edges: SmallVec::new(),
        }
    }

    fn extended(&self, edge: ScopeEdgeId, target: ScopeId) -> Self {
        let mut path = self.clone();
        path.edges.push(edge);
        path.scopes.push(target);
        path
    }

    /// The scope where this path starts.
    #[must_use]
    pub fn start(&self) -> ScopeId {
        self.scopes[0]
    }

    /// The scope where this path currently ends.
    ///
    /// # Panics
    ///
    /// Panics only if the path's internal nonempty-scope invariant is broken.
    #[must_use]
    pub fn end(&self) -> ScopeId {
        *self
            .scopes
            .last()
            .expect("a scope path always contains its start")
    }

    /// Scopes in traversal order, including both endpoints.
    #[must_use]
    pub fn scopes(&self) -> &[ScopeId] {
        &self.scopes
    }

    /// Edge identities in traversal order.
    #[must_use]
    pub fn edges(&self) -> &[ScopeEdgeId] {
        &self.edges
    }

    /// The number of scope-edge hops in this path.
    #[must_use]
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Whether this is the zero-edge path at the query's starting scope.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Whether this path already contains `scope`.
    #[must_use]
    pub fn contains_scope(&self, scope: ScopeId) -> bool {
        self.scopes.contains(&scope)
    }
}

/// Language-defined semantics for one family of scope-graph queries.
///
/// The graph owns only facts. This trait supplies the policies in the
/// scope-graph resolution calculus:
///
/// - `initial_state`, `step`, and `accepts` recognize the regular language of
///   admissible path labels and match relation-tagged data;
/// - `compare_labels` supplies the strict partial order that determines which
///   path shadows another;
/// - `data_leq` supplies the data-specificity preorder that gates shadowing.
/// - `shadows` can replace those default orders with a full path/data order.
///
/// The state may also carry query-specific context such as whether an import
/// edge has been traversed, a redirect target is being sought, or an
/// accessibility boundary has been crossed. Returning `None` from `step`
/// rejects that edge and its descendants.
pub trait ScopeGraphQuery<S, L, R, D, Q> {
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
    /// `path` still ends at `edge.source()`; this lets the policy inspect the
    /// complete prefix before deciding whether the transition is admissible.
    fn step(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        path: &ScopePath,
        state: &Self::State,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, L>,
    ) -> Option<Self::State>;

    /// Visit the data that can possibly match at the end of `path`.
    ///
    /// The default visits every datum owned by the current scope. Queries with
    /// a name, namespace, type, or other secondary index can override this hook
    /// to avoid asking [`accepts`](Self::accepts) about unrelated data. An
    /// override must visit every datum that `accepts` could return `true` for;
    /// omitted data are intentionally invisible to the resolution.
    fn visit_candidate_data(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        _input: &ScopeQuery<'_, Q>,
        path: &ScopePath,
        _state: &Self::State,
        visit: &mut impl FnMut(ScopeDatumId),
    ) {
        for &datum in graph.scope_data(path.end()) {
            visit(datum);
        }
    }

    /// Whether `datum` is a reachable candidate at the end of `path`.
    ///
    /// This hook owns both regular-language acceptance and declaration
    /// matching. It can compare names and namespaces, enforce ordered
    /// declaration positions, visibility, scope opacity, or any other
    /// consumer-defined condition.
    fn accepts(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        path: &ScopePath,
        state: &Self::State,
        datum: ScopeDatumId,
    ) -> bool;

    /// Visit data accepted at the end of `path`.
    ///
    /// The default filters [`visit_candidate_data`](Self::visit_candidate_data)
    /// through [`accepts`](Self::accepts). A query whose secondary index already
    /// performs the complete acceptance check can override this hook to avoid
    /// repeating that work. An override must visit exactly the data accepted by
    /// the query policy.
    fn visit_accepted_data(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        path: &ScopePath,
        state: &Self::State,
        visit: &mut impl FnMut(ScopeDatumId),
    ) {
        self.visit_candidate_data(graph, input, path, state, &mut |datum| {
            if self.accepts(graph, input, path, state, datum) {
                visit(datum);
            }
        });
    }

    /// Whether to explore outgoing edges after testing the current scope.
    ///
    /// `accepted_candidate_count` counts candidates accepted on `path`, before
    /// shadowing. The default keeps exploring. A query may return `false` when
    /// those candidates decisively shadow every candidate reachable through an
    /// extension, or when the input can never resolve (for example, a reference
    /// class excluded by the language policy). Incorrectly pruning here can omit
    /// visible candidates.
    fn should_extend(
        &self,
        _graph: &ScopeGraph<S, L, R, D, Q>,
        _input: &ScopeQuery<'_, Q>,
        _path: &ScopePath,
        _state: &Self::State,
        _accepted_candidate_count: usize,
    ) -> bool {
        true
    }

    /// Compare two extended labels for path specificity.
    ///
    /// [`Ordering::Less`] means `left` is more specific and therefore shadows
    /// `right`. `None` makes the labels incomparable, preserving candidates
    /// reached through both paths as a possible ambiguity.
    fn compare_labels(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        left: ScopeGraphPathLabel<'_, L>,
        right: ScopeGraphPathLabel<'_, L>,
    ) -> Option<Ordering>;

    /// Whether `left` is at least as data-specific as `right`.
    ///
    /// Path shadowing removes the right-hand candidate only when this returns
    /// `true`. The default models declarations without an additional data
    /// order, where every matching datum is comparable.
    fn data_leq(
        &self,
        _graph: &ScopeGraph<S, L, R, D, Q>,
        _input: &ScopeQuery<'_, Q>,
        _left: ScopeDatumId,
        _right: ScopeDatumId,
    ) -> bool {
        true
    }

    /// Whether `left` is strictly more specific than `right`.
    ///
    /// The default is the standard scope-graph order: lexicographically
    /// compare the paths' extended label words, then require `left_datum` to
    /// be at least as data-specific as `right_datum`. Override this hook for a
    /// full path order that also depends on scopes, accessibility, receivers,
    /// or other query-specific context.
    fn shadows(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        left_path: &ScopePath,
        left_datum: ScopeDatumId,
        right_path: &ScopePath,
        right_datum: ScopeDatumId,
    ) -> bool {
        label_path_shadows(graph, input, self, left_path, right_path)
            && self.data_leq(graph, input, left_datum, right_datum)
    }
}

/// Deterministic bounds for a scope-graph resolution query.
///
/// Resolution is finite without bounds because paths never repeat a scope.
/// Bounds are useful when querying adversarial graphs whose number of simple
/// paths is exponential.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ScopeResolutionConfig {
    /// Maximum number of edges in an explored path. `None` is unbounded.
    pub max_path_length: Option<usize>,
    /// Maximum number of path prefixes to examine. `None` is unbounded.
    pub max_path_count: Option<usize>,
}

impl ScopeResolutionConfig {
    /// Construct an unbounded configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_path_length: None,
            max_path_count: None,
        }
    }

    /// Return this configuration with a path-length bound.
    #[must_use]
    pub const fn with_max_path_length(mut self, max_path_length: usize) -> Self {
        self.max_path_length = Some(max_path_length);
        self
    }

    /// Return this configuration with an explored-path-count bound.
    #[must_use]
    pub const fn with_max_path_count(mut self, max_path_count: usize) -> Self {
        self.max_path_count = Some(max_path_count);
        self
    }
}

/// One visible resolution path from a reference to relation-tagged data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeResolutionCandidate {
    datum: ScopeDatumId,
    path: ScopePath,
}

impl ScopeResolutionCandidate {
    /// The relation-tagged datum reached by this path.
    #[must_use]
    pub const fn datum(&self) -> ScopeDatumId {
        self.datum
    }

    /// The visible, cycle-free scope path.
    #[must_use]
    pub const fn path(&self) -> &ScopePath {
        &self.path
    }
}

/// Work and truncation information for one resolution query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ScopeResolutionStats {
    /// Number of distinct path prefixes examined.
    pub explored_path_count: usize,
    /// Number of matching candidates before visibility filtering.
    pub reachable_candidate_count: usize,
    /// Whether at least one otherwise admissible extension hit the length bound.
    pub path_length_limited: bool,
    /// Whether queued work remained when the path-count bound was reached.
    pub path_count_limited: bool,
}

/// A query violated the single-candidate, single-successor contract required
/// by linear resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeLinearResolutionError {
    /// One path prefix accepted more than one datum.
    MultipleCandidates {
        /// Scope where the competing candidates were accepted.
        scope: ScopeId,
    },
    /// One path prefix admitted more than one outgoing edge.
    MultipleEdges {
        /// Scope where the traversal branched.
        scope: ScopeId,
    },
}

impl core::fmt::Display for ScopeLinearResolutionError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MultipleCandidates { scope } => {
                write!(formatter, "scope {scope:?} accepted multiple candidates")
            }
            Self::MultipleEdges { scope } => {
                write!(
                    formatter,
                    "scope {scope:?} admitted multiple outgoing edges"
                )
            }
        }
    }
}

impl core::error::Error for ScopeLinearResolutionError {}

impl ScopeResolutionStats {
    /// Whether configured bounds may have omitted a candidate.
    #[must_use]
    pub const fn is_truncated(&self) -> bool {
        self.path_length_limited || self.path_count_limited
    }
}

/// Visible paths produced for one stored or ephemeral scope-graph query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeResolution {
    start_scope: ScopeId,
    reference: Option<ScopeReferenceId>,
    candidates: SmallVec<[ScopeResolutionCandidate; 1]>,
    stats: ScopeResolutionStats,
}

impl ScopeResolution {
    /// Resolve one stored reference under `query` and `config`.
    ///
    /// Search enumerates cycle-free paths in breadth-first order and preserves
    /// edge insertion order within each prefix. Every matching path is first
    /// recorded as reachable; the label and data orders then remove candidates
    /// shadowed by a more-specific path.
    ///
    /// # Panics
    ///
    /// Panics when `reference` does not belong to `graph`.
    #[must_use]
    pub fn compute<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        reference: ScopeReferenceId,
        query: &P,
        config: ScopeResolutionConfig,
    ) -> Self
    where
        P: ScopeGraphQuery<S, L, R, D, Q>,
    {
        Self::compute_query(
            graph,
            &ScopeQuery::from_reference(graph, reference),
            query,
            config,
        )
    }

    /// Resolve an ephemeral query beginning in `scope`.
    ///
    /// This is the non-mutating entry point for completion and speculative
    /// editor queries. It applies the same path language, shadowing, bounds,
    /// and deterministic ordering as stored-reference resolution.
    ///
    /// # Panics
    ///
    /// Panics when `scope` does not belong to `graph`.
    #[must_use]
    pub fn compute_from<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        scope: ScopeId,
        data: &Q,
        query: &P,
        config: ScopeResolutionConfig,
    ) -> Self
    where
        P: ScopeGraphQuery<S, L, R, D, Q>,
    {
        let _ = graph.scope(scope);
        Self::compute_query(graph, &ScopeQuery::new(scope, data), query, config)
    }

    /// Resolve one stored reference whose policy admits at most one candidate
    /// and one outgoing edge at every path prefix.
    ///
    /// This retains the same path, shadowing, bound, and statistics semantics as
    /// [`compute`](Self::compute), while updating one path in place instead of
    /// maintaining a branching frontier.
    ///
    /// # Errors
    ///
    /// Returns [`ScopeLinearResolutionError`] if a prefix accepts several data
    /// or admits several cycle-free outgoing edges.
    ///
    /// # Panics
    ///
    /// Panics when `reference` does not belong to `graph`.
    pub fn try_compute_linear<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        reference: ScopeReferenceId,
        query: &P,
        config: ScopeResolutionConfig,
    ) -> Result<Self, ScopeLinearResolutionError>
    where
        P: ScopeGraphQuery<S, L, R, D, Q>,
    {
        Self::try_compute_linear_query(
            graph,
            &ScopeQuery::from_reference(graph, reference),
            query,
            config,
        )
    }

    /// Resolve one ephemeral query whose policy admits at most one candidate
    /// and one outgoing edge at every path prefix.
    ///
    /// This is the non-mutating counterpart of
    /// [`try_compute_linear`](Self::try_compute_linear).
    ///
    /// # Errors
    ///
    /// Returns [`ScopeLinearResolutionError`] if a prefix accepts several data
    /// or admits several cycle-free outgoing edges.
    ///
    /// # Panics
    ///
    /// Panics when `scope` does not belong to `graph`.
    pub fn try_compute_linear_from<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        scope: ScopeId,
        data: &Q,
        query: &P,
        config: ScopeResolutionConfig,
    ) -> Result<Self, ScopeLinearResolutionError>
    where
        P: ScopeGraphQuery<S, L, R, D, Q>,
    {
        let _ = graph.scope(scope);
        Self::try_compute_linear_query(graph, &ScopeQuery::new(scope, data), query, config)
    }

    fn try_compute_linear_query<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        query: &P,
        config: ScopeResolutionConfig,
    ) -> Result<Self, ScopeLinearResolutionError>
    where
        P: ScopeGraphQuery<S, L, R, D, Q>,
    {
        let mut path = ScopePath::new(input.scope());
        let mut state = query.initial_state(graph, input);
        let mut reachable: SmallVec<[ScopeResolutionCandidate; 1]> = SmallVec::new();
        let mut resolution_stats = ScopeResolutionStats::default();

        loop {
            if config
                .max_path_count
                .is_some_and(|limit| resolution_stats.explored_path_count >= limit)
            {
                resolution_stats.path_count_limited = true;
                break;
            }
            resolution_stats.explored_path_count += 1;

            let mut accepted = None;
            let mut multiple_candidates = false;
            query.visit_accepted_data(graph, input, &path, &state, &mut |datum| {
                if accepted.replace(datum).is_some() {
                    multiple_candidates = true;
                }
            });
            if multiple_candidates {
                return Err(ScopeLinearResolutionError::MultipleCandidates { scope: path.end() });
            }
            let accepted_candidate_count = usize::from(accepted.is_some());
            if let Some(datum) = accepted {
                reachable.push(ScopeResolutionCandidate {
                    datum,
                    path: path.clone(),
                });
            }
            if !query.should_extend(graph, input, &path, &state, accepted_candidate_count) {
                break;
            }

            let at_length_limit = config
                .max_path_length
                .is_some_and(|limit| path.len() >= limit);
            let mut next = None;
            for edge_id in graph.outgoing(path.end()) {
                let edge = graph.edge(edge_id);
                if path.contains_scope(edge.target()) {
                    continue;
                }
                let Some(next_state) = query.step(graph, input, &path, &state, edge) else {
                    continue;
                };
                if at_length_limit {
                    resolution_stats.path_length_limited = true;
                    continue;
                }
                if next.is_some() {
                    return Err(ScopeLinearResolutionError::MultipleEdges { scope: path.end() });
                }
                next = Some((edge_id, edge.target(), next_state));
            }
            let Some((edge, target, next_state)) = next else {
                break;
            };
            path.edges.push(edge);
            path.scopes.push(target);
            state = next_state;
        }

        resolution_stats.reachable_candidate_count = reachable.len();
        Ok(Self::from_reachable(
            graph,
            input,
            query,
            reachable,
            resolution_stats,
        ))
    }

    fn compute_query<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        query: &P,
        config: ScopeResolutionConfig,
    ) -> Self
    where
        P: ScopeGraphQuery<S, L, R, D, Q>,
    {
        let initial = Work {
            path: ScopePath::new(input.scope()),
            state: query.initial_state(graph, input),
        };
        let mut initial = Some(initial);
        let mut frontier: SmallVec<[Work<P::State>; 4]> = SmallVec::new();
        let mut reachable: SmallVec<[ScopeResolutionCandidate; 1]> = SmallVec::new();
        let mut stats = ScopeResolutionStats::default();

        while let Some(work) = initial.take().or_else(|| {
            if frontier.is_empty() {
                None
            } else {
                Some(frontier.remove(0))
            }
        }) {
            if config
                .max_path_count
                .is_some_and(|limit| stats.explored_path_count >= limit)
            {
                stats.path_count_limited = true;
                break;
            }
            stats.explored_path_count += 1;

            let reachable_before = reachable.len();
            query.visit_accepted_data(graph, input, &work.path, &work.state, &mut |datum| {
                reachable.push(ScopeResolutionCandidate {
                    datum,
                    path: work.path.clone(),
                });
            });
            let accepted_candidate_count = reachable.len() - reachable_before;
            if !query.should_extend(
                graph,
                input,
                &work.path,
                &work.state,
                accepted_candidate_count,
            ) {
                continue;
            }

            let at_length_limit = config
                .max_path_length
                .is_some_and(|limit| work.path.len() >= limit);
            for edge_id in graph.outgoing(work.path.end()) {
                let edge = graph.edge(edge_id);
                if work.path.contains_scope(edge.target()) {
                    continue;
                }
                let Some(next_state) = query.step(graph, input, &work.path, &work.state, edge)
                else {
                    continue;
                };
                if at_length_limit {
                    stats.path_length_limited = true;
                    continue;
                }
                frontier.push(Work {
                    path: work.path.extended(edge_id, edge.target()),
                    state: next_state,
                });
            }
        }

        stats.reachable_candidate_count = reachable.len();
        Self::from_reachable(graph, input, query, reachable, stats)
    }

    fn from_reachable<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        query: &P,
        reachable: SmallVec<[ScopeResolutionCandidate; 1]>,
        stats: ScopeResolutionStats,
    ) -> Self
    where
        P: ScopeGraphQuery<S, L, R, D, Q>,
    {
        let candidates = visible_candidates(graph, input, query, reachable);
        Self {
            start_scope: input.scope(),
            reference: input.reference(),
            candidates,
            stats,
        }
    }

    /// The scope where this query started.
    #[must_use]
    pub const fn start_scope(&self) -> ScopeId {
        self.start_scope
    }

    /// The stored reference this result resolves, or `None` for an ephemeral query.
    #[must_use]
    pub const fn reference(&self) -> Option<ScopeReferenceId> {
        self.reference
    }

    /// All visible paths, in stable discovery order.
    #[must_use]
    pub fn candidates(&self) -> &[ScopeResolutionCandidate] {
        &self.candidates
    }

    /// Query work and truncation information.
    #[must_use]
    pub const fn stats(&self) -> ScopeResolutionStats {
        self.stats
    }

    /// Whether at least one datum is visible.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        !self.candidates.is_empty()
    }

    /// The number of distinct visible data targets.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.candidates
            .iter()
            .map(ScopeResolutionCandidate::datum)
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// Whether more than one distinct datum remains visible.
    #[must_use]
    pub fn is_ambiguous(&self) -> bool {
        self.target_count() > 1
    }
}

/// Resolution results for every stored reference, plus a reverse index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeResolutionIndex {
    resolutions: Vec<ScopeResolution>,
    references_by_datum: Vec<Vec<ScopeReferenceId>>,
}

impl ScopeResolutionIndex {
    /// Resolve every reference in identity order and build the reverse index.
    #[must_use]
    pub fn compute<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        query: &P,
        config: ScopeResolutionConfig,
    ) -> Self
    where
        P: ScopeGraphQuery<S, L, R, D, Q>,
    {
        let resolutions: Vec<_> = graph
            .reference_ids()
            .map(|reference| ScopeResolution::compute(graph, reference, query, config))
            .collect();
        let mut references_by_datum = vec![Vec::new(); graph.datum_count()];
        for (reference, resolution) in graph.reference_ids().zip(&resolutions) {
            let mut seen = BTreeSet::new();
            for candidate in resolution.candidates() {
                if seen.insert(candidate.datum()) {
                    references_by_datum[candidate.datum().index()].push(reference);
                }
            }
        }
        Self {
            resolutions,
            references_by_datum,
        }
    }

    /// Borrow one reference's resolution.
    ///
    /// # Panics
    ///
    /// Panics when `reference` was not present in the resolved graph.
    #[must_use]
    pub fn resolution(&self, reference: ScopeReferenceId) -> &ScopeResolution {
        &self.resolutions[reference.index()]
    }

    /// References with a visible path to `datum`, in reference identity order.
    ///
    /// Each reference occurs at most once even when it has several visible
    /// paths to the same datum.
    ///
    /// # Panics
    ///
    /// Panics when `datum` was not present in the resolved graph.
    #[must_use]
    pub fn references_to(&self, datum: ScopeDatumId) -> &[ScopeReferenceId] {
        &self.references_by_datum[datum.index()]
    }

    /// Results in reference identity order.
    #[must_use]
    pub fn resolutions(&self) -> &[ScopeResolution] {
        &self.resolutions
    }
}

struct Work<T> {
    path: ScopePath,
    state: T,
}

fn visible_candidates<S, L, R, D, Q, P>(
    graph: &ScopeGraph<S, L, R, D, Q>,
    input: &ScopeQuery<'_, Q>,
    query: &P,
    reachable: SmallVec<[ScopeResolutionCandidate; 1]>,
) -> SmallVec<[ScopeResolutionCandidate; 1]>
where
    P: ScopeGraphQuery<S, L, R, D, Q> + ?Sized,
{
    if reachable.len() < 2 {
        return reachable;
    }
    reachable
        .iter()
        .enumerate()
        .filter(|&(index, candidate)| {
            !reachable.iter().enumerate().any(|(other_index, other)| {
                index != other_index && candidate_shadows(graph, input, query, other, candidate)
            })
        })
        .map(|(_, candidate)| candidate.clone())
        .collect()
}

fn candidate_shadows<S, L, R, D, Q, P>(
    graph: &ScopeGraph<S, L, R, D, Q>,
    input: &ScopeQuery<'_, Q>,
    query: &P,
    left: &ScopeResolutionCandidate,
    right: &ScopeResolutionCandidate,
) -> bool
where
    P: ScopeGraphQuery<S, L, R, D, Q> + ?Sized,
{
    query.shadows(
        graph,
        input,
        left.path(),
        left.datum(),
        right.path(),
        right.datum(),
    )
}

fn label_path_shadows<S, L, R, D, Q, P>(
    graph: &ScopeGraph<S, L, R, D, Q>,
    input: &ScopeQuery<'_, Q>,
    query: &P,
    left: &ScopePath,
    right: &ScopePath,
) -> bool
where
    P: ScopeGraphQuery<S, L, R, D, Q> + ?Sized,
{
    let mut index = 0;
    loop {
        let left_edge = left.edges.get(index).copied();
        let right_edge = right.edges.get(index).copied();
        let left_label = left_edge.map_or(ScopeGraphPathLabel::Declaration, |edge| {
            ScopeGraphPathLabel::Edge(graph.edge(edge).data())
        });
        let right_label = right_edge.map_or(ScopeGraphPathLabel::Declaration, |edge| {
            ScopeGraphPathLabel::Edge(graph.edge(edge).data())
        });

        match query.compare_labels(graph, input, left_label, right_label) {
            Some(Ordering::Less) => return true,
            Some(Ordering::Greater) | None => return false,
            Some(Ordering::Equal) => {}
        }

        if left_edge.is_none() || right_edge.is_none() {
            return false;
        }
        index += 1;
    }
}
