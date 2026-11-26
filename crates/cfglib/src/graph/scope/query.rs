//! Generic scope-graph resolution of one query.

extern crate alloc;

use alloc::collections::BTreeSet;
use core::cmp::Ordering;
use smallvec::SmallVec;

use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::search::{
    PathBudget, PathFrontier, PathSearchConfig, PathSearchStats, SearchOrder, retain_unshadowed,
};
use crate::graph::view::assert_in_bound;

use super::index::ScopeReads;
use super::storage::{ScopeDatumId, ScopeEdgeId, ScopeGraph, ScopeId, ScopeReferenceId};

/// Borrowed input for one stored or ephemeral scope-graph query.
///
/// Stored references expose their stable identity through [`reference`](Self::reference).
/// Ephemeral queries support completion, speculative rename checks, and other
/// editor requests without mutating the graph merely to allocate a reference.
/// The query borrows its data, so it is `Copy` whatever the data type.
#[derive(Debug)]
pub struct ScopeQuery<'a, Q> {
    scope: ScopeId,
    data: &'a Q,
    reference: Option<ScopeReferenceId>,
}

impl<Q> Clone for ScopeQuery<'_, Q> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Q> Copy for ScopeQuery<'_, Q> {}

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
    pub(super) fn new(start: ScopeId) -> Self {
        let mut scopes = SmallVec::new();
        scopes.push(start);
        Self {
            scopes,
            edges: SmallVec::new(),
        }
    }

    /// Appends the hop across `edge` to `target`.
    pub(super) fn push(&mut self, edge: ScopeEdgeId, target: ScopeId) {
        self.edges.push(edge);
        self.scopes.push(target);
    }

    fn extended(&self, edge: ScopeEdgeId, target: ScopeId) -> Self {
        let mut path = self.clone();
        path.push(edge, target);
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
///
/// # Incremental updates
///
/// [`ScopeResolutionIndex::update`](super::ScopeResolutionIndex::update)
/// re-resolves only the references whose resolution read a changed scope. A
/// resolution reads the scopes at the ends of the path prefixes it expands:
/// their payloads, their data, and their outgoing edges with the edges'
/// labels and targets. The hooks (`step`, `accepts`, `shadows` and the rest)
/// may read those scopes and the query input, and nothing else of the graph;
/// for example, `step` may not read the payload or the data of an edge's
/// target, which the path has not reached yet. A policy that reads more must
/// report the scopes it read as changed too, or the update can keep a stale
/// resolution.
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
        let label = |&edge: &ScopeEdgeId| graph.edge(edge).data();
        label_word_shadows(
            left_path.edges().iter().map(label),
            right_path.edges().iter().map(label),
            |left, right| self.compare_labels(graph, input, left, right),
        ) && self.data_leq(graph, input, left_datum, right_datum)
    }
}

/// One visible resolution path from a reference to relation-tagged data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeResolutionCandidate {
    datum: ScopeDatumId,
    path: ScopePath,
}

impl ScopeResolutionCandidate {
    pub(super) const fn new(datum: ScopeDatumId, path: ScopePath) -> Self {
        Self { datum, path }
    }

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

/// Visible paths produced for one stored or ephemeral scope-graph query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeResolution {
    start_scope: ScopeId,
    reference: Option<ScopeReferenceId>,
    candidates: SmallVec<[ScopeResolutionCandidate; 1]>,
    stats: PathSearchStats,
}

impl ScopeResolution {
    /// Resolve `input` under `query` and `config`.
    ///
    /// The input is a stored reference ([`ScopeQuery::from_reference`]) or an
    /// ephemeral query ([`ScopeQuery::new`]): completion and speculative
    /// editor requests resolve without a change of the graph, under the same
    /// path language, shadowing, bounds, and ordering.
    ///
    /// Search enumerates cycle-free paths in breadth-first order and preserves
    /// edge insertion order within each prefix. Every matching path is first
    /// recorded as reachable; the label and data orders then remove candidates
    /// shadowed by a more-specific path. The search is breadth-first by
    /// contract and keeps every candidate, so it reads neither `config.order`
    /// nor `config.max_complete_path_count`.
    ///
    /// # Panics
    ///
    /// Panics when `input` starts in a scope that does not belong to `graph`.
    #[must_use]
    pub fn compute<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: ScopeQuery<'_, Q>,
        query: &P,
        config: PathSearchConfig,
    ) -> Self
    where
        P: ScopeGraphQuery<S, L, R, D, Q>,
    {
        Self::compute_query(graph, &input, query, config, None)
    }

    /// Resolves `input` by breadth-first search. With `reads`, records the
    /// scope at the end of each prefix the search expands: the scopes whose
    /// data and outgoing edges the resolution read.
    pub(super) fn compute_query<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        query: &P,
        config: PathSearchConfig,
        mut reads: Option<&mut ScopeReads>,
    ) -> Self
    where
        P: ScopeGraphQuery<S, L, R, D, Q>,
    {
        assert_in_bound(graph, input.scope(), "query scope");
        let initial = Work {
            path: ScopePath::new(input.scope()),
            state: query.initial_state(graph, input),
        };
        // The seed is expanded without touching the frontier, so a query that
        // never branches allocates no queue.
        let mut seed = Some(initial);
        let mut frontier = PathFrontier::new(SearchOrder::BreadthFirst);
        let mut budget = PathBudget::new(&config);
        let mut candidates: SmallVec<[ScopeResolutionCandidate; 1]> = SmallVec::new();

        while let Some(work) = seed.take().or_else(|| frontier.pop()) {
            if !budget.admit_expansion() {
                break;
            }
            if let Some(reads) = reads.as_deref_mut() {
                reads.read(work.path.end());
            }

            let reachable_before = candidates.len();
            query.visit_accepted_data(graph, input, &work.path, &work.state, &mut |datum| {
                candidates.push(ScopeResolutionCandidate {
                    datum,
                    path: work.path.clone(),
                });
            });
            let accepted_candidate_count = candidates.len() - reachable_before;
            if !query.should_extend(
                graph,
                input,
                &work.path,
                &work.state,
                accepted_candidate_count,
            ) {
                continue;
            }

            for edge_id in graph.outgoing(work.path.end()) {
                let edge = graph.edge(edge_id);
                if work.path.contains_scope(edge.target()) {
                    continue;
                }
                let Some(next_state) = query.step(graph, input, &work.path, &work.state, edge)
                else {
                    continue;
                };
                if !budget.admit_length(work.path.len() + 1) {
                    continue;
                }
                frontier.push(Work {
                    path: work.path.extended(edge_id, edge.target()),
                    state: next_state,
                });
            }
        }

        // Every candidate is retained, so the complete paths are the
        // candidates before shadowing.
        let stats = PathSearchStats {
            complete_path_count: candidates.len(),
            ..budget.stats()
        };
        retain_unshadowed(&mut candidates, |other, candidate| {
            query.shadows(
                graph,
                input,
                other.path(),
                other.datum(),
                candidate.path(),
                candidate.datum(),
            )
        });
        Self::from_visible(input, candidates, stats)
    }

    /// A result for `input` whose `candidates` are already the visible ones.
    pub(super) fn from_visible<Q>(
        input: &ScopeQuery<'_, Q>,
        candidates: SmallVec<[ScopeResolutionCandidate; 1]>,
        stats: PathSearchStats,
    ) -> Self {
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
    ///
    /// `complete_path_count` counts the candidates before shadowing.
    #[must_use]
    pub const fn stats(&self) -> PathSearchStats {
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

struct Work<T> {
    path: ScopePath,
    state: T,
}

/// Whether the extended label word of the left path is lexicographically
/// more specific than the one of the right path, under `compare`.
///
/// A word is the edge labels of its path followed by the
/// [`Declaration`](ScopeGraphPathLabel::Declaration) marker. The words are
/// compared label by label: the first pair that `compare` orders decides,
/// `Less` for the left path. A pair it cannot order, or two words that end
/// equal, shadow nothing.
pub(super) fn label_word_shadows<'a, L: 'a>(
    mut left: impl Iterator<Item = &'a L>,
    mut right: impl Iterator<Item = &'a L>,
    mut compare: impl FnMut(ScopeGraphPathLabel<'a, L>, ScopeGraphPathLabel<'a, L>) -> Option<Ordering>,
) -> bool {
    let label = |edge: Option<&'a L>| {
        edge.map_or(ScopeGraphPathLabel::Declaration, ScopeGraphPathLabel::Edge)
    };
    loop {
        let (left_edge, right_edge) = (left.next(), right.next());
        match compare(label(left_edge), label(right_edge)) {
            Some(Ordering::Less) => return true,
            Some(Ordering::Greater) | None => return false,
            Some(Ordering::Equal) => {}
        }
        if left_edge.is_none() || right_edge.is_none() {
            return false;
        }
    }
}
