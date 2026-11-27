//! Path-free scope-graph queries resolved through memoized environments.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
use smallvec::SmallVec;

use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::open::{OpenFold, OpenFoldConfig, OpenFoldMemo, Revisit, open_fold_in};
use crate::graph::scc::{SccDecomposition, SccNumbering};
use crate::graph::search::{PathBudget, PathSearchConfig, PathSearchStats};
use crate::graph::view::{GraphView, assert_in_bound};

use super::index::ScopeResolutionIndex;
use super::query::{
    ScopeGraphPathLabel, ScopePath, ScopeQuery, ScopeResolution, ScopeResolutionCandidate,
};
use super::storage::{ScopeDatumId, ScopeEdgeId, ScopeGraph, ScopeId};

mod fold;
#[cfg(test)]
mod tests;

use fold::{EnvironmentFold, EnvironmentMark, Reach};

/// A scope-graph query whose hooks see neither the path nor the reference,
/// so the visible data of a scope can be computed once and shared.
///
/// A [`ScopeGraphQuery`](super::ScopeGraphQuery) hook sees the whole path
/// and the query input, so no part of one resolution can be reused for
/// another. This form gives each hook only the scope where the path ends and
/// the path-language state there. The *environment* of a scope in a state
/// (every datum visible from there, by key, with the path that reaches it)
/// then depends on nothing else, and a [`ScopeEnvironmentCache`] computes it
/// once for all the references that reach that scope in that state.
///
/// # Contract
///
/// - Equal states have equal futures: [`step`](Self::step) and
///   [`accepts`](Self::accepts) depend only on the graph, the scope, the
///   state and the edge or datum. A policy that needs the path (an
///   accessibility rule, a full path order) stays a `ScopeGraphQuery`, or
///   the caller applies it afterwards to the stored candidate paths.
/// - [`key`](Self::key) projects a datum to what shadowing and lookup
///   compare, typically its name and namespace. A datum matches a reference
///   when the query accepts it and its key equals the
///   [`query_key`](Self::query_key) of the reference's data.
/// - [`compare_labels`](Self::compare_labels) is a strict partial order on
///   the labels, with `Equal` for labels of the same specificity. Within one
///   key, a path shadows another when its label word is lexicographically
///   more specific, the order of
///   [`ScopeGraphQuery::shadows`](super::ScopeGraphQuery::shadows) by
///   default. Incomparable words keep both paths, so ambiguity is preserved.
///
/// Under this contract, on a graph without cycles, the cached resolution of
/// a reference sees the same data as
/// [`ScopeResolution::compute`](super::ScopeResolution::compute) under a
/// `ScopeGraphQuery` with the same hooks.
pub trait ScopeEnvironmentQuery<S, L, R, D, Q> {
    /// The path-language state; equal states have equal futures.
    type State: Clone + Ord;
    /// The projection of a datum that shadowing and lookup compare.
    type Key: Clone + Ord;

    /// The state at the start of every path.
    fn initial_state(&self, graph: &ScopeGraph<S, L, R, D, Q>) -> Self::State;

    /// Advances the state across `edge`, which leaves `scope`, or rejects the
    /// edge.
    fn step(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        scope: ScopeId,
        state: &Self::State,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, L>,
    ) -> Option<Self::State>;

    /// Returns whether `datum` of `scope` is reachable in `state`.
    fn accepts(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        scope: ScopeId,
        state: &Self::State,
        datum: ScopeDatumId,
    ) -> bool;

    /// The key of `datum`.
    fn key(&self, graph: &ScopeGraph<S, L, R, D, Q>, datum: ScopeDatumId) -> Self::Key;

    /// The key a reference with data `query` looks up.
    fn query_key(&self, graph: &ScopeGraph<S, L, R, D, Q>, query: &Q) -> Self::Key;

    /// Compares two extended labels for path specificity, as
    /// [`ScopeGraphQuery::compare_labels`](super::ScopeGraphQuery::compare_labels)
    /// does: `Less` means `left` is more specific.
    fn compare_labels(
        &self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        left: ScopeGraphPathLabel<'_, L>,
        right: ScopeGraphPathLabel<'_, L>,
    ) -> Option<Ordering>;
}

/// The environments of the scopes of one graph, computed once per scope,
/// state and remaining path length, for a [`ScopeEnvironmentQuery`].
///
/// [`index`](Self::index) fills the cache for all the references of a graph.
/// A kept cache serves later queries ([`resolve`](Self::resolve)), stored or
/// ephemeral, from the same environments. The environments are the answers
/// of a memoizing [`open_fold_in`] whose [`OpenFoldMemo`] the cache keeps:
/// the environment of a scope folds the scope's own accepted data and the
/// environments its admitted edges reach.
///
/// - **Cycles.** [`new`](Self::new) finds the strongly connected components
///   of the graph once. A scope on a cycle (in a component of several scopes,
///   or with an edge to itself) is resolved anew on every reach, under a guard
///   that refuses a path that comes back to a scope in a state it already had
///   on that path; its environment depends on the path, so it is not cached.
///   Every other environment is computed once. Paths computed for scopes on a
///   cycle stay in the cache until it is dropped.
/// - **Validity.** A cache belongs to one state of one graph, and one query
///   policy. After an edit of the graph, or with another policy, make a new
///   cache.
/// - **Results.** On a graph without cycles, under the contract of
///   [`ScopeEnvironmentQuery`], each query sees the data that
///   [`ScopeResolution::compute`] shows under a
///   [`ScopeGraphQuery`](super::ScopeGraphQuery) with the same hooks. On a
///   cycle the guard is (scope, state), not the scope alone, so a path
///   language that comes back to a scope in another state (`A A B` on
///   `s1 -A-> s2 -A-> s1 -B-> s3`) can reach data that `compute` cuts off.
/// - **Order.** The candidates of a query come in depth-first edge order: a
///   scope's own data first, then the candidates through each outgoing edge
///   in insertion order. This differs from the breadth-first discovery order
///   of `compute`.
/// - **Bounds.** `max_path_length` applies exactly: environments are cached
///   by the path length that is left. `max_path_count` bounds the
///   environments one query computes, and a cache hit computes none; an
///   environment that the bound cuts short is not cached. The cache reads
///   neither `order` nor `max_complete_path_count`. In the statistics,
///   `expanded_path_count` counts the environments a query computed, and
///   `complete_path_count` its candidates after shadowing.
#[derive(Debug, Clone)]
pub struct ScopeEnvironmentCache<St, K> {
    on_cycle: Vec<bool>,
    /// Shared, so a fold that reaches a stored environment lifts it without
    /// copying it.
    environments: OpenFoldMemo<EnvironmentMark<St>, Arc<Environment<K>>>,
    links: Vec<Link>,
}

/// The visible data of one scope in one state, by key.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Environment<K> {
    entries: BTreeMap<K, SmallVec<[Entry; 1]>>,
    /// Whether the length bound refused an extension below this scope.
    length_limited: bool,
}

/// One visible datum and the path that reaches it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
    datum: ScopeDatumId,
    /// The first link of the path, in the cache's link table.
    path: usize,
}

/// One hop of a stored path: the scope where the rest of the path starts,
/// and the edge out of it with the link at its target, or nothing at the
/// path's end. Environments share the rest of a path with the environment
/// they extend.
#[derive(Debug, Clone, Copy)]
struct Link {
    scope: ScopeId,
    hop: Option<(ScopeEdgeId, usize)>,
}

/// The memoizing walk of the environment fold. The length bound lives in
/// the reaches (see [`EnvironmentFold`]), so the walk has no depth bound.
const MEMOIZE: OpenFoldConfig = OpenFoldConfig {
    max_depth: None,
    revisit: Revisit::Memoize,
};

impl<St: Clone + Ord, K: Clone + Ord> ScopeEnvironmentCache<St, K> {
    /// Creates an empty cache for `graph` and finds its cycles.
    #[must_use]
    pub fn new<S, L, R, D, Q>(graph: &ScopeGraph<S, L, R, D, Q>) -> Self {
        let mut on_cycle = vec![false; graph.node_bound()];
        for component in SccDecomposition::compute(graph, SccNumbering::LeavesFirst).components() {
            let cyclic = component.len() > 1
                || component.iter().any(|&scope| {
                    graph
                        .outgoing(scope)
                        .any(|edge| graph.edge(edge).target() == scope)
                });
            if cyclic {
                for &scope in component {
                    on_cycle[scope.index()] = true;
                }
            }
        }
        Self {
            on_cycle,
            environments: OpenFoldMemo::default(),
            links: Vec::new(),
        }
    }

    /// Returns the number of environments the cache holds.
    #[cfg(test)]
    fn environment_count(&self) -> usize {
        self.environments.answer_count()
    }

    /// Resolves `input`, a stored reference or an ephemeral query on the
    /// cache's graph, as [`ScopeResolution::compute`] does for a
    /// [`ScopeGraphQuery`](super::ScopeGraphQuery).
    ///
    /// # Panics
    ///
    /// Panics when `input` starts in a scope that does not belong to `graph`.
    pub fn resolve<S, L, R, D, Q, P>(
        &mut self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: ScopeQuery<'_, Q>,
        query: &P,
        config: PathSearchConfig,
    ) -> ScopeResolution
    where
        P: ScopeEnvironmentQuery<S, L, R, D, Q, State = St, Key = K>,
    {
        assert_in_bound(graph, input.scope(), "query scope");
        let start = Reach::start(
            input.scope(),
            query.initial_state(graph),
            config.max_path_length,
        );
        let mut fold = EnvironmentFold {
            graph,
            query,
            on_cycle: &self.on_cycle,
            links: &mut self.links,
            budget: PathBudget::new(&config),
        };
        // A stored environment is read in place; only a missing one is
        // folded, and handed over by value.
        let mark = fold.mark(&start);
        let stored = mark
            .as_ref()
            .is_some_and(|mark| self.environments.get(mark).is_some());
        let computed = if stored {
            None
        } else {
            open_fold_in(&mut self.environments, &mut fold, start, MEMOIZE)
                .expect("the memoizing fold never fails")
        };
        let budget = fold.budget.stats();
        let environment = if stored {
            mark.as_ref()
                .and_then(|mark| self.environments.get(mark))
                .flatten()
                .map(|environment| &**environment)
        } else {
            computed.as_deref()
        };
        let key = query.query_key(graph, input.data());
        let entries = environment
            .and_then(|environment| environment.entries.get(&key))
            .map_or(&[][..], |entries| entries.as_slice());
        let candidates = entries
            .iter()
            .map(|entry| ScopeResolutionCandidate::new(entry.datum, self.path(entry.path)))
            .collect();
        let stats = PathSearchStats {
            complete_path_count: entries.len(),
            path_length_limited: environment.is_some_and(|environment| environment.length_limited),
            ..budget
        };
        ScopeResolution::from_visible(&input, candidates, stats)
    }

    /// Resolves every reference of `graph` in identity order through this
    /// cache and builds the reverse index.
    ///
    /// Each environment that no cycle reaches is computed once, so the
    /// references of one block, which climb the same parents and imports,
    /// share that work: filling the cache costs the sum of the environment
    /// sizes over the reachable (scope, state) pairs, and a reference then
    /// costs one cache lookup and the copy of its visible paths. The cache
    /// holds every key, so its memory is the sum of the environment sizes.
    ///
    /// The index records nothing of what its resolutions read, so it is not
    /// incremental: its first
    /// [`update`](ScopeResolutionIndex::update), which takes a
    /// [`ScopeGraphQuery`](super::ScopeGraphQuery), resolves every reference
    /// again, and can report a reference as changed whose candidates stayed
    /// the same, since the statistics of the two resolvers differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::cmp::Ordering;
    ///
    /// use cfglib::{
    ///     EdgeRef, PathSearchConfig, ScopeDatumId, ScopeEdgeId, ScopeEnvironmentCache,
    ///     ScopeEnvironmentQuery, ScopeGraph, ScopeGraphPathLabel, ScopeId,
    /// };
    ///
    /// #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    /// enum Label {
    ///     Import,
    ///     Parent,
    /// }
    ///
    /// type Names = ScopeGraph<(), Label, (), &'static str, &'static str>;
    ///
    /// /// Lexical lookup: parents, then at most one import; a declaration
    /// /// shadows an import, which shadows a parent.
    /// struct Lexical;
    ///
    /// impl ScopeEnvironmentQuery<(), Label, (), &'static str, &'static str> for Lexical {
    ///     /// Whether the path has taken its import.
    ///     type State = bool;
    ///     type Key = &'static str;
    ///
    ///     fn initial_state(&self, _: &Names) -> bool {
    ///         false
    ///     }
    ///     fn step(
    ///         &self,
    ///         _: &Names,
    ///         _: ScopeId,
    ///         imported: &bool,
    ///         edge: EdgeRef<'_, ScopeId, ScopeEdgeId, Label>,
    ///     ) -> Option<bool> {
    ///         match (imported, edge.data()) {
    ///             (false, Label::Parent) => Some(false),
    ///             (false, Label::Import) => Some(true),
    ///             (true, _) => None,
    ///         }
    ///     }
    ///     fn accepts(&self, _: &Names, _: ScopeId, _: &bool, _: ScopeDatumId) -> bool {
    ///         true
    ///     }
    ///     fn key(&self, graph: &Names, datum: ScopeDatumId) -> &'static str {
    ///         graph.datum(datum).data()
    ///     }
    ///     fn query_key(&self, _: &Names, name: &&'static str) -> &'static str {
    ///         name
    ///     }
    ///     fn compare_labels(
    ///         &self,
    ///         _: &Names,
    ///         left: ScopeGraphPathLabel<'_, Label>,
    ///         right: ScopeGraphPathLabel<'_, Label>,
    ///     ) -> Option<Ordering> {
    ///         let rank = |label| match label {
    ///             ScopeGraphPathLabel::Declaration => 0,
    ///             ScopeGraphPathLabel::Edge(Label::Import) => 1,
    ///             ScopeGraphPathLabel::Edge(Label::Parent) => 2,
    ///         };
    ///         Some(rank(left).cmp(&rank(right)))
    ///     }
    /// }
    ///
    /// let mut graph = Names::new();
    /// let module = graph.add_scope(());
    /// let library = graph.add_scope(());
    /// let function = graph.add_scope(());
    /// graph.add_edge(function, module, Label::Parent);
    /// graph.add_edge(module, library, Label::Import);
    /// let local = graph.add_datum(module, (), "x");
    /// let imported = graph.add_datum(library, (), "y");
    /// graph.add_datum(library, (), "x");
    /// // Both references climb the same parent and import; the second one
    /// // reuses the environments the first one computed.
    /// let x = graph.add_reference(function, "x");
    /// let y = graph.add_reference(function, "y");
    ///
    /// let mut cache = ScopeEnvironmentCache::new(&graph);
    /// let index = cache.index(&graph, &Lexical, PathSearchConfig::default());
    /// // The module's own `x` shadows the imported one.
    /// assert_eq!(index.resolution(x).candidates()[0].datum(), local);
    /// assert_eq!(index.resolution(x).candidates().len(), 1);
    /// assert_eq!(index.resolution(y).candidates()[0].datum(), imported);
    /// assert_eq!(index.references_to(local), [x]);
    /// ```
    #[must_use]
    pub fn index<S, L, R, D, Q, P>(
        &mut self,
        graph: &ScopeGraph<S, L, R, D, Q>,
        query: &P,
        config: PathSearchConfig,
    ) -> ScopeResolutionIndex
    where
        P: ScopeEnvironmentQuery<S, L, R, D, Q, State = St, Key = K>,
    {
        let resolutions = graph
            .reference_ids()
            .map(|reference| {
                self.resolve(
                    graph,
                    ScopeQuery::from_reference(graph, reference),
                    query,
                    config,
                )
            })
            .collect();
        ScopeResolutionIndex::from_resolutions(graph, resolutions)
    }

    /// The stored path that starts at `link`.
    fn path(&self, mut link: usize) -> ScopePath {
        let mut path = ScopePath::new(self.links[link].scope);
        while let Some((edge, next)) = self.links[link].hop {
            link = next;
            path.push(edge, self.links[link].scope);
        }
        path
    }
}
