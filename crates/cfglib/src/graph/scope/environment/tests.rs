extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::cmp::Ordering;

use super::super::test_support::{Label, chain, random_graph, rank};
use super::{ScopeEnvironmentCache, ScopeEnvironmentQuery};
use crate::PathSearchConfig;
use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::scope::{
    ScopeDatumId, ScopeEdgeId, ScopeGraph, ScopeGraphPathLabel, ScopeGraphQuery, ScopeId,
    ScopePath, ScopeQuery, ScopeResolution, ScopeResolutionCandidate, ScopeResolutionIndex,
};
use crate::test_util::rng::Rng;

/// Scopes without payload, data and references named by a small number.
type Graph = ScopeGraph<(), Label, (), u8, u8>;

/// Every edge, in one state: the path language is `(P | I)*`.
struct Anywhere;

impl ScopeEnvironmentQuery<(), Label, (), u8, u8> for Anywhere {
    type State = ();
    type Key = u8;

    fn initial_state(&self, _graph: &Graph) {}

    fn step(
        &self,
        _graph: &Graph,
        _scope: ScopeId,
        (): &(),
        _edge: EdgeRef<'_, ScopeId, ScopeEdgeId, Label>,
    ) -> Option<()> {
        Some(())
    }

    fn accepts(&self, _graph: &Graph, _scope: ScopeId, (): &(), _datum: ScopeDatumId) -> bool {
        true
    }

    fn key(&self, graph: &Graph, datum: ScopeDatumId) -> u8 {
        *graph.datum(datum).data()
    }

    fn query_key(&self, _graph: &Graph, query: &u8) -> u8 {
        *query
    }

    fn compare_labels(
        &self,
        _graph: &Graph,
        left: ScopeGraphPathLabel<'_, Label>,
        right: ScopeGraphPathLabel<'_, Label>,
    ) -> Option<Ordering> {
        Some(rank(left).cmp(&rank(right)))
    }
}

/// Parents, then at most one import: the path language is `P* I?`.
struct ParentsThenImport;

impl ScopeEnvironmentQuery<(), Label, (), u8, u8> for ParentsThenImport {
    /// Whether the path has taken its import.
    type State = bool;
    type Key = u8;

    fn initial_state(&self, _graph: &Graph) -> bool {
        false
    }

    fn step(
        &self,
        _graph: &Graph,
        _scope: ScopeId,
        imported: &bool,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, Label>,
    ) -> Option<bool> {
        match (imported, edge.data()) {
            (false, Label::Parent) => Some(false),
            (false, Label::Import) => Some(true),
            _ => None,
        }
    }

    fn accepts(&self, _graph: &Graph, _scope: ScopeId, _imported: &bool, _: ScopeDatumId) -> bool {
        true
    }

    fn key(&self, graph: &Graph, datum: ScopeDatumId) -> u8 {
        *graph.datum(datum).data()
    }

    fn query_key(&self, _graph: &Graph, query: &u8) -> u8 {
        *query
    }

    fn compare_labels(
        &self,
        _graph: &Graph,
        left: ScopeGraphPathLabel<'_, Label>,
        right: ScopeGraphPathLabel<'_, Label>,
    ) -> Option<Ordering> {
        Some(rank(left).cmp(&rank(right)))
    }
}

/// The path language `A A B`, data accepted only at its end.
struct ThenB;

impl ScopeEnvironmentQuery<(), Label, (), u8, u8> for ThenB {
    /// How much of `A A B` the path has read.
    type State = u8;
    type Key = u8;

    fn initial_state(&self, _graph: &Graph) -> u8 {
        0
    }

    fn step(
        &self,
        _graph: &Graph,
        _scope: ScopeId,
        read: &u8,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, Label>,
    ) -> Option<u8> {
        match (read, edge.data()) {
            (0 | 1, Label::A) | (2, Label::B) => Some(read + 1),
            _ => None,
        }
    }

    fn accepts(&self, _graph: &Graph, _scope: ScopeId, read: &u8, _datum: ScopeDatumId) -> bool {
        *read == 3
    }

    fn key(&self, graph: &Graph, datum: ScopeDatumId) -> u8 {
        *graph.datum(datum).data()
    }

    fn query_key(&self, _graph: &Graph, query: &u8) -> u8 {
        *query
    }

    fn compare_labels(
        &self,
        _graph: &Graph,
        left: ScopeGraphPathLabel<'_, Label>,
        right: ScopeGraphPathLabel<'_, Label>,
    ) -> Option<Ordering> {
        Some(rank(left).cmp(&rank(right)))
    }
}

/// An environment policy run as a path-aware query: the hooks of the policy,
/// with the path's end for the scope and a key match for the reference.
struct PathAware<'a, P>(&'a P);

impl<P> ScopeGraphQuery<(), Label, (), u8, u8> for PathAware<'_, P>
where
    P: ScopeEnvironmentQuery<(), Label, (), u8, u8>,
{
    type State = P::State;

    fn initial_state(&self, graph: &Graph, _input: &ScopeQuery<'_, u8>) -> P::State {
        self.0.initial_state(graph)
    }

    fn step(
        &self,
        graph: &Graph,
        _input: &ScopeQuery<'_, u8>,
        path: &ScopePath,
        state: &P::State,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, Label>,
    ) -> Option<P::State> {
        self.0.step(graph, path.end(), state, edge)
    }

    fn accepts(
        &self,
        graph: &Graph,
        input: &ScopeQuery<'_, u8>,
        path: &ScopePath,
        state: &P::State,
        datum: ScopeDatumId,
    ) -> bool {
        self.0.key(graph, datum) == self.0.query_key(graph, input.data())
            && self.0.accepts(graph, path.end(), state, datum)
    }

    fn compare_labels(
        &self,
        graph: &Graph,
        _input: &ScopeQuery<'_, u8>,
        left: ScopeGraphPathLabel<'_, Label>,
        right: ScopeGraphPathLabel<'_, Label>,
    ) -> Option<Ordering> {
        self.0.compare_labels(graph, left, right)
    }
}

/// The visible candidates of a resolution, as an ordered set.
fn candidates(resolution: &ScopeResolution) -> BTreeSet<(ScopeDatumId, Vec<ScopeEdgeId>)> {
    resolution
        .candidates()
        .iter()
        .map(|candidate| (candidate.datum(), candidate.path().edges().to_vec()))
        .collect()
}

/// Checks that every candidate path of `resolution` is a path of `graph`
/// from the query's scope to the scope that owns the datum.
fn assert_paths_are_walks(graph: &Graph, resolution: &ScopeResolution) {
    for candidate in resolution.candidates() {
        let path = candidate.path();
        assert_eq!(path.start(), resolution.start_scope());
        assert_eq!(path.end(), graph.datum(candidate.datum()).scope());
        for (hop, &edge) in path.edges().iter().enumerate() {
            assert_eq!(graph.edge(edge).source(), path.scopes()[hop]);
            assert_eq!(graph.edge(edge).target(), path.scopes()[hop + 1]);
        }
    }
}

/// Checks that the cached index gives every reference the candidates of the
/// general one. With `same_cycles`, the policy has one state, so both
/// resolvers refuse the same returns to a scope and must also agree on
/// whether the length bound refused an extension; with several states the
/// cached resolver admits a return in another state, which a length bound
/// can then refuse.
fn assert_indexes_agree<P>(graph: &Graph, policy: &P, config: PathSearchConfig, same_cycles: bool)
where
    P: ScopeEnvironmentQuery<(), Label, (), u8, u8>,
{
    let general = ScopeResolutionIndex::compute(graph, &PathAware(policy), config);
    let cached = ScopeEnvironmentCache::new(graph).index(graph, policy, config);
    for reference in graph.reference_ids() {
        let expected = general.resolution(reference);
        let actual = cached.resolution(reference);
        assert_eq!(
            candidates(actual),
            candidates(expected),
            "{graph:?} {reference:?} {config:?}"
        );
        if same_cycles {
            assert_eq!(
                actual.stats().path_length_limited,
                expected.stats().path_length_limited,
                "{graph:?} {reference:?} {config:?}"
            );
        }
        assert_paths_are_walks(graph, actual);
    }
    for datum in (0..graph.datum_count()).map(ScopeDatumId::from_index) {
        assert_eq!(cached.references_to(datum), general.references_to(datum));
    }
}

#[test]
fn cached_and_general_indexes_agree_on_random_graphs() {
    let mut random = Rng::new(0x74_5EED);
    for _ in 0..1_000 {
        // Up to nine scopes. Most edges go to an earlier scope, which keeps
        // a region without cycles; one in four goes anywhere and can close a
        // cycle.
        let graph = random_graph(&mut random, 9, Some(4), |_| ());
        for config in [
            PathSearchConfig::default(),
            PathSearchConfig {
                max_path_length: Some(random.below(4)),
                ..PathSearchConfig::default()
            },
        ] {
            assert_indexes_agree(&graph, &Anywhere, config, true);
            assert_indexes_agree(&graph, &ParentsThenImport, config, false);
        }
    }
}

#[test]
fn a_path_language_may_come_back_to_a_scope_in_another_state() {
    // `A A B` on s1 -A-> s2 -A-> s1 -B-> s3. The general resolver refuses
    // every second visit of s1; the cached one refuses a second visit of s1
    // in the same state only, so it reads `A A B` and finds the datum in s3.
    let mut graph = Graph::new();
    let s1 = graph.add_scope(());
    let s2 = graph.add_scope(());
    let s3 = graph.add_scope(());
    let a1 = graph.add_edge(s1, s2, Label::A);
    let a2 = graph.add_edge(s2, s1, Label::A);
    let b = graph.add_edge(s1, s3, Label::B);
    let datum = graph.add_datum(s3, (), 0);
    let reference = graph.add_reference(s1, 0);

    let general =
        ScopeResolutionIndex::compute(&graph, &PathAware(&ThenB), PathSearchConfig::default());
    assert!(!general.resolution(reference).is_resolved());

    let cached =
        ScopeEnvironmentCache::new(&graph).index(&graph, &ThenB, PathSearchConfig::default());
    let resolution = cached.resolution(reference);
    assert_eq!(resolution.candidates().len(), 1);
    assert_eq!(resolution.candidates()[0].datum(), datum);
    assert_eq!(resolution.candidates()[0].path().edges(), [a1, a2, b]);
    assert_eq!(resolution.candidates()[0].path().scopes(), [s1, s2, s1, s3]);
}

#[test]
fn equal_labels_keep_both_candidates() {
    // The block imports two modules that both declare `0`: neither import
    // is more specific, so the reference is ambiguous.
    let mut graph = Graph::new();
    let block = graph.add_scope(());
    let left = graph.add_scope(());
    let right = graph.add_scope(());
    graph.add_edge(block, left, Label::Import);
    graph.add_edge(block, right, Label::Import);
    let in_left = graph.add_datum(left, (), 0);
    let in_right = graph.add_datum(right, (), 0);
    let reference = graph.add_reference(block, 0);
    let index =
        ScopeEnvironmentCache::new(&graph).index(&graph, &Anywhere, PathSearchConfig::default());
    let resolution = index.resolution(reference);
    assert!(resolution.is_ambiguous());
    let data: Vec<_> = resolution
        .candidates()
        .iter()
        .map(ScopeResolutionCandidate::datum)
        .collect();
    assert_eq!(data, [in_left, in_right], "edge order");
}

/// A chain of `count` scopes, each with a parent edge to the previous one,
/// the first declaring `0`.
fn declaring_chain(count: usize) -> (Graph, Vec<ScopeId>) {
    let (mut graph, scopes): (Graph, _) = chain(count);
    graph.add_datum(scopes[0], (), 0);
    (graph, scopes)
}

#[test]
fn references_share_the_environments_of_their_scopes() {
    let (mut graph, scopes) = declaring_chain(4);
    let inner = scopes[3];
    let first = graph.add_reference(inner, 0);
    let second = graph.add_reference(inner, 1);
    let mut cache = ScopeEnvironmentCache::new(&graph);
    let config = PathSearchConfig::default();

    let first = ScopeQuery::from_reference(&graph, first);
    let resolution = cache.resolve(&graph, first, &Anywhere, config);
    assert!(resolution.is_resolved());
    assert_eq!(resolution.stats().expanded_path_count, 4);
    assert_eq!(cache.environment_count(), 4);

    // The second reference starts where the first did: one lookup.
    let second = ScopeQuery::from_reference(&graph, second);
    let resolution = cache.resolve(&graph, second, &Anywhere, config);
    assert!(!resolution.is_resolved());
    assert_eq!(resolution.stats().expanded_path_count, 0);
    assert_eq!(cache.environment_count(), 4);

    // An ephemeral query from a scope on the way is a lookup too, and it
    // sees what the general resolver sees.
    let ephemeral = cache.resolve(&graph, ScopeQuery::new(scopes[2], &0), &Anywhere, config);
    assert_eq!(ephemeral.stats().expanded_path_count, 0);
    let general = ScopeResolution::compute(
        &graph,
        ScopeQuery::new(scopes[2], &0),
        &PathAware(&Anywhere),
        config,
    );
    assert_eq!(candidates(&ephemeral), candidates(&general));
}

#[test]
fn an_environment_a_count_bound_cut_short_is_not_cached() {
    let (mut graph, scopes) = declaring_chain(3);
    let reference = graph.add_reference(scopes[2], 0);
    let reference = ScopeQuery::from_reference(&graph, reference);
    let mut cache = ScopeEnvironmentCache::new(&graph);

    let cut = cache.resolve(
        &graph,
        reference,
        &Anywhere,
        PathSearchConfig {
            max_path_count: Some(1),
            ..PathSearchConfig::default()
        },
    );
    assert!(!cut.is_resolved());
    assert!(cut.stats().path_count_limited);
    assert_eq!(cache.environment_count(), 0);

    let full = cache.resolve(&graph, reference, &Anywhere, PathSearchConfig::default());
    assert!(full.is_resolved());
    assert!(!full.stats().is_truncated());
}

#[test]
fn scopes_on_a_cycle_are_not_cached() {
    // 0 and 1 import each other; 2 has a parent edge to 0 and is off the
    // cycle, so only its environment is cached.
    let mut graph = Graph::new();
    let first = graph.add_scope(());
    let second = graph.add_scope(());
    let outside = graph.add_scope(());
    graph.add_edge(first, second, Label::Import);
    graph.add_edge(second, first, Label::Import);
    graph.add_edge(outside, first, Label::Parent);
    let datum = graph.add_datum(second, (), 0);
    let reference = graph.add_reference(outside, 0);
    let reference = ScopeQuery::from_reference(&graph, reference);
    let mut cache = ScopeEnvironmentCache::new(&graph);
    let resolution = cache.resolve(&graph, reference, &Anywhere, PathSearchConfig::default());
    assert_eq!(resolution.candidates()[0].datum(), datum);
    assert_eq!(cache.environment_count(), 1);
}
