extern crate alloc;

use alloc::format;
use alloc::vec::Vec;

use super::super::test_support::{Label, chain};
use crate::graph::edge_view::EdgeView;
use crate::graph::view::GraphView;
use crate::test_util::rng::Rng;
use crate::{
    EdgeRef, PathSearchConfig, ScopeCursor, ScopeCursorPolicy, ScopeCursorQuery, ScopeDatumId,
    ScopeEdgeId, ScopeFirstResolution, ScopeFirstResolutionError, ScopeGraph, ScopeId, ScopeQuery,
    ScopeResolution, ScopeResolutionCandidate,
};

/// A declaration: its name and whether an import can see it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Declaration {
    name: u8,
    exported: bool,
}

type Graph = ScopeGraph<(), Label, (), Declaration, u8>;

/// Lexical lookup that may cross one import, after which it sees only
/// exported declarations.
struct Lookup {
    stop_at_candidate: bool,
}

impl ScopeCursorQuery<(), Label, (), Declaration, u8> for Lookup {
    /// Whether the path crossed an import.
    type State = bool;

    fn initial_state(&self, _graph: &Graph, _input: &ScopeQuery<'_, u8>) -> bool {
        false
    }

    fn step(
        &self,
        _graph: &Graph,
        _input: &ScopeQuery<'_, u8>,
        _cursor: ScopeCursor,
        &imported: &bool,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, Label>,
    ) -> Option<bool> {
        match edge.data() {
            Label::Parent => Some(imported),
            Label::Import => (!imported).then_some(true),
            Label::Hidden | Label::A | Label::B => None,
        }
    }

    fn accepts(
        &self,
        graph: &Graph,
        input: &ScopeQuery<'_, u8>,
        _cursor: ScopeCursor,
        &imported: &bool,
        datum: ScopeDatumId,
    ) -> bool {
        let declaration = graph.datum(datum).data();
        declaration.name == *input.data() && (!imported || declaration.exported)
    }

    fn should_extend(
        &self,
        _graph: &Graph,
        _input: &ScopeQuery<'_, u8>,
        _cursor: ScopeCursor,
        _imported: &bool,
        accepted_candidate_count: usize,
    ) -> bool {
        !self.stop_at_candidate || accepted_candidate_count == 0
    }
}

const STOP: Lookup = Lookup {
    stop_at_candidate: true,
};
const EXTEND: Lookup = Lookup {
    stop_at_candidate: false,
};
const NAMES: u8 = 4;

/// A graph whose every edge leads to an earlier scope. Most scopes have one
/// parent; some have an import, a hidden edge, a second parent (a branch), or
/// two declarations of a name (an ambiguity).
fn random_upward_graph(random: &mut Rng) -> Graph {
    let scope_count = 1 + random.below(30);
    let mut graph = Graph::new();
    let scopes: Vec<ScopeId> = (0..scope_count).map(|_| graph.add_scope(())).collect();
    for (index, &scope) in scopes.iter().enumerate().skip(1) {
        if random.chance(90) {
            graph.add_edge(scope, scopes[random.below(index)], Label::Parent);
        }
        if random.chance(20) {
            graph.add_edge(scope, scopes[random.below(index)], Label::Import);
        }
        if random.chance(10) {
            graph.add_edge(scope, scopes[random.below(index)], Label::Hidden);
        }
        if random.chance(4) {
            graph.add_edge(scope, scopes[random.below(index)], Label::Parent);
        }
    }
    for &scope in &scopes {
        for _ in 0..random.below(3) {
            let name = u8::try_from(random.below(usize::from(NAMES))).expect("names fit in u8");
            let exported = random.chance(50);
            graph.add_datum(scope, (), Declaration { name, exported });
        }
    }
    graph
}

fn random_config(random: &mut Rng) -> PathSearchConfig {
    match random.below(4) {
        0 => PathSearchConfig {
            max_path_length: Some(random.below(5)),
            ..PathSearchConfig::default()
        },
        1 => PathSearchConfig {
            max_path_count: Some(1 + random.below(6)),
            ..PathSearchConfig::default()
        },
        _ => PathSearchConfig::default(),
    }
}

/// The datum and owning scope of the first candidate of a general search.
fn first_candidate(resolution: &ScopeResolution) -> Option<(ScopeDatumId, ScopeId)> {
    resolution
        .candidates()
        .first()
        .map(|candidate| (candidate.datum(), candidate.path().end()))
}

#[test]
fn first_resolution_is_the_general_search_of_a_policy_that_stops_at_a_candidate() {
    let mut random = Rng::new(0x5C0F_E29A);
    let (mut resolved, mut unresolved, mut failed) = (0, 0, 0);
    for _ in 0..300 {
        let graph = random_upward_graph(&mut random);
        for scope in graph.node_ids() {
            for name in 0..NAMES {
                let config = random_config(&mut random);
                let input = ScopeQuery::new(scope, &name);
                let general =
                    ScopeResolution::compute(&graph, input, &ScopeCursorPolicy(STOP), config);
                match ScopeFirstResolution::try_compute(&graph, input, &STOP, config) {
                    // The walk is the one path the general search expands.
                    Ok(first) => {
                        assert_eq!(first.candidate(), first_candidate(&general));
                        assert_eq!(first.stats(), general.stats());
                        if first.is_resolved() {
                            resolved += 1;
                        } else {
                            unresolved += 1;
                        }
                    }
                    // A cursor policy shadows nothing, so every datum of the
                    // scope stays a candidate of the general search.
                    Err(ScopeFirstResolutionError::MultipleCandidates { scope }) => {
                        assert!(general.candidates().len() >= 2, "{graph:?}");
                        assert!(
                            general
                                .candidates()
                                .iter()
                                .all(|candidate| candidate.path().end() == scope)
                        );
                        failed += 1;
                    }
                    // The general search branches there instead.
                    Err(ScopeFirstResolutionError::MultipleEdges { scope }) => {
                        let unhidden = graph
                            .outgoing(scope)
                            .filter(|&edge| *graph.edge(edge).data() != Label::Hidden)
                            .count();
                        assert!(unhidden >= 2, "{graph:?}");
                        failed += 1;
                    }
                    Err(error @ ScopeFirstResolutionError::EdgeNotUpward { .. }) => {
                        panic!("{error} in an upward graph");
                    }
                }
            }
        }
    }
    assert!(resolved > 1000 && unresolved > 1000 && failed > 100);
}

#[test]
fn every_other_candidate_of_an_extending_search_lies_past_the_first_datum() {
    let mut random = Rng::new(0x0F1E_57C0);
    let mut compared = 0;
    for _ in 0..300 {
        let graph = random_upward_graph(&mut random);
        for scope in graph.node_ids() {
            for name in 0..NAMES {
                let config = PathSearchConfig::default();
                let input = ScopeQuery::new(scope, &name);
                let Ok(first) = ScopeFirstResolution::try_compute(&graph, input, &EXTEND, config)
                else {
                    continue;
                };
                let general =
                    ScopeResolution::compute(&graph, input, &ScopeCursorPolicy(EXTEND), config);
                assert_eq!(first.candidate(), first_candidate(&general));
                // A label order in which a declaration is the most specific
                // label would let the first datum shadow every other one.
                if let [nearest, rest @ ..] = general.candidates() {
                    let prefix = nearest.path().scopes();
                    for candidate in rest {
                        let scopes = candidate.path().scopes();
                        assert!(scopes.len() > prefix.len() && scopes.starts_with(prefix));
                    }
                }
                compared += usize::from(first.is_resolved());
            }
        }
    }
    assert!(compared > 1000);
}

#[test]
fn first_resolution_reports_what_general_search_does_under_every_bound() {
    let (mut graph, scopes): (Graph, _) = chain(4);
    graph.add_datum(scopes[0], (), declaration(0));
    let input = ScopeQuery::new(scopes[3], &0);
    let limits = [None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(5)];
    let mut truncations = [false; 2];
    for max_path_length in limits {
        for max_path_count in limits {
            let config = PathSearchConfig {
                max_path_length,
                max_path_count,
                ..PathSearchConfig::default()
            };
            let first = ScopeFirstResolution::try_compute(&graph, input, &STOP, config)
                .expect("a chain is linear");
            let general = ScopeResolution::compute(&graph, input, &ScopeCursorPolicy(STOP), config);
            assert_eq!(first.candidate(), first_candidate(&general), "{config:?}");
            assert_eq!(first.stats(), general.stats(), "{config:?}");
            truncations[0] |= first.stats().path_length_limited;
            truncations[1] |= first.stats().path_count_limited;
        }
    }
    assert_eq!(truncations, [true; 2], "the sweep hits both bounds");
}

#[test]
fn a_cursor_policy_under_the_general_resolver_shadows_no_candidate() {
    let (mut graph, scopes): (Graph, _) = chain(3);
    let outer = graph.add_datum(scopes[0], (), declaration(0));
    let inner = graph.add_datum(scopes[1], (), declaration(0));
    let input = ScopeQuery::new(scopes[2], &0);
    let config = PathSearchConfig::default();

    let general = ScopeResolution::compute(&graph, input, &ScopeCursorPolicy(EXTEND), config);
    let data: Vec<_> = general
        .candidates()
        .iter()
        .map(ScopeResolutionCandidate::datum)
        .collect();
    assert_eq!(data, [inner, outer]);

    let first = ScopeFirstResolution::try_compute(&graph, input, &EXTEND, config)
        .expect("a chain is linear");
    assert_eq!(first.candidate(), Some((inner, scopes[1])));
}

const fn declaration(name: u8) -> Declaration {
    Declaration {
        name,
        exported: true,
    }
}

#[test]
fn stored_reference_resolves_from_its_scope() {
    let (mut graph, scopes): (Graph, _) = chain(3);
    let outer = graph.add_datum(scopes[0], (), declaration(1));
    graph.add_datum(scopes[2], (), declaration(2));
    let reference = graph.add_reference(scopes[2], 1);

    let first = ScopeFirstResolution::try_compute(
        &graph,
        ScopeQuery::from_reference(&graph, reference),
        &STOP,
        PathSearchConfig::default(),
    )
    .expect("a chain is linear");

    assert_eq!(first.candidate(), Some((outer, scopes[0])));
    assert_eq!(first.stats().expanded_path_count, 3);
    assert_eq!(first.stats().complete_path_count, 1);
    assert!(!first.stats().is_truncated());
}

#[test]
#[should_panic(expected = "query scope is out of range")]
fn a_query_that_starts_outside_the_graph_panics() {
    let (graph, _): (Graph, _) = chain(2);
    let outside = ScopeId::from_index(2);
    let _ = ScopeFirstResolution::try_compute(
        &graph,
        ScopeQuery::new(outside, &0),
        &STOP,
        PathSearchConfig::default(),
    );
}

#[test]
fn admitted_edge_that_does_not_point_upward_is_an_error() {
    let (mut graph, scopes): (Graph, _) = chain(2);
    let later = graph.add_edge(scopes[0], scopes[1], Label::Parent);
    let config = PathSearchConfig::default();

    let error =
        ScopeFirstResolution::try_compute(&graph, ScopeQuery::new(scopes[1], &0), &STOP, config)
            .expect_err("the walk reaches the downward edge");
    assert_eq!(
        error,
        ScopeFirstResolutionError::EdgeNotUpward { edge: later }
    );

    let mut graph = Graph::new();
    let scope = graph.add_scope(());
    let self_loop = graph.add_edge(scope, scope, Label::Import);
    let error =
        ScopeFirstResolution::try_compute(&graph, ScopeQuery::new(scope, &0), &STOP, config)
            .expect_err("an import self loop is admitted");
    assert_eq!(
        error,
        ScopeFirstResolutionError::EdgeNotUpward { edge: self_loop }
    );
}

#[test]
fn rejected_edge_may_point_anywhere() {
    let (mut graph, scopes): (Graph, _) = chain(2);
    graph.add_edge(scopes[0], scopes[1], Label::Hidden);
    graph.add_edge(scopes[1], scopes[1], Label::Hidden);

    let first = ScopeFirstResolution::try_compute(
        &graph,
        ScopeQuery::new(scopes[1], &0),
        &STOP,
        PathSearchConfig::default(),
    )
    .expect("hidden edges are never admitted");

    assert_eq!(first.candidate(), None);
    assert_eq!(first.stats().expanded_path_count, 2);
}

#[test]
fn bounds_stop_the_walk_and_report_truncation() {
    let (mut graph, scopes): (Graph, _) = chain(5);
    graph.add_datum(scopes[0], (), declaration(0));
    let input = ScopeQuery::new(scopes[4], &0);

    let length = ScopeFirstResolution::try_compute(
        &graph,
        input,
        &STOP,
        PathSearchConfig {
            max_path_length: Some(2),
            ..PathSearchConfig::default()
        },
    )
    .expect("a chain is linear");
    assert_eq!(length.candidate(), None);
    assert_eq!(length.stats().expanded_path_count, 3);
    assert!(length.stats().path_length_limited);

    let count = ScopeFirstResolution::try_compute(
        &graph,
        input,
        &STOP,
        PathSearchConfig {
            max_path_count: Some(2),
            ..PathSearchConfig::default()
        },
    )
    .expect("a chain is linear");
    assert_eq!(count.candidate(), None);
    assert_eq!(count.stats().expanded_path_count, 2);
    assert!(count.stats().path_count_limited);
}

#[test]
fn branches_and_ambiguities_are_errors() {
    let (mut graph, scopes): (Graph, _) = chain(3);
    graph.add_edge(scopes[2], scopes[0], Label::Parent);
    graph.add_datum(scopes[1], (), declaration(1));
    graph.add_datum(scopes[1], (), declaration(1));
    let config = PathSearchConfig::default();

    let branch =
        ScopeFirstResolution::try_compute(&graph, ScopeQuery::new(scopes[2], &1), &STOP, config)
            .expect_err("two parents are admitted");
    assert_eq!(
        branch,
        ScopeFirstResolutionError::MultipleEdges { scope: scopes[2] }
    );
    assert_eq!(
        format!("{branch}"),
        format!("scope {:?} admitted multiple outgoing edges", scopes[2])
    );

    let ambiguous =
        ScopeFirstResolution::try_compute(&graph, ScopeQuery::new(scopes[1], &1), &STOP, config)
            .expect_err("two declarations match");
    assert_eq!(
        ambiguous,
        ScopeFirstResolutionError::MultipleCandidates { scope: scopes[1] }
    );
}

#[test]
fn walk_stops_at_the_first_candidate_before_a_later_violation() {
    let (mut graph, scopes): (Graph, _) = chain(3);
    let near = graph.add_datum(scopes[1], (), declaration(1));
    graph.add_edge(scopes[1], scopes[2], Label::Parent);

    let first = ScopeFirstResolution::try_compute(
        &graph,
        ScopeQuery::new(scopes[2], &1),
        &EXTEND,
        PathSearchConfig::default(),
    )
    .expect("the walk stops before the downward edge");

    assert_eq!(first.candidate(), Some((near, scopes[1])));
}

#[test]
fn deep_walk_visits_each_scope_once() {
    let depth = 1500;
    let (mut graph, scopes): (Graph, _) = chain(depth);
    for (index, &scope) in scopes.iter().enumerate() {
        let name = if index == 0 {
            0
        } else {
            1 + u8::from(index % 2 == 0)
        };
        graph.add_datum(scope, (), declaration(name));
    }

    let first = ScopeFirstResolution::try_compute(
        &graph,
        ScopeQuery::new(scopes[depth - 1], &0),
        &STOP,
        PathSearchConfig::default(),
    )
    .expect("a chain is linear");

    assert_eq!(first.candidate().map(|(_, scope)| scope), Some(scopes[0]));
    assert_eq!(first.stats().expanded_path_count, depth);
}
