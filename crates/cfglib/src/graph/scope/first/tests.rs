extern crate alloc;

use alloc::format;
use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::test_util::rng::Rng;
use crate::{
    EdgeRef, ScopeCursor, ScopeCursorQuery, ScopeDatumId, ScopeEdgeId, ScopeFirstResolution,
    ScopeFirstResolutionError, ScopeGraph, ScopeGraphPathLabel, ScopeGraphQuery, ScopeId,
    ScopeLinearResolutionError, ScopePath, ScopeQuery, ScopeResolution, ScopeResolutionConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Label {
    Parent,
    Import,
    Hidden,
}

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
            Label::Hidden => None,
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

    fn compare_labels(
        &self,
        _graph: &Graph,
        _input: &ScopeQuery<'_, u8>,
        left: ScopeGraphPathLabel<'_, Label>,
        right: ScopeGraphPathLabel<'_, Label>,
    ) -> Option<Ordering> {
        fn rank(label: ScopeGraphPathLabel<'_, Label>) -> u8 {
            match label {
                ScopeGraphPathLabel::Declaration => 0,
                ScopeGraphPathLabel::Edge(Label::Parent) => 1,
                ScopeGraphPathLabel::Edge(Label::Import) => 2,
                ScopeGraphPathLabel::Edge(Label::Hidden) => 3,
            }
        }
        Some(rank(left).cmp(&rank(right)))
    }
}

const STOP: Lookup = Lookup {
    stop_at_candidate: true,
};
const EXTEND: Lookup = Lookup {
    stop_at_candidate: false,
};
const NAMES: u8 = 4;

/// [`Lookup`] as a [`ScopeGraphQuery`]: each hook gives the cursor of the
/// stored path to the cursor hook, so linear resolution runs the same policy.
struct Linear<'a>(&'a Lookup);

impl ScopeGraphQuery<(), Label, (), Declaration, u8> for Linear<'_> {
    type State = bool;

    fn initial_state(&self, graph: &Graph, input: &ScopeQuery<'_, u8>) -> bool {
        ScopeCursorQuery::initial_state(self.0, graph, input)
    }

    fn step(
        &self,
        graph: &Graph,
        input: &ScopeQuery<'_, u8>,
        path: &ScopePath,
        state: &bool,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, Label>,
    ) -> Option<bool> {
        ScopeCursorQuery::step(
            self.0,
            graph,
            input,
            ScopeCursor::of_path(path),
            state,
            edge,
        )
    }

    fn accepts(
        &self,
        graph: &Graph,
        input: &ScopeQuery<'_, u8>,
        path: &ScopePath,
        state: &bool,
        datum: ScopeDatumId,
    ) -> bool {
        ScopeCursorQuery::accepts(
            self.0,
            graph,
            input,
            ScopeCursor::of_path(path),
            state,
            datum,
        )
    }

    fn should_extend(
        &self,
        graph: &Graph,
        input: &ScopeQuery<'_, u8>,
        path: &ScopePath,
        state: &bool,
        accepted_candidate_count: usize,
    ) -> bool {
        ScopeCursorQuery::should_extend(
            self.0,
            graph,
            input,
            ScopeCursor::of_path(path),
            state,
            accepted_candidate_count,
        )
    }

    fn compare_labels(
        &self,
        graph: &Graph,
        input: &ScopeQuery<'_, u8>,
        left: ScopeGraphPathLabel<'_, Label>,
        right: ScopeGraphPathLabel<'_, Label>,
    ) -> Option<Ordering> {
        ScopeCursorQuery::compare_labels(self.0, graph, input, left, right)
    }
}

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

fn random_config(random: &mut Rng) -> ScopeResolutionConfig {
    match random.below(4) {
        0 => ScopeResolutionConfig::new().with_max_path_length(random.below(5)),
        1 => ScopeResolutionConfig::new().with_max_path_count(1 + random.below(6)),
        _ => ScopeResolutionConfig::new(),
    }
}

fn same_error(linear: ScopeLinearResolutionError, first: ScopeFirstResolutionError) -> bool {
    match (linear, first) {
        (
            ScopeLinearResolutionError::MultipleCandidates { scope: linear },
            ScopeFirstResolutionError::MultipleCandidates { scope: first },
        )
        | (
            ScopeLinearResolutionError::MultipleEdges { scope: linear },
            ScopeFirstResolutionError::MultipleEdges { scope: first },
        ) => linear == first,
        _ => false,
    }
}

/// The datum and owning scope of the first visible linear candidate.
fn first_linear_candidate(resolution: &ScopeResolution) -> Option<(ScopeDatumId, ScopeId)> {
    resolution
        .candidates()
        .first()
        .map(|candidate| (candidate.datum(), candidate.path().end()))
}

#[test]
fn first_resolution_equals_linear_resolution_on_random_upward_graphs() {
    let mut random = Rng::new(0x5C0F_E29A);
    let (mut resolved, mut unresolved, mut failed) = (0, 0, 0);
    for _ in 0..300 {
        let graph = random_upward_graph(&mut random);
        for scope in graph.scope_ids() {
            for name in 0..NAMES {
                let config = random_config(&mut random);
                let linear = ScopeResolution::try_compute_linear_from(
                    &graph,
                    scope,
                    &name,
                    &Linear(&STOP),
                    config,
                );
                let first =
                    ScopeFirstResolution::try_compute_from(&graph, scope, &name, &STOP, config);
                match (linear, first) {
                    (Ok(linear), Ok(first)) => {
                        assert_eq!(first.candidate(), first_linear_candidate(&linear));
                        assert_eq!(first.stats(), linear.stats());
                        if first.is_resolved() {
                            resolved += 1;
                        } else {
                            unresolved += 1;
                        }
                    }
                    (Err(linear), Err(first)) => {
                        assert!(same_error(linear, first), "{linear:?} != {first:?}");
                        failed += 1;
                    }
                    (linear, first) => panic!("{linear:?} disagrees with {first:?}"),
                }
            }
        }
    }
    assert!(resolved > 1000 && unresolved > 1000 && failed > 100);
}

#[test]
fn first_resolution_is_the_visible_candidate_when_declarations_shadow_edges() {
    let mut random = Rng::new(0x0F1E_57C0);
    let mut compared = 0;
    for _ in 0..300 {
        let graph = random_upward_graph(&mut random);
        for scope in graph.scope_ids() {
            for name in 0..NAMES {
                let config = ScopeResolutionConfig::new();
                let Ok(linear) = ScopeResolution::try_compute_linear_from(
                    &graph,
                    scope,
                    &name,
                    &Linear(&EXTEND),
                    config,
                ) else {
                    continue;
                };
                let first =
                    ScopeFirstResolution::try_compute_from(&graph, scope, &name, &EXTEND, config)
                        .expect("a prefix of a valid linear walk is valid");
                assert!(linear.candidates().len() <= 1);
                assert_eq!(first.candidate(), first_linear_candidate(&linear));
                compared += usize::from(first.is_resolved());
            }
        }
    }
    assert!(compared > 1000);
}

fn chain(depth: usize) -> (Graph, Vec<ScopeId>) {
    let mut graph = Graph::new();
    let scopes: Vec<ScopeId> = (0..depth).map(|_| graph.add_scope(())).collect();
    for pair in scopes.windows(2) {
        graph.add_edge(pair[1], pair[0], Label::Parent);
    }
    (graph, scopes)
}

const fn declaration(name: u8) -> Declaration {
    Declaration {
        name,
        exported: true,
    }
}

#[test]
fn stored_reference_resolves_from_its_scope() {
    let (mut graph, scopes) = chain(3);
    let outer = graph.add_datum(scopes[0], (), declaration(1));
    graph.add_datum(scopes[2], (), declaration(2));
    let reference = graph.add_reference(scopes[2], 1);

    let first =
        ScopeFirstResolution::try_compute(&graph, reference, &STOP, ScopeResolutionConfig::new())
            .expect("a chain is linear");

    assert_eq!(first.candidate(), Some((outer, scopes[0])));
    assert_eq!(first.stats().explored_path_count, 3);
    assert_eq!(first.stats().reachable_candidate_count, 1);
    assert!(!first.stats().is_truncated());
}

#[test]
fn admitted_edge_that_does_not_point_upward_is_an_error() {
    let (mut graph, scopes) = chain(2);
    let later = graph.add_edge(scopes[0], scopes[1], Label::Parent);
    let config = ScopeResolutionConfig::new();

    let error = ScopeFirstResolution::try_compute_from(&graph, scopes[1], &0, &STOP, config)
        .expect_err("the walk reaches the downward edge");
    assert_eq!(
        error,
        ScopeFirstResolutionError::EdgeNotUpward { edge: later }
    );

    let mut graph = Graph::new();
    let scope = graph.add_scope(());
    let self_loop = graph.add_edge(scope, scope, Label::Import);
    let error = ScopeFirstResolution::try_compute_from(&graph, scope, &0, &STOP, config)
        .expect_err("an import self loop is admitted");
    assert_eq!(
        error,
        ScopeFirstResolutionError::EdgeNotUpward { edge: self_loop }
    );
}

#[test]
fn rejected_edge_may_point_anywhere() {
    let (mut graph, scopes) = chain(2);
    graph.add_edge(scopes[0], scopes[1], Label::Hidden);
    graph.add_edge(scopes[1], scopes[1], Label::Hidden);

    let first = ScopeFirstResolution::try_compute_from(
        &graph,
        scopes[1],
        &0,
        &STOP,
        ScopeResolutionConfig::new(),
    )
    .expect("hidden edges are never admitted");

    assert_eq!(first.candidate(), None);
    assert_eq!(first.stats().explored_path_count, 2);
}

#[test]
fn bounds_stop_the_walk_and_report_truncation() {
    let (mut graph, scopes) = chain(5);
    graph.add_datum(scopes[0], (), declaration(0));

    let length = ScopeFirstResolution::try_compute_from(
        &graph,
        scopes[4],
        &0,
        &STOP,
        ScopeResolutionConfig::new().with_max_path_length(2),
    )
    .expect("a chain is linear");
    assert_eq!(length.candidate(), None);
    assert_eq!(length.stats().explored_path_count, 3);
    assert!(length.stats().path_length_limited);

    let count = ScopeFirstResolution::try_compute_from(
        &graph,
        scopes[4],
        &0,
        &STOP,
        ScopeResolutionConfig::new().with_max_path_count(2),
    )
    .expect("a chain is linear");
    assert_eq!(count.candidate(), None);
    assert_eq!(count.stats().explored_path_count, 2);
    assert!(count.stats().path_count_limited);
}

#[test]
fn branches_and_ambiguities_are_errors() {
    let (mut graph, scopes) = chain(3);
    graph.add_edge(scopes[2], scopes[0], Label::Parent);
    graph.add_datum(scopes[1], (), declaration(1));
    graph.add_datum(scopes[1], (), declaration(1));
    let config = ScopeResolutionConfig::new();

    let branch = ScopeFirstResolution::try_compute_from(&graph, scopes[2], &1, &STOP, config)
        .expect_err("two parents are admitted");
    assert_eq!(
        branch,
        ScopeFirstResolutionError::MultipleEdges { scope: scopes[2] }
    );
    assert_eq!(
        format!("{branch}"),
        format!("scope {:?} admitted multiple outgoing edges", scopes[2])
    );

    let ambiguous = ScopeFirstResolution::try_compute_from(&graph, scopes[1], &1, &STOP, config)
        .expect_err("two declarations match");
    assert_eq!(
        ambiguous,
        ScopeFirstResolutionError::MultipleCandidates { scope: scopes[1] }
    );
}

#[test]
fn walk_stops_at_the_first_candidate_before_a_later_violation() {
    let (mut graph, scopes) = chain(3);
    let near = graph.add_datum(scopes[1], (), declaration(1));
    graph.add_edge(scopes[1], scopes[2], Label::Parent);

    let first = ScopeFirstResolution::try_compute_from(
        &graph,
        scopes[2],
        &1,
        &EXTEND,
        ScopeResolutionConfig::new(),
    )
    .expect("the walk stops before the downward edge");

    assert_eq!(first.candidate(), Some((near, scopes[1])));
}

#[test]
fn deep_walk_visits_each_scope_once() {
    let depth = 1500;
    let (mut graph, scopes) = chain(depth);
    for (index, &scope) in scopes.iter().enumerate() {
        let name = if index == 0 {
            0
        } else {
            1 + u8::from(index % 2 == 0)
        };
        graph.add_datum(scope, (), declaration(name));
    }

    let first = ScopeFirstResolution::try_compute_from(
        &graph,
        scopes[depth - 1],
        &0,
        &STOP,
        ScopeResolutionConfig::new(),
    )
    .expect("a chain is linear");

    assert_eq!(first.candidate().map(|(_, scope)| scope), Some(scopes[0]));
    assert_eq!(first.stats().explored_path_count, depth);
}
