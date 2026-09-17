extern crate alloc;

#[cfg(feature = "serde")]
use alloc::string::String;
use alloc::vec;
use core::cmp::Ordering;

use crate::{TraversalDirection, shortest_path};

use super::{
    ScopeDatumId, ScopeEdgeId, ScopeGraph, ScopeGraphPathLabel, ScopeGraphQuery,
    ScopeLinearResolutionError, ScopePath, ScopeQuery, ScopeResolution, ScopeResolutionConfig,
    ScopeResolutionIndex,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Label {
    Parent,
    Import,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Relation {
    Value,
    Type,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScopeMeta {
    ordered: bool,
    opaque_to_nested: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Declaration {
    name: &'static str,
    end: u32,
    visible_to_nested: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Reference {
    name: &'static str,
    relation: Relation,
    start: u32,
    name_match: NameMatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NameMatch {
    Exact,
    Prefix,
}

type Graph = ScopeGraph<ScopeMeta, Label, Relation, Declaration, Reference>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryState {
    Lexical,
    Imported,
}

struct LexicalQuery;

impl ScopeGraphQuery<ScopeMeta, Label, Relation, Declaration, Reference> for LexicalQuery {
    type State = QueryState;

    fn initial_state(&self, _graph: &Graph, _input: &ScopeQuery<'_, Reference>) -> Self::State {
        QueryState::Lexical
    }

    fn step(
        &self,
        _graph: &Graph,
        _input: &ScopeQuery<'_, Reference>,
        _path: &ScopePath,
        state: &Self::State,
        edge: crate::EdgeRef<'_, super::ScopeId, ScopeEdgeId, Label>,
    ) -> Option<Self::State> {
        match (state, edge.data()) {
            (QueryState::Lexical, Label::Parent) => Some(QueryState::Lexical),
            (QueryState::Lexical | QueryState::Imported, Label::Import) => {
                Some(QueryState::Imported)
            }
            (QueryState::Imported, Label::Parent) => None,
        }
    }

    fn accepts(
        &self,
        graph: &Graph,
        input: &ScopeQuery<'_, Reference>,
        path: &ScopePath,
        _state: &Self::State,
        datum: ScopeDatumId,
    ) -> bool {
        let reference = input.data();
        let candidate = graph.datum(datum);
        let declaration = candidate.data();
        let name_matches = match reference.name_match {
            NameMatch::Exact => declaration.name == reference.name,
            NameMatch::Prefix => declaration.name.starts_with(reference.name),
        };
        if candidate.relation() != &reference.relation || !name_matches {
            return false;
        }

        let scope = graph.scope(path.end()).payload();
        if scope.opaque_to_nested && path.start() != path.end() && !declaration.visible_to_nested {
            return false;
        }
        if !scope.ordered {
            return true;
        }

        graph
            .scope_data(path.end())
            .iter()
            .copied()
            .filter(|&candidate_id| {
                let candidate = graph.datum(candidate_id);
                candidate.relation() == &reference.relation
                    && candidate.data().name == declaration.name
                    && candidate.data().end <= reference.start
                    && (!scope.opaque_to_nested
                        || path.start() == path.end()
                        || candidate.data().visible_to_nested)
            })
            .max_by_key(|candidate_id| (graph.datum(*candidate_id).data().end, *candidate_id))
            == Some(datum)
    }

    fn compare_labels(
        &self,
        _graph: &Graph,
        _input: &ScopeQuery<'_, Reference>,
        left: ScopeGraphPathLabel<'_, Label>,
        right: ScopeGraphPathLabel<'_, Label>,
    ) -> Option<Ordering> {
        fn rank(label: ScopeGraphPathLabel<'_, Label>) -> u8 {
            match label {
                ScopeGraphPathLabel::Declaration => 0,
                ScopeGraphPathLabel::Edge(Label::Import) => 1,
                ScopeGraphPathLabel::Edge(Label::Parent) => 2,
            }
        }
        Some(rank(left).cmp(&rank(right)))
    }

    fn data_leq(
        &self,
        graph: &Graph,
        _input: &ScopeQuery<'_, Reference>,
        left: ScopeDatumId,
        right: ScopeDatumId,
    ) -> bool {
        graph.datum(left).data().name == graph.datum(right).data().name
    }
}

struct IndexedQuery {
    candidate: ScopeDatumId,
    extend_after_match: bool,
}

impl ScopeGraphQuery<ScopeMeta, Label, Relation, Declaration, Reference> for IndexedQuery {
    type State = ();

    fn initial_state(&self, _graph: &Graph, _input: &ScopeQuery<'_, Reference>) {}

    fn step(
        &self,
        _graph: &Graph,
        _input: &ScopeQuery<'_, Reference>,
        _path: &ScopePath,
        _state: &(),
        _edge: crate::EdgeRef<'_, super::ScopeId, ScopeEdgeId, Label>,
    ) -> Option<()> {
        None
    }

    fn visit_candidate_data(
        &self,
        _graph: &Graph,
        _input: &ScopeQuery<'_, Reference>,
        _path: &ScopePath,
        _state: &(),
        visit: &mut impl FnMut(ScopeDatumId),
    ) {
        visit(self.candidate);
    }

    fn accepts(
        &self,
        _graph: &Graph,
        _input: &ScopeQuery<'_, Reference>,
        _path: &ScopePath,
        _state: &(),
        _datum: ScopeDatumId,
    ) -> bool {
        true
    }

    fn should_extend(
        &self,
        _graph: &Graph,
        _input: &ScopeQuery<'_, Reference>,
        _path: &ScopePath,
        _state: &(),
        accepted_candidate_count: usize,
    ) -> bool {
        self.extend_after_match || accepted_candidate_count == 0
    }

    fn compare_labels(
        &self,
        _graph: &Graph,
        _input: &ScopeQuery<'_, Reference>,
        _left: ScopeGraphPathLabel<'_, Label>,
        _right: ScopeGraphPathLabel<'_, Label>,
    ) -> Option<Ordering> {
        Some(Ordering::Equal)
    }
}

const PLAIN: ScopeMeta = ScopeMeta {
    ordered: false,
    opaque_to_nested: false,
};

fn declaration(name: &'static str) -> Declaration {
    Declaration {
        name,
        end: 0,
        visible_to_nested: false,
    }
}

fn reference(name: &'static str, relation: Relation) -> Reference {
    Reference {
        name,
        relation,
        start: u32::MAX,
        name_match: NameMatch::Exact,
    }
}

#[test]
fn storage_is_an_edge_aware_graph_view() {
    let mut graph: ScopeGraph<&str, &str> = ScopeGraph::new();
    let child = graph.add_scope("child");
    let parent = graph.add_scope("parent");
    let edge = graph.add_edge(child, parent, "lexical");

    assert_eq!(graph.scope_count(), 2);
    assert_eq!(graph.edge_ids().collect::<alloc::vec::Vec<_>>(), vec![edge]);
    assert_eq!(graph.edge(edge).source(), child);
    assert_eq!(graph.edge(edge).target(), parent);
    assert_eq!(
        shortest_path(&graph, child, parent, TraversalDirection::Outgoing),
        Some(vec![child, parent])
    );

    assert!(graph.remove_edge(edge));
    assert_eq!(graph.edge_count(), 0);
    assert_eq!(graph.edge_bound(), 1);
}

#[test]
fn local_data_shadows_parent_data_without_crossing_namespaces() {
    let mut graph = Graph::new();
    let root = graph.add_scope(PLAIN);
    let local = graph.add_scope(PLAIN);
    graph.add_edge(local, root, Label::Parent);
    let outer = graph.add_datum(root, Relation::Value, declaration("x"));
    let outer_type = graph.add_datum(root, Relation::Type, declaration("x"));
    let inner = graph.add_datum(local, Relation::Value, declaration("x"));
    let use_x = graph.add_reference(local, reference("x", Relation::Value));

    let result =
        ScopeResolution::compute(&graph, use_x, &LexicalQuery, ScopeResolutionConfig::new());
    assert_eq!(result.stats().reachable_candidate_count, 2);
    assert_eq!(result.target_count(), 1);
    assert_eq!(result.candidates()[0].datum(), inner);
    assert!(!result.is_ambiguous());

    let index = ScopeResolutionIndex::compute(&graph, &LexicalQuery, ScopeResolutionConfig::new());
    assert_eq!(index.references_to(inner), &[use_x]);
    assert!(index.references_to(outer).is_empty());
    assert!(index.references_to(outer_type).is_empty());
}

#[test]
fn equal_import_paths_remain_ambiguous_and_do_not_admit_parents_after_imports() {
    let mut graph = Graph::new();
    let start = graph.add_scope(PLAIN);
    let first = graph.add_scope(PLAIN);
    let second = graph.add_scope(PLAIN);
    let parent = graph.add_scope(PLAIN);
    graph.add_edge(start, first, Label::Import);
    graph.add_edge(start, second, Label::Import);
    graph.add_edge(first, parent, Label::Parent);
    graph.add_edge(second, start, Label::Import);
    let first_x = graph.add_datum(first, Relation::Value, declaration("x"));
    let second_x = graph.add_datum(second, Relation::Value, declaration("x"));
    graph.add_datum(parent, Relation::Value, declaration("x"));
    let use_x = graph.add_reference(start, reference("x", Relation::Value));

    let result =
        ScopeResolution::compute(&graph, use_x, &LexicalQuery, ScopeResolutionConfig::new());
    assert_eq!(result.target_count(), 2);
    assert!(result.is_ambiguous());
    assert_eq!(
        result
            .candidates()
            .iter()
            .map(super::ScopeResolutionCandidate::datum)
            .collect::<alloc::vec::Vec<_>>(),
        vec![first_x, second_x]
    );
    assert!(!result.stats().is_truncated());
}

#[test]
fn a_preferred_import_shadows_a_lexical_parent() {
    let mut graph = Graph::new();
    let start = graph.add_scope(PLAIN);
    let imported = graph.add_scope(PLAIN);
    let parent = graph.add_scope(PLAIN);
    graph.add_edge(start, imported, Label::Import);
    graph.add_edge(start, parent, Label::Parent);
    let imported_x = graph.add_datum(imported, Relation::Value, declaration("x"));
    graph.add_datum(parent, Relation::Value, declaration("x"));
    let use_x = graph.add_reference(start, reference("x", Relation::Value));

    let result =
        ScopeResolution::compute(&graph, use_x, &LexicalQuery, ScopeResolutionConfig::new());
    assert_eq!(result.target_count(), 1);
    assert_eq!(result.candidates()[0].datum(), imported_x);
}

#[test]
fn label_words_remain_comparable_after_paths_enter_different_scopes() {
    let mut graph = Graph::new();
    let start = graph.add_scope(PLAIN);
    let direct_branch = graph.add_scope(PLAIN);
    let nested_branch = graph.add_scope(PLAIN);
    let nested_parent = graph.add_scope(PLAIN);
    graph.add_edge(start, nested_branch, Label::Parent);
    graph.add_edge(nested_branch, nested_parent, Label::Parent);
    graph.add_edge(start, direct_branch, Label::Parent);
    let nested_x = graph.add_datum(nested_parent, Relation::Value, declaration("x"));
    let direct_x = graph.add_datum(direct_branch, Relation::Value, declaration("x"));
    let use_x = graph.add_reference(start, reference("x", Relation::Value));

    let result =
        ScopeResolution::compute(&graph, use_x, &LexicalQuery, ScopeResolutionConfig::new());

    assert_eq!(result.target_count(), 1);
    assert_eq!(result.candidates()[0].datum(), direct_x);
    assert!(
        !result
            .candidates()
            .iter()
            .any(|candidate| candidate.datum() == nested_x)
    );
}

#[test]
fn ordered_and_opaque_scope_rules_fit_the_generic_query_contract() {
    let mut graph = Graph::new();
    let ordered = graph.add_scope(ScopeMeta {
        ordered: true,
        opaque_to_nested: false,
    });
    let first = graph.add_datum(
        ordered,
        Relation::Value,
        Declaration {
            name: "x",
            end: 5,
            visible_to_nested: false,
        },
    );
    let second = graph.add_datum(
        ordered,
        Relation::Value,
        Declaration {
            name: "x",
            end: 10,
            visible_to_nested: false,
        },
    );
    let early = graph.add_reference(
        ordered,
        Reference {
            name: "x",
            relation: Relation::Value,
            start: 7,
            name_match: NameMatch::Exact,
        },
    );
    let late = graph.add_reference(
        ordered,
        Reference {
            name: "x",
            relation: Relation::Value,
            start: 12,
            name_match: NameMatch::Exact,
        },
    );

    let early_result =
        ScopeResolution::compute(&graph, early, &LexicalQuery, ScopeResolutionConfig::new());
    let late_result =
        ScopeResolution::compute(&graph, late, &LexicalQuery, ScopeResolutionConfig::new());
    assert_eq!(early_result.candidates()[0].datum(), first);
    assert_eq!(late_result.candidates()[0].datum(), second);

    let opaque = graph.add_scope(ScopeMeta {
        ordered: false,
        opaque_to_nested: true,
    });
    let nested = graph.add_scope(PLAIN);
    graph.add_edge(nested, opaque, Label::Parent);
    graph.add_datum(opaque, Relation::Value, declaration("hidden"));
    let visible = graph.add_datum(
        opaque,
        Relation::Value,
        Declaration {
            name: "visible",
            end: 0,
            visible_to_nested: true,
        },
    );
    let hidden_use = graph.add_reference(nested, reference("hidden", Relation::Value));
    let visible_use = graph.add_reference(nested, reference("visible", Relation::Value));
    assert!(
        !ScopeResolution::compute(
            &graph,
            hidden_use,
            &LexicalQuery,
            ScopeResolutionConfig::new(),
        )
        .is_resolved()
    );
    assert_eq!(
        ScopeResolution::compute(
            &graph,
            visible_use,
            &LexicalQuery,
            ScopeResolutionConfig::new(),
        )
        .candidates()[0]
            .datum(),
        visible
    );
}

#[test]
fn configured_bounds_report_possible_truncation() {
    let mut graph = Graph::new();
    let start = graph.add_scope(PLAIN);
    let middle = graph.add_scope(PLAIN);
    let end = graph.add_scope(PLAIN);
    graph.add_edge(start, middle, Label::Parent);
    graph.add_edge(middle, end, Label::Parent);
    graph.add_datum(end, Relation::Value, declaration("x"));
    let use_x = graph.add_reference(start, reference("x", Relation::Value));

    let length_limited = ScopeResolution::compute(
        &graph,
        use_x,
        &LexicalQuery,
        ScopeResolutionConfig::new().with_max_path_length(1),
    );
    assert!(!length_limited.is_resolved());
    assert!(length_limited.stats().path_length_limited);

    let count_limited = ScopeResolution::compute(
        &graph,
        use_x,
        &LexicalQuery,
        ScopeResolutionConfig::new().with_max_path_count(1),
    );
    assert!(!count_limited.is_resolved());
    assert!(count_limited.stats().path_count_limited);
}

#[test]
fn linear_resolution_matches_general_search_and_rejects_branching() {
    let mut graph = Graph::new();
    let parent = graph.add_scope(PLAIN);
    let local = graph.add_scope(PLAIN);
    graph.add_edge(local, parent, Label::Parent);
    graph.add_datum(parent, Relation::Value, declaration("x"));
    let use_x = graph.add_reference(local, reference("x", Relation::Value));

    let general =
        ScopeResolution::compute(&graph, use_x, &LexicalQuery, ScopeResolutionConfig::new());
    let linear = ScopeResolution::try_compute_linear(
        &graph,
        use_x,
        &LexicalQuery,
        ScopeResolutionConfig::new(),
    )
    .expect("one parent edge is linear");
    assert_eq!(linear, general);

    let sibling = graph.add_scope(PLAIN);
    graph.add_edge(local, sibling, Label::Parent);
    assert_eq!(
        ScopeResolution::try_compute_linear(
            &graph,
            use_x,
            &LexicalQuery,
            ScopeResolutionConfig::new(),
        ),
        Err(ScopeLinearResolutionError::MultipleEdges { scope: local })
    );

    let mut ambiguous = Graph::new();
    let scope = ambiguous.add_scope(PLAIN);
    ambiguous.add_datum(scope, Relation::Value, declaration("x"));
    ambiguous.add_datum(scope, Relation::Value, declaration("x"));
    let use_x = ambiguous.add_reference(scope, reference("x", Relation::Value));
    assert_eq!(
        ScopeResolution::try_compute_linear(
            &ambiguous,
            use_x,
            &LexicalQuery,
            ScopeResolutionConfig::new(),
        ),
        Err(ScopeLinearResolutionError::MultipleCandidates { scope })
    );
}

#[test]
fn query_owned_candidate_index_avoids_unrelated_scope_data() {
    let mut graph = Graph::new();
    let scope = graph.add_scope(PLAIN);
    graph.add_datum(scope, Relation::Value, declaration("unrelated"));
    let candidate = graph.add_datum(scope, Relation::Value, declaration("chosen"));
    let reference = graph.add_reference(scope, reference("chosen", Relation::Value));

    let resolution = ScopeResolution::compute(
        &graph,
        reference,
        &IndexedQuery {
            candidate,
            extend_after_match: true,
        },
        ScopeResolutionConfig::new(),
    );

    assert_eq!(resolution.candidates().len(), 1);
    assert_eq!(resolution.candidates()[0].datum(), candidate);
}

#[test]
fn query_can_prune_extensions_after_a_decisive_local_match() {
    let mut graph = Graph::new();
    let parent = graph.add_scope(PLAIN);
    let local = graph.add_scope(PLAIN);
    graph.add_edge(local, parent, Label::Parent);
    let candidate = graph.add_datum(local, Relation::Value, declaration("chosen"));
    let reference = graph.add_reference(local, reference("chosen", Relation::Value));

    let resolution = ScopeResolution::compute(
        &graph,
        reference,
        &IndexedQuery {
            candidate,
            extend_after_match: false,
        },
        ScopeResolutionConfig::new(),
    );

    assert_eq!(resolution.stats().explored_path_count, 1);
    assert_eq!(resolution.candidates()[0].datum(), candidate);
}

#[test]
fn ephemeral_queries_support_shadow_aware_completion_without_graph_mutation() {
    let mut graph = Graph::new();
    let parent = graph.add_scope(PLAIN);
    let local = graph.add_scope(PLAIN);
    graph.add_edge(local, parent, Label::Parent);
    let parent_x = graph.add_datum(parent, Relation::Value, declaration("x"));
    let parent_y = graph.add_datum(parent, Relation::Value, declaration("y"));
    let local_x = graph.add_datum(local, Relation::Value, declaration("x"));
    let reference_count = graph.reference_count();
    let completion = Reference {
        name: "",
        relation: Relation::Value,
        start: u32::MAX,
        name_match: NameMatch::Prefix,
    };

    let result = ScopeResolution::compute_from(
        &graph,
        local,
        &completion,
        &LexicalQuery,
        ScopeResolutionConfig::new(),
    );

    assert_eq!(result.start_scope(), local);
    assert_eq!(result.reference(), None);
    assert_eq!(graph.reference_count(), reference_count);
    assert_eq!(
        result
            .candidates()
            .iter()
            .map(super::ScopeResolutionCandidate::datum)
            .collect::<alloc::vec::Vec<_>>(),
        vec![local_x, parent_y]
    );
    assert!(
        !result
            .candidates()
            .iter()
            .any(|candidate| candidate.datum() == parent_x)
    );
}

#[cfg(feature = "serde")]
#[test]
fn scope_graph_storage_round_trips() {
    let mut graph: ScopeGraph<String, String, String, String, String> = ScopeGraph::new();
    let parent = graph.add_scope("parent".into());
    let child = graph.add_scope("child".into());
    graph.add_edge(child, parent, "parent".into());
    graph.add_datum(parent, "value".into(), "x".into());
    graph.add_reference(child, "x".into());

    let json = serde_json::to_string(&graph).expect("serializable scope graph");
    let decoded: ScopeGraph<String, String, String, String, String> =
        serde_json::from_str(&json).expect("valid scope-graph encoding");

    assert_eq!(decoded, graph);
}
