//! Integration tests: scope-graph and stack-graph documents are checked on
//! load.
//!
//! A document names its own identities, so one whose tables disagree has to be
//! refused when it is read, not accepted and then fail on the first query.

#![cfg(feature = "serde")]

use core::cmp::Ordering;

use cfglib::{
    EdgeRef, ScopeDatumId, ScopeEdgeId, ScopeGraph, ScopeGraphPathLabel, ScopeGraphQuery, ScopeId,
    ScopePath, ScopeQuery, ScopeResolution, ScopeResolutionConfig, ScopeResolutionIndex,
    StackGraph, StackNodeId, StackPartialPathConfig, StackPartialPathDatabase, StackResolution,
    StackSearchConfig,
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

/// One value of a document, named by its JSON pointer, and what replaces it.
type Replacement<'a> = (&'a str, Value);

/// Asserts that `T` loads none of the documents made by replacing values of
/// `sound`.
///
/// A tampering is a label and the replacements that together make the
/// document invalid; everything else stays as written, so the one rule it
/// breaks is the one that has to reject it.
fn assert_rejected<T: DeserializeOwned>(sound: &Value, tamperings: &[(&str, &[Replacement<'_>])]) {
    for (label, replacements) in tamperings {
        let mut document = sound.clone();
        for (pointer, value) in *replacements {
            *document
                .pointer_mut(pointer)
                .unwrap_or_else(|| panic!("{label}: the encoding has no value at {pointer}")) =
                value.clone();
        }
        assert!(
            serde_json::from_value::<T>(document).is_err(),
            "{label} must not load"
        );
    }
}

mod scope {
    use super::*;

    type Scopes = ScopeGraph<String, String, String, String, String>;

    /// Names are resolved by equality along every edge, which is all these
    /// tests need from a query: the answer depends on every table.
    struct ByName;

    impl ScopeGraphQuery<String, String, String, String, String> for ByName {
        type State = ();

        fn initial_state(&self, _: &Scopes, _: &ScopeQuery<'_, String>) {}

        fn step(
            &self,
            _: &Scopes,
            _: &ScopeQuery<'_, String>,
            _: &ScopePath,
            (): &(),
            _: EdgeRef<'_, ScopeId, ScopeEdgeId, String>,
        ) -> Option<()> {
            Some(())
        }

        fn accepts(
            &self,
            graph: &Scopes,
            input: &ScopeQuery<'_, String>,
            _: &ScopePath,
            (): &(),
            datum: ScopeDatumId,
        ) -> bool {
            graph.datum(datum).data() == input.data()
        }

        fn compare_labels(
            &self,
            _: &Scopes,
            _: &ScopeQuery<'_, String>,
            _: ScopeGraphPathLabel<'_, String>,
            _: ScopeGraphPathLabel<'_, String>,
        ) -> Option<Ordering> {
            None
        }
    }

    /// Two scopes owning interleaved data and references, so that a table
    /// listing the wrong owner, order, or count differs from the real one.
    fn scopes() -> Scopes {
        let mut graph = ScopeGraph::new();
        let global = graph.add_scope("global".into());
        let local = graph.add_scope("local".into());
        graph.add_edge(local, global, "parent".into());
        graph.add_datum(global, "value".into(), "x".into());
        graph.add_datum(local, "value".into(), "y".into());
        graph.add_datum(global, "value".into(), "z".into());
        graph.add_reference(local, "x".into());
        graph.add_reference(global, "z".into());
        graph
    }

    fn resolutions(graph: &Scopes) -> ScopeResolutionIndex {
        ScopeResolutionIndex::compute(graph, &ByName, ScopeResolutionConfig::new())
    }

    #[test]
    fn a_sound_document_loads_and_resolves_like_the_original() {
        let graph = scopes();
        let document = serde_json::to_value(&graph).expect("serializes");

        let loaded: Scopes = serde_json::from_value(document).expect("a sound document loads");

        assert_eq!(loaded, graph);
        let index = resolutions(&loaded);
        assert_eq!(index, resolutions(&graph));
        assert_eq!(index.resolutions().len(), 2);
        assert!(index.resolutions().iter().all(ScopeResolution::is_resolved));
    }

    #[test]
    fn a_document_whose_tables_disagree_is_rejected() {
        let sound = serde_json::to_value(scopes()).expect("serializes");

        assert_rejected::<Scopes>(
            &sound,
            &[
                (
                    "a data table shorter than the scopes",
                    &[("/scope_data", json!([[0, 2]]))],
                ),
                (
                    "a data table longer than the scopes",
                    &[("/scope_data", json!([[0, 2], [1], []]))],
                ),
                (
                    "a references table shorter than the scopes",
                    &[("/scope_references", json!([[1]]))],
                ),
                (
                    "a references table longer than the scopes",
                    &[("/scope_references", json!([[1], [0], []]))],
                ),
                (
                    "a datum listed under another scope",
                    &[("/scope_data", json!([[0], [1, 2]]))],
                ),
                (
                    "a datum listed twice",
                    &[("/scope_data/0", json!([0, 0, 2]))],
                ),
                (
                    "a datum listed out of identity order",
                    &[("/scope_data/0", json!([2, 0]))],
                ),
                ("a datum nobody lists", &[("/scope_data/0", json!([0]))]),
                (
                    "a listed datum that does not exist",
                    &[("/scope_data/1", json!([9]))],
                ),
                (
                    "a datum owned by a scope that does not exist",
                    &[("/data/0/scope", json!(7))],
                ),
                (
                    "a reference listed under another scope",
                    &[("/scope_references", json!([[0], [1]]))],
                ),
                (
                    "a reference listed twice",
                    &[("/scope_references/0", json!([1, 1]))],
                ),
                (
                    "a reference nobody lists",
                    &[("/scope_references/1", json!([]))],
                ),
                (
                    "a listed reference that does not exist",
                    &[("/scope_references/1", json!([9]))],
                ),
                (
                    "a reference starting in a scope that does not exist",
                    &[("/references/1/scope", json!(7))],
                ),
            ],
        );
    }
}

type Stack = StackGraph<String, String, String, ()>;

/// The nodes of [`stack`] a query starts from and ends at.
struct Nodes {
    reference: StackNodeId,
    definition: StackNodeId,
}

/// A reference in one file that reaches a definition in another through the
/// root, by pushing a scoped symbol whose exported scope is later jumped to,
/// beside a spare file whose nodes `clear_file` retired, so the node table
/// holds vacant slots.
///
/// Node numbers, which the tamperings below write as JSON pointers: 0 root,
/// 1 jump-to-scope, 2 retired exported scope and 3 retired scoped push of the
/// spare file, 4 reference, 5 exported scope, 6 scoped push of it, and 7 the
/// matching scoped pop of the consumer file, then 8 exported scope and
/// 9 definition of the provider file. Edges 0 to 5 are
/// 4→6, 6→7, 7→1, 5→0, 0→8 and 8→9.
fn stack() -> (Stack, Nodes) {
    let mut graph = Stack::new();
    let consumer = graph.add_file("consumer.rs".into());
    let provider = graph.add_file("provider.rs".into());
    let spare = graph.add_file("spare.rs".into());
    let stale = graph
        .add_scope_node(spare, true, "stale scope".into())
        .expect("known file");
    graph
        .add_push_scoped_symbol_node(spare, "stale".into(), stale, false, "stale push".into())
        .expect("a file-local exported scope");
    graph.clear_file(spare).expect("known file");

    let reference = graph
        .add_push_symbol_node(consumer, "member".into(), true, "member use".into())
        .expect("known file");
    let receiver = graph
        .add_scope_node(consumer, true, "receiver scope".into())
        .expect("known file");
    let push_receiver = graph
        .add_push_scoped_symbol_node(consumer, "receiver".into(), receiver, false, "pause".into())
        .expect("a file-local exported scope");
    let pop_receiver = graph
        .add_pop_scoped_symbol_node(consumer, "receiver".into(), false, "resume".into())
        .expect("known file");
    let members = graph
        .add_scope_node(provider, true, "members".into())
        .expect("known file");
    let definition = graph
        .add_pop_symbol_node(provider, "member".into(), true, "member def".into())
        .expect("known file");
    let (root, jump) = (graph.root_node(), graph.jump_to_scope_node());
    for (source, target) in [
        (reference, push_receiver),
        (push_receiver, pop_receiver),
        (pop_receiver, jump),
        (receiver, root),
        (root, members),
        (members, definition),
    ] {
        graph
            .add_edge(source, target, 0, ())
            .expect("a file-local edge, or one through the root");
    }
    (
        graph,
        Nodes {
            reference,
            definition,
        },
    )
}

mod stack_graph {
    use super::*;

    /// Resolves from the reference directly and by stitching file summaries.
    fn resolve(graph: &Stack, nodes: &Nodes) -> [StackResolution<String>; 2] {
        let database = StackPartialPathDatabase::compute(graph, StackPartialPathConfig::new());
        [
            StackResolution::compute(graph, nodes.reference, StackSearchConfig::new()),
            StackResolution::compute_from_partials(
                graph,
                &database,
                nodes.reference,
                StackSearchConfig::new(),
            ),
        ]
    }

    #[test]
    fn a_sound_document_loads_and_resolves_like_the_original() {
        let (graph, nodes) = stack();
        let document = serde_json::to_value(&graph).expect("serializes");

        let loaded: Stack = serde_json::from_value(document).expect("a sound document loads");

        assert_eq!(loaded, graph);
        let [direct, stitched] = resolve(&loaded, &nodes);
        assert_eq!([direct.clone(), stitched], resolve(&graph, &nodes));
        assert_eq!(direct.paths().len(), 1);
        assert_eq!(direct.paths()[0].end(), nodes.definition);
    }

    #[test]
    fn a_document_whose_singletons_are_not_the_singletons_is_rejected() {
        let (graph, _) = stack();
        let sound = serde_json::to_value(&graph).expect("serializes");

        assert_rejected::<Stack>(
            &sound,
            &[
                ("a root that is no node", &[("/root", json!(99))]),
                (
                    "a root that is the jump-to-scope node",
                    &[("/root", json!(1))],
                ),
                ("a root that is an ordinary node", &[("/root", json!(4))]),
                ("a root that was retired", &[("/root", json!(2))]),
                (
                    "a jump-to-scope that is the root",
                    &[("/jump_to_scope", json!(0))],
                ),
                (
                    "a jump-to-scope that is no node",
                    &[("/jump_to_scope", json!(99))],
                ),
                (
                    "a root node that is not a root",
                    &[(
                        "/graph/delta_nodes/0/kind",
                        json!({"Scope": {"exported": true}}),
                    )],
                ),
                (
                    "a jump-to-scope node that is not a jump",
                    &[("/graph/delta_nodes/1/kind", json!("DropScopes"))],
                ),
                (
                    "a singleton in a file",
                    &[("/graph/delta_nodes/0/file", json!(0))],
                ),
                (
                    "a singleton with a payload",
                    &[("/graph/delta_nodes/1/payload", json!("payload"))],
                ),
                (
                    "a second root",
                    &[("/graph/delta_nodes/9/kind", json!("Root"))],
                ),
                (
                    "a second jump-to-scope",
                    &[("/graph/delta_nodes/9/kind", json!("JumpToScope"))],
                ),
            ],
        );
    }

    #[test]
    fn a_document_whose_nodes_and_files_disagree_is_rejected() {
        let (graph, _) = stack();
        let sound = serde_json::to_value(&graph).expect("serializes");

        assert_rejected::<Stack>(
            &sound,
            &[
                (
                    "a file table shorter than the per-file lists",
                    &[("/files", json!(["consumer.rs", "provider.rs"]))],
                ),
                (
                    "fewer per-file lists than files",
                    &[("/file_nodes", json!([[4, 5, 6, 7], [8, 9]]))],
                ),
                (
                    "more per-file lists than files",
                    &[("/file_nodes", json!([[4, 5, 6, 7], [8, 9], [], []]))],
                ),
                (
                    "a node in a file that does not exist",
                    &[("/graph/delta_nodes/4/file", json!(9))],
                ),
                (
                    "an ordinary node in no file",
                    &[("/graph/delta_nodes/4/file", json!(null))],
                ),
                (
                    "an ordinary node without a payload",
                    &[("/graph/delta_nodes/4/payload", json!(null))],
                ),
                (
                    "a retired node in a file that does not exist",
                    &[("/graph/delta_nodes/2/file", json!(9))],
                ),
                (
                    "a retired node without a payload",
                    &[("/graph/delta_nodes/2/payload", json!(null))],
                ),
                (
                    "nodes listed under another file",
                    &[("/file_nodes", json!([[4, 5, 6], [7, 8, 9], []]))],
                ),
                (
                    "a node whose file does not list it",
                    &[("/graph/delta_nodes/9/file", json!(0))],
                ),
                (
                    "a node no file lists",
                    &[("/file_nodes/0", json!([4, 5, 6]))],
                ),
                (
                    "a node listed twice",
                    &[("/file_nodes/0", json!([4, 4, 5, 6, 7]))],
                ),
                (
                    "a file's nodes out of identity order",
                    &[("/file_nodes/0", json!([5, 4, 6, 7]))],
                ),
                (
                    "a retired node still listed by its file",
                    &[("/file_nodes/2", json!([2, 3]))],
                ),
                (
                    "a listed node that does not exist",
                    &[("/file_nodes/1", json!([8, 9, 99]))],
                ),
            ],
        );
    }

    #[test]
    fn a_document_whose_scoped_push_names_a_foreign_scope_is_rejected() {
        let (graph, _) = stack();
        let sound = serde_json::to_value(&graph).expect("serializes");
        let live_scope = "/graph/delta_nodes/6/kind/PushScopedSymbol/scope";
        let retired_scope = "/graph/delta_nodes/3/kind/PushScopedSymbol/scope";

        assert_rejected::<Stack>(
            &sound,
            &[
                ("a scope in another file", &[(live_scope, json!(8))]),
                ("a node that is not a scope", &[(live_scope, json!(4))]),
                ("a node that does not exist", &[(live_scope, json!(99))]),
                (
                    "a scope that is not exported",
                    &[("/graph/delta_nodes/5/kind/Scope/exported", json!(false))],
                ),
                (
                    "a retired push naming a node that does not exist",
                    &[(retired_scope, json!(99))],
                ),
                (
                    "a retired push naming a scope of another file",
                    &[(retired_scope, json!(5))],
                ),
            ],
        );
    }

    #[test]
    fn a_document_with_an_edge_no_builder_adds_is_rejected() {
        let (graph, _) = stack();
        let sound = serde_json::to_value(&graph).expect("serializes");

        assert_rejected::<Stack>(
            &sound,
            &[
                // The two singletons swap kinds, which leaves the root's one
                // edge leaving the jump-to-scope node.
                (
                    "an edge leaving the jump-to-scope node",
                    &[
                        ("/root", json!(1)),
                        ("/jump_to_scope", json!(0)),
                        ("/graph/delta_nodes/0/kind", json!("JumpToScope")),
                        ("/graph/delta_nodes/1/kind", json!("Root")),
                    ],
                ),
                // The reference moves to the provider's file, consistently,
                // which leaves its edge crossing from one file to another.
                (
                    "an edge between two files",
                    &[
                        ("/graph/delta_nodes/4/file", json!(1)),
                        ("/file_nodes/0", json!([5, 6, 7])),
                        ("/file_nodes/1", json!([4, 8, 9])),
                    ],
                ),
            ],
        );
    }
}
