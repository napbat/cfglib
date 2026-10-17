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
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

/// Asserts that `T` loads none of `sound` with one value replaced.
///
/// Each tampering names the value to replace by its JSON pointer, so what is
/// left alone stays a well-formed document.
fn assert_rejected<T: DeserializeOwned>(sound: &Value, tamperings: &[(&str, &str, Value)]) {
    for (label, pointer, value) in tamperings {
        let mut document = sound.clone();
        *document
            .pointer_mut(pointer)
            .unwrap_or_else(|| panic!("{label}: the encoding has no value at {pointer}")) =
            value.clone();
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
        assert!(serde_json::from_value::<Scopes>(sound.clone()).is_ok());

        assert_rejected::<Scopes>(
            &sound,
            &[
                (
                    "a data table shorter than the scopes",
                    "/scope_data",
                    json!([[0, 2]]),
                ),
                (
                    "a data table longer than the scopes",
                    "/scope_data",
                    json!([[0, 2], [1], []]),
                ),
                (
                    "a references table shorter than the scopes",
                    "/scope_references",
                    json!([[1]]),
                ),
                (
                    "a references table longer than the scopes",
                    "/scope_references",
                    json!([[1], [0], []]),
                ),
                (
                    "a datum listed under another scope",
                    "/scope_data",
                    json!([[0], [1, 2]]),
                ),
                ("a datum listed twice", "/scope_data/0", json!([0, 0, 2])),
                (
                    "a datum listed out of identity order",
                    "/scope_data/0",
                    json!([2, 0]),
                ),
                ("a datum nobody lists", "/scope_data/0", json!([0])),
                (
                    "a listed datum that does not exist",
                    "/scope_data/1",
                    json!([9]),
                ),
                (
                    "a datum owned by a scope that does not exist",
                    "/data/0/scope",
                    json!(7),
                ),
                (
                    "a reference listed under another scope",
                    "/scope_references",
                    json!([[0], [1]]),
                ),
                (
                    "a reference listed twice",
                    "/scope_references/0",
                    json!([1, 1]),
                ),
                ("a reference nobody lists", "/scope_references/1", json!([])),
                (
                    "a listed reference that does not exist",
                    "/scope_references/1",
                    json!([9]),
                ),
                (
                    "a reference starting in a scope that does not exist",
                    "/references/1/scope",
                    json!(7),
                ),
            ],
        );
    }
}
