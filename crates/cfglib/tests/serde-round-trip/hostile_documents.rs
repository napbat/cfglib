//! Documents that are not a value of the type: truncated, mistyped, naming a
//! variant that does not exist, or internally inconsistent. A loader gets
//! these from disk and from other tools, so each must come back as an error
//! the caller can report, never as a panic and never as a value that panics
//! later.

use cfglib::{
    BasicBlock, Cfg, ClrExceptionRegion, Edge, ExceptionFlow, Graph, HandlerMetadata, HornClauses,
    JumpTable, KeyedGraph, Region, ScopeGraph, SehExceptionRegion, SehRegistrationChain,
    StackGraph, StackPartialPathDatabase, VectoredExceptionModel,
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::consumer::{Op, Route, rich_cfg};
use crate::json::{document, load};

type Store = Graph<String, u32>;

/// Documents no type accepts, however they are shaped.
const NOT_A_VALUE: [&str; 9] = [
    "",
    "   ",
    "null",
    "42",
    "\"text\"",
    "[]",
    "{",
    "{\"graph\": ",
    "[1, 2,",
];

fn assert_rejects_garbage<T: DeserializeOwned>(name: &str) {
    for garbage in NOT_A_VALUE {
        assert!(
            serde_json::from_str::<T>(garbage).is_err(),
            "{name} accepted {garbage:?}"
        );
    }
    let deeply_nested = "[".repeat(100_000);
    assert!(
        serde_json::from_str::<T>(&deeply_nested).is_err(),
        "{name} accepted a document nested 100000 levels deep"
    );
    assert!(
        serde_json::from_str::<T>("{}").is_err(),
        "{name} accepted an empty object, which has none of its fields"
    );
}

#[test]
fn every_document_type_rejects_text_that_is_not_one_of_its_values() {
    assert_rejects_garbage::<Cfg<Op, Route>>("Cfg");
    assert_rejects_garbage::<BasicBlock<Op>>("BasicBlock");
    assert_rejects_garbage::<Edge<Route>>("Edge");
    assert_rejects_garbage::<Region>("Region");
    assert_rejects_garbage::<HandlerMetadata<u32>>("HandlerMetadata");
    assert_rejects_garbage::<Store>("Graph");
    assert_rejects_garbage::<KeyedGraph<String, String, u32>>("KeyedGraph");
    assert_rejects_garbage::<ScopeGraph<String, String, String, String, String>>("ScopeGraph");
    assert_rejects_garbage::<StackGraph<String, String, String, String>>("StackGraph");
    assert_rejects_garbage::<StackPartialPathDatabase>("StackPartialPathDatabase");
    assert_rejects_garbage::<HornClauses>("HornClauses");
    assert_rejects_garbage::<JumpTable<u64>>("JumpTable");
    assert_rejects_garbage::<VectoredExceptionModel<String>>("VectoredExceptionModel");
    assert_rejects_garbage::<SehRegistrationChain<u32, u64>>("SehRegistrationChain");
    assert_rejects_garbage::<SehExceptionRegion>("SehExceptionRegion");
    assert_rejects_garbage::<ClrExceptionRegion<u32>>("ClrExceptionRegion");
    assert_rejects_garbage::<ExceptionFlow<String>>("ExceptionFlow");
}

/// Replace every string equal to `from` anywhere in `value`.
fn replace_string(value: &mut Value, from: &str, to: &str) {
    match value {
        Value::String(text) if text == from => to.clone_into(text),
        Value::Array(items) => items
            .iter_mut()
            .for_each(|item| replace_string(item, from, to)),
        Value::Object(fields) => fields
            .values_mut()
            .for_each(|field| replace_string(field, from, to)),
        _ => {}
    }
}

#[test]
fn a_variant_name_that_does_not_exist_is_rejected_wherever_it_appears() {
    for (known, unknown) in [
        ("Fallthrough", "Sideways"),
        ("Catch", "Rethrow"),
        ("Unknown", "Partial"),
        ("Normal", "Abnormal"),
    ] {
        let value = document(&rich_cfg());
        let mut replaced = value.clone();
        replace_string(&mut replaced, known, unknown);
        assert_ne!(replaced, value, "the fixture contains {known}");
        assert!(
            load::<Cfg<Op, Route>>(replaced).is_err(),
            "{unknown} was accepted for {known}"
        );
    }
}

#[test]
fn a_cfg_document_with_a_missing_or_mistyped_field_is_rejected() {
    let value = document(&rich_cfg());
    assert!(
        load::<Cfg<Op, Route>>(value.clone()).is_ok(),
        "the unedited document loads"
    );

    for field in ["graph", "entry", "regions"] {
        let mut missing = value.clone();
        missing
            .as_object_mut()
            .expect("a CFG is an object")
            .remove(field);
        assert!(
            load::<Cfg<Op, Route>>(missing).is_err(),
            "a document without `{field}` loaded"
        );
    }
    for (field, wrong) in [
        ("entry", json!("bb0")),
        ("entry", json!(-1)),
        ("entry", json!(4_294_967_296_u64)),
        ("regions", json!({})),
        ("cleanups", json!("none")),
        ("graph", json!([])),
    ] {
        let mut edited = value.clone();
        edited[field] = wrong.clone();
        assert!(
            load::<Cfg<Op, Route>>(edited).is_err(),
            "`{field}` = {wrong} was accepted"
        );
    }
}

#[test]
fn a_cfg_whose_entry_is_not_a_live_block_of_its_graph_is_rejected() {
    let original = rich_cfg();
    let removed = (0..original.block_bound())
        .map(cfglib::BlockId::from_index)
        .find(|&block| !original.contains_block(block))
        .expect("the fixture removed a block");

    for entry in [
        json!(original.block_bound()),
        json!(removed.index()),
        json!(u32::MAX),
    ] {
        let mut edited = document(&original);
        edited["entry"] = entry.clone();
        assert!(
            load::<Cfg<Op, Route>>(edited).is_err(),
            "entry {entry} names no live block, yet the document loaded"
        );
    }
}

#[test]
fn a_cfg_whose_handler_names_a_block_it_does_not_have_is_rejected() {
    let mut edited = document(&rich_cfg());
    let handler = &mut edited["regions"][0]["handlers"][0];
    handler["entry"] = json!(10_000);
    assert!(
        load::<Cfg<Op, Route>>(edited).is_err(),
        "a handler entry beyond the block bound was accepted; compaction would panic on it"
    );
}

/// The documented outcome for an inconsistent store document: rejected, or
/// loaded into a store whose own answers agree with each other. A store that
/// loads and then panics or contradicts itself is the failure.
fn assert_rejected_or_self_consistent(edited: Value, what: &str) {
    let Ok(graph) = load::<Store>(edited) else {
        return;
    };
    let live_nodes = graph.node_ids().count();
    let live_edges = graph.edge_ids().count();
    assert_eq!(
        graph.node_count(),
        live_nodes,
        "{what}: node_count contradicts the live nodes"
    );
    assert_eq!(
        graph.edge_count(),
        live_edges,
        "{what}: edge_count contradicts the live edges"
    );
}

fn store_document() -> Value {
    let mut graph = Store::new();
    let a = graph.add_node("a".to_owned());
    let b = graph.add_node("b".to_owned());
    graph.add_edge(a, b, 1);
    graph.compact();
    let c = graph.add_node("c".to_owned());
    graph.add_edge(b, c, 2);
    document(&graph)
}

#[test]
fn a_store_document_whose_live_counters_lie_is_rejected_or_corrected() {
    for key in ["live_node_count", "live_edge_count"] {
        let mut edited = store_document();
        edited[key] = json!(99);
        assert_rejected_or_self_consistent(edited, key);
    }
}

#[test]
fn a_store_document_with_an_adjacency_run_naming_a_missing_edge_is_rejected() {
    let mut edited = store_document();
    edited["out_edges"][0] = json!(9_999);
    assert!(
        load::<Store>(edited).is_err(),
        "an adjacency run naming edge 9999 loaded"
    );
}

#[test]
fn a_store_document_with_an_edge_into_a_missing_node_is_rejected() {
    let mut edited = store_document();
    edited["base_edges"][0]["target"] = json!(77);
    assert!(
        load::<Store>(edited).is_err(),
        "an edge into node 77 loaded"
    );
}

#[test]
fn a_store_document_whose_offsets_do_not_cover_its_nodes_is_rejected() {
    let mut edited = store_document();
    edited["out_offsets"] = json!([0]);
    assert!(
        load::<Store>(edited).is_err(),
        "offsets shorter than the node count loaded"
    );
}

#[test]
fn a_clause_set_whose_tables_disagree_is_rejected() {
    let clauses = |heads: Value, body_lens: Value, dependents: Value| {
        json!({
            "fact_count": 2,
            "heads": heads,
            "body_lens": body_lens,
            "dependents": dependents,
        })
    };
    for (what, document) in [
        (
            "a head beyond the fact count",
            clauses(json!([7]), json!([0]), json!([[], []])),
        ),
        (
            "fewer body lengths than clauses",
            clauses(json!([0]), json!([]), json!([[], []])),
        ),
        (
            "a dependent clause that does not exist",
            clauses(json!([0]), json!([1]), json!([[], [5]])),
        ),
        (
            "fewer dependent lists than facts",
            clauses(json!([0]), json!([0]), json!([[]])),
        ),
    ] {
        assert!(
            load::<HornClauses>(document).is_err(),
            "{what}: the load succeeded, and derivable() would panic on it"
        );
    }
}
