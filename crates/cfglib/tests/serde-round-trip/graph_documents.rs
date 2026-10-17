//! The store and the structures built on it, persisted by an indexer between
//! runs: `Graph`, `KeyedGraph`, `HornClauses`, and recovered `JumpTable`s.

use std::collections::BTreeMap;
use std::fmt::{Debug, Display};

use cfglib::{
    DominatorTree, EdgeId, EdgeKind, Graph, HornClauses, JumpTable, KeyedGraph, NodeId, Rooted,
    SwitchTargets, recover_switch_tables, tarjan_scc,
};
use serde_json::{Value, json};

use crate::consumer::Op;
use crate::golden::assert_golden;
use crate::json::{decode, document, encode, load, round_trip};

type Store = Graph<String, u32>;

/// A store in every state a long-lived index leaves one in: a compacted base,
/// a delta of later nodes and edges, two edges whose endpoints were
/// redirected, and a removed edge and node whose slots are still there.
fn layered_store() -> Store {
    let mut graph = Store::new();
    let [parse, check, lower, emit] =
        ["parse", "check", "lower", "emit"].map(|name| graph.add_node(name.to_owned()));
    let edges = [
        (parse, check),
        (parse, lower),
        (check, lower),
        (lower, emit),
        (parse, emit),
    ]
    .into_iter()
    .zip(0..)
    .map(|((source, target), payload)| graph.add_edge(source, target, payload))
    .collect::<Vec<_>>();
    graph.compact();

    let link = graph.add_node("link".to_owned());
    let doomed = graph.add_node("doomed".to_owned());
    graph.add_edge(emit, link, 5);
    graph.add_edge(link, parse, 6);
    graph.add_edge(check, link, 7);
    graph.add_edge(doomed, parse, 8);
    graph.add_edge(lower, doomed, 9);

    graph.redirect_edge_target(edges[4], link);
    graph.redirect_edge_source(edges[3], check);
    graph.remove_edge(edges[1]);
    graph.remove_node(doomed);
    graph
}

/// The decoded store checked against the original through every accessor,
/// including the slots of removed entities, and through the analyses and
/// renderings a consumer runs on it.
fn assert_same_store<N, E>(actual: &Graph<N, E>, expected: &Graph<N, E>)
where
    N: PartialEq + Debug + Display,
    E: PartialEq + Debug + Display,
{
    assert_eq!(actual, expected);
    assert_eq!(actual.node_bound(), expected.node_bound());
    assert_eq!(actual.node_count(), expected.node_count());
    assert_eq!(actual.edge_bound(), expected.edge_bound());
    assert_eq!(actual.edge_count(), expected.edge_count());

    for index in 0..expected.node_bound() {
        let node = NodeId::from_index(index);
        assert_eq!(
            actual.contains_node(node),
            expected.contains_node(node),
            "{node}"
        );
        assert_eq!(actual.node(node), expected.node(node), "{node}");
    }
    for index in 0..expected.edge_bound() {
        let edge = EdgeId::from_index(index);
        assert_eq!(
            actual.contains_edge(edge),
            expected.contains_edge(edge),
            "{edge}"
        );
        let (actual, expected) = (actual.edge(edge), expected.edge(edge));
        assert_eq!(actual.source(), expected.source(), "{edge}");
        assert_eq!(actual.target(), expected.target(), "{edge}");
        assert_eq!(actual.payload(), expected.payload(), "{edge}");
    }
    for node in expected.node_ids() {
        assert_eq!(
            actual.outgoing(node).collect::<Vec<_>>(),
            expected.outgoing(node).collect::<Vec<_>>(),
            "{node}"
        );
        assert_eq!(
            actual.incoming(node).collect::<Vec<_>>(),
            expected.incoming(node).collect::<Vec<_>>(),
            "{node}"
        );
    }

    let root = expected.node_ids().next().expect("the fixture has nodes");
    assert_eq!(
        DominatorTree::compute(&Rooted::new(actual, root)),
        DominatorTree::compute(&Rooted::new(expected, root))
    );
    assert_eq!(actual.to_text(), expected.to_text());
    assert_eq!(actual.to_dot(), expected.to_dot());
}

#[test]
fn a_store_with_a_base_a_delta_removed_slots_and_redirected_edges_decodes_equal() {
    let original = layered_store();
    assert!(
        original.node_bound() > original.node_count(),
        "a node slot was removed"
    );
    assert!(
        original.edge_bound() > original.edge_count(),
        "edge slots were removed"
    );

    let decoded: Store = round_trip(&original);
    assert_eq!(decoded, original);
}

#[test]
fn the_decoded_store_agrees_with_the_original_through_every_accessor_and_analysis() {
    let original = layered_store();
    assert_same_store(&round_trip(&original), &original);
}

#[test]
fn writing_a_decoded_store_reproduces_the_document_byte_for_byte() {
    let first = encode(&layered_store());
    let second = encode(&decode::<Store>(&first));
    assert_eq!(first, second);
}

/// Where each old slot of a store went through a compaction, by index.
type Renumbered = Vec<Option<usize>>;

/// Apply one fixed edit script and report where every old node and edge went.
fn evolve(mut graph: Store) -> (Store, Renumbered, Renumbered) {
    let first = graph.node_ids().next().expect("the fixture has nodes");
    let last = graph.node_ids().last().expect("the fixture has nodes");
    let fresh = graph.add_node("fresh".to_owned());
    let added = graph.add_edge(last, fresh, 10);
    graph.add_edge(fresh, first, 11);
    graph.redirect_edge_target(added, first);
    graph.remove_node(last);

    let (node_bound, edge_bound) = (graph.node_bound(), graph.edge_bound());
    let renumbering = graph.compact();
    let nodes = (0..node_bound)
        .map(|index| {
            renumbering
                .node(NodeId::from_index(index))
                .map(NodeId::index)
        })
        .collect();
    let edges = (0..edge_bound)
        .map(|index| {
            renumbering
                .edge(EdgeId::from_index(index))
                .map(EdgeId::index)
        })
        .collect();
    (graph, nodes, edges)
}

#[test]
fn the_decoded_store_keeps_evolving_exactly_like_the_original() {
    let (expected, expected_nodes, expected_edges) = evolve(layered_store());
    let (actual, actual_nodes, actual_edges) = evolve(round_trip(&layered_store()));

    assert_same_store(&actual, &expected);
    assert_eq!(
        actual_nodes, expected_nodes,
        "the same renumbering of nodes"
    );
    assert_eq!(
        actual_edges, expected_edges,
        "the same renumbering of edges"
    );
    assert_eq!(
        actual.node_bound(),
        actual.node_count(),
        "compaction closed the holes"
    );
}

#[test]
fn a_store_that_was_never_compacted_and_one_that_just_was_both_round_trip() {
    let mut delta_only = Store::new();
    let a = delta_only.add_node("a".to_owned());
    let b = delta_only.add_node("b".to_owned());
    let first = delta_only.add_edge(a, b, 1);
    delta_only.add_edge(b, a, 2);
    delta_only.remove_edge(first);
    assert_same_store(&round_trip(&delta_only), &delta_only);

    let mut compacted = layered_store();
    compacted.compact();
    let decoded: Store = round_trip(&compacted);
    assert_same_store(&decoded, &compacted);
    assert_eq!(decoded.node_bound(), decoded.node_count());
}

#[test]
fn an_empty_store_round_trips() {
    let empty = Store::new();
    let decoded: Store = round_trip(&empty);
    assert!(decoded.is_empty());
    assert_eq!(decoded, empty);
}

#[test]
fn a_store_that_never_relocated_has_no_overlay_in_its_document_and_loads_without_one() {
    let mut graph = Store::new();
    let a = graph.add_node("a".to_owned());
    let b = graph.add_node("b".to_owned());
    graph.add_edge(a, b, 1);
    graph.compact();

    let value = document(&graph);
    assert!(value.get("relocations").is_none(), "{value}");
    let decoded: Store = load(value.clone()).expect("a document without the optional field loads");
    assert_eq!(decoded, graph);

    let mut explicit_null = value;
    explicit_null["relocations"] = Value::Null;
    let decoded: Store = load(explicit_null).expect("an explicit null is the same as absent");
    assert_eq!(decoded, graph);
}

#[test]
fn the_store_document_is_stable() {
    let mut graph = Store::new();
    let a = graph.add_node("a".to_owned());
    let b = graph.add_node("b".to_owned());
    let first = graph.add_edge(a, b, 1);
    graph.compact();
    let c = graph.add_node("c".to_owned());
    graph.add_edge(b, c, 2);
    graph.remove_edge(first);

    let json = serde_json::to_string_pretty(&graph).expect("the value serializes");
    assert_golden("serde-round-trip/small-store.json", &json);
}

fn dependency_graph() -> KeyedGraph<String, String, u32> {
    let mut graph = KeyedGraph::new();
    for (importer, imported, weight) in [
        ("pkg::b", "pkg::a", 1),
        ("pkg::c", "pkg::a", 2),
        ("pkg::a", "pkg::d", 3),
        ("pkg::b", "pkg::d", 4),
        ("pkg::d", "pkg::b", 5),
    ] {
        let from = graph.intern(&importer.to_owned());
        let to = graph.intern(&imported.to_owned());
        graph.add_edge(from, to, weight);
    }
    graph
}

#[test]
fn a_keyed_graph_keeps_its_keys_ids_and_edges() {
    let original = dependency_graph();
    let decoded: KeyedGraph<String, String, u32> = round_trip(&original);
    assert_eq!(decoded, original);

    assert_eq!(
        decoded.keys().collect::<Vec<_>>(),
        original.keys().collect::<Vec<_>>()
    );
    for (key, id) in original.keys() {
        assert_eq!(decoded.node_id(key), Some(id), "{key}");
        assert_eq!(
            decoded.graph().node(id),
            key,
            "a self-keyed node carries its key"
        );
    }
    assert_same_store(decoded.graph(), original.graph());
    assert_eq!(tarjan_scc(&decoded).len(), tarjan_scc(&original).len());
}

#[test]
fn a_decoded_keyed_graph_keeps_minting_dense_ids_in_the_same_order() {
    let mut original = dependency_graph();
    let mut decoded: KeyedGraph<String, String, u32> = round_trip(&original);

    for graph in [&mut original, &mut decoded] {
        let new = graph.intern(&"pkg::e".to_owned());
        let known = graph.node_id(&"pkg::a".to_owned()).expect("a is minted");
        graph.add_edge(new, known, 6);
        assert_eq!(
            graph.add_edge_between(&"pkg::e".to_owned(), &"pkg::missing".to_owned(), 7),
            None
        );
    }
    assert_eq!(decoded, original);
    assert_eq!(
        decoded.node_id(&"pkg::e".to_owned()),
        Some(NodeId::from_index(4))
    );
}

/// The document of the dependency graph with its key table edited by `edit`.
/// A swap of two keys cannot be detected (a node carries no key of its own
/// unless the consumer made it self-keyed), so only disagreements about which
/// nodes exist are in scope.
fn load_with_key_table(
    edit: impl FnOnce(&mut serde_json::Map<String, Value>),
) -> Result<KeyedGraph<String, String, u32>, serde_json::Error> {
    let mut value = document(&dependency_graph());
    edit(
        value["ids"]
            .as_object_mut()
            .expect("the key table is a map"),
    );
    load(value)
}

#[test]
fn a_key_naming_a_node_the_graph_does_not_have_is_rejected() {
    let loaded = load_with_key_table(|table| {
        table.insert("pkg::a".to_owned(), json!(99));
    });
    assert!(
        loaded.is_err(),
        "a key naming node 99 of a 4-node graph loaded"
    );
}

#[test]
fn two_keys_naming_one_node_are_rejected() {
    let loaded = load_with_key_table(|table| {
        let shared = table["pkg::a"].clone();
        table.insert("pkg::b".to_owned(), shared);
    });
    assert!(loaded.is_err(), "two keys naming one node loaded");
}

#[test]
fn a_key_table_with_fewer_keys_than_live_nodes_is_rejected() {
    let loaded = load_with_key_table(|table| {
        table.remove("pkg::d");
    });
    assert!(loaded.is_err(), "a node with no key loaded");
}

#[test]
fn a_key_table_with_more_keys_than_live_nodes_is_rejected() {
    let loaded = load_with_key_table(|table| {
        table.insert("pkg::extra".to_owned(), json!(0));
    });
    assert!(loaded.is_err(), "five keys for four nodes loaded");
}

/// Grammar nullability over four symbols, the way a grammar tool would
/// persist what it has derived so far: `S -> A B`, `A -> ε`, `B -> A A`, and
/// `C -> "x"` with no nullable production.
fn nullability() -> HornClauses {
    let mut clauses = HornClauses::new(4);
    clauses.add_clause(0, &[1, 2]);
    clauses.add_clause(1, &[]);
    clauses.add_clause(2, &[1, 1]);
    clauses
}

#[test]
fn a_clause_set_decodes_equal_and_solves_to_the_same_fixpoint() {
    let original = nullability();
    let decoded: HornClauses = round_trip(&original);
    assert_eq!(decoded, original);
    assert_eq!(decoded.derivable(), [true, true, true, false]);
    assert_eq!(decoded.derivable(), original.derivable());
}

#[test]
fn a_decoded_clause_set_keeps_accepting_clauses_and_counting_repeated_bodies() {
    let mut original = nullability();
    let mut decoded: HornClauses = round_trip(&original);
    for clauses in [&mut original, &mut decoded] {
        // `C <- D & D` needs D once per occurrence; D has no clause at all.
        clauses.add_clause(3, &[3, 3]);
        clauses.add_clause(3, &[0]);
    }
    assert_eq!(decoded, original);
    assert_eq!(decoded.derivable(), [true, true, true, true]);
}

#[test]
fn cyclic_support_with_no_axiom_stays_underivable_after_a_round_trip() {
    let mut clauses = HornClauses::new(2);
    clauses.add_clause(0, &[1]);
    clauses.add_clause(1, &[0]);
    let decoded: HornClauses = round_trip(&clauses);
    assert_eq!(decoded.derivable(), [false, false]);
}

fn switch_tables<T: Clone>(first: &[T], default: T) -> Vec<JumpTable<T>> {
    vec![
        JumpTable {
            block: cfglib::BlockId::from_raw(1),
            dispatch: SwitchTargets {
                targets: first.to_vec(),
                default_target: Some(default),
            },
        },
        JumpTable {
            block: cfglib::BlockId::from_raw(2),
            dispatch: SwitchTargets {
                targets: Vec::new(),
                default_target: None,
            },
        },
    ]
}

/// A CFG whose blocks 1 and 2 end in an indirect jump, as discovery leaves
/// it before the tables are resolved.
fn unresolved_dispatch() -> cfglib::Cfg<Op> {
    let mut cfg = cfglib::Cfg::<Op>::new();
    let dispatch = cfg.new_block();
    let empty_dispatch = cfg.new_block();
    let [first, second, third, fallback] = [(); 4].map(|()| cfg.new_block());
    cfg.add_edge(cfg.entry(), dispatch, EdgeKind::Fallthrough);
    cfg.add_edge(dispatch, first, EdgeKind::IndirectJump);
    cfg.add_edge(empty_dispatch, second, EdgeKind::IndirectJump);
    cfg.add_edge(dispatch, third, EdgeKind::IndirectJump);
    cfg.add_edge(dispatch, fallback, EdgeKind::Fallthrough);
    cfg
}

#[test]
fn persisted_jump_tables_rewire_a_cfg_like_the_originals() {
    let addresses = switch_tables(&[0x1000_u64, 0x1010, 0x1020], 0x1030);
    let decoded: Vec<JumpTable<u64>> = round_trip(&addresses);
    assert_eq!(decoded, addresses);

    // Blocks 3.. are the case targets, by address.
    let by_address: BTreeMap<u64, _> = [0x1000_u64, 0x1010, 0x1020, 0x1030]
        .into_iter()
        .zip((3..).map(cfglib::BlockId::from_raw))
        .collect();
    let mut original_cfg = unresolved_dispatch();
    let mut decoded_cfg = unresolved_dispatch();
    let from_original = recover_switch_tables(&mut original_cfg, &addresses, |address| {
        by_address.get(address).copied()
    });
    let from_decoded = recover_switch_tables(&mut decoded_cfg, &decoded, |address| {
        by_address.get(address).copied()
    });

    let summary = |recovery: &[cfglib::SwitchRecovery]| {
        recovery
            .iter()
            .map(|each| (each.block, each.case_count))
            .collect::<Vec<_>>()
    };
    assert_eq!(summary(&from_decoded), summary(&from_original));
    assert_eq!(decoded_cfg, original_cfg);
}

#[test]
fn jump_tables_over_source_level_tokens_round_trip() {
    let tokens = switch_tables(
        &["case_a".to_owned(), "case_b".to_owned()],
        "default".to_owned(),
    );
    let decoded: Vec<JumpTable<String>> = round_trip(&tokens);
    assert_eq!(decoded, tokens);
    assert_eq!(decoded[1].dispatch.default_target, None);
}
