extern crate alloc;

use alloc::string::{String, ToString};

use crate::graph::store::Graph;
use crate::test_util::golden::assert_golden;

use super::{TextErrorKind, parse_text};

/// A store whose labels exercise every escape and whose indices have a hole.
fn awkward_store() -> Graph<String, String> {
    let mut graph = Graph::new();
    let first = graph.add_node("says \"hi\"\nand a back\\slash".to_string());
    let removed = graph.add_node("gone".to_string());
    let last = graph.add_node("-> not an arrow".to_string());
    graph.add_edge(first, last, "crosses the hole".to_string());
    graph.add_edge(last, first, String::new());
    graph.add_edge(first, first, "self".to_string());
    graph.remove_node(removed);
    graph
}

#[test]
fn a_store_writes_one_line_per_node_and_edge() {
    assert_golden("text/store-escaped-labels.txt", &awkward_store().to_text());
}

#[test]
fn writing_parsing_and_writing_again_is_a_fixed_point() {
    let once = awkward_store().to_text();
    let twice = parse_text(&once)
        .expect("the writer emits its own grammar")
        .to_text();
    assert_eq!(once, twice);
}

#[test]
fn parsing_reproduces_identity_topology_and_labels() {
    let original = awkward_store();
    let parsed = parse_text(&original.to_text()).expect("the writer emits its own grammar");

    assert_eq!(parsed.node_count(), original.node_count());
    assert_eq!(
        parsed.node_bound(),
        original.node_bound(),
        "the hole survives"
    );
    assert_eq!(parsed.edge_count(), original.edge_count());
    for node in original.node_ids() {
        assert_eq!(parsed.node(node), original.node(node));
    }
    for edge in original.edge_ids() {
        assert_eq!(parsed.edge(edge).source(), original.edge(edge).source());
        assert_eq!(parsed.edge(edge).target(), original.edge(edge).target());
        assert_eq!(parsed.edge(edge).payload(), original.edge(edge).payload());
    }
}

#[test]
fn comments_and_blank_lines_are_not_content() {
    let graph = parse_text("# the entry\n\nn0 first\n  \nn1\nn0 -> n1\n").expect("valid text");
    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.edge_count(), 1);
    assert_eq!(graph.node(graph.node_ids().next().unwrap()), "first");
}

#[test]
fn node_indices_must_ascend() {
    let error = parse_text("n0 a\nn1 b\nn0 again\n").expect_err("n0 is already declared");
    assert_eq!(error.line, 3);
    assert_eq!(error.kind, TextErrorKind::NodeOrder { expected: 2 });
}

#[test]
fn an_edge_cannot_name_an_undeclared_node() {
    let error = parse_text("n0 a\nn0 -> n4\n").expect_err("n4 was never declared");
    assert_eq!(error.line, 2);
    assert_eq!(error.kind, TextErrorKind::UnknownNode { index: 4 });
}

#[test]
fn an_unknown_escape_is_rejected() {
    let error = parse_text("n0 a\\q\n").expect_err("there is no \\q escape");
    assert_eq!(error.kind, TextErrorKind::Escape);
}

#[test]
fn a_malformed_identifier_is_rejected() {
    let error = parse_text("node0 a\n").expect_err("identifiers are n then digits");
    assert_eq!(error.kind, TextErrorKind::Identifier);
}

#[test]
fn an_error_names_its_line_and_what_was_wrong() {
    let error = parse_text("n0 a\nn0 -> n4\n").expect_err("n4 was never declared");
    assert_eq!(
        alloc::format!("{error}"),
        "line 2: node 4 has not been declared"
    );
}
