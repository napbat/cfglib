//! DOT output of a code-intelligence graph: styles, hooks, and escaping,
//! checked against the graph through a scanner rather than substring greps.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt;

use cfglib::{
    DotEdgeAttributes, DotEdgeStyle, DotRankDir, DotStyle, EdgeId, FilteredEdges, Graph, IdTag,
    KeyedGraph, Label, NodeId, Reversed, bind_edge_attributes, display_label, no_label,
    plain_edge_attributes, to_dot, write_dot,
};

use crate::dot_scanner::{Dot, scan};
use crate::golden::assert_golden;
use crate::labels::{DOT_ALPHABET, strings_over};
use crate::sink::assert_failure_is_reported_at_every_length;

/// `node_prefix` followed by the dense index, the way DOT names a node.
fn identifier(prefix: &str, index: usize) -> String {
    format!("{prefix}{index}")
}

/// The scanned DOT checked against the store it was drawn from: the live
/// nodes (and only those) with the lines of their payload, and every live edge
/// in insertion order, parallel edges included.
fn assert_draws_store<N: fmt::Display, E>(graph: &Graph<N, E>, dot: &Dot, prefix: &str) {
    let expected_ids: Vec<_> = graph
        .node_ids()
        .map(|node| identifier(prefix, node.index()))
        .collect();
    assert_eq!(dot.node_ids(), expected_ids);
    for (node, scanned) in graph.node_ids().zip(&dot.nodes) {
        let text = graph.node(node).to_string();
        let expected = (!text.is_empty()).then(|| text.split('\n').map(str::to_owned).collect());
        assert_eq!(scanned.label_lines(), expected, "label of {}", scanned.id);
    }
    let expected_edges: Vec<_> = graph
        .edge_ids()
        .map(|id| {
            let edge = graph.edge(id);
            (
                identifier(prefix, edge.source().index()),
                identifier(prefix, edge.target().index()),
            )
        })
        .collect();
    let drawn: Vec<_> = dot
        .edge_endpoints()
        .into_iter()
        .map(|(source, target)| (source.to_owned(), target.to_owned()))
        .collect();
    assert_eq!(drawn, expected_edges);
}

/// A module's call graph: a removed function in the middle of the numbering,
/// a call made twice, a recursive call, and a call into the removed function.
fn call_graph() -> Graph<&'static str, u32> {
    let mut graph = Graph::new();
    let main = graph.add_node("main");
    let stale = graph.add_node("stale");
    let parse = graph.add_node("parse");
    let emit = graph.add_node("emit");
    graph.add_edge(main, parse, 3);
    graph.add_edge(main, parse, 1);
    graph.add_edge(parse, emit, 2);
    graph.add_edge(emit, main, 1);
    graph.add_edge(parse, parse, 4);
    graph.add_edge(main, stale, 1);
    graph.remove_node(stale);
    graph
}

#[test]
fn nodes_and_edges_match_the_store_including_holes_and_parallel_edges() {
    let graph = call_graph();
    let dot = scan(&graph.to_dot());

    assert_eq!(dot.name, "view");
    assert_eq!(dot.nodes.len(), graph.node_count());
    assert_eq!(dot.edges.len(), graph.edge_count());
    assert_eq!(
        dot.node_ids(),
        ["n0", "n2", "n3"],
        "the removed node is not drawn"
    );
    assert_draws_store(&graph, &dot, "n");
    assert!(
        dot.edges.iter().all(|edge| edge.attributes.is_empty()),
        "the default store style draws edges undecorated"
    );
}

#[test]
fn an_empty_store_is_an_empty_digraph() {
    let dot = scan(&Graph::<String, ()>::new().to_dot());
    assert!(dot.nodes.is_empty());
    assert!(dot.edges.is_empty());
}

#[test]
fn plain_style_draws_identifiers_and_edges_without_attributes() {
    let graph = call_graph();
    let style = DotStyle::<Label<NodeId, &str>, DotEdgeStyle<EdgeId, u32>>::plain();
    let dot = scan(&to_dot(&graph, &style));

    assert_eq!(dot.node_ids(), ["n0", "n2", "n3"]);
    assert!(dot.nodes.iter().all(|node| node.attributes.is_empty()));
    assert!(dot.edges.iter().all(|edge| edge.attributes.is_empty()));
    assert_eq!(dot.edges.len(), graph.edge_count());
}

#[test]
fn name_prefix_and_rank_direction_come_from_the_style() {
    let graph = call_graph();
    let base = DotStyle::new(
        no_label as Label<NodeId, &str>,
        plain_edge_attributes as DotEdgeStyle<EdgeId, u32>,
    );

    let default = scan(&to_dot(&graph, &base));
    assert_eq!(
        default.graph_attributes["rankdir"],
        DotRankDir::default().attribute()
    );
    assert_eq!(default.graph_attributes["rankdir"], "TB");

    for (direction, expected) in [
        (DotRankDir::TopToBottom, "TB"),
        (DotRankDir::LeftToRight, "LR"),
        (DotRankDir::BottomToTop, "BT"),
        (DotRankDir::RightToLeft, "RL"),
    ] {
        let styled = base.named("callgraph").node_prefix("fn").rankdir(direction);
        let dot = scan(&to_dot(&graph, &styled));
        assert_eq!(dot.name, "callgraph");
        assert_eq!(dot.graph_attributes["rankdir"], expected);
        assert_eq!(direction.attribute(), expected);
        assert_eq!(dot.node_ids(), ["fn0", "fn2", "fn3"]);
        assert_eq!(dot.edge_endpoints()[0], ("fn0", "fn2"));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct SymbolTag;

impl IdTag for SymbolTag {
    const PREFIX: &'static str = "sym";
}

#[test]
fn a_tagged_store_names_its_nodes_by_the_tag_prefix() {
    let mut graph = Graph::<&str, (), SymbolTag, SymbolTag>::tagged();
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    graph.add_edge(a, b, ());

    let dot = scan(&graph.to_dot());
    assert_eq!(dot.node_ids(), ["sym0", "sym1"]);
    assert_eq!(dot.edge_endpoints(), [("sym0", "sym1")]);
}

#[test]
fn every_short_string_is_escaped_as_a_node_label() {
    let labels = strings_over(&DOT_ALPHABET, 3);
    let mut graph = Graph::<String, ()>::new();
    for label in &labels {
        graph.add_node(label.clone());
    }

    let dot = scan(&graph.to_dot());
    assert_draws_store(&graph, &dot, "n");
}

#[test]
fn carriage_returns_do_not_split_or_leak_into_a_label() {
    let mut graph = Graph::<&str, ()>::new();
    graph.add_node("dos\r\nlines");
    let dot = scan(&graph.to_dot());
    assert_eq!(
        dot.nodes[0].label_lines(),
        Some(vec!["dos".to_owned(), "lines".to_owned()])
    );
}

#[test]
fn every_short_string_is_escaped_as_an_edge_label() {
    let labels = strings_over(&DOT_ALPHABET, 3);
    let mut graph = Graph::<(), String>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    for label in &labels {
        graph.add_edge(a, b, label.clone());
    }
    let style = DotStyle::new(
        no_label as Label<NodeId, ()>,
        bind_edge_attributes::<EdgeId, String, _>(|_, label| DotEdgeAttributes {
            label: Cow::Borrowed(label),
            ..DotEdgeAttributes::plain()
        }),
    );

    let dot = scan(&to_dot(&graph, &style));
    assert_eq!(dot.edges.len(), labels.len());
    for (label, edge) in labels.iter().zip(&dot.edges) {
        let expected = (!label.is_empty()).then(|| label.clone());
        assert_eq!(edge.label(), expected, "label {label:?}");
    }
}

#[test]
fn each_attribute_is_written_only_when_it_is_set() {
    // One edge per subset of the four attributes.
    let mut graph = Graph::<(), u8>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    for subset in 0..16_u8 {
        graph.add_edge(a, b, subset);
    }
    let style = DotStyle::new(
        no_label as Label<NodeId, ()>,
        bind_edge_attributes::<EdgeId, u8, _>(|_, subset| DotEdgeAttributes {
            label: if subset & 1 != 0 {
                Cow::Borrowed("say \"x\"")
            } else {
                Cow::Borrowed("")
            },
            color: (subset & 2 != 0).then_some("green4"),
            style: (subset & 4 != 0).then_some("dashed"),
            penwidth: (subset & 8 != 0).then_some(2.5),
        }),
    );

    let dot = scan(&to_dot(&graph, &style));
    for (subset, edge) in (0..16_u8).zip(&dot.edges) {
        let keys: BTreeSet<_> = edge.attributes.keys().map(String::as_str).collect();
        let expected: BTreeSet<_> = [(1, "label"), (2, "color"), (4, "style"), (8, "penwidth")]
            .into_iter()
            .filter(|(bit, _)| subset & bit != 0)
            .map(|(_, key)| key)
            .collect();
        assert_eq!(keys, expected, "subset {subset:#06b}");
        if subset & 1 != 0 {
            assert_eq!(edge.label().as_deref(), Some("say \"x\""));
        }
        if subset & 2 != 0 {
            assert_eq!(edge.bare("color"), Some("green4"));
        }
        if subset & 4 != 0 {
            assert_eq!(edge.bare("style"), Some("dashed"));
        }
        if subset & 8 != 0 {
            let width: f64 = edge.bare("penwidth").unwrap().parse().unwrap();
            assert!((width - 2.5).abs() < 1e-9, "{width}");
        }
    }
}

#[test]
fn edge_attributes_report_whether_they_add_anything() {
    assert!(DotEdgeAttributes::plain().is_plain());
    assert!(DotEdgeAttributes::default().is_plain());
    let labeled = DotEdgeAttributes {
        label: Cow::Borrowed("x"),
        ..DotEdgeAttributes::plain()
    };
    assert!(!labeled.is_plain());
    for decorated in [
        DotEdgeAttributes {
            color: Some("red"),
            ..DotEdgeAttributes::plain()
        },
        DotEdgeAttributes {
            style: Some("bold"),
            ..DotEdgeAttributes::plain()
        },
        DotEdgeAttributes {
            penwidth: Some(1.0),
            ..DotEdgeAttributes::plain()
        },
    ] {
        assert!(!decorated.is_plain());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallKind {
    Direct,
    Indirect,
    Tail,
}

#[derive(Debug, Clone, Copy)]
struct Call {
    kind: CallKind,
    count: u32,
}

fn call_attributes(_: EdgeId, call: &Call) -> DotEdgeAttributes<'_> {
    match call.kind {
        CallKind::Direct => DotEdgeAttributes {
            label: Cow::Owned(format!("x{}", call.count)),
            ..DotEdgeAttributes::plain()
        },
        CallKind::Indirect => DotEdgeAttributes {
            label: Cow::Borrowed("via vtable"),
            color: Some("purple"),
            style: Some("dashed"),
            penwidth: None,
        },
        CallKind::Tail => DotEdgeAttributes {
            label: Cow::Borrowed(""),
            color: Some("blue"),
            style: Some("bold"),
            penwidth: Some(2.5),
        },
    }
}

/// A call graph with awkward names, one edge of each call kind, and the
/// bookkeeping a decorating style reads from the payload.
fn decorated_call_graph() -> Graph<String, Call> {
    let mut graph = Graph::new();
    let main = graph.add_node("main\nentry point".to_owned());
    let parse = graph.add_node("parse \"input\"".to_owned());
    let path = graph.add_node("C:\\temp\\lib".to_owned());
    let silent = graph.add_node(String::new());
    graph.add_edge(
        main,
        parse,
        Call {
            kind: CallKind::Direct,
            count: 3,
        },
    );
    graph.add_edge(
        parse,
        path,
        Call {
            kind: CallKind::Indirect,
            count: 1,
        },
    );
    graph.add_edge(
        path,
        silent,
        Call {
            kind: CallKind::Tail,
            count: 9,
        },
    );
    graph
}

fn call_style() -> DotStyle<Label<NodeId, String>, DotEdgeStyle<EdgeId, Call>> {
    DotStyle::new(
        display_label as Label<NodeId, String>,
        call_attributes as DotEdgeStyle<EdgeId, Call>,
    )
    .named("calls")
    .node_prefix("fn")
    .rankdir(DotRankDir::LeftToRight)
}

#[test]
fn a_decorated_call_graph_is_drawn_with_its_hooks() {
    assert_golden(
        "text-dot/store-call-graph.dot",
        &to_dot(&decorated_call_graph(), &call_style()),
    );
}

#[test]
fn hooks_decide_labels_and_attributes_per_node_and_edge() {
    let graph = decorated_call_graph();
    let dot = scan(&to_dot(&graph, &call_style()));

    assert_eq!(dot.name, "calls");
    assert_eq!(dot.node_ids(), ["fn0", "fn1", "fn2", "fn3"]);
    assert_eq!(
        dot.nodes[0].label_lines(),
        Some(vec!["main".to_owned(), "entry point".to_owned()])
    );
    assert_eq!(
        dot.nodes[1].label_lines(),
        Some(vec!["parse \"input\"".to_owned()])
    );
    assert_eq!(
        dot.nodes[2].label_lines(),
        Some(vec!["C:\\temp\\lib".to_owned()])
    );
    assert_eq!(
        dot.nodes[3].label_lines(),
        None,
        "an empty label draws no label attribute"
    );

    assert_eq!(dot.edges[0].label().as_deref(), Some("x3"));
    assert_eq!(dot.edges[0].bare("color"), None);
    assert_eq!(dot.edges[1].label().as_deref(), Some("via vtable"));
    assert_eq!(dot.edges[1].bare("color"), Some("purple"));
    assert_eq!(dot.edges[1].bare("style"), Some("dashed"));
    assert_eq!(dot.edges[2].label(), None);
    assert_eq!(dot.edges[2].bare("style"), Some("bold"));
    assert_eq!(dot.edges[2].bare("penwidth"), Some("2.5"));
}

#[test]
fn an_edge_hook_closure_sees_identity_and_payload() {
    let graph = call_graph();
    let hot: BTreeSet<EdgeId> = [EdgeId::from_index(1), EdgeId::from_index(4)].into();
    let hook = bind_edge_attributes::<EdgeId, u32, _>(|id, count| DotEdgeAttributes {
        label: Cow::Owned(format!("e{}x{count}", id.index())),
        style: hot.contains(&id).then_some("bold"),
        ..DotEdgeAttributes::plain()
    });
    let style = DotStyle::new(no_label as Label<NodeId, &str>, hook);

    let dot = scan(&to_dot(&graph, &style));
    for (id, edge) in graph.edge_ids().zip(&dot.edges) {
        let expected = format!("e{}x{}", id.index(), graph.edge(id).payload());
        assert_eq!(edge.label().as_deref(), Some(expected.as_str()));
        assert_eq!(edge.bare("style"), hot.contains(&id).then_some("bold"));
    }
}

#[test]
fn a_reversed_view_is_drawn_with_every_edge_flipped() {
    let graph = call_graph();
    let forward = scan(&graph.to_dot());
    let style = DotStyle::new(
        display_label as Label<NodeId, &str>,
        plain_edge_attributes as DotEdgeStyle<EdgeId, u32>,
    );
    let reversed = scan(&to_dot(&Reversed::new(&graph), &style));

    assert_eq!(reversed.node_ids(), forward.node_ids());
    let expected: Vec<_> = forward
        .edge_endpoints()
        .into_iter()
        .map(|(source, target)| (target, source))
        .collect();
    assert_eq!(reversed.edge_endpoints(), expected);
}

#[test]
fn a_filtered_view_is_drawn_without_the_edges_it_rejects() {
    let graph = call_graph();
    let heavy = FilteredEdges::new(&graph, |_, count: &u32| *count >= 2);
    let style = DotStyle::new(
        display_label as Label<NodeId, &str>,
        plain_edge_attributes as DotEdgeStyle<EdgeId, u32>,
    );

    let dot = scan(&to_dot(&heavy, &style));
    assert_eq!(
        dot.nodes.len(),
        graph.node_count(),
        "filtering drops edges, not nodes"
    );
    assert_eq!(
        dot.edge_endpoints(),
        [("n0", "n2"), ("n2", "n3"), ("n2", "n2")],
        "calls counted at least twice: main to parse, parse to emit, parse to itself"
    );
}

#[test]
fn a_keyed_graph_is_drawn_by_the_same_writer() {
    let mut graph = KeyedGraph::<String, String, u32>::new();
    for (importer, imported) in [("pkg::b", "pkg::a"), ("pkg::c", "pkg::a")] {
        let from = graph.intern(&importer.to_owned());
        let to = graph.intern(&imported.to_owned());
        graph.add_edge(from, to, 1);
    }
    let style = DotStyle::new(
        display_label as Label<NodeId, String>,
        plain_edge_attributes as DotEdgeStyle<EdgeId, u32>,
    );

    let dot = scan(&to_dot(&graph, &style));
    assert_eq!(dot.node_ids(), ["n0", "n1", "n2"]);
    assert_eq!(dot.nodes[1].label_lines(), Some(vec!["pkg::a".to_owned()]));
    assert_eq!(dot.edge_endpoints(), [("n0", "n1"), ("n2", "n1")]);
}

#[test]
fn write_dot_streams_the_same_text_to_any_sink() {
    let graph = decorated_call_graph();
    let style = call_style();
    let mut streamed = String::new();
    write_dot(&graph, &mut streamed, &style).expect("a String never fails");
    assert_eq!(streamed, to_dot(&graph, &style));

    assert_failure_is_reported_at_every_length(&streamed, |sink| write_dot(&graph, sink, &style));

    let store = call_graph();
    let mut default_streamed = String::new();
    store
        .write_dot(&mut default_streamed)
        .expect("a String never fails");
    assert_eq!(default_streamed, store.to_dot());
    assert_failure_is_reported_at_every_length(&default_streamed, |sink| store.write_dot(sink));
}
