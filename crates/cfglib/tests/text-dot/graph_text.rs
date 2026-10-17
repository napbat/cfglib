//! The line-oriented text form of a code-intelligence graph: written for a
//! human or a diff, parsed back by a tool.

use std::borrow::Cow;
use std::fmt;

use cfglib::{
    EdgeId, FilteredEdges, Graph, IdTag, KeyedGraph, Label, NodeId, Reversed, TextError,
    TextErrorKind, TextStyle, bind_label, display_label, parse_text, to_text, write_text,
};

use crate::golden::assert_golden;
use crate::labels::{TEXT_ALPHABET, strings_over};
use crate::sink::assert_failure_is_reported_at_every_length;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Symbol {
    Module(&'static str),
    Function(&'static str),
    Type(&'static str),
}

impl fmt::Display for Symbol {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Module(name) => write!(formatter, "mod {name}"),
            Self::Function(name) => write!(formatter, "fn {name}"),
            Self::Type(name) => write!(formatter, "type {name}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Relation {
    Defines,
    Calls,
    Reads,
}

impl fmt::Display for Relation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Defines => "defines",
            Self::Calls => "calls",
            Self::Reads => "reads",
        })
    }
}

fn symbol_label(_: NodeId, symbol: &Symbol) -> Cow<'_, str> {
    Cow::Owned(symbol.to_string())
}

fn relation_label(_: EdgeId, relation: &Relation) -> Cow<'_, str> {
    Cow::Owned(relation.to_string())
}

type SymbolGraph = Graph<Symbol, Relation>;

/// An analyzed module: one removed symbol in the middle of the numbering, a
/// call made twice, a recursive call, and an edge that was added and removed.
fn symbol_graph() -> SymbolGraph {
    let mut graph = SymbolGraph::new();
    let app = graph.add_node(Symbol::Module("app"));
    let main = graph.add_node(Symbol::Function("main"));
    let stale = graph.add_node(Symbol::Function("stale"));
    let load = graph.add_node(Symbol::Function("load"));
    let config = graph.add_node(Symbol::Type("Config"));

    graph.add_edge(app, main, Relation::Defines);
    graph.add_edge(app, stale, Relation::Defines);
    graph.add_edge(app, load, Relation::Defines);
    graph.add_edge(app, config, Relation::Defines);
    graph.add_edge(main, load, Relation::Calls);
    graph.add_edge(main, load, Relation::Calls);
    graph.add_edge(stale, main, Relation::Calls);
    let withdrawn = graph.add_edge(load, main, Relation::Calls);
    graph.add_edge(load, config, Relation::Reads);
    graph.add_edge(main, main, Relation::Calls);

    assert!(graph.remove_edge(withdrawn));
    assert!(graph.remove_node(stale));
    graph
}

/// The text form of `graph` read back, checked against the graph itself:
/// every live node keeps its index and label, every removed node stays a
/// hole, and the live edges come back in insertion order.
fn assert_isomorphic<N: fmt::Display, E: fmt::Display>(
    original: &Graph<N, E>,
    parsed: &Graph<String, String>,
) {
    assert_eq!(
        parsed.node_ids().collect::<Vec<_>>(),
        original.node_ids().collect::<Vec<_>>(),
        "the same node indices are live"
    );
    for node in original.node_ids() {
        assert_eq!(parsed.node(node), &original.node(node).to_string());
    }
    let original_edges: Vec<_> = original
        .edge_ids()
        .map(|id| {
            let edge = original.edge(id);
            (edge.source(), edge.target(), edge.payload().to_string())
        })
        .collect();
    let parsed_edges: Vec<_> = parsed
        .edge_ids()
        .map(|id| {
            let edge = parsed.edge(id);
            (edge.source(), edge.target(), edge.payload().clone())
        })
        .collect();
    assert_eq!(parsed_edges, original_edges);
}

#[test]
fn a_symbol_graph_is_written_one_fact_per_line_with_the_hole_visible() {
    assert_golden("text-dot/graph-symbols.txt", &symbol_graph().to_text());
}

#[test]
fn parsing_a_written_graph_reproduces_indices_labels_and_edge_order() {
    let graph = symbol_graph();
    let parsed = parse_text(&graph.to_text()).expect("the writer emits its own grammar");

    assert_isomorphic(&graph, &parsed);
    assert_eq!(parsed.node_count(), 4);
    assert_eq!(
        parsed.node_bound(),
        5,
        "the removed symbol keeps its slot, so every later index keeps its meaning"
    );
    assert!(!parsed.contains_node(NodeId::from_index(2)));
}

#[test]
fn writing_parsing_and_writing_again_is_a_fixed_point() {
    let once = symbol_graph().to_text();
    let twice = parse_text(&once).expect("its own grammar").to_text();
    assert_eq!(once, twice);
}

#[test]
fn a_removed_first_node_leaves_a_leading_hole_that_parses_back() {
    let mut graph = Graph::<&str, &str>::new();
    let first = graph.add_node("first");
    let second = graph.add_node("second");
    let third = graph.add_node("third");
    graph.add_edge(second, third, "next");
    graph.add_edge(first, second, "next");
    graph.remove_node(first);

    let text = graph.to_text();
    assert_eq!(text, "n1 second\nn2 third\nn1 -> n2 next\n");
    let parsed = parse_text(&text).expect("its own grammar");
    assert_isomorphic(&graph, &parsed);
    assert_eq!(parsed.node_bound(), 3);
}

#[test]
fn a_removed_last_node_writes_nothing_and_moves_no_surviving_index() {
    let mut graph = Graph::<&str, &str>::new();
    let kept = graph.add_node("kept");
    let last = graph.add_node("last");
    graph.add_edge(kept, last, "link");
    graph.remove_node(last);

    let text = graph.to_text();
    assert_eq!(text, "n0 kept\n");
    let parsed = parse_text(&text).expect("its own grammar");
    assert_eq!(parsed.node_ids().collect::<Vec<_>>(), [kept]);
    assert_eq!(parsed.edge_count(), 0);
}

#[test]
fn removed_edges_leave_no_line_and_the_rest_keep_insertion_order() {
    let mut graph = Graph::<&str, &str>::new();
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let first = graph.add_edge(a, b, "first");
    graph.add_edge(b, a, "second");
    graph.add_edge(a, b, "third");
    graph.remove_edge(first);

    let parsed = parse_text(&graph.to_text()).expect("its own grammar");
    let labels: Vec<_> = parsed
        .edge_ids()
        .map(|id| parsed.edge(id).payload().as_str())
        .collect();
    assert_eq!(labels, ["second", "third"]);
    assert_eq!(parsed.edge_count(), graph.edge_count());
}

#[test]
fn parallel_edges_and_self_loops_keep_their_multiplicity() {
    let mut graph = Graph::<&str, u32>::new();
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    for weight in 0..3 {
        graph.add_edge(a, b, weight);
    }
    graph.add_edge(a, a, 7);
    graph.add_edge(a, a, 8);

    let parsed = parse_text(&graph.to_text()).expect("its own grammar");
    assert_eq!(parsed.successors(a).collect::<Vec<_>>(), [b, b, b, a, a]);
    assert_eq!(parsed.predecessors(a).collect::<Vec<_>>(), [a, a]);
    assert_isomorphic(&graph, &parsed);
}

#[test]
fn an_empty_graph_is_empty_text_and_empty_text_is_an_empty_graph() {
    let graph = Graph::<String, String>::new();
    assert_eq!(graph.to_text(), "");

    let parsed = parse_text("").expect("nothing to read is not an error");
    assert!(parsed.is_empty());
    assert_eq!(parsed.edge_count(), 0);
}

#[test]
fn every_short_label_round_trips_as_node_text_and_as_edge_text() {
    let labels = strings_over(&TEXT_ALPHABET, 3);
    let mut graph = Graph::<String, String>::new();
    let mut previous = None;
    for label in &labels {
        let node = graph.add_node(label.clone());
        if let Some(previous) = previous {
            graph.add_edge(previous, node, label.clone());
        }
        previous = Some(node);
    }

    let text = graph.to_text();
    assert_eq!(
        text.lines().count(),
        graph.node_count() + graph.edge_count(),
        "an escaped label never spills onto a second line"
    );
    let parsed = parse_text(&text).expect("the writer escapes everything it must");

    let damaged: Vec<_> = graph
        .node_ids()
        .filter(|&node| parsed.node(node) != graph.node(node))
        .map(|node| graph.node(node))
        .take(5)
        .collect();
    assert!(damaged.is_empty(), "node labels changed: {damaged:?}");
    let damaged: Vec<_> = graph
        .edge_ids()
        .filter(|&edge| parsed.edge(edge).payload() != graph.edge(edge).payload())
        .map(|edge| graph.edge(edge).payload())
        .take(5)
        .collect();
    assert!(damaged.is_empty(), "edge labels changed: {damaged:?}");
    assert_eq!(parsed.to_text(), text);
}

#[test]
fn a_label_that_looks_like_an_arrow_is_still_a_node_label() {
    let mut graph = Graph::<&str, &str>::new();
    let a = graph.add_node("-> n0 spoof");
    let b = graph.add_node("-");
    graph.add_edge(a, b, "-> n1 spoof");

    let parsed = parse_text(&graph.to_text()).expect("its own grammar");
    assert_eq!(parsed.node_count(), 2, "no label was read as structure");
    assert_eq!(parsed.edge_count(), 1);
    assert_eq!(parsed.node(a), "-> n0 spoof");
    assert_eq!(parsed.node(b), "-");
    assert_eq!(
        parsed.edge(parsed.edge_ids().next().unwrap()).payload(),
        "-> n1 spoof"
    );
}

#[test]
fn plain_style_writes_topology_only() {
    let graph = symbol_graph();
    let style = TextStyle::<Label<NodeId, Symbol>, Label<EdgeId, Relation>>::plain();
    let text = to_text(&graph, &style);

    let parsed = parse_text(&text).expect("a label-free graph is still the grammar");
    assert_eq!(
        parsed.node_ids().collect::<Vec<_>>(),
        graph.node_ids().collect::<Vec<_>>()
    );
    assert!(parsed.nodes().all(String::is_empty));
    assert!(parsed.edges().all(|edge| edge.payload().is_empty()));
    assert_eq!(parsed.edge_count(), graph.edge_count());
    assert!(
        !text.contains("main") && !text.contains("calls"),
        "labels were not requested: {text}"
    );
}

#[test]
fn label_hooks_decide_what_each_line_says() {
    let graph = symbol_graph();
    let width = 3;
    let node_label = bind_label::<NodeId, Symbol, _>(|id, symbol| {
        Cow::Owned(format!("{:0width$}:{symbol}", id.index()))
    });
    let edge_label = bind_label::<EdgeId, Relation, _>(|id, relation| {
        if *relation == Relation::Calls {
            Cow::Owned(format!("call #{}", id.index()))
        } else {
            Cow::Borrowed("")
        }
    });
    let text = to_text(&graph, &TextStyle::new(node_label, edge_label));

    let parsed = parse_text(&text).expect("its own grammar");
    assert_eq!(parsed.node(NodeId::from_index(3)), "003:fn load");
    let call_edges: Vec<_> = parsed
        .edges()
        .map(|edge| edge.payload().as_str())
        .filter(|label| !label.is_empty())
        .collect();
    assert_eq!(call_edges, ["call #4", "call #5", "call #9"]);
}

#[test]
fn write_text_streams_the_same_text_to_any_sink() {
    let graph = symbol_graph();
    let style = TextStyle::new(symbol_label, relation_label);
    let mut streamed = String::new();
    write_text(&graph, &mut streamed, &style).expect("a String never fails");
    assert_eq!(streamed, to_text(&graph, &style));
    assert_eq!(streamed, graph.to_text());

    assert_failure_is_reported_at_every_length(&streamed, |sink| write_text(&graph, sink, &style));
}

#[test]
fn a_reversed_view_is_written_with_every_edge_flipped() {
    let graph = symbol_graph();
    let style = TextStyle::new(symbol_label, relation_label);
    let reversed = to_text(&Reversed::new(&graph), &style);

    let flipped = parse_text(&reversed).expect("its own grammar");
    let forward = parse_text(&graph.to_text()).expect("its own grammar");
    assert_eq!(
        flipped.node_ids().collect::<Vec<_>>(),
        forward.node_ids().collect::<Vec<_>>()
    );
    let endpoints = |graph: &Graph<String, String>| {
        graph
            .edge_ids()
            .map(|id| {
                let edge = graph.edge(id);
                (edge.source(), edge.target(), edge.payload().clone())
            })
            .collect::<Vec<_>>()
    };
    let expected: Vec<_> = endpoints(&forward)
        .into_iter()
        .map(|(source, target, label)| (target, source, label))
        .collect();
    assert_eq!(endpoints(&flipped), expected);
}

#[test]
fn a_filtered_view_is_written_without_the_edges_it_rejects() {
    let graph = symbol_graph();
    let calls_only = FilteredEdges::new(&graph, |_, relation: &Relation| {
        *relation == Relation::Calls
    });
    let text = to_text(&calls_only, &TextStyle::new(symbol_label, relation_label));

    let parsed = parse_text(&text).expect("its own grammar");
    assert_eq!(
        parsed.node_count(),
        graph.node_count(),
        "filtering drops edges, not nodes"
    );
    assert_eq!(
        parsed.edge_count(),
        3,
        "two calls to load and the recursive call"
    );
    assert!(parsed.edges().all(|edge| edge.payload() == "calls"));
}

#[test]
fn a_keyed_graph_is_written_in_minting_order() {
    let mut graph = KeyedGraph::<String, String, String>::new();
    for (importer, imported) in [
        ("pkg::b", "pkg::a"),
        ("pkg::c", "pkg::a"),
        ("pkg::a", "pkg::c"),
    ] {
        let from = graph.intern(&importer.to_owned());
        let to = graph.intern(&imported.to_owned());
        graph.add_edge(from, to, "imports".to_owned());
    }

    let style = TextStyle::new(
        display_label as Label<NodeId, String>,
        display_label as Label<EdgeId, String>,
    );
    assert_eq!(
        to_text(&graph, &style),
        "n0 pkg::b\nn1 pkg::a\nn2 pkg::c\n\
         n0 -> n1 imports\nn2 -> n1 imports\nn1 -> n2 imports\n"
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FunctionTag;

impl IdTag for FunctionTag {
    const PREFIX: &'static str = "fn";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CallTag;

impl IdTag for CallTag {
    const PREFIX: &'static str = "call";
}

#[test]
fn a_tagged_store_is_still_written_with_n_identifiers() {
    let mut graph = Graph::<&str, &str, FunctionTag, CallTag>::tagged();
    let main = graph.add_node("main");
    let helper = graph.add_node("helper");
    let call = graph.add_edge(main, helper, "direct");
    assert_eq!(format!("{main} {call}"), "fn0 call0");

    let text = graph.to_text();
    assert_eq!(text, "n0 main\nn1 helper\nn0 -> n1 direct\n");
    let parsed = parse_text(&text).expect("a plain store reads any tag's text");
    assert_eq!(parsed.node(NodeId::from_index(1)), "helper");
}

#[test]
fn comments_blank_lines_and_crlf_line_endings_are_not_content() {
    let text = "# exported by a tool\r\n\r\nn0 first\r\n   # a note\r\nn1\r\n\r\nn0 -> n1 link\r\n";
    let parsed = parse_text(text).expect("valid text");
    assert_eq!(parsed.node(NodeId::from_index(0)), "first");
    assert_eq!(parsed.node(NodeId::from_index(1)), "");
    assert_eq!(
        parsed.edge(parsed.edge_ids().next().unwrap()).payload(),
        "link"
    );
}

fn first_error(text: &str) -> TextError {
    parse_text(text).expect_err("the text is malformed")
}

#[test]
fn each_malformed_line_is_reported_with_its_number_and_kind() {
    let cases = [
        (
            "n0 a\nn0 again\n",
            2,
            TextErrorKind::NodeOrder { expected: 1 },
        ),
        (
            "n0 a\nn2 b\nn1 late\n",
            3,
            TextErrorKind::NodeOrder { expected: 3 },
        ),
        (
            "n0 a\nn0 -> n3\n",
            2,
            TextErrorKind::UnknownNode { index: 3 },
        ),
        (
            "n0 a\nn1 -> n0\n",
            2,
            TextErrorKind::UnknownNode { index: 1 },
        ),
        ("n0 a\\q\n", 1, TextErrorKind::Escape),
        ("n0 trailing\\\n", 1, TextErrorKind::Escape),
        ("node0 a\n", 1, TextErrorKind::Identifier),
        ("n a\n", 1, TextErrorKind::Identifier),
        ("n0 a\nn0 -> m0\n", 2, TextErrorKind::Identifier),
        ("n0 a\nn0 -> \n", 2, TextErrorKind::Unrecognized),
    ];
    for (text, line, kind) in cases {
        let error = first_error(text);
        assert_eq!((error.line, error.kind), (line, kind), "{text:?}");
    }
}

#[test]
fn only_the_first_malformed_line_is_reported() {
    let error = first_error("n0 ok\nn0 -> n9\nbad line\n");
    assert_eq!(error.line, 2);
}

#[test]
fn a_text_error_is_a_standard_error_that_names_its_line() {
    let error = first_error("n0 a\nn0 -> n4\n");
    let boxed: Box<dyn std::error::Error> = Box::new(error);
    assert!(boxed.to_string().contains('2'), "{boxed}");
}
