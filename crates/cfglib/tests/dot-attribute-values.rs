//! Integration tests: every value the DOT writer emits is spelled the way DOT
//! reads it, whatever the consumer's hooks and tags hand back.

use std::borrow::Cow;

use cfglib::{
    Cfg, DotEdgeAttributes, DotStyle, EdgeId, EdgeKind, EdgeTag, Graph, IdTag, Label, NodeId,
    bind_edge_attributes, no_label, to_dot,
};

/// The one edge of a two-node graph, drawn with `attributes`.
fn drawn_edge(attributes: DotEdgeAttributes<'static>) -> String {
    let mut graph = Graph::<(), ()>::new();
    let source = graph.add_node(());
    let target = graph.add_node(());
    graph.add_edge(source, target, ());
    let style = DotStyle::new(
        no_label as Label<NodeId, ()>,
        bind_edge_attributes::<EdgeId, (), _>(move |_, ()| attributes.clone()),
    );
    to_dot(&graph, &style)
        .lines()
        .find(|line| line.contains("->"))
        .expect("the graph has an edge")
        .trim()
        .to_owned()
}

#[test]
fn a_color_or_style_is_quoted_exactly_when_dot_requires_it() {
    // (value, spelling): a DOT identifier or numeral stands bare, anything else
    // is a quoted string, and so is a word DOT reserves.
    let cases = [
        ("green4", "green4"),
        ("dashed", "dashed"),
        ("_private", "_private"),
        ("0.5", "0.5"),
        (".5", ".5"),
        ("-3", "-3"),
        ("#ff0000", "\"#ff0000\""),
        ("#ff000080", "\"#ff000080\""),
        ("/blues9/3", "\"/blues9/3\""),
        ("0.650 0.200 1.000", "\"0.650 0.200 1.000\""),
        ("dashed,bold", "\"dashed,bold\""),
        ("setlinewidth(2)", "\"setlinewidth(2)\""),
        ("red:blue", "\"red:blue\""),
        ("9lives", "\"9lives\""),
        ("1e3", "\"1e3\""),
        ("1.2.3", "\"1.2.3\""),
        ("-", "\"-\""),
        (".", "\".\""),
        ("", "\"\""),
        ("node", "\"node\""),
        ("EDGE", "\"EDGE\""),
        ("Subgraph", "\"Subgraph\""),
        ("say \"hi\"", "\"say \\\"hi\\\"\""),
        ("back\\slash", "\"back\\\\slash\""),
        ("two\nlines", "\"two\\nlines\""),
    ];
    for (value, spelling) in cases {
        let value: &'static str = value;
        for (key, attributes) in [
            (
                "color",
                DotEdgeAttributes {
                    color: Some(value),
                    ..DotEdgeAttributes::plain()
                },
            ),
            (
                "style",
                DotEdgeAttributes {
                    style: Some(value),
                    ..DotEdgeAttributes::plain()
                },
            ),
        ] {
            assert_eq!(
                drawn_edge(attributes),
                format!("n0 -> n1 [{key}={spelling}];"),
                "{key} {value:?}"
            );
        }
    }
}

#[test]
fn a_non_finite_pen_width_is_not_written() {
    for width in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let only_width = DotEdgeAttributes {
            penwidth: Some(width),
            ..DotEdgeAttributes::plain()
        };
        assert!(only_width.is_plain(), "{width}: it adds nothing");
        assert_eq!(drawn_edge(only_width), "n0 -> n1;", "{width}");

        let with_color = DotEdgeAttributes {
            color: Some("red"),
            penwidth: Some(width),
            ..DotEdgeAttributes::plain()
        };
        assert_eq!(drawn_edge(with_color), "n0 -> n1 [color=red];", "{width}");
    }
    let finite = DotEdgeAttributes {
        penwidth: Some(2.5),
        ..DotEdgeAttributes::plain()
    };
    assert_eq!(drawn_edge(finite), "n0 -> n1 [penwidth=2.5];");
}

#[test]
fn a_cfg_edge_with_an_unusable_weight_is_drawn_at_the_default_width() {
    for weight in [f64::NAN, f64::INFINITY] {
        let mut cfg = Cfg::<u32>::new();
        let exit = cfg.new_block();
        let edge = cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);
        cfg.edge_mut(edge).set_weight(Some(weight));

        let dot = cfg.to_dot_with(|_| Cow::Borrowed(""));
        assert!(!dot.contains("penwidth"), "{weight}: {dot}");
        assert!(!dot.contains("NaN=") && !dot.contains("inf]"), "{dot}");
        assert!(dot.contains(&format!("label=\"({weight:.2})\"")), "{dot}");
    }
}

/// A node tag whose prefix is not a DOT identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DashedTag;

impl IdTag for DashedTag {
    const PREFIX: &'static str = "my-node";
}

#[test]
fn a_node_prefix_that_is_not_a_dot_identifier_is_quoted() {
    let mut graph = Graph::<&str, (), DashedTag, EdgeTag>::tagged();
    let first = graph.add_node("first");
    let second = graph.add_node("second");
    graph.add_edge(first, second, ());

    let dot = graph.to_dot();
    assert!(dot.contains("\"my-node0\" [label=\"first\\l\"];"), "{dot}");
    assert!(dot.contains("\"my-node0\" -> \"my-node1\";"), "{dot}");
}

#[test]
fn a_graph_name_that_is_not_a_dot_identifier_is_quoted() {
    let graph = Graph::<(), ()>::new();
    let style = DotStyle::<Label<NodeId, ()>, cfglib::DotEdgeStyle<EdgeId, ()>>::plain()
        .named("call graph");
    let dot = to_dot(&graph, &style);
    assert!(dot.starts_with("digraph \"call graph\" {\n"), "{dot}");
}
