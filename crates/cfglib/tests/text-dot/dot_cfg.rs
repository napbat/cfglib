//! DOT output of a control-flow graph: block labels, edge styling by kind and
//! weight, and the writer's reach into consumer styles and views.

use std::borrow::Cow;

use cfglib::{
    BasicBlock, BlockId, Cfg, DotEdgeAttributes, DotEdgeStyle, DotStyle, Edge, EdgeId, EdgeKind,
    FilteredEdges, bind_label, control_flow_edge_attributes, to_dot,
};

use crate::dot_scanner::{Dot, Edge as DrawnEdge, scan};
use crate::fixture::{rich_cfg, try_finally};
use crate::golden::assert_golden;
use crate::instruction::{Isa, Line};
use crate::labels::{DOT_ALPHABET, strings_over};
use crate::sink::assert_failure_is_reported_at_every_length;

fn identifier(prefix: &str, block: BlockId) -> String {
    format!("{prefix}{}", block.index())
}

/// A drawn edge checked against the attributes the public edge hook returns
/// for it: same color, line style, label, and thickness.
fn assert_attributes(drawn: &DrawnEdge, expected: &DotEdgeAttributes<'_>) {
    assert_eq!(drawn.label().unwrap_or_default(), expected.label);
    assert_eq!(drawn.bare("color"), expected.color);
    assert_eq!(drawn.bare("style"), expected.style);
    match (drawn.bare("penwidth"), expected.penwidth) {
        (None, None) => {}
        (Some(text), Some(width)) => {
            let drawn_width: f64 = text.parse().expect("a number");
            assert!(
                (drawn_width - width).abs() <= 0.05,
                "{drawn_width} vs {width}"
            );
        }
        other => panic!("penwidth disagrees: {other:?}"),
    }
}

/// The scanned DOT checked against the CFG it was drawn from: the live blocks
/// and only those, every live edge in insertion order, and each edge dressed
/// by the control-flow hook.
fn assert_draws_cfg<I, E>(cfg: &Cfg<I, E>, dot: &Dot, prefix: &str) {
    let expected_ids: Vec<_> = cfg.block_ids().map(|id| identifier(prefix, id)).collect();
    assert_eq!(dot.node_ids(), expected_ids);
    assert_eq!(dot.edges.len(), cfg.edge_count());
    for (edge, drawn) in cfg.edges().zip(&dot.edges) {
        assert_eq!(drawn.source, identifier(prefix, edge.source()));
        assert_eq!(drawn.target, identifier(prefix, edge.target()));
        assert_attributes(drawn, &control_flow_edge_attributes(edge.id(), edge.data()));
    }
}

#[test]
fn every_edge_kind_is_drawn_with_its_own_color_and_line_style() {
    let mut cfg = Cfg::<Line>::new();
    for kind in EdgeKind::ALL {
        let target = cfg.new_block();
        cfg.add_edge(cfg.entry(), target, kind);
    }
    assert_golden("text-dot/cfg-every-edge-kind.dot", &cfg.to_dot());
}

#[test]
fn a_hot_loop_is_drawn_with_probabilities_and_proportional_width() {
    let mut cfg = Cfg::<Line>::new();
    let header = cfg.new_block();
    let body = cfg.new_block();
    let exit = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(Line::new("i = 0"));
    cfg.block_mut(header).set_label("header");
    cfg.block_mut(header).push(Line::new("i < n"));
    cfg.block_mut(body).push(Line::new("i = i + 1"));
    cfg.block_mut(exit).push(Line::new("return i"));
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_weighted_edge(header, body, EdgeKind::ConditionalTrue, 0.9);
    cfg.add_weighted_edge(header, exit, EdgeKind::ConditionalFalse, 0.1);
    cfg.add_weighted_edge(body, header, EdgeKind::Back, 1.0);

    assert_golden("text-dot/cfg-hot-loop.dot", &cfg.to_dot());
}

#[test]
fn blocks_edges_and_hooks_match_the_cfg_including_holes_and_parallel_edges() {
    let cfg = rich_cfg();
    let dot = scan(&cfg.to_dot());

    assert_eq!(dot.name, "cfg");
    assert_eq!(dot.nodes.len(), cfg.block_count());
    assert_eq!(dot.edges.len(), cfg.edge_count());
    assert_draws_cfg(&cfg, &dot, "bb");
    let ids = dot.node_ids();
    assert!(
        !ids.contains(&"bb3") && !ids.contains(&"bb8"),
        "removed blocks are not drawn: {ids:?}"
    );
}

#[test]
fn each_block_label_names_the_block_and_lists_its_instructions_in_order() {
    let cfg = rich_cfg();
    let dot = scan(&cfg.to_dot());

    for (id, node) in cfg.block_ids().zip(&dot.nodes) {
        let lines = node.label_lines().expect("a CFG block is always labeled");
        let name = id.to_string();
        let at = lines
            .iter()
            .position(|line| *line == name)
            .unwrap_or_else(|| panic!("{name} is not named in {lines:?}"));
        let mut expected = Vec::new();
        for instruction in cfg.block(id).instructions() {
            expected.extend(instruction.0.split('\n').map(str::to_owned));
        }
        if expected.is_empty() {
            assert!(lines.len() > at + 1, "an empty block says so: {lines:?}");
        } else {
            assert_eq!(&lines[at + 1..], expected.as_slice(), "{name}");
        }
        if let Some(label) = cfg.block(id).label() {
            let text = lines.join("\n");
            assert!(text.contains(label), "{label:?} is not in {text:?}");
        }
    }
}

#[test]
fn every_short_instruction_is_escaped_inside_a_block_label() {
    let instructions: Vec<_> = strings_over(&DOT_ALPHABET, 3)
        .into_iter()
        .filter(|text| !text.is_empty())
        .collect();
    let mut cfg = Cfg::<Line>::new();
    for text in &instructions {
        cfg.block_mut(cfg.entry()).push(Line::new(text));
    }

    let dot = scan(&cfg.to_dot());
    let lines = dot.nodes[0].label_lines().expect("a CFG block is labeled");
    assert_eq!(lines[0], "bb0");
    let expected: Vec<_> = instructions
        .iter()
        .flat_map(|text| text.split('\n').map(str::to_owned))
        .collect();
    assert_eq!(&lines[1..], expected.as_slice());
}

#[test]
fn display_instr_text_is_the_label_and_needs_no_flow_classification() {
    let mut cfg = Cfg::<Isa>::new();
    for instruction in [
        Isa::Load {
            destination: 1,
            offset: -8,
        },
        Isa::Return,
    ] {
        cfg.block_mut(cfg.entry()).push(instruction);
    }

    let dot = scan(&cfg.to_dot());
    assert_eq!(
        dot.nodes[0].label_lines(),
        Some(vec![
            "bb0".to_owned(),
            "load r1, [-8]".to_owned(),
            "ret".to_owned()
        ])
    );
}

#[derive(Debug, Clone)]
struct Opaque(u32);

fn opaque_label(op: &Opaque) -> Cow<'_, str> {
    Cow::Owned(format!("op#{}", op.0))
}

#[test]
fn the_bound_free_entry_point_renders_instructions_that_implement_nothing() {
    let mut cfg = Cfg::<Opaque>::new();
    let next = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(Opaque(7));
    cfg.block_mut(next).push(Opaque(8));
    cfg.add_edge(cfg.entry(), next, EdgeKind::Fallthrough);

    let label = opaque_label;
    let dot = scan(&cfg.to_dot_with(label));
    assert_eq!(
        dot.nodes[1].label_lines(),
        Some(vec!["bb1".to_owned(), "op#8".to_owned()])
    );

    let mut streamed = String::new();
    cfg.write_dot_with(&mut streamed, label)
        .expect("a String never fails");
    assert_eq!(streamed, cfg.to_dot_with(label));
    assert_failure_is_reported_at_every_length(&streamed, |sink| cfg.write_dot_with(sink, label));
}

#[test]
fn write_dot_streams_the_same_text_as_to_dot() {
    let cfg = try_finally();
    let mut streamed = String::new();
    cfg.write_dot(&mut streamed).expect("a String never fails");
    assert_eq!(streamed, cfg.to_dot());
    assert_failure_is_reported_at_every_length(&streamed, |sink| cfg.write_dot(sink));
}

#[test]
fn a_consumer_style_over_a_cfg_uses_the_same_writer_and_the_control_flow_hook() {
    let cfg = try_finally();
    let node_label = bind_label::<BlockId, BasicBlock<Line>, _>(|id, block| {
        Cow::Owned(format!("{id} ({} instr)", block.instructions().len()))
    });
    let style = DotStyle::new(
        node_label,
        control_flow_edge_attributes as DotEdgeStyle<EdgeId, Edge>,
    )
    .named("flow")
    .node_prefix("b");

    let dot = scan(&to_dot(&cfg, &style));
    assert_eq!(dot.name, "flow");
    assert_draws_cfg(&cfg, &dot, "b");
    for (id, node) in cfg.block_ids().zip(&dot.nodes) {
        let expected = format!("{id} ({} instr)", cfg.block(id).instructions().len());
        assert_eq!(node.label_lines(), Some(vec![expected]));
    }
}

#[test]
fn a_view_of_the_normal_flow_leaves_the_exceptional_edges_undrawn() {
    let cfg = try_finally();
    let normal = FilteredEdges::new(&cfg, |_, edge: &Edge| !edge.kind().is_exceptional());
    let style = DotStyle::new(
        bind_label::<BlockId, BasicBlock<Line>, _>(|id, _| Cow::Owned(id.to_string())),
        control_flow_edge_attributes as DotEdgeStyle<EdgeId, Edge>,
    )
    .node_prefix("bb");

    let dot = scan(&to_dot(&normal, &style));
    assert_eq!(
        dot.nodes.len(),
        cfg.block_count(),
        "filtering drops edges, not blocks"
    );
    let expected: Vec<_> = cfg
        .edges()
        .filter(|edge| !edge.kind().is_exceptional())
        .map(|edge| {
            (
                identifier("bb", edge.source()),
                identifier("bb", edge.target()),
            )
        })
        .collect();
    let drawn: Vec<_> = dot
        .edge_endpoints()
        .into_iter()
        .map(|(source, target)| (source.to_owned(), target.to_owned()))
        .collect();
    assert_eq!(drawn, expected);
    assert!(
        drawn.len() < cfg.edge_count(),
        "the fixture has an unwind edge to drop"
    );
}

#[test]
fn edges_of_one_kind_look_alike_whatever_their_endpoints_and_payloads() {
    let mut cfg = Cfg::<Line, u32>::with_edge_payload();
    let first = cfg.entry();
    let second = cfg.new_block();
    for kind in EdgeKind::ALL {
        cfg.add_edge_with_payload(first, second, kind, 1);
        cfg.add_edge_with_payload(second, first, kind, 2);
    }

    let attributes: Vec<_> = cfg
        .edges()
        .map(|edge| control_flow_edge_attributes(edge.id(), edge.data()))
        .collect();
    for pair in attributes.chunks(2) {
        assert_eq!(pair[0].color, pair[1].color);
        assert_eq!(pair[0].style, pair[1].style);
        assert_eq!(pair[0].label, pair[1].label);
    }
    for decorated in &attributes {
        assert!(decorated.color.is_some() && decorated.style.is_some());
        assert_eq!(
            decorated.penwidth, None,
            "an unweighted edge has the default width"
        );
    }
}

#[test]
fn a_weighted_edge_names_its_probability_and_the_likelier_it_is_the_thicker() {
    let weights = [0.0, 0.25, 0.5, 0.75, 1.0];
    for kind in EdgeKind::ALL {
        let mut cfg = Cfg::<Line>::new();
        let next = cfg.new_block();
        for weight in weights {
            cfg.add_weighted_edge(cfg.entry(), next, kind, weight);
        }

        let widths: Vec<f64> = cfg
            .edges()
            .zip(weights)
            .map(|(edge, weight)| {
                let attributes = control_flow_edge_attributes(edge.id(), edge.data());
                assert!(
                    attributes.label.contains(&format!("{weight:.2}")),
                    "{kind}: {:?} should name {weight}",
                    attributes.label
                );
                attributes.penwidth.expect("a weighted edge has a width")
            })
            .collect();
        assert!(
            widths.windows(2).all(|pair| pair[0] < pair[1]),
            "{kind}: widths {widths:?} must grow with the weight"
        );
    }
}
