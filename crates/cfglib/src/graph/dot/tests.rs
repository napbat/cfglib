use crate::graph::store::Graph;
use crate::test_util::golden::assert_golden;
use crate::{DotRankDir, DotStyle, to_dot};

#[test]
fn a_store_renders_its_payloads_and_escapes_them() {
    let mut graph = Graph::new();
    let quoted = graph.add_node("say \"hi\"\nback\\slash");
    let plain = graph.add_node("plain");
    graph.add_edge(quoted, plain, ());

    assert_golden("dot/store-escaped-payloads.dot", &graph.to_dot());
}

#[test]
fn the_plain_style_draws_topology_only() {
    let mut graph = Graph::new();
    let first = graph.add_node("ignored");
    let second = graph.add_node("also ignored");
    graph.add_edge(first, second, "ignored too");

    let dot = to_dot(
        &graph,
        &DotStyle::plain()
            .named("topology")
            .rankdir(DotRankDir::LeftToRight),
    );
    assert_golden("dot/store-topology-only.dot", &dot);
}
