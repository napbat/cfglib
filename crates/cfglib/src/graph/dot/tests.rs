use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::store::Graph;
use crate::test_util::golden::assert_golden;
use crate::test_util::{MockInst, ff};
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
fn a_cfg_renders_one_line_per_instruction() {
    let mut cfg = Cfg::new();
    let second = cfg.new_block();
    cfg.block_mut(cfg.entry())
        .instructions_mut()
        .push(ff("entry_inst"));
    cfg.block_mut(second)
        .instructions_mut()
        .push(ff("second_inst"));
    cfg.add_edge(cfg.entry(), second, EdgeKind::Fallthrough);

    assert_golden("dot/cfg-linear.dot", &cfg.to_dot());
}

#[test]
fn conditional_arms_are_colored_and_labeled_by_kind() {
    let mut cfg = Cfg::new();
    let taken = cfg.new_block();
    let fallen = cfg.new_block();
    cfg.block_mut(cfg.entry()).instructions_mut().push(ff("br"));
    cfg.add_edge(cfg.entry(), taken, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), fallen, EdgeKind::ConditionalFalse);

    assert_golden("dot/cfg-conditional.dot", &cfg.to_dot());
}

#[test]
fn an_empty_block_says_so() {
    let cfg: Cfg<MockInst> = Cfg::new();

    assert_golden("dot/cfg-empty.dot", &cfg.to_dot());
}

#[test]
fn a_weighted_edge_is_labeled_and_drawn_thicker() {
    let mut cfg = Cfg::new();
    let target = cfg.new_block();
    cfg.block_mut(cfg.entry()).instructions_mut().push(ff("a"));
    let edge = cfg.add_edge(cfg.entry(), target, EdgeKind::Fallthrough);
    cfg.edge_mut(edge).set_weight(Some(0.75));

    assert_golden("dot/cfg-weighted.dot", &cfg.to_dot());
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
