use super::fixtures::BuilderInst;
use super::{
    BTreeSet, BlockId, Cfg, DenseNodeId, DirectedGraph, DominatorTree, EdgeKind, FlowEffect, NodeId,
};

pub(super) fn assert_cfg_shape<I>(cfg: &Cfg<I>, expected_blocks: usize, expected_edges: usize) {
    assert_eq!(
        cfg.num_blocks(),
        expected_blocks,
        "unexpected CFG block count"
    );
    assert_eq!(cfg.num_edges(), expected_edges, "unexpected CFG edge count");
    let verification = cfglib::verify(cfg);
    assert!(
        verification.is_ok(),
        "invalid benchmark CFG: {:?}",
        verification.errors
    );
}

fn assert_directed_shape<N, E>(
    graph: &DirectedGraph<N, E>,
    expected_nodes: usize,
    expected_edges: usize,
) {
    assert_eq!(
        graph.node_count(),
        expected_nodes,
        "unexpected directed-graph node count"
    );
    assert_eq!(
        graph.edge_count(),
        expected_edges,
        "unexpected directed-graph edge count"
    );

    let mut outgoing_count = 0;
    let mut incoming_count = 0;
    for node in graph.node_ids() {
        let outgoing: BTreeSet<_> = graph.outgoing_edges(node).iter().copied().collect();
        let incoming: BTreeSet<_> = graph.incoming_edges(node).iter().copied().collect();
        assert_eq!(
            outgoing.len(),
            graph.outgoing_edges(node).len(),
            "duplicate outgoing edge identity"
        );
        assert_eq!(
            incoming.len(),
            graph.incoming_edges(node).len(),
            "duplicate incoming edge identity"
        );
        for &edge in graph.outgoing_edges(node) {
            assert_eq!(graph.edge(edge).source(), node);
        }
        for &edge in graph.incoming_edges(node) {
            assert_eq!(graph.edge(edge).target(), node);
        }
        outgoing_count += outgoing.len();
        incoming_count += incoming.len();
    }
    assert_eq!(outgoing_count, expected_edges);
    assert_eq!(incoming_count, expected_edges);
    for edge in graph.edges() {
        assert!(graph.outgoing_edges(edge.source()).contains(&edge.id()));
        assert!(graph.incoming_edges(edge.target()).contains(&edge.id()));
    }
}

pub(super) fn assert_dense_permutation<N>(nodes: &[N], expected_count: usize)
where
    N: Copy + DenseNodeId + core::fmt::Debug,
{
    assert_eq!(nodes.len(), expected_count);
    let mut seen = vec![false; expected_count];
    for &node in nodes {
        assert!(
            node.index() < expected_count,
            "node is out of range: {node:?}"
        );
        assert!(!seen[node.index()], "duplicate node in traversal: {node:?}");
        seen[node.index()] = true;
    }
    assert!(seen.into_iter().all(core::convert::identity));
}

fn branchy_edge_count(node_count: usize) -> usize {
    (node_count - 1) + node_count.saturating_sub(2).div_ceil(2) + node_count.saturating_sub(1) / 32
}

fn has_cfg_edge<I>(cfg: &Cfg<I>, source: BlockId, target: BlockId, kind: EdgeKind) -> bool {
    cfg.successor_edges(source).iter().any(|&edge| {
        let edge = cfg.edge(edge);
        edge.target() == target && edge.kind() == kind
    })
}

pub(super) fn has_directed_edge<N, E>(
    graph: &DirectedGraph<N, E>,
    source: NodeId,
    target: NodeId,
) -> bool {
    graph
        .outgoing_edges(source)
        .iter()
        .any(|&edge| graph.edge(edge).target() == target)
}

pub(super) fn assert_branchy_cfg(cfg: &Cfg<u32>, node_count: usize) {
    assert_cfg_shape(cfg, node_count, branchy_edge_count(node_count));
    for index in 0..node_count {
        let block = BlockId::from_raw(index as u32);
        assert_eq!(
            cfg.block(block).instructions(),
            &[index as u32, (index as u32).wrapping_mul(17)]
        );
        if index + 1 < node_count {
            assert!(has_cfg_edge(
                cfg,
                block,
                BlockId::from_raw((index + 1) as u32),
                EdgeKind::Fallthrough
            ));
        }
    }
    for index in (0..node_count.saturating_sub(2)).step_by(2) {
        assert!(has_cfg_edge(
            cfg,
            BlockId::from_raw(index as u32),
            BlockId::from_raw((index + 2) as u32),
            EdgeKind::ConditionalTrue
        ));
    }
    for index in (32..node_count).step_by(32) {
        assert!(has_cfg_edge(
            cfg,
            BlockId::from_raw(index as u32),
            BlockId::from_raw((index - 16) as u32),
            EdgeKind::Back
        ));
    }
}

pub(super) fn assert_branchy_graph(graph: &DirectedGraph<(), ()>, node_count: usize) {
    assert_directed_shape(graph, node_count, branchy_edge_count(node_count));
    for index in 0..node_count - 1 {
        assert!(has_directed_edge(
            graph,
            NodeId::from_index(index),
            NodeId::from_index(index + 1)
        ));
    }
    for index in (0..node_count.saturating_sub(2)).step_by(2) {
        assert!(has_directed_edge(
            graph,
            NodeId::from_index(index),
            NodeId::from_index(index + 2)
        ));
    }
    for index in (32..node_count).step_by(32) {
        assert!(has_directed_edge(
            graph,
            NodeId::from_index(index),
            NodeId::from_index(index - 16)
        ));
    }
}

pub(super) fn assert_builder_cfg(
    cfg: &Cfg<BuilderInst>,
    expected_blocks: usize,
    expected_edges: usize,
    expected_effects: &[(FlowEffect, usize)],
    expected_edge_kinds: &[(EdgeKind, usize)],
) {
    assert_cfg_shape(cfg, expected_blocks, expected_edges);
    assert_eq!(cfg.dfs_preorder().len(), expected_blocks);

    let instructions: Vec<_> = cfg
        .blocks()
        .iter()
        .flat_map(cfglib::BasicBlock::instructions)
        .collect();
    assert_eq!(
        instructions.len(),
        expected_effects.iter().map(|(_, count)| count).sum()
    );
    for &(effect, expected) in expected_effects {
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| instruction.0 == effect)
                .count(),
            expected,
            "unexpected {effect:?} instruction count"
        );
    }
    assert!(instructions.iter().all(|instruction| {
        expected_effects
            .iter()
            .any(|(effect, _)| instruction.0 == *effect)
    }));
    for &(kind, expected) in expected_edge_kinds {
        assert_eq!(
            cfg.edges().filter(|edge| edge.kind() == kind).count(),
            expected,
            "unexpected {kind:?} edge count"
        );
    }
    assert_eq!(
        expected_edge_kinds
            .iter()
            .map(|(_, count)| count)
            .sum::<usize>(),
        expected_edges
    );
}

pub(super) fn assert_linear_cfg(cfg: &Cfg<u32>, node_count: usize) {
    assert_cfg_shape(cfg, node_count, node_count - 1);
    for index in 0..node_count {
        let block = BlockId::from_raw(index as u32);
        assert_eq!(cfg.block(block).instructions(), &[index as u32]);
        if index + 1 < node_count {
            assert_eq!(cfg.successor_edges(block).len(), 1);
            let edge = cfg.edge(cfg.successor_edges(block)[0]);
            assert_eq!(edge.target(), BlockId::from_raw((index + 1) as u32));
            assert_eq!(edge.kind(), EdgeKind::Fallthrough);
        } else {
            assert_eq!(cfg.successor_edges(block).len(), 0);
        }
    }
}

pub(super) fn assert_empty_chain(cfg: &Cfg<u32>, node_count: usize) {
    assert_cfg_shape(cfg, node_count, node_count - 1);
    assert_eq!(cfg.block(cfg.entry()).instructions(), &[0]);
    for index in 1..node_count - 1 {
        assert_eq!(
            cfg.block(BlockId::from_raw(index as u32))
                .instructions()
                .len(),
            0
        );
    }
    assert_eq!(
        cfg.block(BlockId::from_raw((node_count - 1) as u32))
            .instructions(),
        &[(node_count - 1) as u32]
    );
}

pub(super) fn assert_high_fan_in(
    cfg: &Cfg<u32>,
    predecessor_count: usize,
    old_target: BlockId,
    new_target: BlockId,
    redirected: bool,
) {
    assert_cfg_shape(cfg, predecessor_count + 3, predecessor_count);
    let expected_target = if redirected { new_target } else { old_target };
    let empty_target = if redirected { old_target } else { new_target };
    assert_eq!(cfg.predecessor_edges(empty_target).len(), 0);
    assert_eq!(
        cfg.predecessor_edges(expected_target).len(),
        predecessor_count
    );
    for (index, &edge_id) in cfg.predecessor_edges(expected_target).iter().enumerate() {
        assert_eq!(edge_id.index(), index);
        let edge = cfg.edge(edge_id);
        assert_eq!(edge.source(), BlockId::from_raw((index + 3) as u32));
        assert_eq!(edge.target(), expected_target);
        assert_eq!(edge.kind(), EdgeKind::Unconditional);
        assert!(edge.weight().is_none());
    }
}

pub(super) fn assert_weighted_fan_out(
    cfg: &Cfg<u32>,
    edge_count: usize,
    source: BlockId,
    target: BlockId,
    merged: bool,
    target_retains_instructions: bool,
) {
    let live_edges = if merged { edge_count } else { edge_count + 1 };
    assert_cfg_shape(cfg, 3, live_edges);
    let sink = BlockId::from_raw(2);
    let outgoing_source = if merged { source } else { target };

    if merged {
        assert_eq!(cfg.block(source).instructions(), &[0, 1]);
        if target_retains_instructions {
            assert_eq!(cfg.block(target).instructions(), &[1]);
        } else {
            assert_eq!(cfg.block(target).instructions().len(), 0);
        }
        assert_eq!(cfg.successor_edges(target).len(), 0);
    } else {
        assert_eq!(cfg.block(source).instructions(), &[0]);
        assert_eq!(cfg.block(target).instructions(), &[1]);
        let connecting = cfg.edge(cfglib::EdgeId::from_raw(0));
        assert_eq!(connecting.source(), source);
        assert_eq!(connecting.target(), target);
        assert_eq!(connecting.kind(), EdgeKind::Fallthrough);
    }
    assert_eq!(cfg.block(sink).instructions(), &[2]);
    assert_eq!(cfg.successor_edges(outgoing_source).len(), edge_count);

    assert_weighted_outgoing_edges(cfg, edge_count, outgoing_source, source);
}

fn assert_weighted_outgoing_edges(
    cfg: &Cfg<u32>,
    edge_count: usize,
    outgoing_source: BlockId,
    back_target: BlockId,
) {
    let sink = BlockId::from_raw(2);
    for index in 0..edge_count - 1 {
        let id = cfglib::EdgeId::from_raw((index + 1) as u32);
        let edge = cfg.edge(id);
        assert_eq!(edge.id(), id);
        assert_eq!(edge.source(), outgoing_source);
        assert_eq!(edge.target(), sink);
        assert_eq!(
            edge.kind(),
            if index % 2 == 0 {
                EdgeKind::ConditionalTrue
            } else {
                EdgeKind::ConditionalFalse
            }
        );
        assert_eq!(
            edge.weight().map(f64::to_bits),
            Some((index as f64 + 0.25).to_bits())
        );
    }
    let back = cfg.edge(cfglib::EdgeId::from_raw(edge_count as u32));
    assert_eq!(back.source(), outgoing_source);
    assert_eq!(back.target(), back_target);
    assert_eq!(back.kind(), EdgeKind::Back);
    assert_eq!(back.weight().map(f64::to_bits), Some(0.875_f64.to_bits()));
}

pub(super) fn assert_split_weighted_fan_out(
    cfg: &Cfg<u32>,
    edge_count: usize,
    source: BlockId,
    target: BlockId,
    split: BlockId,
) {
    assert_cfg_shape(cfg, 4, edge_count + 2);
    assert_eq!(split, BlockId::from_raw(3));
    assert_eq!(cfg.block(source).instructions(), &[0]);
    assert_eq!(cfg.block(target).instructions(), &[1]);
    assert_eq!(cfg.block(split).instructions().len(), 0);
    assert_eq!(cfg.block(BlockId::from_raw(2)).instructions(), &[2]);

    let connecting = cfg.edge(cfglib::EdgeId::from_raw(0));
    assert_eq!(connecting.source(), source);
    assert_eq!(connecting.target(), target);
    assert_eq!(connecting.kind(), EdgeKind::Fallthrough);
    assert!(connecting.weight().is_none());

    let [fallthrough] = cfg.successor_edges(target) else {
        panic!("split source should have one outgoing edge");
    };
    assert_eq!(fallthrough.index(), edge_count + 1);
    let fallthrough = cfg.edge(*fallthrough);
    assert_eq!(fallthrough.source(), target);
    assert_eq!(fallthrough.target(), split);
    assert_eq!(fallthrough.kind(), EdgeKind::Fallthrough);
    assert!(fallthrough.weight().is_none());
    assert_eq!(cfg.successor_edges(split).len(), edge_count);
    assert_weighted_outgoing_edges(cfg, edge_count, split, source);
}

pub(super) fn assert_weighted_irreducible(cfg: &Cfg<u32>, made_reducible: bool) {
    let split_count = usize::from(made_reducible);
    assert_cfg_shape(cfg, 4 + split_count, 5 + 2 * split_count);
    for index in 0..4 {
        assert_eq!(
            cfg.block(BlockId::from_index(index)).instructions(),
            &[index as u32]
        );
    }

    let dominators = DominatorTree::compute(cfg);
    assert_eq!(
        cfglib::graph::structure::is_reducible(cfg, &dominators),
        made_reducible
    );

    let entry = BlockId::from_raw(0);
    let first = BlockId::from_raw(1);
    let second = BlockId::from_raw(2);
    let exit = BlockId::from_raw(3);
    let redirected_target = if made_reducible {
        BlockId::from_raw(4)
    } else {
        second
    };
    let expected = [
        (entry, first, EdgeKind::ConditionalTrue, None),
        (
            entry,
            redirected_target,
            EdgeKind::ConditionalFalse,
            Some(0.125_f64),
        ),
        (first, second, EdgeKind::Fallthrough, None),
        (second, first, EdgeKind::Back, Some(0.75_f64)),
        (second, exit, EdgeKind::SwitchCase, Some(0.25_f64)),
    ];
    for (index, (source, target, kind, weight)) in expected.into_iter().enumerate() {
        let edge = cfg.edge(cfglib::EdgeId::from_raw(index as u32));
        assert_eq!(edge.source(), source);
        assert_eq!(edge.target(), target);
        assert_eq!(edge.kind(), kind);
        assert_eq!(edge.weight().map(f64::to_bits), weight.map(f64::to_bits));
    }

    if made_reducible {
        let copy = BlockId::from_raw(4);
        assert_eq!(cfg.block(copy).instructions(), &[2]);
        assert_eq!(
            cfg.successor_edges(copy),
            &[cfglib::EdgeId::from_raw(5), cfglib::EdgeId::from_raw(6)]
        );
        for (index, (target, kind, weight)) in [
            (first, EdgeKind::Back, 0.75_f64),
            (exit, EdgeKind::SwitchCase, 0.25_f64),
        ]
        .into_iter()
        .enumerate()
        {
            let edge = cfg.edge(cfglib::EdgeId::from_raw((5 + index) as u32));
            assert_eq!(edge.source(), copy);
            assert_eq!(edge.target(), target);
            assert_eq!(edge.kind(), kind);
            assert_eq!(edge.weight().map(f64::to_bits), Some(weight.to_bits()));
        }
    }
}

pub(super) fn assert_irreducible_fixture(
    cfg: &Cfg<u32>,
    cycle_nodes: usize,
    external_entries: usize,
    expected_splits: usize,
) {
    let original_blocks = 1 + cycle_nodes + external_entries;
    let original_edges = cycle_nodes + 1 + external_entries * 2;
    assert_cfg_shape(
        cfg,
        original_blocks + expected_splits,
        original_edges + expected_splits,
    );
    assert!(
        cfg.blocks()
            .iter()
            .all(|block| block.instructions().is_empty())
    );

    let dominators = DominatorTree::compute(cfg);
    assert_eq!(
        cfglib::graph::structure::is_reducible(cfg, &dominators),
        expected_splits > 0
    );

    let cycle_entry = BlockId::from_raw(2);
    if expected_splits > 0 {
        assert_eq!(expected_splits, cycle_nodes - 1);
        let first_copy = BlockId::from_index(original_blocks);
        for index in 0..external_entries {
            let external = BlockId::from_index(1 + cycle_nodes + index);
            let outgoing = cfg.successor_edges(external);
            assert_eq!(outgoing.len(), 1);
            let edge = cfg.edge(outgoing[0]);
            assert_eq!(edge.target(), first_copy);
            assert_eq!(edge.kind(), EdgeKind::Unconditional);
            assert!(edge.weight().is_none());
        }
        assert_eq!(cfg.predecessor_edges(cycle_entry).len(), 1);
        for split_index in 0..expected_splits {
            let original = BlockId::from_index(2 + split_index);
            let copy = BlockId::from_index(original_blocks + split_index);
            let original_outgoing = cfg.successor_edges(original);
            let copied_outgoing = cfg.successor_edges(copy);
            assert_eq!(original_outgoing.len(), 1);
            assert_eq!(copied_outgoing.len(), 1);
            let original_edge = cfg.edge(original_outgoing[0]);
            let copied_edge = cfg.edge(copied_outgoing[0]);
            let expected_target = if split_index + 1 < expected_splits {
                BlockId::from_index(original_blocks + split_index + 1)
            } else {
                BlockId::from_index(1)
            };
            assert_eq!(copied_edge.source(), copy);
            assert_eq!(copied_edge.target(), expected_target);
            assert_eq!(copied_edge.kind(), original_edge.kind());
            assert_eq!(
                copied_edge.weight().map(f64::to_bits),
                original_edge.weight().map(f64::to_bits)
            );
            assert_eq!(copied_edge.id().index(), original_edges + split_index);
        }
    } else {
        assert_eq!(
            cfg.predecessor_edges(cycle_entry).len(),
            external_entries + 1
        );
    }
}
