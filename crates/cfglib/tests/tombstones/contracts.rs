//! The documented contract of removed slots, stated directly: what iterates,
//! what stays readable, what a compaction reports, and what an analysis run
//! between a transform and a compaction sees.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, CfgMetrics, DenseId, DominatorTree, EdgeId, EdgeKind, Graph, GraphMetrics,
    GraphView, Rooted, RootedView, detect_loops, find_back_edges, kosaraju_scc, merge_blocks,
    remove_empty_blocks, remove_unreachable, simplify, tarjan_scc,
};

use crate::cfg_analyses::check_cfg_analyses;
use crate::oracle::Reference;
use crate::topology::{GENERAL, Op, Realizations, SEEDS, SSA_SAFE, Slots, Topology, ViewCheck};
use crate::view_analyses::check_general_views;

#[test]
fn a_removed_block_leaves_iteration_but_stays_readable_until_compaction() {
    let mut cfg = Cfg::<u32>::new();
    let entry = cfg.entry();
    let (first, middle, last) = (cfg.new_block(), cfg.new_block(), cfg.new_block());
    cfg.block_mut(first).push(10);
    cfg.block_mut(middle).push(20);
    cfg.block_mut(last).push(30);
    let into_first = cfg.add_edge(entry, first, EdgeKind::Fallthrough);
    let into_middle = cfg.add_edge(first, middle, EdgeKind::ConditionalTrue);
    let out_of_middle = cfg.add_edge(middle, last, EdgeKind::Fallthrough);
    let bypass = cfg.add_edge(first, last, EdgeKind::ConditionalFalse);

    assert!(cfg.remove_block(middle));
    assert!(
        !cfg.remove_block(middle),
        "a second removal reports that nothing changed"
    );

    assert_eq!(cfg.block_ids().collect::<Vec<_>>(), [entry, first, last]);
    assert_eq!(cfg.blocks().count(), 3);
    assert_eq!(cfg.block_count(), 3);
    assert_eq!(
        cfg.block_bound(),
        4,
        "the removed slot is still inside the bound"
    );
    assert_eq!(
        cfg.edge_ids().collect::<Vec<_>>(),
        [into_first, bypass],
        "the removed block's edges went with it; the rest keep insertion order"
    );
    assert_eq!(cfg.edge_count(), 2);
    assert_eq!(cfg.edge_bound(), 4);
    assert!(!cfg.contains_block(middle));
    assert!(!cfg.contains_edge(into_middle) && !cfg.contains_edge(out_of_middle));
    assert_eq!(
        cfg.block(middle).instructions(),
        [20],
        "a removed block stays readable"
    );
    assert_eq!(cfg.successors(first).collect::<Vec<_>>(), [last]);
    assert_eq!(cfg.predecessors(last).collect::<Vec<_>>(), [first]);
    assert!(!cfg.is_compact());

    let renumbering = cfg.compact();
    assert!(cfg.is_compact());
    assert_eq!(cfg.block_bound(), 3);
    assert_eq!(renumbering.node(middle), None);
    assert_eq!(renumbering.node(last).map(DenseId::index), Some(2));
    assert_eq!(cfg.block(BlockId::from_index(2)).instructions(), [30]);
    assert_eq!(cfg.entry(), entry, "the entry kept its number");
}

#[test]
fn a_compaction_is_the_identity_exactly_when_nothing_was_removed() {
    let mut untouched = Cfg::<u32>::new();
    let kept = untouched.new_block();
    untouched.add_edge(untouched.entry(), kept, EdgeKind::Fallthrough);
    let renumbering = untouched.compact();
    assert!(renumbering.is_identity(), "nothing was removed");
    assert_eq!(renumbering.node(kept), Some(kept));

    let mut trailing = Cfg::<u32>::new();
    let kept = trailing.new_block();
    let last = trailing.new_block();
    trailing.add_edge(trailing.entry(), kept, EdgeKind::Fallthrough);
    assert!(trailing.remove_block(last));
    let renumbering = trailing.compact();
    assert_eq!(
        renumbering.node(kept),
        Some(kept),
        "every survivor kept its number"
    );
    assert_eq!(renumbering.node(last), None);
    assert!(
        !renumbering.is_identity(),
        "a removed block is not mapped to itself, so the numbering is not the identity"
    );
}

#[test]
fn compaction_numbers_survivors_densely_in_their_old_order() {
    for seed in SEEDS {
        let realized = Realizations::of_topology(Topology::random(seed, GENERAL))
            .labeled(format!("seed {seed}"));
        let topology = &realized.topology;
        let live: Vec<usize> = (0..topology.slots)
            .filter(|&slot| topology.is_live(slot))
            .collect();
        let live_edges: Vec<usize> = (0..topology.edges.len())
            .filter(|&edge| topology.is_live_edge(edge))
            .collect();

        for slot in 0..topology.slots {
            let expected = live.iter().position(|&candidate| candidate == slot);
            assert_eq!(
                realized
                    .cfg_renumbering
                    .node(BlockId::from_index(slot))
                    .map(DenseId::index),
                expected,
                "seed {seed}: cfg block {slot}"
            );
            assert_eq!(
                realized
                    .graph_renumbering
                    .node(cfglib::NodeId::from_index(slot))
                    .map(DenseId::index),
                expected,
                "seed {seed}: graph node {slot}"
            );
        }
        for edge in 0..topology.edges.len() {
            let expected = live_edges.iter().position(|&candidate| candidate == edge);
            assert_eq!(
                realized
                    .cfg_renumbering
                    .edge(EdgeId::from_index(edge))
                    .map(DenseId::index),
                expected,
                "seed {seed}: cfg edge {edge}"
            );
        }
        assert_eq!(realized.cfg_renumbering.node_bound(), topology.slots);
        assert_eq!(realized.cfg_renumbering.edge_bound(), topology.edges.len());
        assert!(
            !realized.cfg_renumbering.is_identity() && !realized.graph_renumbering.is_identity(),
            "seed {seed}: every generated topology removes a node, so no numbering is the identity"
        );

        let compacted = &realized.cfg_compacted;
        assert!(compacted.is_compact());
        assert_eq!(compacted.block_bound(), live.len(), "seed {seed}");
        assert_eq!(compacted.edge_bound(), live_edges.len(), "seed {seed}");
        for (rank, &slot) in live.iter().enumerate() {
            assert_eq!(
                compacted.block(BlockId::from_index(rank)).instructions(),
                topology.blocks[slot],
                "seed {seed}: instructions of block {slot}"
            );
        }
        for (rank, &edge) in live_edges.iter().enumerate() {
            let (source, target) = topology.edges[edge];
            let renumbered = compacted.edge(EdgeId::from_index(rank));
            let rank_of = |slot: usize| live.iter().position(|&candidate| candidate == slot);
            assert_eq!(
                Some(renumbered.source().index()),
                rank_of(source),
                "seed {seed}: edge {edge}"
            );
            assert_eq!(
                Some(renumbered.target().index()),
                rank_of(target),
                "seed {seed}: edge {edge}"
            );
        }
    }
}

struct AdjacencyContract;

impl ViewCheck for AdjacencyContract {
    fn check<G: RootedView>(
        &self,
        view: &G,
        slots: &Slots<G::NodeId>,
        reference: &Reference<'_>,
        label: &str,
    ) {
        let live: BTreeSet<usize> = reference.live().collect();
        assert!(
            view.node_bound() >= live.len(),
            "{label}: the bound covers every live node"
        );
        let yielded: Vec<usize> = view.node_ids().map(|node| slots.slot_of(node)).collect();
        assert_eq!(
            yielded.iter().copied().collect::<BTreeSet<_>>(),
            live,
            "{label}: node_ids yields the live nodes"
        );
        assert_eq!(
            yielded.len(),
            live.len(),
            "{label}: node_ids yields each node once"
        );
        for node in view.node_ids() {
            assert!(
                node.index() < view.node_bound(),
                "{label}: {node:?} is inside the bound"
            );
        }
        let sorted = |mut slots: Vec<usize>| {
            slots.sort_unstable();
            slots
        };
        for &node in &live {
            let id = slots.id(node);
            assert_eq!(
                sorted(
                    view.successors(id)
                        .map(|next| slots.slot_of(next))
                        .collect()
                ),
                sorted(reference.successors(node).to_vec()),
                "{label}: successors of {node}, parallel edges included"
            );
            assert_eq!(
                sorted(
                    view.predecessors(id)
                        .map(|previous| slots.slot_of(previous))
                        .collect()
                ),
                sorted(reference.predecessors(node).to_vec()),
                "{label}: predecessors of {node}, parallel edges included"
            );
        }
    }
}

#[test]
fn views_report_live_nodes_and_one_adjacency_in_both_directions() {
    for seed in SEEDS {
        let realized = Realizations::of_topology(Topology::random(seed, GENERAL))
            .labeled(format!("seed {seed}"));
        realized.run_views(&AdjacencyContract);
        assert_eq!(realized.cfg_holed.node_bound(), realized.topology.slots);
        let live = Reference::new(&realized.topology).live_count();
        assert_eq!(realized.cfg_compacted.node_bound(), live);
        assert_eq!(realized.graph_compacted.node_bound(), live);
    }
}

fn entry_to_one_block_plus_a_removed_block() -> Cfg<u32> {
    let mut cfg = Cfg::<u32>::new();
    let reached = cfg.new_block();
    let removed = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(1);
    cfg.block_mut(reached).push(2);
    cfg.block_mut(removed).push(3);
    cfg.add_edge(cfg.entry(), reached, EdgeKind::Fallthrough);
    assert!(cfg.remove_block(removed));
    cfg
}

#[test]
fn metrics_count_the_live_blocks_not_the_slots() {
    let mut cfg = entry_to_one_block_plus_a_removed_block();
    let metrics = CfgMetrics::compute(&cfg);
    assert_eq!(metrics.graph.node_count, 2);
    assert_eq!(metrics.graph.edge_count, 1);
    assert_eq!(metrics.graph.reachable_node_count, 2);
    assert_eq!(
        metrics.graph.unreachable_node_count, 0,
        "a removed slot is not unreachable code"
    );
    assert_eq!(metrics.graph.exit_count, 1);
    assert_eq!(metrics.instruction_count, 2);
    assert!((metrics.avg_instructions_per_block - 1.0).abs() < f64::EPSILON);
    assert_eq!(GraphMetrics::compute(&cfg), metrics.graph);

    cfg.compact();
    assert_eq!(
        CfgMetrics::compute(&cfg),
        metrics,
        "compaction does not change a metrics report"
    );
}

#[test]
fn unreachable_blocks_are_counted_only_when_they_are_live() {
    let mut cfg = Cfg::<u32>::new();
    let reached = cfg.new_block();
    let dead_code = cfg.new_block();
    let removed = cfg.new_block();
    cfg.add_edge(cfg.entry(), reached, EdgeKind::Fallthrough);
    cfg.add_edge(dead_code, reached, EdgeKind::Fallthrough);
    assert!(cfg.remove_block(removed));

    let metrics = GraphMetrics::compute(&cfg);
    assert_eq!(metrics.node_count, 3);
    assert_eq!(metrics.reachable_node_count, 2);
    assert_eq!(
        metrics.unreachable_node_count, 1,
        "only the live dead block"
    );
    assert_eq!(metrics.edge_count, 2);
}

#[test]
fn metrics_between_a_transform_and_the_compaction_equal_the_compacted_metrics() {
    for seed in SEEDS {
        let mut cfg = Topology::random(seed, SSA_SAFE).build_cfg();
        remove_unreachable(&mut cfg);
        let holed = CfgMetrics::compute(&cfg);
        assert_eq!(holed.graph.node_count, cfg.block_count(), "seed {seed}");
        assert_eq!(holed.graph.edge_count, cfg.edge_count(), "seed {seed}");
        assert_eq!(
            holed.graph.unreachable_node_count, 0,
            "seed {seed}: nothing unreachable is left"
        );
        cfg.compact();
        assert_eq!(CfgMetrics::compute(&cfg), holed, "seed {seed}");
    }
}

#[test]
fn components_cover_the_live_nodes_only() {
    let mut graph = Graph::<&str, ()>::new();
    let first = graph.add_node("first");
    let second = graph.add_node("second");
    let removed = graph.add_node("removed");
    let last = graph.add_node("last");
    graph.add_edge(first, second, ());
    graph.add_edge(second, first, ());
    graph.add_edge(second, removed, ());
    graph.add_edge(removed, last, ());
    assert!(graph.remove_node(removed));

    for (name, decomposition) in [
        ("tarjan", tarjan_scc(&graph)),
        ("kosaraju", kosaraju_scc(&graph)),
    ] {
        let components: BTreeSet<Vec<_>> = decomposition.components().map(<[_]>::to_vec).collect();
        assert_eq!(
            components,
            BTreeSet::from([vec![first, second], vec![last]]),
            "{name}: the removed node is in no component"
        );
        assert_eq!(decomposition.len(), 2, "{name}");
        assert!(
            !decomposition.is_dag(&graph),
            "{name}: first and second still form a cycle"
        );
    }
}

#[test]
fn a_graph_whose_only_cycle_ran_through_a_removed_node_is_acyclic() {
    let mut graph = Graph::<(), ()>::new();
    let (first, middle, last) = (graph.add_node(()), graph.add_node(()), graph.add_node(()));
    graph.add_edge(first, middle, ());
    graph.add_edge(middle, last, ());
    graph.add_edge(last, first, ());
    assert!(!tarjan_scc(&graph).is_dag(&graph));

    assert!(graph.remove_node(middle));
    for decomposition in [tarjan_scc(&graph), kosaraju_scc(&graph)] {
        assert!(decomposition.is_dag(&graph));
        assert_eq!(decomposition.len(), 2);
    }
    let rooted = Rooted::new(&graph, first);
    assert_eq!(GraphMetrics::compute(&rooted).node_count, 2);
}

#[test]
fn analyses_run_between_a_transform_and_the_compaction() {
    type Transform = fn(&mut Cfg<Op>) -> usize;
    let transforms: [(&str, Transform); 4] = [
        ("remove_unreachable", remove_unreachable::<Op, ()>),
        ("merge_blocks", merge_blocks::<Op, ()>),
        ("remove_empty_blocks", remove_empty_blocks::<Op, ()>),
        ("simplify", simplify::<Op, ()>),
    ];
    for (name, transform) in transforms {
        for seed in SEEDS {
            let mut cfg = Topology::random(seed, SSA_SAFE).build_cfg();
            transform(&mut cfg);
            let realized = Realizations::of_cfg(cfg).labeled(format!("{name} seed {seed}"));
            check_general_views(&realized);
            check_cfg_analyses(&realized);
        }
    }
}

#[test]
fn loops_are_relative_to_the_entry_so_dead_cycles_are_not_loops() {
    let mut cfg = Cfg::<u32>::new();
    let exit = cfg.new_block();
    let dead_self_loop = cfg.new_block();
    let (dead_first, dead_second) = (cfg.new_block(), cfg.new_block());
    cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);
    cfg.add_edge(dead_self_loop, dead_self_loop, EdgeKind::ConditionalTrue);
    cfg.add_edge(dead_first, dead_second, EdgeKind::Fallthrough);
    cfg.add_edge(dead_second, dead_first, EdgeKind::Fallthrough);

    let dominators = DominatorTree::compute(&cfg);
    assert!(
        detect_loops(&cfg, &dominators).is_empty(),
        "no cycle is reachable from the entry, so no loop exists"
    );
    assert_eq!(find_back_edges(&cfg, &dominators), Vec::new());
    assert_eq!(GraphMetrics::compute(&cfg).max_nesting_depth, 0);
}
