#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
// `cfglib_bench_alloc` is an intentionally benchmark-local cfg selected with
// `RUSTFLAGS="--cfg cfglib_bench_alloc"`; it is not a crate feature.
#![allow(unexpected_cfgs)]

use std::alloc::System;
#[cfg(cfglib_bench_alloc)]
use std::alloc::{GlobalAlloc, Layout};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::hint::black_box;
#[cfg(cfglib_bench_alloc)]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
#[cfg(not(cfglib_bench_alloc))]
use std::time::Instant;

use cfglib::dataflow::constprop::ConstantFolder;
use cfglib::{
    BlockId, Cfg, CfgBuilder, CommonAncestor, ConstValue, DenseNodeId, DirectedGraph, Direction,
    DominanceFrontiers, DominatorTree, EdgeKind, EdgeStep, FixpointResult, FlowControl, FlowEffect,
    InstrInfo, IntervalAnalysis, NaturalLoop, NodeFacts, NodeId, NodeProblem, PhiPlacements,
    Problem, ProgramPoint, Rooted, SccResult, SccpResult, SsaForm, SsaValue, TraversalDirection,
    ValueNumberInfo, ValueNumbering, breadth_first, breadth_first_edges, build_ssa,
    common_ancestors, constant_propagation, contract_edge, control_dependence_graph,
    depth_first_preorder, detect_loops, global_value_numbering, interval_analysis, merge_blocks,
    nearest_common_ancestor, place_phis, remove_empty_blocks, sccp, shortest_path,
    shortest_path_edges, solve_node_problem, tarjan_scc,
};

#[path = "performance/analysis_oracles.rs"]
mod analysis_oracles;
#[path = "performance/fixtures.rs"]
mod fixtures;
#[path = "performance/harness.rs"]
mod harness;
#[path = "performance/structural_oracles.rs"]
mod structural_oracles;

use analysis_oracles::{
    assert_bool_cfg_facts, assert_bool_node_facts, assert_branchy_dominators,
    assert_branchy_post_dominators, assert_cfg_intervals, assert_common_ancestor_results,
    assert_control_dependence_graph, assert_dominance_frontiers, assert_edge_path,
    assert_edge_traversal, assert_linear_ssa, assert_node_path, assert_phi_placements,
    assert_phi_ssa, assert_reverse_chain_intervals, assert_wide_cfg_facts, assert_wide_node_facts,
    directed_distances, reference_cfg_breadth_first, reference_cfg_preorder,
};
use fixtures::{
    BuilderInst, CfgReachability, ConstantInst, Reachability, WideCfgFact, WideNodeFact,
    branchy_cfg, branchy_graph, build_conditional_break_chain, build_eight_case_switch_chain,
    build_if_else_chain, build_two_case_switch_chain, empty_chain_cfg, high_fan_in_cfg,
    independent_constants, irreducible_cfg, linear_cfg, linear_constants, many_exit_cfg,
    multi_latch_graph, phi_storm_cfg, reverse_id_chain_graph, weighted_high_fan_out_cfg,
    weighted_irreducible_cfg,
};
use harness::{benchmark, benchmark_target, configuration_error, run_semantic_oracle};
use structural_oracles::{
    assert_branchy_cfg, assert_branchy_graph, assert_builder_cfg, assert_cfg_shape,
    assert_dense_permutation, assert_empty_chain, assert_high_fan_in, assert_irreducible_fixture,
    assert_linear_cfg, assert_split_weighted_fan_out, assert_weighted_fan_out,
    assert_weighted_irreducible,
};

#[allow(clippy::too_many_lines)]
fn main() {
    const NODE_COUNT: usize = 4_096;
    const BUILDER_REGION_COUNT: usize = 2_048;

    // Cargo passes `--bench` to custom benchmark binaries. Treat only a
    // positional argument as the optional name filter.
    let filter = std::env::args()
        .skip(1)
        .find(|argument| !argument.starts_with('-'))
        .unwrap_or_default();
    let (target_ms, target) = benchmark_target();
    #[cfg(cfglib_bench_alloc)]
    let _ = target_ms;
    let cfg = branchy_cfg(NODE_COUNT);
    let cfg_dominators = DominatorTree::compute(&cfg);
    let cfg_post_dominators = DominatorTree::compute_post(&cfg);
    let graph = branchy_graph(NODE_COUNT);
    let (reverse_id_chain, reverse_id_root) = reverse_id_chain_graph(NODE_COUNT);
    let wide_cfg = branchy_cfg(1_024);
    let wide_graph = branchy_graph(1_024);
    let linear = linear_cfg(NODE_COUNT / 2);
    let many_exits = many_exit_cfg(NODE_COUNT - 1);
    let linear_dominators = DominatorTree::compute(&linear);
    let empty_chain = empty_chain_cfg(NODE_COUNT / 2);
    let (high_fan_in, old_target, new_target) = high_fan_in_cfg(NODE_COUNT);
    let (weighted_high_fan_out, fan_out_source, fan_out_target) =
        weighted_high_fan_out_cfg(NODE_COUNT);
    let irreducible_small = irreducible_cfg(2, 512);
    let irreducible_large = irreducible_cfg(512, 512);
    let weighted_irreducible = weighted_irreducible_cfg();
    let (multi_latch, multi_latch_root) = multi_latch_graph(2_048, 256);
    let multi_latch_dominators =
        DominatorTree::compute(&Rooted::new(&multi_latch, multi_latch_root));
    let constants = independent_constants(1_024);
    let constant_dominators = DominatorTree::compute(&constants);
    let constant_ssa = build_ssa(&constants, &constant_dominators);
    let linear_constant_cfg = linear_constants(2_048);
    let linear_constant_dominators = DominatorTree::compute(&linear_constant_cfg);
    let phi_storm = phi_storm_cfg(32, 128);
    let phi_storm_dominators = DominatorTree::compute(&phi_storm);

    println!(
        "cfglib synthetic benchmark: {NODE_COUNT} nodes, {} CFG edges, {} graph edges",
        cfg.num_edges(),
        graph.edge_count()
    );
    #[cfg(cfglib_bench_alloc)]
    println!(
        "mode: allocation instrumentation (counting wrapper over System; CPU timing disabled)"
    );
    #[cfg(not(cfglib_bench_alloc))]
    println!("mode: CPU (direct System global allocator; no counting wrapper)");
    #[cfg(not(cfglib_bench_alloc))]
    println!("timing: median of 7 samples; target >= {target_ms} ms/sample");
    #[cfg(cfglib_bench_alloc)]
    println!(
        "memory: allocations and requested bytes per operation; peak is incremental live bytes"
    );

    let mut registered = 0_usize;
    let mut matched = 0_usize;
    macro_rules! bench {
        ($name:literal, $operation:expr, $oracle:expr) => {{
            registered += 1;
            if filter.is_empty() || $name.contains(&filter) {
                matched += 1;
                let mut operation = $operation;
                run_semantic_oracle(&mut operation, $oracle);
                benchmark($name, target, operation);
            }
        }};
    }

    bench!(
        "cfg_build_branchy",
        || branchy_cfg(NODE_COUNT),
        |result: &Cfg<u32>| assert_branchy_cfg(result, NODE_COUNT)
    );
    bench!(
        "cfg_builder_if_else_chain",
        || build_if_else_chain(BUILDER_REGION_COUNT),
        |result: &Cfg<BuilderInst>| assert_builder_cfg(
            result,
            1 + 3 * BUILDER_REGION_COUNT,
            4 * BUILDER_REGION_COUNT,
            &[
                (FlowEffect::ConditionalOpen, BUILDER_REGION_COUNT),
                (FlowEffect::ConditionalAlternate, BUILDER_REGION_COUNT),
                (FlowEffect::Fallthrough, 2 * BUILDER_REGION_COUNT),
            ],
            &[
                (EdgeKind::ConditionalTrue, BUILDER_REGION_COUNT),
                (EdgeKind::ConditionalFalse, BUILDER_REGION_COUNT),
                (EdgeKind::Fallthrough, 2 * BUILDER_REGION_COUNT),
            ],
        )
    );
    bench!(
        "cfg_builder_conditional_break_chain",
        || { build_conditional_break_chain(BUILDER_REGION_COUNT) },
        |result: &Cfg<BuilderInst>| assert_builder_cfg(
            result,
            1 + 4 * BUILDER_REGION_COUNT,
            5 * BUILDER_REGION_COUNT,
            &[
                (FlowEffect::LoopOpen, BUILDER_REGION_COUNT),
                (FlowEffect::ConditionalBreak, BUILDER_REGION_COUNT),
            ],
            &[
                (EdgeKind::Fallthrough, BUILDER_REGION_COUNT),
                (EdgeKind::ConditionalTrue, BUILDER_REGION_COUNT),
                (EdgeKind::ConditionalFalse, BUILDER_REGION_COUNT),
                (EdgeKind::Back, BUILDER_REGION_COUNT),
                (EdgeKind::Unconditional, BUILDER_REGION_COUNT),
            ],
        )
    );
    bench!(
        "cfg_builder_two_case_switch_chain",
        || { build_two_case_switch_chain(BUILDER_REGION_COUNT) },
        |result: &Cfg<BuilderInst>| assert_builder_cfg(
            result,
            1 + 3 * BUILDER_REGION_COUNT,
            4 * BUILDER_REGION_COUNT,
            &[
                (FlowEffect::SwitchOpen, BUILDER_REGION_COUNT),
                (FlowEffect::SwitchCase, BUILDER_REGION_COUNT),
                (FlowEffect::Fallthrough, 2 * BUILDER_REGION_COUNT),
            ],
            &[
                (EdgeKind::SwitchCase, 2 * BUILDER_REGION_COUNT),
                (EdgeKind::Fallthrough, 2 * BUILDER_REGION_COUNT),
            ],
        )
    );
    bench!(
        "cfg_builder_eight_case_switch_chain",
        || { build_eight_case_switch_chain(BUILDER_REGION_COUNT / 4) },
        |result: &Cfg<BuilderInst>| {
            let regions = BUILDER_REGION_COUNT / 4;
            assert_builder_cfg(
                result,
                1 + 9 * regions,
                16 * regions,
                &[
                    (FlowEffect::SwitchOpen, regions),
                    (FlowEffect::SwitchCase, 7 * regions),
                    (FlowEffect::Fallthrough, 8 * regions),
                ],
                &[
                    (EdgeKind::SwitchCase, 8 * regions),
                    (EdgeKind::Fallthrough, 8 * regions),
                ],
            );
        }
    );
    bench!(
        "directed_build_branchy",
        || branchy_graph(NODE_COUNT),
        |result: &DirectedGraph<(), ()>| assert_branchy_graph(result, NODE_COUNT)
    );
    bench!(
        "cfg_depth_first_preorder",
        || depth_first_preorder(&cfg, cfg.entry(), TraversalDirection::Outgoing),
        |result: &Vec<BlockId>| {
            assert_dense_permutation(result, NODE_COUNT);
            assert_eq!(*result, reference_cfg_preorder(&cfg));
        }
    );
    bench!(
        "cfg_breadth_first",
        || breadth_first(&cfg, cfg.entry(), TraversalDirection::Outgoing),
        |result: &Vec<BlockId>| {
            assert_dense_permutation(result, NODE_COUNT);
            assert_eq!(*result, reference_cfg_breadth_first(&cfg));
        }
    );
    bench!(
        "directed_breadth_first_edges",
        || breadth_first_edges(&graph, NodeId::from_raw(0), TraversalDirection::Outgoing),
        |result: &Vec<EdgeStep>| assert_edge_traversal(result, &graph)
    );
    bench!(
        "directed_shortest_path",
        || {
            shortest_path(
                &graph,
                NodeId::from_raw(0),
                NodeId::from_index(NODE_COUNT - 1),
                TraversalDirection::Outgoing,
            )
            .expect("fixture target is reachable")
        },
        |result: &Vec<NodeId>| assert_node_path(
            result,
            &graph,
            NodeId::from_raw(0),
            NodeId::from_index(NODE_COUNT - 1),
        )
    );
    bench!(
        "directed_shortest_path_edges",
        || {
            shortest_path_edges(
                &graph,
                NodeId::from_raw(0),
                NodeId::from_index(NODE_COUNT - 1),
                TraversalDirection::Outgoing,
            )
            .expect("fixture target is reachable")
        },
        |result: &Vec<cfglib::graph::directed::EdgeId>| assert_edge_path(
            result,
            &graph,
            NodeId::from_raw(0),
            NodeId::from_index(NODE_COUNT - 1),
        )
    );
    bench!(
        "directed_nearest_common_ancestor",
        || {
            nearest_common_ancestor(
                &graph,
                NodeId::from_index(NODE_COUNT - 1),
                NodeId::from_index(NODE_COUNT - 2),
                TraversalDirection::Incoming,
            )
        },
        |result: &Option<NodeId>| {
            let a = NodeId::from_index(NODE_COUNT - 1);
            let b = NodeId::from_index(NODE_COUNT - 2);
            let (from_a, _) = directed_distances(&graph, a, TraversalDirection::Incoming);
            let (from_b, _) = directed_distances(&graph, b, TraversalDirection::Incoming);
            let expected = graph
                .node_ids()
                .filter(|node| {
                    from_a[node.index()] != usize::MAX && from_b[node.index()] != usize::MAX
                })
                .min_by_key(|node| (from_a[node.index()] + from_b[node.index()], node.index()));
            assert_eq!(*result, expected);
        }
    );
    bench!(
        "directed_common_ancestors",
        || {
            common_ancestors(
                &graph,
                NodeId::from_index(NODE_COUNT - 1),
                NodeId::from_index(NODE_COUNT - 2),
                TraversalDirection::Incoming,
                None,
            )
        },
        |result: &Vec<CommonAncestor<NodeId>>| assert_common_ancestor_results(
            result,
            &graph,
            NodeId::from_index(NODE_COUNT - 1),
            NodeId::from_index(NODE_COUNT - 2),
        )
    );
    bench!(
        "cfg_dominators",
        || DominatorTree::compute(&cfg),
        |result: &DominatorTree| assert_branchy_dominators(result, NODE_COUNT)
    );
    bench!(
        "cfg_dominance_frontiers",
        || { DominanceFrontiers::compute(&cfg, &cfg_dominators) },
        |result: &DominanceFrontiers| assert_dominance_frontiers(result, &cfg, &cfg_dominators,)
    );
    bench!(
        "cfg_post_dominators",
        || DominatorTree::compute_post(&cfg),
        |result: &DominatorTree| assert_branchy_post_dominators(result, NODE_COUNT)
    );
    bench!(
        "cfg_post_dominators_many_exits",
        || { DominatorTree::compute_post(&many_exits) },
        |result: &DominatorTree| {
            for index in 0..NODE_COUNT {
                let block = BlockId::from_index(index);
                assert!(result.is_reachable(block));
                assert_eq!(result.idom(block), None);
            }
        }
    );
    bench!(
        "cfg_control_dependence_graph",
        || { control_dependence_graph(&cfg, &cfg_post_dominators) },
        |result: &DirectedGraph<BlockId, ()>| assert_control_dependence_graph(
            result,
            &cfg,
            &cfg_post_dominators,
        )
    );
    bench!(
        "cfg_dominator_depths_linear",
        || { linear_dominators.depths() },
        |result: &Vec<usize>| {
            assert_eq!(result.len(), NODE_COUNT / 2);
            assert!(result.iter().copied().eq(0..NODE_COUNT / 2));
        }
    );
    bench!(
        "directed_tarjan_scc",
        || tarjan_scc(&graph),
        |result: &SccResult<NodeId>| {
            let cycle_count = (NODE_COUNT - 1) / 32;
            assert_eq!(result.len(), NODE_COUNT - 16 * cycle_count);
            let mut seen = vec![false; NODE_COUNT];
            for (component_index, component) in result.components.iter().enumerate() {
                assert!(!component.nodes.is_empty());
                for &node in &component.nodes {
                    assert!(!seen[node.index()]);
                    seen[node.index()] = true;
                    assert_eq!(result.component_index(node), component_index);
                }
            }
            assert!(seen.into_iter().all(core::convert::identity));
            for cycle in 1..=cycle_count {
                let end = cycle * 32;
                let component = result.component(NodeId::from_index(end));
                assert_eq!(component.nodes.len(), 17);
                assert!(
                    (end - 16..=end).all(|index| component.contains(NodeId::from_index(index)))
                );
            }
            assert_eq!(
                result
                    .components
                    .iter()
                    .filter(|component| component.nodes.len() == 17)
                    .count(),
                cycle_count
            );
            assert!(
                result
                    .components
                    .iter()
                    .all(|component| { component.nodes.len() == 1 || component.nodes.len() == 17 })
            );
        }
    );
    bench!(
        "directed_detect_loops_multilatch",
        || detect_loops(&multi_latch, &multi_latch_dominators),
        |result: &Vec<NaturalLoop<NodeId>>| {
            assert_eq!(result.len(), 1);
            let natural_loop = &result[0];
            assert_eq!(natural_loop.header, multi_latch_root);
            assert_eq!(natural_loop.depth, 0);
            assert_eq!(natural_loop.body.len(), 2_048 + 256 + 1);
            assert!(
                (0..=2_048 + 256)
                    .all(|index| natural_loop.body.contains(&NodeId::from_index(index)))
            );
            assert_eq!(natural_loop.latches.len(), 256);
            assert!(
                (2_049..=2_304)
                    .all(|index| natural_loop.latches.contains(&NodeId::from_index(index)))
            );
        }
    );
    bench!(
        "cfg_interval_analysis",
        || interval_analysis(&cfg),
        |result: &IntervalAnalysis| assert_cfg_intervals(result, &cfg)
    );
    bench!(
        "directed_interval_reverse_id_chain",
        || { interval_analysis(&Rooted::new(&reverse_id_chain, reverse_id_root)) },
        |result: &IntervalAnalysis<NodeId>| assert_reverse_chain_intervals(
            result,
            NODE_COUNT,
            reverse_id_root,
        )
    );
    bench!(
        "directed_node_fixpoint_bool",
        || solve_node_problem(&graph, &Reachability),
        |result: &NodeFacts<bool>| assert_bool_node_facts(result, NODE_COUNT)
    );
    bench!(
        "directed_node_fixpoint_wide",
        || solve_node_problem(&wide_graph, &WideNodeFact),
        |result: &NodeFacts<Vec<u64>>| assert_wide_node_facts(result, 1_024)
    );
    bench!(
        "cfg_fixpoint_bool",
        || { cfglib::dataflow::fixpoint::solve(&cfg, &CfgReachability) },
        |result: &FixpointResult<bool>| assert_bool_cfg_facts(result, NODE_COUNT)
    );
    bench!(
        "cfg_fixpoint_wide",
        || { cfglib::dataflow::fixpoint::solve(&wide_cfg, &WideCfgFact) },
        |result: &FixpointResult<Vec<u64>>| assert_wide_cfg_facts(result, 1_024)
    );
    bench!(
        "cfg_sccp_independent_constants",
        || sccp(&constants, &constant_ssa),
        |result: &SccpResult<u32, u64>| {
            assert_eq!(result.reachable_blocks, BTreeSet::from([constants.entry()]));
            assert!(result.executable_edges.is_empty());
            assert_eq!(result.values.len(), 1_024);
            for variable in 0..1_024_u32 {
                assert_eq!(
                    result.values.get(&SsaValue::new(variable, 1)),
                    Some(&ConstValue::Const(u64::from(variable)))
                );
            }
        }
    );
    bench!(
        "cfg_build_ssa_linear",
        || build_ssa(&linear_constant_cfg, &linear_constant_dominators),
        |result: &SsaForm<u32>| assert_linear_ssa(result, 2_048)
    );
    bench!(
        "cfg_place_phis_phi_storm",
        || place_phis(&phi_storm, &phi_storm_dominators),
        |result: &PhiPlacements<u32>| assert_phi_placements(result, 32, 128)
    );
    bench!(
        "cfg_build_ssa_phi_storm",
        || build_ssa(&phi_storm, &phi_storm_dominators),
        |result: &SsaForm<u32>| assert_phi_ssa(result, &phi_storm, 32, 128)
    );
    bench!(
        "cfg_global_value_numbering_linear",
        || { global_value_numbering(&linear_constant_cfg, &linear_constant_dominators) },
        |result: &ValueNumbering| {
            assert_eq!(result.blocks.len(), 2_048);
            assert_eq!(result.num_values, 2_048);
            for index in 0..2_048 {
                let block = result
                    .blocks
                    .get(&BlockId::from_index(index))
                    .expect("value numbering omitted a block");
                assert_eq!(block.inst_vn, [Some(index as u32)]);
                assert_eq!(block.redundant.len(), 0);
            }
        }
    );
    bench!(
        "cfg_constprop_independent_constants",
        || { constant_propagation(&constants) },
        |result: &FixpointResult<BTreeMap<u32, ConstValue<u64>>>| {
            assert_eq!(result.block_in.len(), 1);
            assert_eq!(result.block_out.len(), 1);
            assert!(result.block_in[0].is_empty());
            assert_eq!(result.block_out[0].len(), 1_024);
            for variable in 0..1_024_u32 {
                assert_eq!(
                    result.block_out[0].get(&variable),
                    Some(&ConstValue::Const(u64::from(variable)))
                );
            }
        }
    );
    bench!("cfg_clone_linear", || linear.clone(), |result: &Cfg<
        u32,
    >| {
        assert_linear_cfg(result, NODE_COUNT / 2);
    });
    bench!(
        "cfg_clone_merge_linear",
        || {
            let mut cloned = linear.clone();
            let merged = merge_blocks(&mut cloned);
            (cloned, merged)
        },
        |(result, merged): &(Cfg<u32>, usize)| {
            assert_eq!(*merged, NODE_COUNT / 2 - 1);
            assert_cfg_shape(result, NODE_COUNT / 2, 0);
            assert_eq!(
                result.block(result.entry()).instructions(),
                &(0..NODE_COUNT as u32 / 2).collect::<Vec<_>>()
            );
            assert!(
                result.blocks()[1..]
                    .iter()
                    .all(|block| block.instructions().is_empty())
            );
        }
    );
    bench!(
        "cfg_clone_empty_chain",
        || empty_chain.clone(),
        |result: &Cfg<u32>| assert_empty_chain(result, NODE_COUNT / 2)
    );
    bench!(
        "cfg_clone_remove_empty_chain",
        || {
            let mut cloned = empty_chain.clone();
            let removed = remove_empty_blocks(&mut cloned);
            (cloned, removed)
        },
        |(result, removed): &(Cfg<u32>, usize)| {
            assert_eq!(*removed, NODE_COUNT / 2 - 2);
            assert_cfg_shape(result, NODE_COUNT / 2, 1);
            assert_eq!(result.block(result.entry()).instructions(), &[0]);
            let last = BlockId::from_index(NODE_COUNT / 2 - 1);
            assert_eq!(
                result.block(last).instructions(),
                &[(NODE_COUNT / 2 - 1) as u32]
            );
            let edge = result.edge(cfglib::EdgeId::from_raw(0));
            assert_eq!(edge.source(), result.entry());
            assert_eq!(edge.target(), last);
            assert_eq!(edge.kind(), EdgeKind::Fallthrough);
            assert!(edge.weight().is_none());
        }
    );
    bench!(
        "cfg_clone_high_fan_in",
        || high_fan_in.clone(),
        |result: &Cfg<u32>| assert_high_fan_in(result, NODE_COUNT, old_target, new_target, false,)
    );
    bench!(
        "cfg_clone_redirect_high_fan_in",
        || {
            let mut cloned = high_fan_in.clone();
            cloned.redirect_edges_to(old_target, new_target);
            cloned
        },
        |result: &Cfg<u32>| assert_high_fan_in(result, NODE_COUNT, old_target, new_target, true,)
    );
    bench!(
        "cfg_clone_weighted_high_fan_out",
        || { weighted_high_fan_out.clone() },
        |result: &Cfg<u32>| assert_weighted_fan_out(
            result,
            NODE_COUNT,
            fan_out_source,
            fan_out_target,
            false,
            false,
        )
    );
    bench!(
        "cfg_clone_split_weighted_high_fan_out",
        || {
            let mut cloned = weighted_high_fan_out.clone();
            let split = cloned.split_block(fan_out_target, 1);
            (cloned, split)
        },
        |(result, split): &(Cfg<u32>, BlockId)| assert_split_weighted_fan_out(
            result,
            NODE_COUNT,
            fan_out_source,
            fan_out_target,
            *split,
        )
    );
    bench!(
        "cfg_clone_merge_weighted_high_fan_out",
        || {
            let mut cloned = weighted_high_fan_out.clone();
            let merged = merge_blocks(&mut cloned);
            (cloned, merged)
        },
        |(result, merged): &(Cfg<u32>, usize)| {
            assert_eq!(*merged, 1);
            assert_weighted_fan_out(
                result,
                NODE_COUNT,
                fan_out_source,
                fan_out_target,
                true,
                false,
            );
        }
    );
    bench!(
        "cfg_clone_contract_weighted_high_fan_out",
        || {
            let mut cloned = weighted_high_fan_out.clone();
            let contracted = contract_edge(&mut cloned, fan_out_source, fan_out_target);
            (cloned, contracted)
        },
        |(result, contracted): &(Cfg<u32>, bool)| {
            assert!(*contracted);
            assert_weighted_fan_out(
                result,
                NODE_COUNT,
                fan_out_source,
                fan_out_target,
                true,
                true,
            );
        }
    );
    bench!(
        "cfg_clone_irreducible_small",
        || irreducible_small.clone(),
        |result: &Cfg<u32>| assert_irreducible_fixture(result, 2, 512, 0)
    );
    bench!(
        "cfg_clone_make_reducible_small",
        || {
            let mut cloned = irreducible_small.clone();
            let splits = cfglib::graph::reducible::make_reducible(&mut cloned);
            (cloned, splits)
        },
        |(result, splits): &(Cfg<u32>, usize)| {
            assert_eq!(*splits, 1);
            assert_irreducible_fixture(result, 2, 512, 1);
        }
    );
    bench!(
        "cfg_clone_irreducible_large",
        || irreducible_large.clone(),
        |result: &Cfg<u32>| assert_irreducible_fixture(result, 512, 512, 0)
    );
    bench!(
        "cfg_clone_make_reducible_large",
        || {
            let mut cloned = irreducible_large.clone();
            let splits = cfglib::graph::reducible::make_reducible(&mut cloned);
            (cloned, splits)
        },
        |(result, splits): &(Cfg<u32>, usize)| {
            assert_eq!(*splits, 511);
            assert_irreducible_fixture(result, 512, 512, 511);
        }
    );
    bench!(
        "cfg_clone_weighted_irreducible",
        || weighted_irreducible.clone(),
        |result: &Cfg<u32>| assert_weighted_irreducible(result, false)
    );
    bench!(
        "cfg_clone_make_reducible_weighted",
        || {
            let mut cloned = weighted_irreducible.clone();
            let splits = cfglib::graph::reducible::make_reducible(&mut cloned);
            (cloned, splits)
        },
        |(result, splits): &(Cfg<u32>, usize)| {
            assert_eq!(*splits, 1);
            assert_weighted_irreducible(result, true);
        }
    );

    assert_eq!(registered, 49, "benchmark registration count changed");
    if !filter.is_empty() && matched == 0 {
        configuration_error(&format!(
            "benchmark filter `{filter}` matched no registered cases"
        ));
    }
}
