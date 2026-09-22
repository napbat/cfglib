use super::harness::{BenchmarkSuite, benchmark_coverage};

mod address;
mod analysis;
mod dataflow;
mod fixtures;
mod graph;
mod transform;

pub(super) const PUBLIC_API_FUNCTIONS: &[&str] = &[
    "abstract_interpret",
    "alias_propagation",
    "bind_edge_attributes",
    "bind_label",
    "block_nesting_depths",
    "block_purities",
    "block_purity",
    "breadth_first",
    "breadth_first_edges",
    "breadth_first_edges_with",
    "breadth_first_events",
    "build_address_cfg",
    "call_graph",
    "canonicalize_loops",
    "cfg_block_nesting_depths",
    "cfg_purity",
    "color_graph",
    "common_ancestors",
    "condensation",
    "condensation_of",
    "constant_propagation",
    "control_flow_edge_attributes",
    "contract_edge",
    "contract_edge_mapped",
    "control_dependence_graph",
    "copies_by_predecessor",
    "copy_propagation",
    "copy_propagation_with_exits",
    "dead_code_elimination",
    "dead_code_elimination_with_exits",
    "duplicate_structuring_tails",
    "duplicate_structuring_tails_with_structure",
    "depth_first_edges",
    "depth_first_edges_with",
    "depth_first_events",
    "depth_first_postorder",
    "depth_first_preorder",
    "detect_cfg_patterns",
    "detect_explicit_tail_calls",
    "detect_loops",
    "detect_loops_tagged",
    "detect_patterns",
    "detect_switch_tables",
    "detect_tail_calls",
    "display_label",
    "eliminate_phis",
    "eliminate_pre",
    "find_back_edges",
    "find_back_edges_tagged",
    "find_function",
    "find_loop_invariants",
    "follow",
    "follow_path",
    "index_paths_may_overlap",
    "insert_preheader",
    "install_clr_region",
    "install_seh_region",
    "promote_handler_extents",
    "interference_graph",
    "is_recursive_function",
    "is_reducible",
    "kosaraju_scc",
    "lift",
    "lift_borrowed",
    "lift_borrowed_with_report",
    "lift_predicated",
    "lift_with_report",
    "linearize",
    "loop_exit_blocks",
    "make_reducible",
    "meet_options",
    "merge_blocks",
    "merge_blocks_mapped",
    "min_label_relaxation",
    "nearest_common_ancestor",
    "no_label",
    "open_breadth_first_events",
    "open_breadth_first_paths",
    "open_depth_first_events",
    "open_fold_post_order",
    "open_search",
    "parse_cfg_text",
    "parse_text",
    "plain_edge_attributes",
    "program_dependence_graph",
    "promote_exclusive_extents",
    "propagate_summaries",
    "reachable",
    "recover_block_expressions",
    "recover_expressions",
    "recover_exclusive_extents",
    "recover_exclusive_extents_with",
    "recover_switch_tables",
    "relax_layout",
    "remove_dead_code",
    "remove_dead_code_mapped",
    "remove_empty_blocks",
    "remove_empty_blocks_mapped",
    "remove_unreachable",
    "remove_unreachable_mapped",
    "resolve_jump_edges",
    "reverse_cfg",
    "reverse_postorder",
    "rotate_loop",
    "scan_predecessors",
    "search",
    "search_with_marks",
    "search_with_scratch",
    "set_uniform_edge_weights",
    "shortest_path",
    "shortest_path_edges",
    "simplify",
    "simplify_mapped",
    "solve_edge_problem",
    "solve_edge_problem_from",
    "solve_edge_problem_from_with_config",
    "solve_edge_problem_with_config",
    "solve_node_problem",
    "solve_node_problem_from",
    "solve_node_problem_from_with_config",
    "solve_node_problem_with_config",
    "solve_problem",
    "solve_problem_from",
    "solve_problem_from_with_config",
    "solve_problem_with_config",
    "split_critical_edges",
    "split_critical_edges_mapped",
    "split_critical_edges_with",
    "split_node",
    "split_node_at_points",
    "split_node_with_payload_mapped",
    "tarjan_scc",
    "to_dot",
    "to_text",
    "topological_sort",
    "try_solve_edge_problem",
    "try_solve_edge_problem_from",
    "try_solve_edge_problem_from_with_config",
    "try_solve_edge_problem_with_config",
    "try_solve_node_problem",
    "try_solve_node_problem_from",
    "try_solve_node_problem_from_with_config",
    "try_solve_node_problem_with_config",
    "try_solve_problem",
    "try_solve_problem_from",
    "try_solve_problem_from_with_config",
    "try_solve_problem_with_config",
    "verify",
    "verify_edge_view",
    "verify_view",
    "verify_with",
    "write_dot",
    "write_text",
];

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    address::register(suite);
    benchmark_coverage!(suite, "cfg_depth_first_preorder", [depth_first_preorder]);
    benchmark_coverage!(suite, "cfg_breadth_first", [breadth_first]);
    benchmark_coverage!(
        suite,
        "directed_breadth_first_edges",
        [breadth_first_edges, breadth_first_edges_with]
    );
    benchmark_coverage!(suite, "directed_shortest_path", [shortest_path]);
    benchmark_coverage!(suite, "directed_shortest_path_edges", [shortest_path_edges]);
    benchmark_coverage!(
        suite,
        "directed_nearest_common_ancestor",
        [nearest_common_ancestor]
    );
    benchmark_coverage!(suite, "directed_common_ancestors", [common_ancestors]);
    benchmark_coverage!(
        suite,
        "cfg_control_dependence_graph",
        [control_dependence_graph]
    );
    benchmark_coverage!(suite, "directed_tarjan_scc", [tarjan_scc]);
    benchmark_coverage!(
        suite,
        "directed_detect_loops_multilatch",
        [detect_loops, find_back_edges]
    );
    benchmark_coverage!(
        suite,
        "directed_node_fixpoint_bool",
        [solve_node_problem, solve_node_problem_with_config]
    );
    benchmark_coverage!(
        suite,
        "cfg_fixpoint_bool",
        [solve_problem, solve_problem_with_config]
    );
    benchmark_coverage!(
        suite,
        "cfg_constprop_independent_constants",
        [constant_propagation]
    );
    benchmark_coverage!(suite, "cfg_clone_merge_linear", [merge_blocks]);
    benchmark_coverage!(suite, "cfg_clone_remove_empty_chain", [remove_empty_blocks]);
    benchmark_coverage!(
        suite,
        "cfg_clone_contract_weighted_high_fan_out",
        [contract_edge]
    );
    benchmark_coverage!(suite, "cfg_clone_make_reducible_large", [make_reducible]);

    analysis::register(suite);
    dataflow::register(suite);
    graph::register(suite);
    transform::register(suite);
}
