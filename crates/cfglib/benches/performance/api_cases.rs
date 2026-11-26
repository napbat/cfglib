use super::harness::{BenchmarkSuite, benchmark_coverage};

mod address;
mod analysis;
mod block_order;
mod control_dependence;
mod cut;
mod dataflow;
mod decomposition;
mod dyck;
mod feedback;
mod fixtures;
mod frequency;
mod graph;
mod ifds;
mod incremental_scc;
mod inference;
mod path_expr;
mod pipeline;
mod program_structure;
mod reach_index;
mod relaxation;
mod scope;
mod stack;
mod strong_cuts;
mod summaries;
mod transform;
mod traversal;

pub(super) const PUBLIC_API_FUNCTIONS: &[&str] = &[
    "bidirected_dyck_classes",
    "breadth_first",
    "breadth_first_edges",
    "build_address_cfg",
    "call_graph",
    "call_graph_indexed",
    "canonical_hash",
    "canonical_order",
    "canonicalize_loops",
    "common_ancestors",
    "contract_edge",
    "contract_edge_mapped",
    "control_dependence_graph",
    "control_flow_edge_attributes",
    "depth_first_edges",
    "depth_first_events",
    "depth_first_postorder",
    "depth_first_preorder",
    "detect_patterns",
    "display_label",
    "edge_disjoint_paths",
    "empty_trampolines",
    "ext_tsp_order",
    "ext_tsp_score",
    "falls_through",
    "feedback_arc_set",
    "find_back_edges",
    "find_function",
    "follow",
    "follow_path",
    "hot_cold_order",
    "insert_preheader",
    "interpret",
    "is_recursive_function",
    "is_reducible",
    "linear_arrangement",
    "linearize",
    "make_reducible",
    "make_reducible_by_dispatch",
    "meet_options",
    "merge_blocks",
    "merge_blocks_mapped",
    "min_label_relaxation",
    "multiset_fingerprint",
    "nearest_common_ancestor",
    "no_label",
    "nontermination_sensitive_controllers",
    "nontermination_sensitive_controllers_in",
    "open_breadth_first_events",
    "open_breadth_first_paths",
    "open_depth_first_events",
    "open_fold",
    "open_fold_in",
    "open_search",
    "parse_cfg_listing",
    "parse_text",
    "pass_fn",
    "path_expressions",
    "plain_edge_attributes",
    "promote_exclusive_extents",
    "promote_handler_extents",
    "propagate_summaries",
    "reachable",
    "recover_exclusive_extents",
    "recursive_functions",
    "relax_layout",
    "relax_spans",
    "remove_empty_blocks",
    "remove_empty_blocks_mapped",
    "remove_unreachable_mapped",
    "render",
    "resolve_jump_edges",
    "resume_edge_problem",
    "resume_node_problem",
    "reverse_postorder",
    "scan_predecessors",
    "search",
    "search_in",
    "shortest_path",
    "shortest_path_edges",
    "shortest_path_edges_in",
    "shortest_path_in",
    "simplify_mapped",
    "solve_edge_problem",
    "solve_ifds",
    "solve_node_problem",
    "split_critical_edges_mapped",
    "stream_edge_problem",
    "stream_node_problem",
    "to_dot",
    "to_text",
    "topological_sort",
    "topological_sort_or_cycle",
    "try_meet_options",
    "verify",
    "verify_edge_view",
    "verify_semantics",
    "verify_view",
    "weak_control_closure",
    "weak_control_closure_in",
    "wl_colors",
    "write_dot",
    "write_text",
];

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    address::register(suite);
    benchmark_coverage!(suite, "cfg_depth_first_preorder", [depth_first_preorder]);
    benchmark_coverage!(suite, "cfg_breadth_first", [breadth_first]);
    benchmark_coverage!(suite, "directed_breadth_first_edges", [breadth_first_edges]);
    benchmark_coverage!(suite, "directed_shortest_path", [shortest_path]);
    benchmark_coverage!(suite, "directed_shortest_path_edges", [shortest_path_edges]);
    benchmark_coverage!(suite, "directed_shortest_path_in", [shortest_path_in]);
    benchmark_coverage!(
        suite,
        "directed_shortest_path_edges_in",
        [shortest_path_edges_in]
    );
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
    benchmark_coverage!(suite, "directed_node_fixpoint_bool", [solve_node_problem]);
    benchmark_coverage!(suite, "cfg_clone_merge_linear", [merge_blocks]);
    benchmark_coverage!(suite, "cfg_clone_remove_empty_chain", [remove_empty_blocks]);
    benchmark_coverage!(
        suite,
        "cfg_clone_contract_weighted_high_fan_out",
        [contract_edge]
    );
    benchmark_coverage!(suite, "cfg_clone_make_reducible_large", [make_reducible]);

    analysis::register(suite);
    block_order::register(suite);
    control_dependence::register(suite);
    cut::register(suite);
    dataflow::register(suite);
    decomposition::register(suite);
    dyck::register(suite);
    feedback::register(suite);
    frequency::register(suite);
    graph::register(suite);
    ifds::register(suite);
    incremental_scc::register(suite);
    inference::register(suite);
    path_expr::register(suite);
    pipeline::register(suite);
    program_structure::register(suite);
    reach_index::register(suite);
    relaxation::register(suite);
    scope::register(suite);
    stack::register(suite);
    strong_cuts::register(suite);
    summaries::register(suite);
    transform::register(suite);
    traversal::register(suite);
}
