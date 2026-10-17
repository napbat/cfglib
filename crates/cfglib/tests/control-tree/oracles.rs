//! The one-call checks: structure a graph and hold the tree to every oracle.

use std::collections::BTreeSet;

use cfglib::{BlockId, Cfg, ControlNode, ControlTree, DominatorTree, LiftReport, detect_loops};

use crate::coverage::{
    assert_each_block_in_exactly_one_leaf, assert_instruction_walk_is_a_partition,
    assert_spans_tile_each_block, reachable_blocks,
};
use crate::implied_flow::assert_flow_matches_graph;
use crate::tree_shape::{
    assert_children_precede_parents, assert_exception_edges_are_expressed_by_nesting,
    assert_report_matches_tree, assert_well_formed,
};

fn run_checks(
    cfg: &Cfg<&'static str>,
    covered: &BTreeSet<BlockId>,
    check_flow: bool,
) -> (ControlTree, LiftReport) {
    let (tree, report) = ControlTree::compute_with_report(cfg);
    assert_eq!(
        ControlTree::compute(cfg),
        tree,
        "compute is compute_with_report without the report"
    );
    assert_well_formed(&tree);
    if report.gotos.is_empty() {
        // Goto labels are applied after the walk, which is its own finding.
        assert_children_precede_parents(&tree);
    }
    assert_each_block_in_exactly_one_leaf(cfg, &tree, covered);
    assert_spans_tile_each_block(cfg, &tree, covered);
    assert_instruction_walk_is_a_partition(cfg, &tree, covered);
    assert_report_matches_tree(&tree, &report);
    if report.unstructured_regions.is_empty() {
        assert_exception_edges_are_expressed_by_nesting(cfg, &tree);
    }
    if check_flow {
        assert_flow_matches_graph(cfg, &tree, &report, covered);
    }
    (tree, report)
}

/// Structure `cfg` and hold the result to every invariant any tree has, whether
/// or not the graph structures fully: the tree covers `covered`, hangs together,
/// reports its gotos honestly, and implies the graph's own flow.
pub fn structure_and_check_covering(
    cfg: &Cfg<&'static str>,
    covered: &BTreeSet<BlockId>,
) -> (ControlTree, LiftReport) {
    run_checks(cfg, covered, true)
}

/// [`structure_and_check_covering`] without the flow oracle, for a graph whose
/// handlers the tree sequences: several `finally` handlers concatenate into one
/// body, which reads as flow between them that the graph does not have.
pub fn structure_and_check_shape_only(
    cfg: &Cfg<&'static str>,
    covered: &BTreeSet<BlockId>,
) -> (ControlTree, LiftReport) {
    run_checks(cfg, covered, false)
}

/// [`structure_and_check`] for a graph with a dangling goto: the goto's target
/// holds nothing the tree can follow, so the flow beyond it is not implied.
pub fn structure_and_check_dangling(cfg: &Cfg<&'static str>) -> (ControlTree, LiftReport) {
    let (tree, report) = run_checks(cfg, &reachable_blocks(cfg), false);
    assert_ne!(report.unresolved_labels, []);
    (tree, report)
}

/// [`structure_and_check_covering`] for a graph whose covered blocks are the
/// ones the entry reaches, and whose loops are the natural loops a dominator
/// tree proves: the tree holds one `Loop` node per header.
pub fn structure_and_check(cfg: &Cfg<&'static str>) -> (ControlTree, LiftReport) {
    let (tree, report) = structure_and_check_covering(cfg, &reachable_blocks(cfg));
    let mut headers: Vec<BlockId> = tree
        .preorder()
        .filter_map(|id| match *tree.node(id) {
            ControlNode::Loop { header, .. } => Some(header),
            _ => None,
        })
        .collect();
    let mut natural: Vec<BlockId> = detect_loops(cfg, &DominatorTree::compute(cfg))
        .into_iter()
        .map(|natural| natural.header)
        .collect();
    headers.sort();
    natural.sort();
    assert_eq!(headers, natural, "one Loop node per natural loop\n{tree:?}");
    (tree, report)
}
