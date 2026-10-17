//! Graph metrics — cyclomatic complexity, nesting depth, code density.
//!
//! Provides quantitative measurements of graph complexity that are useful
//! for program analysis, code quality assessment, and heuristic-driven
//! transformation or decompilation. [`GraphMetrics::compute`] serves any rooted
//! graph view; [`CfgMetrics::compute`] adds instruction-level measurements for a
//! [`CfgView`].
//!
//! Every count is over the nodes the view holds: a slot a store removed is not
//! a node, whether or not the store has been compacted.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::dataflow::cfg_view::CfgView;
use crate::graph::dominator::DominatorTree;
use crate::graph::loop_forest::LoopForest;
use crate::graph::view::{DenseId, RootedView};

/// Topology-level metrics for any rooted graph view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphMetrics {
    /// Number of nodes the view holds, not counting removed slots.
    pub node_count: usize,
    /// Number of edges (counted through forward adjacency).
    pub edge_count: usize,
    /// `McCabe` cyclomatic complexity: `E - N + 2P` (P=1 for single function).
    pub cyclomatic_complexity: usize,
    /// Maximum loop nesting depth: 0 without loops, 1 for a loop that
    /// contains no other, and one more for each loop around it. From
    /// dominance-based loop detection.
    pub max_nesting_depth: usize,
    /// Number of reachable nodes from the root.
    pub reachable_node_count: usize,
    /// Number of unreachable nodes.
    pub unreachable_node_count: usize,
    /// Number of exit nodes (nodes with no successors).
    pub exit_count: usize,
}

/// Instruction-aware metrics for a [`CfgView`].
#[derive(Debug, Clone, PartialEq)]
pub struct CfgMetrics {
    /// Topology metrics (loop depth honors [`EdgeKind::Back`] tags, as
    /// [`LoopForest::compute_tagged`] does).
    ///
    /// [`EdgeKind::Back`]: crate::EdgeKind::Back
    pub graph: GraphMetrics,
    /// Total instruction count across all blocks.
    pub instruction_count: usize,
    /// Average instructions per block (0.0 for empty CFG).
    pub avg_instructions_per_block: f64,
}

/// Everything except loop nesting, computed in one pass.
fn topology<G: RootedView>(graph: &G) -> GraphMetrics {
    let n = graph.node_bound();
    let node_count = graph.node_ids().count();

    // Reachability from the root, counting reachable-region edges as we go
    // (an edge's source being reachable implies its target is too).
    let mut visited = vec![false; n];
    let mut stack = vec![graph.root()];
    visited[graph.root().index()] = true;
    let mut reachable_count = 1;
    let mut reachable_edges = 0;
    while let Some(node) = stack.pop() {
        for successor in graph.successors(node) {
            reachable_edges += 1;
            if !visited[successor.index()] {
                visited[successor.index()] = true;
                reachable_count += 1;
                stack.push(successor);
            }
        }
    }

    let edge_count: usize = graph
        .node_ids()
        .map(|node| graph.successors(node).count())
        .sum();

    // Cyclomatic complexity: E - N + 2P (P=1), with BOTH terms scoped to
    // the region reachable from the root — mixing total edges with
    // reachable nodes would let disconnected components inflate the
    // metric arbitrarily.
    let cyclomatic = if reachable_edges >= reachable_count {
        reachable_edges - reachable_count + 2
    } else {
        1
    };

    let exit_count = graph
        .node_ids()
        .filter(|&node| graph.successors(node).next().is_none())
        .count();

    GraphMetrics {
        node_count,
        edge_count,
        cyclomatic_complexity: cyclomatic,
        max_nesting_depth: 0,
        reachable_node_count: reachable_count,
        unreachable_node_count: node_count.saturating_sub(reachable_count),
        exit_count,
    }
}

impl GraphMetrics {
    /// Compute topology metrics for any rooted graph view.
    ///
    /// # Panics
    ///
    /// Panics when the view's root index is outside `0..node_bound()` — a
    /// broken [`RootedView`] implementation; validate consumer views with
    /// [`verify_view`](crate::verify_view).
    #[must_use]
    pub fn compute<G: RootedView>(graph: &G) -> Self {
        let mut metrics = topology(graph);
        if metrics.edge_count > 0 {
            let dom = DominatorTree::compute(graph);
            metrics.max_nesting_depth = LoopForest::compute(graph, &dom).max_nesting_depth();
        }
        metrics
    }
}

impl CfgMetrics {
    /// Compute comprehensive metrics for a CFG.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, CfgMetrics, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let b0 = cfg.entry();
    /// let b1 = cfg.new_block();
    /// let b2 = cfg.new_block();
    /// cfg.add_edge(b0, b1, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(b0, b2, EdgeKind::ConditionalFalse);
    ///
    /// let m = CfgMetrics::compute(&cfg);
    /// assert_eq!(m.graph.node_count, 3);
    /// assert_eq!(m.graph.edge_count, 2);
    /// assert_eq!(m.graph.exit_count, 2);
    /// ```
    #[must_use]
    pub fn compute<G: CfgView>(cfg: &G) -> Self {
        let mut graph = topology(cfg);

        // Honor explicit back-edge tags for loop depth on CFGs (one dominator
        // computation — not GraphMetrics::compute's dominance-only pass plus a
        // second).
        if graph.edge_count > 0 {
            let dom = DominatorTree::compute(cfg);
            graph.max_nesting_depth = LoopForest::compute_tagged(cfg, &dom).max_nesting_depth();
        }

        let instruction_count: usize = cfg.node_ids().map(|b| cfg.instructions(b).len()).sum();
        let avg_instr = if graph.node_count > 0 {
            crate::usize_to_f64(instruction_count) / crate::usize_to_f64(graph.node_count)
        } else {
            0.0
        };

        CfgMetrics {
            graph,
            instruction_count,
            avg_instructions_per_block: avg_instr,
        }
    }
}

/// Compute the nesting depth of each node.
///
/// Returns a vector indexed by dense node index, where each value is the
/// depth of the node's innermost loop counted from 1, and 0 outside every
/// loop — **dominance-based** detection, like every view-generic algorithm.
/// Callers of a [`CfgView`] whose builders tag explicit `Back` edges want
/// [`cfg_block_nesting_depths`], which honors the tags exactly as
/// [`CfgMetrics::compute`] does.
#[must_use]
pub fn block_nesting_depths<G: RootedView>(graph: &G) -> Vec<usize> {
    let dom = DominatorTree::compute(graph);
    LoopForest::compute(graph, &dom).node_depths()
}

/// Compute the nesting depth of each block of a [`CfgView`], honoring
/// explicit [`EdgeKind::Back`](crate::EdgeKind::Back) tags — the tagged
/// counterpart of [`block_nesting_depths`], consistent with
/// [`CfgMetrics::compute`]'s loop depth.
#[must_use]
pub fn cfg_block_nesting_depths<G: CfgView>(cfg: &G) -> Vec<usize> {
    let dom = DominatorTree::compute(cfg);
    LoopForest::compute_tagged(cfg, &dom).node_depths()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::ff;

    #[test]
    fn single_block_metrics() {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("a"));

        let m = CfgMetrics::compute(&cfg);
        assert_eq!(m.graph.node_count, 1);
        assert_eq!(m.instruction_count, 1);
        assert_eq!(m.graph.cyclomatic_complexity, 1);
        assert_eq!(m.graph.max_nesting_depth, 0);
        assert_eq!(m.graph.reachable_node_count, 1);
        assert_eq!(m.graph.unreachable_node_count, 0);
    }

    #[test]
    fn diamond_cyclomatic_complexity() {
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        let merge = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("e"));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        cfg.add_edge(a, merge, EdgeKind::Fallthrough);
        cfg.add_edge(b, merge, EdgeKind::Fallthrough);

        let m = CfgMetrics::compute(&cfg);
        // E=4, N=4, CC = 4-4+2 = 2
        assert_eq!(m.graph.cyclomatic_complexity, 2);
    }

    #[test]
    fn nesting_depth_with_loop() {
        let mut cfg = Cfg::new();
        let header = cfg.new_block();
        let body = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("e"));
        cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
        cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(body, header, EdgeKind::Back);

        let depths = block_nesting_depths(&cfg);
        assert_eq!(depths[header.index()], 1);
        assert_eq!(depths[body.index()], 1);
        assert_eq!(depths[exit.index()], 0);
        assert_eq!(
            CfgMetrics::compute(&cfg).graph.max_nesting_depth,
            1,
            "one loop is depth 1; depth 0 means no loops"
        );
    }

    #[test]
    fn tagged_nesting_depths_agree_with_cfg_metrics() {
        // Tagged back-edge without dominance (irreducible shape): the
        // untagged view walk scores 0, the Cfg-tagged walk sees the loop —
        // and CfgMetrics::compute's loop depth must agree with the tagged depths.
        let mut cfg = Cfg::<crate::test_util::MockInst>::new();
        let b1 = cfg.new_block();
        let b2 = cfg.new_block();
        cfg.add_edge(cfg.entry(), b1, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b2, EdgeKind::ConditionalFalse);
        cfg.add_edge(b1, b2, EdgeKind::Fallthrough);
        cfg.add_edge(b2, b1, EdgeKind::Back);

        let untagged = block_nesting_depths(&cfg);
        let tagged = cfg_block_nesting_depths(&cfg);
        assert_eq!(untagged[b1.index()], 0, "dominance alone cannot see it");
        assert_eq!(tagged[b1.index()], 1, "the tag makes the loop visible");
        assert_eq!(CfgMetrics::compute(&cfg).graph.max_nesting_depth, 1);
    }

    #[test]
    fn graph_metrics_on_consumer_view() {
        use crate::graph::store::Graph;
        use crate::graph::view::Rooted;

        let mut graph = Graph::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        graph.add_edge(a, b, ());
        graph.add_edge(b, a, ());
        assert_eq!(graph.node(c), &"c");

        let m = GraphMetrics::compute(&Rooted::new(&graph, a));
        assert_eq!(m.node_count, 3);
        assert_eq!(m.reachable_node_count, 2);
        assert_eq!(m.unreachable_node_count, 1);
        assert_eq!(m.max_nesting_depth, 1, "the a<->b cycle is one loop");
        assert_eq!(m.exit_count, 1, "c has no successors");
    }

    #[test]
    fn nested_loops_count_every_enclosing_loop() {
        // entry -> outer <-> (inner <-> body) -> exit
        let mut cfg = Cfg::<u32>::new();
        let [outer, inner, body, exit] = [(); 4].map(|()| cfg.new_block());
        cfg.add_edge(cfg.entry(), outer, EdgeKind::Fallthrough);
        cfg.add_edge(outer, inner, EdgeKind::ConditionalTrue);
        cfg.add_edge(outer, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(inner, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(body, inner, EdgeKind::Back);
        cfg.add_edge(inner, outer, EdgeKind::Back);

        let depths = block_nesting_depths(&cfg);
        assert_eq!(depths[outer.index()], 1);
        assert_eq!(depths[inner.index()], 2);
        assert_eq!(depths[body.index()], 2);
        let metrics = CfgMetrics::compute(&cfg).graph;
        assert_eq!(
            metrics.max_nesting_depth,
            depths.iter().copied().max().unwrap_or(0),
            "the deepest loop is the deepest block"
        );
        assert_eq!(metrics.max_nesting_depth, 2);
    }

    #[test]
    fn a_block_that_loops_to_itself_is_a_loop() {
        let mut cfg = Cfg::<u32>::new();
        cfg.add_edge(cfg.entry(), cfg.entry(), EdgeKind::ConditionalTrue);

        assert_eq!(CfgMetrics::compute(&cfg).graph.max_nesting_depth, 1);
        assert_eq!(GraphMetrics::compute(&cfg).max_nesting_depth, 1);
    }

    #[test]
    fn removed_blocks_are_not_nodes() {
        let mut cfg = Cfg::new();
        let live = cfg.new_block();
        let removed = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("e"));
        cfg.add_edge(cfg.entry(), live, EdgeKind::Fallthrough);
        assert!(cfg.remove_block(removed));

        let before = CfgMetrics::compute(&cfg);
        assert_eq!(before.graph.node_count, cfg.block_count());
        assert_eq!(before.graph.unreachable_node_count, 0);
        assert_eq!(before.graph.reachable_node_count, 2);
        assert!((before.avg_instructions_per_block - 0.5).abs() < f64::EPSILON);

        cfg.compact();
        assert_eq!(CfgMetrics::compute(&cfg), before);
        assert_eq!(
            GraphMetrics::compute(&cfg),
            before.graph,
            "no loop, so the tagged and untagged metrics agree"
        );
    }
}
