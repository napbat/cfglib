//! Graph metrics — cyclomatic complexity, nesting depth, code density.
//!
//! Provides quantitative measurements of graph complexity that are useful
//! for program analysis, code quality assessment, and heuristic-driven
//! transformation or decompilation. [`GraphMetrics::compute`] serves any rooted
//! edge view; [`CfgMetrics::compute`] adds instruction-level measurements for a
//! [`CfgView`] on top of the same graph metrics, so the two agree on every
//! graph both accept. The loop depth of each block is
//! [`LoopForest::node_depths`].
//!
//! Every count is over the nodes the view holds: a slot a store removed is not
//! a node, whether or not the store has been compacted.

use crate::dataflow::cfg_view::CfgView;
use crate::edge::KindedEdge;
use crate::graph::dominator::DominatorTree;
use crate::graph::edge_view::EdgeView;
use crate::graph::loop_forest::LoopForest;
use crate::graph::traverse::reachable;
use crate::graph::view::{DenseId, Direction, RootedView};

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
    /// contains no other, and one more for each loop around it. The loops are
    /// those of [`LoopForest::compute`], which honors
    /// [`EdgeKind::Back`](crate::EdgeKind::Back) tags.
    pub max_nesting_depth: usize,
    /// Number of nodes reachable from the root.
    pub reachable_node_count: usize,
    /// Number of exit nodes (nodes with no successors).
    pub exit_count: usize,
}

/// Instruction-aware metrics for a [`CfgView`].
#[derive(Debug, Clone, PartialEq)]
pub struct CfgMetrics {
    /// The [`GraphMetrics`] of the CFG.
    pub graph: GraphMetrics,
    /// Total instruction count across all blocks.
    pub instruction_count: usize,
    /// Average instructions per block (0.0 for empty CFG).
    pub avg_instructions_per_block: f64,
}

impl GraphMetrics {
    /// Compute topology metrics for any rooted edge view.
    ///
    /// # Panics
    ///
    /// Panics when the view's root index is outside `0..node_bound()` — a
    /// broken [`RootedView`] implementation; validate consumer views with
    /// [`verify_view`](crate::verify_view).
    #[must_use]
    pub fn compute<G>(graph: &G) -> Self
    where
        G: RootedView + EdgeView,
        G::EdgeData: KindedEdge,
    {
        let node_count = graph.node_ids().count();
        let edge_count: usize = graph
            .node_ids()
            .map(|node| graph.successors(node).count())
            .sum();

        // Cyclomatic complexity: E - N + 2P (P=1), with BOTH terms scoped to
        // the region reachable from the root — mixing total edges with
        // reachable nodes would let disconnected components inflate the
        // metric arbitrarily. An edge out of a reachable node ends at one.
        let reached = reachable(graph, [graph.root()], Direction::Forward);
        let (reachable_node_count, reachable_edges) = graph
            .node_ids()
            .filter(|node| reached[node.index()])
            .fold((0, 0), |(nodes, edges), node| {
                (nodes + 1, edges + graph.successors(node).count())
            });
        let cyclomatic_complexity = if reachable_edges >= reachable_node_count {
            reachable_edges - reachable_node_count + 2
        } else {
            1
        };

        let exit_count = graph
            .node_ids()
            .filter(|&node| graph.successors(node).next().is_none())
            .count();

        let max_nesting_depth = if edge_count > 0 {
            let dom = DominatorTree::compute(graph);
            LoopForest::compute(graph, &dom).max_nesting_depth()
        } else {
            0
        };

        GraphMetrics {
            node_count,
            edge_count,
            cyclomatic_complexity,
            max_nesting_depth,
            reachable_node_count,
            exit_count,
        }
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
        let graph = GraphMetrics::compute(cfg);
        let instruction_count: usize = cfg.node_ids().map(|b| cfg.instructions(b).len()).sum();
        let avg_instructions_per_block = if graph.node_count > 0 {
            crate::usize_to_f64(instruction_count) / crate::usize_to_f64(graph.node_count)
        } else {
            0.0
        };

        CfgMetrics {
            graph,
            instruction_count,
            avg_instructions_per_block,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::graph::store::NodeOrder;
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

        let depths = LoopForest::compute(&cfg, &DominatorTree::compute(&cfg)).node_depths();
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
    fn a_tagged_loop_dominance_cannot_prove_counts_for_both_metrics() {
        // A tagged back edge without dominance (irreducible shape): the tag
        // makes the loop visible, to the graph metrics and the CFG metrics
        // alike.
        let mut cfg = Cfg::<crate::test_util::MockInst>::new();
        let b1 = cfg.new_block();
        let b2 = cfg.new_block();
        cfg.add_edge(cfg.entry(), b1, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b2, EdgeKind::ConditionalFalse);
        cfg.add_edge(b1, b2, EdgeKind::Fallthrough);
        cfg.add_edge(b2, b1, EdgeKind::Back);

        let depths = LoopForest::compute(&cfg, &DominatorTree::compute(&cfg)).node_depths();
        assert_eq!(depths[b1.index()], 1, "the tag makes the loop visible");
        assert_eq!(GraphMetrics::compute(&cfg).max_nesting_depth, 1);
        assert_eq!(CfgMetrics::compute(&cfg).graph, GraphMetrics::compute(&cfg));
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

        let depths = LoopForest::compute(&cfg, &DominatorTree::compute(&cfg)).node_depths();
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
        assert_eq!(before.graph.reachable_node_count, 2);
        assert!((before.avg_instructions_per_block - 0.5).abs() < f64::EPSILON);

        cfg.compact(NodeOrder::Insertion);
        assert_eq!(CfgMetrics::compute(&cfg), before);
        assert_eq!(GraphMetrics::compute(&cfg), before.graph);
    }
}
