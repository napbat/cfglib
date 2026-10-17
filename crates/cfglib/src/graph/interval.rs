//! First-level Allen-Cocke interval analysis.
//!
//! Partitions the source graph into **intervals** — maximal single-entry
//! regions where the header dominates all other blocks. The current analysis
//! computes only this first partition; it does not construct or iterate over
//! successively derived graphs.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use smallvec::SmallVec;

use crate::block::BlockId;
use crate::graph::traverse::{TraversalDirection, reachable};
use crate::graph::view::{DenseId, RootedView};

/// One interval in the source graph's first-level partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interval<N = BlockId> {
    /// The header node — sole entry point of the interval.
    pub header: N,
    /// All nodes in the interval (including the header).
    pub blocks: BTreeSet<N>,
}

/// Result of the source graph's first-level interval analysis.
///
/// The analysis computes only the partition of the original graph. It does
/// not derive successive graphs from the intervals, so it cannot say whether
/// a graph is reducible, and a graph with several intervals may be either.
/// [`is_reducible`](crate::graph::structure::is_reducible) is the answer to
/// that question.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntervalAnalysis<N = BlockId> {
    /// The intervals of the nodes the root reaches, each node in exactly one,
    /// starting with the interval of the root.
    pub intervals: Vec<Interval<N>>,
}

/// Compute the source graph's first-level intervals.
///
/// Allen & Cocke interval construction: starting from the entry,
/// repeatedly absorb successor blocks whose only header-reaching
/// predecessor is within the current interval.
fn compute_intervals_from_graph<G: RootedView>(graph: &G) -> Vec<Interval<G::NodeId>> {
    if graph.node_bound() == 0 {
        return Vec::new();
    }

    let mut intervals = Vec::new();
    // An unassigned node counts the predecessors not yet admitted to the
    // current interval, and an assigned node's count is never read again.
    // These two tables replace separate in-interval, total-predecessor, and
    // inside-predecessor tables.
    //
    // Only predecessors the root reaches are counted: a block that never runs
    // cannot be in an interval, so it must not keep a live block out of one.
    let live = reachable(graph, [graph.root()], TraversalDirection::Outgoing);
    let mut remaining = alloc::vec![0_usize; graph.node_bound()];
    for node in graph.node_ids() {
        remaining[node.index()] = graph
            .predecessors(node)
            .filter(|predecessor| live[predecessor.index()])
            .count();
    }
    let mut assigned = alloc::vec![false; graph.node_bound()];
    let mut headers = alloc::vec![graph.root()];
    let mut worklist: SmallVec<[G::NodeId; 4]> = SmallVec::new();
    let mut successors: SmallVec<[G::NodeId; 4]> = SmallVec::new();

    while let Some(h) = headers.pop() {
        if assigned[h.index()] {
            continue;
        }
        let mut interval = BTreeSet::new();
        interval.insert(h);
        assigned[h.index()] = true;

        // Grow the interval from newly admitted nodes. Every outgoing edge
        // accounts for one predecessor now inside the interval; reaching zero
        // is exactly the Allen-Cocke admission condition. Each edge is visited
        // once instead of repeatedly scanning every dense node to a fixed
        // point, which also makes the result independent of node-id order.
        worklist.clear();
        worklist.push(h);
        while let Some(block) = worklist.pop() {
            for successor in graph.successors(block) {
                let index = successor.index();
                // A count of zero on an unassigned node means an edge the
                // count never saw, from a view whose successors and
                // predecessors disagree; it admits nothing.
                if assigned[index] || remaining[index] == 0 {
                    continue;
                }
                remaining[index] -= 1;
                if remaining[index] == 0 {
                    assigned[index] = true;
                    interval.insert(successor);
                    worklist.push(successor);
                }
            }
        }

        // Blocks that are successors of the interval but not in it
        // become headers for new intervals.
        for &b in &interval {
            successors.clear();
            successors.extend(graph.successors(b));
            // Counts for nodes outside the completed interval are reused by
            // the next header. Restore every decrement contributed by this
            // interval; admitted nodes are assigned and keep no count.
            for &successor in &successors {
                if !assigned[successor.index()] {
                    remaining[successor.index()] += 1;
                }
            }
            successors.sort_unstable();
            successors.dedup();
            for &s in &successors {
                if !assigned[s.index()] {
                    headers.push(s);
                }
            }
        }

        intervals.push(Interval {
            header: h,
            blocks: interval,
        });
    }

    intervals
}

impl<N: DenseId> IntervalAnalysis<N> {
    /// Compute the source graph's first-level interval partition.
    ///
    /// Computes the Allen-Cocke interval partition of the source graph; no
    /// derived graph iteration is performed. Blocks the root does not reach
    /// belong to no interval, and an edge out of one does not count against
    /// the block it enters.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind, IntervalAnalysis};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let b1 = cfg.new_block();
    /// cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
    ///
    /// let result = IntervalAnalysis::compute(&cfg);
    /// assert_eq!(result.intervals.len(), 1);
    /// assert!(result.intervals[0].blocks.contains(&b1));
    /// ```
    #[must_use]
    pub fn compute<G>(graph: &G) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        Self {
            intervals: compute_intervals_from_graph(graph),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::CfgBuilder;
    use crate::flow::FlowEffect;
    use crate::test_util::{MockInst, ff};
    use crate::{DenseId, Graph, GraphView, NodeId, Rooted, RootedView};
    use alloc::format;
    use alloc::vec;

    struct ReverseNodeIds<'g> {
        graph: &'g Graph<(), ()>,
        root: NodeId,
    }

    impl GraphView for ReverseNodeIds<'_> {
        type NodeId = NodeId;

        fn node_bound(&self) -> usize {
            self.graph.node_bound()
        }

        fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
            (0..self.graph.node_bound())
                .rev()
                .map(<NodeId as DenseId>::from_index)
        }

        fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
            self.graph.successors(node)
        }

        fn predecessors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
            self.graph.predecessors(node)
        }
    }

    impl RootedView for ReverseNodeIds<'_> {
        fn root(&self) -> NodeId {
            self.root
        }
    }

    #[test]
    fn single_block_is_one_interval() {
        let cfg = CfgBuilder::build(vec![ff("a")]).unwrap();
        let result = IntervalAnalysis::compute(&cfg);
        assert_eq!(result.intervals.len(), 1);
    }

    #[test]
    fn empty_view_has_no_intervals() {
        let graph = Graph::<(), ()>::new();
        let view = Rooted::new(&graph, NodeId::from_index(0));

        let result = IntervalAnalysis::compute(&view);

        assert_eq!(result.intervals, Vec::new());
    }

    #[test]
    fn linear_cfg_is_one_interval() {
        let cfg = CfgBuilder::build(vec![ff("a"), ff("b"), ff("c")]).unwrap();
        let result = IntervalAnalysis::compute(&cfg);
        // All blocks should be in a single interval since each block
        // has only one predecessor from within the interval.
        assert_eq!(result.intervals.len(), 1);
    }

    #[test]
    fn diamond_cfg_intervals() {
        // Build a diamond manually to avoid Break-outside-scope.
        let mut cfg = crate::Cfg::<MockInst>::new();
        let b0 = cfg.entry();
        let b1 = cfg.new_block();
        let b2 = cfg.new_block();
        let b3 = cfg.new_block();
        cfg.add_edge(b0, b1, crate::edge::EdgeKind::ConditionalTrue);
        cfg.add_edge(b0, b2, crate::edge::EdgeKind::ConditionalFalse);
        cfg.add_edge(b1, b3, crate::edge::EdgeKind::Fallthrough);
        cfg.add_edge(b2, b3, crate::edge::EdgeKind::Fallthrough);

        let result = IntervalAnalysis::compute(&cfg);
        assert_ne!(result.intervals.len(), 0);
    }

    #[test]
    fn loop_cfg_intervals() {
        let cfg = CfgBuilder::build(vec![
            MockInst(FlowEffect::LoopOpen, "loop"),
            ff("body"),
            MockInst(FlowEffect::LoopClose, "endloop"),
            MockInst(FlowEffect::Return, "ret"),
        ])
        .unwrap();
        let result = IntervalAnalysis::compute(&cfg);
        assert_ne!(result.intervals.len(), 0);
    }

    #[test]
    fn reverse_id_chain_is_one_ordered_interval() {
        let mut graph = Graph::<(), ()>::new();
        let nodes: Vec<_> = (0..8).map(|_| graph.add_node(())).collect();
        for index in 1..nodes.len() {
            graph.add_edge(nodes[index], nodes[index - 1], ());
        }

        let result = IntervalAnalysis::compute(&Rooted::new(&graph, nodes[7]));

        assert_eq!(result.intervals.len(), 1);
        assert_eq!(
            result.intervals[0]
                .blocks
                .iter()
                .map(|node| node.index())
                .collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
    }

    #[test]
    fn permuted_node_iteration_still_indexes_predecessor_counts_by_id() {
        let mut graph = Graph::<(), ()>::new();
        let root = graph.add_node(());
        let child = graph.add_node(());
        graph.add_edge(root, child, ());
        let view = ReverseNodeIds {
            graph: &graph,
            root,
        };

        let result = IntervalAnalysis::compute(&view);

        assert_eq!(result.intervals.len(), 1);
        assert_eq!(
            result.intervals[0]
                .blocks
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![root, child]
        );
    }

    #[test]
    fn parallel_predecessor_edges_are_counted_individually() {
        let mut graph = Graph::<(), ()>::with_capacity(2, 255);
        let root = graph.add_node(());
        let child = graph.add_node(());
        for _ in 0..255 {
            graph.add_edge(root, child, ());
        }

        let result = IntervalAnalysis::compute(&Rooted::new(&graph, root));

        assert_eq!(result.intervals.len(), 1);
        assert_eq!(result.intervals[0].blocks.len(), 2);
        assert!(result.intervals[0].blocks.contains(&child));
    }

    #[test]
    fn non_header_self_loop_starts_a_new_interval() {
        let mut graph = Graph::<(), ()>::new();
        let root = graph.add_node(());
        let loop_header = graph.add_node(());
        let exit = graph.add_node(());
        graph.add_edge(root, loop_header, ());
        graph.add_edge(loop_header, loop_header, ());
        graph.add_edge(loop_header, exit, ());

        let result = IntervalAnalysis::compute(&Rooted::new(&graph, root));

        assert_eq!(result.intervals.len(), 2);
        assert_eq!(result.intervals[0].header, root);
        assert_eq!(result.intervals[0].blocks.len(), 1);
        assert_eq!(result.intervals[1].header, loop_header);
        assert_eq!(
            result.intervals[1]
                .blocks
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![loop_header, exit]
        );
    }

    #[test]
    fn unreachable_nodes_are_not_assigned_to_intervals() {
        let mut graph = Graph::<(), ()>::new();
        let root = graph.add_node(());
        let reachable = graph.add_node(());
        let unreachable = graph.add_node(());
        let unreachable_successor = graph.add_node(());
        graph.add_edge(root, reachable, ());
        graph.add_edge(unreachable, unreachable_successor, ());

        let result = IntervalAnalysis::compute(&Rooted::new(&graph, root));

        assert_eq!(result.intervals.len(), 1);
        assert_eq!(result.intervals[0].blocks.len(), 2);
        assert!(result.intervals[0].blocks.contains(&root));
        assert!(result.intervals[0].blocks.contains(&reachable));
        assert!(!result.intervals[0].blocks.contains(&unreachable));
        assert!(!result.intervals[0].blocks.contains(&unreachable_successor));
    }

    #[test]
    fn a_dead_predecessor_does_not_split_an_interval() {
        // `dead` never runs, so it cannot be the second predecessor that
        // keeps `join` out of the root's interval.
        let mut graph = Graph::<(), ()>::new();
        let root = graph.add_node(());
        let join = graph.add_node(());
        let dead = graph.add_node(());
        graph.add_edge(root, join, ());
        graph.add_edge(dead, join, ());

        let result = IntervalAnalysis::compute(&Rooted::new(&graph, root));

        assert_eq!(result.intervals.len(), 1);
        assert_eq!(
            result.intervals[0].blocks,
            BTreeSet::from([root, join]),
            "the dead block is in no interval"
        );
    }

    #[test]
    fn dead_code_that_branches_into_a_live_diamond_leaves_it_whole() {
        // root -> {left, right} -> merge, with a dead block into both the
        // merge and the left arm.
        let mut graph = Graph::<(), ()>::new();
        let [root, left, right, merge, dead] = [(); 5].map(|()| graph.add_node(()));
        for (from, to) in [
            (root, left),
            (root, right),
            (left, merge),
            (right, merge),
            (dead, merge),
            (dead, left),
        ] {
            graph.add_edge(from, to, ());
        }

        let result = IntervalAnalysis::compute(&Rooted::new(&graph, root));

        assert_eq!(result.intervals.len(), 1);
        assert_eq!(
            result.intervals[0].blocks,
            BTreeSet::from([root, left, right, merge])
        );
    }

    /// The Allen-Cocke definition, checked on every digraph of up to four
    /// nodes: the intervals partition the nodes the root reaches, the first
    /// is the root's, each header dominates its interval, every other member
    /// has all its reachable predecessors inside, and no node of a later
    /// interval could have been absorbed.
    #[test]
    fn intervals_satisfy_the_definition_on_every_small_digraph() {
        use crate::DominatorTree;

        for count in 1..=4_usize {
            let pairs: Vec<(usize, usize)> = (0..count)
                .flat_map(|from| (0..count).map(move |to| (from, to)))
                .collect();
            for mask in 0..1_u32 << pairs.len() {
                let mut graph = Graph::<(), ()>::new();
                let nodes: Vec<NodeId> = (0..count).map(|_| graph.add_node(())).collect();
                for (bit, &(from, to)) in pairs.iter().enumerate() {
                    if mask & (1 << bit) != 0 {
                        graph.add_edge(nodes[from], nodes[to], ());
                    }
                }
                let view = Rooted::new(&graph, nodes[0]);
                let dom = DominatorTree::compute(&view);
                let result = IntervalAnalysis::compute(&view);
                let live = |node: NodeId| dom.is_reachable(node);
                let label = format!("{count} nodes, edge mask {mask:#b}");

                let mut seen = BTreeSet::new();
                for (position, interval) in result.intervals.iter().enumerate() {
                    for &block in &interval.blocks {
                        assert!(seen.insert(block), "{label}: {block:?} in two intervals");
                        assert!(dom.dominates(interval.header, block), "{label}");
                        if block != interval.header {
                            assert!(
                                graph
                                    .predecessors(block)
                                    .filter(|&before| live(before))
                                    .all(|before| interval.blocks.contains(&before)),
                                "{label}: {block:?} has a predecessor outside its interval"
                            );
                        }
                    }
                    // Nothing in a later interval had every predecessor in this
                    // one, or it would have been absorbed.
                    for later in &result.intervals[position + 1..] {
                        for &outside in &later.blocks {
                            let absorbable = graph
                                .predecessors(outside)
                                .filter(|&before| live(before))
                                .all(|before| interval.blocks.contains(&before));
                            assert!(!absorbable, "{label}: {outside:?} should have joined");
                        }
                    }
                }
                assert_eq!(
                    seen,
                    nodes
                        .iter()
                        .copied()
                        .filter(|&n| live(n))
                        .collect::<BTreeSet<_>>(),
                    "{label}"
                );
                if count > 0 {
                    assert_eq!(result.intervals[0].header, nodes[0], "{label}");
                }
            }
        }
    }
}
