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
use crate::graph::view::{DenseNodeId, RootedGraphView};

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
/// The public `levels` shape is retained for compatibility, but the current
/// implementation always populates only `levels[0]`, the intervals of the
/// original graph. It does not compute subsequent derived-graph levels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntervalAnalysis<N = BlockId> {
    /// Interval partitions by level; currently only the source partition at
    /// `levels[0]` is populated.
    pub levels: Vec<Vec<Interval<N>>>,
    /// Whether the first partition contains at most one interval.
    ///
    /// This is sufficient to establish reducibility, but `false` is not a
    /// complete irreducibility result because derived levels are not computed.
    /// Use [`is_reducible`](crate::graph::structure::is_reducible) for the full
    /// graph property.
    pub is_reducible: bool,
}

/// Compute the source graph's first-level intervals.
///
/// Allen & Cocke interval construction: starting from the entry,
/// repeatedly absorb successor blocks whose only header-reaching
/// predecessor is within the current interval.
fn compute_intervals_from_graph<G: RootedGraphView>(graph: &G) -> Vec<Interval<G::NodeId>> {
    if graph.node_count() == 0 {
        return Vec::new();
    }

    let max_predecessors = graph
        .node_ids()
        .map(|node| graph.predecessors(node).count())
        .max()
        .unwrap_or(0);

    if max_predecessors < usize::from(u8::MAX) {
        compute_intervals_with_count::<G, u8>(graph)
    } else if max_predecessors < usize::from(u16::MAX) {
        compute_intervals_with_count::<G, u16>(graph)
    } else if max_predecessors < u32::MAX as usize {
        compute_intervals_with_count::<G, u32>(graph)
    } else {
        compute_intervals_with_count::<G, WideCount>(graph)
    }
}

trait IntervalCount: Copy {
    fn from_usize(count: usize) -> Self;
    fn is_assigned(self) -> bool;
    fn mark_assigned(&mut self);
    fn decrement(&mut self) -> bool;
    fn restore(&mut self);
}

macro_rules! impl_interval_count {
    ($count:ty) => {
        impl IntervalCount for $count {
            fn from_usize(count: usize) -> Self {
                match Self::try_from(count) {
                    Ok(count) => count,
                    Err(_) => unreachable!("predecessor count width was selected incorrectly"),
                }
            }

            fn is_assigned(self) -> bool {
                self == Self::MAX
            }

            fn mark_assigned(&mut self) {
                *self = Self::MAX;
            }

            fn decrement(&mut self) -> bool {
                if self.is_assigned() || *self == 0 {
                    return false;
                }
                *self -= 1;
                *self == 0
            }

            fn restore(&mut self) {
                if !self.is_assigned() {
                    *self += 1;
                }
            }
        }
    };
}

impl_interval_count!(u8);
impl_interval_count!(u16);
impl_interval_count!(u32);

#[derive(Clone, Copy)]
struct WideCount {
    remaining: usize,
    assigned: bool,
}

impl IntervalCount for WideCount {
    fn from_usize(remaining: usize) -> Self {
        Self {
            remaining,
            assigned: false,
        }
    }

    fn is_assigned(self) -> bool {
        self.assigned
    }

    fn mark_assigned(&mut self) {
        self.assigned = true;
    }

    fn decrement(&mut self) -> bool {
        if self.assigned || self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        self.remaining == 0
    }

    fn restore(&mut self) {
        if !self.assigned {
            self.remaining += 1;
        }
    }
}

fn compute_intervals_with_count<G, C>(graph: &G) -> Vec<Interval<G::NodeId>>
where
    G: RootedGraphView,
    C: IntervalCount,
{
    let mut intervals = Vec::new();
    // Unassigned nodes store the number of predecessors not yet admitted to
    // the current interval. Assigned nodes use a sentinel because their count
    // is never consulted again. This single table replaces separate assigned,
    // in-interval, total-predecessor, and inside-predecessor tables.
    let mut remaining_predecessors = alloc::vec![C::from_usize(0); graph.node_count()];
    for node in graph.node_ids() {
        remaining_predecessors[node.index()] = C::from_usize(graph.predecessors(node).count());
    }
    let mut headers = alloc::vec![graph.root()];
    let mut worklist: SmallVec<[G::NodeId; 4]> = SmallVec::new();
    let mut successors: SmallVec<[G::NodeId; 4]> = SmallVec::new();

    while let Some(h) = headers.pop() {
        if remaining_predecessors[h.index()].is_assigned() {
            continue;
        }
        let mut interval = BTreeSet::new();
        interval.insert(h);
        remaining_predecessors[h.index()].mark_assigned();

        // Grow the interval from newly admitted nodes. Every outgoing edge
        // accounts for one predecessor now inside the interval; reaching zero
        // is exactly the Allen-Cocke admission condition. Each edge is visited
        // once instead of repeatedly scanning every dense node to a fixed
        // point, which also makes the result independent of node-id order.
        worklist.clear();
        worklist.push(h);
        while let Some(block) = worklist.pop() {
            for successor in graph.successors(block) {
                let remaining = &mut remaining_predecessors[successor.index()];
                if remaining.decrement() {
                    remaining.mark_assigned();
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
            // interval; admitted nodes keep the sentinel permanently.
            for &successor in &successors {
                let remaining = &mut remaining_predecessors[successor.index()];
                remaining.restore();
            }
            successors.sort_unstable();
            successors.dedup();
            for &s in &successors {
                if !remaining_predecessors[s.index()].is_assigned() {
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

/// Perform interval analysis on a rooted graph view.
///
/// Computes the Allen-Cocke interval partition of the source graph. The
/// returned [`IntervalAnalysis`] contains only this first level; no derived
/// graph iteration is performed. Consequently, `is_reducible == false` means
/// only that the first partition did not collapse to one interval, not that the
/// graph has been proven irreducible.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, interval_analysis};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b1 = cfg.new_block();
/// cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
///
/// let result = interval_analysis(&cfg);
/// assert!(result.is_reducible);
/// ```
#[must_use]
pub fn interval_analysis<G: RootedGraphView>(graph: &G) -> IntervalAnalysis<G::NodeId> {
    let mut levels = Vec::new();

    let intervals = compute_intervals_from_graph(graph);
    let num_intervals = intervals.len();
    levels.push(intervals);

    // A single first-level interval is sufficient to establish reducibility.
    // Use `structure::is_reducible` when a complete Boolean answer is needed.
    IntervalAnalysis {
        is_reducible: num_intervals <= 1,
        levels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::CfgBuilder;
    use crate::flow::FlowEffect;
    use crate::test_util::{MockInst, ff};
    use crate::{DenseNodeId, DirectedGraph, DirectedGraphView, NodeId, Rooted, RootedGraphView};
    use alloc::vec;

    struct ReverseNodeIds<'g> {
        graph: &'g DirectedGraph<(), ()>,
        root: NodeId,
    }

    impl DirectedGraphView for ReverseNodeIds<'_> {
        type NodeId = NodeId;

        fn node_count(&self) -> usize {
            self.graph.node_count()
        }

        fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
            (0..self.graph.node_count())
                .rev()
                .map(<NodeId as DenseNodeId>::from_index)
        }

        fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
            self.graph.successors(node)
        }

        fn predecessors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
            self.graph.predecessors(node)
        }
    }

    impl RootedGraphView for ReverseNodeIds<'_> {
        fn root(&self) -> NodeId {
            self.root
        }
    }

    #[test]
    fn single_block_is_one_interval() {
        let cfg = CfgBuilder::build(vec![ff("a")]).unwrap();
        let result = interval_analysis(&cfg);
        assert_eq!(result.levels.len(), 1);
        assert_eq!(result.levels[0].len(), 1);
        assert!(result.is_reducible);
    }

    #[test]
    fn empty_view_has_one_empty_level() {
        let graph = DirectedGraph::<(), ()>::new();
        let view = Rooted::new(&graph, NodeId::from_index(0));

        let result = interval_analysis(&view);

        assert_eq!(result.levels, vec![Vec::new()]);
        assert!(result.is_reducible);
    }

    #[test]
    fn linear_cfg_is_one_interval() {
        let cfg = CfgBuilder::build(vec![ff("a"), ff("b"), ff("c")]).unwrap();
        let result = interval_analysis(&cfg);
        assert_eq!(result.levels.len(), 1);
        // All blocks should be in a single interval since each block
        // has only one predecessor from within the interval.
        assert_eq!(result.levels[0].len(), 1);
        assert!(result.is_reducible);
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

        let result = interval_analysis(&cfg);
        assert_eq!(result.levels.len(), 1);
        assert_ne!(result.levels[0].len(), 0);
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
        let result = interval_analysis(&cfg);
        assert_eq!(result.levels.len(), 1);
        assert_ne!(result.levels[0].len(), 0);
    }

    #[test]
    fn reverse_id_chain_is_one_ordered_interval() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let nodes: Vec<_> = (0..8).map(|_| graph.add_node(())).collect();
        for index in 1..nodes.len() {
            graph.add_edge(nodes[index], nodes[index - 1], ());
        }

        let result = interval_analysis(&Rooted::new(&graph, nodes[7]));

        assert_eq!(result.levels[0].len(), 1);
        assert_eq!(
            result.levels[0][0]
                .blocks
                .iter()
                .map(|node| node.index())
                .collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
    }

    #[test]
    fn permuted_node_iteration_still_indexes_predecessor_counts_by_id() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let root = graph.add_node(());
        let child = graph.add_node(());
        graph.add_edge(root, child, ());
        let view = ReverseNodeIds {
            graph: &graph,
            root,
        };

        let result = interval_analysis(&view);

        assert_eq!(result.levels[0].len(), 1);
        assert_eq!(
            result.levels[0][0]
                .blocks
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![root, child]
        );
    }

    #[test]
    fn parallel_predecessor_edges_are_counted_individually() {
        let mut graph = DirectedGraph::<(), ()>::with_capacity(2, 255);
        let root = graph.add_node(());
        let child = graph.add_node(());
        for _ in 0..255 {
            graph.add_edge(root, child, ());
        }

        let result = interval_analysis(&Rooted::new(&graph, root));

        assert_eq!(result.levels[0].len(), 1);
        assert_eq!(result.levels[0][0].blocks.len(), 2);
        assert!(result.levels[0][0].blocks.contains(&child));
    }

    #[test]
    fn wider_counts_keep_real_values_distinct_from_assignment() {
        let u32_start = usize::from(u16::MAX);
        let mut compact = <u32 as IntervalCount>::from_usize(u32_start);
        assert!(!compact.is_assigned());
        assert!(!compact.decrement());
        assert_eq!(compact, u32::from(u16::MAX) - 1);
        compact.restore();
        assert_eq!(compact, u32::from(u16::MAX));
        compact.mark_assigned();
        assert!(compact.is_assigned());
        assert!(!compact.decrement());
        assert_eq!(compact, u32::MAX);

        let wide_start = u32::MAX as usize;
        let mut wide = WideCount::from_usize(wide_start);
        assert!(!wide.is_assigned());
        assert!(!wide.decrement());
        assert_eq!(wide.remaining, wide_start - 1);
        wide.restore();
        assert_eq!(wide.remaining, wide_start);
        wide.mark_assigned();
        assert!(wide.is_assigned());
        assert!(!wide.decrement());
        assert_eq!(wide.remaining, wide_start);
    }

    #[test]
    fn non_header_self_loop_starts_a_new_interval() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let root = graph.add_node(());
        let loop_header = graph.add_node(());
        let exit = graph.add_node(());
        graph.add_edge(root, loop_header, ());
        graph.add_edge(loop_header, loop_header, ());
        graph.add_edge(loop_header, exit, ());

        let result = interval_analysis(&Rooted::new(&graph, root));

        assert_eq!(result.levels[0].len(), 2);
        assert_eq!(result.levels[0][0].header, root);
        assert_eq!(result.levels[0][0].blocks.len(), 1);
        assert_eq!(result.levels[0][1].header, loop_header);
        assert_eq!(
            result.levels[0][1]
                .blocks
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![loop_header, exit]
        );
    }

    #[test]
    fn unreachable_nodes_are_not_assigned_to_intervals() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let root = graph.add_node(());
        let reachable = graph.add_node(());
        let unreachable = graph.add_node(());
        let unreachable_successor = graph.add_node(());
        graph.add_edge(root, reachable, ());
        graph.add_edge(unreachable, unreachable_successor, ());

        let result = interval_analysis(&Rooted::new(&graph, root));

        assert_eq!(result.levels[0].len(), 1);
        assert_eq!(result.levels[0][0].blocks.len(), 2);
        assert!(result.levels[0][0].blocks.contains(&root));
        assert!(result.levels[0][0].blocks.contains(&reachable));
        assert!(!result.levels[0][0].blocks.contains(&unreachable));
        assert!(!result.levels[0][0].blocks.contains(&unreachable_successor));
    }
}
