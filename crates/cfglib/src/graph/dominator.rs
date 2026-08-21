//! Dominator tree computation using the Cooper-Harvey-Kennedy iterative
//! algorithm.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::traverse::{TraversalDirection, reverse_postorder};
use crate::graph::view::{DenseNodeId, DirectedGraphView, RootedGraphView};

/// A dominator tree computed from a rooted directed graph.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, DominatorTree};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// let b2 = cfg.new_block();
/// cfg.add_edge(b0, b1, EdgeKind::ConditionalTrue);
/// cfg.add_edge(b0, b2, EdgeKind::ConditionalFalse);
///
/// let dom = DominatorTree::compute(&cfg);
/// assert_eq!(dom.idom(b1), Some(b0));
/// assert_eq!(dom.idom(b2), Some(b0));
/// assert!(dom.dominates(b0, b1));
/// assert!(dom.dominates(b0, b2));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DominatorTree<N = BlockId> {
    /// Immediate dominator for each node. The root has no parent.
    idom: Vec<Option<N>>,
    reachable: Vec<bool>,
}

/// Smallest lossless depth representation for an internal whole-graph pass.
pub(crate) enum AnalysisDepths {
    Compact(Vec<u32>),
    Full(Vec<usize>),
}

/// Integer operations shared by full and compact dominator depth tables.
///
/// This stays private so the two storage widths are implementation details;
/// the generic depth cores monomorphise to the same integer operations as the
/// former hand-written versions.
trait DepthWord: Copy + Ord {
    const UNREACHABLE: Self;
    const ZERO: Self;

    fn add(self, other: Self) -> Self;
    fn next(self) -> Self;
    fn previous(self) -> Self;
}

impl DepthWord for usize {
    const UNREACHABLE: Self = usize::MAX;
    const ZERO: Self = 0;

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn next(self) -> Self {
        self + 1
    }

    fn previous(self) -> Self {
        self - 1
    }
}

impl DepthWord for u32 {
    const UNREACHABLE: Self = u32::MAX;
    const ZERO: Self = 0;

    fn add(self, other: Self) -> Self {
        self.checked_add(other)
            .expect("compact dominator depth exceeds u32")
    }

    fn next(self) -> Self {
        self.checked_add(1)
            .expect("compact dominator depth exceeds u32")
    }

    fn previous(self) -> Self {
        self - 1
    }
}

/// Order of siblings in a compact dominator-child linked list.
#[derive(Clone, Copy)]
pub(crate) enum DominatorChildOrder {
    /// Following links visits children by increasing dense node id.
    Ascending,
    /// Following links visits children by decreasing dense node id.
    Descending,
}

/// Compact, transient child adjacency for whole-tree consumers.
///
/// Keeping this separate from [`DominatorTree`] avoids permanently increasing
/// every tree's memory footprint for the few passes that need repeated child
/// traversal.
pub(crate) struct DominatorChildLinks<N> {
    first_child: Vec<Option<N>>,
    next_sibling: Vec<Option<N>>,
}

impl<N: DenseNodeId> DominatorChildLinks<N> {
    /// First child of `parent` in the selected order.
    pub(crate) fn first_child(&self, parent: N) -> Option<N> {
        self.first_child[parent.index()]
    }

    /// Next sibling after `child` in the selected order.
    pub(crate) fn next_sibling(&self, child: N) -> Option<N> {
        self.next_sibling[child.index()]
    }
}

fn compact_depths_supported(node_count: usize) -> bool {
    u32::try_from(node_count).is_ok()
}

/// A reversed graph with one synthetic root connected to every exit.
///
/// Keeping this as a view avoids copying every node, edge, and adjacency list
/// merely to run the generic dominator algorithm.
struct PostDominatorView<'g, G: DirectedGraphView> {
    graph: &'g G,
    exits: &'g [G::NodeId],
    /// Small exit lists use linear multiplicity counts; large lists are
    /// normalized before constructing the view and use binary search.
    binary_search_exits: bool,
}

const POST_DOMINATOR_BINARY_SEARCH_THRESHOLD: usize = 16;

impl<G: DirectedGraphView> PostDominatorView<'_, G> {
    #[inline]
    fn exit_multiplicity(&self, node: G::NodeId) -> usize {
        if self.binary_search_exits {
            usize::from(self.exits.binary_search(&node).is_ok())
        } else {
            self.exits.iter().filter(|&&exit| exit == node).count()
        }
    }
}

impl<G: DirectedGraphView> DirectedGraphView for PostDominatorView<'_, G> {
    // The virtual exit is private implementation state.  Use `usize` for the
    // augmented view so callers' IDs are never asked to represent the
    // out-of-range index at `graph.node_count()`.
    type NodeId = usize;

    fn node_count(&self) -> usize {
        self.graph.node_count() + 1
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        let original_count = self.graph.node_count();
        let is_virtual = node == original_count;
        let original = (node < original_count).then(|| G::NodeId::from_index(node));
        is_virtual
            .then_some(self.exits)
            .into_iter()
            .flatten()
            .copied()
            .map(DenseNodeId::index)
            .chain(
                original
                    .into_iter()
                    .flat_map(move |node| self.graph.predecessors(node).map(DenseNodeId::index)),
            )
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        let original_count = self.graph.node_count();
        let original = (node < original_count).then(|| G::NodeId::from_index(node));
        let exit_multiplicity = original.map_or(0, |node| self.exit_multiplicity(node));
        original
            .into_iter()
            .flat_map(move |node| self.graph.successors(node).map(DenseNodeId::index))
            .chain(core::iter::repeat_n(original_count, exit_multiplicity))
    }
}

impl<N: DenseNodeId> DominatorTree<N> {
    /// Compute the dominator tree of a rooted graph view using the iterative
    /// algorithm by Cooper, Harvey, and Kennedy.
    #[must_use]
    pub fn compute<G>(graph: &G) -> Self
    where
        G: RootedGraphView<NodeId = N>,
    {
        Self::compute_from(graph, graph.root())
    }

    /// Compute dominators for any directed graph view from an explicit root.
    #[must_use]
    pub fn compute_from<G>(graph: &G, root: N) -> Self
    where
        G: DirectedGraphView<NodeId = N>,
    {
        let order = reverse_postorder(graph, root, TraversalDirection::Outgoing);
        let node_count = graph.node_count();
        let mut order_index = vec![usize::MAX; node_count];
        for (index, node) in order.iter().copied().enumerate() {
            order_index[node.index()] = index;
        }

        let mut dominators = vec![None; order.len()];
        dominators[order_index[root.index()]] = Some(order_index[root.index()]);

        let mut changed = true;
        while changed {
            changed = false;
            for node in order.iter().copied().filter(|node| *node != root) {
                let node_order = order_index[node.index()];
                let mut new_parent_index = None;
                for predecessor in graph.predecessors(node) {
                    let predecessor_order = order_index[predecessor.index()];
                    if predecessor_order == usize::MAX || dominators[predecessor_order].is_none() {
                        continue;
                    }

                    new_parent_index = Some(match new_parent_index {
                        None => predecessor_order,
                        Some(parent) if predecessor_order == parent => parent,
                        Some(parent) => Self::intersect(&dominators, predecessor_order, parent),
                    });
                }
                let Some(new_parent_index) = new_parent_index else {
                    continue;
                };

                if dominators[node_order] != Some(new_parent_index) {
                    dominators[node_order] = Some(new_parent_index);
                    changed = true;
                }
            }
        }

        let mut immediate = vec![None; node_count];
        for (index, parent) in dominators.into_iter().enumerate() {
            let node = order[index];
            immediate[node.index()] = parent.map(|parent_index| order[parent_index]);
        }
        immediate[root.index()] = None;
        let mut reachable = vec![false; node_count];
        for node in order {
            reachable[node.index()] = true;
        }
        Self {
            idom: immediate,
            reachable,
        }
    }

    fn intersect(dominators: &[Option<usize>], mut left: usize, mut right: usize) -> usize {
        while left != right {
            while left > right {
                left = dominators[left].expect("processed dominator must have a parent");
            }
            while right > left {
                right = dominators[right].expect("processed dominator must have a parent");
            }
        }
        left
    }

    /// Return the immediate dominator of `node`, or `None` for a root.
    #[must_use]
    pub fn idom(&self, node: N) -> Option<N> {
        self.idom[node.index()]
    }

    /// Whether `node` was reachable from the root this tree was computed
    /// from. `idom` alone cannot distinguish "is the root" from "was never
    /// reached" (both `None`) — consumers reasoning about dominance must
    /// check this before trusting a `None`.
    #[must_use]
    pub fn is_reachable(&self, node: N) -> bool {
        self.reachable[node.index()]
    }

    /// Return whether `dominator` dominates `node`.
    #[must_use]
    pub fn dominates(&self, dominator: N, node: N) -> bool {
        if dominator == node {
            return true;
        }

        let mut current = node;
        while let Some(parent) = self.idom(current) {
            if parent == dominator {
                return true;
            }
            if parent == current {
                break;
            }
            current = parent;
        }
        false
    }

    /// Query dominance using a caller-reused depth table.
    ///
    /// Whole-graph passes call dominance once per edge; rejecting edges that
    /// point deeper into the tree before walking any parents avoids quadratic
    /// behavior on long acyclic chains while keeping the tree itself compact.
    fn dominates_in_depths<D: DepthWord>(&self, dominator: N, node: N, depths: &[D]) -> bool {
        debug_assert_eq!(depths.len(), self.idom.len());
        if dominator == node {
            return true;
        }

        let dominator_depth = depths[dominator.index()];
        let mut node_depth = depths[node.index()];
        if dominator_depth == D::UNREACHABLE
            || node_depth == D::UNREACHABLE
            || dominator_depth >= node_depth
        {
            return false;
        }

        let mut current = node;
        while node_depth > dominator_depth {
            let Some(parent) = self.idom(current) else {
                return false;
            };
            if parent == current {
                return false;
            }
            current = parent;
            node_depth = node_depth.previous();
        }
        current == dominator
    }

    /// Query dominance using the smallest lossless internal depth table.
    pub(crate) fn dominates_with_analysis_depths(
        &self,
        dominator: N,
        node: N,
        depths: &AnalysisDepths,
    ) -> bool {
        match depths {
            AnalysisDepths::Compact(depths) => self.dominates_in_depths(dominator, node, depths),
            AnalysisDepths::Full(depths) => self.dominates_in_depths(dominator, node, depths),
        }
    }

    /// Build compact child adjacency in a caller-selected sibling order.
    pub(crate) fn child_links(&self, order: DominatorChildOrder) -> DominatorChildLinks<N> {
        let mut first_child = vec![None; self.idom.len()];
        let mut next_sibling = vec![None; self.idom.len()];

        match order {
            DominatorChildOrder::Ascending => {
                for index in (0..self.idom.len()).rev() {
                    self.prepend_child(index, &mut first_child, &mut next_sibling);
                }
            }
            DominatorChildOrder::Descending => {
                for index in 0..self.idom.len() {
                    self.prepend_child(index, &mut first_child, &mut next_sibling);
                }
            }
        }

        DominatorChildLinks {
            first_child,
            next_sibling,
        }
    }

    fn prepend_child(
        &self,
        index: usize,
        first_child: &mut [Option<N>],
        next_sibling: &mut [Option<N>],
    ) {
        let child = N::from_index(index);
        if let Some(parent) = self.idom(child).filter(|&parent| parent != child) {
            next_sibling[index] = first_child[parent.index()];
            first_child[parent.index()] = Some(child);
        }
    }

    /// Return nodes whose immediate dominator is `node`.
    #[must_use]
    pub fn children(&self, node: N) -> Vec<N> {
        self.idom
            .iter()
            .enumerate()
            .filter(|(index, parent)| **parent == Some(node) && *index != node.index())
            .map(|(index, _)| N::from_index(index))
            .collect()
    }

    /// Return a node's depth in the dominator tree.
    #[must_use]
    pub fn depth(&self, node: N) -> Option<usize> {
        if !self.reachable[node.index()] {
            return None;
        }
        let mut depth = 0;
        let mut current = node;
        loop {
            match self.idom[current.index()] {
                None => return Some(depth),
                Some(parent) if parent == current => return Some(depth),
                Some(parent) => {
                    depth += 1;
                    current = parent;
                }
            }
        }
    }

    /// Return depths indexed by dense node index.
    ///
    /// Unreachable nodes have the sentinel depth [`usize::MAX`].
    #[must_use]
    pub fn depths(&self) -> Vec<usize> {
        self.depths_in::<usize>()
    }

    fn depths_in<D: DepthWord>(&self) -> Vec<D> {
        let mut depths = vec![D::UNREACHABLE; self.idom.len()];

        for index in 0..self.idom.len() {
            if !self.reachable[index] || depths[index] != D::UNREACHABLE {
                continue;
            }

            let start = N::from_index(index);
            let mut current = start;
            let mut distance = D::ZERO;
            let base = loop {
                let current_index = current.index();
                if depths[current_index] != D::UNREACHABLE {
                    break depths[current_index];
                }
                match self.idom[current_index] {
                    None => break D::ZERO,
                    Some(parent) if parent == current => break D::ZERO,
                    Some(parent) => {
                        current = parent;
                        distance = distance.next();
                    }
                }
            };

            current = start;
            let mut depth = base.add(distance);
            loop {
                let current_index = current.index();
                if depths[current_index] != D::UNREACHABLE {
                    break;
                }
                depths[current_index] = depth;
                match self.idom[current_index] {
                    None => break,
                    Some(parent) if parent == current => break,
                    Some(parent) => {
                        current = parent;
                        depth = depth.previous();
                    }
                }
            }
        }

        depths
    }

    /// Return the smallest lossless depth table for internal analyses.
    pub(crate) fn analysis_depths(&self) -> AnalysisDepths {
        if compact_depths_supported(self.idom.len()) {
            AnalysisDepths::Compact(self.compact_depths())
        } else {
            AnalysisDepths::Full(self.depths())
        }
    }

    fn compact_depths(&self) -> Vec<u32> {
        debug_assert!(compact_depths_supported(self.idom.len()));
        self.depths_in::<u32>()
    }
}

impl<N: DenseNodeId> DominatorTree<N> {
    /// Compute the **post-dominator** tree of any graph view from its exit
    /// nodes.
    ///
    /// Post-dominators are computed by introducing a virtual exit node
    /// connected from every node in `exits`, then running the dominator
    /// algorithm on the reverse graph from that virtual exit — the
    /// multi-exit story consumers previously had to build by hand. An
    /// empty `exits` yields a tree with nothing reachable.
    ///
    /// # Panics
    ///
    /// Panics if the graph is nonempty and an exit's dense index is outside
    /// the graph.
    #[must_use]
    pub fn compute_post_from<G>(graph: &G, exits: &[N]) -> Self
    where
        G: DirectedGraphView<NodeId = N>,
    {
        let node_count = graph.node_count();
        if node_count == 0 {
            return DominatorTree {
                idom: Vec::new(),
                reachable: Vec::new(),
            };
        }
        assert!(
            exits.iter().all(|exit| exit.index() < node_count),
            "post-dominator exit index is outside the graph"
        );

        // Linear counting wins for tiny exit lists. Sort and deduplicate larger
        // arbitrary lists once: duplicate virtual edges cannot change a
        // dominator tree, and normalization keeps repeated reverse-adjacency
        // queries O(log E) instead of rescanning all E exits for every node.
        let sorted_exits = if exits.len() >= POST_DOMINATOR_BINARY_SEARCH_THRESHOLD
            && (!exits.windows(2).all(|pair| pair[0] <= pair[1])
                || exits.windows(2).any(|pair| pair[0] == pair[1]))
        {
            let mut sorted = exits.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            Some(sorted)
        } else {
            None
        };
        let exits = sorted_exits.as_deref().unwrap_or(exits);
        let binary_search_exits = exits.len() >= POST_DOMINATOR_BINARY_SEARCH_THRESHOLD;
        let reverse = PostDominatorView {
            graph,
            exits,
            binary_search_exits,
        };
        let virtual_exit = node_count;
        let reverse_dominators = DominatorTree::<usize>::compute_from(&reverse, virtual_exit);
        let idom = (0..node_count)
            .map(|node| {
                reverse_dominators
                    .idom(node)
                    .and_then(|parent| (parent != virtual_exit).then(|| N::from_index(parent)))
            })
            .collect();
        let reachable = reverse_dominators.reachable[..node_count].to_vec();
        DominatorTree { idom, reachable }
    }
}

impl DominatorTree<BlockId> {
    /// Compute the **post-dominator** tree for the given CFG.
    ///
    /// Exits are the CFG's blocks with no successors; a CFG with none
    /// (e.g. ending in an infinite loop) falls back to treating the
    /// last-allocated block as the exit, preserving long-standing
    /// behavior. See [`compute_post_from`](Self::compute_post_from) for
    /// the view-generic entry point with caller-chosen exits.
    #[must_use]
    pub fn compute_post<I>(cfg: &Cfg<I>) -> Self {
        let node_count = cfg.num_blocks();
        if node_count == 0 {
            return DominatorTree {
                idom: Vec::new(),
                reachable: Vec::new(),
            };
        }
        let mut exits: Vec<BlockId> = cfg.exit_blocks().collect();
        if exits.is_empty() {
            exits.push(BlockId::from_index(node_count - 1));
        }
        Self::compute_post_from(cfg, &exits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::graph::directed::{DirectedGraph, NodeId};
    use crate::test_util::MockInst;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct BoundedNode(u8);

    impl DenseNodeId for BoundedNode {
        fn from_index(index: usize) -> Self {
            assert!(index < 4, "bounded ID cannot represent a synthetic node");
            Self(u8::try_from(index).expect("test node index fits in u8"))
        }

        fn index(self) -> usize {
            usize::from(self.0)
        }
    }

    struct BoundedDiamond;

    impl DirectedGraphView for BoundedDiamond {
        type NodeId = BoundedNode;

        fn node_count(&self) -> usize {
            4
        }

        fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
            const EMPTY: &[u8] = &[];
            const ENTRY: &[u8] = &[1, 2];
            const TO_EXIT: &[u8] = &[3];
            let successors = match node.0 {
                0 => ENTRY,
                1 | 2 => TO_EXIT,
                _ => EMPTY,
            };
            successors.iter().copied().map(BoundedNode)
        }

        fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
            const EMPTY: &[u8] = &[];
            const FROM_ENTRY: &[u8] = &[0];
            const MERGE: &[u8] = &[1, 2];
            let predecessors = match node.0 {
                1 | 2 => FROM_ENTRY,
                3 => MERGE,
                _ => EMPTY,
            };
            predecessors.iter().copied().map(BoundedNode)
        }
    }

    #[test]
    fn single_block_cfg() {
        let cfg: Cfg<MockInst> = Cfg::new();
        let dom = DominatorTree::compute(&cfg);
        assert_eq!(dom.idom(cfg.entry()), None);
        assert!(dom.dominates(cfg.entry(), cfg.entry()));
        assert_eq!(dom.children(cfg.entry()).len(), 0);
    }

    #[test]
    fn linear_chain_dominance() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let b1 = cfg.new_block();
        let b2 = cfg.new_block();
        cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
        cfg.add_edge(b1, b2, EdgeKind::Fallthrough);
        let dom = DominatorTree::compute(&cfg);
        assert!(dom.dominates(cfg.entry(), b1));
        assert!(dom.dominates(cfg.entry(), b2));
        assert!(dom.dominates(b1, b2));
        assert!(!dom.dominates(b2, b1));
        assert_eq!(dom.idom(b1), Some(cfg.entry()));
        assert_eq!(dom.idom(b2), Some(b1));
    }

    #[test]
    fn diamond_idom_at_merge() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        let merge = cfg.new_block();
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        cfg.add_edge(a, merge, EdgeKind::Fallthrough);
        cfg.add_edge(b, merge, EdgeKind::Fallthrough);
        let dom = DominatorTree::compute(&cfg);
        // Merge block's idom should be entry (not a or b).
        assert_eq!(dom.idom(merge), Some(cfg.entry()));
        assert!(dom.dominates(cfg.entry(), a));
        assert!(dom.dominates(cfg.entry(), b));
        assert!(!dom.dominates(a, b));
        assert!(!dom.dominates(b, a));
    }

    #[test]
    fn self_loop_dominance() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        cfg.add_edge(cfg.entry(), cfg.entry(), EdgeKind::Back);
        let dom = DominatorTree::compute(&cfg);
        assert_eq!(dom.idom(cfg.entry()), None);
        assert!(dom.dominates(cfg.entry(), cfg.entry()));
    }

    #[test]
    fn unreachable_block_not_dominated() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let unreachable = cfg.new_block();
        let dom = DominatorTree::compute(&cfg);
        // Entry still dominates itself.
        assert!(dom.dominates(cfg.entry(), cfg.entry()));
        // Unreachable block has no idom.
        assert_eq!(dom.idom(unreachable), None);
    }

    #[test]
    fn depth_computation() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let b1 = cfg.new_block();
        let b2 = cfg.new_block();
        cfg.add_edge(cfg.entry(), b1, EdgeKind::Fallthrough);
        cfg.add_edge(b1, b2, EdgeKind::Fallthrough);
        let dom = DominatorTree::compute(&cfg);
        assert_eq!(dom.depth(cfg.entry()), Some(0));
        assert_eq!(dom.depth(b1), Some(1));
        assert_eq!(dom.depth(b2), Some(2));
    }

    #[test]
    fn depth_tables_match_queries_when_parents_have_larger_ids() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let leaf = graph.add_node(());
        let middle = graph.add_node(());
        let child = graph.add_node(());
        let root = graph.add_node(());
        let unreachable = graph.add_node(());
        graph.add_edge(root, child, ());
        graph.add_edge(child, middle, ());
        graph.add_edge(middle, leaf, ());

        let dom = DominatorTree::compute_from(&graph, root);
        let depths = dom.depths();
        let AnalysisDepths::Compact(compact_depths) = dom.analysis_depths() else {
            panic!("small test graph should use compact depths");
        };
        for node in [root, child, middle, leaf] {
            let expected = dom.depth(node).expect("reachable node has a depth");
            assert_eq!(depths[node.index()], expected);
            assert_eq!(
                compact_depths[node.index()],
                u32::try_from(expected).expect("test depth fits u32")
            );
        }
        assert_eq!(depths[unreachable.index()], usize::MAX);
        assert_eq!(compact_depths[unreachable.index()], u32::MAX);
    }

    #[test]
    fn full_analysis_depths_match_public_dominance_queries() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let root = graph.add_node(());
        let left = graph.add_node(());
        let right = graph.add_node(());
        let merge = graph.add_node(());
        let unreachable = graph.add_node(());
        graph.add_edge(root, left, ());
        graph.add_edge(root, right, ());
        graph.add_edge(left, merge, ());
        graph.add_edge(right, merge, ());

        let dom = DominatorTree::compute_from(&graph, root);
        let full_depths = AnalysisDepths::Full(dom.depths());
        let nodes = [root, left, right, merge, unreachable];
        for dominator in nodes {
            for node in nodes {
                assert_eq!(
                    dom.dominates_with_analysis_depths(dominator, node, &full_depths),
                    dom.dominates(dominator, node),
                    "mismatch for {dominator:?} dominating {node:?}"
                );
            }
        }
    }

    #[test]
    fn child_links_follow_the_selected_sibling_order() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let root = graph.add_node(());
        let first = graph.add_node(());
        let second = graph.add_node(());
        let grandchild = graph.add_node(());
        let unreachable = graph.add_node(());
        graph.add_edge(root, first, ());
        graph.add_edge(root, second, ());
        graph.add_edge(first, grandchild, ());

        let dom = DominatorTree::compute_from(&graph, root);
        let ascending = dom.child_links(DominatorChildOrder::Ascending);
        let descending = dom.child_links(DominatorChildOrder::Descending);

        let collect = |links: &DominatorChildLinks<NodeId>, parent| {
            let mut children = Vec::new();
            let mut child = links.first_child(parent);
            while let Some(next) = child {
                children.push(next);
                child = links.next_sibling(next);
            }
            children
        };

        assert_eq!(collect(&ascending, root), vec![first, second]);
        assert_eq!(collect(&descending, root), vec![second, first]);
        assert_eq!(collect(&ascending, first), vec![grandchild]);
        assert_eq!(collect(&ascending, unreachable).len(), 0);
    }

    #[test]
    fn compact_depth_selection_falls_back_before_the_sentinel_can_be_a_depth() {
        let largest_compact = usize::try_from(u32::MAX).expect("u32 fits supported usize targets");
        assert!(compact_depths_supported(largest_compact));
        if let Some(too_large) = largest_compact.checked_add(1) {
            assert!(!compact_depths_supported(too_large));
        }
    }

    #[test]
    fn post_dominators_over_a_consumer_view() {
        // Diamond in consumer storage: a -> {b, c} -> d. Everything is
        // post-dominated by d; the branch is post-dominated by the merge.
        let mut graph = DirectedGraph::<&str, ()>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");
        graph.add_edge(a, b, ());
        graph.add_edge(a, c, ());
        graph.add_edge(b, d, ());
        graph.add_edge(c, d, ());

        let post = DominatorTree::compute_post_from(&graph, &[d]);
        assert_eq!(post.idom(a), Some(d));
        assert_eq!(post.idom(b), Some(d));
        assert_eq!(post.idom(c), Some(d));
        assert!(post.dominates(d, a), "d post-dominates the entry");

        // No exits: nothing is reachable on the reverse graph.
        let empty = DominatorTree::compute_post_from(&graph, &[]);
        assert_eq!(empty.idom(a), None);
        assert_eq!(empty.depth(a), None);
    }

    #[test]
    #[should_panic(expected = "post-dominator exit index is outside the graph")]
    fn post_dominators_reject_an_exit_outside_the_graph() {
        let mut graph = DirectedGraph::<(), ()>::new();
        graph.add_node(());

        let outside = NodeId::from_raw(
            u32::try_from(graph.node_count()).expect("test graph size fits in u32"),
        );
        let _ = DominatorTree::compute_post_from(&graph, &[outside]);
    }

    #[test]
    fn post_dominator_view_preserves_duplicate_exit_edges() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let exit = graph.add_node(());
        let exits = [exit, exit];
        let reverse = PostDominatorView {
            graph: &graph,
            exits: &exits,
            binary_search_exits: false,
        };
        let virtual_exit = graph.node_count();

        assert_eq!(
            reverse.successors(virtual_exit).collect::<Vec<_>>(),
            vec![exit.index(), exit.index()]
        );
        assert_eq!(
            reverse.predecessors(exit.index()).collect::<Vec<_>>(),
            vec![virtual_exit, virtual_exit]
        );
    }

    #[test]
    fn post_dominators_do_not_require_consumer_ids_for_the_virtual_exit() {
        let post = DominatorTree::compute_post_from(&BoundedDiamond, &[BoundedNode(3)]);
        assert_eq!(post.idom(BoundedNode(0)), Some(BoundedNode(3)));
        assert_eq!(post.idom(BoundedNode(1)), Some(BoundedNode(3)));
        assert_eq!(post.idom(BoundedNode(2)), Some(BoundedNode(3)));
        assert!(post.dominates(BoundedNode(3), BoundedNode(0)));
    }

    #[test]
    fn post_dominators_accept_large_unsorted_exit_lists() {
        let mut graph = DirectedGraph::<(), ()>::new();
        let entry = graph.add_node(());
        let mut exits = Vec::new();
        for _ in 0..16 {
            let exit = graph.add_node(());
            graph.add_edge(entry, exit, ());
            exits.push(exit);
        }

        let ordered = DominatorTree::compute_post_from(&graph, &exits);
        exits.reverse();
        let reversed = DominatorTree::compute_post_from(&graph, &exits);
        assert_eq!(reversed, ordered);
    }

    #[test]
    fn children_returns_immediate_children() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        let c = cfg.new_block();
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        cfg.add_edge(a, c, EdgeKind::Fallthrough);
        let dom = DominatorTree::compute(&cfg);
        let mut entry_children = dom.children(cfg.entry());
        entry_children.sort();
        assert_eq!(entry_children.len(), 2);
        assert!(entry_children.contains(&a));
        assert!(entry_children.contains(&b));
        assert_eq!(dom.children(a), vec![c]);
    }
}
