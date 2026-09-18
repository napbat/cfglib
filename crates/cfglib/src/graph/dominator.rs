//! Dominator tree computation using the Cooper-Harvey-Kennedy iterative
//! algorithm.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::traverse::{PostorderScratch, TraversalDirection, reverse_postorder_in};
use crate::graph::view::{DenseId, GraphView, RootedView};

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

impl<N: DenseId> DominatorChildLinks<N> {
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
struct PostDominatorView<'g, G: GraphView> {
    graph: &'g G,
    exits: &'g [G::NodeId],
    /// Small exit lists use linear multiplicity counts; large lists are
    /// normalized before constructing the view and use binary search.
    binary_search_exits: bool,
}

const POST_DOMINATOR_BINARY_SEARCH_THRESHOLD: usize = 16;

impl<G: GraphView> PostDominatorView<'_, G> {
    #[inline]
    fn exit_multiplicity(&self, node: G::NodeId) -> usize {
        if self.binary_search_exits {
            usize::from(self.exits.binary_search(&node).is_ok())
        } else {
            self.exits.iter().filter(|&&exit| exit == node).count()
        }
    }
}

impl<G: GraphView> GraphView for PostDominatorView<'_, G> {
    // The virtual exit is private implementation state.  Use `usize` for the
    // augmented view so callers' IDs are never asked to represent the
    // out-of-range index at `graph.node_bound()`.
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.graph.node_bound() + 1
    }

    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph
            .node_ids()
            .map(DenseId::index)
            .chain(core::iter::once(self.graph.node_bound()))
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        let original_count = self.graph.node_bound();
        let is_virtual = node == original_count;
        let original = (node < original_count).then(|| G::NodeId::from_index(node));
        is_virtual
            .then_some(self.exits)
            .into_iter()
            .flatten()
            .copied()
            .map(DenseId::index)
            .chain(
                original
                    .into_iter()
                    .flat_map(move |node| self.graph.predecessors(node).map(DenseId::index)),
            )
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        let original_count = self.graph.node_bound();
        let original = (node < original_count).then(|| G::NodeId::from_index(node));
        let exit_multiplicity = original.map_or(0, |node| self.exit_multiplicity(node));
        original
            .into_iter()
            .flat_map(move |node| self.graph.successors(node).map(DenseId::index))
            .chain(core::iter::repeat_n(original_count, exit_multiplicity))
    }
}

/// Every buffer [`DominatorTree::compute_in`] fills, owned by the caller.
///
/// A whole-codebase pass computes a dominator tree once per procedure, and
/// most procedures are a handful of blocks. At that size the tree is not the
/// cost, the *call* is: a reverse postorder with its own visited row,
/// frontier, and adjacency buffer, then a position index and a working parent
/// array, each a malloc and a free for a four-block answer. This moves all
/// five out of the call, leaving only the two exact-sized arrays the tree
/// itself keeps.
///
/// Hand one scratch to every [`compute_in`](DominatorTree::compute_in) of the
/// pass. The buffers grow to the largest graph the pass meets and are then
/// reused by every graph after it, so the whole pass allocates a bounded
/// number of times instead of a few times per procedure.
///
/// # One scratch, any graph
///
/// Nodes are held as dense indices rather than as a node-id type, so the
/// scratch carries no type parameter: one instance serves a [`Cfg`], a
/// [`Graph`](crate::Graph), and a consumer-defined [`GraphView`], in any
/// order, and it is [`Send`], so a worker thread can own one for the whole
/// corpus it is handed.
///
/// # Sizing
///
/// Nothing is fixed at construction and nothing is released by a call. Every
/// buffer is cleared and resized on entry, so a scratch used on a large graph
/// and then on a small one is correct and allocation-free, and the space it
/// holds is the high-water mark of the sequence.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, DominatorScratch, DominatorTree, EdgeKind};
///
/// let mut cfg = Cfg::<u32>::new();
/// let entry = cfg.entry();
/// let then_block = cfg.new_block();
/// cfg.add_edge(entry, then_block, EdgeKind::ConditionalTrue);
///
/// let mut scratch = DominatorScratch::new();
/// let tree = DominatorTree::compute_in(&mut scratch, &cfg);
/// assert_eq!(tree, DominatorTree::compute(&cfg));
/// ```
#[derive(Debug, Clone, Default)]
pub struct DominatorScratch {
    /// The reverse postorder the iteration walks, and the walk's own buffers.
    postorder: PostorderScratch,
    /// Each node's position in that order, or [`usize::MAX`] when the root
    /// does not reach it.
    order_index: Vec<usize>,
    /// The working immediate-dominator array, in order positions rather than
    /// node indices, which is what makes the intersection a walk down two
    /// integers.
    dominators: Vec<Option<usize>>,
}

impl DominatorScratch {
    /// Scratch holding nothing yet.
    ///
    /// Every buffer is sized by the first graph it sees, so there is no node
    /// count to state here.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<N: DenseId> DominatorTree<N> {
    /// Constructs an internal dominator forest from already validated dense
    /// parent and reachability tables.
    pub(crate) fn from_forest_parts(idom: Vec<Option<N>>, reachable: Vec<bool>) -> Self {
        debug_assert_eq!(idom.len(), reachable.len());
        Self { idom, reachable }
    }

    /// Compute the dominator tree of a rooted graph view using the iterative
    /// algorithm by Cooper, Harvey, and Kennedy.
    #[must_use]
    pub fn compute<G>(graph: &G) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        Self::compute_in(&mut DominatorScratch::new(), graph)
    }

    /// Compute the dominator tree of a rooted graph view over caller-owned
    /// working storage.
    ///
    /// This is [`compute`](Self::compute) with the call's buffers taken out of
    /// it. The tree it returns is the same one, built by the same code: the
    /// allocating entry point is this function over a fresh
    /// [`DominatorScratch`].
    #[must_use]
    pub fn compute_in<G>(scratch: &mut DominatorScratch, graph: &G) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        Self::compute_from_in(scratch, graph, graph.root())
    }

    /// Recompute the dominator tree and report which nodes' immediate
    /// dominators changed relative to `previous`.
    ///
    /// A graph edit (edge insertion or removal) invalidates the tree; this
    /// recomputes it and diffs against `previous`, so downstream analyses can
    /// be updated selectively instead of from scratch.
    ///
    /// # Panics
    ///
    /// Panics when `previous` was computed over a smaller node space than
    /// `graph` currently has.
    #[must_use]
    pub fn compute_with_diff<G>(graph: &G, previous: &Self) -> (Self, Vec<N>)
    where
        G: RootedView<NodeId = N>,
    {
        let next = Self::compute(graph);
        let changed = (0..graph.node_bound())
            .map(N::from_index)
            .filter(|&node| previous.idom(node) != next.idom(node))
            .collect();
        (next, changed)
    }

    /// Compute dominators for any directed graph view from an explicit root.
    #[must_use]
    pub fn compute_from<G>(graph: &G, root: N) -> Self
    where
        G: GraphView<NodeId = N>,
    {
        Self::compute_from_in(&mut DominatorScratch::new(), graph, root)
    }

    /// Compute dominators from an explicit root over caller-owned working
    /// storage.
    ///
    /// This is [`compute_from`](Self::compute_from) with the call's buffers
    /// taken out of it; see [`DominatorScratch`].
    #[must_use]
    pub fn compute_from_in<G>(scratch: &mut DominatorScratch, graph: &G, root: N) -> Self
    where
        G: GraphView<NodeId = N>,
    {
        reverse_postorder_in(
            &mut scratch.postorder,
            graph,
            root,
            TraversalDirection::Outgoing,
        );
        let order = &scratch.postorder.order;
        let node_count = graph.node_bound();
        scratch.order_index.clear();
        scratch.order_index.resize(node_count, usize::MAX);
        let order_index = &mut scratch.order_index;
        for (index, node) in order.iter().copied().enumerate() {
            order_index[node] = index;
        }

        let dominators = &mut scratch.dominators;
        dominators.clear();
        dominators.resize(order.len(), None);
        dominators[order_index[root.index()]] = Some(order_index[root.index()]);

        let mut changed = true;
        while changed {
            changed = false;
            for node in order.iter().copied().filter(|&node| node != root.index()) {
                let node_order = order_index[node];
                let mut new_parent_index = None;
                for predecessor in graph.predecessors(N::from_index(node)) {
                    let predecessor_order = order_index[predecessor.index()];
                    if predecessor_order == usize::MAX || dominators[predecessor_order].is_none() {
                        continue;
                    }

                    new_parent_index = Some(match new_parent_index {
                        None => predecessor_order,
                        Some(parent) if predecessor_order == parent => parent,
                        Some(parent) => Self::intersect(dominators, predecessor_order, parent),
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
        for (index, parent) in dominators.iter().copied().enumerate() {
            let node = order[index];
            immediate[node] = parent.map(|parent_index| N::from_index(order[parent_index]));
        }
        immediate[root.index()] = None;
        let mut reachable = vec![false; node_count];
        for &node in order {
            reachable[node] = true;
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

impl<N: DenseId> DominatorTree<N> {
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
        G: GraphView<NodeId = N>,
    {
        let node_count = graph.node_bound();
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
    /// last-allocated live block as the exit, preserving long-standing
    /// behavior. See [`compute_post_from`](Self::compute_post_from) for
    /// the view-generic entry point with caller-chosen exits.
    #[must_use]
    pub fn compute_post<I, E>(cfg: &Cfg<I, E>) -> Self {
        if cfg.block_count() == 0 {
            return DominatorTree {
                idom: Vec::new(),
                reachable: Vec::new(),
            };
        }
        let mut exits: Vec<BlockId> = cfg.exit_blocks().collect();
        if exits.is_empty() {
            // The fallback names a block, not a quantity: a removed block
            // keeps its slot, so the highest live identity is the last one
            // `block_ids` yields, not `block_count() - 1`.
            exits.extend(cfg.block_ids().last());
        }
        Self::compute_post_from(cfg, &exits)
    }
}

#[cfg(test)]
mod tests;
