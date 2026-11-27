//! A graph behind a virtual root, as a view.

extern crate alloc;
use alloc::vec::Vec;

use super::{DenseId, GraphView, RootedView};

/// Adapter putting one extra node, with an edge to each of a list of roots,
/// in front of a view.
///
/// An analysis that starts from one root sees only what that root reaches.
/// Code the root misses (a handler pad no exception edge enters, a filter
/// funclet the runtime invokes, a loop nothing enters) is reached from the
/// virtual root like any other, and the virtual root over a [`Reversed`]
/// view is the virtual exit a post-dominator tree needs when a graph has
/// several exits. Adding a node with only outgoing edges adds no path from
/// any real node, so every fact about a node one of the roots reaches is
/// unchanged.
///
/// Nodes are the dense indices of the underlying view, as `usize`, with the
/// virtual root one past its [`node_bound`](GraphView::node_bound): the
/// underlying identity type is never asked to name an index past its own
/// bound. The virtual root leads to the roots in the order given, and a root
/// named twice has two edges from it.
///
/// [`Reversed`]: super::Reversed
///
/// # Examples
///
/// ```
/// use cfglib::{DominatorTree, Graph, GraphView, VirtualRoot};
///
/// // Two disconnected entries meet only at the virtual root.
/// let mut graph = Graph::<&str, ()>::new();
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let join = graph.add_node("join");
/// graph.add_edge(a, join, ());
/// graph.add_edge(b, join, ());
///
/// let roots = [a, b];
/// let view = VirtualRoot::new(&graph, &roots);
/// let dominators = DominatorTree::compute(&view);
/// assert_eq!(dominators.idom(join.index()), Some(view.virtual_root()));
/// assert_eq!(view.node_bound(), graph.node_bound() + 1);
/// ```
#[derive(Debug, Clone)]
pub struct VirtualRoot<'g, G: GraphView> {
    graph: &'g G,
    roots: &'g [G::NodeId],
    named: Named<'g>,
}

/// How a node finds out how many edges it has from the virtual root.
#[derive(Debug, Clone)]
enum Named<'g> {
    /// The root indices in ascending order, searched per query.
    Sorted(Vec<usize>),
    /// A caller-owned table that is `true` exactly at the roots.
    Mask(&'g [bool]),
}

impl<'g, G: GraphView> VirtualRoot<'g, G> {
    /// Put a virtual root in front of `roots`, in the order given.
    ///
    /// The view keeps a sorted copy of the roots so that
    /// [`predecessors`](GraphView::predecessors) finds the virtual root by
    /// binary search. A caller that already holds a membership table answers
    /// that in constant time with [`masked`](Self::masked).
    ///
    /// # Panics
    ///
    /// Panics when a root is not below the node bound of `graph`.
    #[must_use]
    pub fn new(graph: &'g G, roots: &'g [G::NodeId]) -> Self {
        let mut sorted: Vec<usize> = roots.iter().map(|root| root.index()).collect();
        sorted.sort_unstable();
        assert!(
            sorted.last().is_none_or(|&last| last < graph.node_bound()),
            "a root of a virtual root is out of range"
        );
        Self {
            graph,
            roots,
            named: Named::Sorted(sorted),
        }
    }

    /// Put a virtual root in front of `roots`, reading membership from
    /// `is_root`, which must be `true` exactly at the nodes of `roots`, each
    /// named once.
    ///
    /// # Panics
    ///
    /// Panics when `is_root` is shorter than the node bound of `graph`, or
    /// when a node of `roots` is not marked in it.
    #[must_use]
    pub fn masked(graph: &'g G, roots: &'g [G::NodeId], is_root: &'g [bool]) -> Self {
        assert!(
            is_root.len() >= graph.node_bound(),
            "the root table of a virtual root is shorter than the node bound"
        );
        assert!(
            roots.iter().all(|root| is_root[root.index()]),
            "a root of a virtual root is not marked in its root table"
        );
        Self {
            graph,
            roots,
            named: Named::Mask(is_root),
        }
    }

    /// Borrow the underlying view.
    #[must_use]
    pub const fn graph(&self) -> &'g G {
        self.graph
    }

    /// The roots the virtual root leads to, in the order given.
    #[must_use]
    pub const fn roots(&self) -> &'g [G::NodeId] {
        self.roots
    }

    /// The dense index of the virtual root: the node bound of the underlying
    /// view.
    #[inline]
    #[must_use]
    pub fn virtual_root(&self) -> usize {
        self.graph.node_bound()
    }

    /// How many edges `node`, a node of the underlying view, has from the
    /// virtual root.
    #[inline]
    fn edges_from_root(&self, node: usize) -> usize {
        match &self.named {
            Named::Sorted(sorted) => {
                let first = sorted.partition_point(|&root| root < node);
                sorted[first..].partition_point(|&root| root == node)
            }
            Named::Mask(is_root) => usize::from(is_root[node]),
        }
    }
}

impl<G: GraphView> GraphView for VirtualRoot<'_, G> {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.virtual_root() + 1
    }

    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph
            .node_ids()
            .map(DenseId::index)
            .chain(core::iter::once(self.virtual_root()))
    }

    #[inline]
    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        let bound = self.virtual_root();
        let original = (node < bound).then(|| G::NodeId::from_index(node));
        (node == bound)
            .then_some(self.roots)
            .into_iter()
            .flatten()
            .map(|root| root.index())
            .chain(
                original
                    .into_iter()
                    .flat_map(|node| self.graph.successors(node).map(DenseId::index)),
            )
    }

    #[inline]
    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        let bound = self.virtual_root();
        let original = (node < bound).then(|| G::NodeId::from_index(node));
        let from_root = if node < bound {
            self.edges_from_root(node)
        } else {
            0
        };
        core::iter::repeat_n(bound, from_root).chain(
            original
                .into_iter()
                .flat_map(|node| self.graph.predecessors(node).map(DenseId::index)),
        )
    }
}

impl<G: GraphView> RootedView for VirtualRoot<'_, G> {
    fn root(&self) -> Self::NodeId {
        self.virtual_root()
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec::Vec;

    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;

    /// Forward and reverse adjacency must be one edge multiset, however the
    /// roots are named.
    fn assert_symmetric<G: GraphView>(view: &VirtualRoot<'_, G>) {
        let mut forward: Vec<(usize, usize)> = view
            .node_ids()
            .flat_map(|node| view.successors(node).map(move |target| (node, target)))
            .collect();
        let mut reverse: Vec<(usize, usize)> = view
            .node_ids()
            .flat_map(|node| view.predecessors(node).map(move |source| (source, node)))
            .collect();
        forward.sort_unstable();
        reverse.sort_unstable();
        assert_eq!(forward, reverse);
    }

    #[test]
    fn every_root_has_one_edge_from_the_virtual_root() {
        let mut cfg = Cfg::<u8>::new();
        let pad = cfg.new_block();
        let other = cfg.new_block();
        cfg.add_edge(pad, other, EdgeKind::Fallthrough);

        let roots = [cfg.entry(), pad, other];
        let view = VirtualRoot::new(&cfg, &roots);
        assert_symmetric(&view);
        let from_root: Vec<usize> = view.successors(view.root()).collect();
        assert_eq!(from_root, [cfg.entry().index(), pad.index(), other.index()]);
        assert_eq!(view.predecessors(cfg.entry().index()).count(), 1);
    }

    #[test]
    fn the_root_precedes_only_the_blocks_it_names() {
        let mut cfg = Cfg::<u8>::new();
        let named = cfg.new_block();
        let unnamed = cfg.new_block();
        let roots = [cfg.entry(), named];
        let view = VirtualRoot::new(&cfg, &roots);
        assert_eq!(view.predecessors(named.index()).count(), 1);
        assert_eq!(view.predecessors(unnamed.index()).count(), 0);
        assert_eq!(view.node_bound(), cfg.block_bound() + 1);
        assert_eq!(view.root(), cfg.block_bound());
    }

    #[test]
    fn roots_keep_the_order_they_are_given_in() {
        let mut cfg = Cfg::<u8>::new();
        let below = cfg.new_block();
        let entry = cfg.new_block();
        let above = cfg.new_block();
        let unnamed = cfg.new_block();
        cfg.set_entry(entry);

        let roots = [above, entry, below];
        let view = VirtualRoot::new(&cfg, &roots);
        assert_symmetric(&view);
        let from_root: Vec<usize> = view.successors(view.root()).collect();
        assert_eq!(from_root, [above.index(), entry.index(), below.index()]);
        for named in [below, entry, above] {
            assert_eq!(view.predecessors(named.index()).count(), 1);
        }
        assert_eq!(view.predecessors(unnamed.index()).count(), 0);
    }

    #[test]
    fn a_root_named_twice_has_two_edges_both_ways() {
        let mut cfg = Cfg::<u8>::new();
        let pad = cfg.new_block();
        let roots = [pad, cfg.entry(), pad];
        let view = VirtualRoot::new(&cfg, &roots);
        assert_symmetric(&view);
        assert_eq!(view.predecessors(pad.index()).count(), 2);
    }

    #[test]
    fn a_mask_answers_like_the_sorted_roots() {
        let mut cfg = Cfg::<u8>::new();
        let pad = cfg.new_block();
        let other = cfg.new_block();
        cfg.add_edge(cfg.entry(), other, EdgeKind::Fallthrough);
        let roots = [pad, cfg.entry()];
        let mut is_root = alloc::vec![false; cfg.block_bound()];
        for root in roots {
            is_root[root.index()] = true;
        }
        let masked = VirtualRoot::masked(&cfg, &roots, &is_root);
        let sorted = VirtualRoot::new(&cfg, &roots);
        assert_symmetric(&masked);
        for node in sorted.node_ids() {
            assert!(masked.predecessors(node).eq(sorted.predecessors(node)));
            assert!(masked.successors(node).eq(sorted.successors(node)));
        }
    }

    #[test]
    fn a_removed_block_is_not_a_node() {
        let mut cfg = Cfg::<u8>::new();
        let gone = cfg.new_block();
        let kept = cfg.new_block();
        cfg.remove_block(gone);
        let roots = [cfg.entry(), kept];
        let view = VirtualRoot::new(&cfg, &roots);
        assert_symmetric(&view);
        assert!(view.node_ids().all(|node| node != gone.index()));
    }

    #[test]
    #[should_panic(expected = "not marked in its root table")]
    fn a_mask_that_misses_a_root_is_rejected() {
        let mut cfg = Cfg::<u8>::new();
        let pad = cfg.new_block();
        let roots = [pad];
        let is_root = alloc::vec![false; cfg.block_bound()];
        let _ = VirtualRoot::masked(&cfg, &roots, &is_root);
    }
}
