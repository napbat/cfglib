//! A graph behind a virtual root.
//!
//! A dominator or post-dominator tree holds only the nodes its root reaches.
//! Code the root misses (a handler pad no exception edge enters, a filter
//! funclet the runtime invokes, a loop nothing enters) would have no tree.
//! [`VirtualRoot`] adds one node, with only outgoing edges, that precedes the
//! graph's own root and each block of such code, so their branches meet like
//! any other.
//!
//! Adding a node with only outgoing edges adds no path from any real node, so
//! every fact about a node the root reaches is unchanged.

use crate::graph::view::{DenseId, GraphView, RootedView};

/// A graph behind a virtual root, over dense indices.
///
/// Nodes are the graph's dense indices, with the virtual root one past the
/// graph's [`node_bound`](GraphView::node_bound).
pub(crate) struct VirtualRoot<'a, G: GraphView> {
    graph: &'a G,
    roots: &'a [G::NodeId],
}

impl<'a, G: GraphView> VirtualRoot<'a, G> {
    /// Put a virtual root in front of `roots`, which must hold each node
    /// once, in ascending order.
    ///
    /// The order is what lets [`predecessors`](GraphView::predecessors) find
    /// the virtual root by binary search. A node named twice would have two
    /// edges from the root forward and one backward, so a list that is not
    /// strictly ascending is rejected rather than trusted.
    ///
    /// # Panics
    ///
    /// Panics when `roots` is not strictly ascending.
    pub(crate) fn new(graph: &'a G, roots: &'a [G::NodeId]) -> Self {
        assert!(
            roots.windows(2).all(|pair| pair[0] < pair[1]),
            "the roots of a virtual root must be strictly ascending"
        );
        Self { graph, roots }
    }

    /// The dense index of the virtual root: one past the last node slot.
    fn virtual_root(&self) -> usize {
        self.graph.node_bound()
    }
}

impl<G: GraphView> GraphView for VirtualRoot<'_, G> {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.graph.node_bound() + 1
    }

    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph
            .node_ids()
            .map(DenseId::index)
            .chain(core::iter::once(self.virtual_root()))
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        let original_bound = self.graph.node_bound();
        let original = (node < original_bound).then(|| G::NodeId::from_index(node));
        (node == original_bound)
            .then_some(self.roots)
            .into_iter()
            .flatten()
            .copied()
            .map(DenseId::index)
            .chain(
                original
                    .into_iter()
                    .flat_map(|block| self.graph.successors(block).map(DenseId::index)),
            )
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        let original_bound = self.graph.node_bound();
        let original = (node < original_bound).then(|| G::NodeId::from_index(node));
        let virtual_predecessor = original
            .filter(|block| self.roots.binary_search(block).is_ok())
            .map(|_| original_bound);
        virtual_predecessor.into_iter().chain(
            original
                .into_iter()
                .flat_map(|block| self.graph.predecessors(block).map(DenseId::index)),
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
    use crate::block::BlockId;
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
    fn roots_on_both_sides_of_a_relocated_entry_have_one_edge_each() {
        let mut cfg = Cfg::<u8>::new();
        let below = cfg.new_block();
        let entry = cfg.new_block();
        let above = cfg.new_block();
        let unnamed = cfg.new_block();
        cfg.set_entry(entry);

        let roots = [below, entry, above];
        let view = VirtualRoot::new(&cfg, &roots);
        assert_symmetric(&view);
        let from_root: Vec<usize> = view.successors(view.root()).collect();
        assert_eq!(from_root, [below.index(), entry.index(), above.index()]);
        for named in [below, entry, above] {
            assert_eq!(view.predecessors(named.index()).count(), 1);
        }
        assert_eq!(view.predecessors(unnamed.index()).count(), 0);
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
    #[should_panic(expected = "strictly ascending")]
    fn a_root_named_twice_is_rejected() {
        let mut cfg = Cfg::<u8>::new();
        let pad = cfg.new_block();
        let roots = [cfg.entry(), pad, pad];
        let _ = VirtualRoot::new(&cfg, &roots);
    }

    #[test]
    #[should_panic(expected = "strictly ascending")]
    fn roots_out_of_order_are_rejected() {
        let mut cfg = Cfg::<u8>::new();
        let pad = cfg.new_block();
        let roots: [BlockId; 2] = [pad, cfg.entry()];
        let _ = VirtualRoot::new(&cfg, &roots);
    }
}
