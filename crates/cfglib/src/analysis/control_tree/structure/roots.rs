//! The graph the post-dominator tree reads.
//!
//! A post-dominator tree holds only the blocks its root reaches. Code that
//! the entry cannot reach but the structurer still emits — a filter funclet
//! the runtime invokes, a handler pad with no exception edge, the body of
//! such a pad — would have no merge points. This view roots the graph at a
//! virtual block that precedes the entry and each such block, so their
//! branches merge like any other.
//!
//! Adding a block with only outgoing edges adds no path from any real block,
//! so every fact about a block the entry reaches is unchanged.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::view::{GraphView, RootedView};

/// A [`Cfg`] behind a virtual root, over dense block indices.
pub(super) struct VirtualRoot<'cfg, I, E> {
    cfg: &'cfg Cfg<I, E>,
    /// The entry first, then the extra roots in ascending order, each once.
    successors: Vec<u32>,
    /// The index of the virtual root: one past the last block slot.
    root: u32,
}

impl<'cfg, I, E> VirtualRoot<'cfg, I, E> {
    /// Root `cfg` at a virtual block that precedes the entry and `extra`.
    ///
    /// A block named twice, or the entry named again, has one edge from the
    /// root: forward and reverse adjacency must describe the same edges.
    pub(super) fn new(cfg: &'cfg Cfg<I, E>, extra: impl IntoIterator<Item = BlockId>) -> Self {
        let root = u32::try_from(cfg.block_bound()).expect("block bound exceeds u32::MAX");
        let extra: BTreeSet<BlockId> = extra
            .into_iter()
            .filter(|&block| block != cfg.entry())
            .collect();
        let successors = core::iter::once(cfg.entry())
            .chain(extra)
            .map(BlockId::raw)
            .collect();
        Self {
            cfg,
            successors,
            root,
        }
    }

    /// Whether the virtual root has an edge to `node`: the entry, or one of
    /// the extra roots, which `new` stores in ascending order.
    fn is_named(&self, node: u32) -> bool {
        let (entry, extra) = self
            .successors
            .split_first()
            .expect("the root always names the entry");
        node == *entry || extra.binary_search(&node).is_ok()
    }
}

impl<I, E> GraphView for VirtualRoot<'_, I, E> {
    type NodeId = u32;

    fn node_bound(&self) -> usize {
        self.cfg.block_bound() + 1
    }

    fn node_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.cfg
            .block_ids()
            .map(BlockId::raw)
            .chain(core::iter::once(self.root))
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        let roots: &[u32] = if node == self.root {
            &self.successors
        } else {
            &[]
        };
        let blocks = (node != self.root)
            .then(|| self.cfg.successors(BlockId::from_raw(node)))
            .into_iter()
            .flatten();
        roots.iter().copied().chain(blocks.map(BlockId::raw))
    }

    fn predecessors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        let from_root = self.is_named(node).then_some(self.root);
        let blocks = (node != self.root)
            .then(|| self.cfg.predecessors(BlockId::from_raw(node)))
            .into_iter()
            .flatten();
        from_root.into_iter().chain(blocks.map(BlockId::raw))
    }
}

impl<I, E> RootedView for VirtualRoot<'_, I, E> {
    fn root(&self) -> u32 {
        self.root
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::edge::EdgeKind;

    /// Forward and reverse adjacency must be one edge multiset, however the
    /// extra roots are named.
    fn assert_symmetric(view: &VirtualRoot<'_, u8, ()>) {
        let mut forward: Vec<(u32, u32)> = view
            .node_ids()
            .flat_map(|node| view.successors(node).map(move |target| (node, target)))
            .collect();
        let mut reverse: Vec<(u32, u32)> = view
            .node_ids()
            .flat_map(|node| view.predecessors(node).map(move |source| (source, node)))
            .collect();
        forward.sort_unstable();
        reverse.sort_unstable();
        assert_eq!(forward, reverse);
    }

    #[test]
    fn a_block_named_twice_has_one_edge_from_the_root() {
        let mut cfg = Cfg::<u8>::new();
        let pad = cfg.new_block();
        let other = cfg.new_block();
        cfg.add_edge(pad, other, EdgeKind::Fallthrough);

        let view = VirtualRoot::new(&cfg, [pad, other, pad, pad]);
        assert_symmetric(&view);
        let from_root: Vec<u32> = view.successors(view.root()).collect();
        assert_eq!(from_root, [cfg.entry().raw(), pad.raw(), other.raw()]);
    }

    #[test]
    fn the_entry_named_as_an_extra_root_is_not_repeated() {
        let mut cfg = Cfg::<u8>::new();
        let pad = cfg.new_block();
        let view = VirtualRoot::new(&cfg, [cfg.entry(), pad, cfg.entry()]);
        assert_symmetric(&view);
        let from_root: Vec<u32> = view.successors(view.root()).collect();
        assert_eq!(from_root, [cfg.entry().raw(), pad.raw()]);
        assert_eq!(view.predecessors(cfg.entry().raw()).count(), 1);
    }

    #[test]
    fn the_root_precedes_only_the_blocks_it_names() {
        let mut cfg = Cfg::<u8>::new();
        let named = cfg.new_block();
        let unnamed = cfg.new_block();
        let view = VirtualRoot::new(&cfg, [named]);
        assert_eq!(view.predecessors(named.raw()).count(), 1);
        assert_eq!(view.predecessors(unnamed.raw()).count(), 0);
        assert_eq!(view.node_bound(), cfg.block_bound() + 1);
    }

    #[test]
    fn extra_roots_on_both_sides_of_a_relocated_entry_have_one_edge_each() {
        let mut cfg = Cfg::<u8>::new();
        let below = cfg.new_block();
        let entry = cfg.new_block();
        let above = cfg.new_block();
        let unnamed = cfg.new_block();
        cfg.set_entry(entry);

        let view = VirtualRoot::new(&cfg, [above, below, entry, above]);
        assert_symmetric(&view);
        let from_root: Vec<u32> = view.successors(view.root()).collect();
        assert_eq!(from_root, [entry.raw(), below.raw(), above.raw()]);
        for named in [below, entry, above] {
            assert_eq!(view.predecessors(named.raw()).count(), 1);
        }
        assert_eq!(view.predecessors(unnamed.raw()).count(), 0);
    }
}
