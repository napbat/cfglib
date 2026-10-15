//! The graph the post-dominator tree reads.
//!
//! A post-dominator tree holds only the blocks its root reaches. Code that
//! the entry cannot reach but the structurer still emits — a filter funclet
//! the runtime invokes, a handler entry with no exception edge — would have no
//! merge points. This view roots the graph at a virtual block that precedes
//! the entry and each such block, so their branches merge like any other.
//!
//! Adding a block with only outgoing edges adds no path from any real block,
//! so every fact about a block the entry reaches is unchanged.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::view::{GraphView, RootedView};

/// A [`Cfg`] behind a virtual root, over dense block indices.
pub(super) struct VirtualRoot<'cfg, I, E> {
    cfg: &'cfg Cfg<I, E>,
    /// The entry first, then the extra roots in the order given.
    successors: Vec<u32>,
    /// The index of the virtual root: one past the last block slot.
    root: u32,
}

impl<'cfg, I, E> VirtualRoot<'cfg, I, E> {
    /// Root `cfg` at a virtual block that precedes the entry and `extra`.
    pub(super) fn new(cfg: &'cfg Cfg<I, E>, extra: impl IntoIterator<Item = BlockId>) -> Self {
        let root = u32::try_from(cfg.block_bound()).expect("block bound exceeds u32::MAX");
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
        let from_root = self.successors.contains(&node).then_some(self.root);
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
