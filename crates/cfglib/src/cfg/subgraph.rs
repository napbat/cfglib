//! Subgraph extraction from a [`Cfg`].

extern crate alloc;

use alloc::vec::Vec;

use crate::block::BlockId;
use crate::rewrite::Rewrite;

use super::Cfg;

impl<I: Clone, E: Clone> Cfg<I, E> {
    /// Extract a sub-CFG containing only the specified blocks.
    ///
    /// The resulting CFG preserves edges between the selected blocks
    /// and remaps block IDs to be contiguous starting from 0.
    /// The first block in `blocks` becomes the entry.
    ///
    /// Edges that cross the boundary (one endpoint outside the set)
    /// are dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let b0 = cfg.entry();
    /// let b1 = cfg.new_block();
    /// let b2 = cfg.new_block();
    /// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
    /// cfg.add_edge(b1, b2, EdgeKind::Fallthrough);
    ///
    /// let sub = cfg.subgraph(&[b0, b1]);
    /// assert_eq!(sub.block_count(), 2);
    /// assert_eq!(sub.edge_count(), 1); // b1→b2 dropped
    /// ```
    #[must_use]
    pub fn subgraph(&self, blocks: &[BlockId]) -> Self {
        self.subgraph_mapped(blocks).0
    }

    /// Extract a sub-CFG and return a complete old-to-new identity mapping.
    #[must_use]
    pub fn subgraph_mapped(&self, blocks: &[BlockId]) -> (Self, Rewrite) {
        let mut mapping = Rewrite::new();
        if blocks.is_empty() {
            for block in self.block_ids() {
                mapping.record_block(block, []);
            }
            for edge in self.edge_ids() {
                mapping.record_edge(edge, []);
            }
            let empty = Self::with_edge_payload();
            mapping.record_created_block(empty.entry());
            return (empty, mapping);
        }

        let mut new_cfg = Self::with_edge_payload();

        // Map old BlockId → new BlockId via a dense vector (O(1) lookup).
        let mut id_map: Vec<Option<BlockId>> = alloc::vec![None; self.block_bound()];
        let entry = new_cfg.entry();
        id_map[blocks[0].index()] = Some(entry);
        mapping.record_block(blocks[0], [entry]);
        mapping.record_created_block(entry);
        copy_block(self, &mut new_cfg, blocks[0], entry);

        for &old in &blocks[1..] {
            let new_id = new_cfg.new_block();
            id_map[old.index()] = Some(new_id);
            mapping.record_block(old, [new_id]);
            mapping.record_created_block(new_id);
            copy_block(self, &mut new_cfg, old, new_id);
        }

        for edge in self.edges() {
            let new_source = id_map.get(edge.source().index()).copied().flatten();
            let new_target = id_map.get(edge.target().index()).copied().flatten();
            if let (Some(source), Some(target)) = (new_source, new_target) {
                let new_edge = new_cfg.add_edge_with_payload(
                    source,
                    target,
                    edge.kind(),
                    edge.payload().clone(),
                );
                if let Some(weight) = edge.weight() {
                    new_cfg.edge_mut(new_edge).set_weight(Some(weight));
                }
                mapping.record_edge(edge.id(), [new_edge]);
                mapping.record_created_edge(new_edge);
            } else {
                mapping.record_edge(edge.id(), []);
            }
        }

        for block in self.block_ids() {
            if id_map[block.index()].is_none() {
                mapping.record_block(block, []);
            }
        }

        (new_cfg, mapping)
    }
}

fn copy_block<I: Clone, E: Clone>(
    source: &Cfg<I, E>,
    target: &mut Cfg<I, E>,
    from: BlockId,
    to: BlockId,
) {
    let original = source.block(from);
    target
        .block_mut(to)
        .instructions_mut()
        .extend(original.instructions().iter().cloned());
    if let Some(label) = original.label() {
        target.block_mut(to).set_label(label);
    }
}
