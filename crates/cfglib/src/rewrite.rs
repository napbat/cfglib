//! What one control-flow-graph transform did to the identities it touched.
//!
//! Two things in this crate answer "what became what", and they are not the
//! same thing. [`Renumbering`] is the **dense** answer a compaction gives:
//! total over the old slot space, every identity affected, an array lookup.
//! [`Rewrite`] is the **sparse** answer a transform gives: only the
//! identities it touched appear, and a missing entry means the transform left
//! that identity alone.
//!
//! Both compose the same way, `first.compose(&next)`, and a renumbering
//! converts into a rewrite, so a consumer that runs a pass and then compacts
//! carries the pass's record across the compaction with
//! `rewrite.compose(&Rewrite::from(&renumbering))` and has one mapping from
//! the identities it held to the identities it now holds.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::block::{BlockId, BlockTag};
use crate::edge::EdgeId;
use crate::graph::store::Renumbering;

/// Old-to-new block and edge identities produced by one graph rewrite.
///
/// Only affected old identities appear. A missing entry therefore means
/// "unchanged"; an entry with no replacements means "removed"; one
/// replacement means retained or redirected; several replacements mean an
/// identity expanded, as when one block or edge is split. Every identity
/// allocated by the rewrite is listed separately, including a new identity
/// that is also one of an old identity's replacements.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Rewrite {
    blocks: BTreeMap<BlockId, Vec<BlockId>>,
    edges: BTreeMap<EdgeId, Vec<EdgeId>>,
    created_blocks: Vec<BlockId>,
    created_edges: Vec<EdgeId>,
}

impl Rewrite {
    /// Create an empty rewrite record.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            blocks: BTreeMap::new(),
            edges: BTreeMap::new(),
            created_blocks: Vec::new(),
            created_edges: Vec::new(),
        }
    }

    /// Whether the rewrite records no affected or created identities.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
            && self.edges.is_empty()
            && self.created_blocks.is_empty()
            && self.created_edges.is_empty()
    }

    /// Replacements of an affected old block, or `None` when it was
    /// unchanged.
    #[must_use]
    pub fn blocks(&self, old: BlockId) -> Option<&[BlockId]> {
        self.blocks.get(&old).map(Vec::as_slice)
    }

    /// Replacements of an affected old edge, or `None` when it was unchanged.
    #[must_use]
    pub fn edges(&self, old: EdgeId) -> Option<&[EdgeId]> {
        self.edges.get(&old).map(Vec::as_slice)
    }

    /// Blocks allocated without an old identity.
    #[must_use]
    pub fn created_blocks(&self) -> &[BlockId] {
        &self.created_blocks
    }

    /// Edges allocated without an old identity.
    #[must_use]
    pub fn created_edges(&self) -> &[EdgeId] {
        &self.created_edges
    }

    /// Iterate over all explicitly affected old blocks in identity order.
    pub fn block_mappings(&self) -> impl Iterator<Item = (BlockId, &[BlockId])> {
        self.blocks
            .iter()
            .map(|(&old, replacements)| (old, replacements.as_slice()))
    }

    /// Iterate over all explicitly affected old edges in identity order.
    pub fn edge_mappings(&self) -> impl Iterator<Item = (EdgeId, &[EdgeId])> {
        self.edges
            .iter()
            .map(|(&old, replacements)| (old, replacements.as_slice()))
    }

    /// Record the complete replacement set of one old block.
    pub fn record_block(&mut self, old: BlockId, replacements: impl IntoIterator<Item = BlockId>) {
        self.blocks.insert(old, unique(replacements));
    }

    /// Record the complete replacement set of one old edge.
    pub fn record_edge(&mut self, old: EdgeId, replacements: impl IntoIterator<Item = EdgeId>) {
        self.edges.insert(old, unique(replacements));
    }

    /// Record a newly allocated block with no old identity.
    pub fn record_created_block(&mut self, block: BlockId) {
        push_unique(&mut self.created_blocks, block);
    }

    /// Record a newly allocated edge with no old identity.
    pub fn record_created_edge(&mut self, edge: EdgeId) {
        push_unique(&mut self.created_edges, edge);
    }

    /// Compose `next`, which ran after this rewrite, into this record.
    ///
    /// Replacements already recorded here are chased through `next`; mappings
    /// introduced only by `next` are then added. This preserves a direct view
    /// from identities before the first rewrite to identities after the last.
    /// `next` may be a compaction converted with [`From`], which makes the
    /// composed record total over the identities that existed before.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{BlockId, Cfg, EdgeKind, NodeOrder, Rewrite, remove_unreachable_mapped};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let dead = cfg.new_block();
    /// let live = cfg.new_block();
    /// cfg.add_edge(cfg.entry(), live, EdgeKind::Fallthrough);
    /// cfg.block_mut(dead).push(7);
    ///
    /// let (_, mut rewrite) = remove_unreachable_mapped(&mut cfg);
    /// let renumbering = cfg.compact(NodeOrder::Insertion);
    /// rewrite.compose(&Rewrite::from(&renumbering));
    /// assert_eq!(rewrite.blocks(dead), Some([].as_slice()));
    /// // The compaction moved `live` down into the dropped slot.
    /// assert_eq!(rewrite.blocks(live), Some([BlockId::from_index(1)].as_slice()));
    /// ```
    pub fn compose(&mut self, next: &Self) {
        compose_axis(&mut self.blocks, &next.blocks);
        compose_axis(&mut self.edges, &next.edges);

        for (&old, replacements) in &next.blocks {
            self.blocks
                .entry(old)
                .or_insert_with(|| replacements.clone());
        }
        for (&old, replacements) in &next.edges {
            self.edges
                .entry(old)
                .or_insert_with(|| replacements.clone());
        }

        self.created_blocks = remap_created(self.created_blocks.drain(..), &self.blocks);
        self.created_edges = remap_created(self.created_edges.drain(..), &self.edges);
        for &block in &next.created_blocks {
            push_unique(&mut self.created_blocks, block);
        }
        for &edge in &next.created_edges {
            push_unique(&mut self.created_edges, edge);
        }
    }
}

/// The record of a compaction: every old block and edge slot maps to the
/// identity the compaction gave it, or to nothing when it was dropped, which
/// is the record's spelling for "removed".
impl From<&Renumbering<BlockTag>> for Rewrite {
    fn from(renumbering: &Renumbering<BlockTag>) -> Self {
        Self {
            blocks: renumbering
                .node_mappings()
                .map(|(old, new)| (old, new.into_iter().collect()))
                .collect(),
            edges: renumbering
                .edge_mappings()
                .map(|(old, new)| (old, new.into_iter().collect()))
                .collect(),
            created_blocks: Vec::new(),
            created_edges: Vec::new(),
        }
    }
}

/// Where a transform records the identities it touched: a [`Rewrite`], or
/// nothing at all.
///
/// A pass whose plain form must not pay for the record (block merging, edge
/// contraction, and empty-block bypass move every edge of a block, so a
/// record costs as much as the move) is written once, generic over the
/// recorder; `()` records nothing and compiles away.
pub(crate) trait Recorder {
    /// Whether this recorder keeps anything, so a pass can skip the walks
    /// that only feed the record.
    const RECORDS: bool;

    /// Record the complete replacement set of one old block.
    fn record_block(&mut self, old: BlockId, replacements: impl IntoIterator<Item = BlockId>);

    /// Record the complete replacement set of one old edge.
    fn record_edge(&mut self, old: EdgeId, replacements: impl IntoIterator<Item = EdgeId>);
}

impl Recorder for Rewrite {
    const RECORDS: bool = true;

    fn record_block(&mut self, old: BlockId, replacements: impl IntoIterator<Item = BlockId>) {
        Self::record_block(self, old, replacements);
    }

    fn record_edge(&mut self, old: EdgeId, replacements: impl IntoIterator<Item = EdgeId>) {
        Self::record_edge(self, old, replacements);
    }
}

impl Recorder for () {
    const RECORDS: bool = false;

    fn record_block(&mut self, _old: BlockId, _replacements: impl IntoIterator<Item = BlockId>) {}

    fn record_edge(&mut self, _old: EdgeId, _replacements: impl IntoIterator<Item = EdgeId>) {}
}

/// Append `value` unless the run already holds it. Runs are a handful of
/// replacements, so a scan beats a set.
fn push_unique<T: Copy + PartialEq>(run: &mut Vec<T>, value: T) {
    if !run.contains(&value) {
        run.push(value);
    }
}

fn unique<T: Copy + PartialEq>(values: impl IntoIterator<Item = T>) -> Vec<T> {
    let mut result = Vec::new();
    for value in values {
        push_unique(&mut result, value);
    }
    result
}

/// What `identity` became under `map`: its recorded replacements, or itself
/// when `map` does not mention it.
fn replacements_of<'a, T: Ord>(map: &'a BTreeMap<T, Vec<T>>, identity: &'a T) -> &'a [T] {
    map.get(identity)
        .map_or_else(|| core::slice::from_ref(identity), Vec::as_slice)
}

fn compose_axis<T: Copy + Ord>(current: &mut BTreeMap<T, Vec<T>>, next: &BTreeMap<T, Vec<T>>) {
    for replacements in current.values_mut() {
        let mut composed = Vec::new();
        for replacement in replacements.drain(..) {
            for &value in replacements_of(next, &replacement) {
                push_unique(&mut composed, value);
            }
        }
        *replacements = composed;
    }
}

fn remap_created<T: Copy + Ord>(
    created: impl Iterator<Item = T>,
    mappings: &BTreeMap<T, Vec<T>>,
) -> Vec<T> {
    let mut remapped = Vec::new();
    for identity in created {
        for &replacement in replacements_of(mappings, &identity) {
            push_unique(&mut remapped, replacement);
        }
    }
    remapped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composition_chases_replacements_and_removals() {
        let a = BlockId::from_index(0);
        let b = BlockId::from_index(1);
        let c = BlockId::from_index(2);
        let d = BlockId::from_index(3);
        let mut first = Rewrite::new();
        first.record_block(a, [b, c]);
        first.record_created_block(c);

        let mut second = Rewrite::new();
        second.record_block(b, [d]);
        second.record_block(c, []);
        first.compose(&second);

        assert_eq!(first.blocks(a), Some([d].as_slice()));
        assert_eq!(first.blocks(c), Some([].as_slice()));
        assert_eq!(first.created_blocks(), []);
    }

    #[test]
    fn composing_a_compaction_maps_every_old_identity_to_the_one_it_holds_now() {
        use crate::cfg::Cfg;
        use crate::edge::{Edge, EdgeKind};
        use crate::graph::store::NodeOrder;
        use crate::test_util::probability;

        let mut cfg = Cfg::<u32>::new();
        let entry = cfg.entry();
        let dead = cfg.new_block();
        let middle = cfg.new_block();
        let exit = cfg.new_block();
        let doomed = cfg.add_edge(dead, exit, EdgeKind::Jump);
        let into_middle = cfg.add_edge(entry, middle, EdgeKind::Fallthrough);
        let half = probability(0.5);
        let out_of_middle = cfg.add_edge(middle, exit, Edge::new(EdgeKind::Jump).with_weight(half));

        // A pass that only removes `dead` and its edge.
        let mut rewrite = Rewrite::new();
        rewrite.record_block(dead, []);
        rewrite.record_edge(doomed, []);
        cfg.remove_block(dead);
        let renumbering = cfg.compact(NodeOrder::Insertion);
        rewrite.compose(&Rewrite::from(&renumbering));

        assert_eq!(rewrite.blocks(dead), Some([].as_slice()));
        assert_eq!(rewrite.edges(doomed), Some([].as_slice()));
        for old in [entry, middle, exit] {
            let new = renumbering.node(old).expect("a live block survives");
            assert_eq!(rewrite.blocks(old), Some([new].as_slice()), "{old}");
        }
        for old in [into_middle, out_of_middle] {
            let new = renumbering.edge(old).expect("a live edge survives");
            assert_eq!(rewrite.edges(old), Some([new].as_slice()), "{old}");
        }
        assert_eq!(
            cfg.edge(rewrite.edges(out_of_middle).unwrap()[0]).weight(),
            Some(half)
        );
    }
}
