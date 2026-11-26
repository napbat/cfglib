//! Compaction of a [`Cfg`]: give the surviving blocks and edges dense
//! numbers again and report what moved.

extern crate alloc;

use alloc::collections::BTreeSet;

use crate::block::{BlockId, BlockTag};
use crate::edge::EdgeTag;
use crate::graph::store::{NodeOrder, Renumbering};

use super::Cfg;
use super::anchors::BlockRefMut;

/// The renumbering one [`Cfg::compact`] produced.
pub type CfgRenumbering = Renumbering<BlockTag, EdgeTag>;

impl<I, E> Cfg<I, E> {
    /// Whether [`compact`](Self::compact) in [`NodeOrder::Insertion`] would
    /// renumber nothing: no block or edge slot has been removed, so every
    /// slot holds a live entity.
    ///
    /// A CFG that was only ever added to is compact, however it was built.
    /// Unlike [`Graph::is_compact`](crate::Graph::is_compact), which asks
    /// whether the store is a pure compressed base, this asks what a consumer
    /// holding identities needs to know: whether they stay valid.
    #[must_use]
    pub fn is_compact(&self) -> bool {
        self.block_count() == self.block_bound() && self.edge_count() == self.edge_bound()
    }

    /// Renumber the surviving blocks in `order` and the surviving edges
    /// densely, and report the change.
    ///
    /// This carries [`Graph::compact`](crate::Graph::compact) over to the
    /// CFG: block identity order follows `order` (under
    /// [`NodeOrder::Insertion`] it still means insertion order), while edge
    /// identity order still means insertion order and every block keeps the
    /// order of its successor and predecessor edges. Reverse postorder from
    /// the entry makes identity order a forward solver's iteration order.
    /// The entry block, the exception regions, and the cleanup records are
    /// renumbered with the graph; a set of blocks (a region's protected
    /// blocks, a handler's known body) simply loses the members that did not
    /// survive.
    ///
    /// Compose the result with a pass's [`Rewrite`](crate::Rewrite) through
    /// [`then_renumbered`](crate::Rewrite::then_renumbered) to keep one
    /// mapping from the identities a consumer held to the ones it now holds.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of
    /// [`Graph::compact`](crate::Graph::compact), before anything changes.
    /// Panics when a handler entry, a filter block, a cleanup exit block, or
    /// a continuation target names a block that did not survive: those are
    /// single references with no meaningful replacement, so the exception
    /// metadata has to be rebuilt before such a block is removed
    /// ([`check_regions`](Self::check_regions) reports them as
    /// [`RegionError::BlockNotLive`](crate::RegionError::BlockNotLive)).
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind, NodeOrder};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let dropped = cfg.new_block();
    /// let kept = cfg.new_block();
    /// cfg.add_edge(cfg.entry(), kept, EdgeKind::Fallthrough);
    /// cfg.remove_block(dropped);
    ///
    /// let renumbering = cfg.compact(NodeOrder::Insertion);
    /// assert!(cfg.is_compact());
    /// assert_eq!(renumbering.node(dropped), None);
    /// assert_eq!(renumbering.node(kept).unwrap().index(), 1);
    /// ```
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind, NodeOrder};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let exit = cfg.new_block();
    /// let body = cfg.new_block();
    /// cfg.add_edge(cfg.entry(), body, EdgeKind::Fallthrough);
    /// cfg.add_edge(body, exit, EdgeKind::Fallthrough);
    ///
    /// let renumbering = cfg.compact(NodeOrder::ReversePostorder(&[cfg.entry()]));
    /// assert_eq!(cfg.entry().index(), 0);
    /// assert_eq!(renumbering.node(body).unwrap().index(), 1);
    /// assert_eq!(renumbering.node(exit).unwrap().index(), 2);
    /// ```
    pub fn compact(&mut self, order: NodeOrder<'_, BlockTag>) -> CfgRenumbering {
        let renumbering = self.graph.compact(order);
        self.renumber_metadata(&renumbering);
        renumbering
    }

    /// Carry the entry block and the exception metadata across `renumbering`.
    fn renumber_metadata(&mut self, renumbering: &CfgRenumbering) {
        self.entry = require(renumbering, self.entry, "entry block");

        self.for_each_block_ref_mut(|reference| match reference {
            BlockRefMut::Set(blocks) => *blocks = retain_surviving(blocks, renumbering),
            BlockRefMut::Anchor(block, role) => *block = require(renumbering, *block, role.name()),
        });
    }
}

fn retain_surviving(blocks: &BTreeSet<BlockId>, renumbering: &CfgRenumbering) -> BTreeSet<BlockId> {
    blocks
        .iter()
        .filter_map(|&block| renumbering.node(block))
        .collect()
}

fn require(renumbering: &CfgRenumbering, block: BlockId, role: &str) -> BlockId {
    renumbering
        .node(block)
        .unwrap_or_else(|| panic!("the {role} {block} was removed before compaction"))
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::edge::EdgeKind;
    use crate::region::{
        CompletionReason, Continuation, Handler, HandlerBody, HandlerKind, HandlerRef, Region,
        RegionId,
    };

    /// The payload of every block names it, so a renumbered reference can be
    /// checked against the block it must still denote.
    fn tagged(cfg: &mut Cfg<u32>) -> BlockId {
        let block = cfg.new_block();
        cfg.block_mut(block)
            .push(u32::try_from(block.index()).unwrap() + 1000);
        block
    }

    fn payload(cfg: &Cfg<u32>, block: BlockId) -> u32 {
        cfg.block(block).instructions()[0]
    }

    /// A graph with a removable block between every pair of blocks that
    /// exception metadata names: a protected block and a block of the
    /// handler's extent, the handler entry, its filter block, the cleanup's
    /// exit block, and its continuation target. Returns the graph and the
    /// payloads in that order.
    fn sparse_metadata() -> (Cfg<u32>, [u32; 6]) {
        let mut cfg = Cfg::<u32>::new();
        let mut gaps = vec![];
        let mut named = vec![];
        for _ in 0..6 {
            gaps.push(tagged(&mut cfg));
            named.push(tagged(&mut cfg));
        }
        let [protected, extent, entry, filter, exit, target] = [0, 1, 2, 3, 4, 5].map(|i| named[i]);
        cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
        let region = cfg
            .add_region(Region {
                protected_blocks: [protected, gaps[0]].into_iter().collect(),
                handlers: vec![Handler {
                    entry,
                    body: HandlerBody::known([entry, extent, gaps[1]]),
                    kind: HandlerKind::Filter {
                        filter_block: filter,
                    },
                }],
                parent: None,
            })
            .unwrap();
        let handler = HandlerRef::new(region, 0);
        cfg.set_cleanup_exit(handler, exit).unwrap();
        cfg.add_continuation(
            handler,
            Continuation {
                reason: CompletionReason::Normal,
                target,
            },
        )
        .unwrap();
        let payloads =
            [protected, extent, entry, filter, exit, target].map(|block| payload(&cfg, block));
        for gap in gaps {
            cfg.remove_block(gap);
        }
        (cfg, payloads)
    }

    fn assert_denotes(cfg: &Cfg<u32>, payloads: [u32; 6]) {
        let [protected, extent, entry, filter, exit, target] = payloads;
        let region = cfg.region(RegionId::from_raw(0));
        let handler = &region.handlers[0];
        let names = |blocks: &BTreeSet<BlockId>| -> alloc::vec::Vec<u32> {
            blocks.iter().map(|&block| payload(cfg, block)).collect()
        };
        assert_eq!(names(&region.protected_blocks), [protected]);
        assert_eq!(payload(cfg, handler.entry), entry);
        assert_eq!(
            names(handler.body.blocks().expect("the body stays known")),
            [extent, entry]
        );
        let HandlerKind::Filter { filter_block } = handler.kind else {
            panic!("the handler stays a filter");
        };
        assert_eq!(payload(cfg, filter_block), filter);
        let (_, cleanup) = cfg.cleanups().iter().next().expect("the record stays");
        assert_eq!(
            payload(cfg, cleanup.exit_block.expect("the exit stays set")),
            exit
        );
        assert_eq!(payload(cfg, cleanup.continuations[0].target), target);
    }

    #[test]
    fn compaction_renumbers_every_block_the_exception_metadata_names() {
        let (mut cfg, payloads) = sparse_metadata();
        assert!(!cfg.is_compact());
        cfg.compact(NodeOrder::Insertion);
        assert!(cfg.is_compact());
        assert_denotes(&cfg, payloads);
    }

    #[test]
    fn ordered_compaction_renumbers_every_block_the_exception_metadata_names() {
        let (cfg, payloads) = sparse_metadata();
        let entry = cfg.entry();
        // A rotation moves every block, yet keeps the handler body's two
        // members in the relative order `assert_denotes` reads a set in.
        let mut rotated: Vec<BlockId> = cfg.graph.node_ids().collect();
        rotated.rotate_left(4);
        let rotated_entry = rotated.len() - 4;
        for (order, entry_index) in [
            (NodeOrder::ReversePostorder(&[entry]), 0),
            (NodeOrder::Explicit(&rotated), rotated_entry),
        ] {
            let mut cfg = cfg.clone();
            let renumbering = cfg.compact(order);
            assert!(cfg.is_compact());
            assert_eq!(Some(cfg.entry()), renumbering.node(entry));
            assert_eq!(cfg.entry().index(), entry_index, "{order:?}");
            assert!(
                cfg.block(cfg.entry()).instructions().is_empty(),
                "the entry is still the one block without a payload"
            );
            assert_denotes(&cfg, payloads);
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn a_compacted_graph_with_metadata_survives_a_serde_round_trip() {
        let (mut cfg, payloads) = sparse_metadata();
        cfg.compact(NodeOrder::Insertion);
        let text = serde_json::to_string(&cfg).expect("a graph serializes");
        let read: Cfg<u32> = serde_json::from_str(&text).expect("a compacted graph decodes");
        assert_eq!(read, cfg);
        assert_denotes(&read, payloads);
    }
}
