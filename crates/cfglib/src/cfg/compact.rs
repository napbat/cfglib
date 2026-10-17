//! Compaction of a [`Cfg`]: give the surviving blocks and edges dense
//! numbers again and report what moved.

extern crate alloc;

use alloc::collections::BTreeSet;

use crate::block::{BlockId, BlockTag};
use crate::edge::EdgeTag;
use crate::graph::store::Renumbering;
use crate::region::{Handler, HandlerBody, HandlerKind, Region};

use super::Cfg;

/// The renumbering one [`Cfg::compact`] produced.
pub type CfgRenumbering = Renumbering<BlockTag, EdgeTag>;

impl<I, E> Cfg<I, E> {
    /// Whether [`compact`](Self::compact) would renumber nothing: no block or
    /// edge slot has been removed, so every slot holds a live entity.
    ///
    /// A CFG that was only ever added to is compact, however it was built.
    /// Unlike [`Graph::is_compact`](crate::Graph::is_compact), which asks
    /// whether the store is a pure compressed base, this asks what a consumer
    /// holding identities needs to know: whether they stay valid.
    #[must_use]
    pub fn is_compact(&self) -> bool {
        self.block_count() == self.block_bound() && self.edge_count() == self.edge_bound()
    }

    /// Renumber the surviving blocks and edges densely and report the change.
    ///
    /// Survivors keep their relative order, so identity order still means
    /// insertion order afterwards. The entry block, the exception regions,
    /// and the cleanup records are renumbered with the graph; a set of blocks
    /// (a region's protected blocks, a handler's known body) simply loses the
    /// members that did not survive.
    ///
    /// Compose the result with a pass's [`Rewrite`](crate::Rewrite) through
    /// [`then_renumbered`](crate::Rewrite::then_renumbered) to keep one
    /// mapping from the identities a consumer held to the ones it now holds.
    ///
    /// # Panics
    ///
    /// Panics when a handler entry, a filter block, or a cleanup resume point
    /// names a block that did not survive: those are single references with
    /// no meaningful replacement, so the exception metadata has to be rebuilt
    /// before such a block is removed.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let dropped = cfg.new_block();
    /// let kept = cfg.new_block();
    /// cfg.add_edge(cfg.entry(), kept, EdgeKind::Fallthrough);
    /// cfg.remove_block(dropped);
    ///
    /// let renumbering = cfg.compact();
    /// assert!(cfg.is_compact());
    /// assert_eq!(renumbering.node(dropped), None);
    /// assert_eq!(renumbering.node(kept).unwrap().index(), 1);
    /// ```
    pub fn compact(&mut self) -> CfgRenumbering {
        let renumbering = self.graph.compact();
        self.entry = require(&renumbering, self.entry, "entry block");

        for region in &mut self.regions {
            renumber_region(region, &renumbering);
        }
        for cleanup in &mut self.cleanups {
            cleanup.resume_from = cleanup
                .resume_from
                .map(|block| require(&renumbering, block, "cleanup resume block"));
            for continuation in &mut cleanup.continuations {
                continuation.resume =
                    require(&renumbering, continuation.resume, "cleanup continuation");
            }
        }
        renumbering
    }
}

fn renumber_region(region: &mut Region, renumbering: &CfgRenumbering) {
    region.protected_blocks = retain_surviving(&region.protected_blocks, renumbering);
    for handler in &mut region.handlers {
        renumber_handler(handler, renumbering);
    }
}

fn renumber_handler(handler: &mut Handler, renumbering: &CfgRenumbering) {
    handler.entry = require(renumbering, handler.entry, "handler entry block");
    if let HandlerBody::Known(blocks) = &handler.body {
        handler.body = HandlerBody::Known(retain_surviving(blocks, renumbering));
    }
    if let HandlerKind::Filter { filter_block } = &mut handler.kind {
        *filter_block = require(renumbering, *filter_block, "handler filter block");
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

    use super::*;
    use crate::edge::EdgeKind;
    use crate::region::{
        CompletionReason, Continuation, HandlerBody, HandlerKind, HandlerRef, RegionId,
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
    /// resume block, and its continuation target. Returns the graph and the
    /// payloads in that order.
    fn sparse_metadata() -> (Cfg<u32>, [u32; 6]) {
        let mut cfg = Cfg::<u32>::new();
        let mut gaps = vec![];
        let mut named = vec![];
        for _ in 0..6 {
            gaps.push(tagged(&mut cfg));
            named.push(tagged(&mut cfg));
        }
        let [protected, extent, entry, filter, resume, target] =
            [0, 1, 2, 3, 4, 5].map(|i| named[i]);
        cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
        let region = cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: [protected, gaps[0]].into_iter().collect(),
            handlers: vec![Handler {
                entry,
                body: HandlerBody::known([entry, extent, gaps[1]]),
                kind: HandlerKind::Filter {
                    filter_block: filter,
                },
            }],
            parent: None,
        });
        let handler = HandlerRef::new(region, 0);
        cfg.set_cleanup_resume(handler, resume);
        cfg.add_continuation(
            handler,
            Continuation {
                reason: CompletionReason::Normal,
                resume: target,
            },
        );
        let payloads =
            [protected, extent, entry, filter, resume, target].map(|block| payload(&cfg, block));
        for gap in gaps {
            cfg.remove_block(gap);
        }
        (cfg, payloads)
    }

    fn assert_denotes(cfg: &Cfg<u32>, payloads: [u32; 6]) {
        let [protected, extent, entry, filter, resume, target] = payloads;
        let region = &cfg.regions()[0];
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
        let cleanup = &cfg.cleanups()[0];
        assert_eq!(
            payload(cfg, cleanup.resume_from.expect("resume stays set")),
            resume
        );
        assert_eq!(payload(cfg, cleanup.continuations[0].resume), target);
    }

    #[test]
    fn compaction_renumbers_every_block_the_exception_metadata_names() {
        let (mut cfg, payloads) = sparse_metadata();
        assert!(!cfg.is_compact());
        cfg.compact();
        assert!(cfg.is_compact());
        assert_denotes(&cfg, payloads);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn a_compacted_graph_with_metadata_survives_a_serde_round_trip() {
        let (mut cfg, payloads) = sparse_metadata();
        cfg.compact();
        let text = serde_json::to_string(&cfg).expect("a graph serializes");
        let read: Cfg<u32> = serde_json::from_str(&text).expect("a compacted graph decodes");
        assert_eq!(read, cfg);
        assert_denotes(&read, payloads);
    }
}
