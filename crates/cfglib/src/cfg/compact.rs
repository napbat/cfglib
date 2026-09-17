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
    /// Whether every block and edge slot holds a live entity.
    ///
    /// True exactly when [`compact`](Self::compact) would change nothing.
    #[must_use]
    pub fn is_compact(&self) -> bool {
        self.graph.is_compact()
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
