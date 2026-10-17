//! The blocks exception metadata names, and what a structural pass must keep
//! true about them.
//!
//! A region's protected set, a handler's known body, and a cleanup's blocks
//! are sets, so a block that leaves the graph just drops out of them at the
//! next [`compact`](Cfg::compact). A handler entry, a filter block, a cleanup
//! resume point, and a continuation target are *single references*: nothing
//! can stand in for them, and `compact` panics when one names a removed
//! block. Those are the **anchors**, and the passes that delete or merge
//! blocks leave them alone.

use crate::block::BlockId;
use crate::region::HandlerKind;

use super::Cfg;

impl<I, E> Cfg<I, E> {
    /// Every block that exception metadata names by a single reference:
    /// handler entries, filter blocks, cleanup resume points, and
    /// continuation targets. A block may repeat, and a named block may
    /// already have been removed.
    ///
    /// This is the same list [`compact`](Cfg::compact) requires to survive.
    pub(crate) fn anchors(&self) -> impl Iterator<Item = BlockId> + '_ {
        let handlers = self
            .regions
            .iter()
            .flat_map(|region| &region.handlers)
            .flat_map(|handler| {
                let filter = match handler.kind {
                    HandlerKind::Filter { filter_block } => Some(filter_block),
                    _ => None,
                };
                core::iter::once(handler.entry).chain(filter)
            });
        let cleanups = self.cleanups.iter().flat_map(|cleanup| {
            cleanup.resume_from.into_iter().chain(
                cleanup
                    .continuations
                    .iter()
                    .map(|continuation| continuation.resume),
            )
        });
        handlers.chain(cleanups)
    }

    /// Whether exception metadata names `block` by a single reference.
    pub(crate) fn is_anchor(&self, block: BlockId) -> bool {
        self.anchors().any(|anchor| anchor == block)
    }

    /// The anchors that name no live block: what makes
    /// [`compact`](Cfg::compact) panic, and what a well-formed CFG never has.
    /// Decoding a serialized CFG and verifying one both ask this.
    pub(crate) fn dangling_anchors(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.anchors()
            .filter(|&anchor| !self.contains_block(anchor))
    }

    /// Whether folding `target` into `source` keeps every promise the
    /// exception metadata makes.
    ///
    /// `target` disappears, so no handler, filter, or cleanup may name it, and
    /// its instructions end up in `source`, so the two must be protected by
    /// the same regions and belong to the same known handler bodies. A handler
    /// body of unknown extent claims no particular block.
    pub(crate) fn can_fold_into(&self, source: BlockId, target: BlockId) -> bool {
        !self.is_anchor(target)
            && self.regions.iter().all(|region| {
                region.protected_blocks.contains(&source)
                    == region.protected_blocks.contains(&target)
                    && region.handlers.iter().all(|handler| {
                        handler.body.blocks().is_none_or(|blocks| {
                            blocks.contains(&source) == blocks.contains(&target)
                        })
                    })
            })
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    extern crate std;

    use alloc::vec;
    use alloc::vec::Vec;

    use crate::edge::EdgeKind;
    use crate::region::{
        CompletionReason, Continuation, Handler, HandlerBody, HandlerKind, HandlerRef, Region,
        RegionId,
    };

    use super::*;

    /// One of each anchor: an entry, a filter block, a resume point, and a
    /// continuation target, plus a plain block nothing names.
    fn every_anchor() -> (Cfg<u32>, [BlockId; 4], BlockId) {
        let mut cfg = Cfg::<u32>::new();
        let entry = cfg.new_block();
        let filter = cfg.new_block();
        let resume = cfg.new_block();
        let target = cfg.new_block();
        let plain = cfg.new_block();
        cfg.add_edge(cfg.entry(), plain, EdgeKind::Fallthrough);
        let region = cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: [cfg.entry()].into_iter().collect(),
            handlers: vec![Handler {
                entry,
                body: HandlerBody::Unknown,
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
        (cfg, [entry, filter, resume, target], plain)
    }

    #[test]
    fn anchors_are_exactly_the_blocks_compaction_requires() {
        let (cfg, anchors, plain) = every_anchor();

        assert_eq!(cfg.anchors().collect::<Vec<_>>().len(), anchors.len());
        for anchor in anchors {
            assert!(cfg.is_anchor(anchor));
            let mut broken = cfg.clone();
            broken.remove_block(anchor);
            let outcome = std::panic::catch_unwind(move || broken.compact());
            assert!(outcome.is_err(), "compacting survived losing {anchor}");
        }
        assert!(!cfg.is_anchor(plain));
        let mut broken = cfg;
        broken.remove_block(plain);
        broken.compact();
    }
}
