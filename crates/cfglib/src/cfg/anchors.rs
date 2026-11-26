//! The blocks exception metadata names, and what a structural pass must keep
//! true about them.
//!
//! A region's protected set and a handler's known body are sets, so a block
//! that leaves the graph just drops out of them at the next
//! [`compact`](Cfg::compact). A handler entry, a filter block, a cleanup exit
//! block, and a continuation target are *single references*: nothing can
//! stand in for them, and `compact` panics when one names a removed block.
//! Those are the **anchors**, and the passes that delete or merge blocks leave
//! them alone.
//!
//! [`Cfg::block_refs`] and [`Cfg::for_each_block_ref_mut`] are the one list
//! of every block reference the metadata holds, read-only and mutable; every
//! walk over the references (anchors, folding, compaction, splitting) goes
//! through them, so a new kind of reference is added in one place.

extern crate alloc;

use alloc::collections::BTreeSet;

use crate::block::BlockId;
use crate::region::{HandlerBody, HandlerKind};

use super::Cfg;

/// What a single block reference of the exception metadata stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnchorRole {
    /// [`Handler::entry`](crate::Handler::entry).
    HandlerEntry,
    /// The block of a [`HandlerKind::Filter`].
    FilterBlock,
    /// [`Cleanup::exit_block`](crate::Cleanup::exit_block).
    CleanupExit,
    /// [`Continuation::target`](crate::Continuation::target).
    ContinuationTarget,
}

impl AnchorRole {
    /// The role's name in a message.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::HandlerEntry => "handler entry block",
            Self::FilterBlock => "handler filter block",
            Self::CleanupExit => "cleanup exit block",
            Self::ContinuationTarget => "cleanup continuation target",
        }
    }

    /// Whether the reference names the block a body *ends* in, so it moves
    /// to the tail when the block is split. Every other anchor names where
    /// control enters, which stays the head.
    pub(crate) const fn ends_a_body(self) -> bool {
        matches!(self, Self::CleanupExit)
    }
}

/// One block reference of the exception metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockRef<'a> {
    /// A single reference.
    Anchor(BlockId, AnchorRole),
    /// A protected set or a known handler body.
    Set(&'a BTreeSet<BlockId>),
}

/// One block reference of the exception metadata, open for rewriting.
#[derive(Debug)]
pub(crate) enum BlockRefMut<'a> {
    /// A single reference.
    Anchor(&'a mut BlockId, AnchorRole),
    /// A protected set or a known handler body.
    Set(&'a mut BTreeSet<BlockId>),
}

impl<I, E> Cfg<I, E> {
    /// Every block reference of the exception metadata: per region its
    /// protected set, then per handler its entry, filter block and known
    /// body; then per cleanup record its exit block and continuation targets.
    ///
    /// [`for_each_block_ref_mut`](Self::for_each_block_ref_mut) visits the
    /// same references in the same order.
    pub(crate) fn block_refs(&self) -> impl Iterator<Item = BlockRef<'_>> + '_ {
        let regions = self.regions.iter().flat_map(|region| {
            let handlers = region.handlers.iter().flat_map(|handler| {
                [
                    Some(BlockRef::Anchor(handler.entry, AnchorRole::HandlerEntry)),
                    handler
                        .kind
                        .filter_block()
                        .map(|block| BlockRef::Anchor(block, AnchorRole::FilterBlock)),
                    handler.body.blocks().map(BlockRef::Set),
                ]
                .into_iter()
                .flatten()
            });
            core::iter::once(BlockRef::Set(&region.protected_blocks)).chain(handlers)
        });
        let cleanups = self.cleanups.iter().flat_map(|(_, cleanup)| {
            let exit = cleanup
                .exit_block
                .map(|block| BlockRef::Anchor(block, AnchorRole::CleanupExit));
            exit.into_iter()
                .chain(cleanup.continuations.iter().map(|continuation| {
                    BlockRef::Anchor(continuation.target, AnchorRole::ContinuationTarget)
                }))
        });
        regions.chain(cleanups)
    }

    /// Visit every block reference of the exception metadata mutably, in the
    /// order of [`block_refs`](Self::block_refs).
    pub(crate) fn for_each_block_ref_mut(&mut self, mut visit: impl FnMut(BlockRefMut<'_>)) {
        for region in &mut self.regions {
            visit(BlockRefMut::Set(&mut region.protected_blocks));
            for handler in &mut region.handlers {
                visit(BlockRefMut::Anchor(
                    &mut handler.entry,
                    AnchorRole::HandlerEntry,
                ));
                if let HandlerKind::Filter { filter_block } = &mut handler.kind {
                    visit(BlockRefMut::Anchor(filter_block, AnchorRole::FilterBlock));
                }
                if let HandlerBody::Known(blocks) = &mut handler.body {
                    visit(BlockRefMut::Set(blocks));
                }
            }
        }
        for cleanup in self.cleanups.values_mut() {
            if let Some(exit) = &mut cleanup.exit_block {
                visit(BlockRefMut::Anchor(exit, AnchorRole::CleanupExit));
            }
            for continuation in &mut cleanup.continuations {
                visit(BlockRefMut::Anchor(
                    &mut continuation.target,
                    AnchorRole::ContinuationTarget,
                ));
            }
        }
    }

    /// Every block that exception metadata names by a single reference:
    /// handler entries, filter blocks, cleanup exit blocks, and
    /// continuation targets. A block may repeat, and a named block may
    /// already have been removed.
    ///
    /// This is the same list [`compact`](Cfg::compact) requires to survive.
    pub(crate) fn anchors(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.block_refs().filter_map(|reference| match reference {
            BlockRef::Anchor(block, _) => Some(block),
            BlockRef::Set(_) => None,
        })
    }

    /// The anchors that name no live block: what makes
    /// [`compact`](Cfg::compact) panic, and what a well-formed CFG never has.
    pub(crate) fn dangling_anchors(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.anchors()
            .filter(|&anchor| !self.contains_block(anchor))
    }

    /// Whether exception metadata names `block` by a single reference.
    pub(crate) fn is_anchor(&self, block: BlockId) -> bool {
        self.anchors().any(|anchor| anchor == block)
    }

    /// Whether folding `target` into `source` keeps every promise the
    /// exception metadata makes.
    ///
    /// `target` disappears, so no handler, filter, or cleanup may name it, and
    /// its instructions end up in `source`, so every protected set and known
    /// handler body must hold both or neither. A handler body of unknown
    /// extent claims no particular block.
    pub(crate) fn can_fold_into(&self, source: BlockId, target: BlockId) -> bool {
        self.block_refs().all(|reference| match reference {
            BlockRef::Anchor(block, _) => block != target,
            BlockRef::Set(blocks) => blocks.contains(&source) == blocks.contains(&target),
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
    use crate::graph::store::NodeOrder;
    use crate::region::{
        CompletionReason, Continuation, Handler, HandlerBody, HandlerKind, HandlerRef, Region,
    };

    use super::*;

    /// One of each anchor: an entry, a filter block, an exit block, and a
    /// continuation target, plus a plain block nothing names.
    fn every_anchor() -> (Cfg<u32>, [BlockId; 4], BlockId) {
        let mut cfg = Cfg::<u32>::new();
        let entry = cfg.new_block();
        let filter = cfg.new_block();
        let exit = cfg.new_block();
        let target = cfg.new_block();
        let plain = cfg.new_block();
        cfg.add_edge(cfg.entry(), plain, EdgeKind::Fallthrough);
        let region = cfg
            .add_region(Region {
                protected_blocks: [cfg.entry()].into_iter().collect(),
                handlers: vec![Handler {
                    entry,
                    body: HandlerBody::known([entry]),
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
        (cfg, [entry, filter, exit, target], plain)
    }

    #[test]
    fn anchors_are_exactly_the_blocks_compaction_requires() {
        let (cfg, anchors, plain) = every_anchor();

        assert_eq!(cfg.anchors().collect::<Vec<_>>(), anchors);
        for anchor in anchors {
            assert!(cfg.is_anchor(anchor));
            let mut broken = cfg.clone();
            broken.remove_block(anchor);
            let outcome = std::panic::catch_unwind(move || broken.compact(NodeOrder::Insertion));
            assert!(outcome.is_err(), "compacting survived losing {anchor}");
        }
        assert!(!cfg.is_anchor(plain));
        let mut broken = cfg;
        broken.remove_block(plain);
        broken.compact(NodeOrder::Insertion);
    }

    #[test]
    fn the_read_and_write_walks_visit_the_same_references() {
        #[derive(Debug, PartialEq, Eq)]
        enum Seen {
            Anchor(BlockId, AnchorRole),
            Set(BTreeSet<BlockId>),
        }
        let (mut cfg, _, _) = every_anchor();
        let read: Vec<Seen> = cfg
            .block_refs()
            .map(|reference| match reference {
                BlockRef::Anchor(block, role) => Seen::Anchor(block, role),
                BlockRef::Set(blocks) => Seen::Set(blocks.clone()),
            })
            .collect();
        let mut written = Vec::new();
        cfg.for_each_block_ref_mut(|reference| {
            written.push(match reference {
                BlockRefMut::Anchor(block, role) => Seen::Anchor(*block, role),
                BlockRefMut::Set(blocks) => Seen::Set(blocks.clone()),
            });
        });
        assert_eq!(read, written);
        assert_eq!(
            read.len(),
            6,
            "a protected set, an entry, a filter, a body, an exit and a target"
        );
    }
}
