//! Exception handling (EH) modelling.
//!
//! Provides first-class support for EH control flow — landing pads, cleanup
//! blocks, handler/unwind/leave/resume/continue edges, and stable links back to
//! caller-owned edge metadata — enabling accurate runtime-neutral analysis.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::edge::{EdgeId, EdgeKind};
use crate::region::{Cleanup, HandlerKind, HandlerRef};

/// Classification of a block's role in exception handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EhBlockKind {
    /// Normal code — not part of an EH construct.
    Normal,
    /// A landing pad — first block of an exception handler.
    LandingPad,
    /// A cleanup block — executes during stack unwinding (finally).
    Cleanup,
    /// A catch dispatch — selects among multiple handlers.
    CatchSwitch,
    /// A resume/rethrow point.
    Resume,
}

impl From<HandlerKind> for EhBlockKind {
    fn from(kind: HandlerKind) -> Self {
        match kind {
            HandlerKind::Catch | HandlerKind::CatchAll => Self::LandingPad,
            HandlerKind::Finally | HandlerKind::Fault => Self::Cleanup,
            HandlerKind::Filter { .. } => Self::CatchSwitch,
        }
    }
}

/// The exception-control meaning retained for an [`EhEdge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EhEdgeKind {
    /// Transfer into a selected exception handler.
    Handler,
    /// Stack-unwind transfer to a handler or cleanup.
    Unwind,
    /// Normal transfer out of a protected region.
    Leave,
    /// Continue searching or rethrow the active exception.
    Resume,
    /// Resume execution after handling the exception in-place.
    Continue,
}

impl EhEdgeKind {
    fn from_cfg(kind: EdgeKind) -> Option<Self> {
        match kind {
            EdgeKind::ExceptionHandler => Some(Self::Handler),
            EdgeKind::ExceptionUnwind => Some(Self::Unwind),
            EdgeKind::ExceptionLeave => Some(Self::Leave),
            EdgeKind::ExceptionResume => Some(Self::Resume),
            EdgeKind::ExceptionContinue => Some(Self::Continue),
            _ => None,
        }
    }

    /// Whether this is specifically a stack-unwind transfer.
    #[must_use]
    pub const fn is_unwind(self) -> bool {
        matches!(self, Self::Unwind)
    }
}

/// An exception handling edge annotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EhEdge {
    /// Stable identity of the source CFG edge.
    ///
    /// Use it to recover caller-owned payload metadata from the original
    /// [`Cfg`](crate::Cfg), including exception dispositions and platform
    /// records.
    pub edge_id: EdgeId,
    /// Source block (may throw).
    pub from: BlockId,
    /// Target block (handler / cleanup).
    pub to: BlockId,
    /// Precise exception-control transfer kind.
    pub kind: EhEdgeKind,
    /// Compatibility projection of [`Self::kind`].
    ///
    /// This is `true` only for [`EhEdgeKind::Unwind`].
    pub is_unwind: bool,
}

/// EH model for a CFG.
///
/// Every block-keyed table is a dense vector indexed by block, sized by the
/// source CFG's block bound: the classification is total over blocks anyway,
/// and the rest answer in one load instead of a tree descent.
#[derive(Debug, Clone)]
pub struct EhModel {
    block_kinds: Vec<EhBlockKind>,
    eh_edges: Vec<EhEdge>,
    protected_by: Vec<BTreeSet<BlockId>>,
    handlers: Vec<Vec<HandlerRef>>,
    cleanups: Vec<Option<Cleanup>>,
}

mod build;

static NO_PROTECTED_BLOCKS: BTreeSet<BlockId> = BTreeSet::new();

impl EhModel {
    /// The block bound this model was computed over.
    ///
    /// Every block-keyed query answers for an index below it.
    #[must_use]
    pub fn block_bound(&self) -> usize {
        self.block_kinds.len()
    }

    /// The exception-handling role of one block.
    ///
    /// A block outside the model — one added after it was computed — reads as
    /// [`EhBlockKind::Normal`].
    #[must_use]
    pub fn block_kind(&self, block: BlockId) -> EhBlockKind {
        self.block_kinds
            .get(block.index())
            .copied()
            .unwrap_or(EhBlockKind::Normal)
    }

    /// Every block and its exception-handling role, in block order.
    pub fn block_kinds(&self) -> impl Iterator<Item = (BlockId, EhBlockKind)> + '_ {
        self.block_kinds
            .iter()
            .enumerate()
            .map(|(index, &kind)| (BlockId::from_index(index), kind))
    }

    /// All exception-control edges, including leave, rethrow, and continue.
    #[must_use]
    pub fn eh_edges(&self) -> &[EhEdge] {
        &self.eh_edges
    }

    /// The blocks `handler_entry` protects, empty when it protects none.
    #[must_use]
    pub fn protected_by(&self, handler_entry: BlockId) -> &BTreeSet<BlockId> {
        self.protected_by
            .get(handler_entry.index())
            .unwrap_or(&NO_PROTECTED_BLOCKS)
    }

    /// The region/handler identities entered at `handler_entry`.
    ///
    /// The identity provides a lossless route back to [`HandlerKind`] and
    /// consumer-owned [`HandlerMetadata`](crate::HandlerMetadata).
    #[must_use]
    pub fn handlers(&self, handler_entry: BlockId) -> &[HandlerRef] {
        self.handlers
            .get(handler_entry.index())
            .map_or(&[], Vec::as_slice)
    }

    /// What the cleanup entered at `handler_entry` does once its body ends,
    /// for the handlers whose frontend recorded it
    /// ([`Cfg::add_continuation`](crate::Cfg::add_continuation)).
    ///
    /// A `finally` lowered as a single shared block is entered by every route
    /// out of its region and edges to all of their destinations, so the graph
    /// alone cannot say which edge belongs to which route. The record does:
    /// [`Cleanup::resumes_for`] answers "where does control go when this
    /// cleanup was entered by a `return`", and [`Cleanup::resume_from`] names
    /// the block those edges leave (`None` when the cleanup diverges, in
    /// which case its recorded routes are unreachable).
    #[must_use]
    pub fn cleanup(&self, handler_entry: BlockId) -> Option<&Cleanup> {
        self.cleanups.get(handler_entry.index())?.as_ref()
    }

    /// Every recorded cleanup with the handler entry block it belongs to.
    pub fn cleanups(&self) -> impl Iterator<Item = (BlockId, &Cleanup)> + '_ {
        self.cleanups
            .iter()
            .enumerate()
            .filter_map(|(index, cleanup)| Some((BlockId::from_index(index), cleanup.as_ref()?)))
    }

    /// All blocks classified as `kind`.
    fn blocks_of_kind(&self, kind: EhBlockKind) -> Vec<BlockId> {
        self.block_kinds()
            .filter(|&(_, candidate)| candidate == kind)
            .map(|(block, _)| block)
            .collect()
    }

    /// Returns all landing pad blocks.
    #[must_use]
    pub fn landing_pads(&self) -> Vec<BlockId> {
        self.blocks_of_kind(EhBlockKind::LandingPad)
    }

    /// Returns all cleanup blocks.
    #[must_use]
    pub fn cleanup_blocks(&self) -> Vec<BlockId> {
        self.blocks_of_kind(EhBlockKind::Cleanup)
    }

    /// Returns blocks that resume, rethrow, or continue an exception.
    #[must_use]
    pub fn resume_blocks(&self) -> Vec<BlockId> {
        self.blocks_of_kind(EhBlockKind::Resume)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::ff;

    #[test]
    fn handler_kinds_map_to_precise_block_roles() {
        assert_eq!(
            EhBlockKind::from(HandlerKind::Catch),
            EhBlockKind::LandingPad
        );
        assert_eq!(
            EhBlockKind::from(HandlerKind::CatchAll),
            EhBlockKind::LandingPad
        );
        assert_eq!(
            EhBlockKind::from(HandlerKind::Finally),
            EhBlockKind::Cleanup
        );
        assert_eq!(EhBlockKind::from(HandlerKind::Fault), EhBlockKind::Cleanup);
        assert_eq!(
            EhBlockKind::from(HandlerKind::Filter {
                filter_block: BlockId::from_raw(7),
            }),
            EhBlockKind::CatchSwitch
        );
    }

    #[test]
    fn no_eh_all_normal() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("a"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let model = EhModel::compute(&cfg);
        assert!(model.eh_edges().is_empty());
        assert!(
            model
                .block_kinds()
                .all(|(_, kind)| kind == EhBlockKind::Normal)
        );
    }

    #[test]
    fn exception_edge_creates_landing_pad() {
        let mut cfg = Cfg::new();
        let handler = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(ff("call"));
        cfg.block_mut(handler).instructions_mut().push(ff("catch"));
        cfg.add_edge(cfg.entry(), handler, EdgeKind::ExceptionHandler);
        let model = EhModel::compute(&cfg);
        assert_eq!(model.eh_edges().len(), 1);
        assert_eq!(model.block_kind(handler), EhBlockKind::LandingPad);
        assert!(model.protected_by(handler).contains(&cfg.entry()));
    }

    #[test]
    fn unknown_handler_body_still_has_region_and_edge_identity() {
        use crate::region::{Handler, HandlerBody, HandlerKind, HandlerRef, Region, RegionId};

        let mut cfg = Cfg::<()>::new();
        let handler = cfg.new_block();
        let entry = cfg.entry();
        let edge = cfg.add_edge(entry, handler, EdgeKind::ExceptionHandler);
        let region = cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: [entry].into_iter().collect(),
            handlers: alloc::vec![Handler {
                entry: handler,
                body: HandlerBody::unknown(),
                kind: HandlerKind::Catch,
            }],
            parent: None,
        });

        let model = EhModel::compute(&cfg);
        assert_eq!(model.eh_edges()[0].edge_id, edge);
        assert_eq!(model.block_kind(handler), EhBlockKind::LandingPad);
        assert_eq!(
            model.handlers(handler),
            alloc::vec![HandlerRef::new(region, 0)]
        );
        assert!(model.protected_by(handler).contains(&entry));
    }

    #[test]
    fn cleanup_continuations_reach_the_model_by_entry_block() {
        use crate::region::{
            CompletionReason, Continuation, Handler, HandlerBody, HandlerKind, Region, RegionId,
        };

        let mut cfg = Cfg::new();
        let cleanup = cfg.new_block();
        let after = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(ff("try"));
        cfg.block_mut(cleanup)
            .instructions_mut()
            .push(ff("finally"));
        let region = cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: [cfg.entry()].into_iter().collect(),
            handlers: alloc::vec![Handler {
                entry: cleanup,
                body: HandlerBody::known([cleanup]),
                kind: HandlerKind::Finally,
            }],
            parent: None,
        });

        // Without records the model is exactly what it always was.
        assert!(EhModel::compute(&cfg).cleanups().next().is_none());

        let handler = HandlerRef::new(region, 0);
        cfg.set_cleanup_resume(handler, cleanup);
        cfg.add_continuation(
            handler,
            Continuation {
                reason: CompletionReason::Normal,
                resume: after,
            },
        );
        cfg.add_continuation(
            handler,
            Continuation {
                reason: CompletionReason::Return,
                resume: exit,
            },
        );
        // Both routes leave the same block, so the edges alone are opaque.
        cfg.add_edge(cleanup, after, EdgeKind::Fallthrough);
        cfg.add_edge(cleanup, exit, EdgeKind::Fallthrough);

        let model = EhModel::compute(&cfg);
        assert_eq!(model.block_kind(cleanup), EhBlockKind::Cleanup);
        let recorded = model.cleanup(cleanup).expect("the cleanup was recorded");
        assert_eq!(recorded.handler, handler);
        assert_eq!(recorded.resume_from, Some(cleanup));
        assert_eq!(
            recorded
                .resumes_for(CompletionReason::Return)
                .collect::<alloc::vec::Vec<_>>(),
            alloc::vec![exit],
            "the reason selects the route the shared edges cannot"
        );
        assert_eq!(
            recorded
                .resumes_for(CompletionReason::Normal)
                .collect::<alloc::vec::Vec<_>>(),
            alloc::vec![after]
        );
        assert!(
            recorded
                .resumes_for(CompletionReason::Transfer)
                .next()
                .is_none()
        );
    }

    #[test]
    fn landing_pads_query() {
        let mut cfg = Cfg::new();
        let lp = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(ff("try"));
        cfg.block_mut(lp).instructions_mut().push(ff("handler"));
        cfg.add_edge(cfg.entry(), lp, EdgeKind::ExceptionHandler);
        let model = EhModel::compute(&cfg);
        let pads = model.landing_pads();
        assert_eq!(pads.len(), 1);
        assert_eq!(pads[0], lp);
    }

    #[test]
    fn payload_cfg_retains_every_exception_transfer_and_edge_identity() {
        use crate::exception::{ExceptionDisposition, ExceptionFlow, ExceptionPhase};

        let mut cfg = Cfg::<(), ExceptionFlow<u32>>::with_edge_payload();
        let handler = cfg.new_block();
        let leave = cfg.new_block();
        let rethrow = cfg.new_block();
        let outer = cfg.new_block();
        let continue_decision = cfg.new_block();
        let resume_target = cfg.new_block();
        let entry = cfg.entry();

        let handler_edge = cfg.add_edge_with_payload(
            entry,
            handler,
            EdgeKind::ExceptionHandler,
            ExceptionFlow::exceptional(
                ExceptionPhase::Unwind,
                Some(ExceptionDisposition::ExecuteHandler),
                11,
            ),
        );
        cfg.add_edge_with_payload(
            entry,
            leave,
            EdgeKind::ExceptionLeave,
            ExceptionFlow::normal(12),
        );
        cfg.add_edge_with_payload(
            rethrow,
            outer,
            EdgeKind::ExceptionResume,
            ExceptionFlow::exceptional(
                ExceptionPhase::Search,
                Some(ExceptionDisposition::ContinueSearch),
                13,
            ),
        );
        cfg.add_edge_with_payload(
            continue_decision,
            resume_target,
            EdgeKind::ExceptionContinue,
            ExceptionFlow::exceptional(
                ExceptionPhase::Search,
                Some(ExceptionDisposition::ContinueExecution),
                14,
            ),
        );

        let model = EhModel::compute(&cfg);
        assert_eq!(
            model
                .eh_edges()
                .iter()
                .map(|edge| edge.kind)
                .collect::<alloc::vec::Vec<_>>(),
            alloc::vec![
                EhEdgeKind::Handler,
                EhEdgeKind::Leave,
                EhEdgeKind::Resume,
                EhEdgeKind::Continue,
            ]
        );
        assert_eq!(model.eh_edges()[0].edge_id, handler_edge);
        assert_eq!(cfg[model.eh_edges()[0].edge_id].payload().metadata(), &11);
        assert_eq!(
            model.resume_blocks(),
            alloc::vec![rethrow, continue_decision]
        );
    }

    #[test]
    fn explicit_cleanup_region_overrides_incoming_unwind_inference() {
        use crate::region::{Handler, HandlerBody, HandlerKind, Region, RegionId};

        let mut cfg = Cfg::<()>::new();
        let cleanup = cfg.new_block();
        let entry = cfg.entry();
        cfg.add_edge(entry, cleanup, EdgeKind::ExceptionUnwind);
        let region = cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: [entry].into_iter().collect(),
            handlers: alloc::vec![Handler {
                entry: cleanup,
                body: HandlerBody::known([cleanup]),
                kind: HandlerKind::Finally,
            }],
            parent: None,
        });

        let model = EhModel::compute(&cfg);
        assert_eq!(model.block_kind(cleanup), EhBlockKind::Cleanup);
        assert_eq!(
            model.handlers(cleanup),
            alloc::vec![HandlerRef::new(region, 0)]
        );
    }
}
