//! Exception handling (EH) modelling.
//!
//! Provides first-class support for EH control flow — landing pads,
//! cleanup blocks, and unwind edges — enabling accurate modelling of
//! try/catch/finally in decompilation and analysis.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::region::{Cleanup, HandlerRef};

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

/// An exception handling edge annotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EhEdge {
    /// Source block (may throw).
    pub from: BlockId,
    /// Target block (handler / cleanup).
    pub to: BlockId,
    /// Whether this is an unwind edge (vs normal flow).
    pub is_unwind: bool,
}

/// EH model for a CFG.
#[derive(Debug, Clone)]
pub struct EhModel {
    /// Classification of each block.
    pub block_kinds: BTreeMap<BlockId, EhBlockKind>,
    /// All EH (unwind) edges.
    pub eh_edges: Vec<EhEdge>,
    /// Landing pad → set of blocks it protects.
    pub protected_by: BTreeMap<BlockId, BTreeSet<BlockId>>,
    /// Cleanup handler entry block → what the cleanup does once its body
    /// ends, for the handlers whose frontend recorded it
    /// ([`Cfg::add_continuation`]).
    ///
    /// A `finally` lowered as a single shared block is entered by every route
    /// out of its region and edges to all of their destinations, so the graph
    /// alone cannot say which edge belongs to which route. The record does:
    /// [`Cleanup::resumes_for`] answers "where does control go when this
    /// cleanup was entered by a `return`", and [`Cleanup::resume_from`] names
    /// the block those edges leave (`None` when the cleanup diverges, in
    /// which case its recorded routes are unreachable).
    pub cleanups: BTreeMap<BlockId, Cleanup>,
}

/// Build an EH model by analysing edge kinds and region metadata.
///
/// Blocks reachable only via `Exception` edges are classified as
/// landing pads. Blocks that are targets of the existing `Region`
/// handlers are also incorporated.
///
/// Cleanup records the frontend attached to a handler
/// ([`Cfg::add_continuation`]) are carried into [`EhModel::cleanups`], keyed
/// by that handler's entry block, so an analysis reads cleanup-then-continue
/// structure instead of a fan of indistinguishable out-edges.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, build_eh_model};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
///
/// let model = build_eh_model(&cfg);
/// // No exception edges, so no landing pads.
/// assert!(model.eh_edges.is_empty());
/// ```
#[must_use]
pub fn build_eh_model<I>(cfg: &Cfg<I>) -> EhModel {
    let mut block_kinds = BTreeMap::new();
    let mut eh_edges = Vec::new();
    let mut protected_by: BTreeMap<BlockId, BTreeSet<BlockId>> = BTreeMap::new();
    let mut cleanups: BTreeMap<BlockId, Cleanup> = BTreeMap::new();

    // Classify from edge kinds.
    for edge in cfg.edges() {
        match edge.kind() {
            EdgeKind::ExceptionHandler | EdgeKind::ExceptionUnwind => {
                eh_edges.push(EhEdge {
                    from: edge.source(),
                    to: edge.target(),
                    is_unwind: matches!(edge.kind(), EdgeKind::ExceptionUnwind),
                });
                block_kinds
                    .entry(edge.target())
                    .or_insert(EhBlockKind::LandingPad);
                protected_by
                    .entry(edge.target())
                    .or_default()
                    .insert(edge.source());
            }
            _ => {}
        }
    }

    // Classify from region metadata.
    for region in cfg.regions() {
        for (index, handler) in region.handlers.iter().enumerate() {
            let target = handler.entry;
            if let Some(cleanup) = cfg.cleanup(HandlerRef::new(region.id, index)) {
                cleanups.insert(target, cleanup.clone());
            }
            block_kinds.entry(target).or_insert(match handler.kind {
                crate::region::HandlerKind::Catch | crate::region::HandlerKind::CatchAll => {
                    EhBlockKind::LandingPad
                }
                crate::region::HandlerKind::Finally | crate::region::HandlerKind::Fault => {
                    EhBlockKind::Cleanup
                }
                crate::region::HandlerKind::Filter { .. } => EhBlockKind::CatchSwitch,
            });
            for &bid in &region.protected_blocks {
                protected_by.entry(target).or_default().insert(bid);
            }
        }
    }

    // All remaining blocks are Normal.
    for block in cfg.blocks() {
        block_kinds.entry(block.id()).or_insert(EhBlockKind::Normal);
    }

    EhModel {
        block_kinds,
        eh_edges,
        protected_by,
        cleanups,
    }
}

/// Returns all landing pad blocks.
#[must_use]
pub fn landing_pads(model: &EhModel) -> Vec<BlockId> {
    model
        .block_kinds
        .iter()
        .filter(|&(_, k)| *k == EhBlockKind::LandingPad)
        .map(|(&bid, _)| bid)
        .collect()
}

/// Returns all cleanup blocks.
#[must_use]
pub fn cleanup_blocks(model: &EhModel) -> Vec<BlockId> {
    model
        .block_kinds
        .iter()
        .filter(|&(_, k)| *k == EhBlockKind::Cleanup)
        .map(|(&bid, _)| bid)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::ff;

    #[test]
    fn no_eh_all_normal() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("a"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let model = build_eh_model(&cfg);
        assert_eq!(model.eh_edges.len(), 0);
        assert!(
            model
                .block_kinds
                .values()
                .all(|&k| k == EhBlockKind::Normal)
        );
    }

    #[test]
    fn exception_edge_creates_landing_pad() {
        let mut cfg = Cfg::new();
        let handler = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("call"));
        cfg.block_mut(handler)
            .instructions_vec_mut()
            .push(ff("catch"));
        cfg.add_edge(cfg.entry(), handler, EdgeKind::ExceptionHandler);
        let model = build_eh_model(&cfg);
        assert_eq!(model.eh_edges.len(), 1);
        assert_eq!(model.block_kinds[&handler], EhBlockKind::LandingPad);
        assert!(model.protected_by[&handler].contains(&cfg.entry()));
    }

    #[test]
    fn cleanup_continuations_reach_the_model_by_entry_block() {
        use crate::region::{
            CompletionReason, Continuation, Handler, HandlerKind, Region, RegionId,
        };

        let mut cfg = Cfg::new();
        let cleanup = cfg.new_block();
        let after = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("try"));
        cfg.block_mut(cleanup)
            .instructions_vec_mut()
            .push(ff("finally"));
        let region = cfg.add_region(Region {
            id: RegionId::from_raw(0),
            protected_blocks: [cfg.entry()].into_iter().collect(),
            handlers: alloc::vec![Handler {
                entry: cleanup,
                body: [cleanup].into_iter().collect(),
                kind: HandlerKind::Finally,
            }],
            parent: None,
        });

        // Without records the model is exactly what it always was.
        assert!(build_eh_model(&cfg).cleanups.is_empty());

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

        let model = build_eh_model(&cfg);
        assert_eq!(model.block_kinds[&cleanup], EhBlockKind::Cleanup);
        let recorded = &model.cleanups[&cleanup];
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
            .instructions_vec_mut()
            .push(ff("try"));
        cfg.block_mut(lp).instructions_vec_mut().push(ff("handler"));
        cfg.add_edge(cfg.entry(), lp, EdgeKind::ExceptionHandler);
        let model = build_eh_model(&cfg);
        let pads = landing_pads(&model);
        assert_eq!(pads.len(), 1);
        assert_eq!(pads[0], lp);
    }
}
