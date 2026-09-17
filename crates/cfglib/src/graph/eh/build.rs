//! Construction of [`EhModel`] from a CFG's edges and regions.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::region::{Cleanup, HandlerRef};

use super::{EhBlockKind, EhEdge, EhEdgeKind, EhModel};

impl EhModel {
    /// Compute an EH model by analyzing edge kinds and region metadata.
    ///
    /// Targets of handler/unwind edges are classified as landing pads. Sources
    /// of resume/continue edges are classified as resume points. Explicit
    /// [`Region`] metadata is authoritative, so a `finally` or `fault` target
    /// remains a cleanup even when an unwind edge also reaches it.
    ///
    /// [`Region`]: crate::Region
    ///
    /// Cleanup records the frontend attached to a handler
    /// ([`Cfg::add_continuation`]) are carried into [`EhModel::cleanup`],
    /// keyed by that handler's entry block, so an analysis reads
    /// cleanup-then-continue structure instead of a fan of indistinguishable
    /// out-edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind, EhModel};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let b0 = cfg.entry();
    /// let b1 = cfg.new_block();
    /// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
    ///
    /// let model = EhModel::compute(&cfg);
    /// // No exception edges, so no landing pads.
    /// assert!(model.eh_edges().is_empty());
    /// ```
    #[must_use]
    pub fn compute<I, E>(cfg: &Cfg<I, E>) -> Self {
        EhModelBuilder::from_cfg(cfg).finish()
    }
}

/// The model under construction, already sized to the source CFG so every
/// table is a dense block-indexed vector from the start.
struct EhModelBuilder {
    block_kinds: Vec<EhBlockKind>,
    eh_edges: Vec<EhEdge>,
    protected_by: Vec<BTreeSet<BlockId>>,
    handlers: Vec<Vec<HandlerRef>>,
    cleanups: Vec<Option<Cleanup>>,
}

impl EhModelBuilder {
    fn from_cfg<I, E>(cfg: &Cfg<I, E>) -> Self {
        let blocks = cfg.block_bound();
        let mut builder = Self {
            block_kinds: vec![EhBlockKind::Normal; blocks],
            eh_edges: Vec::new(),
            protected_by: vec![BTreeSet::new(); blocks],
            handlers: vec![Vec::new(); blocks],
            cleanups: vec![None; blocks],
        };
        builder.classify_edges(cfg);
        builder.classify_regions(cfg);
        builder
    }

    fn classify_edges<I, E>(&mut self, cfg: &Cfg<I, E>) {
        for edge in cfg.edges() {
            let Some(kind) = EhEdgeKind::from_cfg(edge.kind()) else {
                continue;
            };
            self.eh_edges.push(EhEdge {
                edge_id: edge.id(),
                from: edge.source(),
                to: edge.target(),
                kind,
                is_unwind: kind.is_unwind(),
            });
            match kind {
                EhEdgeKind::Handler | EhEdgeKind::Unwind => {
                    self.infer_kind(edge.target(), EhBlockKind::LandingPad);
                    self.protected_by[edge.target().index()].insert(edge.source());
                }
                EhEdgeKind::Resume | EhEdgeKind::Continue => {
                    self.infer_kind(edge.source(), EhBlockKind::Resume);
                }
                EhEdgeKind::Leave => {}
            }
        }
    }

    /// Record a role inferred from an edge, which never overrides the more
    /// precise role region metadata gives the same block.
    fn infer_kind(&mut self, block: BlockId, kind: EhBlockKind) {
        let slot = &mut self.block_kinds[block.index()];
        if *slot == EhBlockKind::Normal {
            *slot = kind;
        }
    }

    fn classify_regions<I, E>(&mut self, cfg: &Cfg<I, E>) {
        for region in cfg.regions() {
            for (index, handler) in region.handlers.iter().enumerate() {
                let target = handler.entry.index();
                let handler_ref = HandlerRef::new(region.id, index);
                self.handlers[target].push(handler_ref);
                if let Some(cleanup) = cfg.cleanup(handler_ref) {
                    self.cleanups[target] = Some(cleanup.clone());
                }
                // Region metadata is more precise than the role inferred from
                // an exception edge, so this intentionally overwrites it.
                self.block_kinds[target] = handler.kind.into();
                self.protected_by[target].extend(region.protected_blocks.iter().copied());
            }
        }
    }

    fn finish(self) -> EhModel {
        EhModel {
            block_kinds: self.block_kinds,
            eh_edges: self.eh_edges,
            protected_by: self.protected_by,
            handlers: self.handlers,
            cleanups: self.cleanups,
        }
    }
}
