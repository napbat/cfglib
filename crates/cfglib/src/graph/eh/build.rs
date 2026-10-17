//! Construction of [`EhModel`] from a CFG's edges and regions.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use crate::cfg::Cfg;
use crate::region::HandlerRef;

use super::{EhBlockKind, EhEdge, EhEdgeKind, EhModel};

impl EhModel {
    /// Compute an EH model by analyzing edge kinds and region metadata.
    ///
    /// Targets of handler/unwind edges are classified as landing pads. Sources
    /// of resume/continue edges are classified as resume points, unless the
    /// block is also the target of a handler/unwind edge: being entered by
    /// dispatch is the role that decides, whatever order the edges were added
    /// in. Explicit [`Region`] metadata is authoritative, so a `finally` or
    /// `fault` target remains a cleanup even when an unwind edge also reaches
    /// it, and a block several handlers name takes the role of the last of
    /// them, in region and handler order.
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
        let blocks = cfg.block_bound();
        let mut model = Self {
            block_kinds: vec![EhBlockKind::Normal; blocks],
            eh_edges: Vec::new(),
            protected_by: vec![BTreeSet::new(); blocks],
            handlers: vec![Vec::new(); blocks],
            cleanups: vec![None; blocks],
        };
        model.classify_edges(cfg);
        model.classify_regions(cfg);
        model
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
            });
            match kind {
                EhEdgeKind::Handler | EhEdgeKind::Unwind => {
                    // Entry by dispatch outranks being a resume point, so the
                    // role does not depend on which edge is met first.
                    self.block_kinds[edge.target().index()] = EhBlockKind::LandingPad;
                    self.protected_by[edge.target().index()].insert(edge.source());
                }
                EhEdgeKind::Resume | EhEdgeKind::Continue => {
                    let role = &mut self.block_kinds[edge.source().index()];
                    if *role == EhBlockKind::Normal {
                        *role = EhBlockKind::Resume;
                    }
                }
                EhEdgeKind::Leave => {}
            }
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
}
