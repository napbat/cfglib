//! Construction of [`EhModel`] from a CFG's edges and regions.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use crate::cfg::Cfg;
use crate::edge::EdgeKind;

use super::{EhBlockKind, EhModel};

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
        };
        model.classify_edges(cfg);
        model.classify_regions(cfg);
        model
    }

    fn classify_edges<I, E>(&mut self, cfg: &Cfg<I, E>) {
        for edge in cfg.edges() {
            match edge.kind() {
                EdgeKind::ExceptionHandler | EdgeKind::ExceptionUnwind => {
                    // Entry by dispatch outranks being a resume point, so the
                    // role does not depend on which edge is met first.
                    self.block_kinds[edge.target().index()] = EhBlockKind::LandingPad;
                    self.protected_by[edge.target().index()].insert(edge.source());
                }
                EdgeKind::ExceptionResume | EdgeKind::ExceptionContinue => {
                    let role = &mut self.block_kinds[edge.source().index()];
                    if *role == EhBlockKind::Normal {
                        *role = EhBlockKind::Resume;
                    }
                }
                EdgeKind::ExceptionLeave => {}
                // Listed rather than matched by a wildcard, so a new edge kind
                // has to say here whether the exception model carries it.
                EdgeKind::Fallthrough
                | EdgeKind::ConditionalTrue
                | EdgeKind::ConditionalFalse
                | EdgeKind::Unconditional
                | EdgeKind::Back
                | EdgeKind::Call
                | EdgeKind::CallReturn
                | EdgeKind::SwitchCase
                | EdgeKind::Jump
                | EdgeKind::IndirectJump
                | EdgeKind::IndirectCall => continue,
            }
            self.eh_edges.push(edge.id());
        }
    }

    fn classify_regions<I, E>(&mut self, cfg: &Cfg<I, E>) {
        for (_, region) in cfg.regions() {
            for handler in &region.handlers {
                let target = handler.entry.index();
                // Region metadata is more precise than the role inferred from
                // an exception edge, so this intentionally overwrites it.
                self.block_kinds[target] = handler.kind.into();
                self.protected_by[target].extend(region.protected_blocks.iter().copied());
            }
        }
    }
}
