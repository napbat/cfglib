//! Structural fidelity reporting for control-tree structuring.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::region::RegionId;

/// Why one [`ControlNode::Goto`](super::ControlNode::Goto) was emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GotoReason {
    /// The graph carries an explicit [`EdgeKind::Jump`](crate::EdgeKind)
    /// that no loop context could absorb as a break or continue.
    ExplicitJump,
    /// A transfer left the enclosing structural bound (a try body, a
    /// declared handler extent, or a natural-loop body).
    BoundaryEscape,
    /// A transfer re-entered a block the structured walk already emitted
    /// (shared tails, irreducible entries).
    RevisitedTarget,
}

/// One emitted goto and the structural reason it was required.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GotoDiagnostic {
    /// The block the goto targets.
    pub target: BlockId,
    /// Why structured flow could not represent the transfer.
    pub reason: GotoReason,
}

/// Exactly which parts of a structuring degraded to unstructured flow.
///
/// Consumers use this to decide between structured emission and a
/// lower-level fallback per construct — and to attach precise diagnostics —
/// instead of inferring fidelity from the shape of the returned tree.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LiftReport {
    /// Every emitted goto in emission order.
    pub gotos: Vec<GotoDiagnostic>,
    /// Blocks emitted by the completeness sweep rather than the structured
    /// walk (goto targets, handler pads reached only through exception
    /// edges, unreachable code).
    pub swept_blocks: Vec<BlockId>,
    /// Regions left as ordinary control flow because a handler extent was
    /// unknown or the region was shadowed by an enclosing structured region
    /// at the same anchor.
    pub unstructured_regions: Vec<RegionId>,
    /// Goto targets whose label could not be attached to any emitted node;
    /// the tree contains a dangling goto and the consumer should fall back.
    pub unresolved_labels: Vec<BlockId>,
}

impl LiftReport {
    /// Returns whether every transfer was represented structurally.
    #[must_use]
    pub fn is_fully_structured(&self) -> bool {
        self.gotos.is_empty()
            && self.unstructured_regions.is_empty()
            && self.unresolved_labels.is_empty()
    }
}
