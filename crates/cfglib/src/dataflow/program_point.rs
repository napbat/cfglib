//! Program points: where in a CFG a definition or a use sits.

use crate::block::BlockId;

/// A point in the program identified by a (block, instruction-index) pair.
///
/// Used to refer to both definition sites and use sites in data flow
/// analyses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramPoint {
    /// The block containing the instruction.
    pub block: BlockId,
    /// Index of the instruction within the block.
    pub inst_idx: usize,
}

impl core::fmt::Display for ProgramPoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}:{}", self.block, self.inst_idx)
    }
}

/// Alias for [`ProgramPoint`] used in definition contexts.
pub type DefSite = ProgramPoint;
/// Alias for [`ProgramPoint`] used in use-site contexts.
pub type UseSite = ProgramPoint;
