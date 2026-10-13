//! Facts that the generic algorithms compute over a [`FunctionView`].
//!
//! Every fact is a side table. The algorithms read the function through its
//! view and never change it, so a consumer decides what to do with a fact.
//! The facts build on each other in this order:
//!
//! 1. [`RootTable`] collects the roots that the function names and gives each
//!    root a dense index.
//! 2. [`BlockSummaries`] holds the roots that each block reads, writes, and
//!    overwrites.
//! 3. [`RootLiveness`] solves whole-root liveness over the summaries.
//! 4. [`ParamPlacement`] finds the blocks that need a block parameter for a
//!    root: the pruned iterated dominance frontier.
//! 5. [`SelectionResolution`] finds the indexed selections whose index is a
//!    constant.
//!
//! A bit set in these facts holds dense indices of one [`RootTable`]. The
//! facts of one function use the table of that function.
//!
//! [`FunctionView`]: super::FunctionView

mod liveness;
mod placement;
mod resolution;
mod roots;
mod summary;
mod words;

#[cfg(test)]
mod tests;

pub use liveness::{ExitSeed, RootLiveness};
pub use placement::ParamPlacement;
pub use resolution::{ResolvedSelection, RootConstant, SelectionResolution};
pub use roots::RootTable;
pub use summary::{BlockSummaries, BlockSummary};
