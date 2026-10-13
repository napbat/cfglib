//! The medium-level contract: a flat function in static single assignment
//! form with block parameters.
//!
//! Every instruction defines zero or more values and reads values by handle.
//! There is no expression tree and no mutable variable. A block receives its
//! join values as ordered parameters, and the terminator that ends each block
//! owns its successor records: the target block, the values passed to the
//! target's parameters, and the edge metadata. The parameters of the entry
//! block are the parameters of the function. Predecessors are derived from
//! the successor records, never stored.
//!
//! [`FunctionView`] returns slices for contiguous data, so an implementation
//! keeps its parameters, instructions, operands, and results in dense pools.
//! [`AnalysisDialect`] supplies the semantic hooks that the generic analyses
//! of this crate need.
//!
//! The generic algorithms read a function through these traits and report
//! facts. [`UseIndex`] lists the uses of each value, [`ValueLiveness`] solves
//! liveness, [`SccpFacts`] holds the result of sparse conditional constant
//! propagation, and [`DeadInstructions`] marks the instructions that a pass can
//! delete. None of them changes the function.

mod analysis;
mod csr;
mod dce;
mod dialect;
mod liveness;
mod sccp;
mod uses;
mod view;

#[cfg(test)]
mod tests;

pub use analysis::{AnalysisDialect, Effects, Lattice};
pub use dce::DeadInstructions;
pub use dialect::Dialect;
pub use liveness::ValueLiveness;
pub use sccp::SccpFacts;
pub use uses::{Use, UseIndex};
pub use view::{FunctionView, Successor, ValueDef};
