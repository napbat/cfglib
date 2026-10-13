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

mod analysis;
mod dialect;
mod view;

pub use analysis::{AnalysisDialect, Effects, Lattice};
pub use dialect::Dialect;
pub use view::{FunctionView, Successor, ValueDef};
