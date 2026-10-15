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
//!
//! The module holds the read-only view contract and nothing else. Value
//! analyses over a function, such as use lists, liveness, constant
//! propagation, and dead-instruction marking, belong to the consumer. The
//! consumer owns the semantics of its dialect, so it also owns the analyses
//! that depend on them.

mod dialect;
mod view;

pub use dialect::Dialect;
pub use view::{FunctionView, Successor, ValueDef};
