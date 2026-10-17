//! Consumer model: reify-style analyses over a medium-level function that the
//! consumer stores itself.
//!
//! `glint` is a toy shader-style SSA bytecode with block parameters, written
//! as a consumer would: its own storage, its own dialect, and views that
//! implement `cfglib::ir::mlil::FunctionView` and the graph view traits. The
//! documented contract of the view is checked on valid and broken functions,
//! the functions are run through the view alone, and the generic graph
//! algorithms are compared with brute-force oracles over the same functions.

mod algorithms;
mod contract;
mod execution;
mod fixtures;
mod glint;
mod interpreter;
mod oracle;
mod random_functions;
mod skewed;
mod view_contract;
