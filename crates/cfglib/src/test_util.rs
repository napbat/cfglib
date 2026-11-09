//! Shared test helpers for cfglib.
//!
//! Provides mock instruction types used across all test modules:
//!
//! - [`MockInst`] — minimal, flow-effect only (for graph/transform tests).
//! - [`DfInst`] — full-featured: flow + defs/uses + effects + calls.
//! - [`LiveVariables`] — a backward block problem over [`DfInst`] code
//!   for the fixpoint solver tests.
//! - [`MemInst`] — uses/defs plus explicit memory events, generic over
//!   the test module's location and fence vocabularies.
//! - [`shapes`] — block-and-edge shapes shared by the reusable-scratch
//!   differential tests.
//! - [`golden`] — exact-output comparison against `tests/golden` files.
//! - [`rng`] — a seeded generator for reproducible randomized tests.
//! - [`protected_region`] — one block that can unwind into a handler.
//!
//! `DfInst` implements every instruction trait the unit tests need
//! (`FlowControl`, `DisplayInstr`, `InstrInfo`, `CallInfo`) so
//! test modules don't need to define their own instruction types.

mod dataflow_inst;
pub(crate) mod golden;
mod live_variables;
mod memory_inst;
mod mock_inst;
mod protected;
pub(crate) mod rng;
pub(crate) mod shapes;

pub(crate) use dataflow_inst::{
    DfInst, df_call, df_def, df_ff, df_throwing, df_use, df_with_effect,
};
pub(crate) use live_variables::LiveVariables;
pub(crate) use memory_inst::MemInst;
pub(crate) use mock_inst::{MockInst, diamond_cfg, ff};
pub(crate) use protected::protected_region;
