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
//! - [`digraphs`] — every small digraph, and a graph from an edge list.
//! - [`golden`] — exact-output comparison against `tests/golden` files.
//! - [`postdominance`] — post-dominance without a unique exit, by fixpoint.
//! - [`rng`] — a seeded generator for reproducible randomized tests.
//! - [`protected_region`] — one block that can unwind into a handler.
//! - [`probability`] — an edge weight from a decimal literal.
//!
//! `DfInst` implements every instruction trait the unit tests need
//! (`FlowControl`, `DisplayInstr`, `InstrInfo`, `CallInfo`) so
//! test modules don't need to define their own instruction types.

mod dataflow_inst;
pub(crate) mod digraphs;
pub(crate) mod golden;
mod live_variables;
mod memory_inst;
mod mock_inst;
pub(crate) mod postdominance;
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

/// The edge weight `value`, which must lie in `0.0..=1.0`.
pub(crate) fn probability(value: f64) -> crate::edge::Probability {
    crate::edge::Probability::from_f64(value).expect("a test weight lies in 0.0..=1.0")
}
