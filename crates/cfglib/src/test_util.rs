//! Shared test helpers for cfglib.
//!
//! Provides mock instruction types used across all test modules:
//!
//! - [`MockInst`] — minimal, flow-effect only (for graph/transform tests).
//! - [`DfInst`] — full-featured: flow + defs/uses + effects + optional
//!   copy semantics, expression decomposition, and constant values.
//! - [`VnInst`] — operator-carrying, for value-numbering and PRE tests.
//! - [`UdInst`] — bare uses/defs plus a test-specific payload, for test
//!   modules that attach their own semantic trait impl.
//! - [`MemInst`] — uses/defs plus explicit memory events, generic over
//!   the test module's location and fence vocabularies.
//! - [`shapes`] — block-and-edge shapes shared by the reusable-scratch
//!   differential tests.
//! - [`toy`] — the shared core of the per-level IR "toy dialect"
//!   fixtures (source spans, lifted-statement metadata).
//!
//! `DfInst` implements **all** instruction traits (`FlowControl`,
//! `DisplayInstr`, `InstrInfo`, `EffectInfo`, `Predicated`, `CopySource`,
//! `ExprInstr`, `ConstantFolder`, `CallInfo`) so test modules don't need
//! to define their own instruction types.

mod dataflow_inst;
pub(crate) mod golden;
mod memory_inst;
mod mock_inst;
pub(crate) mod shapes;
pub(crate) mod toy;
mod ud_inst;
mod vn_inst;

pub(crate) use dataflow_inst::{
    DfInst, TestEffect, df_call, df_const, df_copy, df_def, df_ff, df_impure, df_op, df_pred,
    df_pure, df_use, df_with_effect,
};
pub(crate) use memory_inst::MemInst;
pub(crate) use mock_inst::{MockInst, diamond_cfg, ff};
pub(crate) use ud_inst::{UdInst, ud_inst};
pub(crate) use vn_inst::{VnInst, vn_impure, vn_inst};
