//! Intermediate-representation traits.
//!
//! This module defines contracts only. It holds no IR storage, no builder
//! implementation, and no editing machinery: a consumer owns those and
//! implements the traits here on its own dense storage.
//!
//! - [`rtl`] is the register-transfer level. Instruction semantics describe one
//!   atomic state update through the write-only
//!   [`InstructionBuilder`](rtl::InstructionBuilder). Analyses read sealed
//!   records through [`RecordView`](rtl::RecordView) and
//!   [`FunctionView`](rtl::FunctionView).
//! - [`mlil`] is the medium level: a flat function in static single assignment
//!   form with block parameters, read through
//!   [`FunctionView`](mlil::FunctionView). Value analyses over it belong to the
//!   consumer.
//!
//! The generic algorithms of this crate read through these views and report
//! facts. The consumer applies every change.

pub mod mlil;
pub mod rtl;
