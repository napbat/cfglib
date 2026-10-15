//! Intermediate-representation traits and the generic algorithms over them.
//!
//! This module defines contracts. It holds no IR storage, no builder
//! implementation, and no editing machinery: a consumer owns those and
//! implements the traits here on its own dense storage. Where an algorithm
//! creates a function, it drives a builder trait that the consumer
//! implements, and the consumer's builder seals the function.
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
//! - [`hlil`] is the structured level: statement and expression trees read
//!   through a [`FunctionView`](hlil::FunctionView) and built through a
//!   write-only [`FunctionBuilder`](hlil::FunctionBuilder) that the consumer
//!   implements. Generic algorithms verify, print, analyze, and rebuild such a
//!   function, and the consumer stores it.
//!
//! The generic algorithms of this crate read through these views and report
//! facts. An algorithm never edits the function it reads, and the consumer
//! applies every change.

pub mod hlil;
pub mod mlil;
pub mod rtl;
