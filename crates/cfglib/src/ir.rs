//! Intermediate-representation traits and vocabulary.
//!
//! This module defines contracts. It holds no IR storage, no builder
//! implementation, no editing machinery, and no algorithm over the contents
//! of an instruction: a consumer owns those and implements the traits here
//! on its own dense storage. A producer that creates a function drives a
//! builder trait that the consumer implements, and the consumer's builder
//! seals the function.
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
//!   implements, with the walks that every reader of the shapes needs. The
//!   consumer stores the function.
//!
//! Every level is language-neutral. The library defines the shapes and the
//! traits, and names no language, runtime, ABI, or instruction set: a
//! consumer-defined `Dialect` of each level supplies the vocabulary, as plain
//! data that the consumer's storage holds.

pub mod hlil;
pub mod mlil;
pub mod rtl;
