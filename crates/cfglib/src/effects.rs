//! The effects of code in one shared vocabulary, and the one algorithm that
//! computes them from register-transfer statements.
//!
//! [`Effect`] states what code reads, writes, accesses in memory, raises, and
//! how control leaves it. Its places, locations, and outcomes are type
//! parameters, so every level and every instruction set uses the same shape:
//! a register root, a stack slot, and a value of a higher level are all
//! places.
//!
//! The register-transfer level gets its effects from statements:
//!
//! - [`Summarizer`] turns the statements of one instruction and its control
//!   boundary into an [`Effect`]. It reads any source of
//!   [`Statement`](crate::ir::rtl::Statement)s, so the raw semantics of an
//!   instruction and a sealed record of a consumer get one set of rules.
//! - [`StatementTree`] turns the same statements into expression trees, the
//!   generic part of a readable and exact view.
//! - [`Captured`] keeps the statements of one instruction as owned data. It
//!   implements [`InstructionBuilder`](crate::ir::rtl::InstructionBuilder), so
//!   instruction semantics emit into it without a consumer storage.
//! - [`DialectEffects`] is the only knowledge that a dialect supplies: the
//!   memory access of each effect operation, the state of an intrinsic, the
//!   instruction check, the select operators, and the operator names.
//!
//! Unlike the rest of [`ir`](crate::ir), this module holds an algorithm over
//! the contents of an instruction and an owned instruction buffer. Both are
//! shared vocabulary: each level and each consumer must compute effects with
//! the same rules, and a producer must capture semantics without a consumer.

mod captured;
mod dialect;
mod exprs;
mod vocabulary;

pub use captured::{Captured, CapturedValue};
pub use dialect::{AccessShape, DialectEffects, SelectShape};
pub use exprs::{Expr, ExprId, Exprs, Opaque};
pub use vocabulary::{
    Access, Address, Effect, Exit, RaiseOrigin, Raised, StateTouch, Touch, Write, WriteKind,
};
