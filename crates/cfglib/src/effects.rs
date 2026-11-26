//! The effects of code in one shared vocabulary, and the one algorithm that
//! computes them from register-transfer statements.
//!
//! [`Effect`] states what code reads, writes, accesses in memory, raises, and
//! how control leaves it. Its roots, locations, and outcomes are type
//! parameters, so every level and every instruction set uses the same shape:
//! a register root, a stack slot, and a value of a higher level are all
//! roots.
//!
//! The register-transfer level gets its effects from statements:
//!
//! - [`Effect::compute`] turns the statements of one record and its control
//!   boundary into an [`Effect`]. It reads any
//!   [`RecordView`](crate::ir::rtl::RecordView), so the raw semantics of an
//!   instruction and a sealed record of a consumer get one set of rules.
//! - [`StatementTree::compute`] turns the same statements into expression
//!   trees, the generic part of a readable and exact view.
//! - [`Captured`] keeps the statements of one instruction as owned data. Its
//!   [`CaptureBuilder`] implements
//!   [`InstructionBuilder`](crate::ir::rtl::InstructionBuilder), so
//!   instruction semantics emit into it without a consumer storage.
//! - [`DialectEffects`] is the only knowledge that a dialect supplies: the
//!   memory access of each effect operation, the state of an intrinsic, the
//!   select operators, and the operator names. What only one consumer knows,
//!   such as the value of a relocated operator or the check that one
//!   instruction set runs before each instruction, stays with that consumer:
//!   it substitutes values in the computed expressions and finds the outcomes
//!   of one effect by the position of the effect.
//!
//! Unlike the rest of [`ir`](crate::ir), this module holds an algorithm over
//! the contents of an instruction and an owned instruction buffer. Both are
//! shared vocabulary: each level and each consumer must compute effects with
//! the same rules, and a producer must capture semantics without a consumer.

mod captured;
mod defs;
mod dialect;
mod exprs;
mod pool;
mod summarizer;
mod tree;
mod vocabulary;

pub use captured::{CaptureBuilder, Captured, CapturedValue};
pub use dialect::{AccessShape, DialectEffects, SelectShape};
pub use exprs::{Expr, ExprId, Exprs, Opaque};
pub use summarizer::RtlEffect;
pub use tree::{StatementTree, TreeItem, TreeNode, TreeNodeId, TreeSpan, TreeSpanValue};
pub use vocabulary::{
    Access, Effect, MemoryAddress, RaiseOrigin, Raised, StateTouch, Write, WriteKind,
};
