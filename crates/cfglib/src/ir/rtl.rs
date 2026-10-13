//! The register-transfer contract.
//!
//! One machine instruction is one atomic state update:
//!
//! - every read observes the single state before the instruction;
//! - pure computation, loads, stores, and intrinsics follow in a stated order;
//! - writes collect into one delta that commits at the instruction boundary;
//! - one control boundary ends the instruction.
//!
//! A state update is never partially committed. A fault is a typed outcome
//! that an effect or a raise declares; it is not a control-flow successor. An
//! instruction has no control flow of its own: a repeated instruction branches
//! to its own address, and a choice inside one instruction is a select or a
//! guarded write.
//!
//! Instruction semantics emit a record through the write-only
//! [`InstructionBuilder`]. The consumer owns the storage behind the builder and
//! exposes sealed records through [`RecordView`] and whole functions through
//! [`FunctionView`]. A sealed record is read-only.
//!
//! Every RTL value is a signless bit vector ([`Bits`]). Signedness, a floating
//! format, and a lane structure are properties of an operator, not of a value.

mod bits;
mod builder;
mod dialect;
mod fold;
mod statement;
mod view;

pub use bits::{BitSlice, Bits};
pub use builder::InstructionBuilder;
pub use dialect::{Dialect, Place};
pub use fold::{ConstBits, Fold};
pub use statement::{
    Action, Control, Raise, Selection, SpanAction, Statement, ValueRange, WriteTarget,
};
pub use view::{FunctionView, RecordView};
