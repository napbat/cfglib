//! The structured contract: statement and expression trees over declared
//! variables.
//!
//! HLIL is the structured, expression-oriented level of the IR stack.
//! Statements form trees (conditionals, loops, switches, exception regions)
//! and values nest as typed expression trees. The shapes of both are
//! universal and owned by the library ([`StmtShape`], [`ExprShape`]). A
//! consumer-defined [`Dialect`] supplies everything semantic: the operations,
//! the constants, the value types, and the metadata of a variable. The
//! neutrality contract is the one of [`ir::mlil`](crate::ir::mlil).
//!
//! The consumer owns the storage. It implements [`FunctionView`] on its dense
//! statement, expression, and variable arenas, and the view lends the shapes
//! out as borrowed slices. It implements [`FunctionBuilder`] to receive a new
//! function: a write-only protocol in which the algorithm appends children
//! before parents, and the consumer seals the function in
//! [`finish`](FunctionBuilder::finish). No algorithm edits the function it
//! reads.
//!
//! The generic algorithms:
//!
//! - [`verify`] checks the structural invariants and calls the dialect's
//!   [`VerifyDialect`] hook;
//! - [`pseudocode`] writes the indented text of a function through the
//!   dialect's [`Pseudocode`] hooks;
//! - [`VariablePlacements`] anchors each variable's declaration in the
//!   narrowest lexical body that holds all of its occurrences;
//! - [`recover_structure`] rebuilds a function through a builder with counted
//!   loops, conditional selections, and paired regions restored, steered by
//!   the dialect's [`RecoverDialect`] hooks;
//! - [`for_each_child_body`], [`visit_statements`], [`visit_expression_tree`],
//!   [`expressions_equal`], and [`compound_assignment`] are the walks that
//!   every consumer of the shapes needs.

mod builder;
mod correspondence;
mod dialect;
mod error;
mod name;
mod placement;
mod pseudocode;
mod recover;
mod shape;
mod verify;
mod view;

pub use builder::FunctionBuilder;
pub use correspondence::Correspondence;
pub use dialect::{Dialect, Pseudocode, RecoverDialect, VerifyDialect};
pub use error::{LiftError, LowerError, RecoverError};
pub use placement::VariablePlacements;
pub use pseudocode::{pseudocode, write_pseudocode};
pub use recover::{Recovery, has_recovery_candidates, recover_structure};
pub use shape::{ExprShape, Handler, HandlerKind, StmtShape, SwitchArm};
pub use verify::{VerificationIssue, VerificationReport, verify};
pub use view::{
    ChildBodyKind, Entity, FunctionView, compound_assignment, expressions_equal,
    for_each_child_body, for_each_statement_expression, visit_expression_tree, visit_statements,
};

#[cfg(test)]
mod tests;
