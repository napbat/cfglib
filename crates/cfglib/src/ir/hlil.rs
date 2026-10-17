//! The structured contract: statement and expression trees over declared
//! variables.
//!
//! HLIL is the structured, expression-oriented level of the IR stack.
//! Statements form trees (conditionals, loops, switches, exception regions)
//! and values nest as typed expression trees. The shapes of both are
//! universal and owned by the library ([`StmtShape`], [`ExprShape`]). A
//! consumer-defined [`Dialect`] supplies everything semantic: the operations,
//! the constants, the value types, and the metadata of a variable. The
//! neutrality contract is the one of [`ir`](crate::ir).
//!
//! The consumer owns the storage. It implements [`FunctionView`] on its dense
//! statement, expression, and variable arenas, and the view lends the shapes
//! out as borrowed slices. It implements [`FunctionBuilder`] to receive a new
//! function: a write-only protocol in which the algorithm appends children
//! before parents, and the consumer seals the function in
//! [`finish`](FunctionBuilder::finish). No algorithm edits the function it
//! reads.
//!
//! A flat function is written the same way. The consumer implements
//! [`FlatBuilder`] over its own [`Cfg`](crate::Cfg) of instructions and edge
//! payloads, and [`lower`](fn@lower) names each instruction and edge in HLIL
//! terms ([`Lowered`], [`LowerEdge`]). The consumer chooses the flat operation
//! and the [`EdgeKind`](crate::EdgeKind).
//!
//! A flat function is read through [`FlatFunction`]: the consumer's
//! [`Cfg`](crate::Cfg) with its variable table and signature. The instruction
//! type of the graph implements [`LiftInstruction`] and the edge payload
//! [`LiftEdge`] to say what each of them stands for ([`InstructionShape`]),
//! and the marker's [`LiftDialect`] holds the policy that has no subject.
//!
//! The generic algorithms:
//!
//! - [`lift`](fn@lift) structures a [`FlatFunction`] into a
//!   [`ControlTree`](crate::ControlTree) and translates it through a
//!   [`FunctionBuilder`], inlining each definition into its consumer when that
//!   provably keeps the evaluation order, and returns the [`Correspondence`]
//!   of flat instructions to structured entities;
//! - [`lower`](fn@lower) flattens a function into blocks and edges through a
//!   [`FlatBuilder`], linearizes the expression trees into typed temporaries,
//!   and registers the declared exception regions;
//! - [`verify`] checks the structural invariants and calls the dialect's
//!   [`VerifyDialect`] hook;
//! - [`pseudocode`] writes the indented text of a function through the
//!   dialect's [`Pseudocode`] hooks;
//! - [`VariablePlacements`] anchors each variable's declaration in the
//!   narrowest lexical body that holds all of its occurrences;
//! - [`recover_structure`] rebuilds a function through a builder with counted
//!   loops, conditional selections, and paired regions restored, steered by
//!   the dialect's [`RecoverDialect`] hooks;
//! - [`for_each_child_body`], [`for_each_statement_expression`],
//!   [`visit_statements`], [`visit_expression_tree`], [`expressions_equal`],
//!   and [`compound_assignment`] are the walks that every consumer of the
//!   shapes needs.
//!
//! The algorithms compose. A caller lifts a flat function, recovers the
//! structure of the lifted function when [`has_recovery_candidates`] finds
//! something to recover, and moves the [`Correspondence`] along with
//! [`Correspondence::remap`].
//!
//! # Depth
//!
//! The depth of an expression tree costs no call frames in any algorithm: a
//! chain of operands as deep as generated code produces is walked, lowered,
//! recovered, and printed on a stack of the algorithm's own. Statements nest
//! as deep as the program text does, and [`verify`], [`lift`](fn@lift),
//! [`lower`](fn@lower), [`recover_structure`], [`pseudocode`], and
//! [`VariablePlacements`] recurse once per level of that nesting. The walks
//! over statements, [`visit_statements`] and [`for_each_child_body`], do not
//! recurse.

mod builder;
mod correspondence;
mod dialect;
mod error;
mod flat;
mod flat_builder;
mod lift;
mod lower;
mod name;
mod placement;
mod pseudocode;
mod recover;
mod shape;
mod transfer;
mod verify;
mod view;

pub use builder::FunctionBuilder;
pub use correspondence::Correspondence;
pub use dialect::{Dialect, LiftDialect, Pseudocode, RecoverDialect, VerifyDialect};
pub use error::{LiftError, LowerError, RecoverError};
pub use flat::{FlatFunction, InstructionShape, LiftEdge, LiftInstruction};
pub use flat_builder::{Appended, FlatBuilder, LowerEdge, Lowered, Occurrence, VariableOrigin};
pub use lift::{LiftMetadata, LiftedFunction, lift, lift_trampolines, lift_with_tree};
pub use lower::{LoweredFunction, lower};
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
