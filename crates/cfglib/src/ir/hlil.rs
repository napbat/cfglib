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
//! function: a write-only protocol in which the producer appends children
//! before parents, and the consumer seals the function in
//! [`finish`](FunctionBuilder::finish).
//!
//! [`for_each_child_body`], [`for_each_statement_expression`],
//! [`visit_statements`], and [`visit_expression_tree`] are the walks that
//! every reader of the shapes
//! needs. The algorithms over this contract (lifting, lowering, structure
//! recovery, verification, printing) live with the consumers of the
//! contract, not here.
//!
//! # Depth
//!
//! The walks over statements, [`visit_statements`] and
//! [`for_each_child_body`], and the walk over an expression tree,
//! [`visit_expression_tree`], keep a stack of their own. A nest as deep as
//! generated code or untrusted input produces costs memory, not stack
//! frames of the thread.

mod builder;
mod dialect;
mod shape;
mod view;

pub use builder::FunctionBuilder;
pub use dialect::Dialect;
pub use shape::{ExprShape, Handler, HandlerKind, StmtShape, SwitchArm};
pub use view::{
    ChildBodyKind, Entity, FunctionView, for_each_child_body, for_each_statement_expression,
    visit_expression_tree, visit_statements,
};

#[cfg(test)]
mod tests;
