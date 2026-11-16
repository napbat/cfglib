//! Consumer model: instruction semantics that emit statements, and the
//! effects and the expression trees of those statements.
//!
//! `wren` is a toy dialect with partial registers, flags, a slot file for
//! indexed access, a sparse bank, memory in two address spaces, an
//! instruction check, and a select operator. Its operators and effects are
//! not `Copy`, so the algorithms only borrow them.

mod captured;
mod summarizer;
mod tree;
mod wren;
