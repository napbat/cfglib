//! Integration tests: a consumer recovering expression trees from blocks
//! whose length it does not control, such as generated or obfuscated code.
//!
//! A block of chained single-use temporaries recovers to one tree as deep as
//! the block is long, so every operation on a tree must keep its own stack.

mod chain;
mod instruction;
