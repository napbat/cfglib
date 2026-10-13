//! CFG transformation passes.
//!
//! All mutation passes operate in-place and return the number of
//! blocks, edges, or instructions affected. Every pass entry point is
//! re-exported at the crate root.
//!
//! - [`cleanup`] — unreachable removal, block merging, empty-block bypass, combined simplify.
//! - [`coloring`] — interference graphs and greedy graph coloring.
//! - [`contract`] — edge contraction and node splitting.
//! - [`critical`] — critical edge splitting (required for SSA).
//! - [`dce`] — dead code elimination via liveness analysis.
//! - [`linearize`] — re-serialize a CFG back to a flat instruction stream.
//! - [`loops`] — loop rotation and loop-invariant detection.
//! - [`pass`] — named, ordered, fallible pass composition and reports.
//! - [`pre`] — partial redundancy elimination.

pub mod cleanup;
pub mod coloring;
pub mod contract;
pub mod critical;
pub mod dce;
pub mod layout;
pub mod linearize;
pub mod loops;
pub mod pass;
pub mod pre;
