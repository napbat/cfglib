//! A compiler middle end over a toy three-address language, the way a
//! frontend that owns its instruction set drives cfglib end to end: build the
//! CFG, compute dominance, SSA, and SCCP, fold and propagate, remove dead
//! code, lower SSA back to copies, and verify.
//!
//! The oracle is an interpreter. Every program runs on many input vectors
//! before and after every transformation, and the observable behavior
//! (printed values, the returned value, a trap) must be identical. The
//! structural claims are checked against brute-force definitions written in
//! the test: dominance by "remove the node and see what is still reachable",
//! frontiers and phi placement from those, and SSA's single-definition and
//! dominated-use laws.

/// The package's golden-file comparison, shared with the unit tests.
#[path = "../../src/test_util/golden.rs"]
mod golden;

mod check;
mod cleanup;
mod folding;
mod interp;
mod oracle;
mod passes;
mod pipeline;
mod programs;
mod structure;
mod toy;
