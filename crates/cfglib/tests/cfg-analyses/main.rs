//! Integration tests: a code-quality tool consuming cfglib's CFG analyses
//! (purity, tail calls, profiles, value numbering, alias sets, expression
//! recovery, pattern detection, loop depths, interval partitions) over a toy
//! three-address statement language with named locals.
//!
//! Every analysis is judged by hand-labeled fixtures or by a brute-force
//! reference written in the test: closure of merges, symbolic expression
//! trees, flow equations, definition-level dominance, textbook intervals.

mod aliases;
mod expressions;
mod intervals;
mod nesting;
mod numbering;
mod patterns;
mod profile;
mod purity;
mod random;
mod splitting;
mod stmt;
mod tail_calls;
