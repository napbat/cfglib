//! Integration tests: a code-quality tool consuming cfglib's CFG analyses
//! (pattern detection, loop depths, interval partitions) over a toy
//! three-address statement language with named locals.
//!
//! Every analysis is judged by hand-labeled fixtures or by a brute-force
//! reference written in the test: definition-level dominance and textbook
//! intervals.

mod intervals;
mod nesting;
mod patterns;
mod random;
mod splitting;
mod stmt;
