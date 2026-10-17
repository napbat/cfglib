//! Integration tests: a compiler-backend consumer composing cfglib's CFG
//! transforms into pipelines over a toy register machine.
//!
//! Semantics-preserving passes are judged by running the program before and
//! after on a set of inputs; structural contracts (no critical edges, proper
//! colorings, complete linearizations, honest `Rewrite` records) are judged by
//! brute-force oracles written against the documents. Generic passes run on
//! both `Cfg<Inst>` and `Cfg<Inst, Route>`, the way a consumer that keeps edge
//! payloads would call them.

mod cleanup;
mod coloring;
mod critical_edges;
mod dead_code;
mod linearize;
mod loops;
mod oracles;
mod passes;
mod programs;
mod redundancy;
mod relaxation;
mod renumbering;
mod reversal;
mod toy;
