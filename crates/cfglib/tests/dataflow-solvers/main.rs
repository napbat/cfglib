//! Consumer-style tests of cfglib's dataflow solvers.
//!
//! The block-level, node-level, and edge-sensitive solvers are driven the way
//! reify (reaching and stack analyses), symtree-xref (taint over a symbol
//! graph), and symtree-cfg (liveness, reaching definitions) drive them. The
//! oracles are naive round-robin fixpoint solvers that read adjacency from
//! the test's own graph data, collecting-semantics exploration of program
//! points for the instruction analyses, and concrete executions for abstract
//! interpretation. Random graphs are generated deterministically and include
//! unreachable blocks, removed block slots, and unwind edges.

mod abstract_domains;
mod analysis_oracles;
mod block_solver;
mod edge_solver;
mod graphs;
mod instruction_analyses;
mod node_solver;
mod reference;
