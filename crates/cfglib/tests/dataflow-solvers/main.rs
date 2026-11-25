//! Consumer-style tests of cfglib's dataflow solvers.
//!
//! The block-level, node-level, and edge-sensitive solvers are driven the way
//! reify (reaching and stack analyses) and symtree-xref (taint over a symbol
//! graph) drive them. The oracles are naive round-robin fixpoint solvers that
//! read adjacency from the test's own graph data. Random graphs are generated
//! deterministically and include unreachable blocks, removed block slots, and
//! unwind edges. Every iteration order is checked against the index order,
//! an interval domain checks widening and narrowing, and every stream is
//! checked against the solve that keeps its facts.

mod block_solver;
mod edge_solver;
mod graphs;
mod lifted;
mod node_solver;
mod reference;
mod streaming;
mod widening;
