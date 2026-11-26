//! Data flow analysis framework.
//!
//! Generic fixpoint solvers over graph views:
//!
//! - [`node_fixpoint`] — one fact per node of any
//!   [`GraphView`](crate::GraphView): the per-block problems of a
//!   [`CfgView`](cfg_view::CfgView) such as [`Cfg`](crate::Cfg), and taint,
//!   reachability-with-facts or closure over graphs that are not CFGs
//! - [`edge_fixpoint`] — problems whose facts differ per outgoing edge, over
//!   any [`EdgeView`](crate::EdgeView)
//!
//! Both take one [`Solve`](solve::Solve) descriptor of the seeds and the
//! [`SolveConfig`](solve::SolveConfig), and share the vocabulary of
//! [`solve`]. The [`stream`] module runs the same solves and hands each
//! node's facts out as soon as they are final, instead of keeping them all,
//! and the [`resume`] module re-solves after an edit from the facts of the
//! solve before it. The [`slot_fixpoint`] module schedules units over slots
//! that the caller owns, for a fixpoint that has no graph view and no problem
//! trait. The [`ifds`] module solves interprocedural, context-sensitive
//! problems over a consumer supergraph by IFDS tabulation.
//!
//! # Usage
//!
//! A per-block problem implements [`NodeProblem`](node_fixpoint::NodeProblem)
//! for every [`CfgView`](cfg_view::CfgView) whose instruction type it reads,
//! so one problem serves [`Cfg`](crate::Cfg) and every consumer view alike;
//! [`InstrInfo`](instruction::InstrInfo) is the instruction-side contract
//! such a problem reads (which variables an instruction reads and writes,
//! and whether it may unwind), and [`ProgramPoint`](program_point::ProgramPoint)
//! names an instruction in a block. Variables and memory locations are
//! supplied by the IR adapter; the framework does not impose a numbering
//! scheme. Instruction-level analyses such as liveness, reaching
//! definitions, SSA and memory SSA live in the `il` crate.

pub mod bits;
pub mod cfg_view;
pub mod edge_fixpoint;
pub mod ifds;
pub mod instruction;
pub mod node_fixpoint;
pub mod program_point;
pub mod resume;
mod schedule;
pub mod slot_fixpoint;
pub mod solve;
pub mod stream;
pub(crate) mod worklist;
