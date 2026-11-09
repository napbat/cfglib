//! Data flow analysis framework.
//!
//! Provides generic fixpoint solvers over any
//! [`CfgView`](cfg_view::CfgView):
//!
//! - [`fixpoint`] — block-level problems, forward or backward
//! - [`edge_fixpoint`] — problems whose facts differ per outgoing edge
//!
//! [`CfgView`](cfg_view::CfgView) is the one read-only trait that the
//! instruction-level machinery reads, and [`Cfg`](crate::Cfg) implements it.
//! The [`node_fixpoint`] module is the graph-shaped counterpart, solving
//! per-node fact problems over any
//! [`GraphView`](crate::GraphView) (taint or reachability
//! over a value-flow graph, closure over an import graph).
//! The [`slot_fixpoint`] module schedules units over slots that the caller
//! owns, for a fixpoint that has no graph view and no problem trait.
//!
//! # Usage
//!
//! Implement [`InstrInfo`](instruction::InstrInfo) for your instruction type to declare which
//! IR variables each instruction reads from and writes to, then state a
//! problem over them. The [`MemoryEventInfo`](crate::MemoryEventInfo)
//! extension declares ordered memory events and an instruction-level
//! read/modify/write summary. Variables and memory locations are supplied by
//! the IR adapter; the framework does not impose a numbering scheme.
//! Instruction-level analyses such as liveness, reaching definitions, SSA
//! and memory SSA live in the `il` crate.

pub mod bits;
pub mod cfg_view;
pub mod edge_fixpoint;
pub mod fixpoint;
pub mod instruction;
pub mod node_fixpoint;
pub mod program_point;
pub mod slot_fixpoint;
mod worklist;
