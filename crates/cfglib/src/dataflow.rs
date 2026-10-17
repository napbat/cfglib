//! Data flow analysis framework.
//!
//! Provides generic infrastructure for computing data flow properties
//! over any [`CfgView`]:
//!
//! - **Reaching definitions** — which writes can reach a given point
//! - **Liveness** — which variables are live at each point
//! - **Def-use / use-def chains** — linking writers to readers
//!
//! [`CfgView`] is the one read-only trait that the
//! instruction-level machinery reads, and [`Cfg`](crate::Cfg) implements it.
//! The [`node_fixpoint`] module is the graph-shaped counterpart, solving
//! per-node fact problems over any
//! [`GraphView`](crate::GraphView) (taint or reachability
//! over a value-flow graph, closure over an import graph).
//!
//! # Usage
//!
//! Implement [`InstrInfo`] for your instruction type to declare which
//! IR variables each instruction reads from and writes to, then run any
//! of the provided analyses. The [`MemoryEventInfo`](crate::MemoryEventInfo)
//! extension adds ordered memory data flow and an instruction-level
//! read/modify/write summary. Variables and memory locations are supplied by
//! the IR adapter; the framework does not impose a numbering scheme.

pub mod abstract_interpretation;
pub mod bits;
pub mod cfg_view;
pub mod constant_propagation;
pub mod copy_propagation;
pub mod def_use;
pub mod edge_fixpoint;
pub mod fixpoint;
pub mod instruction;
pub mod liveness;
pub mod memory;
pub mod node_fixpoint;
pub mod phi_web;
pub mod program_point;
pub mod reaching;
pub mod sccp;
pub mod ssa;
pub mod ssa_destruction;
pub mod unwind;
mod worklist;
