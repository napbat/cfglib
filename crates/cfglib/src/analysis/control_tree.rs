//! Control-tree structuring: the structured form of a control-flow graph.
//!
//! [`ControlTree::compute`] reads a [`Cfg`](crate::Cfg) and produces a tree of
//! `if / else`, loops, `switch`, `try / catch`, and linear sequences. This is
//! the core of what a decompiler does with a flat graph.
//!
//! # A fact about the graph
//!
//! The tree holds no instruction. A leaf names the instructions it stands for
//! with a [`BlockSpan`], and every other reference is a
//! [`BlockId`](crate::BlockId), an [`EdgeId`](crate::EdgeId), a
//! [`RegionId`](crate::RegionId), or a [`ProgramPoint`](crate::ProgramPoint).
//! It is a fact about the graph, in the way [`DominatorTree`](crate::DominatorTree)
//! and [`LoopNestingTree`](crate::LoopNestingTree) are, so the tree needs no
//! copy of the instruction type. A consumer that lifts to a higher IR walks the
//! tree and reads each leaf's instructions from its own graph.
//!
//! The nodes live in one flat arena: [`ControlTree::node`] turns a
//! [`ControlNodeId`] into a [`ControlNode`], and each body of statements is a
//! [`ControlBody`] run read with [`ControlTree::body`]. Traversal is
//! [`ControlTree::preorder`], a [`ControlVisitor`] driven by
//! [`ControlTree::walk`], and [`ControlTree::for_each_instruction`] for the
//! instructions in program-structure order.
//! [`ControlTree::write_pseudocode`] prints the tree, with the instruction text
//! supplied by the caller.
//!
//! # What is reported
//!
//! [`ControlTree::compute_with_report`] also returns a [`LiftReport`], which
//! lists exactly which transfers fell back to a `Goto` ([`GotoDiagnostic`],
//! [`GotoReason`]), which blocks only the completeness sweep reached, which
//! exception regions stayed ordinary flow, and which goto labels found no
//! anchor. No owned code is dropped: the tree holds the blocks the entry
//! reaches and the blocks a handler owns, a dead catch pad among them. A
//! protected block the entry cannot reach is dead code and is dropped.
//!
//! # Variants
//!
//! [`ControlTree::compute_predicated`] also splits blocks into runs that share
//! a [`Predicated`](crate::Predicated) guard and wraps each guarded run in a
//! [`ControlNode::Guarded`]; a block then appears in several [`BlockSpan`]s.
//! [`ControlTree::compute_with_report_treating_empty`] structures the graph as
//! if some blocks held no instructions, for a lifter that has already decided
//! that a block's only instruction is a transfer the structure expresses. Both
//! have a `_with_report` form.
//!
//! # Merge points
//!
//! A branch or switch rejoins at the immediate post-dominator of its block. A
//! region that no path leaves has no exit, so the post-dominator tree cuts it
//! and a branch with an arm that never exits has no merge point; see
//! [`ControlTree::compute_with_report`] for what that means for the shape of
//! the tree.
//!
//! # Examples
//!
//! ```
//! use cfglib::{Cfg, ControlNode, ControlTree, EdgeKind};
//!
//! // entry: test; --true--> then; --false--> merge; then --> merge
//! let mut cfg = Cfg::<&str>::new();
//! let then_block = cfg.new_block();
//! let merge = cfg.new_block();
//! cfg.block_mut(cfg.entry()).push("test");
//! cfg.block_mut(then_block).push("then");
//! cfg.block_mut(merge).push("merge");
//! cfg.add_edge(cfg.entry(), then_block, EdgeKind::ConditionalTrue);
//! cfg.add_edge(cfg.entry(), merge, EdgeKind::ConditionalFalse);
//! cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);
//!
//! let (tree, report) = ControlTree::compute_with_report(&cfg);
//! assert!(report.is_fully_structured());
//!
//! let text = tree
//!     .to_pseudocode(|sink, point| {
//!         sink.write_str(cfg.block(point.block).instructions()[point.inst_idx])
//!     })
//!     .unwrap();
//! assert_eq!(text, "if (test) {\n    then\n}\nmerge\n");
//! assert!(tree
//!     .preorder()
//!     .any(|id| matches!(tree.node(id), ControlNode::IfThenElse { .. })));
//! ```

mod display;
mod node;
mod predicated;
mod report;
mod structure;
mod tree;
mod walk;

#[cfg(test)]
mod tests;

pub use node::{
    BlockSpan, CatchHandler, ControlBody, ControlNode, ControlNodeId, ControlRun, LoopKind,
    SwitchCase,
};
pub use report::{GotoDiagnostic, GotoReason, LiftReport};
pub use tree::{ChildBodies, ControlTree};
pub use walk::{ControlDescent, ControlVisitor, Preorder};
