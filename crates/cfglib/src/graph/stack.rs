//! Incremental, language-parametric stack graphs for name resolution.
//!
//! [`StackGraph`] models references, definitions, scopes, and type-directed
//! lookup using the standard symbol-stack and scope-stack node semantics.
//! [`StackResolution`] executes those semantics directly over a combined graph;
//! partial-path extraction and stitching provide the file-incremental form.
//! Every search takes its bounds as a [`PathSearchConfig`](crate::PathSearchConfig)
//! and reports [`PathSearchStats`](crate::PathSearchStats) within its own
//! statistics.
//!
//! Misuse follows one policy, the one of the scope graphs: an identity that
//! is not part of the graph (an unknown file, a node that is out of range or
//! retired, a seed that is not a source reference) is a caller bug and
//! panics; a rule of the graph structure, a stack transition, a branch of a
//! linear walk, or a stale summary is reported as an error.

mod partial;
mod path;
mod search;
mod stitching;
mod storage;

#[cfg(test)]
mod tests;

pub use partial::{
    StackPartialPath, StackPartialPathDatabase, StackPartialPathId, StackPartialPathSet,
    StackPartialPathStats,
};
pub use path::{StackPath, StackPathError, StackPathStep, StackScopedSymbol};
pub use search::{
    StackLinearResolutionError, StackResolution, StackResolutionIndex, StackSearchStats,
};
pub use storage::{
    StackEdge, StackEdgeId, StackEdgeTag, StackFileId, StackGraph, StackGraphError, StackNode,
    StackNodeId, StackNodeKind, StackNodeTag,
};
