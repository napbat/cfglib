//! Incremental, language-parametric stack graphs for name resolution.
//!
//! [`StackGraph`] models references, definitions, scopes, and type-directed
//! lookup using the standard symbol-stack and scope-stack node semantics.
//! [`StackResolution`] executes those semantics directly over a combined graph;
//! partial-path extraction and stitching provide the file-incremental form.

mod partial;
mod path;
mod search;
mod stitching;
mod storage;

#[cfg(test)]
mod tests;

pub use partial::{
    StackPartialPath, StackPartialPathConfig, StackPartialPathDatabase, StackPartialPathId,
    StackPartialPathSet, StackPartialPathStats,
};
pub use path::{StackPath, StackPathError, StackPathStep, StackScopedSymbol};
pub use search::{
    StackLinearResolutionError, StackResolution, StackResolutionIndex, StackReverseIndex,
    StackSearchConfig, StackSearchStats,
};
pub use storage::{
    StackEdge, StackEdgeId, StackEdgeTag, StackFileId, StackFileTag, StackGraph, StackGraphError,
    StackNode, StackNodeId, StackNodeKind, StackNodeTag,
};
