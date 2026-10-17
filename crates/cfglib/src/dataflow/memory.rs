//! Memory-state and memory-carried value data flow.
//!
//! [`MemorySsa`] constructs versioned state for consumer-defined memory
//! locations. [`MemoryValueFlow`] connects ordinary SSA values to those states
//! through exact, ordered memory events.

mod ssa;
mod value;

pub use ssa::{
    ConservativeMemoryAlias, ExactMemoryAlias, MemoryAlias, MemoryClassId, MemoryDefinition,
    MemoryEventSite, MemoryLocationClass, MemoryPhi, MemorySsa, MemorySsaEvent, MemorySsaScratch,
    MemorySsaValue, MemoryUse, index_paths_may_overlap,
};
pub use value::{
    MemoryValueEdge, MemoryValueFlow, MemoryValueFlowError, MemoryValueFlowScratch,
    MemoryValueNode, MemoryValueRole,
};
