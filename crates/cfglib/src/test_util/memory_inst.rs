//! Mock instruction carrying explicit memory events, generic over the
//! test module's location and fence vocabularies.

extern crate alloc;

use alloc::vec::Vec;

use crate::dataflow::InstrInfo;
use crate::memory::{MemoryEvent, MemoryEventInfo};

/// A mock instruction with ordinary data flow (`u8` variables) plus an
/// explicit list of memory events.
#[derive(Debug, Clone)]
pub struct MemInst<L, F> {
    /// Variables read by this instruction.
    pub uses: Vec<u8>,
    /// Variables written by this instruction.
    pub defs: Vec<u8>,
    /// Memory events, in semantic order.
    pub events: Vec<MemoryEvent<L, u8, F>>,
    /// Whether execution can unwind before this instruction completes.
    pub may_unwind: bool,
}

impl<L, F> MemInst<L, F> {
    /// Create a [`MemInst`] with events only (no register data flow).
    pub fn new(events: impl IntoIterator<Item = MemoryEvent<L, u8, F>>) -> Self {
        Self::with_data_flow([], [], events)
    }

    /// Create a [`MemInst`] with uses, defs, and events.
    pub fn with_data_flow(
        uses: impl IntoIterator<Item = u8>,
        defs: impl IntoIterator<Item = u8>,
        events: impl IntoIterator<Item = MemoryEvent<L, u8, F>>,
    ) -> Self {
        Self {
            uses: uses.into_iter().collect(),
            defs: defs.into_iter().collect(),
            events: events.into_iter().collect(),
            may_unwind: false,
        }
    }

    /// This instruction, able to unwind before it completes.
    #[must_use]
    pub fn unwinding(mut self) -> Self {
        self.may_unwind = true;
        self
    }
}

impl<L, F> InstrInfo for MemInst<L, F> {
    type Variable = u8;

    fn uses(&self) -> &[u8] {
        &self.uses
    }
    fn defs(&self) -> &[u8] {
        &self.defs
    }

    fn may_unwind(&self) -> bool {
        self.may_unwind
    }
}

impl<L: Clone + Ord, F: Clone + Eq> MemoryEventInfo for MemInst<L, F> {
    type Location = L;
    type Fence = F;

    fn memory_events(
        &self,
    ) -> impl Iterator<Item = MemoryEvent<Self::Location, Self::Variable, Self::Fence>> {
        self.events.iter().cloned()
    }
}
