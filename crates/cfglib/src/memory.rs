//! Language-independent memory events.
//!
//! Instructions opt in through [`MemoryEventInfo`]. The common event shape
//! distinguishes reads, writes, compound read/modify/write operations,
//! address-variable dependencies, atomicity, and fences. Locations and fence
//! details remain consumer-owned: an ISA can retain address spaces and
//! hardware scopes while a source language can use fields, objects, and
//! language-level ordering domains.
//!
//! This module owns only the semantic event vocabulary. Memory SSA, event
//! traces and combined value flow live in the `il` crate.

extern crate alloc;

use alloc::vec::Vec;

use crate::InstrInfo;

/// Instruction-level summary of reported memory accesses.
///
/// The three properties are tracked separately at the summary level:
/// [`READ_WRITE`](Self::READ_WRITE) means an instruction has separate reads
/// and writes, while [`READ_MODIFY_WRITE`](Self::READ_MODIFY_WRITE) records at
/// least one compound access whose replacement depends on the previous value.
/// A modification always implies both a read and a write.
/// Detailed locations, ordering, and atomicity remain available through
/// [`MemoryEventInfo::memory_events`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryOperations(u8);

impl MemoryOperations {
    const READ_BIT: u8 = 1 << 0;
    const MODIFY_BIT: u8 = 1 << 1;
    const WRITE_BIT: u8 = 1 << 2;

    /// No memory access. An instruction may still emit a fence event.
    pub const NONE: Self = Self(0);
    /// One or more reads and no writes.
    pub const READ: Self = Self(Self::READ_BIT);
    /// One or more possible writes and no reads.
    pub const WRITE: Self = Self(Self::WRITE_BIT);
    /// Separate reads and writes, with no compound read/modify/write access.
    pub const READ_WRITE: Self = Self(Self::READ_BIT | Self::WRITE_BIT);
    /// At least one compound read/modify/write access.
    pub const READ_MODIFY_WRITE: Self = Self(Self::READ_BIT | Self::MODIFY_BIT | Self::WRITE_BIT);

    /// Whether at least one access consumes a location's value.
    #[must_use]
    pub const fn reads(self) -> bool {
        self.0 & Self::READ_BIT != 0
    }

    /// Whether at least one replacement depends on that location's old value.
    #[must_use]
    pub const fn modifies(self) -> bool {
        self.0 & Self::MODIFY_BIT != 0
    }

    /// Whether at least one access may change a location.
    #[must_use]
    pub const fn writes(self) -> bool {
        self.0 & Self::WRITE_BIT != 0
    }

    /// Whether the summary contains no memory accesses.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Combines the access properties reported by two event groups.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl FromIterator<Self> for MemoryOperations {
    /// Combines the properties of every item, the way
    /// [`union`](Self::union) combines two. An empty iterator yields
    /// [`NONE`](Self::NONE).
    fn from_iter<T: IntoIterator<Item = Self>>(operations: T) -> Self {
        operations.into_iter().fold(Self::NONE, Self::union)
    }
}

/// The direction of one access to a memory location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryAccessKind {
    /// Reads the location without writing it.
    Read,
    /// May write the location without consuming its previous value.
    Write,
    /// Reads a location and may write a replacement derived from that value.
    ///
    /// This includes conditional atomics such as compare-and-store: their
    /// resulting memory state is a definition even when the runtime value is
    /// unchanged because the comparison failed.
    ReadModifyWrite,
}

impl MemoryAccessKind {
    /// Instruction-level properties contributed by this access.
    #[must_use]
    pub const fn operations(self) -> MemoryOperations {
        match self {
            Self::Read => MemoryOperations::READ,
            Self::Write => MemoryOperations::WRITE,
            Self::ReadModifyWrite => MemoryOperations::READ_MODIFY_WRITE,
        }
    }

    /// Whether the access consumes the location's previous value.
    #[must_use]
    pub const fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadModifyWrite)
    }

    /// Whether the replacement depends on the location's previous value.
    #[must_use]
    pub const fn modifies(self) -> bool {
        matches!(self, Self::ReadModifyWrite)
    }

    /// Whether the access may change the location.
    #[must_use]
    pub const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadModifyWrite)
    }
}

/// Whether an access is indivisible with respect to competing accesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryAtomicity {
    /// An ordinary, non-atomic access.
    NonAtomic,
    /// An atomic access.
    Atomic,
}

/// One access to a consumer-defined memory location.
///
/// `address_uses` links the access back to the instruction variables that
/// select its binding or sub-location. Constant address components can be
/// carried by `location`; if they are omitted, the location must denote a
/// conservative region containing every possible selected sub-location.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryAccess<L, V> {
    /// The location or conservative region being accessed.
    location: L,
    /// Instruction uses that determine the accessed binding or sub-location.
    address_uses: Vec<V>,
    /// Instruction uses whose runtime values are written to this location.
    ///
    /// These are value operands, not address operands. A store of `value` to
    /// `base[index]` therefore reports `value` here and reports `base` and
    /// `index` through [`address_uses`](Self::address_uses). Empty is valid for
    /// an opaque clobber whose replacement value is unknown.
    value_uses: Vec<V>,
    /// Instruction definitions receiving values read from this location.
    ///
    /// Empty is valid for a read performed only for its effects. For a
    /// read/modify/write, these definitions receive the value observed before
    /// the replacement is committed.
    value_defs: Vec<V>,
    /// Whether the location is read, written, or modified in place.
    kind: MemoryAccessKind,
    /// Whether competing observers see the access as indivisible.
    atomicity: MemoryAtomicity,
}

impl<L, V> MemoryAccess<L, V> {
    /// Creates a read whose observed value is assigned to `value_defs`.
    #[must_use]
    pub fn read(location: L, value_defs: impl IntoIterator<Item = V>) -> Self {
        Self {
            location,
            address_uses: Vec::new(),
            value_uses: Vec::new(),
            value_defs: value_defs.into_iter().collect(),
            kind: MemoryAccessKind::Read,
            atomicity: MemoryAtomicity::NonAtomic,
        }
    }

    /// Creates a write whose replacement comes from `value_uses`.
    #[must_use]
    pub fn write(location: L, value_uses: impl IntoIterator<Item = V>) -> Self {
        Self {
            location,
            address_uses: Vec::new(),
            value_uses: value_uses.into_iter().collect(),
            value_defs: Vec::new(),
            kind: MemoryAccessKind::Write,
            atomicity: MemoryAtomicity::NonAtomic,
        }
    }

    /// Creates a read/modify/write whose replacement comes from `value_uses`.
    ///
    /// `value_defs` receive the value observed before the replacement is
    /// committed.
    #[must_use]
    pub fn read_modify_write(
        location: L,
        value_uses: impl IntoIterator<Item = V>,
        value_defs: impl IntoIterator<Item = V>,
    ) -> Self {
        Self {
            location,
            address_uses: Vec::new(),
            value_uses: value_uses.into_iter().collect(),
            value_defs: value_defs.into_iter().collect(),
            kind: MemoryAccessKind::ReadModifyWrite,
            atomicity: MemoryAtomicity::NonAtomic,
        }
    }

    /// Creates an access of `kind` whose value operands are unreported.
    ///
    /// Every stored and loaded value stays unknown, so the access acts as an
    /// opaque clobber or effect-only read of `location`. Prefer the
    /// directional constructors when the transferred values are known: only
    /// they let derived analyses connect ordinary values through memory.
    #[must_use]
    pub const fn opaque(location: L, kind: MemoryAccessKind) -> Self {
        Self {
            location,
            address_uses: Vec::new(),
            value_uses: Vec::new(),
            value_defs: Vec::new(),
            kind,
            atomicity: MemoryAtomicity::NonAtomic,
        }
    }

    /// Attaches the ordinary values that select the binding or sub-location.
    #[must_use]
    pub fn with_address_uses(mut self, address_uses: impl IntoIterator<Item = V>) -> Self {
        self.address_uses = address_uses.into_iter().collect();
        self
    }

    /// Sets whether competing observers see the access as indivisible.
    #[must_use]
    pub const fn with_atomicity(mut self, atomicity: MemoryAtomicity) -> Self {
        self.atomicity = atomicity;
        self
    }

    /// Location or conservative region accessed by this event.
    #[must_use]
    pub const fn location(&self) -> &L {
        &self.location
    }

    /// Variables whose values select the accessed binding or sub-location.
    #[must_use]
    pub fn address_uses(&self) -> &[V] {
        &self.address_uses
    }

    /// Ordinary instruction uses whose values are stored by this access.
    #[must_use]
    pub fn value_uses(&self) -> &[V] {
        &self.value_uses
    }

    /// Ordinary instruction definitions receiving values loaded by this
    /// access.
    #[must_use]
    pub fn value_defs(&self) -> &[V] {
        &self.value_defs
    }

    /// Direction of the access.
    #[must_use]
    pub const fn kind(&self) -> MemoryAccessKind {
        self.kind
    }

    /// Whether competing observers see the access as indivisible.
    #[must_use]
    pub const fn atomicity(&self) -> MemoryAtomicity {
        self.atomicity
    }

    /// Whether the access consumes the location's previous value.
    #[must_use]
    pub const fn reads(&self) -> bool {
        self.kind.reads()
    }

    /// Whether the replacement depends on the location's previous value.
    #[must_use]
    pub const fn modifies(&self) -> bool {
        self.kind.modifies()
    }

    /// Whether the access may change the location.
    #[must_use]
    pub const fn writes(&self) -> bool {
        self.kind.writes()
    }

    /// Instruction-level properties contributed by this access.
    #[must_use]
    pub const fn operations(&self) -> MemoryOperations {
        self.kind.operations()
    }
}

/// One memory-relevant event emitted by an instruction.
///
/// Fence payloads retain the consumer's ordering, scope, and execution-barrier
/// details. Generic analyses can still treat every `Fence` as an ordering
/// boundary without pretending those language-specific details are
/// interchangeable.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryEvent<L, V, F> {
    /// An access to one location or conservative region.
    Access(MemoryAccess<L, V>),
    /// A consumer-defined memory fence.
    Fence(F),
}

impl<L, V, F> MemoryEvent<L, V, F> {
    /// Instruction-level properties contributed by this event.
    ///
    /// A fence contributes no access properties; it remains visible as an
    /// explicit [`MemoryEvent::Fence`] in the event stream.
    #[must_use]
    pub const fn operations(&self) -> MemoryOperations {
        match self {
            Self::Access(access) => access.operations(),
            Self::Fence(_) => MemoryOperations::NONE,
        }
    }

    /// Whether this event reads memory.
    #[must_use]
    pub const fn reads(&self) -> bool {
        matches!(self, Self::Access(access) if access.reads())
    }

    /// Whether this event is one compound read/modify/write access.
    #[must_use]
    pub const fn modifies(&self) -> bool {
        matches!(self, Self::Access(access) if access.modifies())
    }

    /// Whether this event writes memory.
    #[must_use]
    pub const fn writes(&self) -> bool {
        matches!(self, Self::Access(access) if access.writes())
    }

    /// Whether this event is a fence rather than a location access.
    #[must_use]
    pub const fn is_fence(&self) -> bool {
        matches!(self, Self::Fence(_))
    }
}

/// Opt-in contract for instructions that expose memory accesses and fences.
///
/// Events must be returned in the instruction's semantic order. Address
/// calculation is ordinary value data flow and belongs in the instruction's
/// [`InstrInfo`] uses. Every variable that selects an
/// access's binding or sub-location must also appear in that access's
/// [`MemoryAccess::address_uses`]. Values transferred through memory use
/// [`MemoryAccess::value_uses`] and [`MemoryAccess::value_defs`]; an
/// adapter may leave either empty when it intentionally exposes only memory
/// state. The location may be a conservative region rather than a concrete
/// runtime address, but an adapter must not claim that unequal locations
/// cannot alias unless its location vocabulary proves that distinction.
pub trait MemoryEventInfo: InstrInfo {
    /// Consumer-defined location or conservative alias region.
    type Location: Clone + Ord;
    /// Consumer-defined fence details, such as ordering and visibility scope.
    type Fence: Clone + Eq;

    /// Returns this instruction's memory events in semantic order.
    fn memory_events(
        &self,
    ) -> impl Iterator<Item = MemoryEvent<Self::Location, Self::Variable, Self::Fence>>;

    /// Summarizes whether the instruction reads, modifies, and writes memory.
    ///
    /// Separate read and write events produce [`MemoryOperations::READ_WRITE`];
    /// a [`MemoryAccessKind::ReadModifyWrite`] event produces
    /// [`MemoryOperations::READ_MODIFY_WRITE`].
    fn memory_operations(&self) -> MemoryOperations {
        self.memory_events()
            .map(|event| event.operations())
            .collect()
    }
}

#[cfg(test)]
#[path = "memory/tests.rs"]
mod tests;
