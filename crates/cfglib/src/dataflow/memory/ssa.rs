//! Location-aware memory SSA and def-use tracing.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use crate::{BlockId, Cfg, DominatorTree, InstrInfo, ProgramPoint, SsaForm, SsaValue, VariableId};

use crate::memory::{
    MemoryAccess, MemoryAccessKind, MemoryAtomicity, MemoryEvent, MemoryEventInfo, MemoryOperations,
};

use shadow::{ShadowInstruction, build_location_classes, build_shadow_cfg};

/// Caller-defined may-alias relation for memory locations.
///
/// Returning `false` promises that the two locations cannot refer to any
/// overlapping runtime storage. Returning `true` is always safe and may only
/// reduce precision. The analysis symmetrizes the relation and takes its
/// transitive closure, so a non-transitive oracle remains conservative.
pub trait MemoryAlias<L> {
    /// Whether two reported locations may overlap at runtime.
    fn may_alias(&self, left: &L, right: &L) -> bool;
}

impl<L, F> MemoryAlias<L> for F
where
    F: Fn(&L, &L) -> bool,
{
    fn may_alias(&self, left: &L, right: &L) -> bool {
        self(left, right)
    }
}

/// Alias oracle that treats only equal locations as overlapping.
///
/// Use this only when the location vocabulary proves that unequal identities
/// denote disjoint storage.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ExactMemoryAlias;

impl<L: Eq> MemoryAlias<L> for ExactMemoryAlias {
    fn may_alias(&self, left: &L, right: &L) -> bool {
        left == right
    }
}

/// Alias oracle that places every reported location in one conservative class.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ConservativeMemoryAlias;

impl<L> MemoryAlias<L> for ConservativeMemoryAlias {
    fn may_alias(&self, _left: &L, _right: &L) -> bool {
        true
    }
}

/// Whether two structured index paths may select overlapping sub-locations.
///
/// The building block for alias oracles over indexed storage
/// (`base[i][j]`-shaped locations): `fixed` projects each path component to
/// its compile-time value when one is known. Paths of unequal arity are
/// conservatively overlapping; a component whose value is unknown matches
/// anything; two paths whose known components ever disagree provably select
/// disjoint storage.
///
/// Returning `false` is a proof of disjointness under the caller's promise
/// that equal-arity paths address the same axes of the same object — the
/// storage-classification half of the oracle stays consumer-owned.
pub fn index_paths_may_overlap<T, C: PartialEq>(
    left: &[T],
    right: &[T],
    mut fixed: impl FnMut(&T) -> Option<C>,
) -> bool {
    if left.len() != right.len() {
        return true;
    }
    left.iter()
        .zip(right)
        .all(|(left, right)| match (fixed(left), fixed(right)) {
            (Some(left), Some(right)) => left == right,
            _ => true,
        })
}

/// Stable identity of one transitive may-alias class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryClassId(usize);

impl MemoryClassId {
    /// Zero-based deterministic class index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// One location class used as a memory-SSA variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryLocationClass<L> {
    id: MemoryClassId,
    locations: Vec<L>,
}

impl<L> MemoryLocationClass<L> {
    /// Stable class identity.
    #[must_use]
    pub const fn id(&self) -> MemoryClassId {
        self.id
    }

    /// Sorted reported locations conservatively merged into this class.
    #[must_use]
    pub fn locations(&self) -> &[L] {
        &self.locations
    }
}

/// Identity of one event within one source instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryEventSite {
    point: ProgramPoint,
    event_index: usize,
}

impl MemoryEventSite {
    /// Creates an event identity from its instruction and local event index.
    #[must_use]
    pub const fn new(point: ProgramPoint, event_index: usize) -> Self {
        Self { point, event_index }
    }

    /// Source instruction containing this event.
    #[must_use]
    pub const fn point(self) -> ProgramPoint {
        self.point
    }

    /// Zero-based semantic position within the source instruction.
    #[must_use]
    pub const fn event_index(self) -> usize {
        self.event_index
    }
}

/// One location-class value in memory SSA.
pub type MemorySsaValue = SsaValue<MemoryClassId>;

/// The producer of one memory-SSA value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryDefinition {
    /// Memory state entering a dominator-tree root.
    LiveIn {
        /// Location class whose initial state enters the CFG component.
        class: MemoryClassId,
    },
    /// A write or read/modify/write event.
    Event {
        /// Event producing the new memory state.
        site: MemoryEventSite,
    },
    /// A merge of predecessor memory states.
    Phi {
        /// Block containing the merge.
        block: BlockId,
        /// Location class being merged.
        class: MemoryClassId,
    },
}

/// One semantic consumer of a memory-SSA value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryUse {
    /// A read or read/modify/write event.
    Event {
        /// Event consuming the memory state.
        site: MemoryEventSite,
    },
    /// One predecessor operand of a memory phi.
    Phi {
        /// Block containing the phi.
        block: BlockId,
        /// Predecessor supplying this operand.
        predecessor: BlockId,
        /// Memory state produced by the phi.
        result: MemorySsaValue,
    },
}

/// A fully renamed location-class phi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPhi {
    block: BlockId,
    result: MemorySsaValue,
    operands: Vec<(BlockId, MemorySsaValue)>,
}

impl MemoryPhi {
    /// Block containing this phi.
    #[must_use]
    pub const fn block(&self) -> BlockId {
        self.block
    }

    /// Memory state produced by this phi.
    #[must_use]
    pub const fn result(&self) -> &MemorySsaValue {
        &self.result
    }

    /// Incoming state for every CFG predecessor, in predecessor order.
    #[must_use]
    pub fn operands(&self) -> &[(BlockId, MemorySsaValue)] {
        &self.operands
    }
}

/// One memory event annotated with its alias class and SSA states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySSAEvent<L, V, F> {
    site: MemoryEventSite,
    event: MemoryEvent<L, V, F>,
    class: Option<MemoryClassId>,
    read_version: Option<MemorySsaValue>,
    written_version: Option<MemorySsaValue>,
    clobbered_version: Option<MemorySsaValue>,
}

impl<L, V, F> MemorySSAEvent<L, V, F> {
    /// Source identity of this event.
    #[must_use]
    pub const fn site(&self) -> MemoryEventSite {
        self.site
    }

    /// Original typed memory event.
    #[must_use]
    pub const fn event(&self) -> &MemoryEvent<L, V, F> {
        &self.event
    }

    /// Location access, or `None` for a fence.
    #[must_use]
    pub const fn access(&self) -> Option<&MemoryAccess<L, V>> {
        match &self.event {
            MemoryEvent::Access(access) => Some(access),
            MemoryEvent::Fence(_) => None,
        }
    }

    /// Transitive may-alias class, or `None` for a fence.
    #[must_use]
    pub const fn class(&self) -> Option<MemoryClassId> {
        self.class
    }

    /// Memory state consumed by a read or read/modify/write.
    #[must_use]
    pub const fn read_version(&self) -> Option<&MemorySsaValue> {
        self.read_version.as_ref()
    }

    /// New memory state produced by a write or read/modify/write.
    #[must_use]
    pub const fn written_version(&self) -> Option<&MemorySsaValue> {
        self.written_version.as_ref()
    }

    /// Prior memory state replaced by a write or read/modify/write.
    ///
    /// A blind write structurally clobbers this state but does not semantically
    /// read it; it therefore does not appear in [`MemorySSA::users`].
    #[must_use]
    pub const fn clobbered_version(&self) -> Option<&MemorySsaValue> {
        self.clobbered_version.as_ref()
    }

    /// Exact reported location, or `None` for a fence.
    #[must_use]
    pub const fn location(&self) -> Option<&L> {
        match &self.event {
            MemoryEvent::Access(access) => Some(access.location()),
            MemoryEvent::Fence(_) => None,
        }
    }

    /// Native variables used to select the binding or sub-location.
    #[must_use]
    pub fn address_uses(&self) -> Option<&[V]> {
        self.access().map(MemoryAccess::address_uses)
    }

    /// Read, modify, and write properties contributed by this event.
    #[must_use]
    pub const fn operations(&self) -> MemoryOperations {
        self.event.operations()
    }

    /// Access direction, or `None` for a fence.
    #[must_use]
    pub const fn kind(&self) -> Option<MemoryAccessKind> {
        match &self.event {
            MemoryEvent::Access(access) => Some(access.kind()),
            MemoryEvent::Fence(_) => None,
        }
    }

    /// Access atomicity, or `None` for a fence.
    #[must_use]
    pub const fn atomicity(&self) -> Option<MemoryAtomicity> {
        match &self.event {
            MemoryEvent::Access(access) => Some(access.atomicity()),
            MemoryEvent::Fence(_) => None,
        }
    }

    /// Consumer-defined fence details, or `None` for an access.
    #[must_use]
    pub const fn fence(&self) -> Option<&F> {
        match &self.event {
            MemoryEvent::Access(_) => None,
            MemoryEvent::Fence(fence) => Some(fence),
        }
    }

    /// Whether this event semantically reads memory.
    #[must_use]
    pub const fn reads(&self) -> bool {
        self.event.reads()
    }

    /// Whether this event is a compound read/modify/write.
    #[must_use]
    pub const fn modifies(&self) -> bool {
        self.event.modifies()
    }

    /// Whether this event semantically writes memory.
    #[must_use]
    pub const fn writes(&self) -> bool {
        self.event.writes()
    }

    /// Whether this event is a fence rather than an access.
    #[must_use]
    pub const fn is_fence(&self) -> bool {
        self.event.is_fence()
    }

    /// Resolves the access's address inputs to their reaching SSA versions.
    ///
    /// `ssa` must describe the same source CFG. Returns `None` for a fence, a
    /// mismatched form, or a violated [`MemoryEventInfo`] address-use contract.
    #[must_use]
    pub fn ssa_address_uses(&self, ssa: &SsaForm<V>) -> Option<Vec<SsaValue<V>>>
    where
        V: VariableId,
    {
        let access = self.access()?;
        let instruction = ssa.instruction(self.site.point)?;
        access
            .address_uses()
            .iter()
            .map(|address| {
                instruction
                    .uses
                    .iter()
                    .find(|value| &value.variable == address)
                    .cloned()
            })
            .collect()
    }
}

/// Location-aware memory SSA plus bidirectional def-use queries.
///
/// The analysis first merges reported locations according to a
/// [`MemoryAlias`] oracle. Each resulting class becomes a variable in a shadow
/// CFG, allowing the ordinary SSA constructor to provide loop-correct phi
/// placement and renaming. A read consumes the current class value, a write
/// defines a new one, and a read/modify/write does both.
///
/// Events retain their exact reported locations and native address inputs;
/// [`MemorySSAEvent::ssa_address_uses`] connects those inputs to ordinary SSA.
/// Location precision is therefore defined by the adapter. A whole object or
/// region remains sound but produces correspondingly conservative def-use
/// chains.
///
/// Fence events remain ordered and queryable but do not define memory values:
/// their cross-thread happens-before meaning is consumer-specific. As with
/// [`SsaForm::compute`], the CFG entry must not be a branch target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySSA<L, V, F> {
    classes: Vec<MemoryLocationClass<L>>,
    class_by_location: BTreeMap<L, MemoryClassId>,
    events: Vec<MemorySSAEvent<L, V, F>>,
    event_by_site: BTreeMap<MemoryEventSite, usize>,
    events_by_point: BTreeMap<ProgramPoint, Range<usize>>,
    phis: Vec<MemoryPhi>,
    phi_by_result: BTreeMap<MemorySsaValue, usize>,
    definitions: BTreeMap<MemorySsaValue, MemoryDefinition>,
    users: BTreeMap<MemorySsaValue, Vec<MemoryUse>>,
}

impl<L, V, F> MemorySSA<L, V, F>
where
    L: Clone + Ord,
    V: VariableId,
    F: Clone + Eq,
{
    /// Computes location-aware memory SSA for `cfg` using `alias`.
    #[must_use]
    pub fn compute<I, E, A>(cfg: &Cfg<I, E>, alias: &A) -> Self
    where
        I: MemoryEventInfo<Location = L, Fence = F> + InstrInfo<Variable = V>,
        A: MemoryAlias<L> + ?Sized,
    {
        Self::compute_in(&mut MemorySsaScratch::new(), cfg, alias)
    }

    /// Computes location-aware memory SSA over caller-owned working storage.
    ///
    /// This is [`compute`](Self::compute) with the call's buffers taken out of
    /// it: the event trace, the alias merge, and the dominator and SSA
    /// buffers of the shadow CFG the analysis runs over. The answer is the
    /// same one, built by the same code — the allocating entry point is this
    /// function over a fresh [`MemorySsaScratch`].
    #[must_use]
    pub fn compute_in<I, E, A>(
        scratch: &mut MemorySsaScratch<L, V, F>,
        cfg: &Cfg<I, E>,
        alias: &A,
    ) -> Self
    where
        I: MemoryEventInfo<Location = L, Fence = F> + InstrInfo<Variable = V>,
        A: MemoryAlias<L> + ?Sized,
    {
        scratch.reset();
        scratch.trace.rebuild(cfg);
        let (classes, class_by_location) = build_location_classes(scratch, alias);
        let shadow = build_shadow_cfg(cfg, scratch.trace.entries(), &class_by_location);
        let dominators = DominatorTree::compute_in(&mut scratch.dominators, &shadow);
        let ssa = SsaForm::compute_in(&mut scratch.ssa, &shadow, &dominators);
        Self::from_shadow_ssa(classes, class_by_location, &shadow, &ssa)
    }

    /// Alias classes in deterministic identity order.
    #[must_use]
    pub fn classes(&self) -> &[MemoryLocationClass<L>] {
        &self.classes
    }

    /// Returns one alias class by identity.
    #[must_use]
    pub fn class(&self, id: MemoryClassId) -> Option<&MemoryLocationClass<L>> {
        self.classes.get(id.index())
    }

    /// Returns the class assigned to an exact reported location.
    #[must_use]
    pub fn class_of(&self, location: &L) -> Option<MemoryClassId> {
        self.class_by_location.get(location).copied()
    }

    /// Whether two reported locations belong to the same may-alias class.
    ///
    /// Returns `false` when either location did not occur in the analyzed CFG.
    #[must_use]
    pub fn may_alias(&self, left: &L, right: &L) -> bool {
        self.class_of(left)
            .zip(self.class_of(right))
            .is_some_and(|(left, right)| left == right)
    }

    /// Every event in stable CFG, instruction, and semantic-event order.
    #[must_use]
    pub fn events(&self) -> &[MemorySSAEvent<L, V, F>] {
        &self.events
    }

    /// Events emitted by one source instruction.
    #[must_use]
    pub fn events_at(&self, point: ProgramPoint) -> &[MemorySSAEvent<L, V, F>] {
        self.events_by_point
            .get(&point)
            .map_or(&[], |range| &self.events[range.clone()])
    }

    /// Events in the observed may-alias class containing `location`.
    pub fn events_for<'a>(
        &'a self,
        location: &'a L,
    ) -> impl Iterator<Item = &'a MemorySSAEvent<L, V, F>> + 'a {
        let class = self.class_of(location);
        self.events
            .iter()
            .filter(move |event| class.is_some() && event.class == class)
    }

    /// Summarizes all events emitted by one source instruction.
    #[must_use]
    pub fn operations_at(&self, point: ProgramPoint) -> MemoryOperations {
        self.events_at(point)
            .iter()
            .fold(MemoryOperations::NONE, |operations, event| {
                operations.union(event.operations())
            })
    }

    /// Looks up one event by its source identity.
    #[must_use]
    pub fn event(&self, site: MemoryEventSite) -> Option<&MemorySSAEvent<L, V, F>> {
        self.event_by_site
            .get(&site)
            .map(|&index| &self.events[index])
    }

    /// Fully renamed memory phis in CFG block order.
    #[must_use]
    pub fn phis(&self) -> &[MemoryPhi] {
        &self.phis
    }

    /// Producer of `value`, including live-ins and phis.
    #[must_use]
    pub fn definition(&self, value: &MemorySsaValue) -> Option<&MemoryDefinition> {
        self.definitions.get(value)
    }

    /// Semantic readers and phi operands consuming `value`.
    ///
    /// A blind write's structural clobber does not appear here.
    #[must_use]
    pub fn users(&self, value: &MemorySsaValue) -> &[MemoryUse] {
        self.users.get(value).map_or(&[], Vec::as_slice)
    }

    /// Immediate producer of the state read at `site`.
    #[must_use]
    pub fn reaching_definition(&self, site: MemoryEventSite) -> Option<&MemoryDefinition> {
        let version = self.event(site)?.read_version()?;
        self.definition(version)
    }

    /// Immediate producer of the state structurally replaced at `site`.
    #[must_use]
    pub fn clobbered_definition(&self, site: MemoryEventSite) -> Option<&MemoryDefinition> {
        let version = self.event(site)?.clobbered_version()?;
        self.definition(version)
    }

    /// Concrete writes and live-ins that can reach the read at `site`.
    ///
    /// Phi chains are expanded transitively and cycles are visited once.
    #[must_use]
    pub fn reaching_definitions(&self, site: MemoryEventSite) -> Vec<&MemoryDefinition> {
        let Some(start) = self.event(site).and_then(MemorySSAEvent::read_version) else {
            return Vec::new();
        };
        let mut pending = vec![start.clone()];
        let mut visited = BTreeSet::new();
        let mut result = Vec::new();

        while let Some(value) = pending.pop() {
            if !visited.insert(value.clone()) {
                continue;
            }
            let Some(definition) = self.definition(&value) else {
                continue;
            };
            if matches!(definition, MemoryDefinition::Phi { .. }) {
                let Some(&phi_index) = self.phi_by_result.get(&value) else {
                    continue;
                };
                pending.extend(
                    self.phis[phi_index]
                        .operands
                        .iter()
                        .rev()
                        .map(|(_, operand)| operand.clone()),
                );
            } else {
                result.push(definition);
            }
        }
        result
    }

    /// Reads transitively data-dependent on the write at `site`.
    ///
    /// Traversal crosses phis and read/modify/write results. Returned sites are
    /// sorted and deduplicated.
    #[must_use]
    pub fn transitive_readers(&self, site: MemoryEventSite) -> Vec<MemoryEventSite> {
        let Some(start) = self.event(site).and_then(MemorySSAEvent::written_version) else {
            return Vec::new();
        };
        let mut pending = vec![start.clone()];
        let mut visited_values = BTreeSet::new();
        let mut readers = BTreeSet::new();

        while let Some(value) = pending.pop() {
            if !visited_values.insert(value.clone()) {
                continue;
            }
            for memory_use in self.users(&value) {
                match memory_use {
                    MemoryUse::Event { site: reader } => {
                        readers.insert(*reader);
                        if let Some(written) = self
                            .event(*reader)
                            .and_then(MemorySSAEvent::written_version)
                        {
                            pending.push(written.clone());
                        }
                    }
                    MemoryUse::Phi { result, .. } => pending.push(result.clone()),
                }
            }
        }
        readers.into_iter().collect()
    }

    /// Events that semantically read memory.
    pub fn reads(&self) -> impl Iterator<Item = &MemorySSAEvent<L, V, F>> {
        self.events.iter().filter(|entry| entry.reads())
    }

    /// Events that semantically write memory.
    pub fn writes(&self) -> impl Iterator<Item = &MemorySSAEvent<L, V, F>> {
        self.events.iter().filter(|entry| entry.writes())
    }

    /// Compound read/modify/write events.
    pub fn modifies(&self) -> impl Iterator<Item = &MemorySSAEvent<L, V, F>> {
        self.events.iter().filter(|entry| entry.modifies())
    }

    /// Fence events in stable structural order.
    pub fn fences(&self) -> impl Iterator<Item = &MemorySSAEvent<L, V, F>> {
        self.events.iter().filter(|entry| entry.is_fence())
    }

    fn from_shadow_ssa(
        classes: Vec<MemoryLocationClass<L>>,
        class_by_location: BTreeMap<L, MemoryClassId>,
        shadow: &Cfg<ShadowInstruction<L, V, F>>,
        ssa: &SsaForm<MemoryClassId>,
    ) -> Self {
        let mut phis = Vec::new();
        let mut phi_by_result = BTreeMap::new();
        let mut definitions = BTreeMap::new();
        let mut users: BTreeMap<MemorySsaValue, Vec<MemoryUse>> = BTreeMap::new();

        for class in &classes {
            definitions.insert(
                SsaValue::live_in(class.id),
                MemoryDefinition::LiveIn { class: class.id },
            );
        }
        for block in ssa.blocks() {
            for phi in &block.phis {
                let phi_index = phis.len();
                phi_by_result.insert(phi.result.clone(), phi_index);
                definitions.insert(
                    phi.result.clone(),
                    MemoryDefinition::Phi {
                        block: block.block,
                        class: phi.result.variable,
                    },
                );
                for (predecessor, operand) in &phi.operands {
                    users
                        .entry(operand.clone())
                        .or_default()
                        .push(MemoryUse::Phi {
                            block: block.block,
                            predecessor: *predecessor,
                            result: phi.result.clone(),
                        });
                }
                phis.push(MemoryPhi {
                    block: block.block,
                    result: phi.result.clone(),
                    operands: phi.operands.clone(),
                });
            }
        }

        let mut events = Vec::new();
        let mut event_by_site = BTreeMap::new();
        let mut events_by_point: BTreeMap<ProgramPoint, Range<usize>> = BTreeMap::new();
        for block in ssa.blocks() {
            let shadow_instructions = shadow.block(block.block).instructions();
            for (instruction_index, renamed) in block.instructions.iter().enumerate() {
                let source = &shadow_instructions[instruction_index];
                let input = renamed.uses.first().cloned();
                let output = renamed.defs.first().cloned();
                let reads = source.event.reads();
                let writes = source.event.writes();
                let event = MemorySSAEvent {
                    site: source.site,
                    event: source.event.clone(),
                    class: source.class,
                    read_version: reads.then(|| {
                        input
                            .clone()
                            .expect("every memory read must consume its shadow class")
                    }),
                    written_version: writes
                        .then(|| output.expect("every memory write must define its shadow class")),
                    clobbered_version: writes
                        .then(|| input.expect("every memory write must replace its shadow class")),
                };
                if let Some(version) = &event.read_version {
                    users
                        .entry(version.clone())
                        .or_default()
                        .push(MemoryUse::Event { site: event.site });
                }
                if let Some(version) = &event.written_version {
                    definitions.insert(
                        version.clone(),
                        MemoryDefinition::Event { site: event.site },
                    );
                }

                let event_index = events.len();
                event_by_site.insert(event.site, event_index);
                events_by_point
                    .entry(event.site.point)
                    .and_modify(|range| range.end = event_index + 1)
                    .or_insert(event_index..event_index + 1);
                events.push(event);
            }
        }

        Self {
            classes,
            class_by_location,
            events,
            event_by_site,
            events_by_point,
            phis,
            phi_by_result,
            definitions,
            users,
        }
    }
}

mod scratch;
mod shadow;

pub use scratch::MemorySsaScratch;

#[cfg(test)]
mod tests;
