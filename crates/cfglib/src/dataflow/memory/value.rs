//! Ordinary-value dependencies carried through memory SSA.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

use smallvec::SmallVec;

use crate::memory::MemoryAccess;
use crate::{BlockId, Graph, NodeId, SsaForm, SsaInstruction, SsaValue, SsaVersion, VariableId};

use super::{MemoryEventSite, MemorySsa, MemorySsaValue};

/// One node in the combined ordinary-value and memory-state dependency graph.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryValueNode<V> {
    /// One renamed ordinary SSA value.
    Value(SsaValue<V>),
    /// One renamed memory-location-class state.
    Memory(MemorySsaValue),
    /// One exact memory event between its inputs and outputs.
    Event(MemoryEventSite),
}

/// Why one dependency edge exists in a [`MemoryValueFlow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryValueEdge {
    /// Every ordinary use of an instruction may contribute to every ordinary
    /// definition it produces.
    Value {
        /// Source instruction performing the transfer.
        point: crate::ProgramPoint,
    },
    /// One incoming ordinary value flows into an ordinary SSA phi.
    ValuePhi {
        /// Block containing the phi.
        block: BlockId,
        /// Departure point supplying this operand, as in
        /// [`SsaPhi::operands`](crate::SsaPhi::operands).
        predecessor: crate::ProgramPoint,
    },
    /// An ordinary SSA value selects an event's binding or sub-location.
    Address {
        /// Event whose location the value selects.
        site: MemoryEventSite,
        /// Position in [`MemoryAccess::address_uses`](crate::MemoryAccess::address_uses).
        input_index: usize,
    },
    /// An ordinary SSA value is stored by an event.
    Store {
        /// Event receiving the stored value.
        site: MemoryEventSite,
        /// Position in [`MemoryAccess::value_uses`](crate::MemoryAccess::value_uses).
        input_index: usize,
    },
    /// A memory state is semantically read by an event.
    Read {
        /// Event consuming the memory state.
        site: MemoryEventSite,
    },
    /// An event produces a new memory state.
    Write {
        /// Event producing the memory state.
        site: MemoryEventSite,
    },
    /// A value read by an event is assigned to an ordinary SSA definition.
    Load {
        /// Event producing the loaded value.
        site: MemoryEventSite,
        /// Position in [`MemoryAccess::value_defs`](crate::MemoryAccess::value_defs).
        output_index: usize,
    },
    /// One incoming memory state flows into a memory phi.
    MemoryPhi {
        /// Block containing the phi.
        block: BlockId,
        /// Departure point supplying this operand, as in
        /// [`MemoryPhi::operands`](crate::MemoryPhi::operands).
        predecessor: crate::ProgramPoint,
    },
}

/// The role an invalid variable was declared to play in a memory event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryValueRole {
    /// Selects an accessed binding or sub-location.
    Address,
    /// Supplies a value written to memory.
    Stored,
    /// Receives a value read from memory.
    Loaded,
}

impl fmt::Display for MemoryValueRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Address => "address input",
            Self::Stored => "stored value",
            Self::Loaded => "loaded value",
        })
    }
}

/// A violated [`MemoryEventInfo`](crate::MemoryEventInfo) value-transfer
/// contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryValueFlowError<V> {
    /// The ordinary SSA form does not describe an event's source instruction.
    MissingInstruction {
        /// Event whose source point is absent.
        site: MemoryEventSite,
    },
    /// A declared event variable is absent from the required instruction side.
    MissingVariable {
        /// Event containing the invalid declaration.
        site: MemoryEventSite,
        /// Variable not found among the renamed uses or definitions.
        variable: V,
        /// Declared role that determines which side was required.
        role: MemoryValueRole,
    },
}

impl<V: fmt::Debug> fmt::Display for MemoryValueFlowError<V> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingInstruction { site } => {
                write!(
                    formatter,
                    "memory event {site:?} has no matching SSA instruction"
                )
            }
            Self::MissingVariable {
                site,
                variable,
                role,
            } => write!(
                formatter,
                "memory event {site:?} declares {variable:?} as a {role}, but the source instruction does not expose it on that side",
            ),
        }
    }
}

impl<V: fmt::Debug> core::error::Error for MemoryValueFlowError<V> {}

/// A combined dependency graph over ordinary SSA values, memory SSA states,
/// and exact memory events.
///
/// The graph makes the complete path through memory explicit:
///
/// `ordinary value -> store event -> memory state -> load event -> ordinary value`.
///
/// Address inputs enter their event through [`MemoryValueEdge::Address`], so
/// callers can distinguish location dependence from transferred data. A blind
/// write has no edge from the clobbered state; a read/modify/write has both a
/// [`Read`](MemoryValueEdge::Read) and a [`Write`](MemoryValueEdge::Write).
/// Fence events are retained as isolated event nodes because their happens-before
/// semantics remain consumer-defined.
#[derive(Debug, PartialEq, Eq)]
pub struct MemoryValueFlow<V> {
    graph: Graph<MemoryValueNode<V>, MemoryValueEdge>,
    nodes: NodeIndex<V>,
}

/// The nodes of one variable's or memory class's versions, indexed by version.
///
/// Most have a handful of versions, so the table lives inline until one grows
/// past that.
type VersionTable = SmallVec<[Option<NodeId>; 4]>;

/// Graph identity of every node, by the value it stands for.
///
/// The versions of one variable or memory class are dense, so each keeps a
/// version-indexed table instead of one ordered-map entry per renamed value.
/// Only events, which have no dense index, stay in a map.
#[derive(Debug, PartialEq, Eq)]
struct NodeIndex<V> {
    values: BTreeMap<V, VersionTable>,
    memory: Vec<VersionTable>,
    events: BTreeMap<MemoryEventSite, NodeId>,
}

impl<V: VariableId> NodeIndex<V> {
    const fn new() -> Self {
        Self {
            values: BTreeMap::new(),
            memory: Vec::new(),
            events: BTreeMap::new(),
        }
    }

    fn value(&self, value: &SsaValue<V>) -> Option<NodeId> {
        self.values
            .get(&value.variable)
            .and_then(|table| table.get(value.version))
            .copied()
            .flatten()
    }

    fn memory(&self, value: &MemorySsaValue) -> Option<NodeId> {
        self.memory
            .get(value.variable.index())
            .and_then(|table| table.get(value.version))
            .copied()
            .flatten()
    }
}

/// The node of `version` in `table`, made by `create` when it has none yet.
fn version_node(
    table: &mut VersionTable,
    version: SsaVersion,
    create: impl FnOnce() -> NodeId,
) -> NodeId {
    if table.len() <= version {
        table.resize(version + 1, None);
    }
    *table[version].get_or_insert_with(create)
}

/// Every buffer [`MemoryValueFlow::compute_in`] fills, owned by the caller.
///
/// The graph is the answer and its storage has to be allocated. What does not
/// is the resolution of one event's declared address, stored, and loaded
/// variables to the SSA values its source instruction gives them: three
/// vectors per event. On a procedure with a few thousand memory events that
/// is most of what the call allocates.
///
/// This holds those three buffers, refilled per event and never released.
///
/// # One scratch, many procedures
///
/// The type parameter is the ordinary SSA variable, so one scratch serves
/// every form over that vocabulary whatever its size, and it is [`Send`] when
/// the variable is.
///
/// # Examples
///
/// ```
/// use cfglib::{
///     Cfg, DominatorTree, ExactMemoryAlias, MemorySsa, MemoryValueFlow, MemoryValueFlowScratch,
///     SsaForm,
/// };
/// # use cfglib::{InstrInfo, MemoryAccess, MemoryEvent, MemoryEventInfo};
/// # #[derive(Clone)]
/// # struct Inst(Vec<u8>, Vec<MemoryEvent<u32, u8, ()>>);
/// # impl InstrInfo for Inst {
/// #     type Variable = u8;
/// #     fn uses(&self) -> &[u8] { &self.0 }
/// #     fn defs(&self) -> &[u8] { &[] }
/// # }
/// # impl MemoryEventInfo for Inst {
/// #     type Location = u32;
/// #     type Fence = ();
/// #     fn memory_events(&self) -> impl Iterator<Item = MemoryEvent<u32, u8, ()>> {
/// #         self.1.iter().cloned()
/// #     }
/// # }
/// let mut cfg = Cfg::<Inst>::new();
/// cfg.block_mut(cfg.entry()).push(Inst(
///     vec![1],
///     vec![MemoryEvent::Access(MemoryAccess::write(7, [1]))],
/// ));
/// let dominators = DominatorTree::compute(&cfg);
/// let ssa = SsaForm::compute(&cfg, &dominators);
/// let memory: MemorySsa<u32, u8, ()> = MemorySsa::compute(&cfg, &ExactMemoryAlias);
///
/// let mut scratch = MemoryValueFlowScratch::new();
/// let flow = MemoryValueFlow::compute_in(&mut scratch, &memory, &ssa).unwrap();
/// assert_eq!(flow, MemoryValueFlow::compute(&memory, &ssa).unwrap());
/// ```
#[derive(Debug)]
pub struct MemoryValueFlowScratch<V> {
    /// The SSA values selecting one event's binding or sub-location.
    addresses: Vec<SsaValue<V>>,
    /// The SSA values one event stores.
    stored: Vec<SsaValue<V>>,
    /// The SSA values one event loads.
    loaded: Vec<SsaValue<V>>,
}

impl<V> Default for MemoryValueFlowScratch<V> {
    fn default() -> Self {
        Self {
            addresses: Vec::new(),
            stored: Vec::new(),
            loaded: Vec::new(),
        }
    }
}

impl<V> MemoryValueFlowScratch<V> {
    /// Scratch holding nothing yet.
    ///
    /// Every buffer is sized by the widest event it sees, so there is no
    /// count to state here.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<V: VariableId> MemoryValueFlowScratch<V> {
    /// Resolve one access's three declared variable lists into the buffers.
    fn resolve<L>(
        &mut self,
        site: MemoryEventSite,
        instruction: &SsaInstruction<V>,
        access: &MemoryAccess<L, V>,
    ) -> Result<(), MemoryValueFlowError<V>> {
        let used = |variable: &V| instruction.use_of(variable);
        resolve_into(
            &mut self.addresses,
            site,
            access.address_uses(),
            MemoryValueRole::Address,
            used,
        )?;
        resolve_into(
            &mut self.stored,
            site,
            access.value_uses(),
            MemoryValueRole::Stored,
            used,
        )?;
        resolve_into(
            &mut self.loaded,
            site,
            access.value_defs(),
            MemoryValueRole::Loaded,
            |variable| instruction.def_of(variable),
        )
    }
}

impl<V> MemoryValueFlow<V>
where
    V: VariableId,
{
    /// Computes ordinary-value dependencies for matching memory and ordinary
    /// SSA forms.
    ///
    /// # Errors
    ///
    /// Returns a structured error when an event names a source point absent
    /// from `ssa`, declares an address/stored variable outside the source
    /// instruction's uses, or declares a loaded variable outside its
    /// definitions.
    pub fn compute<L, F>(
        memory: &MemorySsa<L, V, F>,
        ssa: &SsaForm<V>,
    ) -> Result<Self, MemoryValueFlowError<V>>
    where
        L: Clone + Ord,
        F: Clone + Eq,
    {
        Self::compute_in(&mut MemoryValueFlowScratch::new(), memory, ssa)
    }

    /// Computes ordinary-value dependencies over caller-owned working
    /// storage.
    ///
    /// This is [`compute`](Self::compute) with the per-event resolution
    /// buffers taken out of it; see [`MemoryValueFlowScratch`]. The graph it
    /// returns is the same one, built by the same code, and it reports the
    /// same errors.
    ///
    /// # Errors
    ///
    /// The same as [`compute`](Self::compute).
    pub fn compute_in<L, F>(
        scratch: &mut MemoryValueFlowScratch<V>,
        memory: &MemorySsa<L, V, F>,
        ssa: &SsaForm<V>,
    ) -> Result<Self, MemoryValueFlowError<V>>
    where
        L: Clone + Ord,
        F: Clone + Eq,
    {
        let mut flow = Self {
            graph: Graph::new(),
            nodes: NodeIndex::new(),
        };
        flow.add_values(ssa);
        flow.add_memory_states(memory);
        flow.add_events(scratch, memory, ssa)?;
        Ok(flow)
    }

    /// Combined dependency graph in deterministic construction order.
    #[must_use]
    pub const fn graph(&self) -> &Graph<MemoryValueNode<V>, MemoryValueEdge> {
        &self.graph
    }

    /// Graph identity of an ordinary SSA value participating in memory flow.
    #[must_use]
    pub fn value_node(&self, value: &SsaValue<V>) -> Option<NodeId> {
        self.nodes.value(value)
    }

    /// Graph identity of a memory SSA state.
    #[must_use]
    pub fn memory_node(&self, value: &MemorySsaValue) -> Option<NodeId> {
        self.nodes.memory(value)
    }

    /// Graph identity of one exact memory event.
    #[must_use]
    pub fn event_node(&self, site: MemoryEventSite) -> Option<NodeId> {
        self.nodes.events.get(&site).copied()
    }

    /// Number of ordinary-value, memory-state, and event nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of dependency edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    fn intern_value(&mut self, value: &SsaValue<V>) -> NodeId {
        let table = match self.nodes.values.get_mut(&value.variable) {
            Some(table) => table,
            None => self.nodes.values.entry(value.variable.clone()).or_default(),
        };
        version_node(table, value.version, || {
            self.graph.add_node(MemoryValueNode::Value(value.clone()))
        })
    }

    fn intern_memory(&mut self, value: &MemorySsaValue) -> NodeId {
        let class = value.variable.index();
        if self.nodes.memory.len() <= class {
            self.nodes.memory.resize_with(class + 1, VersionTable::new);
        }
        version_node(&mut self.nodes.memory[class], value.version, || {
            self.graph.add_node(MemoryValueNode::Memory(value.clone()))
        })
    }

    fn intern_event(&mut self, site: MemoryEventSite) -> NodeId {
        *self
            .nodes
            .events
            .entry(site)
            .or_insert_with(|| self.graph.add_node(MemoryValueNode::Event(site)))
    }

    /// Adds every ordinary-value node and the dependency edges between them.
    ///
    /// Each value is interned once, where its edges are added, so no value is
    /// looked up again to find the node it already made.
    fn add_values(&mut self, ssa: &SsaForm<V>) {
        let mut uses: SmallVec<[NodeId; 4]> = SmallVec::new();
        let mut defs: SmallVec<[NodeId; 4]> = SmallVec::new();
        for block in ssa.blocks() {
            for phi in &block.phis {
                let target = self.intern_value(&phi.result);
                for (predecessor, operand) in phi.operands() {
                    let source = self.intern_value(operand);
                    self.graph.add_edge(
                        source,
                        target,
                        MemoryValueEdge::ValuePhi {
                            block: block.block,
                            predecessor: *predecessor,
                        },
                    );
                }
            }
            for instruction in &block.instructions {
                uses.clear();
                defs.clear();
                for value in instruction.uses() {
                    uses.push(self.intern_value(value));
                }
                for value in instruction.defs() {
                    defs.push(self.intern_value(value));
                }
                for &source in &uses {
                    for &target in &defs {
                        self.graph.add_edge(
                            source,
                            target,
                            MemoryValueEdge::Value {
                                point: instruction.point,
                            },
                        );
                    }
                }
            }
        }
    }

    /// Adds every memory-state node a phi or a class's live-in makes, and the
    /// phi edges between them.
    fn add_memory_states<L, F>(&mut self, memory: &MemorySsa<L, V, F>)
    where
        L: Clone + Ord,
        F: Clone + Eq,
    {
        for class in memory.classes() {
            self.intern_memory(&SsaValue::live_in(class.id()));
        }
        for phi in memory.phis() {
            let target = self.intern_memory(phi.result());
            for (predecessor, operand) in phi.operands() {
                let source = self.intern_memory(operand);
                self.graph.add_edge(
                    source,
                    target,
                    MemoryValueEdge::MemoryPhi {
                        block: phi.block(),
                        predecessor: *predecessor,
                    },
                );
            }
        }
    }

    /// Adds every event node, the nodes its access names, and its edges.
    ///
    /// An event's memory states are interned before its resolved values, and
    /// the values in the order their edges are added: address, stored, then
    /// loaded.
    fn add_events<L, F>(
        &mut self,
        scratch: &mut MemoryValueFlowScratch<V>,
        memory: &MemorySsa<L, V, F>,
        ssa: &SsaForm<V>,
    ) -> Result<(), MemoryValueFlowError<V>>
    where
        L: Clone + Ord,
        F: Clone + Eq,
    {
        for event in memory.events() {
            let site = event.site();
            let event_node = self.intern_event(site);
            let read = event.read_version().map(|value| self.intern_memory(value));
            let written = event
                .written_version()
                .map(|value| self.intern_memory(value));
            if let Some(value) = event.clobbered_version() {
                self.intern_memory(value);
            }
            let instruction = ssa
                .instruction(site.point())
                .ok_or(MemoryValueFlowError::MissingInstruction { site })?;
            let Some(access) = event.access() else {
                continue;
            };
            scratch.resolve(site, instruction, access)?;
            for (input_index, value) in scratch.addresses.iter().enumerate() {
                let source = self.intern_value(value);
                self.graph.add_edge(
                    source,
                    event_node,
                    MemoryValueEdge::Address { site, input_index },
                );
            }
            for (input_index, value) in scratch.stored.iter().enumerate() {
                let source = self.intern_value(value);
                self.graph.add_edge(
                    source,
                    event_node,
                    MemoryValueEdge::Store { site, input_index },
                );
            }
            if let Some(source) = read {
                self.graph
                    .add_edge(source, event_node, MemoryValueEdge::Read { site });
            }
            if let Some(target) = written {
                self.graph
                    .add_edge(event_node, target, MemoryValueEdge::Write { site });
            }
            for (output_index, value) in scratch.loaded.iter().enumerate() {
                let target = self.intern_value(value);
                self.graph.add_edge(
                    event_node,
                    target,
                    MemoryValueEdge::Load { site, output_index },
                );
            }
        }
        Ok(())
    }
}

/// Resolve declared variables into `buffer`, replacing its contents.
///
/// `lookup` names each variable's value on the side of the instruction its
/// `role` requires: [`SsaInstruction::use_of`] for operands the event reads,
/// [`SsaInstruction::def_of`] for the definitions it loads into.
fn resolve_into<'a, V: VariableId + 'a>(
    buffer: &mut Vec<SsaValue<V>>,
    site: MemoryEventSite,
    variables: &[V],
    role: MemoryValueRole,
    lookup: impl Fn(&V) -> Option<&'a SsaValue<V>>,
) -> Result<(), MemoryValueFlowError<V>> {
    buffer.clear();
    for variable in variables {
        let value = lookup(variable).ok_or_else(|| MemoryValueFlowError::MissingVariable {
            site,
            variable: variable.clone(),
            role,
        })?;
        buffer.push(value.clone());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
