//! Ordinary-value dependencies carried through memory SSA.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

use crate::memory::MemoryAccess;
use crate::{BlockId, Graph, NodeId, SsaForm, SsaValue, VariableId};

use super::{MemoryEventSite, MemorySSA, MemorySsaValue};

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
    /// One predecessor ordinary value flows into an ordinary SSA phi.
    ValuePhi {
        /// Block containing the phi.
        block: BlockId,
        /// Predecessor supplying this operand.
        predecessor: BlockId,
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
    /// One predecessor memory state flows into a memory phi.
    MemoryPhi {
        /// Block containing the phi.
        block: BlockId,
        /// Predecessor supplying this operand.
        predecessor: BlockId,
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
    nodes: BTreeMap<MemoryValueNode<V>, NodeId>,
}

/// Every buffer [`MemoryValueFlow::compute_in`] fills, owned by the caller.
///
/// The graph is the answer and its storage has to be allocated. What does not
/// is the resolution of one event's declared address, stored, and loaded
/// variables to the SSA values its source instruction gives them: three
/// vectors per event, and the walk visits every event twice, once to intern
/// its nodes and once to add its edges. On a procedure with a few thousand
/// memory events that is most of what the call allocates.
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
///     Cfg, DominatorTree, ExactMemoryAlias, MemorySSA, MemoryValueFlow, MemoryValueFlowScratch,
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
/// let memory: MemorySSA<u32, u8, ()> = MemorySSA::compute(&cfg, &ExactMemoryAlias);
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
        instruction: &crate::SsaInstruction<V>,
        access: &MemoryAccess<L, V>,
    ) -> Result<(), MemoryValueFlowError<V>> {
        resolve_uses(
            &mut self.addresses,
            site,
            instruction,
            access.address_uses(),
            MemoryValueRole::Address,
        )?;
        resolve_uses(
            &mut self.stored,
            site,
            instruction,
            access.value_uses(),
            MemoryValueRole::Stored,
        )?;
        resolve_defs(&mut self.loaded, site, instruction, access.value_defs())
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
        memory: &MemorySSA<L, V, F>,
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
        memory: &MemorySSA<L, V, F>,
        ssa: &SsaForm<V>,
    ) -> Result<Self, MemoryValueFlowError<V>>
    where
        L: Clone + Ord,
        F: Clone + Eq,
    {
        let mut flow = Self {
            graph: Graph::new(),
            nodes: BTreeMap::new(),
        };
        flow.add_nodes(scratch, memory, ssa)?;
        flow.add_value_edges(ssa);
        flow.add_phi_edges(memory);
        flow.add_event_edges(scratch, memory, ssa)?;
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
        self.node(&MemoryValueNode::Value(value.clone()))
    }

    /// Graph identity of a memory SSA state.
    #[must_use]
    pub fn memory_node(&self, value: &MemorySsaValue) -> Option<NodeId> {
        self.node(&MemoryValueNode::Memory(value.clone()))
    }

    /// Graph identity of one exact memory event.
    #[must_use]
    pub fn event_node(&self, site: MemoryEventSite) -> Option<NodeId> {
        self.node(&MemoryValueNode::Event(site))
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

    fn node(&self, node: &MemoryValueNode<V>) -> Option<NodeId> {
        self.nodes.get(node).copied()
    }

    fn intern(&mut self, node: MemoryValueNode<V>) -> NodeId {
        if let Some(&id) = self.nodes.get(&node) {
            return id;
        }
        let id = self.graph.add_node(node.clone());
        self.nodes.insert(node, id);
        id
    }

    fn add_nodes<L, F>(
        &mut self,
        scratch: &mut MemoryValueFlowScratch<V>,
        memory: &MemorySSA<L, V, F>,
        ssa: &SsaForm<V>,
    ) -> Result<(), MemoryValueFlowError<V>>
    where
        L: Clone + Ord,
        F: Clone + Eq,
    {
        for block in ssa.blocks() {
            for phi in &block.phis {
                self.intern(MemoryValueNode::Value(phi.result.clone()));
                for (_, operand) in &phi.operands {
                    self.intern(MemoryValueNode::Value(operand.clone()));
                }
            }
            for instruction in &block.instructions {
                for value in instruction.uses.iter().chain(&instruction.defs) {
                    self.intern(MemoryValueNode::Value(value.clone()));
                }
            }
        }
        for class in memory.classes() {
            self.intern(MemoryValueNode::Memory(SsaValue::live_in(class.id())));
        }
        for phi in memory.phis() {
            self.intern(MemoryValueNode::Memory(phi.result().clone()));
            for (_, operand) in phi.operands() {
                self.intern(MemoryValueNode::Memory(operand.clone()));
            }
        }
        for event in memory.events() {
            self.intern(MemoryValueNode::Event(event.site()));
            if let Some(value) = event.read_version() {
                self.intern(MemoryValueNode::Memory(value.clone()));
            }
            if let Some(value) = event.written_version() {
                self.intern(MemoryValueNode::Memory(value.clone()));
            }
            if let Some(value) = event.clobbered_version() {
                self.intern(MemoryValueNode::Memory(value.clone()));
            }
            let instruction = ssa
                .instruction(event.site().point())
                .ok_or(MemoryValueFlowError::MissingInstruction { site: event.site() })?;
            let Some(access) = event.access() else {
                continue;
            };
            scratch.resolve(event.site(), instruction, access)?;
            for value in scratch
                .addresses
                .iter()
                .chain(&scratch.stored)
                .chain(&scratch.loaded)
            {
                self.intern(MemoryValueNode::Value(value.clone()));
            }
        }
        Ok(())
    }

    fn add_value_edges(&mut self, ssa: &SsaForm<V>) {
        for block in ssa.blocks() {
            for phi in &block.phis {
                let target = self.intern(MemoryValueNode::Value(phi.result.clone()));
                for (predecessor, operand) in &phi.operands {
                    let source = self.intern(MemoryValueNode::Value(operand.clone()));
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
                for source in &instruction.uses {
                    let source = self.intern(MemoryValueNode::Value(source.clone()));
                    for target in &instruction.defs {
                        let target = self.intern(MemoryValueNode::Value(target.clone()));
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

    fn add_phi_edges<L, F>(&mut self, memory: &MemorySSA<L, V, F>)
    where
        L: Clone + Ord,
        F: Clone + Eq,
    {
        for phi in memory.phis() {
            let target = self.intern(MemoryValueNode::Memory(phi.result().clone()));
            for (predecessor, operand) in phi.operands() {
                let source = self.intern(MemoryValueNode::Memory(operand.clone()));
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

    fn add_event_edges<L, F>(
        &mut self,
        scratch: &mut MemoryValueFlowScratch<V>,
        memory: &MemorySSA<L, V, F>,
        ssa: &SsaForm<V>,
    ) -> Result<(), MemoryValueFlowError<V>>
    where
        L: Clone + Ord,
        F: Clone + Eq,
    {
        for event in memory.events() {
            let site = event.site();
            let event_node = self.intern(MemoryValueNode::Event(site));
            let instruction = ssa
                .instruction(site.point())
                .ok_or(MemoryValueFlowError::MissingInstruction { site })?;
            let Some(access) = event.access() else {
                continue;
            };
            scratch.resolve(site, instruction, access)?;
            for input_index in 0..scratch.addresses.len() {
                let value = scratch.addresses[input_index].clone();
                let source = self.intern(MemoryValueNode::Value(value));
                self.graph.add_edge(
                    source,
                    event_node,
                    MemoryValueEdge::Address { site, input_index },
                );
            }
            for input_index in 0..scratch.stored.len() {
                let value = scratch.stored[input_index].clone();
                let source = self.intern(MemoryValueNode::Value(value));
                self.graph.add_edge(
                    source,
                    event_node,
                    MemoryValueEdge::Store { site, input_index },
                );
            }
            if let Some(value) = event.read_version() {
                let source = self.intern(MemoryValueNode::Memory(value.clone()));
                self.graph
                    .add_edge(source, event_node, MemoryValueEdge::Read { site });
            }
            if let Some(value) = event.written_version() {
                let target = self.intern(MemoryValueNode::Memory(value.clone()));
                self.graph
                    .add_edge(event_node, target, MemoryValueEdge::Write { site });
            }
            for output_index in 0..scratch.loaded.len() {
                let value = scratch.loaded[output_index].clone();
                let target = self.intern(MemoryValueNode::Value(value));
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

/// Resolve declared use variables into `buffer`, replacing its contents.
fn resolve_uses<V: VariableId>(
    buffer: &mut Vec<SsaValue<V>>,
    site: MemoryEventSite,
    instruction: &crate::SsaInstruction<V>,
    variables: &[V],
    role: MemoryValueRole,
) -> Result<(), MemoryValueFlowError<V>> {
    buffer.clear();
    for variable in variables {
        let value = instruction
            .uses
            .iter()
            .find(|value| value.variable == *variable)
            .cloned()
            .ok_or_else(|| MemoryValueFlowError::MissingVariable {
                site,
                variable: variable.clone(),
                role,
            })?;
        buffer.push(value);
    }
    Ok(())
}

/// Resolve declared definition variables into `buffer`, replacing its
/// contents.
fn resolve_defs<V: VariableId>(
    buffer: &mut Vec<SsaValue<V>>,
    site: MemoryEventSite,
    instruction: &crate::SsaInstruction<V>,
    variables: &[V],
) -> Result<(), MemoryValueFlowError<V>> {
    buffer.clear();
    for variable in variables {
        let value = instruction
            .defs
            .iter()
            .find(|value| value.variable == *variable)
            .cloned()
            .ok_or_else(|| MemoryValueFlowError::MissingVariable {
                site,
                variable: variable.clone(),
                role: MemoryValueRole::Loaded,
            })?;
        buffer.push(value);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
