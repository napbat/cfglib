extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::{
    Cfg, DominatorTree, EdgeKind, ExactMemoryAlias, InstrInfo, MemoryAccess, MemoryEvent,
    MemoryEventInfo, MemoryEventSite, MemorySSA, ProgramPoint, SsaForm,
};

use super::{
    MemoryValueEdge, MemoryValueFlow, MemoryValueFlowError, MemoryValueFlowScratch, MemoryValueRole,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Location {
    Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fence {
    Order,
}

#[derive(Debug, Clone)]
struct Instruction {
    uses: Vec<u8>,
    defs: Vec<u8>,
    events: Vec<MemoryEvent<Location, u8, Fence>>,
}

impl Instruction {
    fn plain(uses: impl IntoIterator<Item = u8>, defs: impl IntoIterator<Item = u8>) -> Self {
        Self {
            uses: uses.into_iter().collect(),
            defs: defs.into_iter().collect(),
            events: Vec::new(),
        }
    }

    fn access(
        uses: impl IntoIterator<Item = u8>,
        defs: impl IntoIterator<Item = u8>,
        access: MemoryAccess<Location, u8>,
    ) -> Self {
        Self {
            uses: uses.into_iter().collect(),
            defs: defs.into_iter().collect(),
            events: vec![MemoryEvent::Access(access)],
        }
    }

    fn fence() -> Self {
        Self {
            uses: Vec::new(),
            defs: Vec::new(),
            events: vec![MemoryEvent::Fence(Fence::Order)],
        }
    }
}

impl InstrInfo for Instruction {
    type Variable = u8;

    fn uses(&self) -> &[Self::Variable] {
        &self.uses
    }

    fn defs(&self) -> &[Self::Variable] {
        &self.defs
    }
}

impl MemoryEventInfo for Instruction {
    type Location = Location;
    type Fence = Fence;

    fn memory_events(
        &self,
    ) -> impl Iterator<Item = MemoryEvent<Self::Location, Self::Variable, Self::Fence>> {
        self.events.iter().cloned()
    }
}

fn site(block: crate::BlockId, inst_idx: usize) -> MemoryEventSite {
    MemoryEventSite::new(ProgramPoint { block, inst_idx }, 0)
}

fn analyses(
    cfg: &Cfg<Instruction>,
) -> (
    SsaForm<u8>,
    MemorySSA<Location, u8, Fence>,
    MemoryValueFlow<u8>,
) {
    let dominators = DominatorTree::compute(cfg);
    let ssa = SsaForm::compute(cfg, &dominators);
    let memory = MemorySSA::compute(cfg, &ExactMemoryAlias);
    let flow = MemoryValueFlow::compute(&memory, &ssa).unwrap();
    (ssa, memory, flow)
}

fn has_edge(
    flow: &MemoryValueFlow<u8>,
    source: crate::NodeId,
    target: crate::NodeId,
    payload: MemoryValueEdge,
) -> bool {
    flow.graph().edges().any(|edge| {
        edge.source() == source && edge.target() == target && *edge.payload() == payload
    })
}

#[test]
fn ordinary_ssa_transfers_share_the_combined_graph() {
    let mut cfg = Cfg::new();
    let block = cfg.entry();
    cfg.block_mut(block).push(Instruction::plain([], [1]));
    cfg.block_mut(block).push(Instruction::plain([1], [2]));

    let (ssa, _, flow) = analyses(&cfg);
    let instruction = ssa
        .instruction(ProgramPoint { block, inst_idx: 1 })
        .unwrap();
    let source = flow.value_node(&instruction.uses[0]).unwrap();
    let target = flow.value_node(&instruction.defs[0]).unwrap();

    assert!(has_edge(
        &flow,
        source,
        target,
        MemoryValueEdge::Value {
            point: instruction.point,
        }
    ));
}

#[test]
fn ordinary_ssa_phis_share_the_combined_graph() {
    let mut cfg = Cfg::new();
    let left = cfg.new_block();
    let right = cfg.new_block();
    let merge = cfg.new_block();
    cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    cfg.block_mut(left).push(Instruction::plain([], [1]));
    cfg.block_mut(right).push(Instruction::plain([], [1]));
    cfg.block_mut(merge).push(Instruction::plain([1], [2]));

    let (ssa, _, flow) = analyses(&cfg);
    let phi = &ssa.block(merge).phis[0];
    let target = flow.value_node(&phi.result).unwrap();
    for (predecessor, operand) in &phi.operands {
        assert!(has_edge(
            &flow,
            flow.value_node(operand).unwrap(),
            target,
            MemoryValueEdge::ValuePhi {
                block: merge,
                predecessor: *predecessor,
            }
        ));
    }
}

#[test]
fn stored_values_reach_loaded_definitions_through_memory_state() {
    let mut cfg = Cfg::new();
    let block = cfg.entry();
    cfg.block_mut(block).push(Instruction::plain([], [1]));
    cfg.block_mut(block).push(Instruction::access(
        [1, 7],
        [],
        MemoryAccess::write(Location::Value, [1]).with_address_uses([7]),
    ));
    cfg.block_mut(block).push(Instruction::access(
        [],
        [2],
        MemoryAccess::read(Location::Value, [2]),
    ));

    let (ssa, memory, flow) = analyses(&cfg);
    let store_site = site(block, 1);
    let load_site = site(block, 2);
    let store_event = memory.event(store_site).unwrap();
    let load_event = memory.event(load_site).unwrap();
    let store_instruction = ssa.instruction(store_site.point()).unwrap();
    let load_instruction = ssa.instruction(load_site.point()).unwrap();
    let stored = store_instruction
        .uses
        .iter()
        .find(|value| value.variable == 1)
        .unwrap();
    let address = store_instruction
        .uses
        .iter()
        .find(|value| value.variable == 7)
        .unwrap();
    let loaded = load_instruction
        .defs
        .iter()
        .find(|value| value.variable == 2)
        .unwrap();
    let stored_node = flow.value_node(stored).unwrap();
    let address_node = flow.value_node(address).unwrap();
    let write_node = flow.event_node(store_site).unwrap();
    let load_node = flow.event_node(load_site).unwrap();
    let state = store_event.written_version().unwrap();
    assert_eq!(Some(state), load_event.read_version());
    let state_node = flow.memory_node(state).unwrap();
    let loaded_node = flow.value_node(loaded).unwrap();

    assert!(has_edge(
        &flow,
        stored_node,
        write_node,
        MemoryValueEdge::Store {
            site: store_site,
            input_index: 0,
        }
    ));
    assert!(has_edge(
        &flow,
        address_node,
        write_node,
        MemoryValueEdge::Address {
            site: store_site,
            input_index: 0,
        }
    ));
    assert!(has_edge(
        &flow,
        write_node,
        state_node,
        MemoryValueEdge::Write { site: store_site }
    ));
    assert!(has_edge(
        &flow,
        state_node,
        load_node,
        MemoryValueEdge::Read { site: load_site }
    ));
    assert!(has_edge(
        &flow,
        load_node,
        loaded_node,
        MemoryValueEdge::Load {
            site: load_site,
            output_index: 0,
        }
    ));
}

#[test]
fn blind_write_does_not_depend_on_the_clobbered_state() {
    let mut cfg = Cfg::new();
    let block = cfg.entry();
    cfg.block_mut(block).push(Instruction::access(
        [],
        [],
        MemoryAccess::write(Location::Value, []),
    ));

    let (_, memory, flow) = analyses(&cfg);
    let event_site = site(block, 0);
    let event = memory.event(event_site).unwrap();
    let event_node = flow.event_node(event_site).unwrap();
    let clobbered = flow
        .memory_node(event.clobbered_version().unwrap())
        .unwrap();

    assert!(!flow.graph().edges().any(|edge| {
        edge.source() == clobbered
            && edge.target() == event_node
            && matches!(edge.payload(), MemoryValueEdge::Read { .. })
    }));
}

#[test]
fn read_modify_write_connects_every_value_and_state_role() {
    let mut cfg = Cfg::new();
    let block = cfg.entry();
    cfg.block_mut(block).push(Instruction::access(
        [1, 7],
        [2],
        MemoryAccess::read_modify_write(Location::Value, [1], [2]).with_address_uses([7]),
    ));

    let (ssa, memory, flow) = analyses(&cfg);
    let event_site = site(block, 0);
    let event = memory.event(event_site).unwrap();
    let instruction = ssa.instruction(event_site.point()).unwrap();
    let event_node = flow.event_node(event_site).unwrap();
    let input_node = flow.value_node(&instruction.uses[0]).unwrap();
    let address_node = flow.value_node(&instruction.uses[1]).unwrap();
    let output_node = flow.value_node(&instruction.defs[0]).unwrap();
    let read_node = flow.memory_node(event.read_version().unwrap()).unwrap();
    let written_node = flow.memory_node(event.written_version().unwrap()).unwrap();

    assert!(has_edge(
        &flow,
        input_node,
        event_node,
        MemoryValueEdge::Store {
            site: event_site,
            input_index: 0,
        }
    ));
    assert!(has_edge(
        &flow,
        address_node,
        event_node,
        MemoryValueEdge::Address {
            site: event_site,
            input_index: 0,
        }
    ));
    assert!(has_edge(
        &flow,
        read_node,
        event_node,
        MemoryValueEdge::Read { site: event_site }
    ));
    assert!(has_edge(
        &flow,
        event_node,
        written_node,
        MemoryValueEdge::Write { site: event_site }
    ));
    assert!(has_edge(
        &flow,
        event_node,
        output_node,
        MemoryValueEdge::Load {
            site: event_site,
            output_index: 0,
        }
    ));
}

#[test]
fn merge_states_flow_through_memory_phis() {
    let mut cfg = Cfg::new();
    let left = cfg.new_block();
    let right = cfg.new_block();
    let merge = cfg.new_block();
    cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    cfg.block_mut(left).push(Instruction::access(
        [1],
        [],
        MemoryAccess::write(Location::Value, [1]),
    ));
    cfg.block_mut(right).push(Instruction::access(
        [2],
        [],
        MemoryAccess::write(Location::Value, [2]),
    ));
    cfg.block_mut(merge).push(Instruction::access(
        [],
        [3],
        MemoryAccess::read(Location::Value, [3]),
    ));

    let (_, memory, flow) = analyses(&cfg);
    let [phi] = memory.phis() else {
        panic!("the merge must contain one memory phi");
    };
    let result = flow.memory_node(phi.result()).unwrap();
    for (predecessor, operand) in phi.operands() {
        let source = flow.memory_node(operand).unwrap();
        assert!(has_edge(
            &flow,
            source,
            result,
            MemoryValueEdge::MemoryPhi {
                block: merge,
                predecessor: *predecessor,
            }
        ));
    }

    let read_site = site(merge, 0);
    assert!(has_edge(
        &flow,
        result,
        flow.event_node(read_site).unwrap(),
        MemoryValueEdge::Read { site: read_site }
    ));
}

#[test]
fn memory_value_edges_preserve_operand_order_and_repetition() {
    let mut cfg = Cfg::new();
    let block = cfg.entry();
    cfg.block_mut(block).push(Instruction::access(
        [7, 1, 2],
        [3, 4],
        MemoryAccess::read_modify_write(Location::Value, [2, 1, 2], [4, 3])
            .with_address_uses([7, 7]),
    ));

    let (ssa, _, flow) = analyses(&cfg);
    let event_site = site(block, 0);
    let event = flow.event_node(event_site).unwrap();
    let instruction = ssa.instruction(event_site.point()).unwrap();
    let used = |variable| {
        flow.value_node(
            instruction
                .uses
                .iter()
                .find(|value| value.variable == variable)
                .unwrap(),
        )
        .unwrap()
    };
    let defined = |variable| {
        flow.value_node(
            instruction
                .defs
                .iter()
                .find(|value| value.variable == variable)
                .unwrap(),
        )
        .unwrap()
    };

    for input_index in 0..2 {
        assert!(has_edge(
            &flow,
            used(7),
            event,
            MemoryValueEdge::Address {
                site: event_site,
                input_index,
            },
        ));
    }
    for (input_index, variable) in [2, 1, 2].into_iter().enumerate() {
        assert!(has_edge(
            &flow,
            used(variable),
            event,
            MemoryValueEdge::Store {
                site: event_site,
                input_index,
            },
        ));
    }
    for (output_index, variable) in [4, 3].into_iter().enumerate() {
        assert!(has_edge(
            &flow,
            event,
            defined(variable),
            MemoryValueEdge::Load {
                site: event_site,
                output_index,
            },
        ));
    }
}

#[test]
fn invalid_event_variables_report_their_semantic_role() {
    let mut cfg = Cfg::new();
    cfg.block_mut(cfg.entry()).push(Instruction::access(
        [1],
        [],
        MemoryAccess::write(Location::Value, [9]),
    ));
    let dominators = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dominators);
    let memory = MemorySSA::compute(&cfg, &ExactMemoryAlias);

    let Err(error) = MemoryValueFlow::compute(&memory, &ssa) else {
        panic!("the undeclared stored value must be rejected");
    };
    assert_eq!(
        error,
        MemoryValueFlowError::MissingVariable {
            site: site(cfg.entry(), 0),
            variable: 9,
            role: MemoryValueRole::Stored,
        }
    );
}

#[test]
fn fences_remain_explicit_without_claiming_value_dependencies() {
    let mut cfg = Cfg::new();
    cfg.block_mut(cfg.entry()).push(Instruction::fence());

    let (_, _, flow) = analyses(&cfg);
    assert!(flow.event_node(site(cfg.entry(), 0)).is_some());
    assert_eq!(flow.edge_count(), 0);
}

/// Give every block of a shared shape an address-selected load and a store,
/// so every event resolves address, stored, and loaded variables.
fn with_value_flow(mut cfg: Cfg<Instruction>) -> Cfg<Instruction> {
    for (index, block) in cfg.block_ids().collect::<Vec<_>>().into_iter().enumerate() {
        let address = u8::try_from(index % 3).expect("a small index fits in u8");
        cfg.block_mut(block).push(Instruction::plain([], [address]));
        cfg.block_mut(block).push(Instruction::access(
            [address],
            [10],
            MemoryAccess::read(Location::Value, [10]).with_address_uses([address]),
        ));
        cfg.block_mut(block).push(Instruction::access(
            [10],
            [],
            MemoryAccess::write(Location::Value, [10]),
        ));
    }
    cfg
}

#[test]
fn one_scratch_reused_down_a_sequence_computes_the_allocating_answer() {
    let sequence: Vec<_> = crate::test_util::shapes::scratch_sequence::<Instruction>()
        .into_iter()
        .map(with_value_flow)
        .collect();
    let mut scratch = MemoryValueFlowScratch::new();
    for _ in 0..2 {
        for cfg in &sequence {
            let dominators = DominatorTree::compute(cfg);
            let ssa = SsaForm::compute(cfg, &dominators);
            let memory: MemorySSA<Location, u8, Fence> = MemorySSA::compute(cfg, &ExactMemoryAlias);
            assert_eq!(
                MemoryValueFlow::compute_in(&mut scratch, &memory, &ssa),
                MemoryValueFlow::compute(&memory, &ssa),
                "{} blocks",
                cfg.block_count()
            );
        }
    }
}

#[test]
fn a_reused_scratch_reports_the_same_error_after_a_valid_procedure() {
    // The buffers are refilled per event, so a partly filled one from the
    // event that failed must not reach the next call.
    let valid = with_value_flow(crate::test_util::shapes::diamond_chain::<Instruction>(4));
    let mut invalid = Cfg::<Instruction>::new();
    invalid.block_mut(invalid.entry()).push(Instruction::access(
        [],
        [],
        MemoryAccess::read(Location::Value, [9]),
    ));

    let mut scratch = MemoryValueFlowScratch::new();
    let dominators = DominatorTree::compute(&valid);
    let ssa = SsaForm::compute(&valid, &dominators);
    let memory: MemorySSA<Location, u8, Fence> = MemorySSA::compute(&valid, &ExactMemoryAlias);
    assert!(MemoryValueFlow::compute_in(&mut scratch, &memory, &ssa).is_ok());

    let dominators = DominatorTree::compute(&invalid);
    let ssa = SsaForm::compute(&invalid, &dominators);
    let memory: MemorySSA<Location, u8, Fence> = MemorySSA::compute(&invalid, &ExactMemoryAlias);
    assert_eq!(
        MemoryValueFlow::compute_in(&mut scratch, &memory, &ssa),
        MemoryValueFlow::compute(&memory, &ssa)
    );
    assert!(matches!(
        MemoryValueFlow::compute_in(&mut scratch, &memory, &ssa),
        Err(MemoryValueFlowError::MissingVariable {
            role: MemoryValueRole::Loaded,
            ..
        })
    ));

    let dominators = DominatorTree::compute(&valid);
    let ssa = SsaForm::compute(&valid, &dominators);
    let memory: MemorySSA<Location, u8, Fence> = MemorySSA::compute(&valid, &ExactMemoryAlias);
    assert_eq!(
        MemoryValueFlow::compute_in(&mut scratch, &memory, &ssa),
        MemoryValueFlow::compute(&memory, &ssa)
    );
}

#[test]
fn the_public_accessors_agree_across_a_reused_scratch() {
    // Derived equality compares the stored graph; this walks the answer the
    // way a consumer reads it, so the proof does not rest on the derive.
    let large = with_value_flow(crate::test_util::shapes::diamond_chain::<Instruction>(12));
    let cfg = with_value_flow(crate::test_util::shapes::diamond_chain::<Instruction>(2));
    let mut scratch = MemoryValueFlowScratch::new();
    let (large_ssa, large_memory, _) = analyses(&large);
    drop(MemoryValueFlow::compute_in(&mut scratch, &large_memory, &large_ssa).unwrap());

    let dominators = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dominators);
    let memory: MemorySSA<Location, u8, Fence> = MemorySSA::compute(&cfg, &ExactMemoryAlias);
    let computed = MemoryValueFlow::compute_in(&mut scratch, &memory, &ssa).unwrap();
    let expected = MemoryValueFlow::compute(&memory, &ssa).unwrap();

    assert_eq!(computed.node_count(), expected.node_count());
    assert_eq!(computed.edge_count(), expected.edge_count());
    for node in expected.graph().node_ids() {
        assert_eq!(computed.graph().node(node), expected.graph().node(node));
    }
    for edge in expected.graph().edges() {
        assert!(has_edge(
            &computed,
            edge.source(),
            edge.target(),
            *edge.payload()
        ));
    }
    for event in memory.events() {
        assert_eq!(
            computed.event_node(event.site()),
            expected.event_node(event.site())
        );
    }
}
