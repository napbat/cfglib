extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::test_util::MemInst;
use crate::{
    Cfg, DominatorTree, EdgeKind, ExactMemoryAlias, MemoryAccess, MemoryAccessKind,
    MemoryDefinition, MemoryEvent, MemoryEventSite, MemorySSA, MemorySSAEvent, MemorySsaScratch,
    MemoryUse, ProgramPoint, SsaForm,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Location {
    A,
    B,
    C,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fence {
    Group,
}

type Instruction = MemInst<Location, Fence>;

/// Test-specific constructors over the shared memory mock.
impl MemInst<Location, Fence> {
    fn plain(uses: Vec<u8>, defs: Vec<u8>) -> Self {
        Self::with_data_flow(uses, defs, [])
    }

    fn with_events(events: Vec<MemoryEvent<Location, u8, Fence>>) -> Self {
        Self::new(events)
    }

    fn access(location: Location, kind: MemoryAccessKind) -> Self {
        Self::with_events(vec![memory_access(location, Vec::new(), kind)])
    }
}

fn memory_access(
    location: Location,
    address_uses: Vec<u8>,
    kind: MemoryAccessKind,
) -> MemoryEvent<Location, u8, Fence> {
    let access = match kind {
        MemoryAccessKind::Read => MemoryAccess::read(location, Vec::new()),
        MemoryAccessKind::Write => MemoryAccess::write(location, Vec::new()),
        MemoryAccessKind::ReadModifyWrite => {
            MemoryAccess::read_modify_write(location, Vec::new(), Vec::new())
        }
    };
    MemoryEvent::Access(access.with_address_uses(address_uses))
}

fn site(block: crate::BlockId, inst_idx: usize) -> MemoryEventSite {
    MemoryEventSite::new(ProgramPoint { block, inst_idx }, 0)
}

fn linear_cfg(instructions: impl IntoIterator<Item = Instruction>) -> Cfg<Instruction> {
    let mut cfg = Cfg::new();
    cfg.block_mut(cfg.entry())
        .instructions_mut()
        .extend(instructions);
    cfg
}

#[test]
fn blind_writes_clobber_without_becoming_semantic_readers() {
    let cfg = linear_cfg([
        Instruction::access(Location::A, MemoryAccessKind::Write),
        Instruction::access(Location::A, MemoryAccessKind::Read),
        Instruction::access(Location::A, MemoryAccessKind::Write),
        Instruction::access(Location::A, MemoryAccessKind::Read),
    ]);
    let memory = MemorySSA::compute(&cfg, &ExactMemoryAlias);
    let block = cfg.entry();
    let first_write = site(block, 0);
    let first_read = site(block, 1);
    let second_write = site(block, 2);
    let second_read = site(block, 3);

    assert_eq!(
        memory.reaching_definition(first_read),
        Some(&MemoryDefinition::Event { site: first_write })
    );
    assert_eq!(
        memory.clobbered_definition(second_write),
        Some(&MemoryDefinition::Event { site: first_write })
    );
    assert_eq!(
        memory.reaching_definition(second_read),
        Some(&MemoryDefinition::Event { site: second_write })
    );

    let first_version = memory
        .event(first_write)
        .and_then(MemorySSAEvent::written_version)
        .unwrap();
    assert_eq!(
        memory.users(first_version),
        &[MemoryUse::Event { site: first_read }]
    );
    assert_eq!(memory.transitive_readers(first_write), vec![first_read]);
}

#[test]
fn exact_locations_have_independent_live_in_states() {
    let cfg = linear_cfg([
        Instruction::access(Location::A, MemoryAccessKind::Write),
        Instruction::access(Location::B, MemoryAccessKind::Read),
    ]);
    let memory = MemorySSA::compute(&cfg, &ExactMemoryAlias);
    let read = site(cfg.entry(), 1);

    let Some(MemoryDefinition::LiveIn { class }) = memory.reaching_definition(read) else {
        panic!("an unrelated read must consume its own live-in state");
    };
    assert_eq!(Some(*class), memory.class_of(&Location::B));
    assert_ne!(memory.class_of(&Location::A), memory.class_of(&Location::B));
}

#[test]
fn caller_alias_relation_is_symmetrized_and_closed_transitively() {
    let cfg = linear_cfg([
        Instruction::access(Location::A, MemoryAccessKind::Write),
        Instruction::access(Location::B, MemoryAccessKind::Read),
        Instruction::access(Location::C, MemoryAccessKind::Read),
    ]);
    let alias = |left: &Location, right: &Location| {
        matches!(
            (left, right),
            (Location::A, Location::B) | (Location::B, Location::C)
        )
    };
    let memory = MemorySSA::compute(&cfg, &alias);

    assert_eq!(memory.classes().len(), 1);
    assert!(memory.may_alias(&Location::A, &Location::C));
    assert_eq!(memory.events_for(&Location::B).count(), 3);
    assert_eq!(
        memory.reaching_definition(site(cfg.entry(), 2)),
        Some(&MemoryDefinition::Event {
            site: site(cfg.entry(), 0),
        })
    );
}

#[test]
fn branch_merge_expands_memory_phi_to_both_writes() {
    let mut cfg = Cfg::<Instruction>::new();
    let left = cfg.new_block();
    let right = cfg.new_block();
    let merge = cfg.new_block();
    cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    cfg.block_mut(left)
        .push(Instruction::access(Location::A, MemoryAccessKind::Write));
    cfg.block_mut(right)
        .push(Instruction::access(Location::A, MemoryAccessKind::Write));
    cfg.block_mut(merge)
        .push(Instruction::access(Location::A, MemoryAccessKind::Read));

    let memory = MemorySSA::compute(&cfg, &ExactMemoryAlias);
    let read = site(merge, 0);
    assert!(matches!(
        memory.reaching_definition(read),
        Some(MemoryDefinition::Phi { block, .. }) if *block == merge
    ));

    let mut definitions: Vec<_> = memory
        .reaching_definitions(read)
        .into_iter()
        .filter_map(|definition| match definition {
            MemoryDefinition::Event { site } => Some(*site),
            MemoryDefinition::LiveIn { .. } | MemoryDefinition::Phi { .. } => None,
        })
        .collect();
    definitions.sort_unstable();
    assert_eq!(definitions, [site(left, 0), site(right, 0)]);
}

#[test]
fn loop_header_phi_retains_live_in_and_backedge_write() {
    let mut cfg = Cfg::<Instruction>::new();
    let header = cfg.new_block();
    let body = cfg.new_block();
    let exit = cfg.new_block();
    cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, header, EdgeKind::Back);
    cfg.block_mut(header)
        .push(Instruction::access(Location::A, MemoryAccessKind::Read));
    cfg.block_mut(body)
        .push(Instruction::access(Location::A, MemoryAccessKind::Write));

    let memory = MemorySSA::compute(&cfg, &ExactMemoryAlias);
    let definitions = memory.reaching_definitions(site(header, 0));
    assert_eq!(definitions.len(), 2);
    assert!(definitions.iter().any(|definition| matches!(
        definition,
        MemoryDefinition::LiveIn { class }
            if Some(*class) == memory.class_of(&Location::A)
    )));
    assert!(definitions.iter().any(|definition| matches!(
        definition,
        MemoryDefinition::Event { site: definition_site }
            if *definition_site == site(body, 0)
    )));
}

#[test]
fn read_modify_write_links_both_sides_of_the_data_flow() {
    let cfg = linear_cfg([
        Instruction::access(Location::A, MemoryAccessKind::Write),
        Instruction::access(Location::A, MemoryAccessKind::ReadModifyWrite),
        Instruction::access(Location::A, MemoryAccessKind::Read),
    ]);
    let memory = MemorySSA::compute(&cfg, &ExactMemoryAlias);
    let block = cfg.entry();
    let write = site(block, 0);
    let modify = site(block, 1);
    let read = site(block, 2);

    let event = memory.event(modify).unwrap();
    assert!(event.reads() && event.modifies() && event.writes());
    assert_eq!(
        memory.reaching_definition(modify),
        Some(&MemoryDefinition::Event { site: write })
    );
    assert_eq!(
        memory.reaching_definition(read),
        Some(&MemoryDefinition::Event { site: modify })
    );
    assert_eq!(memory.transitive_readers(write), [modify, read]);
}

#[test]
fn event_order_and_fences_remain_visible_without_changing_memory_state() {
    let instruction = Instruction::with_events(vec![
        memory_access(Location::A, Vec::new(), MemoryAccessKind::Write),
        MemoryEvent::Fence(Fence::Group),
        memory_access(Location::A, Vec::new(), MemoryAccessKind::Read),
    ]);
    let cfg = linear_cfg([instruction]);
    let memory = MemorySSA::compute(&cfg, &ExactMemoryAlias);
    let point = ProgramPoint {
        block: cfg.entry(),
        inst_idx: 0,
    };
    let events = memory.events_at(point);

    assert_eq!(events.len(), 3);
    assert_eq!(events[0].site().event_index(), 0);
    assert!(events[1].is_fence());
    assert_eq!(events[2].site().event_index(), 2);
    assert_eq!(memory.fences().count(), 1);
    assert_eq!(
        memory.operations_at(point),
        crate::MemoryOperations::READ_WRITE
    );
    assert_eq!(
        memory.reaching_definition(events[2].site()),
        Some(&MemoryDefinition::Event {
            site: events[0].site(),
        })
    );
}

#[test]
fn address_dependencies_resolve_to_ordinary_ssa_values() {
    let mut read = Instruction::with_events(vec![memory_access(
        Location::A,
        vec![7],
        MemoryAccessKind::Read,
    )]);
    read.uses.push(7);
    let cfg = linear_cfg([Instruction::plain(Vec::new(), vec![7]), read]);
    let dominators = DominatorTree::compute(&cfg);
    let values = SsaForm::compute(&cfg, &dominators);
    let memory = MemorySSA::compute(&cfg, &ExactMemoryAlias);
    let event = memory.event(site(cfg.entry(), 1)).unwrap();

    assert_eq!(event.location(), Some(&Location::A));
    assert_eq!(event.address_uses(), Some(&[7][..]));
    assert_eq!(
        event.ssa_address_uses(&values).unwrap(),
        values
            .instruction(ProgramPoint {
                block: cfg.entry(),
                inst_idx: 1,
            })
            .unwrap()
            .uses
    );
}

/// Give every block of a shared shape a load, a store, and a
/// read/modify/write over a small set of locations that repeats across
/// blocks, so the shadow CFG takes memory phis.
fn with_memory(mut cfg: Cfg<Instruction>) -> Cfg<Instruction> {
    const LOCATIONS: [Location; 3] = [Location::A, Location::B, Location::C];
    for (index, block) in cfg.block_ids().collect::<Vec<_>>().into_iter().enumerate() {
        let location = LOCATIONS[index % LOCATIONS.len()];
        let next = LOCATIONS[(index + 1) % LOCATIONS.len()];
        cfg.block_mut(block)
            .push(Instruction::access(location, MemoryAccessKind::Write));
        cfg.block_mut(block)
            .push(Instruction::access(next, MemoryAccessKind::Read));
        cfg.block_mut(block).push(Instruction::access(
            location,
            MemoryAccessKind::ReadModifyWrite,
        ));
    }
    cfg
}

#[test]
fn one_scratch_reused_down_a_sequence_computes_the_allocating_answer() {
    let sequence: Vec<_> = crate::test_util::shapes::scratch_sequence::<Instruction>()
        .into_iter()
        .map(with_memory)
        .collect();
    let mut scratch = MemorySsaScratch::new();
    // Twice, so the second pass sees a scratch every buffer of which is
    // already at the sequence's high-water mark.
    for _ in 0..2 {
        for cfg in &sequence {
            let computed: MemorySSA<Location, u8, Fence> =
                MemorySSA::compute_in(&mut scratch, cfg, &ExactMemoryAlias);
            assert_eq!(
                computed,
                MemorySSA::compute(cfg, &ExactMemoryAlias),
                "{} blocks",
                cfg.block_count()
            );
        }
    }
}

#[test]
fn a_reused_scratch_survives_a_large_procedure_before_a_small_one() {
    let large = with_memory(crate::test_util::shapes::diamond_chain::<Instruction>(30));
    let small = with_memory(crate::test_util::shapes::diamond_chain::<Instruction>(1));
    let mut scratch = MemorySsaScratch::new();

    let first: MemorySSA<Location, u8, Fence> =
        MemorySSA::compute_in(&mut scratch, &large, &ExactMemoryAlias);
    assert_eq!(first, MemorySSA::compute(&large, &ExactMemoryAlias));
    let second: MemorySSA<Location, u8, Fence> =
        MemorySSA::compute_in(&mut scratch, &small, &ExactMemoryAlias);
    assert_eq!(second, MemorySSA::compute(&small, &ExactMemoryAlias));
}

#[test]
fn a_reused_scratch_crosses_procedures_with_disjoint_locations() {
    // The alias merge is the buffer that could carry one procedure's
    // locations into the next one's class numbering.
    let first = linear_cfg([
        Instruction::access(Location::C, MemoryAccessKind::Write),
        Instruction::access(Location::A, MemoryAccessKind::Read),
    ]);
    let second = linear_cfg([Instruction::access(Location::B, MemoryAccessKind::Write)]);

    let mut scratch = MemorySsaScratch::new();
    for cfg in [&first, &second, &first] {
        let computed: MemorySSA<Location, u8, Fence> =
            MemorySSA::compute_in(&mut scratch, cfg, &ExactMemoryAlias);
        assert_eq!(computed, MemorySSA::compute(cfg, &ExactMemoryAlias));
    }
    let only_b: MemorySSA<Location, u8, Fence> =
        MemorySSA::compute_in(&mut scratch, &second, &ExactMemoryAlias);
    assert_eq!(only_b.classes().len(), 1);
    assert_eq!(only_b.class_of(&Location::A), None);
}

#[test]
fn the_public_accessors_agree_element_by_element_across_a_reused_scratch() {
    // Derived equality compares the stored maps; this walks the answer the
    // way a consumer reads it, so the proof does not rest on the derive.
    let cfg = with_memory(crate::test_util::shapes::diamond_chain::<Instruction>(3));
    let mut scratch = MemorySsaScratch::new();
    drop(MemorySSA::<Location, u8, Fence>::compute_in(
        &mut scratch,
        &with_memory(crate::test_util::shapes::diamond_chain::<Instruction>(12)),
        &ExactMemoryAlias,
    ));
    let computed: MemorySSA<Location, u8, Fence> =
        MemorySSA::compute_in(&mut scratch, &cfg, &ExactMemoryAlias);
    let expected: MemorySSA<Location, u8, Fence> = MemorySSA::compute(&cfg, &ExactMemoryAlias);

    assert_eq!(computed.classes(), expected.classes());
    assert_eq!(computed.phis(), expected.phis());
    assert_eq!(computed.events(), expected.events());
    for location in [Location::A, Location::B, Location::C] {
        assert_eq!(computed.class_of(&location), expected.class_of(&location));
    }
    for event in computed.events() {
        let site = event.site();
        assert_eq!(computed.event(site), expected.event(site));
        assert_eq!(
            computed.events_at(site.point()),
            expected.events_at(site.point())
        );
        assert_eq!(
            computed.reaching_definition(site),
            expected.reaching_definition(site)
        );
        assert_eq!(
            computed.clobbered_definition(site),
            expected.clobbered_definition(site)
        );
        assert_eq!(
            computed.reaching_definitions(site),
            expected.reaching_definitions(site)
        );
        assert_eq!(
            computed.transitive_readers(site),
            expected.transitive_readers(site)
        );
        for version in [
            event.read_version(),
            event.written_version(),
            event.clobbered_version(),
        ]
        .into_iter()
        .flatten()
        {
            assert_eq!(computed.definition(version), expected.definition(version));
            assert_eq!(computed.users(version), expected.users(version));
        }
    }
}
