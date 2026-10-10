extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::test_util::MemInst;
use crate::{
    Cfg, DominatorTree, InstrInfo, MemoryAccessKind, MemoryAtomicity, MemoryOperations,
    ProgramPoint, SsaForm, SsaValue,
};

use super::{MemoryAccess, MemoryEvent, MemoryEventInfo, MemoryTrace};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Location {
    Object,
    Field,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Fence {
    Function,
}

type Instruction = MemInst<Location, Fence>;

fn access(
    location: Location,
    kind: MemoryAccessKind,
    atomicity: MemoryAtomicity,
) -> MemoryEvent<Location, u8, Fence> {
    MemoryEvent::Access(MemoryAccess::opaque(location, kind).with_atomicity(atomicity))
}

#[test]
fn trace_preserves_instruction_and_event_order() {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(Instruction::new([
        access(
            Location::Object,
            MemoryAccessKind::Read,
            MemoryAtomicity::NonAtomic,
        ),
        access(
            Location::Field,
            MemoryAccessKind::ReadModifyWrite,
            MemoryAtomicity::Atomic,
        ),
    ]));
    cfg.block_mut(entry)
        .push(Instruction::new([MemoryEvent::Fence(Fence::Function)]));
    cfg.block_mut(entry).push(Instruction::new([access(
        Location::Object,
        MemoryAccessKind::Write,
        MemoryAtomicity::NonAtomic,
    )]));

    let trace = MemoryTrace::compute(&cfg);
    let points: Vec<_> = trace
        .entries()
        .iter()
        .map(|entry| (entry.point().inst_idx, entry.event_index()))
        .collect();

    assert_eq!(points, vec![(0, 0), (0, 1), (1, 0), (2, 0)]);
    assert_eq!(trace.entries_in(entry).count(), 4);
}

#[test]
fn read_modify_write_is_one_entry_in_both_direction_queries() {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(Instruction::new([access(
        Location::Object,
        MemoryAccessKind::ReadModifyWrite,
        MemoryAtomicity::Atomic,
    )]));

    let trace = MemoryTrace::compute(&cfg);

    assert_eq!(trace.entries().len(), 1);
    assert_eq!(trace.reads().count(), 1);
    assert_eq!(trace.writes().count(), 1);
    assert_eq!(trace.modifies().count(), 1);
    assert!(trace.entries()[0].event().modifies());
    assert_eq!(
        trace.operations_at(ProgramPoint {
            block: entry,
            inst_idx: 0,
        }),
        MemoryOperations::READ_MODIFY_WRITE
    );
    let MemoryEvent::Access(access) = trace.entries()[0].event() else {
        panic!("the trace entry must be a memory access");
    };
    assert_eq!(access.kind(), MemoryAccessKind::ReadModifyWrite);
    assert_eq!(access.atomicity(), MemoryAtomicity::Atomic);
}

#[test]
fn instruction_summary_distinguishes_read_write_and_modify() {
    let read = Instruction::new([access(
        Location::Object,
        MemoryAccessKind::Read,
        MemoryAtomicity::NonAtomic,
    )]);
    let write = Instruction::new([access(
        Location::Object,
        MemoryAccessKind::Write,
        MemoryAtomicity::NonAtomic,
    )]);
    let separate_read_write = Instruction::new([
        access(
            Location::Object,
            MemoryAccessKind::Read,
            MemoryAtomicity::NonAtomic,
        ),
        access(
            Location::Field,
            MemoryAccessKind::Write,
            MemoryAtomicity::NonAtomic,
        ),
    ]);
    let read_modify_write = Instruction::new([access(
        Location::Object,
        MemoryAccessKind::ReadModifyWrite,
        MemoryAtomicity::Atomic,
    )]);
    let fence = Instruction::new([MemoryEvent::Fence(Fence::Function)]);

    assert_eq!(read.memory_operations(), MemoryOperations::READ);
    assert_eq!(write.memory_operations(), MemoryOperations::WRITE);
    assert_eq!(
        separate_read_write.memory_operations(),
        MemoryOperations::READ_WRITE
    );
    assert_eq!(
        read_modify_write.memory_operations(),
        MemoryOperations::READ_MODIFY_WRITE
    );
    assert!(read_modify_write.memory_operations().reads());
    assert!(read_modify_write.memory_operations().modifies());
    assert!(read_modify_write.memory_operations().writes());
    assert_eq!(fence.memory_operations(), MemoryOperations::NONE);
}

#[test]
fn access_identifies_its_address_dependencies() {
    let instruction = Instruction::with_data_flow(
        [7, 9],
        [],
        [MemoryEvent::Access(
            MemoryAccess::read(Location::Object, Vec::new()).with_address_uses([7]),
        )],
    );

    let events: Vec<_> = instruction.memory_events().collect();
    let MemoryEvent::Access(access) = &events[0] else {
        panic!("the test instruction must contain one memory access");
    };
    assert_eq!(access.address_uses(), [7]);
    assert!(instruction.uses().contains(&access.address_uses()[0]));
    assert!(!access.address_uses().contains(&9));
}

#[test]
fn trace_resolves_address_dependencies_to_reaching_ssa_values() {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    cfg.block_mut(entry)
        .push(Instruction::with_data_flow([], [7], []));
    cfg.block_mut(entry).push(Instruction::with_data_flow(
        [7],
        [],
        [MemoryEvent::Access(
            MemoryAccess::read(Location::Object, Vec::new()).with_address_uses([7]),
        )],
    ));

    let dominators = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dominators);
    let trace = MemoryTrace::compute(&cfg);

    assert_eq!(
        trace.entries()[0].ssa_address_uses(&ssa),
        Some(vec![SsaValue::new(7, 1)])
    );
}

#[test]
fn point_lookup_and_fence_query_are_exact() {
    let mut cfg = Cfg::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(Instruction::new([]));
    cfg.block_mut(entry)
        .push(Instruction::new([MemoryEvent::Fence(Fence::Function)]));

    let trace = MemoryTrace::compute(&cfg);

    assert_eq!(
        trace.entries_at(ProgramPoint {
            block: entry,
            inst_idx: 0,
        }),
        []
    );
    assert_eq!(
        trace
            .entries_at(ProgramPoint {
                block: entry,
                inst_idx: 1,
            })
            .len(),
        1
    );
    assert_eq!(trace.fences().count(), 1);
    assert!(!trace.is_empty());
}
