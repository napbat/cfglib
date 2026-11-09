extern crate alloc;

use alloc::vec::Vec;

use crate::test_util::MemInst;
use crate::{InstrInfo, MemoryAccessKind, MemoryAtomicity, MemoryOperations};

use super::{MemoryAccess, MemoryEvent, MemoryEventInfo};

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
fn operations_collect_to_the_union_of_their_properties() {
    let collected = |operations: &[MemoryOperations]| -> MemoryOperations {
        operations.iter().copied().collect()
    };

    assert_eq!(collected(&[]), MemoryOperations::NONE);
    assert_eq!(
        collected(&[MemoryOperations::READ, MemoryOperations::WRITE]),
        MemoryOperations::READ_WRITE
    );
    assert_eq!(
        collected(&[MemoryOperations::READ, MemoryOperations::READ_MODIFY_WRITE]),
        MemoryOperations::READ_MODIFY_WRITE
    );
    assert_eq!(
        collected(&[MemoryOperations::NONE, MemoryOperations::NONE]),
        MemoryOperations::NONE
    );
}
