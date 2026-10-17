//! The memory event vocabulary and the static `MemoryTrace`: constructors,
//! summaries, fences, atomicity, and the order and filters of the trace.

use cfglib::{
    BlockId, Cfg, DominatorTree, EdgeKind, MemoryAccess, MemoryAccessKind, MemoryAlias,
    MemoryAtomicity, MemoryEvent, MemoryEventInfo, MemoryOperations, MemorySSA, MemoryTrace,
    ProgramPoint, SsaForm, SsaValue,
};

use crate::machine::{Barrier, Inst, chain};

fn point(block: usize, inst_idx: usize) -> ProgramPoint {
    ProgramPoint {
        block: BlockId::from_index(block),
        inst_idx,
    }
}

const ALL_KINDS: [MemoryAccessKind; 3] = [
    MemoryAccessKind::Read,
    MemoryAccessKind::Write,
    MemoryAccessKind::ReadModifyWrite,
];

const SUMMARIES: [MemoryOperations; 5] = [
    MemoryOperations::NONE,
    MemoryOperations::READ,
    MemoryOperations::WRITE,
    MemoryOperations::READ_WRITE,
    MemoryOperations::READ_MODIFY_WRITE,
];

#[test]
fn each_summary_constant_reports_exactly_the_properties_it_names() {
    let properties =
        |summary: MemoryOperations| (summary.reads(), summary.modifies(), summary.writes());
    assert_eq!(properties(MemoryOperations::NONE), (false, false, false));
    assert_eq!(properties(MemoryOperations::READ), (true, false, false));
    assert_eq!(properties(MemoryOperations::WRITE), (false, false, true));
    assert_eq!(
        properties(MemoryOperations::READ_WRITE),
        (true, false, true)
    );
    assert_eq!(
        properties(MemoryOperations::READ_MODIFY_WRITE),
        (true, true, true)
    );
    assert!(MemoryOperations::NONE.is_empty());
    assert_eq!(MemoryOperations::default(), MemoryOperations::NONE);
    for summary in &SUMMARIES[1..] {
        assert!(!summary.is_empty());
    }
}

#[test]
fn combining_summaries_is_the_union_of_their_properties() {
    for left in SUMMARIES {
        assert_eq!(left.union(left), left, "idempotent");
        assert_eq!(
            left.union(MemoryOperations::NONE),
            left,
            "none is the identity"
        );
        for right in SUMMARIES {
            let combined = left.union(right);
            assert_eq!(combined, right.union(left), "commutative");
            assert_eq!(combined.reads(), left.reads() || right.reads());
            assert_eq!(combined.writes(), left.writes() || right.writes());
            assert_eq!(combined.modifies(), left.modifies() || right.modifies());
        }
    }
    assert_eq!(
        MemoryOperations::READ.union(MemoryOperations::WRITE),
        MemoryOperations::READ_WRITE,
        "separate reads and writes are not a modification"
    );
    assert_eq!(
        MemoryOperations::READ_WRITE.union(MemoryOperations::READ_MODIFY_WRITE),
        MemoryOperations::READ_MODIFY_WRITE
    );
}

#[test]
fn an_access_kind_implies_its_summary_and_a_modification_implies_a_read_and_a_write() {
    for kind in ALL_KINDS {
        let summary = kind.operations();
        assert_eq!(kind.reads(), summary.reads());
        assert_eq!(kind.writes(), summary.writes());
        assert_eq!(kind.modifies(), summary.modifies());
        if kind.modifies() {
            assert!(kind.reads() && kind.writes());
        }
    }
    assert_eq!(MemoryAccessKind::Read.operations(), MemoryOperations::READ);
    assert_eq!(
        MemoryAccessKind::Write.operations(),
        MemoryOperations::WRITE
    );
    assert_eq!(
        MemoryAccessKind::ReadModifyWrite.operations(),
        MemoryOperations::READ_MODIFY_WRITE
    );
}

#[test]
fn access_constructors_place_each_operand_in_its_role() {
    let read = MemoryAccess::read("slot", [9_u8]);
    assert_eq!(read.kind(), MemoryAccessKind::Read);
    assert_eq!((read.value_defs(), read.value_uses()), (&[9][..], &[][..]));

    let write = MemoryAccess::write("slot", [3_u8, 4]);
    assert_eq!(write.kind(), MemoryAccessKind::Write);
    assert_eq!(
        (write.value_uses(), write.value_defs()),
        (&[3, 4][..], &[][..])
    );

    let exchange = MemoryAccess::read_modify_write("slot", [3_u8], [9]);
    assert_eq!(exchange.kind(), MemoryAccessKind::ReadModifyWrite);
    assert_eq!(
        (exchange.value_uses(), exchange.value_defs()),
        (&[3][..], &[9][..])
    );

    for kind in ALL_KINDS {
        let opaque = MemoryAccess::<_, u8>::opaque("slot", kind);
        assert_eq!(opaque.kind(), kind);
        assert_eq!(opaque.value_uses(), []);
        assert_eq!(opaque.value_defs(), []);
        assert_eq!(opaque.operations(), kind.operations());
        assert_eq!(
            (opaque.reads(), opaque.modifies(), opaque.writes()),
            (kind.reads(), kind.modifies(), kind.writes())
        );
    }
}

#[test]
fn an_access_is_non_atomic_until_marked_and_keeps_its_address_inputs() {
    let plain = MemoryAccess::write("slot", [1_u8]);
    assert_eq!(plain.atomicity(), MemoryAtomicity::NonAtomic);
    assert_eq!(plain.address_uses(), []);
    assert_eq!(*plain.location(), "slot");

    let atomic = plain
        .with_address_uses([5, 6])
        .with_atomicity(MemoryAtomicity::Atomic);
    assert_eq!(atomic.atomicity(), MemoryAtomicity::Atomic);
    assert_eq!(atomic.address_uses(), [5, 6]);
    assert_eq!(
        atomic.value_uses(),
        [1],
        "marking atomicity leaves the operands alone"
    );
}

#[test]
fn a_fence_event_contributes_no_access_properties() {
    let fence = MemoryEvent::<&str, u8, Barrier>::Fence(Barrier::Release);
    assert!(fence.is_fence());
    assert_eq!(fence.operations(), MemoryOperations::NONE);
    assert!(!fence.reads() && !fence.writes() && !fence.modifies());

    for kind in ALL_KINDS {
        let access = MemoryEvent::<&str, u8, Barrier>::Access(MemoryAccess::opaque("slot", kind));
        assert!(!access.is_fence());
        assert_eq!(access.operations(), kind.operations());
        assert_eq!(access.reads(), kind.reads());
        assert_eq!(access.writes(), kind.writes());
        assert_eq!(access.modifies(), kind.modifies());
    }
}

#[test]
fn an_instruction_summarizes_the_union_of_its_events() {
    let read = MemoryEvent::Access(MemoryAccess::read("x", [1_u8]));
    let write = MemoryEvent::Access(MemoryAccess::write("y", [2_u8]));
    let exchange = MemoryEvent::Access(MemoryAccess::read_modify_write("z", [3_u8], [4]));
    let fence = MemoryEvent::Fence(Barrier::Acquire);
    let summary = |events: Vec<_>| Inst::new(&[], &[], events).memory_operations();

    assert_eq!(summary(vec![]), MemoryOperations::NONE);
    assert_eq!(
        summary(vec![fence.clone()]),
        MemoryOperations::NONE,
        "a fence alone is not an access"
    );
    assert_eq!(
        summary(vec![read.clone(), read.clone()]),
        MemoryOperations::READ
    );
    assert_eq!(
        summary(vec![read.clone(), write.clone()]),
        MemoryOperations::READ_WRITE
    );
    assert_eq!(summary(vec![fence, write]), MemoryOperations::WRITE);
    assert_eq!(
        summary(vec![read, exchange]),
        MemoryOperations::READ_MODIFY_WRITE
    );
}

/// Blocks 0, 1, 2 are allocated in that order but run 0, 2, 1.
fn out_of_order_program() -> Cfg<Inst<&'static str>> {
    let mut cfg = Cfg::new();
    let (first, last, middle) = (cfg.entry(), cfg.new_block(), cfg.new_block());
    for (block, code) in [
        (
            first,
            vec![
                Inst::compute(1, &[]),
                Inst::new(
                    &[],
                    &[],
                    vec![
                        MemoryEvent::Access(MemoryAccess::write("a", [1_u8])),
                        MemoryEvent::Fence(Barrier::Release),
                        MemoryEvent::Access(MemoryAccess::read("a", [2])),
                    ],
                ),
            ],
        ),
        (last, vec![Inst::exchange("b", 1, 3)]),
        (
            middle,
            vec![Inst::load("a", 4), Inst::fence(Barrier::Acquire)],
        ),
    ] {
        cfg.block_mut(block).instructions_mut().extend(code);
    }
    cfg.add_edge(first, middle, EdgeKind::Fallthrough);
    cfg.add_edge(middle, last, EdgeKind::Fallthrough);
    cfg
}

#[test]
fn the_trace_follows_block_allocation_instruction_and_event_order() {
    let cfg = out_of_order_program();
    let trace = MemoryTrace::compute(&cfg);
    let order: Vec<(ProgramPoint, usize)> = trace
        .entries()
        .iter()
        .map(|entry| (entry.point(), entry.event_index()))
        .collect();
    // Allocation order is block 0, 1, 2 even though control runs 0, 2, 1.
    assert_eq!(
        order,
        [
            (point(0, 1), 0),
            (point(0, 1), 1),
            (point(0, 1), 2),
            (point(1, 0), 0),
            (point(2, 0), 0),
            (point(2, 1), 0),
        ]
    );
    assert!(!trace.is_empty());
}

#[test]
fn the_trace_can_be_queried_by_instruction_and_by_block() {
    let cfg = out_of_order_program();
    let trace = MemoryTrace::compute(&cfg);
    assert_eq!(trace.entries_at(point(0, 1)).len(), 3);
    assert_eq!(
        trace.entries_at(point(0, 0)),
        [],
        "a pure instruction emits nothing"
    );
    assert_eq!(
        trace.entries_at(point(9, 9)),
        [],
        "an absent point emits nothing"
    );
    assert_eq!(trace.entries_in(BlockId::from_index(2)).count(), 2);
    assert_eq!(trace.entries_in(BlockId::from_index(0)).count(), 3);

    assert_eq!(
        trace.operations_at(point(0, 1)),
        MemoryOperations::READ_WRITE
    );
    assert_eq!(
        trace.operations_at(point(1, 0)),
        MemoryOperations::READ_MODIFY_WRITE
    );
    assert_eq!(
        trace.operations_at(point(2, 1)),
        MemoryOperations::NONE,
        "a fence is not an access"
    );
    assert_eq!(trace.operations_at(point(0, 0)), MemoryOperations::NONE);

    let entry = &trace.entries_at(point(1, 0))[0];
    assert_eq!(entry.point(), point(1, 0));
    assert!(matches!(entry.event(), MemoryEvent::Access(access) if *access.location() == "b"));
}

#[test]
fn a_read_modify_write_appears_once_in_each_filter_it_belongs_to() {
    let cfg = out_of_order_program();
    let trace = MemoryTrace::compute(&cfg);
    let sites =
        |entries: Vec<&cfglib::MemoryTraceEntry<&str, u8, Barrier>>| -> Vec<(ProgramPoint, usize)> {
            entries
                .iter()
                .map(|entry| (entry.point(), entry.event_index()))
                .collect()
        };
    assert_eq!(
        sites(trace.reads().collect()),
        [(point(0, 1), 2), (point(1, 0), 0), (point(2, 0), 0)]
    );
    assert_eq!(
        sites(trace.writes().collect()),
        [(point(0, 1), 0), (point(1, 0), 0)]
    );
    assert_eq!(sites(trace.modifies().collect()), [(point(1, 0), 0)]);
    assert_eq!(
        sites(trace.fences().collect()),
        [(point(0, 1), 1), (point(2, 1), 0)]
    );
}

#[test]
fn removed_blocks_and_an_empty_cfg_contribute_no_events() {
    let empty = Cfg::<Inst<&str>>::new();
    assert!(MemoryTrace::compute(&empty).is_empty());

    let mut cfg = chain(vec![vec![Inst::store("x", 1)], vec![Inst::store("y", 2)]]);
    let removed = cfg.block_ids().nth(1).expect("the second block");
    assert!(cfg.remove_block(removed));
    let trace = MemoryTrace::compute(&cfg);
    assert_eq!(trace.entries().len(), 1);
    assert_eq!(trace.entries_in(removed).count(), 0);
}

#[test]
fn address_dependencies_resolve_to_the_versions_the_instruction_reads() {
    // v0 and v1 are written twice before the store; the store's address
    // inputs are the second versions.
    let cfg = chain(vec![vec![
        Inst::compute(0, &[]),
        Inst::compute(1, &[]),
        Inst::compute(0, &[0]),
        Inst::store_indexed("table", 0, 1, 2),
        Inst::fence(Barrier::SequentiallyConsistent),
    ]]);
    let dominators = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dominators);
    let trace = MemoryTrace::compute(&cfg);
    let memory: MemorySSA<&str, u8, Barrier> = MemorySSA::compute(&cfg, &cfglib::ExactMemoryAlias);

    let store = point(0, 3);
    let reads = ssa
        .instruction(store)
        .expect("the store is annotated")
        .uses();
    let expected = vec![reads[0].clone(), reads[1].clone()];
    assert_eq!(
        expected[0],
        SsaValue::new(0, 2),
        "the second definition of v0"
    );
    assert_eq!(
        trace.entries_at(store)[0].ssa_address_uses(&ssa),
        Some(expected.clone())
    );
    assert_eq!(
        memory.events_at(store)[0].ssa_address_uses(&ssa),
        Some(expected)
    );

    let fence = point(0, 4);
    assert_eq!(
        trace.entries_at(fence)[0].ssa_address_uses(&ssa),
        None,
        "a fence selects no location"
    );
    assert_eq!(memory.events_at(fence)[0].ssa_address_uses(&ssa), None);

    // A form computed from a smaller CFG does not describe the point.
    let other = chain(vec![vec![Inst::<u8>::compute(0, &[])]]);
    let mismatched = SsaForm::compute(&other, &DominatorTree::compute(&other));
    assert_eq!(
        trace.entries_at(store)[0].ssa_address_uses(&mismatched),
        None
    );
}

#[test]
fn an_address_variable_the_instruction_does_not_use_resolves_to_nothing() {
    // The access claims variable 7 selects its sub-location, but the
    // instruction reports no such use: an adapter contract violation.
    let event = MemoryEvent::Access(MemoryAccess::write("table", [1_u8]).with_address_uses([7]));
    let cfg = chain(vec![vec![Inst::new(&[1], &[], vec![event])]]);
    let ssa = SsaForm::compute(&cfg, &DominatorTree::compute(&cfg));
    let trace = MemoryTrace::compute(&cfg);
    assert_eq!(trace.entries()[0].ssa_address_uses(&ssa), None);
}

/// A non-aliasing oracle is only a function of two locations.
#[test]
fn a_plain_function_is_a_memory_alias() {
    let adjacent = |left: &u8, right: &u8| left.abs_diff(*right) <= 1;
    assert!(adjacent.may_alias(&3, &4));
    assert!(!adjacent.may_alias(&3, &5));
}
