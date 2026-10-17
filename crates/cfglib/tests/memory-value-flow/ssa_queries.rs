//! `MemorySSA`: versions, clobbers, phis at joins, unwind points, fences,
//! atomicity, and its queries, on hand-derived programs and on random ones
//! against the reaching-store exploration.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{
    BlockId, Cfg, ConservativeMemoryAlias, EdgeKind, ExactMemoryAlias, MemoryAccess,
    MemoryAtomicity, MemoryDefinition, MemoryEvent, MemoryEventSite, MemoryOperations, MemorySSA,
    MemoryUse, ProgramPoint, SsaValue,
};

use crate::machine::{Barrier, Inst, Program, Rng, chain, diamond};
use crate::oracle::{Def, Reaching, alias_classes, expand, explore};

type Memory = MemorySSA<u8, u8, Barrier>;

fn point(block: usize, inst_idx: usize) -> ProgramPoint {
    ProgramPoint {
        block: BlockId::from_index(block),
        inst_idx,
    }
}

fn site(block: usize, inst_idx: usize) -> MemoryEventSite {
    MemoryEventSite::new(point(block, inst_idx), 0)
}

fn exact(cfg: &Cfg<Inst<u8>>) -> Memory {
    MemorySSA::compute(cfg, &ExactMemoryAlias)
}

#[test]
fn a_blind_write_replaces_the_state_without_reading_it() {
    // store x; store x; load x
    let cfg = chain(vec![vec![
        Inst::store(0, 1),
        Inst::store(0, 2),
        Inst::load(0, 3),
    ]]);
    let memory = exact(&cfg);
    let (first, second, load) = (site(0, 0), site(0, 1), site(0, 2));

    assert_eq!(
        memory.reaching_definition(load),
        Some(&MemoryDefinition::Event { site: second })
    );
    assert_eq!(
        memory.reaching_definitions(load),
        [&MemoryDefinition::Event { site: second }]
    );
    assert_eq!(
        memory.clobbered_definition(second),
        Some(&MemoryDefinition::Event { site: first })
    );
    let class = memory.class_of(&0).expect("the location has a class");
    assert_eq!(
        memory.clobbered_definition(first),
        Some(&MemoryDefinition::LiveIn { class })
    );

    let first_state = memory
        .event(first)
        .unwrap()
        .written_version()
        .unwrap()
        .clone();
    let second_state = memory
        .event(second)
        .unwrap()
        .written_version()
        .unwrap()
        .clone();
    assert_eq!(
        memory.users(&first_state),
        [],
        "a blind write is not a reader of what it replaces"
    );
    assert_eq!(
        memory.users(&second_state),
        [MemoryUse::Event { site: load }]
    );
    assert_eq!(
        memory.definition(&second_state),
        Some(&MemoryDefinition::Event { site: second })
    );
    assert_eq!(
        memory.definition(&SsaValue::live_in(class)),
        Some(&MemoryDefinition::LiveIn { class })
    );

    // The versions of one class chain through the writes.
    assert_eq!(
        memory.event(second).unwrap().clobbered_version(),
        Some(&first_state)
    );
    assert_eq!(
        memory.event(load).unwrap().read_version(),
        Some(&second_state)
    );
    assert!(memory.event(load).unwrap().written_version().is_none());
    assert!(memory.event(first).unwrap().read_version().is_none());

    assert_eq!(memory.transitive_readers(first), []);
    assert_eq!(memory.transitive_readers(second), [load]);
    assert_eq!(
        memory.transitive_readers(load),
        [],
        "a read has no written state"
    );
}

#[test]
fn a_read_modify_write_both_reads_and_replaces_and_carries_the_dependency_on() {
    // store x; exchange x; load x
    let cfg = chain(vec![vec![
        Inst::store(0, 1),
        Inst::exchange(0, 2, 3),
        Inst::load(0, 4),
    ]]);
    let memory = exact(&cfg);
    let (store, exchange, load) = (site(0, 0), site(0, 1), site(0, 2));

    let stored = memory
        .event(store)
        .unwrap()
        .written_version()
        .unwrap()
        .clone();
    assert_eq!(
        memory.users(&stored),
        [MemoryUse::Event { site: exchange }],
        "the exchange reads the store"
    );
    let swapped = memory.event(exchange).unwrap();
    assert_eq!(swapped.read_version(), Some(&stored));
    assert_eq!(swapped.clobbered_version(), Some(&stored));
    assert_ne!(swapped.written_version(), Some(&stored));
    assert_eq!(
        memory.reaching_definition(load),
        Some(&MemoryDefinition::Event { site: exchange })
    );

    assert_eq!(
        memory.transitive_readers(store),
        [exchange, load],
        "through the exchange's own write"
    );
    assert_eq!(memory.transitive_readers(exchange), [load]);
    assert_eq!(
        memory.operations_at(point(0, 1)),
        MemoryOperations::READ_MODIFY_WRITE
    );
    assert_eq!(memory.modifies().count(), 1);
    assert_eq!(memory.reads().count(), 2);
    assert_eq!(memory.writes().count(), 2);
}

#[test]
fn a_join_merges_the_state_each_arm_left_in_one_phi() {
    let cfg = diamond(
        vec![],
        vec![Inst::store(0, 1)],
        vec![Inst::store(0, 2)],
        vec![Inst::load(0, 3)],
    );
    let memory = exact(&cfg);
    let (left, right, load) = (site(1, 0), site(2, 0), site(3, 0));
    let class = memory.class_of(&0).unwrap();

    let [phi] = memory.phis() else {
        panic!("one class merges at the join");
    };
    assert_eq!(phi.block(), BlockId::from_index(3));
    assert_eq!(phi.result().variable, class);
    let left_state = memory
        .event(left)
        .unwrap()
        .written_version()
        .unwrap()
        .clone();
    let right_state = memory
        .event(right)
        .unwrap()
        .written_version()
        .unwrap()
        .clone();
    assert_eq!(
        phi.operands(),
        [
            (point(1, 1), left_state.clone()),
            (point(2, 1), right_state.clone())
        ],
        "one operand per predecessor, departing at its block's end"
    );
    assert_eq!(
        memory.definition(phi.result()),
        Some(&MemoryDefinition::Phi {
            block: phi.block(),
            class
        })
    );
    assert_eq!(
        memory.reaching_definition(load),
        Some(&MemoryDefinition::Phi {
            block: phi.block(),
            class
        })
    );

    let reaching: BTreeSet<_> = memory
        .reaching_definitions(load)
        .into_iter()
        .map(|definition| match definition {
            MemoryDefinition::Event { site } => *site,
            other => panic!("only stores reach the load, not {other:?}"),
        })
        .collect();
    assert_eq!(reaching, BTreeSet::from([left, right]));
    assert_eq!(
        memory.reaching_definitions(load).len(),
        2,
        "each concrete definition is reported once"
    );

    assert_eq!(
        memory.users(&left_state),
        [MemoryUse::Phi {
            block: phi.block(),
            predecessor: point(1, 1),
            result: phi.result().clone(),
        }]
    );
    assert_eq!(memory.transitive_readers(left), [load], "across the phi");
    assert_eq!(memory.transitive_readers(right), [load]);
}

#[test]
fn only_the_classes_written_in_an_arm_get_a_phi_and_the_other_arm_supplies_the_live_in() {
    // x is written on the left only, y on both.
    let cfg = diamond(
        vec![],
        vec![Inst::store(0, 1), Inst::store(1, 1)],
        vec![Inst::store(1, 2)],
        vec![Inst::load(0, 3), Inst::load(1, 4)],
    );
    let memory = exact(&cfg);
    let (x, y) = (memory.class_of(&0).unwrap(), memory.class_of(&1).unwrap());
    assert_eq!(memory.phis().len(), 2);
    let phi_x = memory
        .phis()
        .iter()
        .find(|phi| phi.result().variable == x)
        .unwrap();
    // The right arm never wrote x, so the initial state flows in from there.
    assert_eq!(phi_x.operands()[1], (point(2, 1), SsaValue::live_in(x)));
    assert_eq!(
        memory.reaching_definitions(site(3, 0)).len(),
        2,
        "the left store and the initial state"
    );
    assert!(
        memory
            .reaching_definitions(site(3, 0))
            .contains(&&MemoryDefinition::LiveIn { class: x })
    );
    assert!(
        !memory
            .reaching_definitions(site(3, 1))
            .contains(&&MemoryDefinition::LiveIn { class: y })
    );
}

#[test]
fn a_loop_header_merges_the_state_from_before_the_loop_and_from_the_back_edge() {
    // entry: store x; header: load x; body: store x, back to the header; exit.
    let mut cfg = Cfg::new();
    let blocks = [
        cfg.entry(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
    ];
    cfg.block_mut(blocks[0]).push(Inst::store(0, 1));
    cfg.block_mut(blocks[1]).push(Inst::load(0, 2));
    cfg.block_mut(blocks[2]).push(Inst::store(0, 3));
    cfg.add_edge(blocks[0], blocks[1], EdgeKind::Fallthrough);
    cfg.add_edge(blocks[1], blocks[2], EdgeKind::ConditionalFalse);
    cfg.add_edge(blocks[1], blocks[3], EdgeKind::ConditionalTrue);
    cfg.add_edge(blocks[2], blocks[1], EdgeKind::Back);
    let memory = exact(&cfg);

    let load = site(1, 0);
    let reaching: BTreeSet<_> = memory
        .reaching_definitions(load)
        .into_iter()
        .map(|definition| format!("{definition:?}"))
        .collect();
    assert_eq!(reaching.len(), 2, "the entry store and the loop store");
    let [phi] = memory.phis() else {
        panic!("one class merges at the header");
    };
    assert_eq!(phi.block(), blocks[1]);
    assert_eq!(phi.operands().len(), 2);
    assert_eq!(
        memory.transitive_readers(site(2, 0)),
        [load],
        "the loop store feeds the next iteration's load"
    );
    assert_eq!(memory.transitive_readers(site(0, 0)), [load]);
}

#[test]
fn an_unwind_carries_the_memory_state_before_the_throwing_instruction() {
    // entry: store x; call (throws); the handler and the normal successor
    // both load x. The store precedes the throw, so both see it.
    let mut cfg = Cfg::new();
    let (entry, normal, handler) = (cfg.entry(), cfg.new_block(), cfg.new_block());
    cfg.block_mut(entry).push(Inst::store(0, 1));
    cfg.block_mut(entry).push(Inst::call());
    cfg.block_mut(normal).push(Inst::load(0, 2));
    cfg.block_mut(handler).push(Inst::load(0, 3));
    cfg.add_edge(entry, normal, EdgeKind::Fallthrough);
    cfg.add_edge(entry, handler, EdgeKind::ExceptionUnwind);
    let memory = exact(&cfg);
    let store = site(0, 0);
    assert_eq!(
        memory.reaching_definitions(site(1, 0)),
        [&MemoryDefinition::Event { site: store }]
    );
    assert_eq!(
        memory.reaching_definitions(site(2, 0)),
        [&MemoryDefinition::Event { site: store }]
    );
}

#[test]
fn a_throwing_store_does_not_reach_its_own_handler() {
    // The faulting store may throw before it commits: the normal successor
    // sees its write, the handler sees the state from before it.
    let mut cfg = Cfg::new();
    let (entry, normal, handler) = (cfg.entry(), cfg.new_block(), cfg.new_block());
    cfg.block_mut(entry).push(Inst::store(0, 1).throwing());
    cfg.block_mut(normal).push(Inst::load(0, 2));
    cfg.block_mut(handler).push(Inst::load(0, 3));
    cfg.add_edge(entry, normal, EdgeKind::Fallthrough);
    cfg.add_edge(entry, handler, EdgeKind::ExceptionUnwind);
    let memory = exact(&cfg);
    let class = memory.class_of(&0).unwrap();

    assert_eq!(
        memory.reaching_definitions(site(1, 0)),
        [&MemoryDefinition::Event { site: site(0, 0) }]
    );
    assert_eq!(
        memory.reaching_definitions(site(2, 0)),
        [&MemoryDefinition::LiveIn { class }]
    );

    // The handler merges the state before the throwing instruction; the
    // operand names that instruction as its departure.
    let [phi] = memory.phis() else {
        panic!("a write at the throwing point places a phi at the unwind successor");
    };
    assert_eq!(phi.block(), handler);
    assert_eq!(phi.operands(), [(point(0, 0), SsaValue::live_in(class))]);
}

#[test]
fn fences_stay_in_the_event_stream_but_define_nothing() {
    let cfg = chain(vec![vec![
        Inst::store(0, 1),
        Inst::fence(Barrier::Release),
        Inst::load(0, 2),
    ]]);
    let memory = exact(&cfg);
    let fence = memory.event(site(0, 1)).unwrap();
    assert!(fence.is_fence() && fence.access().is_none());
    assert_eq!(fence.fence(), Some(&Barrier::Release));
    assert_eq!(fence.class(), None);
    assert_eq!(fence.location(), None);
    assert_eq!(fence.kind(), None);
    assert_eq!(fence.atomicity(), None);
    assert_eq!(fence.address_uses(), None);
    assert_eq!(fence.operations(), MemoryOperations::NONE);
    assert!(!fence.reads() && !fence.writes() && !fence.modifies());
    assert!(
        fence.read_version().is_none()
            && fence.written_version().is_none()
            && fence.clobbered_version().is_none()
    );
    assert_eq!(memory.fences().count(), 1);
    assert_eq!(memory.operations_at(point(0, 1)), MemoryOperations::NONE);

    // The fence does not interrupt the flow from the store to the load.
    assert_eq!(
        memory.reaching_definitions(site(0, 2)),
        [&MemoryDefinition::Event { site: site(0, 0) }]
    );
    assert_eq!(memory.phis(), []);
    assert_eq!(memory.events().len(), 3);
}

#[test]
fn atomic_accesses_report_their_atomicity_and_fences_report_none() {
    let cfg = chain(vec![vec![
        Inst::store(0, 1).atomic(),
        Inst::load(0, 2),
        Inst::exchange(1, 3, 4).atomic(),
    ]]);
    let memory = exact(&cfg);
    assert_eq!(
        memory.event(site(0, 0)).unwrap().atomicity(),
        Some(MemoryAtomicity::Atomic)
    );
    assert_eq!(
        memory.event(site(0, 1)).unwrap().atomicity(),
        Some(MemoryAtomicity::NonAtomic)
    );
    assert_eq!(
        memory.event(site(0, 2)).unwrap().atomicity(),
        Some(MemoryAtomicity::Atomic)
    );
    // Atomicity does not change what reaches what.
    assert_eq!(
        memory.reaching_definitions(site(0, 1)),
        [&MemoryDefinition::Event { site: site(0, 0) }]
    );
}

#[test]
fn events_are_grouped_by_instruction_and_by_alias_class() {
    let cfg = chain(vec![vec![
        Inst::store(0, 1),
        Inst::store(1, 1),
        Inst::new(
            &[],
            &[2],
            vec![
                MemoryEvent::Access(MemoryAccess::read(0, [2])),
                MemoryEvent::Access(MemoryAccess::read(1, [])),
            ],
        ),
    ]]);
    let memory = exact(&cfg);
    assert_eq!(memory.events_at(point(0, 2)).len(), 2);
    assert_eq!(memory.events_at(point(0, 9)), []);
    assert_eq!(
        memory.events_for(&0).count(),
        2,
        "the store and the first read"
    );
    assert_eq!(
        memory.events_for(&7).count(),
        0,
        "an unseen location has no events"
    );
    assert_eq!(
        memory
            .event(MemoryEventSite::new(point(0, 2), 1))
            .unwrap()
            .location(),
        Some(&1)
    );
    assert_eq!(
        memory
            .event(MemoryEventSite::new(point(0, 2), 5))
            .map(|_| ()),
        None
    );
    assert_eq!(memory.operations_at(point(0, 2)), MemoryOperations::READ);

    let merged: Memory = MemorySSA::compute(&cfg, &ConservativeMemoryAlias);
    assert_eq!(
        merged.events_for(&0).count(),
        4,
        "one class holds every access"
    );
}

#[test]
fn a_removed_block_slot_leaves_the_others_events_and_identities_alone() {
    let mut cfg = Cfg::new();
    let (entry, gone, kept) = (cfg.entry(), cfg.new_block(), cfg.new_block());
    cfg.block_mut(entry).push(Inst::store(0, 1));
    cfg.block_mut(gone).push(Inst::store(0, 2));
    cfg.block_mut(kept).push(Inst::load(0, 3));
    cfg.add_edge(entry, gone, EdgeKind::Fallthrough);
    cfg.add_edge(entry, kept, EdgeKind::Fallthrough);
    cfg.add_edge(gone, kept, EdgeKind::Fallthrough);
    assert!(cfg.remove_block(gone));
    let memory = exact(&cfg);
    assert_eq!(memory.events().len(), 2);
    assert_eq!(memory.events_at(point(1, 0)), []);
    assert_eq!(
        memory.reaching_definitions(site(2, 0)),
        [&MemoryDefinition::Event { site: site(0, 0) }]
    );
}

#[test]
fn memory_ssa_matches_the_reaching_store_exploration_on_random_programs() {
    let mut rng = Rng::new(61);
    let mut round = 0;
    let (mut merged_reads, mut phi_count, mut handler_phis) = (0, 0, 0);
    while round < 300 {
        let program = Program::random(&mut rng, true);
        // A phi operand from unreachable code names that code's own states,
        // which no execution from the entry visits.
        if !program.every_live_block_is_reachable() {
            continue;
        }
        round += 1;
        let cfg = program.cfg();
        // Alternate between alias vocabularies, including a non-transitive
        // one whose closure merges a run of neighbors.
        let relation: &dyn Fn(&u8, &u8) -> bool = match round % 3 {
            0 => &|left, right| left == right,
            1 => &|left, right| left.abs_diff(*right) <= 1,
            _ => &|left, right| left % 2 == right % 2,
        };
        let memory: Memory = MemorySSA::compute(&cfg, &relation);
        let locations = program.accessed_locations();
        let representative = alias_classes(&locations, relation);
        let reaching = explore(&program, &representative);

        assert_classes_are_alias_components(&memory, &locations, &representative, round);
        assert_classes_are_sorted_and_addressable(&memory);

        assert_reads_match_exploration(&memory, &reaching, round);
        assert_clobbers_match_exploration(&memory, &reaching, round);
        assert_users_invert_versions(&memory);

        merged_reads += reaching
            .reads
            .values()
            .filter(|defs| defs.len() >= 2)
            .count();
        phi_count += memory.phis().len();
        let handlers: BTreeSet<usize> = program
            .live_edges()
            .into_iter()
            .filter(|(_, _, kind)| *kind == EdgeKind::ExceptionUnwind)
            .map(|(_, target, _)| target)
            .collect();
        handler_phis += memory
            .phis()
            .iter()
            .filter(|phi| handlers.contains(&phi.block().index()))
            .count();
    }
    // A generator that never produced a merge, a phi, or a phi at a handler
    // would let every comparison above pass without exercising it.
    assert!(
        merged_reads >= 20,
        "reads with several reaching writes: {merged_reads}"
    );
    assert!(phi_count >= 20, "memory phis: {phi_count}");
    assert!(handler_phis >= 1, "phis at unwind handlers: {handler_phis}");
}

/// The classes are exactly the connected components of the alias relation.
fn assert_classes_are_alias_components(
    memory: &Memory,
    locations: &BTreeSet<u8>,
    representative: &BTreeMap<u8, u8>,
    round: usize,
) {
    for left in locations {
        for right in locations {
            if let (Some(a), Some(b)) = (memory.class_of(left), memory.class_of(right)) {
                assert_eq!(
                    a == b,
                    representative[left] == representative[right],
                    "round {round}"
                );
            }
        }
    }
}

/// Each class lists its locations in ascending order and is found by its id.
fn assert_classes_are_sorted_and_addressable(memory: &Memory) {
    for class in memory.classes() {
        assert!(
            class.locations().windows(2).all(|pair| pair[0] < pair[1]),
            "sorted locations"
        );
        assert_eq!(memory.class(class.id()), Some(class));
    }
}

/// Each read sees exactly the definitions the exploration finds reaching it,
/// none of them twice.
fn assert_reads_match_exploration(memory: &Memory, reaching: &Reaching<u8>, round: usize) {
    for (read, expected) in &reaching.reads {
        let actual: Vec<Def<u8>> = memory
            .reaching_definitions(*read)
            .into_iter()
            .map(|definition| match definition {
                MemoryDefinition::Event { site } => Def::Write(*site),
                MemoryDefinition::LiveIn { class } => {
                    Def::LiveIn(memory.class(*class).unwrap().locations()[0])
                }
                MemoryDefinition::Phi { .. } => panic!("phis are expanded"),
            })
            .collect();
        assert_eq!(
            actual.len(),
            expected.len(),
            "no definition is reported twice, round {round}"
        );
        assert_eq!(
            &actual.into_iter().collect::<BTreeSet<_>>(),
            expected,
            "reads at {read:?}, round {round}"
        );
    }
}

/// Each write replaces exactly the definitions the exploration finds, and its
/// transitive readers are those the exploration finds.
fn assert_clobbers_match_exploration(memory: &Memory, reaching: &Reaching<u8>, round: usize) {
    for (write, expected) in &reaching.clobbers {
        let version = memory.event(*write).unwrap().clobbered_version().unwrap();
        assert_eq!(
            &expand(memory, version),
            expected,
            "clobbers at {write:?}, round {round}"
        );
        let replaced = memory
            .clobbered_definition(*write)
            .expect("a write has a clobbered state");
        assert_eq!(memory.definition(version), Some(replaced));
        assert_eq!(
            memory.transitive_readers(*write),
            reaching
                .transitive_readers(*write)
                .into_iter()
                .collect::<Vec<_>>(),
            "transitive readers of {write:?}, round {round}"
        );
    }
}

/// The users of a version are exactly the reads of it and the phi operands
/// that name it; a write's replaced version is not a use.
fn assert_users_invert_versions(memory: &Memory) {
    let mut expected: BTreeMap<_, Vec<MemoryUse>> = BTreeMap::new();
    for event in memory.events() {
        if let Some(version) = event.read_version() {
            expected
                .entry(version.clone())
                .or_default()
                .push(MemoryUse::Event { site: event.site() });
        }
    }
    for phi in memory.phis() {
        for (predecessor, operand) in phi.operands() {
            expected
                .entry(operand.clone())
                .or_default()
                .push(MemoryUse::Phi {
                    block: phi.block(),
                    predecessor: *predecessor,
                    result: phi.result().clone(),
                });
        }
    }
    for (version, uses) in &expected {
        let mut actual = memory.users(version).to_vec();
        let mut wanted = uses.clone();
        let key = |memory_use: &MemoryUse| format!("{memory_use:?}");
        actual.sort_by_key(key);
        wanted.sort_by_key(key);
        assert_eq!(actual, wanted, "users of {version:?}");
    }
    for event in memory.events() {
        if let Some(version) = event.written_version() {
            assert_eq!(
                memory.definition(version),
                Some(&MemoryDefinition::Event { site: event.site() })
            );
            if !expected.contains_key(version) {
                assert_eq!(memory.users(version), []);
            }
        }
    }
}
