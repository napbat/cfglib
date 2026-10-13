//! Selection resolution.

use super::toy::{Block, Effect, Function, Inst, Operator, Root, SLOTS, Toy, whole_places};
use crate::ir::rtl::{
    BitSlice, Bits, Control, InstructionBuilder, Raise, ResolvedSelection, RootConstant, RootTable,
    Selection, SelectionResolution, WriteTarget,
};

fn resolve(function: &Function, entry: &[RootConstant<'_, Toy>]) -> SelectionResolution<Inst> {
    let table = RootTable::collect(function);
    SelectionResolution::compute(function, &table, entry)
}

fn top(value: &[u64]) -> RootConstant<'_, Toy> {
    RootConstant {
        root: Root::Top,
        words: value,
    }
}

fn three_bits() -> Bits {
    Bits::new(3).unwrap()
}

/// Reads `Top` and uses it as the index of a read over the stack slots. The
/// read is the second statement of the record.
fn read_slot_at_top(function: &mut Function, block: Block) -> Inst {
    let places = whole_places(&SLOTS);
    let mut record = function.record_in(block);
    let index = record.read_whole(Root::Top);
    record.read(Selection::FiniteIndexed {
        index,
        candidates: &places,
    });
    record.end(Control::Advance)
}

/// Overwrites `Top` with a constant.
fn set_top(function: &mut Function, block: Block, value: u64) {
    let mut record = function.record_in(block);
    let constant = record.constant_value(value, 3);
    record.write_whole(Root::Top, constant);
    record.end(Control::Advance);
}

#[test]
fn a_constant_index_in_one_record_resolves() {
    let mut function = Function::new();
    let entry = function.block();
    let places = whole_places(&SLOTS);
    let mut record = function.record_in(entry);
    let index = record.constant_value(2, 3);
    record.read(Selection::FiniteIndexed {
        index,
        candidates: &places,
    });
    let inst = record.end(Control::Advance);

    let resolution = resolve(&function, &[]);
    assert_eq!(resolution.candidate(inst, 1), Some(2));
    assert_eq!(resolution.candidate(inst, 0), None);
    assert_eq!(
        resolution.resolved(),
        [ResolvedSelection {
            inst,
            statement: 1,
            candidate: 2
        }]
    );
}

#[test]
fn an_entry_constant_and_an_operator_resolve_across_blocks() {
    let mut function = Function::new();
    let first = function.block();
    let second = function.block();
    let third = function.block();
    function.flow(first, &[second]);
    function.flow(second, &[third]);
    let places = whole_places(&SLOTS);

    let first_inst = read_slot_at_top(&mut function, first);

    // `Top` becomes `Top + 1`, which the fold hook of the dialect evaluates.
    let mut record = function.record_in(second);
    let current = record.read_whole(Root::Top);
    let one = record.constant_value(1, 3);
    let next = record.apply(&Operator::Add, &[current, one], &[three_bits()]);
    record.write_whole(Root::Top, next.first);
    record.end(Control::Advance);

    // The indexed write of the third block uses the incremented `Top`.
    let mut record = function.record_in(third);
    let index = record.read_whole(Root::Top);
    let value = record.constant_value(7, 64);
    record.write(
        WriteTarget::FiniteIndexed {
            index,
            candidates: &places,
            value,
        },
        (),
    );
    let third_inst = record.end(Control::Advance);

    let resolution = resolve(&function, &[top(&[1])]);
    assert_eq!(resolution.candidate(first_inst, 1), Some(1));
    assert_eq!(resolution.candidate(third_inst, 2), Some(2));
}

#[test]
fn the_and_operator_folds_too() {
    let mut function = Function::new();
    let entry = function.block();
    let places = whole_places(&SLOTS);
    let mut record = function.record_in(entry);
    let current = record.read_whole(Root::Top);
    let mask = record.constant_value(1, 3);
    let masked = record.apply(&Operator::And, &[current, mask], &[three_bits()]);
    record.read(Selection::FiniteIndexed {
        index: masked.first,
        candidates: &places,
    });
    let inst = record.end(Control::Advance);

    assert_eq!(resolve(&function, &[top(&[3])]).candidate(inst, 3), Some(1));
}

/// Builds `entry → left, right → join` where each arm sets `Top`, and the
/// join reads a slot at `Top`.
fn diamond_with_tops(left: u64, right: u64) -> (Function, Inst) {
    let mut function = Function::new();
    let entry = function.block();
    let left_block = function.block();
    let right_block = function.block();
    let join = function.block();
    function.flow(entry, &[left_block, right_block]);
    function.flow(left_block, &[join]);
    function.flow(right_block, &[join]);
    set_top(&mut function, left_block, left);
    set_top(&mut function, right_block, right);
    let inst = read_slot_at_top(&mut function, join);
    (function, inst)
}

#[test]
fn a_join_keeps_a_constant_that_every_path_agrees_on() {
    let (function, inst) = diamond_with_tops(3, 3);
    assert_eq!(resolve(&function, &[]).candidate(inst, 1), Some(3));
}

#[test]
fn a_join_drops_a_constant_that_the_paths_disagree_on() {
    let (function, inst) = diamond_with_tops(1, 2);
    assert_eq!(resolve(&function, &[]).candidate(inst, 1), None);
}

/// Builds `entry → header → body → header`, with `header → exit`. The entry
/// sets `Top` to zero. The body increments `Top` when `counting` is set.
fn loop_over_top(counting: bool) -> (Function, Inst) {
    let mut function = Function::new();
    let entry = function.block();
    let header = function.block();
    let body = function.block();
    let exit = function.block();
    function.flow(entry, &[header]);
    function.flow(header, &[body, exit]);
    function.flow(body, &[header]);
    set_top(&mut function, entry, 0);
    let inst = read_slot_at_top(&mut function, header);

    let mut record = function.record_in(body);
    if counting {
        let current = record.read_whole(Root::Top);
        let one = record.constant_value(1, 3);
        let next = record.apply(&Operator::Add, &[current, one], &[three_bits()]);
        record.write_whole(Root::Top, next.first);
    } else {
        let value = record.constant_value(1, 64);
        record.write_whole(Root::Acc, value);
    }
    record.end(Control::Advance);
    (function, inst)
}

#[test]
fn a_loop_that_leaves_the_root_alone_keeps_its_constant() {
    let (function, inst) = loop_over_top(false);
    assert_eq!(resolve(&function, &[]).candidate(inst, 1), Some(0));
}

#[test]
fn a_loop_that_changes_the_root_loses_its_constant() {
    let (function, inst) = loop_over_top(true);
    assert_eq!(resolve(&function, &[]).candidate(inst, 1), None);
}

/// How the first block of `top_after` ends.
#[derive(Clone, Copy)]
enum Ending {
    Advance,
    Call,
    Dispatch,
}

/// Builds `entry → next`, where `entry` sets `Top` and then ends with
/// `ending`, and `next` reads a slot at `Top`. Returns the candidate that the
/// read of `next` picks.
fn top_after(ending: Ending) -> Option<usize> {
    let mut function = Function::new();
    let entry = function.block();
    let next = function.block();
    function.flow(entry, &[next]);
    set_top(&mut function, entry, 1);
    let mut record = function.record_in(entry);
    let operand = record.constant_value(0, 64);
    let control = match ending {
        Ending::Advance => Control::Advance,
        Ending::Call => Control::Call { target: operand },
        Ending::Dispatch => Control::Dispatch { scrutinee: operand },
    };
    record.end(control);
    let inst = read_slot_at_top(&mut function, next);
    resolve(&function, &[]).candidate(inst, 1)
}

#[test]
fn a_call_makes_every_root_unknown() {
    assert_eq!(top_after(Ending::Advance), Some(1));
    assert_eq!(top_after(Ending::Call), None);
}

#[test]
fn a_dispatch_stays_inside_the_function_and_keeps_the_constants() {
    assert_eq!(top_after(Ending::Dispatch), Some(1));
}

#[test]
fn a_partial_write_keeps_the_bits_it_preserves() {
    let mut function = Function::new();
    let entry = function.block();
    set_top(&mut function, entry, 2);
    let mut record = function.record_in(entry);
    let bit = record.constant_value(1, 1);
    record.write_slice(Root::Top, BitSlice::new(0, Bits::BOOL), bit);
    record.end(Control::Advance);
    let inst = read_slot_at_top(&mut function, entry);

    // 0b010 with bit 0 set is 0b011.
    assert_eq!(resolve(&function, &[]).candidate(inst, 1), Some(3));
}

#[test]
fn a_partial_write_of_an_unknown_root_leaves_it_unknown() {
    let mut function = Function::new();
    let entry = function.block();
    let mut record = function.record_in(entry);
    let bit = record.constant_value(1, 1);
    record.write_slice(Root::Top, BitSlice::new(0, Bits::BOOL), bit);
    record.end(Control::Advance);
    let inst = read_slot_at_top(&mut function, entry);

    assert_eq!(resolve(&function, &[]).candidate(inst, 1), None);
}

#[test]
fn an_indexed_write_with_a_constant_index_changes_one_candidate() {
    let mut function = Function::new();
    let entry = function.block();
    let places = whole_places(&SLOTS);
    let mut record = function.record_in(entry);
    let index = record.constant_value(2, 3);
    let value = record.constant_value(1, 64);
    record.write(
        WriteTarget::FiniteIndexed {
            index,
            candidates: &places,
            value,
        },
        (),
    );
    record.end(Control::Advance);

    // `Slot2` is now one. `Slot0` is still unknown.
    let mut record = function.record_in(entry);
    let from_slot = record.read_whole(Root::Slot2);
    record.read(Selection::FiniteIndexed {
        index: from_slot,
        candidates: &places,
    });
    let known = record.end(Control::Advance);
    let mut record = function.record_in(entry);
    let from_slot = record.read_whole(Root::Slot0);
    record.read(Selection::FiniteIndexed {
        index: from_slot,
        candidates: &places,
    });
    let unknown = record.end(Control::Advance);

    // The first record has the write at statement 2 with a constant index.
    let resolution = resolve(&function, &[]);
    assert_eq!(resolution.candidate(function.inst(entry, 0), 2), Some(2));
    assert_eq!(resolution.candidate(known, 1), Some(1));
    assert_eq!(resolution.candidate(unknown, 1), None);
}

#[test]
fn an_indexed_write_with_an_unknown_index_forgets_every_candidate() {
    let mut function = Function::new();
    let entry = function.block();
    let places = whole_places(&SLOTS);
    let mut record = function.record_in(entry);
    let value = record.constant_value(1, 64);
    record.write_whole(Root::Slot2, value);
    record.end(Control::Advance);
    let mut record = function.record_in(entry);
    let index = record.nondet(three_bits(), ());
    let value = record.constant_value(5, 64);
    record.write(
        WriteTarget::FiniteIndexed {
            index,
            candidates: &places,
            value,
        },
        (),
    );
    record.end(Control::Advance);
    let mut record = function.record_in(entry);
    let from_slot = record.read_whole(Root::Slot2);
    record.read(Selection::FiniteIndexed {
        index: from_slot,
        candidates: &places,
    });
    let inst = record.end(Control::Advance);

    assert_eq!(resolve(&function, &[]).candidate(inst, 1), None);
}

/// Builds a record that reads `Bank` as the index of a read over the slots,
/// after a sparse write to `Bank` when `write` is set. Returns the candidate
/// that the read picks when `Bank` starts as one.
fn bank_after(write: bool) -> Option<usize> {
    let mut function = Function::new();
    let entry = function.block();
    let places = whole_places(&SLOTS);
    if write {
        let mut record = function.record_in(entry);
        let key = record.constant_value(0x10, 32);
        let value = record.constant_value(1, 64);
        record.write(
            WriteTarget::SparseIndexed {
                bank: Root::Bank,
                key,
                slice: BitSlice::whole(Bits::new(64).unwrap()),
                value,
            },
            (),
        );
        record.end(Control::Advance);
    }
    let mut record = function.record_in(entry);
    let from_bank = record.read_whole(Root::Bank);
    record.read(Selection::FiniteIndexed {
        index: from_bank,
        candidates: &places,
    });
    let inst = record.end(Control::Advance);

    let bank = RootConstant {
        root: Root::Bank,
        words: &[1],
    };
    resolve(&function, &[bank]).candidate(inst, 1)
}

#[test]
fn a_sparse_write_forgets_its_bank() {
    assert_eq!(bank_after(false), Some(1));
    assert_eq!(bank_after(true), None);
}

#[test]
fn a_constant_outside_the_candidates_is_not_resolved() {
    let mut function = Function::new();
    let entry = function.block();
    let places = whole_places(&SLOTS);
    let mut record = function.record_in(entry);
    let index = record.constant_value(5, 3);
    record.read(Selection::FiniteIndexed {
        index,
        candidates: &places,
    });
    record.end(Control::Advance);

    assert_eq!(resolve(&function, &[]).resolved(), []);
}

#[test]
fn effects_nondeterminism_and_raises_give_no_constants() {
    let mut function = Function::new();
    let entry = function.block();
    let places = whole_places(&SLOTS);
    let mut record = function.record_in(entry);
    let address = record.constant_value(0x40, 64);
    let loaded = record.effect(&Effect::Load, &[address], &[three_bits()], &[]);
    let chosen = record.nondet(three_bits(), ());
    let guard = record.constant_value(0, 1);
    record.raise(Raise {
        outcome: (),
        guard: Some(guard),
        payload: &[],
        binds: &[],
    });
    for index in [loaded.first, chosen] {
        record.read(Selection::FiniteIndexed {
            index,
            candidates: &places,
        });
    }
    record.end(Control::Advance);

    assert_eq!(resolve(&function, &[]).resolved(), []);
}

#[test]
fn an_unreachable_block_is_not_reported() {
    let mut function = Function::new();
    let entry = function.block();
    let orphan = function.block();
    let places = whole_places(&SLOTS);
    let mut record = function.record_in(orphan);
    let index = record.constant_value(1, 3);
    record.read(Selection::FiniteIndexed {
        index,
        candidates: &places,
    });
    record.end(Control::Advance);
    let mut record = function.record_in(entry);
    record.read_whole(Root::Acc);
    record.end(Control::Advance);

    assert_eq!(resolve(&function, &[]).resolved(), []);
}

#[test]
fn entry_constants_name_roots_by_value_and_ignore_unnamed_roots() {
    let mut function = Function::new();
    let entry = function.block();
    let inst = read_slot_at_top(&mut function, entry);

    // `Flag` is not a root of the function. The value of `Top` has a bit past
    // the three bits of the root, and the analysis drops it.
    let flag = RootConstant {
        root: Root::Flag,
        words: &[1],
    };
    let resolution = resolve(&function, &[flag, top(&[0b1001])]);
    assert_eq!(resolution.candidate(inst, 1), Some(1));
}
