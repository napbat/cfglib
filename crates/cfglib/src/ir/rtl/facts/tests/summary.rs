//! The root table and the per-block summaries.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::toy::{Function, Root, SLOTS, Toy, roots_in, whole_places};
use crate::ir::rtl::{
    BitSlice, Bits, BlockSummaries, Control, InstructionBuilder, Place, RootTable, Selection,
    WriteTarget,
};

fn summarize(function: &Function) -> (RootTable<Toy>, BlockSummaries) {
    let table = RootTable::collect(function);
    let summaries = BlockSummaries::compute(function, &table);
    (table, summaries)
}

/// The roots of one block as `(reads before write, writes, kills)`.
fn sets(function: &Function, block: super::toy::Block) -> (Vec<Root>, Vec<Root>, Vec<Root>) {
    let (table, summaries) = summarize(function);
    let summary = summaries.block(block);
    (
        roots_in(&table, summary.reads_before_write()),
        roots_in(&table, summary.writes()),
        roots_in(&table, summary.kills()),
    )
}

#[test]
fn the_table_names_every_selected_written_and_candidate_root_once() {
    let mut function = Function::new();
    let entry = function.block();
    let places = whole_places(&SLOTS);
    let mut record = function.record_in(entry);
    let index = record.constant_value(1, 3);
    record.read_whole(Root::Acc);
    record.read_whole(Root::Acc);
    record.read(Selection::FiniteIndexed {
        index,
        candidates: &places,
    });
    record.read(Selection::SparseIndexed {
        bank: Root::Bank,
        key: index,
        slice: BitSlice::whole(Bits::new(64).unwrap()),
    });
    let flag = record.constant_value(1, 1);
    record.write_whole(Root::Flag, flag);
    record.end(Control::Advance);

    let table = RootTable::collect(&function);
    assert_eq!(
        table.roots(),
        [
            Root::Acc,
            Root::Flag,
            Root::Slot0,
            Root::Slot1,
            Root::Slot2,
            Root::Slot3,
            Root::Bank
        ]
    );
    assert_eq!(table.len(), 7);
    assert_eq!(table.index_of(Root::Flag), Some(1));
    assert_eq!(table.index_of(Root::Top), None);
    assert_eq!(table.root(6), Root::Bank);
}

#[test]
fn a_whole_write_kills_its_root() {
    let mut function = Function::new();
    let entry = function.block();
    let mut record = function.record_in(entry);
    let value = record.constant_value(1, 64);
    record.write_whole(Root::Acc, value);
    record.end(Control::Advance);

    let (reads, writes, kills) = sets(&function, entry);
    assert_eq!(reads, []);
    assert_eq!(writes, [Root::Acc]);
    assert_eq!(kills, [Root::Acc]);
}

#[test]
fn a_partial_write_reads_and_writes_its_root_and_does_not_kill_it() {
    let mut function = Function::new();
    let entry = function.block();
    let mut record = function.record_in(entry);
    let value = record.constant_value(1, 8);
    record.write_slice(Root::Acc, BitSlice::new(0, Bits::new(8).unwrap()), value);
    record.end(Control::Advance);

    let (reads, writes, kills) = sets(&function, entry);
    assert_eq!(reads, [Root::Acc]);
    assert_eq!(writes, [Root::Acc]);
    assert_eq!(kills, []);
}

#[test]
fn a_read_after_a_kill_is_not_upward_exposed() {
    let mut function = Function::new();
    let entry = function.block();
    let mut first = function.record_in(entry);
    let value = first.constant_value(1, 64);
    first.write_whole(Root::Acc, value);
    first.end(Control::Advance);
    let mut second = function.record_in(entry);
    let read = second.read_whole(Root::Acc);
    second.write_whole(Root::Tmp, read);
    second.end(Control::Advance);

    let (reads, writes, kills) = sets(&function, entry);
    assert_eq!(reads, []);
    assert_eq!(writes, [Root::Acc, Root::Tmp]);
    assert_eq!(kills, [Root::Acc, Root::Tmp]);
}

#[test]
fn every_read_of_one_record_sees_the_state_before_its_writes() {
    let mut function = Function::new();
    let entry = function.block();
    let mut record = function.record_in(entry);
    let value = record.constant_value(1, 64);
    record.write_whole(Root::Acc, value);
    let read = record.read_whole(Root::Acc);
    record.write_whole(Root::Tmp, read);
    record.end(Control::Advance);

    let (reads, _, kills) = sets(&function, entry);
    assert_eq!(reads, [Root::Acc]);
    assert_eq!(kills, [Root::Acc, Root::Tmp]);
}

#[test]
fn an_indexed_write_is_a_may_write_of_every_candidate() {
    let mut function = Function::new();
    let entry = function.block();
    let places = whole_places(&SLOTS[..2]);
    let mut record = function.record_in(entry);
    let index = record.constant_value(0, 3);
    let value = record.constant_value(9, 64);
    record.write(
        WriteTarget::FiniteIndexed {
            index,
            candidates: &places,
            value,
        },
        (),
    );
    record.end(Control::Advance);

    let (reads, writes, kills) = sets(&function, entry);
    assert_eq!(reads, [Root::Slot0, Root::Slot1]);
    assert_eq!(writes, [Root::Slot0, Root::Slot1]);
    assert_eq!(kills, []);
}

#[test]
fn an_indexed_write_does_not_expose_a_candidate_that_an_earlier_write_killed() {
    let mut function = Function::new();
    let entry = function.block();
    let places = whole_places(&SLOTS[..2]);
    let mut first = function.record_in(entry);
    let value = first.constant_value(9, 64);
    first.write_whole(Root::Slot0, value);
    first.end(Control::Advance);
    let mut second = function.record_in(entry);
    let index = second.constant_value(1, 3);
    let value = second.constant_value(5, 64);
    second.write(
        WriteTarget::FiniteIndexed {
            index,
            candidates: &places,
            value,
        },
        (),
    );
    second.end(Control::Advance);

    let (reads, writes, kills) = sets(&function, entry);
    assert_eq!(reads, [Root::Slot1]);
    assert_eq!(writes, [Root::Slot0, Root::Slot1]);
    assert_eq!(kills, [Root::Slot0]);
}

#[test]
fn a_sparse_write_is_a_may_write_of_its_bank() {
    let mut function = Function::new();
    let entry = function.block();
    let mut record = function.record_in(entry);
    let key = record.constant_value(0x1b, 32);
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

    let (reads, writes, kills) = sets(&function, entry);
    assert_eq!(reads, [Root::Bank]);
    assert_eq!(writes, [Root::Bank]);
    assert_eq!(kills, []);
}

#[test]
fn a_block_without_instructions_has_an_empty_summary() {
    let mut function = Function::new();
    let entry = function.block();
    let empty = function.block();
    function.flow(entry, &[empty]);
    let mut record = function.record_in(entry);
    record.read(Selection::Direct(Place::whole(Root::Acc)));
    record.end(Control::Advance);

    let (reads, writes, kills) = sets(&function, empty);
    assert_eq!((reads, writes, kills), (vec![], vec![], vec![]));
}
