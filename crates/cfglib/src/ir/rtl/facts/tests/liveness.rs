//! Whole-root liveness.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::toy::{Block, Function, Root, Toy, roots_in};
use crate::ir::rtl::{BlockSummaries, Control, RootLiveness, RootTable};

/// The blocks of the loop that the tests share.
struct Loop {
    entry: Block,
    header: Block,
    body: Block,
    exit: Block,
}

/// Builds `entry → header → body → header`, with `header → exit`.
///
/// - `entry` overwrites `Acc` and `Tmp`.
/// - `header` reads `Acc` and `Flag`.
/// - `body` reads `Tmp` and overwrites `Acc`.
fn counted_loop(function: &mut Function) -> Loop {
    let entry = function.block();
    let header = function.block();
    let body = function.block();
    let exit = function.block();
    function.flow(entry, &[header]);
    function.flow(header, &[body, exit]);
    function.flow(body, &[header]);

    let mut record = function.record_in(entry);
    let value = record.constant_value(1, 64);
    record.write_whole(Root::Acc, value);
    record.write_whole(Root::Tmp, value);
    record.end(Control::Advance);

    let mut record = function.record_in(header);
    record.read_whole(Root::Acc);
    record.read_whole(Root::Flag);
    record.end(Control::Advance);

    let mut record = function.record_in(body);
    record.read_whole(Root::Tmp);
    let value = record.constant_value(2, 64);
    record.write_whole(Root::Acc, value);
    record.end(Control::Advance);

    let mut record = function.record_in(exit);
    record.read_whole(Root::Tmp);
    record.end(Control::Advance);

    Loop {
        entry,
        header,
        body,
        exit,
    }
}

fn facts(function: &Function) -> (RootTable<Toy>, BlockSummaries) {
    let table = RootTable::collect(function);
    let summaries = BlockSummaries::compute(function, &table);
    (table, summaries)
}

fn live(table: &RootTable<Toy>, set: &crate::DenseBits) -> Vec<Root> {
    roots_in(table, set)
}

#[test]
fn the_default_seed_keeps_every_root_live_after_the_exit() {
    let mut function = Function::new();
    let shape = counted_loop(&mut function);
    let (table, summaries) = facts(&function);
    let liveness = RootLiveness::compute(&function, &table, &summaries);

    let all = vec![Root::Acc, Root::Tmp, Root::Flag];
    assert_eq!(live(&table, liveness.live_out(shape.exit)), all);
    assert_eq!(live(&table, liveness.live_in(shape.exit)), all);
    assert_eq!(live(&table, liveness.live_in(shape.header)), all);
    // The entry block overwrites `Acc` and `Tmp`, so only `Flag` comes from
    // the caller.
    assert_eq!(live(&table, liveness.live_in(shape.entry)), [Root::Flag]);
}

#[test]
fn a_seed_states_which_roots_the_caller_reads_after_the_exit() {
    let mut function = Function::new();
    let shape = counted_loop(&mut function);
    let (table, summaries) = facts(&function);
    let liveness = RootLiveness::compute_seeded(&function, &table, &summaries, |_, seed| {
        assert!(seed.insert(Root::Acc));
        assert!(!seed.insert(Root::Acc), "a second insert changes nothing");
        assert!(!seed.insert(Root::Top), "an unnamed root is ignored");
    });

    assert_eq!(live(&table, liveness.live_out(shape.exit)), [Root::Acc]);
    // The exit block reads `Tmp`, so `Tmp` is live at its start.
    assert_eq!(
        live(&table, liveness.live_in(shape.exit)),
        [Root::Acc, Root::Tmp]
    );
    // The body overwrites `Acc`, and the loop keeps `Tmp` and `Flag` alive.
    assert_eq!(
        live(&table, liveness.live_in(shape.body)),
        [Root::Tmp, Root::Flag]
    );
    assert_eq!(
        live(&table, liveness.live_out(shape.header)),
        [Root::Acc, Root::Tmp, Root::Flag]
    );
    assert_eq!(live(&table, liveness.live_in(shape.entry)), [Root::Flag]);
}

#[test]
fn a_kill_ends_the_liveness_of_a_root_above_it() {
    let mut function = Function::new();
    let first = function.block();
    let second = function.block();
    function.flow(first, &[second]);
    let mut record = function.record_in(first);
    let value = record.constant_value(1, 64);
    record.write_whole(Root::Acc, value);
    record.end(Control::Advance);
    let mut record = function.record_in(second);
    record.read_whole(Root::Acc);
    record.end(Control::Advance);

    let (table, summaries) = facts(&function);
    let liveness = RootLiveness::compute_seeded(&function, &table, &summaries, |_, _| {});
    assert_eq!(live(&table, liveness.live_in(second)), [Root::Acc]);
    assert_eq!(live(&table, liveness.live_out(first)), [Root::Acc]);
    assert_eq!(live(&table, liveness.live_in(first)), []);
}

#[test]
fn a_partial_write_keeps_its_root_live_above_it() {
    let mut function = Function::new();
    let first = function.block();
    let second = function.block();
    function.flow(first, &[second]);
    let mut record = function.record_in(first);
    let value = record.constant_value(1, 8);
    record.write_slice(
        Root::Acc,
        crate::ir::rtl::BitSlice::new(0, crate::ir::rtl::Bits::new(8).unwrap()),
        value,
    );
    record.end(Control::Advance);
    let mut record = function.record_in(second);
    record.read_whole(Root::Acc);
    record.end(Control::Advance);

    let (table, summaries) = facts(&function);
    let liveness = RootLiveness::compute_seeded(&function, &table, &summaries, |_, _| {});
    assert_eq!(live(&table, liveness.live_in(first)), [Root::Acc]);
}

#[test]
fn each_exit_block_gets_its_own_seed() {
    let mut function = Function::new();
    let entry = function.block();
    let left = function.block();
    let right = function.block();
    function.flow(entry, &[left, right]);
    let mut record = function.record_in(entry);
    record.read_whole(Root::Acc);
    record.read_whole(Root::Tmp);
    record.end(Control::Advance);
    for exit in [left, right] {
        let mut record = function.record_in(exit);
        record.read_whole(Root::Flag);
        record.end(Control::Advance);
    }

    let (table, summaries) = facts(&function);
    let mut seeded = Vec::new();
    let liveness = RootLiveness::compute_seeded(&function, &table, &summaries, |block, seed| {
        seeded.push(block);
        seed.insert(if block == left { Root::Acc } else { Root::Tmp });
    });

    assert_eq!(seeded, [left, right]);
    assert_eq!(live(&table, liveness.live_out(left)), [Root::Acc]);
    assert_eq!(live(&table, liveness.live_out(right)), [Root::Tmp]);
    assert_eq!(
        live(&table, liveness.live_out(entry)),
        [Root::Acc, Root::Tmp, Root::Flag]
    );
}
