//! `BlockSummaries`: what each block reads, writes, and overwrites.
//!
//! The hand-derived tests state each rule of the documentation on a one-block
//! Kestrel function. `assert_summaries_match` compares the summaries of any
//! function with the per-root scan of the oracle.

use std::collections::BTreeSet;
use std::panic::{AssertUnwindSafe, catch_unwind};

use cfglib::DenseId;
use cfglib::ir::rtl::{
    Action, BitSlice, BlockSummaries, Control, Dialect, FunctionView, InstructionBuilder, Place,
    Raise, RootTable, Selection, SpanAction, WriteTarget,
};

use crate::bits;
use crate::kestrel::{Fault, Kestrel, Policy, Reg, Undef};
use crate::kestrel_store::{BlockRef, KestrelBuilder, KestrelFunction, ValueRef};
use crate::model::{Model, Target};
use crate::oracle;
use crate::recipe::neutral_set;

/// Compares every live block of `function` with the oracle.
pub fn assert_summaries_match<D: Target, F: FunctionView<D>>(
    function: &F,
    roots: &RootTable<D>,
    summaries: &BlockSummaries,
    model: &Model,
    label: &str,
) {
    for (index, _) in model.live() {
        let expected = oracle::summary(model, index);
        let actual = summaries.block(F::NodeId::from_index(index));
        assert_eq!(
            neutral_set(roots, actual.reads_before_write()),
            expected.reads_before_write,
            "{label}: reads_before_write of block {index}"
        );
        assert_eq!(
            neutral_set(roots, actual.writes()),
            expected.writes,
            "{label}: writes of block {index}"
        );
        assert_eq!(
            neutral_set(roots, actual.kills()),
            expected.kills,
            "{label}: kills of block {index}"
        );
    }
    // A removed slot below the bound has an empty summary.
    for index in 0..function.node_bound() {
        if !model.is_live(index) {
            let actual = summaries.block(F::NodeId::from_index(index));
            assert_eq!(actual.writes().count_ones(), 0, "{label}: removed {index}");
            assert_eq!(actual.kills().count_ones(), 0, "{label}: removed {index}");
            assert_eq!(
                actual.reads_before_write().count_ones(),
                0,
                "{label}: removed {index}"
            );
        }
    }
}

fn full_write(builder: &mut KestrelBuilder<'_>, root: Reg) {
    let width = Kestrel::root_width(root);
    let value = builder.constant(&vec![0; width.words()], width);
    let plan = [SpanAction {
        span: BitSlice::whole(width),
        action: Action::TakeInput { value, offset: 0 },
    }];
    builder.write(WriteTarget::Root { root, plan: &plan }, Policy::Zeroing);
}

/// Replaces the low `low` bits of `root` and keeps the rest.
fn partial_write(builder: &mut KestrelBuilder<'_>, root: Reg, low: u16) {
    let width = Kestrel::root_width(root).width();
    let value = builder.constant(&[0], bits(low));
    let plan = [
        SpanAction {
            span: BitSlice::new(0, bits(low)),
            action: Action::TakeInput { value, offset: 0 },
        },
        SpanAction {
            span: BitSlice::new(low, bits(width - low)),
            action: Action::PreservePreState,
        },
    ];
    builder.write(WriteTarget::Root { root, plan: &plan }, Policy::Merging);
}

fn read(builder: &mut KestrelBuilder<'_>, root: Reg) {
    builder.read(Selection::Direct(Place::whole(root)));
}

/// The roots that the entry block reads before writing, writes, and kills.
fn entry_sets(function: &KestrelFunction) -> [BTreeSet<Reg>; 3] {
    let roots = RootTable::collect(function);
    let summaries = BlockSummaries::compute(function, &roots);
    let summary = summaries.block(function.entry());
    [
        summary.reads_before_write(),
        summary.writes(),
        summary.kills(),
    ]
    .map(|row| row.ones().map(|index| roots.root(index)).collect())
}

fn set<const N: usize>(roots: [Reg; N]) -> BTreeSet<Reg> {
    BTreeSet::from(roots)
}

type Record = Box<dyn FnOnce(&mut KestrelBuilder<'_>) -> Control<ValueRef>>;

/// A record whose statements `body` emits and that ends by advancing.
fn record(body: impl FnOnce(&mut KestrelBuilder<'_>) + 'static) -> Record {
    Box::new(move |builder| {
        body(builder);
        Control::Advance
    })
}

/// A function whose entry block holds one sealed record per closure.
fn entry_records(records: Vec<Record>) -> KestrelFunction {
    let mut function = KestrelFunction::new();
    let entry = function.entry();
    for record in records {
        function
            .append(entry, |mut builder| {
                let control = record(&mut builder);
                builder.finish(control)
            })
            .expect("the record seals");
    }
    function
}

#[test]
fn a_complete_write_kills_its_root_and_exposes_nothing() {
    let function = entry_records(vec![record(|b| full_write(b, Reg::R0))]);
    let [exposed, writes, kills] = entry_sets(&function);
    assert!(exposed.is_empty());
    assert_eq!(writes, set([Reg::R0]));
    assert_eq!(kills, set([Reg::R0]));
}

#[test]
fn every_read_of_one_record_sees_the_state_before_its_writes() {
    let function = entry_records(vec![record(|b| {
        read(b, Reg::R0);
        full_write(b, Reg::R0);
    })]);
    let [exposed, writes, kills] = entry_sets(&function);
    assert_eq!(exposed, set([Reg::R0]));
    assert_eq!(writes, set([Reg::R0]));
    assert_eq!(kills, set([Reg::R0]));
}

#[test]
fn a_read_after_a_killing_record_is_not_exposed() {
    let function = entry_records(vec![
        record(|b| full_write(b, Reg::R0)),
        record(|b| read(b, Reg::R0)),
    ]);
    let [exposed, _, kills] = entry_sets(&function);
    assert!(exposed.is_empty());
    assert_eq!(kills, set([Reg::R0]));
}

#[test]
fn a_partial_write_reads_and_writes_its_root_and_never_kills() {
    let function = entry_records(vec![record(|b| partial_write(b, Reg::R1, 8))]);
    let [exposed, writes, kills] = entry_sets(&function);
    assert_eq!(exposed, set([Reg::R1]));
    assert_eq!(writes, set([Reg::R1]));
    assert!(kills.is_empty());
}

#[test]
fn a_partial_write_after_a_kill_is_not_exposed() {
    let function = entry_records(vec![
        record(|b| full_write(b, Reg::R1)),
        record(|b| partial_write(b, Reg::R1, 16)),
    ]);
    let [exposed, writes, kills] = entry_sets(&function);
    assert!(exposed.is_empty());
    assert_eq!(writes, set([Reg::R1]));
    assert_eq!(kills, set([Reg::R1]));
}

#[test]
fn a_plan_that_keeps_nothing_kills_even_with_a_nondeterministic_span() {
    let function = entry_records(vec![record(|b| {
        let value = b.constant(&[0], bits(16));
        let plan = [
            SpanAction {
                span: BitSlice::new(0, bits(16)),
                action: Action::TakeInput { value, offset: 0 },
            },
            SpanAction {
                span: BitSlice::new(16, bits(16)),
                action: Action::Nondeterministic(Undef::Scratch),
            },
        ];
        b.write(
            WriteTarget::Root {
                root: Reg::R2,
                plan: &plan,
            },
            Policy::Zeroing,
        );
    })]);
    let [exposed, writes, kills] = entry_sets(&function);
    assert!(exposed.is_empty());
    assert_eq!(writes, set([Reg::R2]));
    assert_eq!(kills, set([Reg::R2]));
}

#[test]
fn an_indexed_write_may_write_every_candidate_and_kills_none() {
    let function = entry_records(vec![record(|b| {
        let index = b.constant(&[1], bits(8));
        let value = b.constant(&[0], bits(32));
        let candidates = [Place::whole(Reg::R0), Place::whole(Reg::R1)];
        b.write(
            WriteTarget::FiniteIndexed {
                index,
                candidates: &candidates,
                value,
            },
            Policy::Merging,
        );
    })]);
    let [exposed, writes, kills] = entry_sets(&function);
    assert_eq!(exposed, set([Reg::R0, Reg::R1]));
    assert_eq!(writes, set([Reg::R0, Reg::R1]));
    assert!(kills.is_empty());
}

#[test]
fn a_sparse_write_touches_only_the_bank_root() {
    let function = entry_records(vec![record(|b| {
        let key = b.constant(&[0x1b], bits(16));
        let value = b.constant(&[0], bits(32));
        b.write(
            WriteTarget::SparseIndexed {
                bank: Reg::Csr,
                key,
                slice: BitSlice::whole(bits(32)),
                value,
            },
            Policy::Merging,
        );
    })]);
    let [exposed, writes, kills] = entry_sets(&function);
    assert_eq!(exposed, set([Reg::Csr]));
    assert_eq!(writes, set([Reg::Csr]));
    assert!(kills.is_empty());
}

#[test]
fn indexed_and_sparse_reads_expose_every_root_they_may_read() {
    let function = entry_records(vec![record(|b| {
        let index = b.constant(&[1], bits(8));
        let candidates = [Place::whole(Reg::R2), Place::whole(Reg::R3)];
        b.read(Selection::FiniteIndexed {
            index,
            candidates: &candidates,
        });
        let key = b.constant(&[9], bits(16));
        b.read(Selection::SparseIndexed {
            bank: Reg::Csr,
            key,
            slice: BitSlice::new(8, bits(8)),
        });
    })]);
    let [exposed, writes, kills] = entry_sets(&function);
    assert_eq!(exposed, set([Reg::R2, Reg::R3, Reg::Csr]));
    assert!(writes.is_empty());
    assert!(kills.is_empty());
}

#[test]
fn a_raise_binding_is_neither_a_read_nor_a_write_but_names_its_root() {
    let function = entry_records(vec![record(|b| {
        let guard = b.constant(&[1], bits(1));
        let bound = b.constant(&[0], bits(1));
        b.raise(Raise {
            outcome: Fault::DivideError,
            guard: Some(guard),
            payload: &[],
            binds: &[(Reg::Carry, bound)],
        });
    })]);
    let [exposed, writes, kills] = entry_sets(&function);
    assert!(exposed.is_empty());
    assert!(writes.is_empty());
    assert!(kills.is_empty());
    assert_eq!(RootTable::collect(&function).roots(), [Reg::Carry]);
}

#[test]
fn a_call_reads_and_writes_every_root_after_its_own_statements() {
    let mut function = KestrelFunction::new();
    let entry = function.entry();
    function
        .append(entry, |mut b| {
            full_write(&mut b, Reg::Zero);
            b.finish(Control::Advance)
        })
        .expect("seals");
    function
        .append(entry, |mut b| {
            full_write(&mut b, Reg::R0);
            let target = b.constant(&[0x100], bits(32));
            b.finish(Control::Call { target })
        })
        .expect("seals");
    // Another block names `R1`, so the table holds three roots.
    let other = function.add_block();
    function.add_edge(entry, other);
    function
        .append(other, |mut b| {
            read(&mut b, Reg::R1);
            b.finish(Control::Advance)
        })
        .expect("seals");
    let [exposed, writes, kills] = entry_sets(&function);
    // `Zero` and `R0` were overwritten before the callee runs; `R1` is only
    // reached by the call.
    assert_eq!(exposed, set([Reg::R1]));
    assert_eq!(writes, set([Reg::R0, Reg::R1, Reg::Zero]));
    assert_eq!(kills, set([Reg::R0, Reg::Zero]));
}

#[test]
fn a_call_kills_nothing_so_a_later_write_does_not_hide_its_exposure() {
    let mut function = KestrelFunction::new();
    let entry = function.entry();
    function
        .append(entry, |mut b| {
            let target = b.constant(&[0x100], bits(32));
            b.finish(Control::Call { target })
        })
        .expect("seals");
    function
        .append(entry, |mut b| {
            full_write(&mut b, Reg::R0);
            b.finish(Control::Advance)
        })
        .expect("seals");
    let [exposed, writes, kills] = entry_sets(&function);
    assert_eq!(exposed, set([Reg::R0]));
    assert_eq!(writes, set([Reg::R0]));
    assert_eq!(kills, set([Reg::R0]));
}

#[test]
fn an_empty_block_has_an_empty_summary() {
    let mut function = KestrelFunction::new();
    let entry = function.entry();
    let empty = function.add_block();
    function.add_edge(entry, empty);
    function
        .append(entry, |mut b| {
            full_write(&mut b, Reg::R0);
            b.finish(Control::Advance)
        })
        .expect("seals");
    let roots = RootTable::collect(&function);
    let summaries = BlockSummaries::compute(&function, &roots);
    let summary = summaries.block(empty);
    assert_eq!(summary.reads_before_write().count_ones(), 0);
    assert_eq!(summary.writes().count_ones(), 0);
    assert_eq!(summary.kills().count_ones(), 0);
    assert_eq!(summary.writes().len(), roots.len());
}

#[test]
fn a_removed_block_has_an_empty_summary_and_names_no_root() {
    let mut function = KestrelFunction::new();
    let entry = function.entry();
    let doomed = function.add_block();
    function.add_edge(entry, doomed);
    function
        .append(entry, |mut b| {
            full_write(&mut b, Reg::R0);
            b.finish(Control::Advance)
        })
        .expect("seals");
    function
        .append(doomed, |mut b| {
            read(&mut b, Reg::R1);
            full_write(&mut b, Reg::R2);
            b.finish(Control::Advance)
        })
        .expect("seals");
    function.remove_block(doomed);

    let roots = RootTable::collect(&function);
    assert_eq!(roots.roots(), [Reg::R0]);
    let summaries = BlockSummaries::compute(&function, &roots);
    let summary = summaries.block(doomed);
    assert_eq!(summary.writes().count_ones(), 0);
    assert_eq!(summary.reads_before_write().count_ones(), 0);
}

#[test]
fn asking_for_a_block_past_the_node_bound_panics() {
    let function = KestrelFunction::new();
    let roots = RootTable::collect(&function);
    let summaries = BlockSummaries::compute(&function, &roots);
    let asked = catch_unwind(AssertUnwindSafe(|| {
        summaries.block(BlockRef::from_index(1)).writes().len()
    }));
    assert!(asked.is_err(), "block 1 is past the node bound of one");
}
