//! `RootTable`: the roots that one function names, with their dense indices.

use std::panic::{AssertUnwindSafe, catch_unwind};

use cfglib::ir::rtl::{
    Action, BitSlice, Control, InstructionBuilder, Place, Raise, RootTable, Selection, SpanAction,
    WriteTarget,
};

use crate::bits;
use crate::fixtures;
use crate::kestrel::{Fault, KEffect, KOp, Kestrel, Policy, Reg, Undef};
use crate::kestrel_store::KestrelFunction;
use crate::model::{Target, build_kestrel, build_tessera};
use crate::tessera::{Slot, Tessera};

#[test]
fn every_read_write_and_binding_shape_names_its_roots() {
    let mut function = KestrelFunction::new();
    let entry = function.entry();
    function
        .append(entry, |mut b| {
            b.read(Selection::Direct(Place::whole(Reg::R0)));
            let index = b.constant(&[0], bits(8));
            let candidates = [Place::whole(Reg::R1), Place::whole(Reg::R2)];
            b.read(Selection::FiniteIndexed {
                index,
                candidates: &candidates,
            });
            let key = b.constant(&[3], bits(16));
            b.read(Selection::SparseIndexed {
                bank: Reg::Csr,
                key,
                slice: BitSlice::new(0, bits(8)),
            });
            let value = b.constant(&[0], bits(32));
            let plan = [SpanAction {
                span: BitSlice::whole(bits(32)),
                action: Action::TakeInput { value, offset: 0 },
            }];
            b.write(
                WriteTarget::Root {
                    root: Reg::R3,
                    plan: &plan,
                },
                Policy::Zeroing,
            );
            let flag = b.constant(&[0], bits(1));
            let flags = [Place::whole(Reg::Zero), Place::whole(Reg::Neg)];
            b.write(
                WriteTarget::FiniteIndexed {
                    index,
                    candidates: &flags,
                    value: flag,
                },
                Policy::Merging,
            );
            let carry = b.constant(&[1], bits(1));
            b.raise(Raise {
                outcome: Fault::PageFault,
                guard: Some(carry),
                payload: &[],
                binds: &[(Reg::Carry, flag)],
            });
            b.finish(Control::Advance)
        })
        .expect("seals");
    let roots = RootTable::collect(&function);
    assert_eq!(
        roots.roots(),
        [
            Reg::R0,
            Reg::R1,
            Reg::R2,
            Reg::R3,
            Reg::Zero,
            Reg::Neg,
            Reg::Carry,
            Reg::Csr
        ]
    );
    assert_eq!(roots.len(), 8);
    assert!(!roots.is_empty());
}

#[test]
fn statements_that_touch_no_root_leave_the_table_empty() {
    let mut function = KestrelFunction::new();
    let entry = function.entry();
    function
        .append(entry, |mut b| {
            let a = b.constant(&[1], bits(8));
            let chosen = b.nondet(bits(8), Undef::AllFlags);
            let mixed = b.apply(&KOp::Add, &[a, chosen], &[bits(8)]);
            b.effect(&KEffect::Store, &[mixed.first], &[], &[Fault::PageFault]);
            b.finish(Control::Advance)
        })
        .expect("seals");
    let roots = RootTable::collect(&function);
    assert!(roots.is_empty());
    assert_eq!(roots.len(), 0);
    assert_eq!(roots.roots(), []);
}

#[test]
fn the_table_is_ascending_and_a_dense_index_inverts_to_its_root() {
    for (name, model) in fixtures::all() {
        let function = build_kestrel(&model);
        let roots = RootTable::collect(&function);
        assert!(
            roots.roots().windows(2).all(|pair| pair[0] < pair[1]),
            "{name}"
        );
        assert_eq!(roots.len(), roots.roots().len());
        for (index, &root) in roots.roots().iter().enumerate() {
            assert_eq!(roots.root(index), root, "{name}");
            assert_eq!(roots.index_of(root), Some(index), "{name}");
        }
        for root in Reg::ALL {
            let named = model.named_roots().contains(&Kestrel::id_of(root));
            assert_eq!(roots.index_of(root).is_some(), named, "{name}: {root:?}");
        }
    }
}

#[test]
fn a_removed_block_names_nothing_and_an_unreachable_live_block_does() {
    // `Over` and `Wide` appear only in the stale records of a removed slot.
    let function = build_kestrel(&fixtures::loop_with_exits());
    let roots = RootTable::collect(&function);
    assert_eq!(roots.index_of(Reg::Over), None);
    assert_eq!(roots.index_of(Reg::Wide), None);

    // `Csr` is named only by a live block that the entry cannot reach.
    let function = build_kestrel(&fixtures::cut_off_block());
    let roots = RootTable::collect(&function);
    assert_eq!(roots.roots(), [Reg::R0, Reg::Csr]);
}

#[test]
fn collecting_twice_gives_equal_tables() {
    let function = build_kestrel(&fixtures::irreducible());
    let first = RootTable::collect(&function);
    assert_eq!(first, RootTable::collect(&function));
    assert_eq!(first.clone(), first);
}

#[test]
fn the_order_of_a_table_is_the_order_of_its_dialect() {
    let function = build_tessera(&fixtures::diamond());
    let roots = RootTable::collect(&function);
    // Slot 93 holds neutral root 1 and slot 100 holds root 0.
    assert_eq!(roots.roots(), [Slot::of(1), Slot::of(0)]);
    assert_eq!(roots.index_of(Slot::of(0)), Some(1));
    assert_eq!(roots.index_of(Tessera::root_of(5)), None);
}

#[test]
fn asking_for_the_root_at_the_length_panics() {
    let function = build_kestrel(&fixtures::diamond());
    let roots = RootTable::collect(&function);
    let asked = catch_unwind(AssertUnwindSafe(|| roots.root(roots.len())));
    assert!(
        asked.is_err(),
        "the table holds roots below its length only"
    );
}
