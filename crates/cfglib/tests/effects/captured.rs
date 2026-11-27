//! `Captured`: the owned statements of one instruction.

use cfglib::ir::rtl::{
    BitSlice, Bits, Control, InstructionBuilder, Place, Raise, RecordView, Selection, Statement,
    ValueRange, WriteTarget,
};
use cfglib::{CaptureBuilder, Captured, CapturedValue};

use crate::wren::{Eff, Fault, Op, Policy, Reg, Undef, Wren, bits, capture, constant, take, whole};

fn value(index: usize) -> CapturedValue {
    CapturedValue::from_index(index)
}

/// Emits one statement of each kind and jumps to the indexed read.
fn emit_every_statement(
    b: &mut CaptureBuilder<'_, Wren>,
    candidates: &[Place<Reg>; 2],
) -> Control<CapturedValue> {
    let index = b.read(Selection::Direct(whole(Reg::Z)));
    let slot = b.read(Selection::FiniteIndexed { index, candidates });
    let element = b.read(Selection::SparseIndexed {
        bank: Reg::Bank,
        key: slot,
        slice: BitSlice::new(8, bits(8)),
    });
    let wide = b.constant(&[1, 2], bits(128));
    let pair = b.apply(&Op::DivMod, &[slot, element], &[bits(16), bits(16)]);
    let undef = b.nondet(Bits::BOOL, Undef::Flags);
    let loaded = b.effect(&Eff::LoadIf, &[undef, slot], &[bits(8)], &[Fault::Page]);
    b.raise(Raise {
        outcome: Fault::Divide,
        guard: Some(undef),
        payload: &[pair.first],
        binds: &[(Reg::C, undef)],
    });
    b.write(
        WriteTarget::Root {
            root: Reg::Slot(2),
            plan: &[take(0, 16, pair.first, 0)],
        },
        Policy::Zero,
    );
    b.write(
        WriteTarget::FiniteIndexed {
            index,
            candidates,
            value: slot,
        },
        Policy::Merge,
    );
    b.write(
        WriteTarget::SparseIndexed {
            bank: Reg::Bank,
            key: slot,
            slice: BitSlice::new(0, bits(8)),
            value: loaded.first,
        },
        Policy::Merge,
    );
    let _ = wide;
    Control::Jump { target: slot }
}

#[test]
fn a_capture_gives_back_every_statement_with_dense_values() {
    let candidates = [whole(Reg::Slot(0)), whole(Reg::Slot(1))];
    let captured = capture(|b| emit_every_statement(b, &candidates));

    let plan = [take(0, 16, value(4), 0)];
    let expected = [
        Statement::Read {
            result: value(0),
            source: Selection::Direct(whole(Reg::Z)),
        },
        Statement::Read {
            result: value(1),
            source: Selection::FiniteIndexed {
                index: value(0),
                candidates: &candidates,
            },
        },
        Statement::Read {
            result: value(2),
            source: Selection::SparseIndexed {
                bank: Reg::Bank,
                key: value(1),
                slice: BitSlice::new(8, bits(8)),
            },
        },
        Statement::Constant {
            result: value(3),
            words: &[1, 2],
            width: bits(128),
        },
        Statement::Apply {
            results: ValueRange::new(value(4), 2),
            operator: &Op::DivMod,
            args: &[value(1), value(2)],
        },
        Statement::Nondet {
            result: value(6),
            width: Bits::BOOL,
            domain: Undef::Flags,
        },
        Statement::Effect {
            results: ValueRange::new(value(7), 1),
            operation: &Eff::LoadIf,
            args: &[value(6), value(1)],
            outcomes: &[Fault::Page],
        },
        Statement::Raise(Raise {
            outcome: Fault::Divide,
            guard: Some(value(6)),
            payload: &[value(4)],
            binds: &[(Reg::C, value(6))],
        }),
        Statement::Write {
            target: WriteTarget::Root {
                root: Reg::Slot(2),
                plan: &plan,
            },
            policy: Policy::Zero,
        },
        Statement::Write {
            target: WriteTarget::FiniteIndexed {
                index: value(0),
                candidates: &candidates,
                value: value(1),
            },
            policy: Policy::Merge,
        },
        Statement::Write {
            target: WriteTarget::SparseIndexed {
                bank: Reg::Bank,
                key: value(1),
                slice: BitSlice::new(0, bits(8)),
                value: value(7),
            },
            policy: Policy::Merge,
        },
    ];
    let actual: Vec<Statement<'_, Wren, CapturedValue>> =
        RecordView::statements(&captured).collect();
    assert_eq!(actual, expected);
    assert_eq!(captured.statement_count(), expected.len());
    assert_eq!(captured.value_count(), 8);
    assert_eq!(
        RecordView::control(&captured),
        Control::Jump { target: value(1) }
    );
    let widths: Vec<Bits> = (0..8).map(|index| captured.width(value(index))).collect();
    assert_eq!(
        widths,
        [
            Bits::BOOL,
            bits(16),
            bits(8),
            bits(128),
            bits(16),
            bits(16),
            Bits::BOOL,
            bits(8),
        ]
    );
}

#[test]
fn begin_starts_a_new_instruction_with_values_from_zero() {
    let mut captured = Captured::<Wren>::new();
    let mut builder = captured.begin();
    let a = constant(&mut builder, 1, 32);
    let b = constant(&mut builder, 2, 32);
    builder.apply(&Op::Add, &[a, b], &[bits(32)]);
    builder
        .finish(Control::Advance)
        .expect("a capture accepts every record");
    assert_eq!(captured.value_count(), 3);
    assert_eq!(captured.finished_control(), Some(Control::Advance));

    let _ = captured.begin();
    assert_eq!(captured.value_count(), 0);
    assert_eq!(captured.statement_count(), 0);
    assert_eq!(captured.finished_control(), None);

    let mut builder = captured.begin();
    let first = builder.read(Selection::Direct(whole(Reg::A)));
    builder
        .finish(Control::Return { target: first })
        .expect("a capture accepts every record");
    assert_eq!(first, value(0));
    assert_eq!(
        RecordView::control(&captured),
        Control::Return { target: value(0) }
    );
}

#[test]
#[should_panic(expected = "only after the producer finished it")]
fn an_unfinished_capture_has_no_control() {
    let captured = Captured::<Wren>::new();
    let _ = RecordView::control(&captured);
}
