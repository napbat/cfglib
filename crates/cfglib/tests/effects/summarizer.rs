//! Each rule of the summarizer, on one Wren instruction each.

use cfglib::ir::rtl::{
    BitSlice, Bits, Control, InstructionBuilder, Place, Raise, Selection, SpanAction, Statement,
    ValueRange, WriteTarget,
};
use cfglib::{
    Access, Address, CaptureBuilder, CapturedValue, Exit, Expr, MemoryAccessKind, MemoryAtomicity,
    Opaque, RaiseOrigin, Raised, StateTouch, Summarizer, Touch, Write, WriteKind,
};

use crate::wren::{
    BASE, Eff, Fault, Op, Policy, Reg, Space, Undef, Wren, WrenEffects, bits, capture, constant,
    effect_of, keep, part, take, undefined, write_plan, write_whole,
};

fn touch(place: Reg, offset: u16, width: u16) -> Touch<Reg> {
    Touch {
        place,
        bits: BitSlice::new(offset, bits(width)),
    }
}

fn write(place: Reg, offset: u16, width: u16, kind: WriteKind) -> Write<Reg> {
    Write {
        place,
        bits: BitSlice::new(offset, bits(width)),
        kind,
    }
}

#[test]
fn every_read_is_upward_exposed_and_one_root_is_listed_once() {
    let effect = effect_of(|b| {
        let low = b.read(Selection::Direct(part(Reg::A, 0, 16)));
        let high = b.read(Selection::Direct(part(Reg::A, 8, 24)));
        let other = b.read(Selection::Direct(part(Reg::B, 24, 8)));
        let again = b.read(Selection::Direct(part(Reg::B, 0, 8)));
        let sum = b.apply(&Op::Add, &[low, high], &[bits(16)]).first;
        let _ = (other, again);
        write_plan(b, Reg::A, &[take(0, 16, sum, 0), keep(16, 16)]);
        Control::Advance
    });
    assert_eq!(
        effect.reads,
        [
            touch(Reg::A, 0, 32),
            touch(Reg::B, 0, 8),
            touch(Reg::B, 24, 8)
        ]
    );
    assert_eq!(effect.writes, [write(Reg::A, 0, 16, WriteKind::Always)]);
    assert_eq!(effect.exit, Exit::Next);
    assert!(!effect.nondeterministic);
    assert_eq!(effect.memory, []);
    assert_eq!(effect.raises, []);
    assert!(effect.exprs.is_empty());
}

#[test]
fn a_select_of_the_old_bits_is_a_conditional_write() {
    let effect = effect_of(|b| {
        let guard = b.read(Selection::Direct(Place::whole(Reg::Z)));
        let new = b.read(Selection::Direct(Place::whole(Reg::B)));
        let old = b.read(Selection::Direct(Place::whole(Reg::A)));
        let chosen = b.apply(&Op::Select, &[guard, new, old], &[bits(32)]).first;
        write_whole(b, Reg::A, chosen);
        Control::Advance
    });
    let [
        Write {
            place: Reg::A,
            kind: WriteKind::Conditional(guard),
            bits: written,
        },
    ] = effect.writes[..]
    else {
        panic!("one conditional write of A: {:?}", effect.writes);
    };
    assert_eq!(written, BitSlice::whole(bits(32)));
    assert_eq!(
        effect.exprs.get(guard),
        Expr::Read(touch(Reg::Z, 0, 1)),
        "the guard of the write is the guard of the select"
    );
    // The select reads the old value, so the conditional write keeps it live.
    assert_eq!(
        effect.reads,
        [
            touch(Reg::A, 0, 32),
            touch(Reg::B, 0, 32),
            touch(Reg::Z, 0, 1)
        ]
    );
}

#[test]
fn a_select_with_its_arms_in_the_other_order_is_conditional_too() {
    let effect = effect_of(|b| {
        let guard = b.read(Selection::Direct(Place::whole(Reg::Z)));
        let old = b.read(Selection::Direct(part(Reg::A, 16, 16)));
        let new = constant(b, 7, 16);
        let chosen = b.apply(&Op::Mux, &[guard, old, new], &[bits(16)]).first;
        write_plan(b, Reg::A, &[keep(0, 16), take(16, 16, chosen, 0)]);
        Control::Advance
    });
    assert!(matches!(
        effect.writes[..],
        [Write {
            place: Reg::A,
            kind: WriteKind::Conditional(_),
            ..
        }]
    ));
    assert_eq!(effect.writes[0].bits, BitSlice::new(16, bits(16)));
}

#[test]
fn a_select_of_other_bits_is_an_unconditional_write() {
    let effect = effect_of(|b| {
        let guard = b.read(Selection::Direct(Place::whole(Reg::Z)));
        let new = b.read(Selection::Direct(Place::whole(Reg::B)));
        let other = b.read(Selection::Direct(Place::whole(Reg::A)));
        let low = b.read(Selection::Direct(part(Reg::A, 0, 16)));
        let high = b.read(Selection::Direct(part(Reg::A, 16, 16)));
        let wide = b
            .apply(&Op::Select, &[guard, new, other], &[bits(32)])
            .first;
        let half = b.apply(&Op::Select, &[guard, low, high], &[bits(16)]).first;
        write_whole(b, Reg::B, wide);
        // The low half of A receives the high half of A when Z is zero: the
        // old bits of the select are not the bits that the span receives.
        write_plan(b, Reg::A, &[take(0, 16, half, 0), keep(16, 16)]);
        Control::Advance
    });
    assert_eq!(
        effect.writes,
        [
            write(Reg::B, 0, 32, WriteKind::Always),
            write(Reg::A, 0, 16, WriteKind::Always),
        ]
    );
}

#[test]
fn a_select_of_a_part_of_the_root_matches_the_span_offset() {
    let effect = effect_of(|b| {
        let guard = b.read(Selection::Direct(Place::whole(Reg::Z)));
        let old = b.read(Selection::Direct(Place::whole(Reg::A)));
        let new = b.read(Selection::Direct(Place::whole(Reg::B)));
        let chosen = b.apply(&Op::Select, &[guard, new, old], &[bits(32)]).first;
        // The span at bit 8 takes the select from bit 8: these are the old
        // bits 8..16 of A when Z is zero.
        write_plan(
            b,
            Reg::A,
            &[keep(0, 8), take(8, 8, chosen, 8), keep(16, 16)],
        );
        Control::Advance
    });
    assert!(matches!(
        effect.writes[..],
        [Write {
            kind: WriteKind::Conditional(_),
            ..
        }]
    ));
    assert_eq!(effect.writes[0].bits, BitSlice::new(8, bits(8)));
}

#[test]
fn a_nondeterministic_span_is_an_undefined_write() {
    let effect = effect_of(|b| {
        let choice = b.nondet(Bits::BOOL, Undef::Flags);
        write_plan(b, Reg::Z, &[undefined(0, 1)]);
        write_whole(b, Reg::C, choice);
        Control::Advance
    });
    assert_eq!(
        effect.writes,
        [
            write(Reg::Z, 0, 1, WriteKind::Undefined),
            write(Reg::C, 0, 1, WriteKind::Undefined),
        ]
    );
    assert!(effect.nondeterministic);
    assert_eq!(effect.reads, []);
}

#[test]
fn a_kept_span_is_no_write_and_neighbor_spans_of_one_kind_merge() {
    let effect = effect_of(|b| {
        let value = b.read(Selection::Direct(Place::whole(Reg::B)));
        let zero = constant(b, 0, 8);
        write_plan(
            b,
            Reg::A,
            &[
                take(0, 8, value, 0),
                take(8, 8, zero, 0),
                keep(16, 8),
                take(24, 4, value, 24),
                undefined(28, 4),
            ],
        );
        Control::Advance
    });
    assert_eq!(
        effect.writes,
        [
            write(Reg::A, 0, 16, WriteKind::Always),
            write(Reg::A, 24, 4, WriteKind::Always),
            write(Reg::A, 28, 4, WriteKind::Undefined),
        ]
    );
}

#[test]
fn an_indexed_access_touches_every_candidate() {
    let candidates = [Place::whole(Reg::Slot(0)), Place::whole(Reg::Slot(1))];
    let effect = effect_of(|b| {
        let index = b.read(Selection::Direct(Place::whole(Reg::Z)));
        let key = b.read(Selection::Direct(Place::whole(Reg::B)));
        let value = b.read(Selection::FiniteIndexed {
            index,
            candidates: &candidates,
        });
        let element = b.read(Selection::SparseIndexed {
            bank: Reg::Bank,
            key,
            slice: BitSlice::new(0, bits(16)),
        });
        b.write(
            WriteTarget::FiniteIndexed {
                index,
                candidates: &candidates,
                value: element,
            },
            Policy::Merge,
        );
        b.write(
            WriteTarget::SparseIndexed {
                bank: Reg::Bank,
                key,
                slice: BitSlice::new(16, bits(16)),
                value,
            },
            Policy::Zero,
        );
        Control::Advance
    });
    assert_eq!(
        effect.reads,
        [
            touch(Reg::B, 0, 32),
            touch(Reg::Z, 0, 1),
            touch(Reg::Slot(0), 0, 16),
            touch(Reg::Slot(1), 0, 16),
            touch(Reg::Bank, 0, 16),
        ]
    );
    assert_eq!(
        effect.writes,
        [
            write(Reg::Slot(0), 0, 16, WriteKind::Indexed),
            write(Reg::Slot(1), 0, 16, WriteKind::Indexed),
            write(Reg::Bank, 16, 16, WriteKind::Indexed),
        ]
    );
}

#[test]
fn guarded_loads_and_stores_keep_their_guards_and_shapes() {
    let effect = effect_of(|b| {
        let guard = b.read(Selection::Direct(Place::whole(Reg::Z)));
        let base = b.read(Selection::Direct(Place::whole(Reg::A)));
        let four = constant(b, 4, 32);
        let address = b.apply(&Op::Add, &[base, four], &[bits(32)]).first;
        let loaded = b
            .effect(&Eff::LoadIf, &[guard, address], &[bits(16)], &[Fault::Page])
            .first;
        b.effect(&Eff::StoreIf, &[base, loaded, guard], &[], &[Fault::Page]);
        b.effect(&Eff::Store(Space::Stack), &[address, base], &[], &[]);
        b.effect(&Eff::AtomicAdd, &[base, four], &[bits(32)], &[]);
        Control::Advance
    });
    let guard = effect.memory[0].guard.expect("the load is guarded");
    let address = effect.memory[0].location.address;
    assert_eq!(effect.exprs.get(guard), Expr::Read(touch(Reg::Z, 0, 1)));
    let Expr::Op {
        name: "add",
        args: &[base, four],
        result: 0,
        width,
    } = effect.exprs.get(address)
    else {
        panic!("the address is the sum: {:?}", effect.exprs.get(address));
    };
    assert_eq!(width, bits(32));
    assert_eq!(effect.exprs.get(base), Expr::Read(touch(Reg::A, 0, 32)));
    assert!(matches!(
        effect.exprs.get(four),
        Expr::Const { words: &[4], .. }
    ));
    let access = |kind, atomicity, space, address, bytes, guard| Access {
        kind,
        atomicity,
        location: Address { space, address },
        bytes,
        guard,
    };
    assert_eq!(
        effect.memory,
        [
            access(
                MemoryAccessKind::Read,
                MemoryAtomicity::NonAtomic,
                Space::Data,
                address,
                2,
                Some(guard)
            ),
            access(
                MemoryAccessKind::Write,
                MemoryAtomicity::NonAtomic,
                Space::Data,
                base,
                2,
                Some(guard)
            ),
            access(
                MemoryAccessKind::Write,
                MemoryAtomicity::NonAtomic,
                Space::Stack,
                address,
                4,
                None
            ),
            access(
                MemoryAccessKind::ReadModifyWrite,
                MemoryAtomicity::Atomic,
                Space::Data,
                base,
                4,
                None
            ),
        ]
    );
    let environment = |guard| Raised {
        outcome: Fault::Page,
        guard,
        origin: RaiseOrigin::Environment,
        check: false,
    };
    assert_eq!(
        effect.raises,
        [environment(Some(guard)), environment(Some(guard))]
    );
}

#[test]
fn a_raise_keeps_its_guard_and_its_bindings_are_not_writes() {
    let effect = effect_of(|b| {
        let divisor = b.read(Selection::Direct(Place::whole(Reg::B)));
        let zero = constant(b, 0, 32);
        let is_zero = b.apply(&Op::Eq, &[divisor, zero], &[Bits::BOOL]).first;
        let flag = constant(b, 1, 1);
        b.raise(Raise {
            outcome: Fault::Divide,
            guard: Some(is_zero),
            payload: &[divisor],
            binds: &[(Reg::C, flag)],
        });
        let dividend = b.read(Selection::Direct(Place::whole(Reg::A)));
        let results = b.apply(&Op::DivMod, &[dividend, divisor], &[bits(32), bits(32)]);
        write_whole(b, Reg::A, results.first);
        write_whole(b, Reg::B, results.get(1).expect("two results"));
        Control::Advance
    });
    let [
        Raised {
            outcome: Fault::Divide,
            guard: Some(guard),
            origin: RaiseOrigin::Raise,
            check: false,
        },
    ] = effect.raises[..]
    else {
        panic!("one guarded raise: {:?}", effect.raises);
    };
    assert!(matches!(
        effect.exprs.get(guard),
        Expr::Op { name: "eq", .. }
    ));
    assert_eq!(
        effect.writes,
        [
            write(Reg::A, 0, 32, WriteKind::Always),
            write(Reg::B, 0, 32, WriteKind::Always),
        ]
    );
}

#[test]
fn an_unconditional_raise_terminates() {
    let effect = effect_of(|b| {
        b.raise(Raise {
            outcome: Fault::Divide,
            guard: None,
            payload: &[],
            binds: &[],
        });
        Control::Terminating
    });
    assert_eq!(
        effect.raises,
        [Raised {
            outcome: Fault::Divide,
            guard: None,
            origin: RaiseOrigin::Raise,
            check: false,
        }]
    );
    assert_eq!(effect.exit, Exit::Terminating);
}

#[test]
fn each_control_boundary_gives_its_exit() {
    let exit = |control: fn(cfglib::CapturedValue) -> Control<cfglib::CapturedValue>| {
        effect_of(|b| {
            let target = b.read(Selection::Direct(Place::whole(Reg::A)));
            control(target)
        })
        .exit
    };
    assert_eq!(exit(|_| Control::Advance), Exit::Next);
    assert_eq!(exit(|target| Control::Jump { target }), Exit::Jump);
    assert_eq!(exit(|target| Control::Call { target }), Exit::Call);
    assert_eq!(exit(|target| Control::Return { target }), Exit::Return);
    assert_eq!(
        exit(|scrutinee| Control::Dispatch { scrutinee }),
        Exit::Dispatch
    );
    assert_eq!(
        exit(|taken| Control::Branch {
            condition: taken,
            taken,
            not_taken: None
        }),
        Exit::Branch
    );
    assert_eq!(exit(|_| Control::Terminating), Exit::Terminating);
}

#[test]
fn each_exit_has_a_name() {
    let names = [
        Exit::Next,
        Exit::Jump,
        Exit::Branch,
        Exit::Call,
        Exit::Return,
        Exit::Dispatch,
        Exit::Terminating,
    ]
    .map(|exit| exit.to_string());
    assert_eq!(
        names,
        [
            "next",
            "jump",
            "branch",
            "call",
            "return",
            "dispatch",
            "terminating"
        ]
    );
}

#[test]
fn a_call_lists_its_reads_and_writes_and_follows_no_callee() {
    let effect = effect_of(|b| {
        let target = b.read(Selection::Direct(Place::whole(Reg::A)));
        let stack = b.read(Selection::Direct(Place::whole(Reg::B)));
        let four = constant(b, 4, 32);
        let below = b.apply(&Op::Sub, &[stack, four], &[bits(32)]).first;
        b.effect(
            &Eff::Store(Space::Stack),
            &[below, target],
            &[],
            &[Fault::Page],
        );
        write_whole(b, Reg::B, below);
        Control::Call { target }
    });
    assert_eq!(effect.exit, Exit::Call);
    assert_eq!(effect.reads, [touch(Reg::A, 0, 32), touch(Reg::B, 0, 32)]);
    assert_eq!(effect.writes, [write(Reg::B, 0, 32, WriteKind::Always)]);
    assert_eq!(effect.memory.len(), 1);
}

#[test]
fn intrinsics_report_the_state_they_can_change_and_the_check() {
    let quiet = effect_of(|b| {
        b.effect(&Eff::Fence, &[], &[], &[]);
        Control::Advance
    });
    assert_eq!(quiet.intrinsic_state, StateTouch::Preserved);
    assert!(!quiet.instruction_check);

    let loud = effect_of(|b| {
        b.effect(&Eff::Fence, &[], &[], &[]);
        b.effect(&Eff::Opaque, &[], &[], &[Fault::Page]);
        Control::Advance
    });
    assert_eq!(loud.intrinsic_state, StateTouch::Unrestricted);
    assert_eq!(
        loud.raises,
        [Raised {
            outcome: Fault::Page,
            guard: None,
            origin: RaiseOrigin::Environment,
            check: false,
        }]
    );
}

#[test]
fn the_raises_of_the_instruction_check_are_marked() {
    let effect = effect_of(|b| {
        let pending = b.read(Selection::Direct(Place::whole(Reg::Check)));
        let fault = b
            .effect(
                &Eff::InstructionCheck,
                &[pending],
                &[Bits::BOOL],
                &[Fault::Check],
            )
            .first;
        let not = b.apply(&Op::Not, &[fault], &[Bits::BOOL]).first;
        let again = b.apply(&Op::Not, &[not], &[Bits::BOOL]).first;
        b.raise(Raise {
            outcome: Fault::Check,
            guard: Some(again),
            payload: &[],
            binds: &[],
        });
        let flag = b.read(Selection::Direct(Place::whole(Reg::Z)));
        b.raise(Raise {
            outcome: Fault::Divide,
            guard: Some(flag),
            payload: &[],
            binds: &[],
        });
        let zero = constant(b, 0, 1);
        write_whole(b, Reg::Check, zero);
        Control::Advance
    });
    assert!(effect.instruction_check);
    assert_eq!(effect.intrinsic_state, StateTouch::Preserved);
    let checks: Vec<bool> = effect.raises.iter().map(|raised| raised.check).collect();
    assert_eq!(checks, [true, true, false]);
    let guard = effect.raises[1].guard.expect("the check raise is guarded");
    let Expr::Op { args: &[inner], .. } = effect.exprs.get(guard) else {
        panic!("the guard is a negation");
    };
    let Expr::Op {
        args: &[result], ..
    } = effect.exprs.get(inner)
    else {
        panic!("the inner guard is a negation");
    };
    assert_eq!(
        effect.exprs.get(result),
        Expr::Opaque {
            origin: Opaque::Effect {
                ordinal: 0,
                result: 0
            },
            width: Bits::BOOL,
        }
    );
}

#[test]
fn a_value_that_two_rules_need_has_one_expression() {
    let effect = effect_of(|b| {
        let base = b.read(Selection::Direct(Place::whole(Reg::A)));
        let flag = b.read(Selection::Direct(Place::whole(Reg::Z)));
        let old = b.read(Selection::Direct(Place::whole(Reg::B)));
        let chosen = b.apply(&Op::Select, &[flag, base, old], &[bits(32)]).first;
        b.effect(&Eff::Load(Space::Data), &[base], &[bits(8)], &[]);
        b.effect(&Eff::StoreIf, &[base, base, flag], &[], &[]);
        write_whole(b, Reg::B, chosen);
        Control::Advance
    });
    // The address A, the guard Z: one expression each, in the order of need.
    assert_eq!(effect.exprs.len(), 2);
    assert_eq!(
        effect.memory[0].location.address,
        effect.memory[1].location.address
    );
    assert_eq!(
        effect.writes[0].kind,
        WriteKind::Conditional(effect.memory[1].guard.unwrap())
    );
}

/// Emits a conditional load, a select of the old value, an indexed read,
/// a raise on a choice, and a branch.
fn emit_mixed(
    b: &mut CaptureBuilder<'_, Wren>,
    candidates: &[Place<Wren>; 2],
) -> Control<CapturedValue> {
    let flag = b.read(Selection::Direct(Place::whole(Reg::Z)));
    let base = b.read(Selection::Direct(Place::whole(Reg::A)));
    let old = b.read(Selection::Direct(Place::whole(Reg::B)));
    let slot = b.read(Selection::FiniteIndexed {
        index: flag,
        candidates,
    });
    let four = constant(b, 4, 32);
    let address = b.apply(&Op::Add, &[base, four], &[bits(32)]).first;
    let loaded = b
        .effect(&Eff::LoadIf, &[flag, address], &[bits(32)], &[Fault::Page])
        .first;
    let chosen = b
        .apply(&Op::Select, &[flag, loaded, old], &[bits(32)])
        .first;
    let undef = b.nondet(Bits::BOOL, Undef::Flags);
    b.raise(Raise {
        outcome: Fault::Divide,
        guard: Some(undef),
        payload: &[slot],
        binds: &[],
    });
    write_whole(b, Reg::B, chosen);
    write_plan(b, Reg::A, &[take(0, 16, address, 0), undefined(16, 16)]);
    Control::Branch {
        condition: flag,
        taken: address,
        not_taken: None,
    }
}

/// The statements of one instruction, written as plain data with value
/// handles from 1000, and the same instruction emitted into a capture.
#[test]
fn a_capture_and_the_same_statements_from_another_source_give_one_effect() {
    let candidates = [Place::whole(Reg::Slot(2)), Place::whole(Reg::Slot(3))];
    let captured = capture(|b| emit_mixed(b, &candidates));

    let v = |index: usize| 1000 + index;
    let four = [4_u64];
    let add_args = [v(1), v(4)];
    let load_args = [v(0), v(5)];
    let page = [Fault::Page];
    let select_args = [v(0), v(6), v(2)];
    let payload = [v(3)];
    let b_plan = [take(0, 32, v(7), 0)];
    let a_plan: [SpanAction<Wren, usize>; 2] = [take(0, 16, v(5), 0), undefined(16, 16)];
    let statements = [
        Statement::Read {
            result: v(0),
            source: Selection::Direct(Place::whole(Reg::Z)),
        },
        Statement::Read {
            result: v(1),
            source: Selection::Direct(Place::whole(Reg::A)),
        },
        Statement::Read {
            result: v(2),
            source: Selection::Direct(Place::whole(Reg::B)),
        },
        Statement::Read {
            result: v(3),
            source: Selection::FiniteIndexed {
                index: v(0),
                candidates: &candidates,
            },
        },
        Statement::Constant {
            result: v(4),
            words: &four,
            width: bits(32),
        },
        Statement::Apply {
            results: ValueRange::new(v(5), 1),
            operator: &Op::Add,
            args: &add_args,
        },
        Statement::Effect {
            results: ValueRange::new(v(6), 1),
            operation: &Eff::LoadIf,
            args: &load_args,
            outcomes: &page,
        },
        Statement::Apply {
            results: ValueRange::new(v(7), 1),
            operator: &Op::Select,
            args: &select_args,
        },
        Statement::Nondet {
            result: v(8),
            width: Bits::BOOL,
            domain: Undef::Flags,
        },
        Statement::Raise(Raise {
            outcome: Fault::Divide,
            guard: Some(v(8)),
            payload: &payload,
            binds: &[],
        }),
        Statement::Write {
            target: WriteTarget::Root {
                root: Reg::B,
                plan: &b_plan,
            },
            policy: Policy::Merge,
        },
        Statement::Write {
            target: WriteTarget::Root {
                root: Reg::A,
                plan: &a_plan,
            },
            policy: Policy::Merge,
        },
    ];
    let control = Control::Branch {
        condition: v(0),
        taken: v(5),
        not_taken: None,
    };
    let width = |value: usize| match value - 1000 {
        0 | 8 => Bits::BOOL,
        3 => bits(16),
        _ => bits(32),
    };

    let summarizer = Summarizer::new(&WrenEffects);
    let direct = summarizer.summarize(statements, control, width);
    let from_capture = summarizer.summarize_record(&captured);
    assert_eq!(direct, from_capture);
}

#[test]
fn the_mixed_instruction_has_a_conditional_an_assigned_and_an_undefined_write() {
    let candidates = [Place::whole(Reg::Slot(2)), Place::whole(Reg::Slot(3))];
    let effect = effect_of(|b| emit_mixed(b, &candidates));
    let flag = effect.raises[0].guard.expect("the load is guarded by Z");
    assert_eq!(
        effect.writes,
        [
            write(Reg::B, 0, 32, WriteKind::Conditional(flag)),
            write(Reg::A, 0, 16, WriteKind::Always),
            write(Reg::A, 16, 16, WriteKind::Undefined),
        ]
    );
    assert!(effect.nondeterministic);
    assert_eq!(effect.exit, Exit::Branch);
}

#[test]
fn a_record_with_no_statement_has_an_empty_effect() {
    let effect = effect_of(|_| Control::Advance);
    assert!(effect.reads.is_empty() && effect.writes.is_empty());
    assert_eq!(effect.exit, Exit::Next);
    assert_eq!(effect.intrinsic_state, StateTouch::Preserved);
}

#[test]
fn an_operator_that_the_dialect_knows_as_a_number_is_a_constant() {
    let effect = effect_of(|b| {
        let base = b.apply(&Op::Base, &[], &[bits(32)]).first;
        b.effect(&Eff::Load(Space::Data), &[base], &[bits(8)], &[]);
        Control::Advance
    });
    assert_eq!(
        effect.exprs.get(effect.memory[0].location.address),
        Expr::Const {
            words: &[BASE],
            width: bits(32)
        }
    );
}
