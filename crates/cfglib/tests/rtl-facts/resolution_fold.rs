//! `Fold` and `ConstBits` through selection resolution.
//!
//! The analysis hands the operands of an operator to the dialect as little-
//! endian words and reads the results back from one buffer. Each case below
//! folds a small expression into an index and compares the chosen candidate
//! with the same arithmetic done on plain `u64` and `u128` values.

use cfglib::DenseId;
use cfglib::ir::rtl::{Control, InstructionBuilder};

use crate::bits;
use crate::kestrel::{KOp, Kestrel};
use crate::kestrel_store::{KestrelBuilder, KestrelFunction, ValueRef};
use crate::model::Store;
use crate::resolution_kit::{
    data, expect_candidate, masked, resolve, select, selection_positions, word,
};

/// The corner values of an integer of up to 64 bits.
const CORNERS: [u64; 9] = [
    0,
    1,
    2,
    0xF,
    0x10,
    0x7FFF_FFFF,
    0x8000_0000,
    0xFFFF_FFFF,
    0x1234_5678,
];

/// The binary operators whose result is one value of the data width.
#[derive(Clone, Copy, Debug)]
enum Plain {
    Add,
    Sub,
    And,
    Xor,
}

/// The mask of the data width.
fn mask(width: u16) -> u64 {
    if width == 64 {
        u64::MAX
    } else {
        (1 << width) - 1
    }
}

fn plain(op: Plain, left: u64, right: u64, width: u16) -> u64 {
    let mask = mask(width);
    let (left, right) = (left & mask, right & mask);
    (match op {
        Plain::Add => left.wrapping_add(right),
        Plain::Sub => left.wrapping_sub(right),
        Plain::And => left & right,
        Plain::Xor => left ^ right,
    }) & mask
}

in_both_dialects! {
    fn folded_arithmetic_selects_the_candidate_that_plain_arithmetic_gives<D, S>() {
        let width = data::<D>();
        for op in [Plain::Add, Plain::Sub, Plain::And, Plain::Xor] {
            for left in CORNERS {
                for right in CORNERS {
                    let mut function = S::start();
                    let entry = S::NodeId::from_index(0);
                    let inst = function
                        .seal(entry, |mut b| {
                            let a = word::<D, _>(&mut b, left, width);
                            let c = word::<D, _>(&mut b, right, width);
                            let operator = match op {
                                Plain::Add => D::add(),
                                Plain::Sub => D::sub(),
                                Plain::And => D::and(),
                                Plain::Xor => D::xor(),
                            };
                            let result = b.apply(&operator, &[a, c], &[width]).first;
                            let low = masked::<D, _>(&mut b, result, 0xF);
                            select::<D, _>(&mut b, low, 16);
                            b.finish(Control::Advance)
                        })
                        .expect("seals");
                    let resolution = resolve::<D, S>(&function, &[]);
                    let position = selection_positions::<D, S>(&function, inst)[0];
                    let expected = plain(op, left, right, width.width()) & 0xF;
                    assert_eq!(
                        resolution.candidate(inst, position),
                        expect_candidate(expected, 16),
                        "{op:?}({left:#x}, {right:#x}) at {} bits",
                        width.width()
                    );
                }
            }
        }
    }

    fn folded_equality_gives_a_one_bit_index<D, S>() {
        let width = data::<D>();
        for left in CORNERS {
            for right in CORNERS {
                let mut function = S::start();
                let entry = S::NodeId::from_index(0);
                let inst = function
                    .seal(entry, |mut b| {
                        let a = word::<D, _>(&mut b, left, width);
                        let c = word::<D, _>(&mut b, right, width);
                        let same = b.apply(&D::equal(), &[a, c], &[bits(1)]).first;
                        select::<D, _>(&mut b, same, 2);
                        b.finish(Control::Advance)
                    })
                    .expect("seals");
                let resolution = resolve::<D, S>(&function, &[]);
                let position = selection_positions::<D, S>(&function, inst)[0];
                assert_eq!(
                    resolution.candidate(inst, position),
                    expect_candidate(u64::from(left == right), 2),
                    "eq({left:#x}, {right:#x})"
                );
            }
        }
    }
}

/// Values that make carries and wraps cross the word boundary of a `u128`.
const WIDE_CORNERS: [u128; 7] = [
    0,
    1,
    u64::MAX as u128,
    1 << 64,
    1 << 127,
    u128::MAX,
    0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210,
];

fn wide(builder: &mut KestrelBuilder<'_>, value: u128) -> ValueRef {
    let low = u64::try_from(value & u128::from(u64::MAX)).expect("masked");
    let high = u64::try_from(value >> 64).expect("two words");
    builder.constant(&[low, high], bits(128))
}

/// Folds `build` over 128-bit corner pairs and compares the chosen candidate of
/// its sixteen-way selection with `expected(left, right)`.
fn check_wide(
    what: &str,
    build: impl Fn(
        &mut crate::kestrel_store::KestrelBuilder<'_>,
        crate::kestrel_store::ValueRef,
        crate::kestrel_store::ValueRef,
    ),
    expected: impl Fn(u128, u128) -> u128,
) {
    for left in WIDE_CORNERS {
        for right in WIDE_CORNERS {
            let mut function = KestrelFunction::new();
            let entry = function.entry();
            let inst = function
                .seal(entry, |mut b| {
                    let a = wide(&mut b, left);
                    let c = wide(&mut b, right);
                    build(&mut b, a, c);
                    b.finish(Control::Advance)
                })
                .expect("seals");
            let resolution = resolve::<Kestrel, KestrelFunction>(&function, &[]);
            let position = selection_positions::<Kestrel, KestrelFunction>(&function, inst)[0];
            assert_eq!(
                resolution.candidate(inst, position),
                expect_candidate(u64::try_from(expected(left, right)).expect("a nibble"), 16),
                "{what}({left:#x}, {right:#x})"
            );
        }
    }
}

/// `value >> amount` and then the low four bits, as a sixteen-way selection.
fn select_top_nibble(
    builder: &mut crate::kestrel_store::KestrelBuilder<'_>,
    value: crate::kestrel_store::ValueRef,
    amount: u64,
) {
    let by = wide(builder, u128::from(amount));
    let shifted = builder.apply(&KOp::Lsr, &[value, by], &[bits(128)]).first;
    let mask = wide(builder, 0xF);
    let nibble = builder
        .apply(&KOp::And, &[shifted, mask], &[bits(128)])
        .first;
    select::<Kestrel, _>(builder, nibble, 16);
}

#[test]
fn a_sum_that_wraps_across_the_word_boundary_matches_u128_arithmetic() {
    check_wide(
        "add",
        |b, left, right| {
            let sum = b.apply(&KOp::Add, &[left, right], &[bits(128)]).first;
            select_top_nibble(b, sum, 124);
        },
        |left, right| (left.wrapping_add(right) >> 124) & 0xF,
    );
}

#[test]
fn a_difference_of_wide_values_matches_u128_arithmetic() {
    check_wide(
        "sub",
        |b, left, right| {
            let difference = b.apply(&KOp::Sub, &[left, right], &[bits(128)]).first;
            select_top_nibble(b, difference, 60);
        },
        |left, right| (left.wrapping_sub(right) >> 60) & 0xF,
    );
}

#[test]
fn the_carry_result_of_a_two_result_operator_is_read_at_its_own_offset() {
    // `AddCarry` returns the 128-bit sum in two output words and the carry in
    // a third: the carry sits at word offset two.
    check_wide(
        "carry",
        |b, left, right| {
            let both = b.apply(&KOp::AddCarry, &[left, right], &[bits(128), bits(1)]);
            select::<Kestrel, _>(b, both.get(1).expect("two results"), 2);
        },
        |left, right| u128::from(left.overflowing_add(right).1),
    );
}

#[test]
fn concatenated_halves_are_shifted_as_one_number() {
    check_wide(
        "concat",
        |b, left, right| {
            // The high half is the low 64 bits of `left`, the low half those
            // of `right`.
            let high = b.apply(&KOp::Trunc, &[left], &[bits(64)]).first;
            let low = b.apply(&KOp::Trunc, &[right], &[bits(64)]).first;
            let joined = b.apply(&KOp::Concat, &[high, low], &[bits(128)]).first;
            select_top_nibble(b, joined, 62);
        },
        |left, right| {
            let (high, low) = (left & u128::from(u64::MAX), right & u128::from(u64::MAX));
            (((high << 64) | low) >> 62) & 0xF
        },
    );
}

#[test]
fn a_zero_extended_value_keeps_its_bits_and_a_truncated_one_loses_the_rest() {
    check_wide(
        "extend",
        |b, left, _| {
            // Cut to 70 bits, widen again, and read the nibble at bit 64.
            let cut = b.apply(&KOp::Trunc, &[left], &[bits(70)]).first;
            let widened = b.apply(&KOp::Zext, &[cut], &[bits(128)]).first;
            select_top_nibble(b, widened, 64);
        },
        |left, _| ((left & ((1 << 70) - 1)) >> 64) & 0xF,
    );
}

#[test]
fn a_shift_by_at_least_the_width_gives_zero() {
    for (amount, left) in [(128_u64, u128::MAX), (200, 1), (127, 1 << 127), (0, 7)] {
        let mut function = KestrelFunction::new();
        let entry = function.entry();
        let inst = function
            .seal(entry, |mut b| {
                let value = wide(&mut b, left);
                let shifted_left = {
                    let by = wide(&mut b, u128::from(amount));
                    b.apply(&KOp::Shl, &[value, by], &[bits(128)]).first
                };
                let by = wide(&mut b, u128::from(amount));
                let shifted_right = b.apply(&KOp::Lsr, &[value, by], &[bits(128)]).first;
                for shifted in [shifted_left, shifted_right] {
                    let mask = wide(&mut b, 0xF);
                    let nibble = b.apply(&KOp::And, &[shifted, mask], &[bits(128)]).first;
                    select::<Kestrel, _>(&mut b, nibble, 16);
                }
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<Kestrel, KestrelFunction>(&function, &[]);
        let positions = selection_positions::<Kestrel, KestrelFunction>(&function, inst);
        let left_shifted = left
            .checked_shl(u32::try_from(amount).expect("small"))
            .unwrap_or(0);
        let right_shifted = left
            .checked_shr(u32::try_from(amount).expect("small"))
            .unwrap_or(0);
        for (position, expected) in positions.into_iter().zip([left_shifted, right_shifted]) {
            assert_eq!(
                resolution.candidate(inst, position),
                expect_candidate(u64::try_from(expected & 0xF).expect("a nibble"), 16),
                "shift by {amount} of {left:#x}"
            );
        }
    }
}

#[test]
fn comparison_and_selection_operators_fold_over_wide_operands() {
    for (left, right) in [
        (1_u128 << 100, 2_u128),
        (5, 5),
        (u128::MAX, 0),
        (0, 1 << 127),
    ] {
        let mut function = KestrelFunction::new();
        let entry = function.entry();
        let inst = function
            .seal(entry, |mut b| {
                let a = wide(&mut b, left);
                let c = wide(&mut b, right);
                let less = b.apply(&KOp::Ult, &[a, c], &[bits(1)]).first;
                select::<Kestrel, _>(&mut b, less, 2);
                let chosen = b.apply(&KOp::Pick, &[less, a, c], &[bits(128)]).first;
                let mask = wide(&mut b, 0xF);
                let nibble = b.apply(&KOp::And, &[chosen, mask], &[bits(128)]).first;
                select::<Kestrel, _>(&mut b, nibble, 16);
                b.finish(Control::Advance)
            })
            .expect("seals");
        let resolution = resolve::<Kestrel, KestrelFunction>(&function, &[]);
        let positions = selection_positions::<Kestrel, KestrelFunction>(&function, inst);
        let is_less = left < right;
        assert_eq!(
            resolution.candidate(inst, positions[0]),
            Some(usize::from(is_less))
        );
        let picked = if is_less { left } else { right };
        assert_eq!(
            resolution.candidate(inst, positions[1]),
            expect_candidate(u64::try_from(picked & 0xF).expect("a nibble"), 16)
        );
    }
}

#[test]
fn bitwise_or_and_not_of_wide_values_match_u128_arithmetic() {
    check_wide(
        "or",
        |b, left, right| {
            let either = b.apply(&KOp::Or, &[left, right], &[bits(128)]).first;
            select_top_nibble(b, either, 124);
        },
        |left, right| ((left | right) >> 124) & 0xF,
    );
    check_wide(
        "not",
        |b, left, _| {
            let flipped = b.apply(&KOp::Not, &[left], &[bits(128)]).first;
            select_top_nibble(b, flipped, 124);
        },
        |left, _| (!left >> 124) & 0xF,
    );
}
