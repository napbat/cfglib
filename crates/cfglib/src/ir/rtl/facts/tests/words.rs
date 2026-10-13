//! Bit-field copies over words.

use crate::ir::rtl::Bits;
use crate::ir::rtl::facts::words::{copy_bits, normalized, to_usize};

#[test]
fn a_copy_crosses_a_word_boundary_of_the_source() {
    let source = [0xf000_0000_0000_0000, 0x0f];
    let mut target = [0_u64; 1];
    copy_bits(&mut target, 4, &source, 60, 8);
    assert_eq!(target, [0xff0]);
}

#[test]
fn a_copy_crosses_a_word_boundary_of_the_target() {
    let source = [0xabcd];
    let mut target = [0_u64; 2];
    copy_bits(&mut target, 56, &source, 0, 16);
    assert_eq!(target, [0xcd00_0000_0000_0000, 0xab]);
}

#[test]
fn a_copy_keeps_the_other_bits_of_the_target() {
    let mut target = [u64::MAX];
    copy_bits(&mut target, 8, &[0], 0, 8);
    assert_eq!(target, [0xffff_ffff_ffff_00ff]);
}

#[test]
fn a_copy_of_a_whole_word_is_exact() {
    let mut target = [0];
    copy_bits(&mut target, 0, &[u64::MAX], 0, 64);
    assert_eq!(target, [u64::MAX]);
}

#[test]
fn source_bits_past_the_last_word_read_as_zero() {
    let mut target = [u64::MAX; 2];
    copy_bits(&mut target, 0, &[1], 0, 128);
    assert_eq!(target, [1, 0]);
}

#[test]
fn a_normalized_value_has_no_bits_past_its_width() {
    let three = Bits::new(3).unwrap();
    assert_eq!(*normalized(&[u64::MAX], three), [7]);
    let wide = Bits::new(70).unwrap();
    assert_eq!(
        *normalized(&[u64::MAX, u64::MAX, u64::MAX], wide),
        [u64::MAX, 0x3f]
    );
    assert_eq!(*normalized(&[], three), [0]);
}

#[test]
fn a_number_converts_when_it_fits() {
    assert_eq!(to_usize(&[5]), Some(5));
    assert_eq!(to_usize(&[5, 0, 0]), Some(5));
    assert_eq!(to_usize(&[5, 1]), None);
    assert_eq!(to_usize(&[]), None);
}
