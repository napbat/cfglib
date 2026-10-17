//! `Bits`, `BitSlice`, and `Place`: the value type of every RTL record.

use std::collections::{BTreeSet, HashSet};

use cfglib::ir::rtl::{BitSlice, Bits, Dialect, Place};

use crate::bits;
use crate::kestrel::{Kestrel, Reg};
use crate::tessera::{Slot, Tessera};

#[test]
fn bits_accept_exactly_the_widths_one_through_the_maximum() {
    assert_eq!(Bits::MAX_WIDTH, 8192);
    assert_eq!(Bits::new(0), None);
    assert_eq!(Bits::new(1), Some(Bits::BOOL));
    assert!(Bits::new(8192).is_some());
    assert_eq!(Bits::new(8193), None);
    assert_eq!(Bits::new(u16::MAX), None);
    for width in 0..=u16::MAX {
        let made = Bits::new(width);
        assert_eq!(made.is_some(), (1..=8192).contains(&width), "width {width}");
        if let Some(made) = made {
            assert_eq!(made.width(), width);
        }
    }
}

#[test]
fn the_boolean_type_is_one_bit_wide() {
    assert_eq!(Bits::BOOL.width(), 1);
    assert_eq!(Bits::BOOL.words(), 1);
    assert_eq!(Bits::BOOL, bits(1));
}

#[test]
fn words_round_up_to_whole_sixty_four_bit_words() {
    for (width, words) in [
        (1, 1),
        (2, 1),
        (63, 1),
        (64, 1),
        (65, 2),
        (127, 2),
        (128, 2),
        (129, 3),
        (8191, 128),
        (8192, 128),
    ] {
        assert_eq!(bits(width).words(), words, "width {width}");
    }
}

#[test]
fn words_hold_the_width_with_less_than_one_word_to_spare() {
    for width in 1..=Bits::MAX_WIDTH {
        let words = bits(width).words();
        let width = usize::from(width);
        assert!(words * 64 >= width, "{words} words hold {width} bits");
        assert!(
            (words - 1) * 64 < width,
            "{words} words are too many for {width} bits"
        );
    }
}

#[test]
fn bits_order_by_width_and_hash_by_value() {
    assert!(bits(8) < bits(9));
    assert!(Bits::BOOL < bits(8192));
    let ordered: BTreeSet<_> = [bits(64), bits(8), bits(64), bits(1)].into_iter().collect();
    assert_eq!(
        ordered.into_iter().map(Bits::width).collect::<Vec<_>>(),
        [1, 8, 64]
    );
    let hashed: HashSet<_> = [bits(32), bits(32), bits(33)].into_iter().collect();
    assert_eq!(hashed.len(), 2);
}

#[test]
fn bits_display_names_their_width() {
    assert!(bits(130).to_string().contains("130"));
}

#[test]
fn a_slice_ends_at_the_bit_after_its_last_bit_even_past_sixteen_bits() {
    assert_eq!(BitSlice::new(0, bits(8)).end(), 8);
    assert_eq!(BitSlice::new(24, bits(8)).end(), 32);
    // The offset and the width each fit in sixteen bits, the sum does not.
    assert_eq!(BitSlice::new(u16::MAX, bits(8192)).end(), 65_535 + 8_192);
}

#[test]
fn the_whole_slice_of_a_type_starts_at_bit_zero() {
    let whole = BitSlice::whole(bits(130));
    assert_eq!(whole, BitSlice::new(0, bits(130)));
    assert_eq!(whole.offset, 0);
    assert_eq!(whole.width, bits(130));
    assert_eq!(whole.end(), 130);
}

#[test]
fn slices_order_by_offset_before_width() {
    assert!(BitSlice::new(0, bits(64)) < BitSlice::new(1, bits(1)));
    assert!(BitSlice::new(8, bits(8)) < BitSlice::new(8, bits(9)));
}

#[test]
fn the_whole_place_of_every_root_covers_its_width_in_both_dialects() {
    for root in Reg::ALL {
        let place = Place::<Kestrel>::whole(root);
        assert_eq!(place.root, root);
        assert_eq!(place.slice, BitSlice::whole(Kestrel::root_width(root)));
    }
    for id in 0..10 {
        let root = Slot::of(id);
        let place = Place::<Tessera>::whole(root);
        assert_eq!(place.root, root);
        assert_eq!(place.slice.offset, 0);
        assert_eq!(place.slice.width, Tessera::root_width(root));
    }
    assert_eq!(Place::<Kestrel>::whole(Reg::Wide).slice.width.words(), 3);
    assert_eq!(Place::<Tessera>::whole(Slot::of(8)).slice.width.words(), 4);
}

#[test]
fn places_order_by_root_before_slice() {
    let low = Place::<Kestrel> {
        root: Reg::R0,
        slice: BitSlice::new(8, bits(8)),
    };
    let high = Place::<Kestrel> {
        root: Reg::R1,
        slice: BitSlice::new(0, bits(8)),
    };
    assert!(low < high);
}
