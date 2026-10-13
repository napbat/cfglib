//! Bit-field copies over little-endian 64-bit words.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec;

use crate::ir::rtl::Bits;

/// Copies `width` bits of `source` to `target`.
///
/// The copy reads from bit `source_offset` of `source` and writes from bit
/// `target_offset` of `target`. Bits of `source` past its last word read as
/// zero. Other bits of `target` keep their value.
///
/// # Panics
///
/// Panics when `target` is too short for the written range.
pub(super) fn copy_bits(
    target: &mut [u64],
    target_offset: usize,
    source: &[u64],
    source_offset: usize,
    width: usize,
) {
    let mut copied = 0;
    while copied < width {
        let to = target_offset + copied;
        let from = source_offset + copied;
        let to_shift = to % 64;
        let from_shift = from % 64;
        let chunk = (width - copied).min(64 - to_shift).min(64 - from_shift);
        let mask = if chunk == 64 {
            u64::MAX
        } else {
            (1_u64 << chunk) - 1
        };
        let bits = (source.get(from / 64).copied().unwrap_or(0) >> from_shift) & mask;
        let word = &mut target[to / 64];
        *word = (*word & !(mask << to_shift)) | (bits << to_shift);
        copied += chunk;
    }
}

/// Returns the low `width` bits of `words` as a value of type `width`: the
/// result has [`Bits::words`] words and its bits past the width are zero.
pub(super) fn normalized(words: &[u64], width: Bits) -> Box<[u64]> {
    let mut value = vec![0; width.words()].into_boxed_slice();
    copy_bits(&mut value, 0, words, 0, usize::from(width.width()));
    value
}

/// Returns the number that `words` hold, or `None` when it exceeds `usize`.
pub(super) fn to_usize(words: &[u64]) -> Option<usize> {
    let (&low, high) = words.split_first()?;
    if high.iter().any(|&word| word != 0) {
        return None;
    }
    usize::try_from(low).ok()
}
