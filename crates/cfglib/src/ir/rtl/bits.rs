//! The RTL value type and bit ranges within a storage root.

use core::fmt;
use core::num::NonZeroU16;

/// The type of every RTL value: a signless bit vector of a fixed width.
///
/// The width is in `1..=`[`Bits::MAX_WIDTH`]. Signedness, a floating format,
/// and a lane structure are properties of the operator that reads the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bits(NonZeroU16);

impl Bits {
    /// The largest width of one RTL value.
    pub const MAX_WIDTH: u16 = 8192;

    /// The one-bit type, which is also the Boolean type.
    pub const BOOL: Self = Self(NonZeroU16::MIN);

    /// Creates a bit-vector type.
    ///
    /// Returns `None` when `width` is zero or larger than
    /// [`Self::MAX_WIDTH`].
    #[must_use]
    pub const fn new(width: u16) -> Option<Self> {
        if width > Self::MAX_WIDTH {
            return None;
        }
        match NonZeroU16::new(width) {
            Some(width) => Some(Self(width)),
            None => None,
        }
    }

    /// Returns the width in bits.
    #[must_use]
    pub const fn width(self) -> u16 {
        self.0.get()
    }

    /// Returns how many 64-bit words hold one value of this type.
    #[must_use]
    pub const fn words(self) -> usize {
        (self.0.get() as usize).div_ceil(64)
    }
}

impl fmt::Display for Bits {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "bits<{}>", self.0)
    }
}

/// A contiguous range of bits inside one storage root.
///
/// The range starts at bit `offset`, counted from the least significant bit,
/// and holds `width` bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BitSlice {
    /// The first bit of the range.
    pub offset: u16,
    /// The number of bits in the range, which is also the type of the value
    /// that the range holds.
    pub width: Bits,
}

impl BitSlice {
    /// Creates a range of `width` bits that starts at `offset`.
    #[must_use]
    pub const fn new(offset: u16, width: Bits) -> Self {
        Self { offset, width }
    }

    /// Returns the range that covers a whole root of type `width`.
    #[must_use]
    pub const fn whole(width: Bits) -> Self {
        Self { offset: 0, width }
    }

    /// Returns the bit after the last bit of the range.
    #[must_use]
    pub const fn end(self) -> u32 {
        self.offset as u32 + self.width.width() as u32
    }
}
