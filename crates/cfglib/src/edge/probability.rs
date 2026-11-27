//! The branch probability an edge stores.

use core::fmt;

/// The most fractional digits [`Probability`]'s decimal form needs: ten
/// digits resolve steps of `10^-10`, finer than half of `2^-31`.
const MAX_DIGITS: usize = 10;

/// `"0."` and [`MAX_DIGITS`] digits.
const DECIMAL_CAPACITY: usize = 2 + MAX_DIGITS;

/// A branch probability in fixed point: `numerator / 2^31`.
///
/// An [`Edge`](crate::Edge) stores one as its weight: the likelihood that
/// control takes the edge, relative to the other edges that leave its source.
/// Its [`Display`](fmt::Display) form is the shortest decimal that
/// [`from_f64`](Self::from_f64) reads back as the same numerator, so `0.9`
/// prints as `0.9`; the DOT form and the `serde` form use it.
///
/// # Examples
///
/// ```
/// use cfglib::Probability;
///
/// let half = Probability::from_f64(0.5).expect("0.5 is a probability");
/// assert_eq!(half.numerator(), Probability::DENOMINATOR / 2);
/// assert_eq!(Probability::from_f64(1.5), None);
/// assert_eq!(Probability::from_f64(f64::NAN), None);
///
/// let likely = Probability::from_f64(0.9).expect("0.9 is a probability");
/// assert_eq!(likely.to_string(), "0.9");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Probability(u32);

impl Probability {
    /// The denominator of every probability, `2^31`.
    pub const DENOMINATOR: u32 = 1 << 31;

    /// The probability of an edge that is never taken.
    pub const ZERO: Self = Self(0);

    /// The probability of an edge that is always taken.
    pub const ONE: Self = Self(Self::DENOMINATOR);

    /// The probability `numerator / 2^31`, or `None` when `numerator` is
    /// above [`DENOMINATOR`](Self::DENOMINATOR).
    #[must_use]
    pub const fn from_numerator(numerator: u32) -> Option<Self> {
        if numerator <= Self::DENOMINATOR {
            Some(Self(numerator))
        } else {
            None
        }
    }

    /// Round `probability` to the nearest fixed-point probability, ties
    /// upward, or `None` when it is NaN or outside `0.0..=1.0`.
    #[must_use]
    pub fn from_f64(probability: f64) -> Option<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return None;
        }
        // Scaling by a power of two is exact, and so is the fraction of a
        // value below 2^52, so the rounding is exact too.
        let scaled = probability * f64::from(Self::DENOMINATOR);
        let floor = truncate(scaled);
        let round_up = scaled - f64::from(floor) >= 0.5;
        Some(Self(floor + u32::from(round_up)))
    }

    /// The numerator over [`DENOMINATOR`](Self::DENOMINATOR).
    #[must_use]
    pub const fn numerator(self) -> u32 {
        self.0
    }

    /// The probability as a float, exactly: `numerator / 2^31`.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        f64::from(self.0) / f64::from(Self::DENOMINATOR)
    }

    /// The nearest decimal with `digits` fractional digits, as an integer
    /// over `10^digits`, ties upward.
    fn nearest_decimal(self, digits: usize) -> u128 {
        let denominator = u128::from(Self::DENOMINATOR);
        (u128::from(self.0) * power_of_ten(digits) + denominator / 2) / denominator
    }

    /// Writes the shortest decimal that reads back as `self` into `buffer`
    /// and returns its length.
    ///
    /// The nearest decimal of each length is tried, shortest first. The
    /// nearest decimal of [`MAX_DIGITS`] digits is within `0.11 / 2^31` of
    /// the probability, far inside the half step that rounds back to it, so
    /// it ends the search unchecked. For a numerator in `1..2^31` it lies
    /// strictly between `0` and `10^MAX_DIGITS`, so it has a fraction to
    /// write.
    fn write_shortest(self, buffer: &mut [u8; DECIMAL_CAPACITY]) -> usize {
        if self.0 == 0 || self.0 == Self::DENOMINATOR {
            buffer[0] = if self.0 == 0 { b'0' } else { b'1' };
            return 1;
        }
        for digits in 1..MAX_DIGITS {
            let decimal = self.nearest_decimal(digits);
            if decimal == 0 || decimal == power_of_ten(digits) {
                continue;
            }
            let len = write_fraction(buffer, decimal, digits);
            if Self::read(&buffer[..len]) == Some(self) {
                return len;
            }
        }
        write_fraction(buffer, self.nearest_decimal(MAX_DIGITS), MAX_DIGITS)
    }

    /// The probability that the ASCII decimal `text` reads as.
    fn read(text: &[u8]) -> Option<Self> {
        core::str::from_utf8(text)
            .ok()?
            .parse::<f64>()
            .ok()
            .and_then(Self::from_f64)
    }
}

/// `10^digits`.
fn power_of_ten(digits: usize) -> u128 {
    (0..digits).fold(1, |power, _| power * 10)
}

/// Writes `0.` and `decimal` as `digits` fractional digits, without trailing
/// zeros, and returns the length. `decimal` is nonzero and below
/// `10^digits`.
fn write_fraction(buffer: &mut [u8; DECIMAL_CAPACITY], mut decimal: u128, digits: usize) -> usize {
    buffer[0] = b'0';
    buffer[1] = b'.';
    let mut len = 2 + digits;
    for slot in (2..len).rev() {
        buffer[slot] = b'0' + digit(decimal % 10);
        decimal /= 10;
    }
    while buffer[len - 1] == b'0' {
        len -= 1;
    }
    len
}

/// The digit `value`, which is below ten.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is a remainder by ten"
)]
const fn digit(value: u128) -> u8 {
    value as u8
}

/// The integer part of a value in `0.0..=2^31`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is a scaled probability in 0.0..=2^31, whose integer part fits u32"
)]
fn truncate(scaled: f64) -> u32 {
    scaled as u32
}

impl fmt::Display for Probability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buffer = [0; DECIMAL_CAPACITY];
        let len = self.write_shortest(&mut buffer);
        let text = core::str::from_utf8(&buffer[..len]).map_err(|_| fmt::Error)?;
        formatter.pad(text)
    }
}

/// A probability serializes as the float of its shortest decimal, so a
/// JSON document shows `0.9`, and deserializes through
/// [`Probability::from_f64`].
#[cfg(feature = "serde")]
impl serde::Serialize for Probability {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut buffer = [0; DECIMAL_CAPACITY];
        let len = self.write_shortest(&mut buffer);
        let value = core::str::from_utf8(&buffer[..len])
            .ok()
            .and_then(|text| text.parse::<f64>().ok())
            .ok_or_else(|| serde::ser::Error::custom("a probability has no decimal form"))?;
        serializer.serialize_f64(value)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Probability {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = f64::deserialize(deserializer)?;
        Self::from_f64(value).ok_or_else(|| {
            serde::de::Error::invalid_value(
                serde::de::Unexpected::Float(value),
                &"a probability in 0.0..=1.0",
            )
        })
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::string::ToString;

    use super::{Probability, power_of_ten};
    use crate::test_util::rng::Rng;

    fn of(value: f64) -> Probability {
        Probability::from_f64(value).expect("a probability")
    }

    /// Whether `text` reads back as `probability`.
    fn reads_back(text: &str, probability: Probability) -> bool {
        text.parse::<f64>().ok().and_then(Probability::from_f64) == Some(probability)
    }

    #[test]
    fn short_decimals_print_as_written() {
        for (value, text) in [
            (0.0, "0"),
            (1.0, "1"),
            (0.5, "0.5"),
            (0.9, "0.9"),
            (0.1, "0.1"),
            (0.75, "0.75"),
            (0.875, "0.875"),
            (0.3, "0.3"),
            (0.999, "0.999"),
            (0.0001, "0.0001"),
        ] {
            assert_eq!(of(value).to_string(), text, "{value}");
        }
        let smallest = Probability::from_numerator(1).expect("1 is a numerator");
        assert_eq!(smallest.to_string(), "0.0000000005");
        assert_eq!(alloc::format!("[{:>5}]", of(0.5)), "[  0.5]");
    }

    #[test]
    fn every_decimal_reads_back_and_no_shorter_one_does() {
        let mut rng = Rng::new(17);
        let edges = [
            1,
            2,
            3,
            Probability::DENOMINATOR / 3,
            Probability::DENOMINATOR / 2 - 1,
            Probability::DENOMINATOR / 2 + 1,
            Probability::DENOMINATOR - 2,
            Probability::DENOMINATOR - 1,
        ];
        let draws = (0..20_000).map(|_| {
            let bound = usize::try_from(Probability::DENOMINATOR).expect("2^31 fits usize");
            u32::try_from(rng.below(bound + 1)).expect("a draw fits u32")
        });
        for numerator in edges.into_iter().chain(draws) {
            let probability = Probability::from_numerator(numerator).expect("in range");
            let text = probability.to_string();
            assert!(reads_back(&text, probability), "{numerator} prints {text}");
            assert!(text.len() <= 12, "{numerator} prints {text}");
            // The nearest decimal one digit shorter reads as another
            // probability, so no shorter decimal reads back as this one.
            if let Some(fraction) = text.strip_prefix("0.")
                && fraction.len() > 1
            {
                let digits = fraction.len() - 1;
                let decimal = probability.nearest_decimal(digits);
                let shorter = if decimal >= power_of_ten(digits) {
                    "1".to_string()
                } else {
                    alloc::format!("0.{decimal:0>digits$}")
                };
                assert!(
                    !reads_back(&shorter, probability),
                    "{numerator} prints {text}, but {shorter} reads back too"
                );
            }
        }
    }

    #[test]
    fn the_float_form_is_exact() {
        assert_eq!(Probability::ONE.to_f64(), 1.0);
        assert_eq!(Probability::ZERO.to_f64(), 0.0);
        assert_eq!(of(0.25).to_f64(), 0.25);
        let third = of(1.0 / 3.0);
        assert_eq!(Probability::from_f64(third.to_f64()), Some(third));
    }

    #[test]
    fn probabilities_round_to_the_nearest_fixed_point_value() {
        assert_eq!(Probability::from_f64(0.0), Some(Probability::ZERO));
        assert_eq!(Probability::from_f64(-0.0), Some(Probability::ZERO));
        assert_eq!(Probability::from_f64(1.0), Some(Probability::ONE));
        assert_eq!(
            Probability::from_f64(0.9).map(Probability::numerator),
            Some(1_932_735_283)
        );
        let half_unit = 0.5 / f64::from(Probability::DENOMINATOR);
        assert_eq!(
            Probability::from_f64(half_unit).map(Probability::numerator),
            Some(1),
            "a tie rounds upward"
        );
        for outside in [-0.1, 1.000_001, f64::NAN, f64::INFINITY] {
            assert_eq!(Probability::from_f64(outside), None, "{outside}");
        }
        assert_eq!(
            Probability::from_numerator(Probability::DENOMINATOR),
            Some(Probability::ONE)
        );
        assert_eq!(
            Probability::from_numerator(Probability::DENOMINATOR + 1),
            None
        );
    }
}
