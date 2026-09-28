//! Lane constraints and value shapes carried by RTL expressions.
//!
//! The constraint domain is dialect-owned: numeric machine lanes use the
//! provided [`ScalarType`], while managed-language dialects supply richer
//! domains — exact and unknown reference types, null, verifier-polymorphic
//! zero, uninitialized objects, hierarchy-dependent merges — by
//! implementing [`Constraint`] on their own type. cfglib owns the folding
//! inference ([`Inference`]) and the web mechanics; the dialect owns what
//! the constraints mean and how they merge. Consumer-facing typing enters
//! at MLIL through [`Lift::value_type`](super::Lift::value_type).

use core::fmt::Debug;
use core::num::NonZeroU16;

/// One dialect-owned lane constraint domain.
///
/// Web typing folds every observation of a web (read interpretations and
/// assigned value shapes) through [`Inference`], which relies on this
/// contract. `merge` must be commutative, and merging equal constraints
/// must yield that constraint back.
pub trait Constraint: Clone + Debug + Eq {
    /// External context merging consults — a class hierarchy, a type
    /// table — passed through [`lift`](super::lift()) by the caller.
    /// `()` when the constraint values are self-contained.
    type Context: ?Sized;

    /// The unconstrained element: observing it imposes nothing.
    fn free() -> Self;

    /// The constraint a genuinely conflicted web resolves to.
    fn conflicted() -> Self;

    /// The merge of two observations, or `None` for a genuine conflict.
    fn merge(&self, other: &Self, context: &Self::Context) -> Option<Self>;

    /// The lane width in bits, when the constraint fixes one.
    ///
    /// Reinterpretation validation treats an unknown width as compatible
    /// with anything, and constants carry
    /// [`word_count`](Self::word_count) 64-bit words per lane.
    fn width(&self) -> Option<u32> {
        None
    }

    /// The number of 64-bit words one lane's bit pattern occupies in a
    /// constant — the width rounded up to whole words, one word when the
    /// constraint fixes no width.
    fn word_count(&self) -> usize {
        self.width().map_or(1, |width| width.div_ceil(64) as usize)
    }
}

/// The interpretation of one numeric storage lane.
///
/// `Bits` is the unknown interpretation: raw storage whose meaning no
/// operation has constrained yet. Reads that impose `Bits` (transports,
/// bitwise moves) leave type inference untouched; a web whose reads
/// genuinely conflict (float against integer) also resolves to `Bits`,
/// and the consumer then renders explicit reinterpretations at each
/// access. This is the provided [`Constraint`] domain for machine-numeric
/// dialects; managed dialects define their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScalarType {
    /// 16-bit IEEE-754 float.
    F16,
    /// 32-bit IEEE-754 float.
    F32,
    /// 64-bit IEEE-754 float.
    F64,
    /// 80-bit x87 extended-precision float.
    F80,
    /// 128-bit IEEE-754 float.
    F128,
    /// 8-bit signed integer.
    I8,
    /// 16-bit signed integer.
    I16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 128-bit signed integer.
    I128,
    /// 256-bit signed integer.
    I256,
    /// 512-bit signed integer.
    I512,
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit unsigned integer.
    U16,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit unsigned integer.
    U64,
    /// 128-bit unsigned integer.
    U128,
    /// 256-bit unsigned integer.
    U256,
    /// 512-bit unsigned integer.
    U512,
    /// Integer with an explicit nonzero bit width.
    ///
    /// Prefer [`ScalarType::integer`] to obtain the canonical named variant
    /// for standard widths.
    Integer {
        /// Number of bits in the lane.
        bits: NonZeroU16,
        /// Whether arithmetic interprets the high bit as a sign bit.
        signed: bool,
    },
    /// Boolean truth value of dialect-defined width.
    Bool,
    /// Uninterpreted bits.
    Bits,
}

impl ScalarType {
    /// Returns the canonical integer constraint, or `None` for zero width.
    #[must_use]
    pub const fn integer(bits: u16, signed: bool) -> Option<Self> {
        Some(match (bits, signed) {
            (8, true) => Self::I8,
            (16, true) => Self::I16,
            (32, true) => Self::I32,
            (64, true) => Self::I64,
            (128, true) => Self::I128,
            (256, true) => Self::I256,
            (512, true) => Self::I512,
            (8, false) => Self::U8,
            (16, false) => Self::U16,
            (32, false) => Self::U32,
            (64, false) => Self::U64,
            (128, false) => Self::U128,
            (256, false) => Self::U256,
            (512, false) => Self::U512,
            (_, signed) => match NonZeroU16::new(bits) {
                Some(bits) => Self::Integer { bits, signed },
                None => return None,
            },
        })
    }

    /// Returns the signedness of an integer interpretation.
    #[must_use]
    pub const fn integer_signedness(self) -> Option<bool> {
        match self {
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::I128 | Self::I256 | Self::I512 => {
                Some(true)
            }
            Self::U8 | Self::U16 | Self::U32 | Self::U64 | Self::U128 | Self::U256 | Self::U512 => {
                Some(false)
            }
            Self::Integer { signed, .. } => Some(signed),
            _ => None,
        }
    }

    /// The lane width in bits, when the interpretation fixes one.
    ///
    /// `Bool` and `Bits` leave the width to the storage and return
    /// `None`; reinterpretation validation treats an unknown width as
    /// compatible with anything.
    #[must_use]
    pub const fn width(self) -> Option<u32> {
        match self {
            Self::I8 | Self::U8 => Some(8),
            Self::F16 | Self::I16 | Self::U16 => Some(16),
            Self::F32 | Self::I32 | Self::U32 => Some(32),
            Self::F64 | Self::I64 | Self::U64 => Some(64),
            Self::F80 => Some(80),
            Self::F128 | Self::I128 | Self::U128 => Some(128),
            Self::I256 | Self::U256 => Some(256),
            Self::I512 | Self::U512 => Some(512),
            Self::Integer { bits, .. } => Some(bits.get() as u32),
            Self::Bool | Self::Bits => None,
        }
    }

    /// The number of 64-bit words one lane's bit pattern occupies in a
    /// constant — `width` rounded up to whole words, one word when the
    /// interpretation fixes no width.
    #[must_use]
    pub const fn word_count(self) -> usize {
        match self.width() {
            Some(width) => width.div_ceil(64) as usize,
            None => 1,
        }
    }

    /// The signed form of a same-width signed/unsigned integer pair.
    ///
    /// The conversion between the two is value-representation-exact, so
    /// inference merges them instead of conflicting; the consumer can
    /// spell the difference as an ordinary conversion.
    fn integer_merge(self, other: Self) -> Option<Self> {
        let left = self.integer_signedness()?;
        let right = other.integer_signedness()?;
        let width = self.width()?;
        if width != other.width()? {
            return None;
        }
        Self::integer(u16::try_from(width).ok()?, left || right)
    }
}

impl Constraint for ScalarType {
    type Context = ();

    fn free() -> Self {
        Self::Bits
    }

    fn conflicted() -> Self {
        Self::Bits
    }

    fn merge(&self, other: &Self, (): &()) -> Option<Self> {
        if self == other {
            return Some(*self);
        }
        self.integer_merge(*other)
    }

    fn width(&self) -> Option<u32> {
        Self::width(*self)
    }
}

/// Folding constraint inference over one web's observations.
///
/// A three-point lattice — unconstrained, one known constraint, and
/// conflict — so a genuine conflict is never forgotten: unlike a pairwise
/// merge whose "unknown" and "conflict" share one value, `F32` then `U32`
/// then `F32` resolves to [`Constraint::conflicted`] here no matter the
/// observation order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Inference<C> {
    /// No observation has constrained the lane yet.
    #[default]
    Unseen,
    /// Every observation so far merged into one constraint.
    Known(C),
    /// Observations imposed unmergeable constraints.
    Conflict,
}

impl<C: Constraint> Inference<C> {
    /// Folds one observed constraint into the inference.
    ///
    /// [`Constraint::free`] observations impose nothing and leave the
    /// state untouched; unmergeable observations conflict permanently.
    pub fn observe(&mut self, constraint: &C, context: &C::Context) {
        if *constraint == C::free() {
            return;
        }
        *self = match &*self {
            Self::Unseen => Self::Known(constraint.clone()),
            Self::Known(known) => match known.merge(constraint, context) {
                Some(merged) => Self::Known(merged),
                None => Self::Conflict,
            },
            Self::Conflict => Self::Conflict,
        };
    }

    /// The inferred constraint: unconstrained webs resolve to
    /// [`Constraint::free`] and conflicted webs to
    /// [`Constraint::conflicted`].
    #[must_use]
    pub fn resolve(&self) -> C {
        match self {
            Self::Known(constraint) => constraint.clone(),
            Self::Unseen => C::free(),
            Self::Conflict => C::conflicted(),
        }
    }
}

/// Scalar-type inference — [`Inference`] over the numeric domain.
pub type ScalarInference = Inference<ScalarType>;

/// The shape of one value: a lane constraint across `lanes` lanes.
///
/// A lane's bit pattern travels as [`Constraint::word_count`] little-endian
/// 64-bit words in constants, so constraints up to 512 bits stay one
/// lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Shape<C> {
    /// Lane constraint.
    pub scalar: C,
    /// Number of lanes (1 for scalars).
    pub lanes: u8,
}

impl<C> Shape<C> {
    /// A one-lane shape.
    #[must_use]
    pub const fn scalar(scalar: C) -> Self {
        Self { scalar, lanes: 1 }
    }

    /// A multi-lane shape.
    #[must_use]
    pub const fn vector(scalar: C, lanes: u8) -> Self {
        Self { scalar, lanes }
    }
}

impl<C: Constraint> Shape<C> {
    /// Whether `bits` holds exactly this shape's constant words —
    /// [`Constraint::word_count`] words per lane across every lane.
    #[must_use]
    pub fn holds_words(&self, bits: &[u64]) -> bool {
        usize::from(self.lanes).checked_mul(self.scalar.word_count()) == Some(bits.len())
    }
}

/// A numeric value shape — [`Shape`] over [`ScalarType`].
pub type ValueShape = Shape<ScalarType>;

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec;

    use super::{Constraint, ScalarType, Shape};

    #[test]
    fn exact_integer_widths_preserve_constant_storage() {
        for (width, words) in [(1, 1), (6, 1), (65, 2), (80, 2), (513, 9)] {
            let scalar = ScalarType::integer(width, false).expect("nonzero width");
            assert_eq!(scalar.width(), Some(u32::from(width)));
            assert_eq!(scalar.word_count(), words);
            assert!(Shape::scalar(scalar).holds_words(&vec![0; words]));
            assert!(!Shape::scalar(scalar).holds_words(&vec![0; words + 1]));
        }
        assert_eq!(ScalarType::integer(0, false), None);
        assert_eq!(ScalarType::integer(32, false), Some(ScalarType::U32));
    }

    #[test]
    fn exact_integer_merges_preserve_width_and_signedness() {
        let unsigned = ScalarType::integer(6, false).expect("nonzero width");
        let signed = ScalarType::integer(6, true).expect("nonzero width");
        assert_eq!(unsigned.merge(&signed, &()), Some(signed));
        assert_eq!(signed.merge(&unsigned, &()), Some(signed));
        assert_eq!(unsigned.merge(&ScalarType::U8, &()), None);
        assert_eq!(unsigned.merge(&ScalarType::F80, &()), None);
        let explicit = ScalarType::Integer {
            bits: core::num::NonZeroU16::new(32).expect("nonzero width"),
            signed: false,
        };
        assert_eq!(explicit.merge(&ScalarType::I32, &()), Some(ScalarType::I32));
    }
}
