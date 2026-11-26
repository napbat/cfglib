//! Width relaxation for symbolic linear layouts.
//!
//! Encoding a linearized function against real branch encodings is a fixed
//! point: an item's width depends on where its targets land, and where
//! everything lands depends on every width. Assemblers solve it by
//! *relaxation* — start every branch in its narrow form, lay items out,
//! resolve labels, widen whatever no longer reaches, and repeat until
//! nothing changes. Two relaxers share that fixed point:
//!
//! - [`relax_layout`] owns the loop generically: the caller keeps its item
//!   vocabulary, width table, label binding, and widening rule, and gets
//!   back the converged offsets, total length, and its own final label
//!   context. Widths may depend on the item's own offset (alignment
//!   padding), which is why offsets are recomputed from scratch each round.
//! - [`relax_spans`] is the incremental form for items whose only dependency
//!   is the displacement to one other item ([`SpanItem`]): a round re-checks
//!   only the spans over an item whose width changed. It also relaxes from
//!   the other side ([`RelaxMode::Shrink`]): every item starts long and
//!   shrinks where a shorter form reaches, as a linker does.
//!
//! Termination is the caller's monotonicity: widening must only grow an
//! item (never shrink it back), so each iteration either widens something
//! or converges.

use core::convert::Infallible;
use core::fmt;

pub use general::relax_layout;
pub use spans::{RelaxMode, SpanItem, relax_spans};

mod general;
mod spans;

/// Why one [`relax_layout`] or [`relax_spans`] run failed.
///
/// `E` is the error of the caller hooks of [`relax_layout`]. [`relax_spans`]
/// has no fallible hook, so it reports a `RelaxError` with the default `E`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelaxError<E = Infallible> {
    /// The layout's total length overflowed `usize`, or a displacement of
    /// [`relax_spans`] overflowed `isize`.
    Overflow,
    /// A caller hook failed.
    Item(E),
}

impl<E: fmt::Display> fmt::Display for RelaxError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => formatter.write_str("layout length overflows the address space"),
            Self::Item(error) => error.fmt(formatter),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> core::error::Error for RelaxError<E> {}
