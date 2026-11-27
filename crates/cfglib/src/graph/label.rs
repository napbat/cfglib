//! Label hooks shared by the graph output formats.
//!
//! Both renderers — [`dot`](super::dot) and [`text`](super::text) — ask the
//! same question of a consumer: given one identity and the payload it
//! addresses, what text stands for it? A hook is any closure or function item
//! of the shape `for<'d> Fn(Id, &'d Data) -> Cow<'d, str>`, and the two
//! supplied answers cover the ordinary cases: render nothing, or render the
//! payload with [`Display`](core::fmt::Display).

extern crate alloc;
use alloc::borrow::Cow;
use alloc::format;
use core::fmt;

/// Text standing for one identity and the payload it addresses, as a plain
/// function pointer.
///
/// This is the hook type of the default styles (`plain()` and the store's
/// own), which have to name their hooks. A consumer style takes any closure of
/// the same shape. The return may borrow the payload, so a store that already
/// holds rendered text labels its entities without copying.
pub type Label<Id, Data> = for<'d> fn(Id, &'d Data) -> Cow<'d, str>;

/// A label hook that renders nothing.
///
/// The entity keeps its identifier, so the output still reads as a graph.
#[must_use]
pub fn no_label<Id, Data: ?Sized>(_id: Id, _data: &Data) -> Cow<'_, str> {
    Cow::Borrowed("")
}

/// A label hook that renders the payload with [`fmt::Display`].
#[must_use]
pub fn display_label<Id, Data>(_id: Id, data: &Data) -> Cow<'_, str>
where
    Data: fmt::Display + ?Sized,
{
    Cow::Owned(format!("{data}"))
}
