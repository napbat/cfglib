//! Label hooks shared by the graph output formats.
//!
//! Both renderers — [`dot`](super::dot) and [`text`](super::text) — ask the
//! same question of a consumer: given one identity and the payload it
//! addresses, what text stands for it? [`Label`] is that question, and the
//! two supplied answers cover the ordinary cases: render nothing, or render
//! the payload with [`Display`](core::fmt::Display).

extern crate alloc;
use alloc::borrow::Cow;
use alloc::format;
use core::fmt;

/// Text standing for one identity and the payload it addresses.
///
/// The return may borrow the payload, so a store that already holds rendered
/// text labels its entities without copying.
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

/// Bind a label closure to the higher-ranked signature the writers require.
///
/// A closure that returns text built from the payload it was handed infers
/// one fixed payload lifetime unless its signature is stated first. Passing
/// it through this function states it:
///
/// ```
/// use std::borrow::Cow;
/// use cfglib::{Graph, NodeId, TextStyle, to_text};
///
/// let mut graph = Graph::<u32, ()>::new();
/// graph.add_node(7);
///
/// let labels = cfglib::bind_label::<NodeId, u32, _>(|_, value| {
///     Cow::Owned(format!("{value:#x}"))
/// });
/// let text = to_text(&graph, &TextStyle::new(labels, cfglib::no_label));
/// assert_eq!(text, "n0 0x7\n");
/// ```
#[must_use]
pub fn bind_label<Id, Data, Hook>(hook: Hook) -> Hook
where
    Data: ?Sized,
    Hook: for<'d> Fn(Id, &'d Data) -> Cow<'d, str>,
{
    hook
}
