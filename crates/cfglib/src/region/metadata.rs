//! Consumer-owned handler data: the side table keyed by [`HandlerRef`].

extern crate alloc;
use alloc::vec::Vec;

use super::HandlerRef;

/// Consumer-owned metadata for handlers, keyed by [`HandlerRef`].
///
/// [`HandlerKind::Filter`](super::HandlerKind::Filter) names a *block* holding
/// the predicate, which fits bytecode where a filter is its own funclet.
/// Source lowerings frequently have no such block: C#'s
/// `catch (E e) when (cond)` evaluates its predicate in the pad itself, so the
/// handler registers as [`HandlerKind::Catch`](super::HandlerKind::Catch) and
/// the only useful description of the filter is whatever identifies the
/// predicate in the frontend's own world — a syntax-node id, an interned
/// expression, a caught type, a bitmask. That payload is therefore consumer
/// DATA rather than a library type, and it is orthogonal to
/// [`HandlerKind`](super::HandlerKind): any handler may carry one. The same
/// table also carries a CLR catch-type token without forcing that token's type
/// into [`Handler`](super::Handler) or [`Cfg`](crate::Cfg). Use the
/// [`HandlerFilters`] and [`HandlerTypes`] aliases when the role should be
/// explicit at a call site.
///
/// It is a **side table** and not a field or a type parameter on
/// [`Handler`](super::Handler) on purpose. A `Handler<F>` would force
/// `Region<F>` and `Cfg<I, F>`, infecting every algorithm signature in the
/// crate, and a new field would break every existing struct-literal
/// construction. A separate table keeps [`Cfg`](crate::Cfg) object-simple,
/// keeps existing frontends compiling, and is dropped by consumers that have
/// no handler metadata at all.
///
/// Storage is a sorted `Vec` of pairs rather than a map: `serde` renders it
/// as a sequence, so it survives formats that only admit string map keys,
/// and lookups stay a binary search. A serialized table whose entries are not
/// in strictly ascending handler order is rejected at decode, because the
/// lookups would silently miss a key it holds.
///
/// # Examples
///
/// ```
/// use cfglib::{HandlerFilters, HandlerRef, RegionId};
///
/// // `catch (IOException e) when (e.Retryable)` — a plain `Catch` handler
/// // whose predicate the frontend identifies by its own expression handle.
/// let handler = HandlerRef::new(RegionId::from_raw(0), 0);
/// let mut filters = HandlerFilters::new();
/// assert!(filters.set(handler, "expr#17").is_none());
///
/// assert_eq!(filters.metadata(handler), Some(&"expr#17"));
/// assert_eq!(filters.len(), 1);
/// // Re-registering returns the payload it replaced.
/// assert_eq!(filters.set(handler, "expr#18"), Some("expr#17"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        try_from = "document::MetadataDocument<M>",
        bound(deserialize = "M: serde::Deserialize<'de>")
    )
)]
pub struct HandlerMetadata<M> {
    /// Sorted by key, so lookups binary-search and iteration is stable.
    entries: Vec<(HandlerRef, M)>,
}

impl<M> HandlerMetadata<M> {
    /// An empty table.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Attach `metadata` to `handler`, returning the payload it replaced.
    pub fn set(&mut self, handler: HandlerRef, metadata: M) -> Option<M> {
        match self.entries.binary_search_by_key(&handler, |(key, _)| *key) {
            Ok(at) => Some(core::mem::replace(&mut self.entries[at].1, metadata)),
            Err(at) => {
                self.entries.insert(at, (handler, metadata));
                None
            }
        }
    }

    /// The metadata attached to `handler`, if any.
    #[must_use]
    pub fn metadata(&self, handler: HandlerRef) -> Option<&M> {
        let at = self
            .entries
            .binary_search_by_key(&handler, |(key, _)| *key)
            .ok()?;
        Some(&self.entries[at].1)
    }

    /// Mutable access to the metadata attached to `handler`.
    pub fn metadata_mut(&mut self, handler: HandlerRef) -> Option<&mut M> {
        let at = self
            .entries
            .binary_search_by_key(&handler, |(key, _)| *key)
            .ok()?;
        Some(&mut self.entries[at].1)
    }

    /// Detach and return the metadata attached to `handler`.
    pub fn remove(&mut self, handler: HandlerRef) -> Option<M> {
        let at = self
            .entries
            .binary_search_by_key(&handler, |(key, _)| *key)
            .ok()?;
        Some(self.entries.remove(at).1)
    }

    /// The number of handlers carrying metadata.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no handler carries metadata.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over every `(handler, metadata)` pair in handler order.
    pub fn iter(&self) -> impl Iterator<Item = (HandlerRef, &M)> {
        self.entries.iter().map(|(key, metadata)| (*key, metadata))
    }
}

impl<M> Default for HandlerMetadata<M> {
    fn default() -> Self {
        Self::new()
    }
}

/// Handler metadata used as a filter-predicate side table.
pub type HandlerFilters<F> = HandlerMetadata<F>;

/// Handler metadata used as a caught-type side table.
pub type HandlerTypes<T> = HandlerMetadata<T>;

/// The serialized form of a table and the order a document must keep true
/// before it becomes one.
#[cfg(feature = "serde")]
mod document {
    extern crate alloc;
    use alloc::vec::Vec;

    use super::{HandlerMetadata, HandlerRef};

    /// The fields a [`HandlerMetadata`] serializes, before they are checked.
    #[derive(serde::Deserialize)]
    #[serde(bound(deserialize = "M: serde::Deserialize<'de>"))]
    pub(super) struct MetadataDocument<M> {
        entries: Vec<(HandlerRef, M)>,
    }

    /// A handler that does not come after the one listed before it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) struct OutOfOrder(HandlerRef);

    impl core::fmt::Display for OutOfOrder {
        fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(
                formatter,
                "handler {} is out of order: entries must list each handler once, in ascending order",
                self.0
            )
        }
    }

    impl<M> TryFrom<MetadataDocument<M>> for HandlerMetadata<M> {
        type Error = OutOfOrder;

        fn try_from(document: MetadataDocument<M>) -> Result<Self, OutOfOrder> {
            match document
                .entries
                .windows(2)
                .find(|pair| pair[0].0 >= pair[1].0)
            {
                Some(pair) => Err(OutOfOrder(pair[1].0)),
                None => Ok(Self {
                    entries: document.entries,
                }),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::region::RegionId;

    fn handler(region: u32, index: usize) -> HandlerRef {
        HandlerRef::new(RegionId::from_raw(region), index)
    }

    #[test]
    fn handler_filters_attach_consumer_payloads_to_any_handler_kind() {
        // The filtered `catch` registers as a plain `Catch` — its predicate
        // lives in the pad — so the payload is the frontend's own handle.
        let catch = handler(0, 0);
        let finally = handler(0, 1);

        let mut filters: HandlerFilters<&str> = HandlerFilters::new();
        assert!(filters.is_empty());
        assert!(filters.set(finally, "unreachable").is_none());
        assert!(filters.set(catch, "e.Retryable").is_none());

        assert_eq!(filters.len(), 2);
        assert_eq!(filters.metadata(catch), Some(&"e.Retryable"));
        assert_eq!(
            filters.set(catch, "e.Retryable && !e.Fatal"),
            Some("e.Retryable")
        );
        if let Some(filter) = filters.metadata_mut(catch) {
            *filter = "e.Fatal";
        }
        assert_eq!(filters.metadata(catch), Some(&"e.Fatal"));
        // Iteration is in handler order regardless of insertion order.
        assert_eq!(
            filters.iter().map(|(key, _)| key).collect::<Vec<_>>(),
            vec![catch, finally]
        );

        assert_eq!(filters.remove(finally), Some("unreachable"));
        assert_eq!(filters.remove(finally), None);
        assert_eq!(filters.len(), 1);
        assert_eq!(
            HandlerFilters::<u32>::default(),
            HandlerFilters::<u32>::new()
        );
    }

    /// The table `{r2.h0, r0.h0, r1.h0}` as a document whose `entries` array
    /// is rewritten by `arrange`.
    #[cfg(feature = "serde")]
    fn document_with(arrange: impl FnOnce(&mut Vec<serde_json::Value>)) -> serde_json::Value {
        let mut table = HandlerTypes::new();
        for region in [2, 0, 1] {
            table.set(handler(region, 0), region);
        }
        let mut document = serde_json::to_value(&table).expect("a table serializes");
        arrange(
            document["entries"]
                .as_array_mut()
                .expect("entries serialize as a sequence"),
        );
        document
    }

    #[cfg(feature = "serde")]
    #[test]
    fn a_serialized_table_reads_back_equal() {
        let document = document_with(|_| {});
        let table: HandlerTypes<u32> = serde_json::from_value(document).expect("sorted entries");
        assert_eq!(table.len(), 3);
        for region in 0..3 {
            assert_eq!(table.metadata(handler(region, 0)), Some(&region));
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn entries_out_of_handler_order_are_rejected_at_decode() {
        let reversed = document_with(|entries| entries.reverse());
        assert!(
            serde_json::from_value::<HandlerTypes<u32>>(reversed).is_err(),
            "a table whose lookups would miss its own keys must not load"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn a_repeated_handler_is_rejected_at_decode() {
        let repeated = document_with(|entries| {
            let first = entries[0].clone();
            entries.insert(1, first);
        });
        assert!(
            serde_json::from_value::<HandlerTypes<u32>>(repeated).is_err(),
            "two payloads for one handler must not load"
        );
    }
}
