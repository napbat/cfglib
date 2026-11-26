//! Crate-internal dense identities and dense `u32` tables.
//!
//! [`define_dense_id!`] defines a dense `u32`-backed identity newtype: a dense
//! index into a plain array — a region, a datum, a reference, a partial path,
//! a control-tree node. An identity that addresses a *store* entity is
//! [`Id<T>`](crate::Id) instead, so the store mints it and one definition
//! carries ordering, hashing, display, and serialization for every kind of
//! node and edge in the crate.
//!
//! The functions below are the one statement of how an analysis stores a
//! dense index in a `u32` table: [`check_bound`] once on entry, [`word`] to
//! store, [`slot`] to load, and [`NO_INDEX`] for a slot that names nothing.
//! Halving the width of a table that holds node indices is what makes the
//! tables worth keeping in `u32`; stating the rules once keeps every table's
//! sentinel and range check the same.

/// The word a dense `u32` table stores for "no index": no node, no edge, no
/// class, not yet visited.
///
/// [`check_bound`] keeps it out of every bound, so it never names a real slot.
pub(crate) const NO_INDEX: u32 = u32::MAX;

/// The word a dense `u32` table stores for `index`.
///
/// # Panics
///
/// Panics when `index` does not fit in a `u32`. A table whose bound passed
/// [`check_bound`] never panics here, so a failure is a view yielding an
/// index past its own bound.
#[inline]
#[must_use]
pub(crate) fn word(index: usize) -> u32 {
    u32::try_from(index).expect("a dense index below a checked bound fits in u32")
}

/// The dense index a stored word names.
#[inline]
#[must_use]
pub(crate) const fn slot(word: u32) -> usize {
    word as usize
}

/// Reject a bound that a dense `u32` table cannot index, naming `what` in the
/// message.
///
/// Every index below an accepted bound is below `u32::MAX`, so it fits in a
/// [`word`] and never collides with [`NO_INDEX`].
///
/// # Panics
///
/// Panics when `bound` exceeds `u32::MAX`.
#[track_caller]
pub(crate) fn check_bound(bound: usize, what: &str) {
    assert!(
        u32::try_from(bound).is_ok(),
        "{what} exceeds the dense u32 range"
    );
}

/// Defines a dense identity newtype backed by a `u32`.
///
/// Generates the newtype struct with the standard identity derive set
/// (`Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, `Hash`, `PartialOrd`,
/// `Ord`) plus the `from_raw` and `index` accessors. Attributes written on
/// the struct — doc comments and `serde` `cfg_attr`s — pass through
/// unchanged, as does the tuple field's visibility.
///
/// Optional trailing items, each on its own line and in this order:
///
/// - `display = "prefix";` implements `core::fmt::Display` as the prefix
///   followed by the raw value.
/// - a doc comment followed by `from_index = "overflow message";` adds the
///   panicking `from_index` constructor; the doc comment (including its
///   `# Panics` section) and the `expect` message come from the call site.
///
/// Type-specific extras — the `DenseId` impl of an identity that is a graph
/// node, private helpers, extra constructors — stay outside the macro,
/// written next to the invocation.
macro_rules! define_dense_id {
    (
        $(#[$meta:meta])*
        pub struct $name:ident($fvis:vis u32);
        $($extras:tt)*
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name($fvis u32);

        impl $name {
            /// Creates an identity from its dense raw index.
            #[inline]
            #[must_use]
            pub const fn from_raw(raw: u32) -> Self {
                Self(raw)
            }

            /// Returns the dense zero-based index.
            #[inline]
            #[must_use]
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }

        crate::identity::define_dense_id! { @extras $name; $($extras)* }
    };
    (@extras $name:ident;) => {};
    (@extras $name:ident;
        display = $prefix:literal;
        $($rest:tt)*
    ) => {
        impl ::core::fmt::Display for $name {
            fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::write!(formatter, "{}{}", $prefix, self.0)
            }
        }

        crate::identity::define_dense_id! { @extras $name; $($rest)* }
    };
    (@extras $name:ident;
        $(#[$fmeta:meta])+
        from_index = $overflow:literal;
        $($rest:tt)*
    ) => {
        impl $name {
            $(#[$fmeta])+
            #[must_use]
            pub fn from_index(index: usize) -> Self {
                Self(u32::try_from(index).expect($overflow))
            }
        }

        crate::identity::define_dense_id! { @extras $name; $($rest)* }
    };
}

pub(crate) use define_dense_id;
