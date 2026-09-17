//! Crate-internal macro for defining dense `u32`-backed identity newtypes.
//!
//! This is for a dense index into a plain array — a region, a datum, a
//! reference, a partial path, an IR statement. An identity that addresses a
//! *store* entity is [`Id<T>`](crate::Id) instead, so the store mints it and
//! one definition carries ordering, hashing, display, and serialization for
//! every kind of node and edge in the crate.

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
/// - `raw;` adds the `raw` accessor returning the backing `u32`.
/// - a doc comment followed by `from_index = "overflow message";` adds the
///   panicking `from_index` constructor; the doc comment (including its
///   `# Panics` section) and the `expect` message come from the call site.
///
/// Type-specific extras — `DenseId`/`DenseId` impls, private
/// helpers, extra constructors — stay outside the macro, written next to
/// the invocation.
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
        raw;
        $($rest:tt)*
    ) => {
        impl $name {
            /// Returns the compact raw identity.
            #[inline]
            #[must_use]
            pub const fn raw(self) -> u32 {
                self.0
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
