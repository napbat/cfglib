//! Tagged dense identities addressing one kind of stored entity.
//!
//! Every store in this crate mints a `u32`-backed newtype per entity kind and
//! then needs a conversion shim wherever two of them meet. [`Id`] replaces the
//! family with one type parameterized by a zero-sized tag, so a node identity
//! and an edge identity stay distinct at compile time while sharing a single
//! implementation of ordering, hashing, display, and serialization.

use core::fmt;
use core::hash::Hash;
use core::marker::PhantomData;

use crate::graph::edge_view::DenseEdgeId;
use crate::graph::view::DenseNodeId;

/// The entity kind a dense [`Id`] addresses.
///
/// A tag is a zero-sized marker; it exists only to keep identities of
/// different kinds from being confused and to give them a display prefix.
/// The supertraits are the ones every derive on a containing store needs, so
/// `#[derive(Clone, Debug, PartialEq, Eq)]` on a type holding an `Id<T>`
/// compiles without restating them.
pub trait IdTag: Copy + Ord + Hash + fmt::Debug {
    /// Prefix printed before the raw value by [`Display`](fmt::Display) and
    /// [`Debug`](fmt::Debug).
    const PREFIX: &'static str;
}

/// A dense zero-based identity of one kind of entity.
///
/// The identity is the entity's slot index. Slots are never reused while the
/// store holds them, so an identity stays valid — and keeps addressing the
/// same entity — until a compaction renumbers the store and reports the
/// change as a [`Renumbering`](super::Renumbering).
///
/// `Debug` prints the same text as `Display` (`n7`, `e12`) because the raw
/// slot index is the whole content of an identity and a derived
/// `Id(7, PhantomData)` only obscures it.
///
/// # Examples
///
/// ```
/// use cfglib::graph::store::{Id, NodeTag};
///
/// let node = Id::<NodeTag>::from_index(7);
/// assert_eq!(node.index(), 7);
/// assert_eq!(format!("{node}"), "n7");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Id<T: IdTag>(u32, PhantomData<fn() -> T>);

impl<T: IdTag> Id<T> {
    /// Create an identity from its dense raw index.
    #[inline]
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw, PhantomData)
    }

    /// Return the compact raw identity.
    #[inline]
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Return the dense zero-based index.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Create an identity from a dense zero-based index.
    ///
    /// # Panics
    ///
    /// Panics when `index` exceeds `u32::MAX`.
    #[inline]
    #[must_use]
    pub fn from_index(index: usize) -> Self {
        Self::from_raw(u32::try_from(index).expect("dense identity index exceeds u32::MAX"))
    }
}

impl<T: IdTag> fmt::Display for Id<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}{}", T::PREFIX, self.0)
    }
}

impl<T: IdTag> fmt::Debug for Id<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}{}", T::PREFIX, self.0)
    }
}

impl<T: IdTag> DenseNodeId for Id<T> {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }

    fn index(self) -> usize {
        self.index()
    }
}

impl<T: IdTag> DenseEdgeId for Id<T> {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }

    fn index(self) -> usize {
        self.index()
    }
}

#[cfg(feature = "serde")]
impl<T: IdTag> serde::Serialize for Id<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(self.0)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: IdTag> serde::Deserialize<'de> for Id<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        <u32 as serde::Deserialize<'de>>::deserialize(deserializer).map(Self::from_raw)
    }
}

/// Tag marking an identity that addresses a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeTag;

impl IdTag for NodeTag {
    const PREFIX: &'static str = "n";
}

/// Tag marking an identity that addresses an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeTag;

impl IdTag for EdgeTag {
    const PREFIX: &'static str = "e";
}

/// A node identity in the default-tagged [`Graph`](super::Graph).
///
/// This alias stays inside `graph::store` rather than joining the crate
/// facade: the facade's `NodeId` is still
/// [`graph::directed::NodeId`](crate::graph::directed::NodeId) until the two
/// stores are unified.
pub type NodeId = Id<NodeTag>;

/// An edge identity in the default-tagged [`Graph`](super::Graph).
///
/// This alias stays inside `graph::store`; the facade's `EdgeId` is still
/// [`edge::EdgeId`](crate::edge::EdgeId) until the two stores are unified.
pub type EdgeId = Id<EdgeTag>;
