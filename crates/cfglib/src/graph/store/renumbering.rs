//! Old-to-new identities produced by one compaction.

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use super::id::{EdgeTag, Id, IdTag, NodeTag};
use crate::identity::NO_INDEX;

/// Every old identity's replacement after one compaction.
///
/// Named for what it is: compaction assigns new dense numbers, and this is
/// the numbering. It is total over the old slot space rather than sparse like
/// [`Rewrite`](crate::Rewrite) — a compaction touches every identity,
/// so a lookup is an array read, not a tree descent, and "absent" can mean
/// only one thing: the entity was removed.
///
/// Renumberings [`compose`](Self::compose), so a consumer holding identities
/// from before several compactions folds them into one lookup instead of
/// replaying each step.
///
/// A [`compact`](super::Graph::compact) renumbering keeps relative order for
/// edges: a lower old identity gets a lower new one. It keeps it for nodes
/// too under [`NodeOrder::Insertion`](super::NodeOrder::Insertion), and may
/// map them to any order under every other [`NodeOrder`](super::NodeOrder);
/// every lookup, and [`compose`](Self::compose), works the same for all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Renumbering<NT: IdTag = NodeTag, ET: IdTag = EdgeTag> {
    nodes: Vec<u32>,
    edges: Vec<u32>,
    tags: PhantomData<fn() -> (NT, ET)>,
}

impl<NT: IdTag, ET: IdTag> Renumbering<NT, ET> {
    pub(super) const fn new(nodes: Vec<u32>, edges: Vec<u32>) -> Self {
        Self {
            nodes,
            edges,
            tags: PhantomData,
        }
    }

    /// The new identity of an old node, or `None` when it did not survive.
    ///
    /// An identity minted after the compaction this describes is also `None`:
    /// it has no old numbering to translate.
    #[must_use]
    pub fn node(&self, old: Id<NT>) -> Option<Id<NT>> {
        translate(&self.nodes, old.index())
    }

    /// The new identity of an old edge, or `None` when it did not survive.
    #[must_use]
    pub fn edge(&self, old: Id<ET>) -> Option<Id<ET>> {
        translate(&self.edges, old.index())
    }

    /// Fold `later`, which renumbered the result of this one, into a single
    /// mapping from this renumbering's old identities to `later`'s new ones.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Graph, NodeOrder};
    ///
    /// let mut graph = Graph::<&'static str, ()>::new();
    /// let first = graph.add_node("first");
    /// let second = graph.add_node("second");
    /// graph.remove_node(first);
    ///
    /// let once = graph.compact(NodeOrder::Insertion);
    /// let twice = once.compose(&graph.compact(NodeOrder::Insertion));
    /// assert_eq!(twice.node(first), None);
    /// assert_eq!(graph.node(twice.node(second).unwrap()), &"second");
    /// ```
    #[must_use]
    pub fn compose(&self, later: &Self) -> Self {
        let nodes = compose_axis(&self.nodes, &later.nodes);
        let edges = compose_axis(&self.edges, &later.edges);
        Self::new(nodes, edges)
    }

    /// Whether every covered identity kept its number.
    ///
    /// For a [`compact`](super::Graph::compact) renumbering in
    /// [`NodeOrder::Insertion`](super::NodeOrder::Insertion) this is true
    /// exactly when nothing was removed: a removed entity has no new number,
    /// even when it was the last slot and nothing moved down behind it. A
    /// compaction that only folded the delta into the base removes nothing,
    /// and is the identity. Under any other order every surviving node must
    /// also have kept its number.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        is_identity_axis(&self.nodes) && is_identity_axis(&self.edges)
    }
}

fn translate<T: IdTag>(mapping: &[u32], old: usize) -> Option<Id<T>> {
    let new = *mapping.get(old)?;
    if new == NO_INDEX {
        return None;
    }
    Some(Id::from_raw(new))
}

fn compose_axis(first: &[u32], second: &[u32]) -> Vec<u32> {
    first
        .iter()
        .map(|&intermediate| {
            if intermediate == NO_INDEX {
                return NO_INDEX;
            }
            second
                .get(intermediate as usize)
                .copied()
                .unwrap_or(NO_INDEX)
        })
        .collect()
}

fn is_identity_axis(mapping: &[u32]) -> bool {
    mapping
        .iter()
        .enumerate()
        .all(|(old, &new)| u32::try_from(old).is_ok_and(|old| old == new))
}
