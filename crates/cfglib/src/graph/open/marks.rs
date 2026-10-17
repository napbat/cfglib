//! The visited marks every open walk keeps.

extern crate alloc;

use alloc::collections::BTreeSet;
use smallvec::SmallVec;

/// Nodes a walk keeps inline before it allocates: its stack or frontier, and
/// the marks it has set.
pub(super) const INLINE_NODES: usize = 8;

/// The nodes a walk has marked.
///
/// The open walks consumers run mostly touch a handful of nodes, where one
/// linear scan of an inline array costs less than a tree's allocation and
/// pointer chasing. A walk that outgrows the array moves its marks into a
/// [`BTreeSet`] once and stays there, so a large walk keeps the logarithmic
/// lookups the walks document. Both forms answer by the node's `Ord`.
pub(super) struct Marks<N>(Repr<N>);

enum Repr<N> {
    Few(SmallVec<[N; INLINE_NODES]>),
    Many(BTreeSet<N>),
}

/// Marks `node` in the inline list, reporting whether it was unmarked, or hands
/// `node` back when the list is full.
fn insert_inline<N: Ord>(nodes: &mut SmallVec<[N; INLINE_NODES]>, node: N) -> Result<bool, N> {
    if nodes.contains(&node) {
        Ok(false)
    } else if nodes.len() < INLINE_NODES {
        nodes.push(node);
        Ok(true)
    } else {
        Err(node)
    }
}

/// The set a full inline list becomes when `node` joins it. Out of line: it
/// runs once per walk that outgrows the list, and inlining it would bloat
/// every mark inserted on the hot path.
#[inline(never)]
fn spill<N: Ord>(nodes: &mut SmallVec<[N; INLINE_NODES]>, node: N) -> BTreeSet<N> {
    let mut set: BTreeSet<N> = nodes.drain(..).collect();
    set.insert(node);
    set
}

impl<N: Ord> Marks<N> {
    pub(super) fn new() -> Self {
        Self(Repr::Few(SmallVec::new()))
    }

    pub(super) fn contains(&self, node: &N) -> bool {
        match &self.0 {
            Repr::Few(nodes) => nodes.contains(node),
            Repr::Many(nodes) => nodes.contains(node),
        }
    }

    /// Marks `node`, reporting whether it was unmarked.
    #[inline]
    pub(super) fn insert(&mut self, node: N) -> bool {
        match &mut self.0 {
            Repr::Few(nodes) => match insert_inline(nodes, node) {
                Ok(inserted) => inserted,
                Err(node) => {
                    self.0 = Repr::Many(spill(nodes, node));
                    true
                }
            },
            Repr::Many(nodes) => nodes.insert(node),
        }
    }

    /// Takes `node`'s mark back; a node that is not marked is left alone.
    pub(super) fn remove(&mut self, node: &N) {
        match &mut self.0 {
            Repr::Few(nodes) => {
                if let Some(at) = nodes.iter().position(|marked| marked == node) {
                    nodes.swap_remove(at);
                }
            }
            Repr::Many(nodes) => {
                nodes.remove(node);
            }
        }
    }
}
