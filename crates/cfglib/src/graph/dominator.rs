//! Dominator trees by the simple Lengauer-Tarjan algorithm.
//!
//! [`DominatorTree::compute`] finds immediate dominators in O(m log n) time
//! for a graph of n nodes and m edges, and the bound holds on irreducible
//! graphs and on adversarial inputs. Every traversal runs on an explicit
//! stack, so a chain of 10⁵ blocks needs no more call stack than a diamond.
//!
//! The finished tree is numbered once, in preorder, with children and roots in
//! ascending dense index. That numbering makes [`dominates`](DominatorTree::dominates)
//! an O(1) interval test, [`subtree`](DominatorTree::subtree) a slice, and
//! [`children`](DominatorTree::children) a walk that skips whole subtrees. It
//! depends only on the set of dominator pairs, never on the order of successor
//! lists or on how many parallel edges join two nodes.
//!
//! The work is split by responsibility. `construction` runs the algorithm over
//! any [`GraphView`], `numbering` numbers a forest from its parent table,
//! `frontier` answers dominance-frontier queries over the numbered tree, and
//! `post` builds the post-dominator tree on the first two and answers its
//! frontier queries with the third. This module holds the tree and its queries.

extern crate alloc;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::ops::Range;

use crate::block::BlockId;
use crate::graph::view::{DenseId, GraphView, RootedView};
use construction::ConstructionBuffers;
pub use frontier::FrontierScratch;
use numbering::{Numbered, NumberingBuffers};
pub use post::{PostDominatorScratch, PostDominatorTree, PostNode};

mod construction;
mod frontier;
#[cfg(test)]
mod iterative;
mod numbering;
mod post;
#[cfg(test)]
mod tests;

/// Marks a root or an unreachable node in a parent table, and an unreachable
/// node in every other table.
const ABSENT: u32 = u32::MAX;

/// The dense index a stored word names.
const fn slot(stored: u32) -> usize {
    stored as usize
}

/// The word that stores a dense index.
///
/// Every table is sized by a node bound that [`check_bound`] has already
/// shown to fit, so a failure here is a view that yields an index past its own
/// bound.
fn word(index: usize) -> u32 {
    u32::try_from(index).expect("a dense index below the checked node bound fits in u32")
}

/// Reject a node bound that dense `u32` tables cannot index.
fn check_bound(bound: usize) {
    assert!(
        u32::try_from(bound).is_ok(),
        "node bound exceeds the dense u32 range of a dominator tree"
    );
}

/// A dominator tree computed from a rooted directed graph.
///
/// A node `a` dominates `b` when every path from the root to `b` passes
/// through `a`. The immediate dominators form a tree, and `a` dominates `b`
/// exactly when `a` is an ancestor of `b`, counting `b` itself.
///
/// The tree is a forest in general: a tree computed from one root has one
/// tree, and the forests that SSA construction and post-dominance build have
/// several. Nodes the root does not reach belong to no tree.
///
/// Every table is a dense `u32` array indexed by [`DenseId::index`] and sized
/// by the graph's node bound when the tree was computed. Querying a node
/// beyond that bound panics.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, DominatorTree};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// let b2 = cfg.new_block();
/// cfg.add_edge(b0, b1, EdgeKind::ConditionalTrue);
/// cfg.add_edge(b0, b2, EdgeKind::ConditionalFalse);
///
/// let dom = DominatorTree::compute(&cfg);
/// assert_eq!(dom.idom(b1), Some(b0));
/// assert_eq!(dom.idom(b2), Some(b0));
/// assert!(dom.dominates(b0, b1));
/// assert!(dom.strictly_dominates(b0, b2));
/// assert!(!dom.dominates(b1, b2));
/// assert_eq!(dom.level(b2), Some(1));
/// assert_eq!(dom.nearest_common_dominator(b1, b2), Some(b0));
/// assert_eq!(dom.preorder(), [b0, b1, b2]);
/// assert_eq!(dom.children(b0).collect::<Vec<_>>(), [b1, b2]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DominatorTree<N = BlockId> {
    /// Parent of each node, [`ABSENT`] for a root and for an unreachable node.
    idom: Vec<u32>,
    /// Preorder index of each node, [`ABSENT`] when it is unreachable.
    pre: Vec<u32>,
    /// Preorder index of each node's last descendant.
    last: Vec<u32>,
    /// Distance from each node to the root of its tree, which has level 0.
    level: Vec<u32>,
    /// The reachable nodes in preorder. The subtree of `b` is
    /// `order[pre[b] ..= last[b]]`.
    order: Vec<N>,
}

/// Every buffer [`DominatorTree::compute_in`] fills, owned by the caller.
///
/// A whole-codebase pass computes a dominator tree once per procedure, and
/// most procedures are a handful of blocks. At that size the tree is not the
/// cost, the *call* is: the depth-first numbering, the semidominator and
/// link-eval tables, the buckets, the two search stacks, and the sibling
/// chains of the numbering walk, each a malloc and a free for a four-block
/// answer. This moves all of them out of the call, leaving only the five
/// arrays the tree itself keeps.
///
/// Hand one scratch to every [`compute_in`](DominatorTree::compute_in) of the
/// pass. The buffers grow to the largest graph the pass meets and are then
/// reused by every graph after it, so the whole pass allocates a bounded
/// number of times instead of a few times per procedure.
///
/// # One scratch, any graph
///
/// Nodes are held as dense indices rather than as a node-id type, so the
/// scratch carries no type parameter: one instance serves a [`Cfg`](crate::Cfg), a
/// [`Graph`](crate::Graph), and a consumer-defined [`GraphView`], in any
/// order, and it is [`Send`], so a worker thread can own one for the whole
/// corpus it is handed.
///
/// # Sizing
///
/// Nothing is fixed at construction and nothing is released by a call. Every
/// buffer is cleared and resized on entry, so a scratch used on a large graph
/// and then on a small one is correct and allocation-free, and the space it
/// holds is the high-water mark of the sequence.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, DominatorScratch, DominatorTree, EdgeKind};
///
/// let mut cfg = Cfg::<u32>::new();
/// let entry = cfg.entry();
/// let then_block = cfg.new_block();
/// cfg.add_edge(entry, then_block, EdgeKind::ConditionalTrue);
///
/// let mut scratch = DominatorScratch::new();
/// let tree = DominatorTree::compute_in(&mut scratch, &cfg);
/// assert_eq!(tree, DominatorTree::compute(&cfg));
/// ```
#[derive(Debug, Clone, Default)]
pub struct DominatorScratch {
    /// The depth-first numbering and the Lengauer-Tarjan tables.
    construction: ConstructionBuffers,
    /// The sibling chains the preorder numbering walks.
    numbering: NumberingBuffers,
}

impl DominatorScratch {
    /// Scratch holding nothing yet.
    ///
    /// Every buffer is sized by the first graph it sees, so there is no node
    /// count to state here.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<N: DenseId> DominatorTree<N> {
    /// Constructs an internal dominator forest from already validated dense
    /// parent and reachability tables.
    ///
    /// A reachable node without a parent is a root. An unreachable node has
    /// no parent.
    pub(crate) fn from_forest_parts(idom: &[Option<N>], reachable: &[bool]) -> Self {
        debug_assert_eq!(idom.len(), reachable.len());
        check_bound(idom.len());
        let parents: Vec<u32> = idom
            .iter()
            .map(|parent| parent.map_or(ABSENT, |parent| word(parent.index())))
            .collect();
        debug_assert!(
            parents
                .iter()
                .zip(reachable)
                .all(|(&parent, &reachable)| reachable || parent == ABSENT),
            "an unreachable node has no immediate dominator"
        );
        let roots: Vec<u32> = (0..parents.len())
            .filter(|&node| reachable[node] && parents[node] == ABSENT)
            .map(word)
            .collect();
        let reached = reachable.iter().filter(|&&reachable| reachable).count();
        let numbered = NumberingBuffers::default().number(&parents, &roots, reached);
        Self::from_numbered(parents, numbered)
    }

    /// Joins a parent table with the numbering of its forest.
    fn from_numbered(idom: Vec<u32>, numbered: Numbered<N>) -> Self {
        let Numbered {
            pre,
            last,
            level,
            order,
        } = numbered;
        Self {
            idom,
            pre,
            last,
            level,
            order,
        }
    }

    /// Compute the dominator tree of a rooted graph view with the simple
    /// Lengauer-Tarjan algorithm.
    ///
    /// The running time is O(m log n) for a graph of n nodes and m edges.
    ///
    /// # Panics
    ///
    /// Panics when the root is not a node of `graph`, or when
    /// [`node_bound`](GraphView::node_bound) exceeds the range of a dense
    /// `u32` table.
    #[must_use]
    pub fn compute<G>(graph: &G) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        Self::compute_in(&mut DominatorScratch::new(), graph)
    }

    /// Compute the dominator tree of a rooted graph view over caller-owned
    /// working storage.
    ///
    /// This is [`compute`](Self::compute) with the call's buffers taken out of
    /// it. The tree it returns is the same one, built by the same code: the
    /// allocating entry point is this function over a fresh
    /// [`DominatorScratch`].
    ///
    /// # Panics
    ///
    /// Panics under the conditions of [`compute`](Self::compute).
    #[must_use]
    pub fn compute_in<G>(scratch: &mut DominatorScratch, graph: &G) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        Self::compute_from_in(scratch, graph, graph.root())
    }

    /// Recompute the dominator tree and report which nodes' immediate
    /// dominators changed relative to `previous`.
    ///
    /// A graph edit (edge insertion or removal) invalidates the tree; this
    /// recomputes it and diffs against `previous`, so downstream analyses can
    /// be updated selectively instead of from scratch.
    ///
    /// # Panics
    ///
    /// Panics when `previous` was computed over a smaller node space than
    /// `graph` currently has, and under the conditions of
    /// [`compute`](Self::compute).
    #[must_use]
    pub fn compute_with_diff<G>(graph: &G, previous: &Self) -> (Self, Vec<N>)
    where
        G: RootedView<NodeId = N>,
    {
        let next = Self::compute(graph);
        let changed = (0..graph.node_bound())
            .map(N::from_index)
            .filter(|&node| previous.idom(node) != next.idom(node))
            .collect();
        (next, changed)
    }

    /// Compute dominators for any directed graph view from an explicit root.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of [`compute`](Self::compute).
    #[must_use]
    pub fn compute_from<G>(graph: &G, root: N) -> Self
    where
        G: GraphView<NodeId = N>,
    {
        Self::compute_from_in(&mut DominatorScratch::new(), graph, root)
    }

    /// Compute dominators from an explicit root over caller-owned working
    /// storage.
    ///
    /// This is [`compute_from`](Self::compute_from) with the call's buffers
    /// taken out of it; see [`DominatorScratch`].
    ///
    /// # Panics
    ///
    /// Panics under the conditions of [`compute`](Self::compute).
    #[must_use]
    pub fn compute_from_in<G>(scratch: &mut DominatorScratch, graph: &G, root: N) -> Self
    where
        G: GraphView<NodeId = N>,
    {
        check_bound(graph.node_bound());
        assert!(
            root.index() < graph.node_bound(),
            "dominator root is not a node of the graph"
        );
        let idom = scratch
            .construction
            .immediate_dominators(graph, root.index());
        let numbered =
            scratch
                .numbering
                .number(&idom, &[word(root.index())], scratch.construction.reached());
        let tree = Self::from_numbered(idom, numbered);
        debug_assert!(
            tree.satisfies_parent_property(graph),
            "an edge ends at a node whose immediate dominator is not an ancestor of its tail"
        );
        tree
    }

    /// Return the immediate dominator of `node`, or `None` for a root and for
    /// an unreachable node.
    #[must_use]
    pub fn idom(&self, node: N) -> Option<N> {
        let parent = self.idom[node.index()];
        (parent != ABSENT).then(|| N::from_index(slot(parent)))
    }

    /// Whether `node` is in a tree of this forest, which for a tree computed
    /// from one root means the root reaches it. `idom` alone cannot
    /// distinguish "is the root" from "was never reached" (both `None`) —
    /// consumers reasoning about dominance must check this before trusting a
    /// `None`.
    #[must_use]
    pub fn is_reachable(&self, node: N) -> bool {
        self.pre[node.index()] != ABSENT
    }

    /// Return whether `dominator` dominates `node`, in constant time.
    ///
    /// A node dominates itself, reachable or not. Any other pair with an
    /// unreachable member, and any two nodes in different trees of a forest,
    /// answer `false`.
    #[must_use]
    pub fn dominates(&self, dominator: N, node: N) -> bool {
        if dominator == node {
            return true;
        }
        let (dominator, node) = (dominator.index(), node.index());
        let (start, position) = (self.pre[dominator], self.pre[node]);
        start != ABSENT
            && position != ABSENT
            && start <= position
            && position <= self.last[dominator]
    }

    /// Return whether `dominator` dominates `node` and is a different node.
    #[must_use]
    pub fn strictly_dominates(&self, dominator: N, node: N) -> bool {
        dominator != node && self.dominates(dominator, node)
    }

    /// Return a node's level: the number of strict dominators it has. A root
    /// has level 0, and an unreachable node has none.
    #[must_use]
    pub fn level(&self, node: N) -> Option<u32> {
        let index = node.index();
        (self.pre[index] != ABSENT).then(|| self.level[index])
    }

    /// Return the deepest node that dominates both `first` and `second`.
    ///
    /// The walk climbs from the deeper node until the two meet, so it costs
    /// O(depth). The answer is `None` when either node is unreachable and when
    /// the two sit in different trees of a forest.
    #[must_use]
    pub fn nearest_common_dominator(&self, first: N, second: N) -> Option<N> {
        let (mut first, mut second) = (first.index(), second.index());
        if self.pre[first] == ABSENT || self.pre[second] == ABSENT {
            return None;
        }
        while first != second {
            match self.level[first].cmp(&self.level[second]) {
                Ordering::Greater => first = slot(self.idom[first]),
                Ordering::Less => second = slot(self.idom[second]),
                Ordering::Equal => {
                    // Distinct nodes at one level climb together. Two roots
                    // have no parent to climb to and share no tree.
                    if self.idom[first] == ABSENT {
                        return None;
                    }
                    first = slot(self.idom[first]);
                    second = slot(self.idom[second]);
                }
            }
        }
        Some(N::from_index(first))
    }

    /// Return the deepest node that dominates every reachable node of `nodes`.
    ///
    /// Unreachable nodes are skipped. The answer is `None` when no node is
    /// reachable and when the reachable ones share no tree.
    #[must_use]
    pub fn nearest_common_dominator_of(&self, nodes: impl IntoIterator<Item = N>) -> Option<N> {
        let mut common = None;
        for node in nodes.into_iter().filter(|&node| self.is_reachable(node)) {
            common = Some(match common {
                None => node,
                Some(common) => self.nearest_common_dominator(common, node)?,
            });
        }
        common
    }

    /// Return every reachable node in preorder.
    ///
    /// Roots come in ascending dense index, and the children of a node come in
    /// ascending dense index after it, so a node always precedes the nodes it
    /// dominates.
    #[must_use]
    pub fn preorder(&self) -> &[N] {
        &self.order
    }

    /// Return the nodes `node` dominates, itself first, in preorder. An
    /// unreachable node has none.
    #[must_use]
    pub fn subtree(&self, node: N) -> &[N] {
        &self.order[self.span(node)]
    }

    /// Iterate over the nodes whose immediate dominator is `node`, in ascending
    /// dense index.
    ///
    /// The walk skips each child's subtree, so it costs one step per child.
    pub fn children(&self, node: N) -> impl Iterator<Item = N> + '_ {
        let span = self.span(node);
        let first = span.start + 1;
        core::iter::successors((first < span.end).then_some(first), move |&position| {
            let next = slot(self.last[self.order[position].index()]) + 1;
            (next < span.end).then_some(next)
        })
        .map(|position| self.order[position])
    }

    /// The range of preorder positions `node` dominates, empty when it is
    /// unreachable.
    fn span(&self, node: N) -> Range<usize> {
        let index = node.index();
        if self.pre[index] == ABSENT {
            return 0..0;
        }
        slot(self.pre[index])..slot(self.last[index]) + 1
    }

    /// Walk the trees below `roots` in preorder, giving each node a scope:
    /// `enter` runs when the walk reaches a node, and `exit` runs once every
    /// node it dominates has been entered and exited.
    ///
    /// This is the explicit-stack walk that a pass over the dominator tree
    /// with scoped state (a renaming stack, a table of available values) is
    /// built on, stated once. `enter` returns what `exit` needs to undo it,
    /// so a scope costs the walk one stack entry and no allocation of its
    /// own. Roots are walked in the order given and the children of a node in
    /// ascending dense index, which is the order of
    /// [`preorder`](Self::preorder); a root that is in no tree has nothing
    /// to walk and is skipped. A walk does not recurse, so a chain of 10⁵
    /// nodes costs no more host stack than a diamond.
    ///
    /// `state` is handed to both callbacks, so they can share one mutable
    /// value without two closures borrowing it. `open` holds the scopes
    /// entered and not yet exited, one per level of the tree at most. It is
    /// cleared on entry and empty on return, and the caller owns it so a pass
    /// over many procedures allocates it once.
    pub(crate) fn scoped_walk<S, T>(
        &self,
        roots: impl IntoIterator<Item = N>,
        state: &mut S,
        open: &mut Vec<(N, T)>,
        mut enter: impl FnMut(&mut S, N) -> T,
        mut exit: impl FnMut(&mut S, N, T),
    ) {
        open.clear();
        for root in roots {
            for position in self.span(root) {
                // A scope ends where the preorder leaves its subtree.
                while let Some(&(top, _)) = open.last()
                    && slot(self.last[top.index()]) < position
                {
                    let (node, scope) = open.pop().expect("a scope was just seen on top");
                    exit(state, node, scope);
                }
                let node = self.order[position];
                let scope = enter(state, node);
                open.push((node, scope));
            }
            while let Some((node, scope)) = open.pop() {
                exit(state, node, scope);
            }
        }
    }
}
