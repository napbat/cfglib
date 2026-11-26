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
//! A tree can come with a certificate. [`DominatorTree::compute_certified_in`]
//! also returns a [`LowHighOrder`], built in linear time from what the
//! construction already computes, and [`DominatorTree::verify_certificate`]
//! checks in O(n + m) that a tree is the dominator tree of a graph, given
//! such an order: a tree that is not is rejected with any certificate. The
//! certified construction costs about twice the plain one, and the check
//! about as much as the plain one. Debug builds check every tree they build
//! this way.
//!
//! A tree can also follow its graph through edge edits.
//! [`DominatorTree::apply_updates_in`] replays a batch of
//! [`DominatorUpdate`]s on the tree it already has, where a full rebuild
//! would pay for the whole graph to learn about one edge.
//!
//! The work is split by responsibility. `construction` runs the algorithm over
//! any [`GraphView`](crate::GraphView) and builds a low-high order from it, `certificate` checks
//! a tree against one, `incremental` replays edits, `numbering` numbers a
//! forest from its parent table, `forest` extends a tree over the code its
//! root misses, `frontier` answers dominance-frontier queries over the
//! numbered tree, `frontier_table` computes the frontier of every node at
//! once, `edge_dominance` answers whether every path to a node takes one
//! edge, and `post` builds the post-dominator tree on the first two and
//! answers its frontier queries with `frontier`. This module holds the tree,
//! its node queries and the one scratch every construction and query shares.

extern crate alloc;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::ops::Range;

use crate::block::BlockId;
use crate::graph::view::{DenseId, RootedView};
use crate::identity::{NO_INDEX, slot, word};
pub use certificate::{CertificateError, LowHighOrder};
use construction::{ConstructionBuffers, LowHighBuffers};
use frontier::FrontierBuffers;
pub use frontier_table::{DominanceFrontiers, FrontierTree};
pub use incremental::DominatorUpdate;
use incremental::UpdateBuffers;
use numbering::{Numbered, NumberingBuffers};
pub(crate) use post::CutBuffers;
pub use post::{ExitPolicy, PostDominatorTree, PostNode, SinkId};

mod certificate;
mod construction;
mod edge_dominance;
mod forest;
mod frontier;
mod frontier_table;
mod incremental;
#[cfg(test)]
mod iterative;
mod numbering;
mod post;
#[cfg(test)]
mod tests;

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
    /// Parent of each node, [`NO_INDEX`] for a root and for an unreachable node.
    idom: Vec<u32>,
    /// Preorder index of each node, [`NO_INDEX`] when it is unreachable.
    pre: Vec<u32>,
    /// Preorder index of each node's last descendant.
    last: Vec<u32>,
    /// Distance from each node to the root of its tree, which has level 0.
    level: Vec<u32>,
    /// The reachable nodes in preorder. The subtree of `b` is
    /// `order[pre[b] ..= last[b]]`.
    order: Vec<N>,
}

/// Every buffer a dominator computation or query fills, owned by the caller.
///
/// A whole-codebase pass computes a dominator tree once per procedure, and
/// most procedures are a handful of blocks. At that size the tree is not the
/// cost, the *call* is: the depth-first numbering, the semidominator and
/// link-eval tables, the buckets, the search stack, and the sibling chains of
/// the numbering walk, each a malloc and a free for a four-block answer. This
/// moves all of them out of the call, leaving only the five arrays the tree
/// itself keeps.
///
/// One scratch serves every `_in` function of the dominator family:
/// [`DominatorTree::compute_in`] and its certified, incremental and forest
/// forms, [`PostDominatorTree::compute_in`] (which also keeps the buffers that
/// find the exits and the cut here), and the iterated-frontier queries
/// [`DominatorTree::iterated_frontier_in`],
/// [`PostDominatorTree::iterated_frontier_in`] and
/// [`DominanceFrontiers::iterated_frontier_in`]. Hand one scratch to every
/// call of the pass. The buffers grow to the largest graph the pass meets and
/// are then reused by every graph after it, so the whole pass allocates a
/// bounded number of times instead of a few times per procedure.
///
/// # One scratch, any graph
///
/// Nodes are held as dense indices rather than as a node-id type, so the
/// scratch carries no type parameter: one instance serves a [`Cfg`](crate::Cfg), a
/// [`Graph`](crate::Graph), and a consumer-defined [`GraphView`](crate::GraphView), in any
/// order, and it is [`Send`], so a worker thread can own one for the whole
/// corpus it is handed.
///
/// # Sizing
///
/// It starts empty ([`Default`]), nothing is fixed at construction and nothing
/// is released by a call. Every buffer is cleared and grown on entry, so a
/// scratch used on a large graph and then on a small one is correct and
/// allocation-free, and the space it holds is the high-water mark of the
/// sequence.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, DominatorScratch, DominatorTree, EdgeKind, ExitPolicy, PostDominatorTree};
///
/// let mut cfg = Cfg::<u32>::new();
/// let entry = cfg.entry();
/// let then_block = cfg.new_block();
/// cfg.add_edge(entry, then_block, EdgeKind::ConditionalTrue);
///
/// let mut scratch = DominatorScratch::default();
/// let tree = DominatorTree::compute_in(&mut scratch, &cfg);
/// assert_eq!(tree, DominatorTree::compute(&cfg));
/// let post = PostDominatorTree::compute_in(&mut scratch, &cfg, ExitPolicy::CutLoops);
/// assert_eq!(post, PostDominatorTree::compute(&cfg));
/// ```
#[derive(Debug, Clone, Default)]
pub struct DominatorScratch {
    /// The depth-first numbering and the Lengauer-Tarjan tables.
    construction: ConstructionBuffers,
    /// The sibling chains the preorder numbering walks.
    numbering: NumberingBuffers,
    /// The tables a low-high order is built in, when one is asked for.
    low_high: LowHighBuffers,
    /// The marks and worklists of an incremental update.
    updates: UpdateBuffers,
    /// The exits and the cut of a post-dominator tree.
    cut: CutBuffers,
    /// The marks and buckets of an iterated-frontier query.
    frontier: FrontierBuffers,
}

impl<N: DenseId> DominatorTree<N> {
    /// Constructs a dominator forest from already validated dense parent and
    /// reachability tables.
    ///
    /// A reachable node without a parent is a root. An unreachable node has
    /// no parent.
    #[cfg(test)]
    pub(crate) fn from_forest_parts(idom: &[Option<N>], reachable: &[bool]) -> Self {
        debug_assert_eq!(idom.len(), reachable.len());
        crate::identity::check_bound(idom.len(), "node bound of a dominator tree");
        let parents: Vec<u32> = idom
            .iter()
            .map(|parent| parent.map_or(NO_INDEX, |parent| word(parent.index())))
            .collect();
        debug_assert!(
            parents
                .iter()
                .zip(reachable)
                .all(|(&parent, &reachable)| reachable || parent == NO_INDEX),
            "an unreachable node has no immediate dominator"
        );
        let roots: Vec<u32> = (0..parents.len())
            .filter(|&node| reachable[node] && parents[node] == NO_INDEX)
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

    /// The forest a construction from a virtual root leaves once the virtual
    /// root is dropped.
    ///
    /// `parent` is the parent table of a construction over a view with one
    /// node more than the graph, the virtual root, at its last index, and
    /// `reached` is how many nodes of the graph that construction reached.
    /// The virtual root has no parent, so dropping it leaves a table over the
    /// graph's nodes, and its children become the roots of the forest. This
    /// is how a post-dominator tree leaves its virtual exit and how a forest
    /// over unreachable code leaves its virtual entry.
    fn from_virtual_root(
        numbering: &mut NumberingBuffers,
        mut parent: Vec<u32>,
        reached: usize,
    ) -> Self {
        let bound = parent.len() - 1;
        parent.truncate(bound);
        let virtual_root = word(bound);
        let mut roots = Vec::new();
        for (node, up) in parent.iter_mut().enumerate() {
            if *up == virtual_root {
                *up = NO_INDEX;
                roots.push(word(node));
            }
        }
        let numbered = numbering.number(&parent, &roots, reached);
        Self::from_numbered(parent, numbered)
    }

    /// Compute the dominator tree of a rooted graph view with the simple
    /// Lengauer-Tarjan algorithm.
    ///
    /// The running time is O(m log n) for a graph of n nodes and m edges. To
    /// start from another root, pass [`Rooted::new(&graph, root)`](crate::Rooted);
    /// for dominators of the reverse graph, pass a
    /// [`Reversed`](crate::Reversed) view under it.
    ///
    /// # Panics
    ///
    /// Panics when the root is not a node of `graph`, or when
    /// [`node_bound`](crate::GraphView::node_bound) exceeds the range of a dense
    /// `u32` table.
    #[must_use]
    pub fn compute<G>(graph: &G) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        Self::compute_in(&mut DominatorScratch::default(), graph)
    }

    /// Compute the dominator tree of a rooted graph view over caller-owned
    /// working storage.
    ///
    /// This is [`compute`](Self::compute) with the call's buffers taken out of
    /// it. The tree it returns is the same one, built by the same code: the
    /// allocating entry point is this function over a fresh
    /// [`DominatorScratch`].
    ///
    /// Debug builds also build a low-high order and check the tree against
    /// it with [`verify_certificate`](Self::verify_certificate).
    ///
    /// # Panics
    ///
    /// Panics under the conditions of [`compute`](Self::compute).
    #[must_use]
    pub fn compute_in<G>(scratch: &mut DominatorScratch, graph: &G) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        if cfg!(debug_assertions) {
            let (tree, certificate) = Self::compute_certified_in(scratch, graph);
            debug_assert_eq!(
                tree.verify_certificate(graph, &certificate),
                Ok(()),
                "the construction built a tree its own certificate rejects"
            );
            return tree;
        }
        Self::construct(scratch, graph, graph.root(), None)
    }

    /// Return the immediate dominator of `node`, or `None` for a root and for
    /// an unreachable node.
    #[must_use]
    pub fn idom(&self, node: N) -> Option<N> {
        let parent = self.idom[node.index()];
        (parent != NO_INDEX).then(|| N::from_index(slot(parent)))
    }

    /// Whether `node` is in a tree of this forest, which for a tree computed
    /// from one root means the root reaches it. `idom` alone cannot
    /// distinguish "is the root" from "was never reached" (both `None`) —
    /// consumers reasoning about dominance must check this before trusting a
    /// `None`.
    #[must_use]
    pub fn is_reachable(&self, node: N) -> bool {
        self.pre[node.index()] != NO_INDEX
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
        start != NO_INDEX
            && position != NO_INDEX
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
        (self.pre[index] != NO_INDEX).then(|| self.level[index])
    }

    /// Return the deepest node that dominates both `first` and `second`.
    ///
    /// The walk climbs from the deeper node until the two meet, so it costs
    /// O(depth). The answer is `None` when either node is unreachable and when
    /// the two sit in different trees of a forest.
    #[must_use]
    pub fn nearest_common_dominator(&self, first: N, second: N) -> Option<N> {
        let (mut first, mut second) = (first.index(), second.index());
        if self.pre[first] == NO_INDEX || self.pre[second] == NO_INDEX {
            return None;
        }
        while first != second {
            match self.level[first].cmp(&self.level[second]) {
                Ordering::Greater => first = slot(self.idom[first]),
                Ordering::Less => second = slot(self.idom[second]),
                Ordering::Equal => {
                    // Distinct nodes at one level climb together. Two roots
                    // have no parent to climb to and share no tree.
                    if self.idom[first] == NO_INDEX {
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
    pub fn nearest_common_dominator_among(&self, nodes: impl IntoIterator<Item = N>) -> Option<N> {
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
        self.below(Some(node))
    }

    /// Iterate over the roots of the forest, in ascending dense index: the
    /// root of a tree computed from one root, every root of a forest, and
    /// nothing for a tree of no node.
    ///
    /// The walk skips each root's tree, so it costs one step per root.
    pub fn roots(&self) -> impl Iterator<Item = N> + '_ {
        self.below(None)
    }

    /// The children of `parent`, or the roots of the forest for `None`, in
    /// ascending dense index: the nodes atop the whole subtrees that tile the
    /// preorder positions below `parent`, found by jumping from each one past
    /// its subtree.
    fn below(&self, parent: Option<N>) -> impl Iterator<Item = N> + '_ {
        let Range { start, end } = match parent {
            Some(node) => {
                let span = self.span(node);
                span.start + 1..span.end
            }
            None => 0..self.order.len(),
        };
        core::iter::successors((start < end).then_some(start), move |&position| {
            let next = slot(self.last[self.order[position].index()]) + 1;
            (next < end).then_some(next)
        })
        .map(|position| self.order[position])
    }

    /// The range of preorder positions `node` dominates, empty when it is
    /// unreachable.
    fn span(&self, node: N) -> Range<usize> {
        let index = node.index();
        if self.pre[index] == NO_INDEX {
            return 0..0;
        }
        slot(self.pre[index])..slot(self.last[index]) + 1
    }
}
