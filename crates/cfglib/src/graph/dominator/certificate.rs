//! Certification of a dominator tree by a low-high order.
//!
//! A tree rooted at the root of a graph is its dominator tree exactly when it
//! holds the parent property and has a low-high order (Georgiadis and Tarjan,
//! "Dominator Tree Certification and Divergent Spanning Trees", TALG 12(1),
//! 2016, theorems 2.5 and 2.9):
//!
//! - **Parent property:** every edge out of a node of the tree ends at a node
//!   of the tree whose parent is an ancestor of the edge's tail, or at the
//!   root. This alone makes every parent a dominator.
//! - **Low-high order:** a preorder of the tree in which every node other than
//!   the root has an edge from its parent, or an edge from a node before it
//!   and an edge from a node after it that it does not dominate. This rules
//!   out a tree that is too shallow, one where a node hangs beside a sibling
//!   that in fact dominates it.
//!
//! The order is the certificate. Building it takes the by-products of the
//! construction (see `construction`), but checking it takes one pass over the
//! nodes and their edges with the tree's O(1) ancestor test, so the checker
//! is small enough to trust: [`DominatorTree::verify_certificate`] and
//! [`PostDominatorTree::verify_certificate`](super::PostDominatorTree::verify_certificate)
//! are both this one function, [`check`], over two ways of reading a tree.

extern crate alloc;
use alloc::vec::Vec;
use core::fmt;

use super::{ABSENT, DominatorScratch, DominatorTree, check_bound, slot, word};
use crate::block::BlockId;
use crate::graph::view::{DenseId, GraphView};

/// A low-high order of a dominator tree: the certificate that the tree is the
/// dominator tree of the graph it was computed from.
///
/// It is a preorder of the tree in which every node other than the root has
/// an edge from its parent, or an edge from a node before it and an edge from
/// a node after it that it does not dominate.
/// [`DominatorTree::compute_certified_in`] builds one with the tree, and
/// [`DominatorTree::verify_certificate`] checks a tree against one in linear
/// time.
///
/// The order is generally not the ascending-index preorder the tree itself
/// stores. It follows the depth-first search of the construction, and so the
/// order of successor lists: two views of the same edges can yield two
/// different orders, each a valid certificate of the same tree.
///
/// For a [`PostDominatorTree`](super::PostDominatorTree) the order is over
/// the forest of blocks, with the virtual exit, its root, left out: the
/// virtual exit comes first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LowHighOrder<N = BlockId> {
    /// The nodes of the tree in the order.
    pub(super) order: Vec<N>,
    /// The position of each node in `order`, [`ABSENT`] for a node outside
    /// the tree.
    pub(super) position: Vec<u32>,
}

impl<N: DenseId> LowHighOrder<N> {
    /// An order of no node, to be filled by a construction.
    pub(super) const fn empty() -> Self {
        Self {
            order: Vec::new(),
            position: Vec::new(),
        }
    }

    /// The order of `order` over a node space of `bound` nodes, as a
    /// consumer might hand one back: nothing about it is checked.
    #[cfg(test)]
    pub(super) fn from_order(order: Vec<N>, bound: usize) -> Self {
        let mut position = alloc::vec![ABSENT; bound];
        for (index, node) in order.iter().enumerate() {
            position[node.index()] = word(index);
        }
        Self { order, position }
    }

    /// Return the nodes of the tree in the order.
    #[must_use]
    pub fn order(&self) -> &[N] {
        &self.order
    }

    /// Return the position of `node` in the order, or `None` when the order
    /// does not hold it.
    #[must_use]
    pub fn position(&self, node: N) -> Option<usize> {
        let position = *self.position.get(node.index())?;
        (position != ABSENT).then(|| slot(position))
    }
}

/// Why [`DominatorTree::verify_certificate`] or
/// [`PostDominatorTree::verify_certificate`](super::PostDominatorTree::verify_certificate)
/// rejected a tree.
///
/// The first failure found is reported. A tree that is not the dominator tree
/// is rejected whatever certificate comes with it, and the dominator tree is
/// rejected only with a certificate that is not a low-high order of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CertificateError<N> {
    /// The tree and the certificate were not made for a graph with this
    /// node bound.
    NodeBound {
        /// The graph's node bound.
        graph: usize,
        /// The node bound the tree was computed with.
        tree: usize,
        /// The node bound of the certificate's position table.
        certificate: usize,
    },
    /// The root is not the root of the tree, or the tree has other roots.
    Root(N),
    /// The blocks a post-dominator tree holds, its exits or its cut roots
    /// are not those of the graph, or an exit or a cut root does not hang
    /// from the virtual exit.
    ExitStructure,
    /// The certificate is not a preorder of the tree at this node: the node
    /// is out of place, missing, listed twice, or not in the tree.
    Preorder(N),
    /// The edge `tail -> head` breaks the parent property: `head` is not in
    /// the tree, or its parent is not an ancestor of `tail`. For a
    /// post-dominator tree the edge is read on the reverse graph, so it is
    /// the edge `head -> tail` of the graph.
    ParentProperty {
        /// The tail of the edge, a node of the tree.
        tail: N,
        /// The head of the edge.
        head: N,
    },
    /// This node has no edge from its parent, and no edge from a node before
    /// it that comes with an edge from a node after it that it does not
    /// dominate.
    LowHigh(N),
}

impl<N: fmt::Debug> fmt::Display for CertificateError<N> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeBound {
                graph,
                tree,
                certificate,
            } => write!(
                formatter,
                "the graph has node bound {graph}, the tree {tree} and the certificate {certificate}"
            ),
            Self::Root(root) => write!(formatter, "{root:?} is not the one root of the tree"),
            Self::ExitStructure => formatter
                .write_str("the tree's blocks, exits or cut roots are not those of the graph"),
            Self::Preorder(node) => {
                write!(
                    formatter,
                    "the certificate is not a preorder of the tree at {node:?}"
                )
            }
            Self::ParentProperty { tail, head } => write!(
                formatter,
                "the edge {tail:?} -> {head:?} breaks the parent property"
            ),
            Self::LowHigh(node) => write!(formatter, "{node:?} breaks the low-high property"),
        }
    }
}

impl<N: fmt::Debug> core::error::Error for CertificateError<N> {}

impl<N: DenseId> DominatorTree<N> {
    /// Compute the dominator tree of `graph` from `root` together with a
    /// low-high order that certifies it.
    ///
    /// The tree is the one [`compute_from_in`](Self::compute_from_in) builds.
    /// The construction also keeps, for each node, the predecessor that
    /// attains its semidominator and its relative dominator, and the order is
    /// built from those in linear time (Georgiadis and Tarjan, section 6).
    /// Release builds pay for this only here.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of [`compute`](Self::compute).
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, DominatorScratch, DominatorTree, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let entry = cfg.entry();
    /// let left = cfg.new_block();
    /// let right = cfg.new_block();
    /// let join = cfg.new_block();
    /// cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(entry, right, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(left, join, EdgeKind::Unconditional);
    /// cfg.add_edge(right, join, EdgeKind::Unconditional);
    ///
    /// let mut scratch = DominatorScratch::new();
    /// let (tree, certificate) = DominatorTree::compute_certified_in(&mut scratch, &cfg, entry);
    /// assert_eq!(tree, DominatorTree::compute(&cfg));
    /// assert_eq!(certificate.order()[0], entry);
    /// assert_eq!(tree.verify_certificate(&cfg, entry, &certificate), Ok(()));
    /// ```
    #[must_use]
    pub fn compute_certified_in<G>(
        scratch: &mut DominatorScratch,
        graph: &G,
        root: N,
    ) -> (Self, LowHighOrder<N>)
    where
        G: GraphView<NodeId = N>,
    {
        let mut certificate = LowHighOrder::empty();
        let tree = Self::construct(scratch, graph, root, Some(&mut certificate));
        (tree, certificate)
    }

    /// Build the tree, and fill `certificate` when one is asked for.
    pub(super) fn construct<G>(
        scratch: &mut DominatorScratch,
        graph: &G,
        root: N,
        certificate: Option<&mut LowHighOrder<N>>,
    ) -> Self
    where
        G: GraphView<NodeId = N>,
    {
        check_bound(graph.node_bound());
        assert!(
            root.index() < graph.node_bound(),
            "dominator root is not a node of the graph"
        );
        let idom =
            scratch
                .construction
                .immediate_dominators(graph, root.index(), certificate.is_some());
        let numbered =
            scratch
                .numbering
                .number(&idom, &[word(root.index())], scratch.construction.reached());
        if let Some(certificate) = certificate {
            scratch.construction.low_high_order(
                graph,
                &idom,
                &numbered,
                &mut scratch.low_high,
                certificate,
            );
        }
        Self::from_numbered(idom, numbered)
    }

    /// Check that this is the dominator tree of `graph` from `root`, given a
    /// low-high order of it.
    ///
    /// The check reads every node of the tree and every edge at it once, with
    /// the tree's O(1) ancestor test, so it costs O(n + m) and allocates
    /// nothing:
    ///
    /// - `root` is the one root of the tree;
    /// - `certificate` is a preorder of the tree;
    /// - the parent property: every edge out of a node of the tree ends at a
    ///   node of the tree, and at the root or at a node whose parent is an
    ///   ancestor of the edge's tail. So the tree holds every node the root
    ///   reaches;
    /// - the low-high property: every node but the root has an edge from its
    ///   parent, or an edge from a node before it in the order and an edge
    ///   from a node after it that it does not dominate. So the root reaches
    ///   every node of the tree, and no node of the tree dominates a sibling.
    ///
    /// Together these hold exactly when the tree is the dominator tree, so a
    /// wrong tree is rejected with any certificate. The tree's own preorder
    /// numbering is trusted to match its parents, which every constructor
    /// guarantees.
    ///
    /// # Errors
    ///
    /// Returns the first failure found, as a [`CertificateError`].
    pub fn verify_certificate<G>(
        &self,
        graph: &G,
        root: N,
        certificate: &LowHighOrder<N>,
    ) -> Result<(), CertificateError<N>>
    where
        G: GraphView<NodeId = N>,
    {
        let bound = graph.node_bound();
        if self.idom.len() != bound || certificate.position.len() != bound {
            return Err(CertificateError::NodeBound {
                graph: bound,
                tree: self.idom.len(),
                certificate: certificate.position.len(),
            });
        }
        let index = root.index();
        if index >= bound
            || !self.is_reachable(root)
            || self.idom(root).is_some()
            || self.subtree(root).len() != self.order.len()
        {
            return Err(CertificateError::Root(root));
        }
        let candidate = Candidate {
            tree: self,
            root: index,
        };
        check(graph, &candidate, certificate).map_err(|failure| failure.lift(N::from_index))
    }
}

/// A tree as the checker reads it, in dense indices.
pub(super) trait Tree {
    /// The root, which has no parent.
    fn root(&self) -> usize;
    /// How many nodes the tree holds.
    fn size(&self) -> usize;
    /// Whether `node` is in the tree.
    fn contains(&self, node: usize) -> bool;
    /// The parent of `node`, a node of the tree.
    fn parent(&self, node: usize) -> Option<usize>;
    /// Whether `ancestor` is an ancestor of `node`, or `node` itself.
    fn dominates(&self, ancestor: usize, node: usize) -> bool;
    /// Every node of the tree.
    fn nodes(&self) -> impl Iterator<Item = usize> + '_;
}

/// A certificate as the checker reads it, in dense indices.
pub(super) trait Order {
    /// How many nodes it lists.
    fn len(&self) -> usize;
    /// The node at `position`.
    fn at(&self, position: usize) -> usize;
    /// The position of `node`, if it is listed.
    fn position(&self, node: usize) -> Option<usize>;
}

impl<N: DenseId> Order for LowHighOrder<N> {
    fn len(&self) -> usize {
        self.order.len()
    }

    fn at(&self, position: usize) -> usize {
        self.order[position].index()
    }

    fn position(&self, node: usize) -> Option<usize> {
        let position = *self.position.get(node)?;
        (position != ABSENT).then(|| slot(position))
    }
}

/// A dominator tree read from one of its roots.
struct Candidate<'a, N> {
    tree: &'a DominatorTree<N>,
    root: usize,
}

impl<N: DenseId> Tree for Candidate<'_, N> {
    fn root(&self) -> usize {
        self.root
    }

    fn size(&self) -> usize {
        self.tree.order.len()
    }

    /// A node index past the tree's tables, which a certificate made for
    /// another graph can name, is not in the tree.
    fn contains(&self, node: usize) -> bool {
        self.tree.pre.get(node).is_some_and(|&pre| pre != ABSENT)
    }

    fn parent(&self, node: usize) -> Option<usize> {
        let parent = self.tree.idom[node];
        (parent != ABSENT).then(|| slot(parent))
    }

    fn dominates(&self, ancestor: usize, node: usize) -> bool {
        self.tree
            .dominates(N::from_index(ancestor), N::from_index(node))
    }

    fn nodes(&self) -> impl Iterator<Item = usize> + '_ {
        self.tree.order.iter().map(|node| node.index())
    }
}

/// What [`check`] found wrong, in dense indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Failure {
    Preorder(usize),
    ParentProperty(usize, usize),
    LowHigh(usize),
}

impl Failure {
    /// The same failure with every index named by `name`.
    pub(super) fn lift<N>(self, name: impl Fn(usize) -> N) -> CertificateError<N> {
        match self {
            Self::Preorder(node) => CertificateError::Preorder(name(node)),
            Self::ParentProperty(tail, head) => CertificateError::ParentProperty {
                tail: name(tail),
                head: name(head),
            },
            Self::LowHigh(node) => CertificateError::LowHigh(name(node)),
        }
    }
}

/// Check that `tree`, whose root has no parent, is the dominator tree of
/// `graph` from its root, given the low-high order `order`.
///
/// Every node of `order` must be in `tree` at the position `order` gives it,
/// the first must be the root, and every later one must have a parent that is
/// an ancestor of the node before it, which is what makes the sequence a
/// preorder. With as many entries as the tree has nodes, it lists them all.
/// Then every edge out of the tree must keep the parent property, and every
/// node but the root must have its low and high in-edges.
pub(super) fn check<G: GraphView>(
    graph: &G,
    tree: &impl Tree,
    order: &impl Order,
) -> Result<(), Failure> {
    let root = tree.root();
    for position in 0..order.len() {
        let node = order.at(position);
        let placed = tree.contains(node)
            && order.position(node) == Some(position)
            && if position == 0 {
                node == root
            } else {
                tree.parent(node)
                    .is_some_and(|parent| tree.dominates(parent, order.at(position - 1)))
            };
        if !placed {
            return Err(Failure::Preorder(node));
        }
    }
    if order.len() != tree.size() {
        let missing = tree
            .nodes()
            .find(|&node| order.position(node).is_none())
            .unwrap_or(root);
        return Err(Failure::Preorder(missing));
    }

    for position in 0..order.len() {
        let tail = order.at(position);
        for head in graph.successors(G::NodeId::from_index(tail)) {
            let head = head.index();
            let kept = tree.contains(head)
                && (head == root
                    || tree
                        .parent(head)
                        .is_some_and(|parent| tree.dominates(parent, tail)));
            if !kept {
                return Err(Failure::ParentProperty(tail, head));
            }
        }
    }

    for position in 1..order.len() {
        let node = order.at(position);
        let parent = tree.parent(node);
        let (mut low, mut high) = (false, false);
        for before in graph.predecessors(G::NodeId::from_index(node)) {
            let before = before.index();
            if Some(before) == parent {
                (low, high) = (true, true);
                break;
            }
            if !tree.contains(before) {
                continue;
            }
            match order.position(before) {
                Some(other) if other < position => low = true,
                Some(other) if other > position && !tree.dominates(node, before) => high = true,
                _ => {}
            }
        }
        if !(low && high) {
            return Err(Failure::LowHigh(node));
        }
    }
    Ok(())
}
