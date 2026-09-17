//! Read-only graph view traits consumed by the generic algorithms.
//!
//! [`GraphView`] abstracts forward and reverse adjacency over dense node
//! identities, so traversals, SCC computation, dominance, and coloring run on
//! [`Graph`](crate::Graph), [`Cfg`](crate::Cfg), or consumer-owned storage
//! without migration. [`NodeView`] adds access to graph-owned node payloads,
//! [`EdgeView`](super::edge_view::EdgeView) adds edge identity and data, and
//! [`RootedView`] adds a distinguished entry node for the algorithms that need
//! one (dominators, reachability, structural analysis); [`Rooted`] roots any
//! plain view at a chosen node.
//!
//! # The dense-identity contract
//!
//! A view has a **bound** and a set of **live identities**. The bound is an
//! exclusive upper limit on every identity the view yields, so an analysis
//! sizes a side table by [`GraphView::node_bound`] and indexes it by
//! [`DenseId::index`] without a bounds failure. The live identities are what
//! [`GraphView::node_ids`] yields, and a store that removed a node simply
//! stops yielding it: a removed node is not a node of the view, not an
//! isolated one.
//!
//! The two numbers coincide exactly when nothing has been removed, which is
//! the ordinary case; keeping them distinct is what lets a mutable store
//! serve an analysis without first being compacted.

use core::fmt::Debug;
use core::hash::Hash;

/// A copyable, ordered identity backed by a dense zero-based index.
///
/// One trait covers nodes and edges: the contract is identical for both, and
/// splitting it only forced every dense identity in the crate to implement
/// the same two methods twice. The supertraits are what the algorithms need
/// of an identity — comparison for ordered sets, hashing for keyed side
/// tables, and formatting for diagnostics.
///
/// Implementations must round-trip: `Self::from_index(id.index()) == id` for
/// every identity a view yields. Dense `u32` and `usize` handles implement
/// this trait directly.
pub trait DenseId: Copy + Ord + Hash + Debug {
    /// Construct an identity from a valid dense zero-based index.
    fn from_index(index: usize) -> Self;

    /// Return the identity's dense zero-based index.
    fn index(self) -> usize;
}

impl DenseId for usize {
    fn from_index(index: usize) -> Self {
        index
    }

    fn index(self) -> usize {
        self
    }
}

impl DenseId for u32 {
    fn from_index(index: usize) -> Self {
        Self::try_from(index).expect("dense index exceeds u32::MAX")
    }

    fn index(self) -> usize {
        usize::try_from(self).expect("u32 dense index exceeds usize::MAX")
    }
}

/// Read-only directed adjacency consumed by generic graph algorithms.
///
/// A view may be backed by [`Graph`](crate::Graph), [`Cfg`](crate::Cfg), or a
/// consumer-owned structure. Node identities follow the [`DenseId`] contract.
///
/// # Contract
///
/// - Every identity yielded by [`node_ids`](Self::node_ids), by
///   [`successors`](Self::successors), or by
///   [`predecessors`](Self::predecessors) has an
///   [`index`](DenseId::index) below [`node_bound`](Self::node_bound).
/// - [`node_ids`](Self::node_ids) yields every live node exactly once. The
///   stores in this crate yield ascending index order; an algorithm must not
///   depend on that, because a consumer-owned view need not.
/// - Forward and reverse adjacency describe the same edge **multiset**: every
///   occurrence of `target` in `successors(source)` has exactly one matching
///   occurrence of `source` in `predecessors(target)`, parallel edges
///   included. Algorithms may combine the two directions without rebuilding
///   or deduplicating either one.
pub trait GraphView {
    /// Node identity used by this view.
    type NodeId: DenseId;

    /// An exclusive upper bound on every node index this view yields.
    ///
    /// This is the correct size for a node-indexed side table. It is at least
    /// the number of live nodes and may exceed it when the backing store has
    /// removed nodes without compacting.
    fn node_bound(&self) -> usize;

    /// Iterate over every live node identity exactly once.
    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_;

    /// Iterate over the outgoing neighbors of `node`.
    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_;

    /// Iterate over the incoming neighbors of `node`.
    ///
    /// A view that stores only forward adjacency can keep this contract
    /// honest with [`scan_predecessors`] — a documented O(nodes × edges)
    /// scan — instead of maintaining a reverse index it never queries.
    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_;
}

/// Read-only node payloads associated with a [`GraphView`].
///
/// Adjacency-only algorithms keep depending on [`GraphView`]. Consumers that
/// need graph-owned node data use this companion trait without depending on a
/// particular store.
pub trait NodeView: GraphView {
    /// Data exposed for each node.
    type NodeData: ?Sized;

    /// Borrow one node's data.
    ///
    /// # Panics
    ///
    /// Panics when `node` does not belong to this view.
    fn node(&self, node: Self::NodeId) -> &Self::NodeData;
}

/// A graph view with a distinguished root/entry node.
///
/// Entry-requiring algorithms (dominance, reachability metrics, interval and
/// loop analysis) take this trait instead of a separate root argument, so a
/// [`Cfg`](crate::Cfg) participates directly through its entry block while
/// consumer graphs opt in via [`Rooted`] or their own implementation.
pub trait RootedView: GraphView {
    /// The root node from which reachability, dominance, and orderings are
    /// computed.
    fn root(&self) -> Self::NodeId;
}

/// Adapter that roots any [`GraphView`] at a chosen node.
///
/// Consumer-owned graphs that have no intrinsic entry (value-flow graphs,
/// type-relation graphs) use this to run entry-requiring algorithms without
/// implementing [`RootedView`] on their storage.
///
/// # Examples
///
/// ```
/// use cfglib::{DominatorTree, Graph, Rooted};
///
/// let mut graph = Graph::new();
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// graph.add_edge(a, b, ());
///
/// let rooted = Rooted::new(&graph, a);
/// let dominators = DominatorTree::compute(&rooted);
/// assert_eq!(dominators.idom(b), Some(a));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Rooted<'g, G: GraphView> {
    graph: &'g G,
    root: G::NodeId,
}

impl<'g, G: GraphView> Rooted<'g, G> {
    /// Root `graph` at `root`.
    #[must_use]
    pub const fn new(graph: &'g G, root: G::NodeId) -> Self {
        Self { graph, root }
    }

    /// Borrow the underlying view.
    #[must_use]
    pub const fn graph(&self) -> &'g G {
        self.graph
    }
}

impl<G: GraphView> GraphView for Rooted<'_, G> {
    type NodeId = G::NodeId;

    fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.node_ids()
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.successors(node)
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.predecessors(node)
    }
}

impl<G: NodeView> NodeView for Rooted<'_, G> {
    type NodeData = G::NodeData;

    fn node(&self, node: Self::NodeId) -> &Self::NodeData {
        self.graph.node(node)
    }
}

impl<G: GraphView> RootedView for Rooted<'_, G> {
    fn root(&self) -> Self::NodeId {
        self.root
    }
}

/// Adapter presenting a view with every edge reversed.
///
/// Successors become predecessors and vice versa, so forward algorithms
/// run backwards without copying the graph: reverse reachability,
/// dominators on the reverse relation, backward walks.
///
/// # Examples
///
/// ```
/// use cfglib::{DominatorTree, Graph, Reversed, Rooted};
///
/// let mut graph = Graph::new();
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// graph.add_edge(a, b, ());
///
/// let reversed = Reversed::new(&graph);
/// let dominators = DominatorTree::compute(&Rooted::new(&reversed, b));
/// assert_eq!(dominators.idom(a), Some(b));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Reversed<'g, G: GraphView> {
    graph: &'g G,
}

impl<'g, G: GraphView> Reversed<'g, G> {
    /// Reverse `graph`.
    #[must_use]
    pub const fn new(graph: &'g G) -> Self {
        Self { graph }
    }

    /// Borrow the underlying (unreversed) view.
    #[must_use]
    pub const fn graph(&self) -> &'g G {
        self.graph
    }
}

impl<G: GraphView> GraphView for Reversed<'_, G> {
    type NodeId = G::NodeId;

    fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.node_ids()
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.predecessors(node)
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.successors(node)
    }
}

impl<G: NodeView> NodeView for Reversed<'_, G> {
    type NodeData = G::NodeData;

    fn node(&self, node: Self::NodeId) -> &Self::NodeData {
        self.graph.node(node)
    }
}

/// The incoming neighbors of `node` by scanning every node's successors.
///
/// The honest [`GraphView::predecessors`] implementation for a view that
/// stores only forward adjacency: O(nodes × edges) per query, correct by
/// construction, and free of a reverse index the consumer never queries.
/// Views on an algorithm's hot reverse path should maintain real reverse
/// adjacency instead.
pub fn scan_predecessors<G: GraphView>(
    graph: &G,
    node: G::NodeId,
) -> impl Iterator<Item = G::NodeId> + '_ {
    graph.node_ids().filter(move |&candidate| {
        graph
            .successors(candidate)
            .any(|successor| successor == node)
    })
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec::Vec;

    use super::{GraphView, scan_predecessors};
    use crate::graph::store::Graph;

    #[test]
    fn scanning_matches_stored_reverse_adjacency() {
        let mut graph = Graph::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        graph.add_edge(a, c, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, a, ());

        for node in [a, b, c] {
            let mut scanned: Vec<_> = scan_predecessors(&graph, node).collect();
            let mut stored: Vec<_> = GraphView::predecessors(&graph, node).collect();
            scanned.sort_unstable();
            stored.sort_unstable();
            assert_eq!(scanned, stored);
        }
        assert!(scan_predecessors(&graph, b).next().is_none());
    }

    #[test]
    fn a_removed_node_leaves_the_view() {
        let mut graph = Graph::new();
        let kept = graph.add_node("kept");
        let dropped = graph.add_node("dropped");
        graph.add_edge(kept, dropped, ());
        graph.remove_node(dropped);

        assert_eq!(graph.node_ids().collect::<Vec<_>>(), [kept]);
        assert_eq!(graph.node_bound(), 2, "the bound still covers both slots");
        assert!(GraphView::successors(&graph, kept).next().is_none());
    }
}
