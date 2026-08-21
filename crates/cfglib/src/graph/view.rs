//! Read-only graph view traits consumed by the generic algorithms.
//!
//! [`DirectedGraphView`] abstracts forward/reverse adjacency over dense node
//! identities so traversals, SCC computation, dominance, and coloring run on
//! [`DirectedGraph`](super::directed::DirectedGraph),
//! [`Cfg`](crate::Cfg), or consumer-owned storage without migration.
//! [`RootedGraphView`] adds a distinguished entry node for algorithms that
//! need one (dominators, reachability, structural analysis); [`Rooted`] roots
//! any plain view at a chosen node.

/// A copyable, ordered node identity backed by a dense zero-based index.
///
/// Implementations of [`DirectedGraphView`] must yield every index in
/// `0..node_count()` exactly once. This contract lets graph algorithms use
/// compact vectors instead of imposing hashing on consumer identities. Dense
/// `u32` and `usize` handles implement this trait directly.
pub trait DenseNodeId: Copy + Ord {
    /// Construct an identity from a valid dense zero-based index.
    fn from_index(index: usize) -> Self;

    /// Return the identity's dense zero-based index.
    fn index(self) -> usize;
}

impl DenseNodeId for usize {
    fn from_index(index: usize) -> Self {
        index
    }

    fn index(self) -> usize {
        self
    }
}

impl DenseNodeId for u32 {
    fn from_index(index: usize) -> Self {
        Self::try_from(index).expect("node index exceeds u32::MAX")
    }

    fn index(self) -> usize {
        usize::try_from(self).expect("u32 node index exceeds usize::MAX")
    }
}

/// Read-only directed adjacency consumed by generic graph algorithms.
///
/// A view may be backed by
/// [`DirectedGraph`](super::directed::DirectedGraph), [`Cfg`](crate::Cfg),
/// or a consumer-owned structure. Node identities must follow the
/// [`DenseNodeId`] contract.
///
/// Forward and reverse adjacency must describe the same edge **multiset**.
/// Every occurrence of `target` in `successors(source)` must have exactly one
/// matching occurrence of `source` in `predecessors(target)`, including each
/// parallel edge. Algorithms may combine the two directions without rebuilding
/// or deduplicating either one.
pub trait DirectedGraphView {
    /// Node identity used by this view.
    type NodeId: DenseNodeId;

    /// Return the number of nodes in the view.
    fn node_count(&self) -> usize;

    /// Iterate over every node identity exactly once.
    ///
    /// Dense identities make this implementation universal, so adapters only
    /// need to expose node count plus forward and reverse adjacency.
    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
        (0..self.node_count()).map(Self::NodeId::from_index)
    }

    /// Iterate over the outgoing neighbors of `node`.
    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_;

    /// Iterate over the incoming neighbors of `node`.
    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_;
}

/// A directed-graph view with a distinguished root/entry node.
///
/// Entry-requiring algorithms (dominance, reachability metrics, interval and
/// loop analysis) take this trait instead of a separate root argument, so a
/// [`Cfg`](crate::Cfg) participates directly through its entry block while
/// consumer graphs opt in via [`Rooted`] or their own implementation.
pub trait RootedGraphView: DirectedGraphView {
    /// The root node from which reachability, dominance, and orderings are
    /// computed.
    fn root(&self) -> Self::NodeId;
}

/// Adapter that roots any [`DirectedGraphView`] at a chosen node.
///
/// Consumer-owned graphs that have no intrinsic entry (value-flow graphs,
/// type-relation graphs) use this to run entry-requiring algorithms without
/// implementing [`RootedGraphView`] on their storage.
///
/// # Examples
///
/// ```
/// use cfglib::{DirectedGraph, DominatorTree, Rooted};
///
/// let mut graph = DirectedGraph::new();
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// graph.add_edge(a, b, ());
///
/// let rooted = Rooted::new(&graph, a);
/// let dominators = DominatorTree::compute(&rooted);
/// assert_eq!(dominators.idom(b), Some(a));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Rooted<'g, G: DirectedGraphView> {
    graph: &'g G,
    root: G::NodeId,
}

impl<'g, G: DirectedGraphView> Rooted<'g, G> {
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

impl<G: DirectedGraphView> DirectedGraphView for Rooted<'_, G> {
    type NodeId = G::NodeId;

    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.successors(node)
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.predecessors(node)
    }
}

impl<G: DirectedGraphView> RootedGraphView for Rooted<'_, G> {
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
/// use cfglib::{DirectedGraph, Reversed, Rooted, DominatorTree};
///
/// let mut graph = DirectedGraph::new();
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// graph.add_edge(a, b, ());
///
/// let reversed = Reversed::new(&graph);
/// let dominators = DominatorTree::compute(&Rooted::new(&reversed, b));
/// assert_eq!(dominators.idom(a), Some(b));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Reversed<'g, G: DirectedGraphView> {
    graph: &'g G,
}

impl<'g, G: DirectedGraphView> Reversed<'g, G> {
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

impl<G: DirectedGraphView> DirectedGraphView for Reversed<'_, G> {
    type NodeId = G::NodeId;

    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.predecessors(node)
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.successors(node)
    }
}
