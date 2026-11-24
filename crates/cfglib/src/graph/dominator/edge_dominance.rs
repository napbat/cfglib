//! Edge dominance: whether every path from the root to a node takes one edge.
//!
//! An edge `u -> v` dominates `w` when every path from the root to `w` takes
//! it. For a reachable `v` other than a root, that holds exactly when `v`
//! dominates `w` and the edge is the only in-edge of `v` whose tail is
//! reachable and not dominated by `v`:
//!
//! - every path to `w` meets `v`, and the edge that first enters `v` comes
//!   from a tail reached without passing `v`, which `v` does not dominate, so
//!   it is that one edge;
//! - an in-edge from any other such tail starts a path that enters `v`
//!   without the edge and then goes on to `w`.
//!
//! The test reads only the in-edges of `v` and the tree's O(1) dominance
//! test, so it needs no table of its own. Parallel edges are separate
//! in-edges: neither of two parallel edges dominates anything.

use super::DominatorTree;
use crate::graph::edge_view::EdgeView;
use crate::graph::view::{DenseId, GraphView};

impl<N: DenseId> DominatorTree<N> {
    /// Return whether every path from the root to `node` takes the edge
    /// `from -> to`.
    ///
    /// The edge dominates exactly the nodes `to` dominates when it is the
    /// [`dominating_in_edge`](Self::dominating_in_edge) of `to`, and nothing
    /// otherwise. So the answer is `false` when the graph has no such edge or
    /// has it more than once, when `to` is a root, which no path re-enters,
    /// and when `from` or `node` is unreachable. A self-loop dominates
    /// nothing.
    ///
    /// `graph` must be the graph the tree was computed from. In a forest
    /// built by [`complete_forest_in`](Self::complete_forest_in), each root
    /// counts as entered from the virtual root that joins them, so an edge
    /// into a root dominates nothing there either.
    ///
    /// The call reads the predecessors of `to` once, so it costs
    /// O(indegree(`to`)) and allocates nothing.
    ///
    /// # Panics
    ///
    /// Panics when `to` or `node`, or a predecessor of `to`, has an index
    /// beyond the node bound the tree was computed with.
    ///
    /// # Examples
    ///
    /// A fact a branch establishes on one arm holds in exactly the blocks
    /// that arm dominates:
    ///
    /// ```
    /// use cfglib::{Cfg, DominatorTree, EdgeKind};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let branch = cfg.entry();
    /// let then_block = cfg.new_block();
    /// let else_block = cfg.new_block();
    /// let join = cfg.new_block();
    /// cfg.add_edge(branch, then_block, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(branch, else_block, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(then_block, join, EdgeKind::Unconditional);
    /// cfg.add_edge(else_block, join, EdgeKind::Unconditional);
    ///
    /// let dom = DominatorTree::compute(&cfg);
    /// assert!(dom.edge_dominates(&cfg, branch, then_block, then_block));
    /// assert!(!dom.edge_dominates(&cfg, branch, then_block, join));
    /// assert!(!dom.edge_dominates(&cfg, then_block, join, join));
    /// assert_eq!(dom.dominating_in_edge(&cfg, else_block), Some(branch));
    /// assert_eq!(dom.dominating_in_edge(&cfg, join), None);
    /// ```
    #[must_use]
    pub fn edge_dominates<G>(&self, graph: &G, from: N, to: N, node: N) -> bool
    where
        G: GraphView<NodeId = N>,
    {
        self.dominates(to, node) && self.dominating_in_edge(graph, to) == Some(from)
    }

    /// Return the tail of the one in-edge of `node` that every path from the
    /// root to `node` takes, if there is one.
    ///
    /// That edge is the bridge into `node`: removing it cuts `node`, and
    /// everything `node` dominates, off from the root. It exists when `node`
    /// is reachable, is not a root, and has exactly one in-edge whose tail is
    /// reachable and not dominated by `node`. Each parallel edge counts, so a
    /// node entered over two parallel edges and nothing else has none.
    ///
    /// `graph` must be the graph the tree was computed from. The call costs
    /// O(indegree(`node`)) and allocates nothing.
    ///
    /// # Panics
    ///
    /// Panics when `node`, or one of its predecessors, has an index beyond
    /// the node bound the tree was computed with.
    #[must_use]
    pub fn dominating_in_edge<G>(&self, graph: &G, node: N) -> Option<N>
    where
        G: GraphView<NodeId = N>,
    {
        if !self.is_reachable(node) || self.idom(node).is_none() {
            return None;
        }
        let mut entries = graph
            .predecessors(node)
            .filter(|&tail| self.enters(node, tail));
        let entry = entries.next()?;
        entries.next().is_none().then_some(entry)
    }

    /// Return whether every path from the root to `node` takes `edge`.
    ///
    /// This is [`edge_dominates`](Self::edge_dominates) for a view with
    /// stable edge identities, where parallel edges are distinct: `edge`
    /// dominates `node` when its head dominates `node`, is reachable and is
    /// not a root, and every other in-edge of its head comes from a node the
    /// head dominates or from an unreachable node. Over a
    /// [`FilteredEdges`](crate::FilteredEdges) view the question is asked of
    /// the edges the filter keeps, so the tree must be computed over that same
    /// view.
    ///
    /// The call reads the in-edges of the head once, so it costs
    /// O(indegree) and allocates nothing.
    ///
    /// # Panics
    ///
    /// Panics when `edge` is not an edge of `graph`, and when its head,
    /// `node`, or the tail of an in-edge of its head has an index beyond the
    /// node bound the tree was computed with.
    #[must_use]
    pub fn edge_id_dominates<G>(&self, graph: &G, edge: G::EdgeId, node: N) -> bool
    where
        G: EdgeView<NodeId = N>,
    {
        let head = graph.edge(edge).target();
        if !self.is_reachable(head) || self.idom(head).is_none() || !self.dominates(head, node) {
            return false;
        }
        let mut entries = graph
            .incoming(head)
            .filter(|&incoming| self.enters(head, graph.edge(incoming).source()));
        entries.next() == Some(edge) && entries.next().is_none()
    }

    /// Whether an edge from `tail` can be the first to enter `head`: the
    /// root reaches `tail` without passing `head`.
    fn enters(&self, head: N, tail: N) -> bool {
        self.is_reachable(tail) && !self.dominates(head, tail)
    }
}
