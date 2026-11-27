//! The subgraph a node set induces, as a view.

use super::{DenseId, GraphView, NodeView};
use crate::graph::edge_view::{EdgeRef, EdgeView};

/// Adapter presenting the subgraph of a view that a node set induces.
///
/// A node is a node of the view when `member` holds `true` at its index, and
/// an edge of the underlying view is an edge here exactly when both of its
/// ends are members, once for each time the underlying view has it. A node
/// outside the set has no edges, so a walk from a member never leaves the set.
///
/// The membership table is borrowed, so confining an algorithm to a region —
/// the blocks the entry reaches, one loop body, the part of a cycle left once
/// its header is removed — costs no copy of the graph.
///
/// # Examples
///
/// ```
/// use cfglib::{Direction, Graph, Induced, reachable};
///
/// // a -> b -> c, and a -> c directly.
/// let mut graph = Graph::<&str, ()>::new();
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let c = graph.add_node("c");
/// graph.add_edge(a, b, ());
/// graph.add_edge(b, c, ());
///
/// // Without `b`, nothing leads from `a` to `c`.
/// let member = [true, false, true];
/// let region = Induced::new(&graph, &member);
/// assert_eq!(reachable(&region, [a], Direction::Forward), [true, false, false]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Induced<'g, G: GraphView> {
    graph: &'g G,
    member: &'g [bool],
}

impl<'g, G: GraphView> Induced<'g, G> {
    /// The subgraph of `graph` induced by the nodes whose `member` entry is
    /// `true`.
    ///
    /// # Panics
    ///
    /// Panics when `member` is shorter than the node bound of `graph`.
    #[must_use]
    pub fn new(graph: &'g G, member: &'g [bool]) -> Self {
        assert!(
            member.len() >= graph.node_bound(),
            "the membership table of an induced view covers {} nodes but the graph has {}",
            member.len(),
            graph.node_bound()
        );
        Self { graph, member }
    }

    /// Borrow the underlying view.
    #[must_use]
    pub const fn graph(&self) -> &'g G {
        self.graph
    }

    /// Whether `node` is a node of this view.
    ///
    /// # Panics
    ///
    /// Panics when `node` is outside the membership table.
    #[inline]
    #[must_use]
    pub fn contains(&self, node: G::NodeId) -> bool {
        self.member[node.index()]
    }
}

impl<G: GraphView> GraphView for Induced<'_, G> {
    type NodeId = G::NodeId;

    fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.graph.node_ids().filter(|&node| self.contains(node))
    }

    #[inline]
    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        let inside = self.contains(node);
        self.graph
            .successors(node)
            .filter(move |&next| inside && self.contains(next))
    }

    #[inline]
    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        let inside = self.contains(node);
        self.graph
            .predecessors(node)
            .filter(move |&before| inside && self.contains(before))
    }
}

impl<G: NodeView> NodeView for Induced<'_, G> {
    type NodeData = G::NodeData;

    fn node(&self, node: Self::NodeId) -> &Self::NodeData {
        self.graph.node(node)
    }
}

impl<G: EdgeView> EdgeView for Induced<'_, G> {
    type EdgeId = G::EdgeId;
    type EdgeData = G::EdgeData;

    fn edge_bound(&self) -> usize {
        self.graph.edge_bound()
    }

    fn edge_ids(&self) -> impl Iterator<Item = Self::EdgeId> + '_ {
        self.graph.edge_ids().filter(|&edge| {
            let edge = self.graph.edge(edge);
            self.contains(edge.source()) && self.contains(edge.target())
        })
    }

    fn outgoing(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        let inside = self.contains(node);
        self.graph
            .outgoing(node)
            .filter(move |&edge| inside && self.contains(self.graph.edge(edge).target()))
    }

    fn incoming(&self, node: Self::NodeId) -> impl Iterator<Item = Self::EdgeId> + '_ {
        let inside = self.contains(node);
        self.graph
            .incoming(node)
            .filter(move |&edge| inside && self.contains(self.graph.edge(edge).source()))
    }

    fn edge(&self, edge: Self::EdgeId) -> EdgeRef<'_, Self::NodeId, Self::EdgeId, Self::EdgeData> {
        self.graph.edge(edge)
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec::Vec;

    use super::*;
    use crate::graph::store::Graph;

    #[test]
    fn only_edges_between_members_remain() {
        let mut graph = Graph::new();
        let [a, b, c] = ["a", "b", "c"].map(|name| graph.add_node(name));
        graph.add_edge(a, b, ());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, a, ());

        let member = [true, true, false];
        let view = Induced::new(&graph, &member);
        assert_eq!(view.node_ids().collect::<Vec<_>>(), [a, b]);
        assert_eq!(view.successors(a).collect::<Vec<_>>(), [b, b]);
        assert_eq!(view.predecessors(a).count(), 0);
        assert_eq!(view.successors(c).count(), 0, "a non-member has no edges");
        assert_eq!(view.edge_ids().count(), 2);
        assert_eq!(view.outgoing(b).count(), 0);
        assert_eq!(view.incoming(b).count(), 2);
    }

    #[test]
    #[should_panic(expected = "covers 2 nodes but the graph has 3")]
    fn a_short_membership_table_is_rejected() {
        let mut graph = Graph::<(), ()>::new();
        for () in [(); 3] {
            graph.add_node(());
        }
        let _ = Induced::new(&graph, &[true, true]);
    }
}
