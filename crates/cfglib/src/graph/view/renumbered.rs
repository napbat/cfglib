//! A node subset as a graph of its own, renumbered densely.

extern crate alloc;
use alloc::vec::Vec;

use super::{DenseId, GraphView};
use crate::identity::{NO_INDEX, slot, word};

/// The position of each node of a graph in a node list, the table a
/// [`Renumbered`] view reads.
///
/// A scratch: it starts empty, grows to the graph on [`number`](Self::number),
/// and [`clear`](Self::clear) resets only the entries a list set, so a pass
/// that renumbers many small components of one graph pays O(component) per
/// component rather than O(graph).
#[derive(Debug, Clone, Default)]
pub(crate) struct Positions {
    /// Per node of the graph, its position in the list, [`NO_INDEX`] for none.
    local: Vec<u32>,
}

impl Positions {
    /// Number `nodes` by their position in the list, over a graph of
    /// `node_bound` nodes.
    ///
    /// # Panics
    ///
    /// Panics when a node appears twice, or when a node is not below
    /// `node_bound`.
    pub(crate) fn number<N: DenseId>(&mut self, node_bound: usize, nodes: &[N]) {
        if self.local.len() < node_bound {
            self.local.resize(node_bound, NO_INDEX);
        }
        for (position, &node) in nodes.iter().enumerate() {
            assert!(
                node.index() < node_bound,
                "a renumbered node is out of range"
            );
            let entry = &mut self.local[node.index()];
            assert!(
                *entry == NO_INDEX,
                "node {} appears twice in the renumbered list",
                node.index()
            );
            *entry = word(position);
        }
    }

    /// Forget the positions of `nodes`, the list last numbered.
    pub(crate) fn clear<N: DenseId>(&mut self, nodes: &[N]) {
        for node in nodes {
            self.local[node.index()] = NO_INDEX;
        }
    }

    /// The position of the node at dense index `index`, when it is listed.
    #[inline]
    fn position(&self, index: usize) -> Option<usize> {
        let position = self.local[index];
        (position != NO_INDEX).then_some(slot(position))
    }
}

/// A list of nodes of a view as a graph of its own, each named by its
/// position in the list.
///
/// Every edge of the view between two listed nodes is an edge here, once for
/// each time the view has it, and no other edge is. Dense positions let an
/// analysis that runs once per component (a strongly connected component, a
/// sink class, a region) size its tables by the component instead of by the
/// whole graph.
pub(crate) struct Renumbered<'a, G: GraphView> {
    graph: &'a G,
    nodes: &'a [G::NodeId],
    positions: &'a Positions,
}

impl<'a, G: GraphView> Renumbered<'a, G> {
    /// The view of `nodes`, which `positions` must have numbered.
    pub(crate) fn new(graph: &'a G, nodes: &'a [G::NodeId], positions: &'a Positions) -> Self {
        debug_assert!(
            nodes
                .iter()
                .enumerate()
                .all(|(position, node)| positions.position(node.index()) == Some(position)),
            "the positions of a renumbered view number its nodes"
        );
        Self {
            graph,
            nodes,
            positions,
        }
    }

    /// The position of `node`, when it is listed.
    #[inline]
    pub(crate) fn position(&self, node: G::NodeId) -> Option<usize> {
        self.positions.position(node.index())
    }
}

impl<G: GraphView> GraphView for Renumbered<'_, G> {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.nodes.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.nodes.len()
    }

    #[inline]
    fn successors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.graph
            .successors(self.nodes[node])
            .filter_map(|next| self.position(next))
    }

    #[inline]
    fn predecessors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.graph
            .predecessors(self.nodes[node])
            .filter_map(|before| self.position(before))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::store::Graph;

    #[test]
    fn listed_nodes_keep_the_edges_between_them() {
        let mut graph = Graph::new();
        let [a, b, c, d] = ["a", "b", "c", "d"].map(|name| graph.add_node(name));
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, b, ());
        graph.add_edge(c, d, ());

        let nodes = [c, b];
        let mut positions = Positions::default();
        positions.number(graph.node_bound(), &nodes);
        let view = Renumbered::new(&graph, &nodes, &positions);
        assert_eq!(view.node_bound(), 2);
        assert!(view.successors(0).eq([1]));
        assert!(view.predecessors(1).eq([0]));
        assert_eq!(view.position(a), None);

        positions.clear(&nodes);
        positions.number(graph.node_bound(), &[a]);
        let single = [a];
        let view = Renumbered::new(&graph, &single, &positions);
        assert_eq!(view.successors(0).count(), 0);
    }

    #[test]
    #[should_panic(expected = "appears twice")]
    fn a_repeated_node_is_rejected() {
        let mut graph = Graph::<(), ()>::new();
        let a = graph.add_node(());
        Positions::default().number(graph.node_bound(), &[a, a]);
    }
}
