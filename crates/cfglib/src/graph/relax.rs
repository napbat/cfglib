//! Edge-defined minimum-label relaxation over [`Graph`].
//!
//! Unlike node-only traversals, relaxation retains parallel edge identities and
//! may revisit a node whenever a smaller label reaches it. This supports
//! shortest-path-shaped analyses whose transfer is defined by each edge rather
//! than by the destination node.

extern crate alloc;
use alloc::vec::Vec;

use super::store::{EdgeRecord, Graph, NodeId};
use super::traverse::TraversalDirection;

/// Compute the minimum label reachable at every node from `seeds`.
///
/// `relax` receives each real edge in its stored direction plus the current
/// label at the node being expanded. It returns the candidate label for the
/// far endpoint, or `None` to reject that edge. A node is expanded again every
/// time its known label decreases, so improvements propagate through cycles
/// and through nodes visited earlier with a larger label. Incoming traversal
/// changes which endpoint is considered far but does not reverse the edge
/// passed to `relax`.
///
/// Labels need only be ordered; they do not need to be cloneable. For a stable,
/// order-independent result, edge transfer must preserve improvements: a
/// smaller input label cannot lose a candidate or produce a larger candidate
/// than a larger input label. The reachable label space must also have no
/// infinite strictly descending chain so the worklist terminates.
///
/// # Panics
///
/// Panics when a seed node does not belong to `graph`.
///
/// # Examples
///
/// ```rust
/// use cfglib::{Graph, TraversalDirection, min_label_relaxation};
///
/// let mut graph = Graph::new();
/// let start = graph.add_node(());
/// let middle = graph.add_node(());
/// let end = graph.add_node(());
/// graph.add_edge(start, end, 9_u32);
/// graph.add_edge(start, middle, 2);
/// graph.add_edge(middle, end, 3);
///
/// let distances = min_label_relaxation(
///     &graph,
///     [(start, 0_u32)],
///     TraversalDirection::Outgoing,
///     |edge, distance| distance.checked_add(*edge.payload()),
/// );
/// assert_eq!(distances, [Some(0), Some(2), Some(5)]);
/// ```
#[must_use]
pub fn min_label_relaxation<N, E, L>(
    graph: &Graph<N, E>,
    seeds: impl IntoIterator<Item = (NodeId, L)>,
    direction: TraversalDirection,
    mut relax: impl FnMut(&EdgeRecord<E>, &L) -> Option<L>,
) -> Vec<Option<L>>
where
    L: Ord,
{
    let mut labels: Vec<Option<L>> = (0..graph.node_bound()).map(|_| None).collect();
    let mut worklist: Vec<(NodeId, L)> = seeds.into_iter().collect();
    let forward = matches!(direction, TraversalDirection::Outgoing);

    while let Some((node, candidate)) = worklist.pop() {
        assert!(node.index() < labels.len(), "seed node is out of range");
        if labels[node.index()]
            .as_ref()
            .is_some_and(|known| known <= &candidate)
        {
            continue;
        }
        labels[node.index()] = Some(candidate);
        let label = labels[node.index()]
            .as_ref()
            .expect("the candidate label was just stored");
        let adjacency: Vec<_> = if forward {
            graph.outgoing(node).collect()
        } else {
            graph.incoming(node).collect()
        };
        for edge_id in adjacency {
            let edge = graph.edge(edge_id);
            if let Some(candidate) = relax(edge, label) {
                let far = if forward {
                    edge.target()
                } else {
                    edge.source()
                };
                worklist.push((far, candidate));
            }
        }
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    enum Rule {
        Mint(u32),
        Carry,
        Block,
    }

    fn apply_rule(edge: &EdgeRecord<Rule>, label: u32) -> Option<u32> {
        match *edge.payload() {
            Rule::Mint(label) => Some(label),
            Rule::Carry => Some(label),
            Rule::Block => None,
        }
    }

    #[test]
    fn smaller_label_reexpands_a_previously_visited_node() {
        let mut graph = Graph::new();
        let root = graph.add_node(());
        let bridge = graph.add_node(());
        let join = graph.add_node(());
        let tail = graph.add_node(());
        let blocked = graph.add_node(());
        graph.add_edge(root, bridge, Rule::Mint(2));
        graph.add_edge(root, join, Rule::Mint(9));
        graph.add_edge(bridge, join, Rule::Carry);
        graph.add_edge(join, tail, Rule::Carry);
        graph.add_edge(root, blocked, Rule::Block);

        let labels = min_label_relaxation(
            &graph,
            [(root, 0)],
            TraversalDirection::Outgoing,
            |edge, label| apply_rule(edge, *label),
        );

        assert_eq!(labels, [Some(0), Some(2), Some(2), Some(2), None]);
    }

    #[test]
    fn incoming_relaxation_uses_the_real_edge_payload() {
        let mut graph = Graph::new();
        let source = graph.add_node(());
        let middle = graph.add_node(());
        let sink = graph.add_node(());
        graph.add_edge(source, middle, Rule::Carry);
        graph.add_edge(middle, sink, Rule::Carry);

        let labels = min_label_relaxation(
            &graph,
            [(sink, 4)],
            TraversalDirection::Incoming,
            |edge, label| apply_rule(edge, *label),
        );

        assert_eq!(labels, [Some(4), Some(4), Some(4)]);
    }

    #[test]
    fn parallel_edges_and_competing_seeds_choose_the_minimum() {
        let mut graph = Graph::new();
        let source = graph.add_node(());
        let target = graph.add_node(());
        graph.add_edge(source, target, Rule::Carry);
        graph.add_edge(source, target, Rule::Mint(3));

        let labels = min_label_relaxation(
            &graph,
            [(source, 1), (source, 7)],
            TraversalDirection::Outgoing,
            |edge, label| apply_rule(edge, *label),
        );

        assert_eq!(labels, [Some(1), Some(1)]);
    }

    #[test]
    #[should_panic(expected = "seed node is out of range")]
    fn rejects_a_seed_from_outside_the_graph() {
        let graph = Graph::<(), ()>::new();
        let _ = min_label_relaxation(
            &graph,
            [(NodeId::from_index(0), 0_u32)],
            TraversalDirection::Outgoing,
            |_, label| Some(*label),
        );
    }
}
