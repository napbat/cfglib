//! Edge-defined minimum-label relaxation over any [`EdgeView`].
//!
//! Unlike node-only traversals, relaxation retains parallel edge identities and
//! may revisit a node whenever a smaller label reaches it. This supports
//! shortest-path-shaped analyses whose transfer is defined by each edge rather
//! than by the destination node.

extern crate alloc;
use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Reverse;

use crate::graph::edge_traverse::EdgeAdjacency;
use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::traverse::by_axis;
use crate::graph::view::{DenseId, Direction, assert_in_bound};

/// Compute the minimum label reachable at every node from `seeds`.
///
/// `relax` receives each edge of the view as the view orients it, plus the
/// current label at the node being expanded. It returns the candidate label
/// for the far endpoint, or `None` to reject that edge. A node is expanded
/// again every time its known label decreases, so improvements propagate
/// through cycles and through nodes visited earlier with a larger label.
/// [`Direction::Backward`] walks the incoming edges, which changes which
/// endpoint is considered far but does not reverse the edge passed to
/// `relax`.
///
/// Labels need only be ordered; they do not need to be cloneable. For a stable,
/// order-independent result, edge transfer must preserve improvements: a
/// smaller input label cannot lose a candidate or produce a larger candidate
/// than a larger input label. The reachable label space must also have no
/// infinite strictly descending chain so the worklist terminates.
///
/// The result is indexed by [`DenseId::index`] and sized by the view's node
/// bound; a node no seed reaches holds `None`.
///
/// # Complexity
///
/// Pending candidates are expanded smallest label first, ties to the smaller
/// node index. When every transfer is extensive (`relax(e, l)` is `None` or at
/// least `l`, such as nonnegative edge costs added to a distance), the first
/// candidate a node expands already carries its final label, so each node
/// expands at most once: at most one `relax` call per edge and
/// O((m + s) log(m + s)) time for `m` edges and `s` seeds. A transfer that can
/// lower a label still reaches the same fixpoint but has no such bound.
///
/// # Panics
///
/// Panics when a seed node does not belong to `graph`.
///
/// # Examples
///
/// ```rust
/// use cfglib::{Graph, Direction, min_label_relaxation};
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
///     Direction::Forward,
///     |edge, distance| distance.checked_add(*edge.data()),
/// );
/// assert_eq!(distances, [Some(0), Some(2), Some(5)]);
/// ```
#[must_use]
pub fn min_label_relaxation<G, L>(
    graph: &G,
    seeds: impl IntoIterator<Item = (G::NodeId, L)>,
    direction: Direction,
    relax: impl FnMut(EdgeRef<'_, G::NodeId, G::EdgeId, G::EdgeData>, &L) -> Option<L>,
) -> Vec<Option<L>>
where
    G: EdgeView,
    L: Ord,
{
    by_axis!(direction, relax_along(graph, seeds, relax))
}

fn relax_along<G, L, A>(
    axis: A,
    graph: &G,
    seeds: impl IntoIterator<Item = (G::NodeId, L)>,
    mut relax: impl FnMut(EdgeRef<'_, G::NodeId, G::EdgeId, G::EdgeData>, &L) -> Option<L>,
) -> Vec<Option<L>>
where
    G: EdgeView,
    L: Ord,
    A: EdgeAdjacency,
{
    let mut labels: Vec<Option<L>> = (0..graph.node_bound()).map(|_| None).collect();
    // A min-heap of pending candidates, smallest label first and ties to the
    // smaller node index, so the expansion order is fixed by the labels alone.
    let mut worklist: BinaryHeap<Reverse<(L, usize)>> = seeds
        .into_iter()
        .map(|(node, label)| {
            assert_in_bound(graph, node, "seed node");
            Reverse((label, node.index()))
        })
        .collect();

    while let Some(Reverse((candidate, index))) = worklist.pop() {
        if labels[index]
            .as_ref()
            .is_some_and(|known| known <= &candidate)
        {
            continue;
        }
        labels[index] = Some(candidate);
        let label = labels[index]
            .as_ref()
            .expect("the candidate label was just stored");
        for edge_id in axis.edges(graph, G::NodeId::from_index(index)) {
            let edge = graph.edge(edge_id);
            if let Some(candidate) = relax(edge, label) {
                let far = axis.next(edge).index();
                // A candidate no smaller than the known label would be
                // discarded when popped; leaving it out keeps the heap small.
                if labels[far].as_ref().is_none_or(|known| &candidate < known) {
                    worklist.push(Reverse((candidate, far)));
                }
            }
        }
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeId;
    use crate::graph::store::{Graph, NodeId};

    #[derive(Clone, Copy)]
    enum Rule {
        Mint(u32),
        Carry,
        Block,
    }

    fn apply_rule(edge: EdgeRef<'_, NodeId, EdgeId, Rule>, label: u32) -> Option<u32> {
        match *edge.data() {
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

        let labels =
            min_label_relaxation(&graph, [(root, 0)], Direction::Forward, |edge, label| {
                apply_rule(edge, *label)
            });

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

        let labels =
            min_label_relaxation(&graph, [(sink, 4)], Direction::Backward, |edge, label| {
                apply_rule(edge, *label)
            });

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
            Direction::Forward,
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
            Direction::Forward,
            |_, label| Some(*label),
        );
    }

    /// `k` if/else diamonds whose costlier arm is added second, the edge order
    /// that made the former last-in-first-out worklist relabel the whole
    /// downstream of every arm: 2^(k+2) - 4 transfers.
    #[test]
    fn a_diamond_chain_calls_the_transfer_once_per_edge() {
        const DIAMONDS: u32 = 20;
        let mut graph = Graph::<(), u64>::new();
        let start = graph.add_node(());
        let mut spine = start;
        for i in 0..DIAMONDS {
            let cheap = graph.add_node(());
            let costly = graph.add_node(());
            let next = graph.add_node(());
            graph.add_edge(spine, cheap, 0);
            graph.add_edge(spine, costly, 1 << (DIAMONDS - i));
            graph.add_edge(cheap, next, 0);
            graph.add_edge(costly, next, 0);
            spine = next;
        }
        let mut calls = 0_usize;
        let labels = min_label_relaxation(
            &graph,
            [(start, 0_u64)],
            Direction::Forward,
            |edge, label| {
                calls += 1;
                label.checked_add(*edge.data())
            },
        );
        assert_eq!(labels[spine.index()], Some(0));
        assert!(calls <= graph.edge_count(), "{calls} transfers");
    }

    /// Nonnegative random weights on random graphs with cycles, in both
    /// directions, against Bellman-Ford: the same labels, and at most one
    /// transfer per edge.
    #[test]
    fn extensive_transfers_match_bellman_ford_and_expand_each_node_once() {
        let mut rng = crate::test_util::rng::Rng::new(70);
        for _ in 0..200 {
            let node_count = 1 + rng.below(12);
            let mut graph = Graph::<(), u64>::new();
            let nodes: Vec<_> = (0..node_count).map(|_| graph.add_node(())).collect();
            let mut edges = Vec::new();
            for _ in 0..rng.below(3 * node_count + 1) {
                let (source, target) = (rng.below(node_count), rng.below(node_count));
                let weight = rng.below(10) as u64;
                graph.add_edge(nodes[source], nodes[target], weight);
                edges.push((source, target, weight));
            }
            for direction in [Direction::Forward, Direction::Backward] {
                let seed = rng.below(node_count);
                let mut expected = alloc::vec![None; node_count];
                expected[seed] = Some(0_u64);
                for _ in 0..node_count {
                    for &(source, target, weight) in &edges {
                        let (near, far) = match direction {
                            Direction::Forward => (source, target),
                            Direction::Backward => (target, source),
                        };
                        if let Some(label) = expected[near] {
                            let candidate = label + weight;
                            if expected[far].is_none_or(|known| candidate < known) {
                                expected[far] = Some(candidate);
                            }
                        }
                    }
                }
                let mut calls = 0_usize;
                let labels = min_label_relaxation(
                    &graph,
                    [(nodes[seed], 0_u64)],
                    direction,
                    |edge, label| {
                        calls += 1;
                        label.checked_add(*edge.data())
                    },
                );
                assert_eq!(labels, expected);
                assert!(calls <= graph.edge_count());
            }
        }
    }
}
