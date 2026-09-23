//! Generic graph coloring and interference-graph construction.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::cfg::Cfg;
use crate::dataflow::liveness::Liveness;
use crate::dataflow::{InstrInfo, VariableId};
use crate::graph::store::Graph;
use crate::graph::view::GraphView;

fn connect<V: VariableId>(adjacency: &mut BTreeMap<V, BTreeSet<V>>, left: &V, right: &V) {
    adjacency.entry(left.clone()).or_default();
    adjacency.entry(right.clone()).or_default();
    if left == right {
        return;
    }
    adjacency
        .entry(left.clone())
        .or_default()
        .insert(right.clone());
    adjacency
        .entry(right.clone())
        .or_default()
        .insert(left.clone());
}

fn connect_clique<V: VariableId>(
    adjacency: &mut BTreeMap<V, BTreeSet<V>>,
    variables: &BTreeSet<V>,
) {
    for variable in variables {
        adjacency.entry(variable.clone()).or_default();
    }
    for (index, variable) in variables.iter().enumerate() {
        for other in variables.iter().skip(index + 1) {
            connect(adjacency, variable, other);
        }
    }
}

/// Build a symmetric interference relation in the common graph storage.
///
/// Variable payloads are nodes and `()` edges connect variables that are live
/// together. Each undirected relation is stored once; [`color_graph`] treats
/// both incoming and outgoing adjacency as conflicts.
#[must_use]
pub fn interference_graph<I, V, E>(cfg: &Cfg<I, E>, live: &Liveness<V>) -> Graph<V, ()>
where
    I: InstrInfo<Variable = V>,
    V: VariableId,
{
    let mut adjacency: BTreeMap<V, BTreeSet<V>> = BTreeMap::new();
    for block_id in cfg.block_ids() {
        let block = cfg.block(block_id);
        let mut active = live.live_out(block_id).clone();
        connect_clique(&mut adjacency, &active);

        for instruction in block.instructions().iter().rev() {
            let definitions: BTreeSet<_> = instruction.defs().iter().cloned().collect();
            connect_clique(&mut adjacency, &definitions);
            for definition in &definitions {
                for variable in &active {
                    connect(&mut adjacency, definition, variable);
                }
            }
            for definition in &definitions {
                active.remove(definition);
            }
            active.extend(instruction.uses().iter().cloned());
            connect_clique(&mut adjacency, &active);
        }
    }

    let mut graph = Graph::with_capacity(adjacency.len(), adjacency.len());
    let nodes: BTreeMap<V, _> = adjacency
        .keys()
        .cloned()
        .map(|variable| {
            let node = graph.add_node(variable.clone());
            (variable, node)
        })
        .collect();

    for (variable, neighbors) in adjacency {
        for neighbor in neighbors {
            if variable < neighbor {
                graph.add_edge(nodes[&variable], nodes[&neighbor], ());
            }
        }
    }
    graph
}

/// Result of greedy graph coloring, keyed by the graph's native node identity.
#[derive(Debug, Clone)]
pub struct ColorAssignment<N> {
    /// Assigned color for each node.
    pub assignment: BTreeMap<N, usize>,
    /// Number of colors used.
    pub num_colors: usize,
}

/// Greedily color a directed graph as an undirected conflict relation.
///
/// Edge orientation is ignored: both successors and predecessors are treated
/// as neighbors. The function therefore works with an interference graph
/// emitted by [`interference_graph`] and with consumer-owned graph views.
#[must_use]
pub fn color_graph<G: GraphView>(graph: &G) -> ColorAssignment<G::NodeId> {
    let mut neighbors = BTreeMap::new();
    for node in graph.node_ids() {
        let adjacent: BTreeSet<_> = graph
            .successors(node)
            .chain(graph.predecessors(node))
            .filter(|&neighbor| neighbor != node)
            .collect();
        neighbors.insert(node, adjacent);
    }

    let mut nodes: Vec<_> = graph.node_ids().collect();
    nodes.sort_by_key(|node| neighbors[node].len());

    let mut assignment = BTreeMap::new();
    let mut num_colors = 0;
    for node in nodes {
        let used_colors: BTreeSet<_> = neighbors[&node]
            .iter()
            .filter_map(|neighbor| assignment.get(neighbor).copied())
            .collect();
        let mut color = 0;
        for used_color in used_colors {
            if used_color == color {
                color += 1;
            } else if used_color > color {
                break;
            }
        }
        assignment.insert(node, color);
        num_colors = num_colors.max(color + 1);
    }

    ColorAssignment {
        assignment,
        num_colors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataflow::liveness::Liveness;
    use crate::test_util::df_op;

    #[test]
    fn triangle_needs_three_colors() {
        let mut graph = Graph::<(), ()>::new();
        let first = graph.add_node(());
        let second = graph.add_node(());
        let third = graph.add_node(());
        graph.add_edge(first, second, ());
        graph.add_edge(first, third, ());
        graph.add_edge(second, third, ());

        let result = color_graph(&graph);
        assert_eq!(result.num_colors, 3);
        assert_ne!(result.assignment[&first], result.assignment[&second]);
        assert_ne!(result.assignment[&first], result.assignment[&third]);
        assert_ne!(result.assignment[&second], result.assignment[&third]);
    }

    #[test]
    fn independent_nodes_share_one_color() {
        let mut graph = Graph::<(), ()>::new();
        graph.add_node(());
        graph.add_node(());
        assert_eq!(color_graph(&graph).num_colors, 1);
    }

    #[test]
    fn empty_graph_uses_no_colors() {
        let graph = Graph::<(), ()>::new();
        assert_eq!(color_graph(&graph).num_colors, 0);
    }

    #[test]
    fn interference_builder_returns_generic_graph_with_variable_payloads() {
        let mut cfg = Cfg::<_, &str>::with_edge_payload();
        cfg.block_mut(cfg.entry())
            .push(df_op("sum", "add", 2, &[0, 1]));
        let live = Liveness::compute(&cfg);

        let graph = interference_graph(&cfg, &live);
        let left = graph
            .node_ids()
            .find(|&node| *graph.node(node) == 0)
            .unwrap();
        let right = graph
            .node_ids()
            .find(|&node| *graph.node(node) == 1)
            .unwrap();
        let result = color_graph(&graph);

        assert_eq!(graph.node_count(), 3);
        assert!(
            graph.successors(left).any(|node| node == right)
                || graph.predecessors(left).any(|node| node == right)
        );
        assert_ne!(result.assignment[&left], result.assignment[&right]);
    }
}
