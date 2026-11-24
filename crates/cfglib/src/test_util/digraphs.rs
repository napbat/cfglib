//! Small directed graphs by node index, for exhaustive tests of analyses
//! that take any [`GraphView`](crate::graph::view::GraphView).

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::graph::store::{Graph, NodeId};

/// The graph over `count` nodes with `edges`, by node index.
pub(crate) fn graph_of(count: usize, edges: &[(usize, usize)]) -> Graph<(), ()> {
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..count).map(|_| graph.add_node(())).collect();
    for &(from, to) in edges {
        graph.add_edge(nodes[from], nodes[to], ());
    }
    graph
}

/// Visit every graph over `count` nodes in which each ordered pair, self-loops
/// included, is an edge `0..=copies` times.
pub(crate) fn for_each_multigraph(
    count: usize,
    copies: usize,
    mut visit: impl FnMut(&[(usize, usize)]),
) {
    let mut multiplicity = vec![0; count * count];
    loop {
        let edges: Vec<(usize, usize)> = multiplicity
            .iter()
            .enumerate()
            .flat_map(|(pair, &times)| core::iter::repeat_n((pair / count, pair % count), times))
            .collect();
        visit(&edges);
        let Some(digit) = multiplicity.iter().position(|&times| times < copies) else {
            return;
        };
        multiplicity[..digit].fill(0);
        multiplicity[digit] += 1;
    }
}
