use cfglib::{
    EdgeCut, Graph, NodeId, VertexCut, edge_disjoint_paths, min_edge_cut, min_vertex_cut,
};

use super::BenchmarkSuite;
use crate::harness::benchmark_case;

/// Nodes in one layer of the layered graph.
const WIDTH: usize = 8;
/// Layers of the layered graph.
const LAYERS: usize = 128;

/// `LAYERS` layers of `WIDTH` nodes, each node leading to the node below it
/// and to the one below and to the right, wrapping around: every layer is a
/// vertex cut of `WIDTH`, and every gap between layers an edge cut of
/// `2 * WIDTH`. The sources are the first layer, the sinks the last.
fn layered_graph() -> (Graph<(), ()>, Vec<NodeId>, Vec<NodeId>) {
    let mut graph = Graph::with_capacity(WIDTH * LAYERS, 2 * WIDTH * (LAYERS - 1));
    let nodes: Vec<NodeId> = (0..WIDTH * LAYERS).map(|_| graph.add_node(())).collect();
    for layer in 0..LAYERS - 1 {
        for column in 0..WIDTH {
            let from = nodes[layer * WIDTH + column];
            graph.add_edge(from, nodes[(layer + 1) * WIDTH + column], ());
            graph.add_edge(from, nodes[(layer + 1) * WIDTH + (column + 1) % WIDTH], ());
        }
    }
    let sources = nodes[..WIDTH].to_vec();
    let sinks = nodes[(LAYERS - 1) * WIDTH..].to_vec();
    (graph, sources, sinks)
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let (graph, sources, sinks) = layered_graph();
    let (sources, sinks) = (sources.as_slice(), sinks.as_slice());
    benchmark_case!(
        suite,
        "api_min_edge_cut",
        covers[min_edge_cut],
        || min_edge_cut(&graph, sources.iter().copied(), sinks.iter().copied(), None),
        |cut: &Option<EdgeCut<_>>| {
            let cut = cut.as_ref().expect("the sources are not sinks");
            assert_eq!((cut.value, cut.edges.len()), (2 * WIDTH, 2 * WIDTH));
        }
    );
    benchmark_case!(
        suite,
        "api_min_edge_cut_limit_3",
        || min_edge_cut(
            &graph,
            sources.iter().copied(),
            sinks.iter().copied(),
            Some(3)
        ),
        |cut: &Option<EdgeCut<_>>| {
            assert_eq!(cut.as_ref().map(|cut| cut.value), Some(4));
        }
    );
    benchmark_case!(
        suite,
        "api_min_vertex_cut",
        covers[min_vertex_cut],
        || min_vertex_cut(&graph, sources.iter().copied(), sinks.iter().copied(), None),
        |cut: &Option<VertexCut<NodeId>>| {
            let cut = cut.as_ref().expect("no source is next to a sink");
            assert_eq!((cut.value, cut.vertices.len()), (WIDTH, WIDTH));
        }
    );
    benchmark_case!(
        suite,
        "api_edge_disjoint_paths",
        covers[edge_disjoint_paths],
        || edge_disjoint_paths(&graph, sources.iter().copied(), sinks.iter().copied(), None),
        |paths: &Option<Vec<Vec<_>>>| {
            let paths = paths.as_ref().expect("the sources are not sinks");
            assert_eq!(paths.len(), 2 * WIDTH);
            assert!(paths.iter().all(|path| path.len() == LAYERS - 1));
        }
    );
}
