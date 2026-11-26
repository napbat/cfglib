//! `StrongCuts::compute_in` over one large strongly connected component whose
//! cuts are known by construction, alone and behind the decomposition that
//! finds the component.

use cfglib::{Graph, NodeId, SccDecomposition, SccNumbering, StrongCutScratch, StrongCuts};

use super::BenchmarkSuite;
use crate::harness::benchmark_case;

/// Segments in the ring of the strong-cut cases.
const STRONG_CUT_SEGMENTS: usize = 128;
/// Nodes in each segment of that ring.
const STRONG_CUT_SEGMENT_NODES: usize = 800;

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let (graph, expected) = segment_ring();
    let component: Vec<NodeId> = graph.node_ids().collect();
    let mut scratch = StrongCutScratch::default();
    benchmark_case!(
        suite,
        "api_strong_cuts_segment_ring",
        || graph_cuts(&graph, &mut scratch),
        |cuts: &StrongCuts<NodeId>| assert_eq!(cuts, &expected)
    );
    let mut scratch = StrongCutScratch::default();
    benchmark_case!(
        suite,
        "api_strong_cuts_of_segment_ring",
        || StrongCuts::compute_in(&mut scratch, &graph, &component),
        |cuts: &StrongCuts<NodeId>| assert_eq!(cuts, &expected)
    );
}

/// The cuts of every component of `graph`, merged in dense-index order: what
/// a consumer pays for the cuts of a graph it has not decomposed yet.
fn graph_cuts(graph: &Graph<(), ()>, scratch: &mut StrongCutScratch) -> StrongCuts<NodeId> {
    let mut cuts = StrongCuts::default();
    for component in SccDecomposition::compute(graph, SccNumbering::LeavesFirst).components() {
        let piece = StrongCuts::compute_in(scratch, graph, component);
        cuts.articulation_points.extend(piece.articulation_points);
        cuts.bridges.extend(piece.bridges);
    }
    cuts.articulation_points.sort_unstable();
    cuts.bridges
        .sort_unstable_by_key(|&(tail, head)| (tail.index(), head.index()));
    cuts
}

/// One strongly connected component of about 10⁵ nodes and 2·10⁵ edges, and
/// its strong cuts: a ring of segments, each a path with an edge each way
/// between neighbors and a shortcut from its first node to its last, joined
/// last to first by one edge.
///
/// The first and last node of each segment are its only ways in and out, so
/// they are the strong articulation points, and the joining edges are the
/// strong bridges. The shortcut and the backward edges spare the rest.
fn segment_ring() -> (Graph<(), ()>, StrongCuts<NodeId>) {
    let nodes = STRONG_CUT_SEGMENTS * STRONG_CUT_SEGMENT_NODES;
    let mut graph = Graph::with_capacity(nodes, 2 * nodes + STRONG_CUT_SEGMENTS);
    let ids: Vec<NodeId> = (0..nodes).map(|_| graph.add_node(())).collect();
    let mut cuts = StrongCuts::default();
    for segment in ids.chunks(STRONG_CUT_SEGMENT_NODES) {
        for pair in segment.windows(2) {
            graph.add_edge(pair[0], pair[1], ());
            graph.add_edge(pair[1], pair[0], ());
        }
        let (first, last) = (segment[0], segment[segment.len() - 1]);
        graph.add_edge(first, last, ());
        cuts.articulation_points.extend([first, last]);
    }
    for segment in 0..STRONG_CUT_SEGMENTS {
        let last = ids[(segment + 1) * STRONG_CUT_SEGMENT_NODES - 1];
        let next = ids[(segment + 1) % STRONG_CUT_SEGMENTS * STRONG_CUT_SEGMENT_NODES];
        graph.add_edge(last, next, ());
        cuts.bridges.push((last, next));
    }
    cuts.bridges
        .sort_unstable_by_key(|&(tail, head)| (tail.index(), head.index()));
    (graph, cuts)
}
