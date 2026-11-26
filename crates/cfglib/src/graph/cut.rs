//! Minimum source-to-sink edge and vertex cuts, and edge-disjoint paths
//! (Menger), over any view, with unit capacities.
//!
//! [`min_edge_cut`] finds the fewest edges whose removal leaves no path from
//! any source to any sink. [`min_vertex_cut`] finds the fewest nodes, other
//! than the sources and sinks, whose removal does the same.
//! [`edge_disjoint_paths`] finds as many paths from a source to a sink that
//! share no edge as a minimum edge cut has edges, which is Menger's theorem.
//! Several sources and sinks join a virtual super-source and super-sink.
//!
//! # Algorithm
//!
//! Each function copies the shape of the view once into a [`FlowNetwork`],
//! the residual representation of the weighted solvers, and runs Dinic's
//! algorithm: breadth-first levels, and blocking flows found over explicit
//! stacks in adjacency order. Every edge has capacity one. A vertex cut splits
//! each node that is neither a source nor a sink into an `in` and an `out` node
//! joined by an arc of capacity one, and gives the edges unbounded capacity,
//! so that only nodes are cut. After S. Even and R. E. Tarjan, *Network Flow
//! and Testing Graph Connectivity* (SIAM J. Comput. 1975), that costs
//! O(m·min(√m, n^(2/3))) for edges and O(√n·m) for vertices, after the O(n + m)
//! copy.
//!
//! With `limit: Some(k)`, the search stops once it has found `k + 1` paths.
//! Every phase finds at least one, so the cost is O(k·m), which answers "is
//! there a separator of at most `k`" without finding the whole flow.
//!
//! # The canonical cut
//!
//! A graph can have many minimum cuts. The one returned is the one closest to
//! the sources: its source side is the set of nodes that the sources still
//! reach in the residual network of the maximum flow. That set is the same for
//! every maximum flow, so the cut depends neither on which flow the search
//! found nor on the adjacency order. Parallel edges count separately, and a
//! self-loop is never cut.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::edge_view::EdgeView;
use super::network_flow::{ArcId, FlowNetwork};
use super::view::{DenseId, GraphView, assert_in_bound};
use crate::identity::{NO_INDEX, check_bound, slot, word};

/// The capacity of an arc that no cut takes.
const UNBOUNDED: i64 = i64::MAX;

/// A minimum edge cut between sources and sinks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeCut<E> {
    /// The number of edges in a minimum cut, or `limit + 1` when every cut has
    /// more than `limit` edges.
    pub value: usize,
    /// The edges of the canonical minimum cut: those from a node on the source
    /// side to a node off it, in the order the view yields the nodes and their
    /// outgoing edges. Empty when the limit is exceeded.
    pub edges: Vec<E>,
    /// Whether each node is on the source side of the canonical cut, indexed by
    /// dense node index. Empty when the limit is exceeded.
    pub source_side: Vec<bool>,
}

/// A minimum vertex cut between sources and sinks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexCut<N> {
    /// The number of nodes in a minimum cut, or `limit + 1` when every cut has
    /// more than `limit` nodes.
    pub value: usize,
    /// The nodes of the canonical minimum cut, in the order the view yields
    /// them. Empty when the limit is exceeded.
    pub vertices: Vec<N>,
    /// Whether each node is on the source side of the canonical cut, indexed by
    /// dense node index: the sources are, the cut nodes and the sinks are not.
    /// Every edge from a node on the source side leads to a node on it or to
    /// a cut node. Empty when the limit is exceeded.
    pub source_side: Vec<bool>,
}

/// Find a minimum set of edges whose removal leaves no path from any of
/// `sources` to any of `sinks`.
///
/// `None` when a source is also a sink, because no set of edges separates
/// them. See the [module documentation](self) for the cost, the `limit`, and
/// which minimum cut is returned.
///
/// # Panics
///
/// Panics when a source or a sink is not below the node bound of the view.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, min_edge_cut, min_vertex_cut};
///
/// // entry -> a, b; a -> join; b -> join; join -> exit.
/// let mut cfg = Cfg::<u32>::new();
/// let [a, b, join, exit] = [(); 4].map(|()| cfg.new_block());
/// cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
/// cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
/// cfg.add_edge(a, join, EdgeKind::Fallthrough);
/// cfg.add_edge(b, join, EdgeKind::Fallthrough);
/// let last = cfg.add_edge(join, exit, EdgeKind::Fallthrough);
///
/// let cut = min_edge_cut(&cfg, [cfg.entry()], [exit], None).expect("entry is not exit");
/// assert_eq!((cut.value, cut.edges), (1, vec![last]));
///
/// let cut = min_vertex_cut(&cfg, [cfg.entry()], [exit], None).expect("not adjacent");
/// assert_eq!(cut.vertices, [join]);
///
/// // Two edges separate the entry from the join: is there a cut of one?
/// let cut = min_edge_cut(&cfg, [cfg.entry()], [join], Some(1)).expect("entry is not join");
/// assert_eq!(cut.value, 2, "more than the limit of one");
/// ```
pub fn min_edge_cut<G: EdgeView>(
    graph: &G,
    sources: impl IntoIterator<Item = G::NodeId>,
    sinks: impl IntoIterator<Item = G::NodeId>,
    limit: Option<usize>,
) -> Option<EdgeCut<G::EdgeId>> {
    let flow = EdgeFlow::compute(graph, sources, sinks, limit)?;
    if limit.is_some_and(|limit| flow.value > limit) {
        return Some(EdgeCut {
            value: flow.value,
            edges: Vec::new(),
            source_side: Vec::new(),
        });
    }
    let mut side = flow.network.residual_reachable(flow.source);
    let edges = flow
        .edges
        .iter()
        .filter(|&&(_, tail, head)| side[tail] && !side[head])
        .map(|&(edge, ..)| edge)
        .collect();
    side.truncate(graph.node_bound());
    Some(EdgeCut {
        value: flow.value,
        edges,
        source_side: side,
    })
}

/// Find a minimum set of nodes, other than `sources` and `sinks`, whose
/// removal leaves no path from any source to any sink.
///
/// `None` when a source is also a sink or has an edge to a sink, because no
/// set of other nodes separates them. See the [module documentation](self) for
/// the cost, the `limit`, and which minimum cut is returned.
///
/// # Panics
///
/// Panics when a source or a sink is not below the node bound of the view.
pub fn min_vertex_cut<G: GraphView>(
    graph: &G,
    sources: impl IntoIterator<Item = G::NodeId>,
    sinks: impl IntoIterator<Item = G::NodeId>,
    limit: Option<usize>,
) -> Option<VertexCut<G::NodeId>> {
    let terminals = Terminals::new(graph, sources, sinks)?;
    let adjacent = terminals
        .sources
        .iter()
        .any(|&source| graph.successors(source).any(|next| terminals.is_sink(next)));
    if adjacent {
        return None;
    }
    // Node `v` is `in` at `2v` and `out` at `2v + 1`.
    let bound = graph.node_bound();
    let (mut network, source, sink) = network_with_terminals(2 * bound);
    for node in graph.node_ids() {
        if !terminals.is_source(node) && !terminals.is_sink(node) {
            network.add_arc(2 * node.index(), 2 * node.index() + 1, 1, 0);
        }
    }
    for &node in &terminals.sources {
        network.add_arc(source, 2 * node.index() + 1, UNBOUNDED, 0);
    }
    for &node in &terminals.sinks {
        network.add_arc(2 * node.index(), sink, UNBOUNDED, 0);
    }
    for node in graph.node_ids() {
        for next in graph.successors(node) {
            network.add_arc(2 * node.index() + 1, 2 * next.index(), UNBOUNDED, 0);
        }
    }
    let value = unit_count(network.max_flow_up_to(source, sink, flow_limit(limit)));
    if limit.is_some_and(|limit| value > limit) {
        return Some(VertexCut {
            value,
            vertices: Vec::new(),
            source_side: Vec::new(),
        });
    }
    let side = network.residual_reachable(source);
    let vertices = graph
        .node_ids()
        .filter(|node| side[2 * node.index()] && !side[2 * node.index() + 1])
        .collect();
    Some(VertexCut {
        value,
        vertices,
        source_side: (0..bound).map(|slot| side[2 * slot + 1]).collect(),
    })
}

/// Find as many paths from a source to a sink that share no edge as a
/// minimum edge cut has edges, each as its edges in order.
///
/// A path starts at a source and ends at a sink, visits no node twice, and
/// passes no other sink. `None` when a source is also a sink. The number of
/// paths is the [`value`](EdgeCut::value) of [`min_edge_cut`] with the same
/// inputs, so with `limit: Some(k)` it is at most `k + 1`.
///
/// # Panics
///
/// Panics when a source or a sink is not below the node bound of the view, or
/// when that bound exceeds `u32::MAX`.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, edge_disjoint_paths};
///
/// let mut cfg = Cfg::<u32>::new();
/// let [a, b, exit] = [(); 3].map(|()| cfg.new_block());
/// let to_a = cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
/// let to_b = cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
/// let from_a = cfg.add_edge(a, exit, EdgeKind::Fallthrough);
/// let from_b = cfg.add_edge(b, exit, EdgeKind::Fallthrough);
///
/// let paths = edge_disjoint_paths(&cfg, [cfg.entry()], [exit], None).expect("entry is not exit");
/// assert_eq!(paths, [vec![to_a, from_a], vec![to_b, from_b]]);
/// ```
pub fn edge_disjoint_paths<G: EdgeView>(
    graph: &G,
    sources: impl IntoIterator<Item = G::NodeId>,
    sinks: impl IntoIterator<Item = G::NodeId>,
    limit: Option<usize>,
) -> Option<Vec<Vec<G::EdgeId>>> {
    let terminals = Terminals::new(graph, sources, sinks)?;
    let flow = EdgeFlow::build(graph, &terminals, limit);
    Some(flow.paths(&terminals))
}

/// The distinct sources and sinks, and which node slots they hold.
struct Terminals<N> {
    sources: Vec<N>,
    sinks: Vec<N>,
    /// Whether each node slot is a source or a sink.
    source: Vec<bool>,
    sink: Vec<bool>,
}

impl<N: DenseId> Terminals<N> {
    /// `None` when a node is both a source and a sink.
    #[track_caller]
    fn new<G: GraphView<NodeId = N>>(
        graph: &G,
        sources: impl IntoIterator<Item = N>,
        sinks: impl IntoIterator<Item = N>,
    ) -> Option<Self> {
        let bound = graph.node_bound();
        let mut terminals = Self {
            sources: Vec::new(),
            sinks: Vec::new(),
            source: vec![false; bound],
            sink: vec![false; bound],
        };
        for node in sources {
            assert_in_bound(graph, node, "a source");
            if !core::mem::replace(&mut terminals.source[node.index()], true) {
                terminals.sources.push(node);
            }
        }
        for node in sinks {
            assert_in_bound(graph, node, "a sink");
            if !core::mem::replace(&mut terminals.sink[node.index()], true) {
                terminals.sinks.push(node);
            }
        }
        let shared = terminals
            .sources
            .iter()
            .any(|&node| terminals.is_sink(node));
        (!shared).then_some(terminals)
    }

    fn is_source(&self, node: N) -> bool {
        self.source[node.index()]
    }

    fn is_sink(&self, node: N) -> bool {
        self.sink[node.index()]
    }
}

/// A network of `nodes` nodes plus a super-source and a super-sink, which it
/// returns with it.
fn network_with_terminals(nodes: usize) -> (FlowNetwork, usize, usize) {
    let mut network = FlowNetwork::new();
    for _ in 0..nodes + 2 {
        network.add_node();
    }
    (network, nodes, nodes + 1)
}

/// The flow to stop at: one unit past the limit, or none.
fn flow_limit(limit: Option<usize>) -> i64 {
    limit.map_or(i64::MAX, |limit| {
        i64::try_from(limit).map_or(i64::MAX, |limit| limit.saturating_add(1))
    })
}

fn unit_count(value: i64) -> usize {
    usize::try_from(value).expect("a flow of unit paths counts in-memory arcs")
}

/// The maximum flow over the edges of a view, each of capacity one.
struct EdgeFlow<E> {
    network: FlowNetwork,
    source: usize,
    /// The arc from the super-source into each source, in order.
    source_arcs: Vec<(usize, ArcId)>,
    /// The flow from each sink slot into the super-sink.
    drain: Vec<i64>,
    /// Each edge with its tail and head slot. Edge `i` is arc
    /// `first_edge + i`, and the edges out of one node are consecutive.
    edges: Vec<(E, usize, usize)>,
    first_edge: usize,
    /// Where the edges out of each node slot start and end in `edges`.
    out: Vec<(usize, usize)>,
    value: usize,
}

impl<E: Copy> EdgeFlow<E> {
    fn compute<G: EdgeView<EdgeId = E>>(
        graph: &G,
        sources: impl IntoIterator<Item = G::NodeId>,
        sinks: impl IntoIterator<Item = G::NodeId>,
        limit: Option<usize>,
    ) -> Option<Self> {
        let terminals = Terminals::new(graph, sources, sinks)?;
        Some(Self::build(graph, &terminals, limit))
    }

    fn build<G: EdgeView<EdgeId = E>>(
        graph: &G,
        terminals: &Terminals<G::NodeId>,
        limit: Option<usize>,
    ) -> Self {
        let bound = graph.node_bound();
        let (mut network, source, sink) = network_with_terminals(bound);
        let source_arcs = terminals
            .sources
            .iter()
            .map(|&node| {
                (
                    node.index(),
                    network.add_arc(source, node.index(), UNBOUNDED, 0),
                )
            })
            .collect();
        let sink_arcs: Vec<ArcId> = terminals
            .sinks
            .iter()
            .map(|&node| network.add_arc(node.index(), sink, UNBOUNDED, 0))
            .collect();
        let first_edge = network.arc_count();
        let mut edges = Vec::new();
        let mut out = vec![(0, 0); bound];
        for node in graph.node_ids() {
            let start = edges.len();
            for edge in graph.outgoing(node) {
                let (tail, head) = (node.index(), graph.edge(edge).target().index());
                network.add_arc(tail, head, 1, 0);
                edges.push((edge, tail, head));
            }
            out[node.index()] = (start, edges.len());
        }
        let value = unit_count(network.max_flow_up_to(source, sink, flow_limit(limit)));
        let mut drain = vec![0; bound];
        for (&node, &arc) in terminals.sinks.iter().zip(&sink_arcs) {
            drain[node.index()] = network.flow(arc);
        }
        Self {
            network,
            source,
            source_arcs,
            drain,
            edges,
            first_edge,
            out,
            value,
        }
    }

    /// Split the flow into paths, one for each unit that leaves the
    /// super-source. A walk follows edges that carry flow and have not been
    /// walked, ends at the first sink it may drain into, and drops every cycle
    /// it closes, so its edges form a path. The path is then cut down to its
    /// part from the last source before its first sink to that sink.
    fn paths<N: DenseId>(&self, terminals: &Terminals<N>) -> Vec<Vec<E>> {
        let bound = self.out.len();
        let mut carries: Vec<bool> = (0..self.edges.len())
            .map(|index| {
                self.network
                    .flow(ArcId::from_position(self.first_edge + index))
                    > 0
            })
            .collect();
        let mut drain = self.drain.clone();
        let mut next: Vec<usize> = self.out.iter().map(|&(start, _)| start).collect();
        check_bound(bound, "node bound of a cut");
        let mut position = vec![NO_INDEX; bound];
        let (mut nodes, mut edges) = (Vec::new(), Vec::new());
        let mut paths = Vec::with_capacity(self.value);
        for &(start, arc) in &self.source_arcs {
            for _ in 0..self.network.flow(arc) {
                nodes.clear();
                edges.clear();
                let mut at = start;
                position[at] = 0;
                nodes.push(at);
                while drain[at] == 0 {
                    let index = (next[at]..self.out[at].1)
                        .find(|&index| carries[index])
                        .expect("flow that enters a node and does not drain there leaves it");
                    next[at] = index + 1;
                    carries[index] = false;
                    let head = self.edges[index].2;
                    if position[head] == NO_INDEX {
                        position[head] = word(nodes.len());
                        nodes.push(head);
                        edges.push(index);
                    } else {
                        let keep = slot(position[head]);
                        for &node in &nodes[keep + 1..] {
                            position[node] = NO_INDEX;
                        }
                        nodes.truncate(keep + 1);
                        edges.truncate(keep);
                    }
                    at = head;
                }
                drain[at] -= 1;
                for &node in &nodes {
                    position[node] = NO_INDEX;
                }
                let last = nodes[1..]
                    .iter()
                    .position(|&node| terminals.sink[node])
                    .map_or(edges.len(), |index| index + 1);
                let first = nodes[..last]
                    .iter()
                    .rposition(|&node| terminals.source[node])
                    .expect("a walk starts at a source");
                paths.push(
                    edges[first..last]
                        .iter()
                        .map(|&index| self.edges[index].0)
                        .collect(),
                );
            }
        }
        paths
    }
}

#[cfg(test)]
mod tests;
