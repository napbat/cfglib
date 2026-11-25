//! The graph that cycle equivalence and the program structure tree read: the
//! blocks the root reaches, closed into one strongly connected graph through a
//! virtual exit.
//!
//! Every reachable block without successors gets an edge to the virtual exit
//! `X*`, as does every block that the post-dominator cut joins to it, so every
//! reachable block reaches `X*`. One more edge, from `X*` to the root, closes
//! the graph. The edge into the root plays the part of the edge out of a
//! distinguished start node: it is the first edge of every path from the root,
//! and the last edge of no path inside the graph.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::{ABSENT, SeseEdge, slot, word};
use crate::graph::dominator::CutBuffers;
use crate::graph::edge_view::EdgeView;
use crate::graph::view::{DenseId, RootedView};

/// The closed graph as edge slots and incident lists over dense indices.
///
/// Nodes are the reachable blocks by dense index and the virtual exit at index
/// `bound`. Edge slots are the edges of the view by dense edge index, then the
/// virtual edge into the root at `edge_bound`, then one virtual edge to the
/// virtual exit per outlet, in ascending block order. A slot of the view whose
/// source the root misses, or that names no live edge, is in no list and has
/// no endpoints.
pub(super) struct ClosedGraph<N> {
    /// The node bound of the view, which is also the index of the virtual exit.
    pub(super) bound: usize,
    /// The edge bound of the view, which is also the slot of the virtual edge
    /// into the root.
    pub(super) edge_bound: usize,
    /// The blocks with an edge to the virtual exit: the exits and the cut
    /// roots, in ascending dense index.
    pub(super) outlets: Vec<N>,
    /// The source of each slot, [`ABSENT`] for a slot outside the closed graph.
    source: Vec<u32>,
    /// The target of each slot, [`ABSENT`] for a slot outside the closed graph.
    target: Vec<u32>,
    /// Where each node's incident slots start in `incident`, with one more
    /// entry for the end.
    starts: Vec<u32>,
    /// Where each node's outgoing slots end and its incoming slots start.
    out_ends: Vec<u32>,
    /// The slots incident to each node: its outgoing slots in adjacency order,
    /// then its incoming ones. A self-loop is in both parts.
    incident: Vec<u32>,
}

impl<N: DenseId> ClosedGraph<N> {
    /// Close the part of `graph` that its root reaches.
    ///
    /// # Panics
    ///
    /// Panics when the root is not a node of `graph`, or when the nodes or the
    /// edge slots of the closed graph exceed the range of a dense `u32` table.
    pub(super) fn new<G>(graph: &G) -> Self
    where
        G: RootedView<NodeId = N> + EdgeView,
    {
        let bound = graph.node_bound();
        let edge_bound = graph.edge_bound();
        let root = graph.root();
        assert!(
            root.index() < bound,
            "program structure root is not a node of the graph"
        );
        let mut cut = CutBuffers::default();
        let structure = cut.analyze(graph, root);
        let reachable = &cut.reachable;
        let outlets = merge_ascending(structure.exits, structure.cut_roots);
        let slots = edge_bound + 1 + outlets.len();
        assert!(
            bound < slot(ABSENT) && slots < slot(ABSENT),
            "the closed graph exceeds the dense u32 range of a program structure tree"
        );

        let mut source = vec![ABSENT; slots];
        let mut target = vec![ABSENT; slots];
        let mut out_degree = vec![0_u32; bound + 1];
        let mut in_degree = vec![0_u32; bound + 1];
        let mut link = |edge: usize, from: usize, to: usize| {
            source[edge] = word(from);
            target[edge] = word(to);
            out_degree[from] += 1;
            in_degree[to] += 1;
        };
        for block in (0..bound).filter(|&block| reachable[block]) {
            for edge in graph.outgoing(N::from_index(block)) {
                link(edge.index(), block, graph.edge(edge).target().index());
            }
        }
        link(edge_bound, bound, root.index());
        for (offset, outlet) in outlets.iter().enumerate() {
            link(edge_bound + 1 + offset, outlet.index(), bound);
        }

        let mut starts = Vec::with_capacity(bound + 2);
        let mut out_ends = Vec::with_capacity(bound + 1);
        let mut end = 0_u32;
        starts.push(end);
        for (outgoing, incoming) in out_degree.iter().zip(&in_degree) {
            out_ends.push(end + outgoing);
            end += outgoing + incoming;
            starts.push(end);
        }

        // The outgoing part fills in adjacency order, block by block; every
        // incoming part fills in the same pass, in the order its sources come.
        let mut out_cursor = starts[..=bound].to_vec();
        let mut in_cursor = out_ends.clone();
        let mut incident = vec![0_u32; slot(end)];
        let mut place = |edge: usize| {
            let (from, to) = (slot(source[edge]), slot(target[edge]));
            incident[slot(out_cursor[from])] = word(edge);
            out_cursor[from] += 1;
            incident[slot(in_cursor[to])] = word(edge);
            in_cursor[to] += 1;
        };
        for block in (0..bound).filter(|&block| reachable[block]) {
            for edge in graph.outgoing(N::from_index(block)) {
                place(edge.index());
            }
        }
        for edge in edge_bound..slots {
            place(edge);
        }

        Self {
            bound,
            edge_bound,
            outlets,
            source,
            target,
            starts,
            out_ends,
            incident,
        }
    }

    /// The number of nodes, the virtual exit included.
    pub(super) fn node_count(&self) -> usize {
        self.bound + 1
    }

    /// The number of edge slots.
    pub(super) fn slot_count(&self) -> usize {
        self.source.len()
    }

    /// The target of the slot `edge` of the closed graph.
    pub(super) fn target(&self, edge: usize) -> usize {
        slot(self.target[edge])
    }

    /// The endpoint of the slot `edge` that is not `node`, `node` itself for
    /// a self-loop.
    pub(super) fn other(&self, edge: usize, node: usize) -> usize {
        let source = slot(self.source[edge]);
        if source == node {
            slot(self.target[edge])
        } else {
            source
        }
    }

    /// The slots leaving `node`, in adjacency order.
    pub(super) fn outgoing(&self, node: usize) -> &[u32] {
        &self.incident[slot(self.starts[node])..slot(self.out_ends[node])]
    }

    /// The slots incident to `node`, leaving it first.
    pub(super) fn incident(&self, node: usize) -> &[u32] {
        &self.incident[slot(self.starts[node])..slot(self.starts[node + 1])]
    }
}

/// The edge a slot names.
pub(super) fn slot_edge<N: DenseId, E: DenseId>(
    edge_bound: usize,
    outlets: &[N],
    edge: usize,
) -> SeseEdge<N, E> {
    match edge.checked_sub(edge_bound) {
        None => SeseEdge::Edge(E::from_index(edge)),
        Some(0) => SeseEdge::Entry,
        Some(offset) => SeseEdge::Exit(outlets[offset - 1]),
    }
}

/// The slot an edge names, `None` for a virtual exit edge from a block that
/// has none or for an edge beyond the edge bound.
pub(super) fn edge_slot<N: DenseId, E: DenseId>(
    edge_bound: usize,
    outlets: &[N],
    edge: SeseEdge<N, E>,
) -> Option<usize> {
    match edge {
        SeseEdge::Edge(edge) => (edge.index() < edge_bound).then_some(edge.index()),
        SeseEdge::Entry => Some(edge_bound),
        SeseEdge::Exit(block) => outlets
            .binary_search_by_key(&block.index(), |outlet| outlet.index())
            .ok()
            .map(|offset| edge_bound + 1 + offset),
    }
}

/// Two ascending lists of distinct blocks merged into one.
///
/// An exit has no successor and a cut root has one, so no block is in both.
fn merge_ascending<N: DenseId>(first: Vec<N>, second: Vec<N>) -> Vec<N> {
    if second.is_empty() {
        return first;
    }
    let mut merged = Vec::with_capacity(first.len() + second.len());
    let mut second = second.into_iter().peekable();
    for block in first {
        while let Some(earlier) = second.next_if(|other| other.index() < block.index()) {
            merged.push(earlier);
        }
        merged.push(block);
    }
    merged.extend(second);
    merged
}
