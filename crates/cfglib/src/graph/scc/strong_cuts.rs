//! Strong articulation points and strong bridges, from dominators.
//!
//! A strong articulation point of a strongly connected component is a node
//! whose removal leaves the component's other nodes no longer strongly
//! connected, and a strong bridge is an edge whose removal does the same.
//! Italiano, Laura and Santaroni ("Finding strong bridges and strong
//! articulation points in linear time", TCS 447, 2012) read both off the
//! dominator trees of the component and of its reverse, from any one root
//! `s`:
//!
//! - a node other than `s` is a strong articulation point exactly when it
//!   has a child in either tree, because that child is cut off from `s`, or
//!   `s` from it, once the node is gone, and a component that loses no such
//!   pair stays strongly connected;
//! - an edge is a strong bridge exactly when it dominates its head in the
//!   component from `s`, or its reversal dominates its tail in the reverse;
//! - `s` itself is tested directly: the component without it is strongly
//!   connected when one other node reaches, and is reached by, all the rest.
//!
//! The component is read through a view that renumbers its nodes densely
//! from zero, so no subgraph is copied and the trees cost the component's
//! size, not the graph's. The root is the node of lowest dense index, so the
//! answer does not depend on the order of successor lists.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::graph::dominator::{DominatorScratch, DominatorTree};
use crate::graph::view::{DenseId, GraphView, Positions, Renumbered, Reversed, Rooted};

/// The strong articulation points and strong bridges of one strongly
/// connected component of a graph.
///
/// Both lists are sorted by the dense indices of their nodes, so the answer
/// does not depend on the order of successor lists. The cuts of a whole graph
/// are those of its components, which share no node and no edge.
///
/// # Examples
///
/// ```
/// use cfglib::{Graph, NodeId, StrongCuts};
///
/// // Two cycles that share `hub`: a -> hub -> a, and hub -> b -> c -> hub.
/// let mut graph = Graph::<&str, ()>::new();
/// let hub = graph.add_node("hub");
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let c = graph.add_node("c");
/// graph.add_edge(a, hub, ());
/// graph.add_edge(hub, a, ());
/// graph.add_edge(hub, b, ());
/// graph.add_edge(b, c, ());
/// graph.add_edge(c, hub, ());
///
/// let component: Vec<NodeId> = graph.node_ids().collect();
/// let cuts = StrongCuts::compute(&graph, &component);
/// assert_eq!(cuts.articulation_points, [hub, b, c]);
/// assert_eq!(cuts.bridges.len(), 5);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrongCuts<N = BlockId> {
    /// The strong articulation points, ascending by dense index.
    pub articulation_points: Vec<N>,
    /// The strong bridges as `(tail, head)`, ascending by the dense indices
    /// of the tail and then of the head.
    pub bridges: Vec<(N, N)>,
}

/// No articulation point and no bridge, the answer for a component of one
/// node.
impl<N> Default for StrongCuts<N> {
    fn default() -> Self {
        Self {
            articulation_points: Vec::new(),
            bridges: Vec::new(),
        }
    }
}

/// The buffers [`StrongCuts::compute_in`] fills, owned by the caller.
///
/// It starts empty and grows on entry to fit the graph it is handed, so one
/// instance serves every component and every graph: the dominator scratch is
/// reused by every tree, and the renumbering table and the search buffers
/// grow to the largest graph seen and are then reused.
#[derive(Debug, Clone, Default)]
pub struct StrongCutScratch {
    /// The working storage of every dominator tree.
    dominators: DominatorScratch,
    /// The position of each node in the component being read.
    positions: Positions,
    /// Whether each node of the component was reached by the search that
    /// tests the root.
    reached: Vec<bool>,
    /// The nodes that search has yet to expand.
    stack: Vec<usize>,
}

impl<N: DenseId> StrongCuts<N> {
    /// Compute the strong articulation points and strong bridges of one
    /// strongly connected component of `graph`, given as its nodes.
    ///
    /// The nodes are a slice of
    /// [`SccDecomposition::components`](crate::SccDecomposition::components),
    /// under either numbering. A component of one node has neither cut. A
    /// parallel edge is never a strong bridge, because its twin keeps the
    /// component connected, and a self-loop never is either.
    ///
    /// The cost is two dominator trees, of the component and of its reverse,
    /// and two extra searches, all over a view of that component alone: with
    /// the simple Lengauer-Tarjan construction, O(m log n) for a component of
    /// `n` nodes and `m` edges. [`compute_in`](Self::compute_in) reuses its
    /// buffers across components.
    ///
    /// # Panics
    ///
    /// Panics when `component` lists a node twice, when the subgraph it
    /// induces is not strongly connected, when a node has an index beyond the
    /// graph's node bound, and when that bound exceeds the range of a dense
    /// `u32` table.
    #[must_use]
    pub fn compute<G>(graph: &G, component: &[N]) -> Self
    where
        G: GraphView<NodeId = N>,
    {
        Self::compute_in(&mut StrongCutScratch::default(), graph, component)
    }

    /// [`compute`](Self::compute) over caller-owned buffers, which grow to
    /// fit `graph` and are reused by every later call.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of [`compute`](Self::compute).
    #[must_use]
    pub fn compute_in<G>(scratch: &mut StrongCutScratch, graph: &G, component: &[N]) -> Self
    where
        G: GraphView<NodeId = N>,
    {
        let mut cuts = Self::default();
        if component.len() < 2 {
            return cuts;
        }
        scratch.positions.number(graph.node_bound(), component);
        collect(graph, component, scratch, &mut cuts);
        scratch.positions.clear(component);
        sort(&mut cuts);
        cuts
    }
}

/// Put both lists in ascending dense-index order, each entry once.
///
/// An edge can be a bridge of a component and of its reverse at once, and a
/// node can have children in both trees, so both lists can hold repeats.
fn sort<N: DenseId>(cuts: &mut StrongCuts<N>) {
    cuts.articulation_points
        .sort_unstable_by_key(|node| node.index());
    cuts.articulation_points.dedup();
    cuts.bridges
        .sort_unstable_by_key(|&(tail, head)| (tail.index(), head.index()));
    cuts.bridges.dedup();
}

/// Append the cuts of `component`, of at least two nodes and numbered in
/// `scratch.positions`, to `cuts`.
fn collect<G: GraphView>(
    graph: &G,
    component: &[G::NodeId],
    scratch: &mut StrongCutScratch,
    cuts: &mut StrongCuts<G::NodeId>,
) {
    let view = Renumbered::new(graph, component, &scratch.positions);
    let root = (0..component.len())
        .min_by_key(|&position| component[position].index())
        .expect("a component of two nodes has a lowest one");
    let forward = DominatorTree::compute_in(&mut scratch.dominators, &Rooted::new(&view, root));
    let reverse = Reversed::new(&view);
    let backward = DominatorTree::compute_in(&mut scratch.dominators, &Rooted::new(&reverse, root));
    assert!(
        forward.preorder().len() == component.len() && backward.preorder().len() == component.len(),
        "the component is not strongly connected"
    );

    for node in 0..component.len() {
        for parent in [forward.idom(node), backward.idom(node)]
            .into_iter()
            .flatten()
        {
            if parent != root {
                cuts.articulation_points.push(component[parent]);
            }
        }
        if let Some(tail) = forward.dominating_in_edge(&view, node) {
            cuts.bridges.push((component[tail], component[node]));
        }
        if let Some(head) = backward.dominating_in_edge(&reverse, node) {
            cuts.bridges.push((component[node], component[head]));
        }
    }
    // The component without the root stays strongly connected exactly when
    // one other node reaches every node but the root, and is reached by every
    // one, along paths that avoid the root.
    let (reached, stack) = (&mut scratch.reached, &mut scratch.stack);
    if !reaches_all_without(&view, root, reached, stack)
        || !reaches_all_without(&reverse, root, reached, stack)
    {
        cuts.articulation_points.push(component[root]);
    }
}

/// Whether one node of `view` other than `removed` reaches every other node
/// but `removed` along paths that avoid it.
///
/// The answer is a count, so the order is free and a stack keeps only the
/// frontier. On the breadth-first core, whose queue keeps every node it
/// reaches, over epoch marks, `api_strong_cuts_component_segment_ring` measured 13%
/// to 31% slower than this loop.
fn reaches_all_without<V: GraphView<NodeId = usize>>(
    view: &V,
    removed: usize,
    reached: &mut Vec<bool>,
    stack: &mut Vec<usize>,
) -> bool {
    let start = usize::from(removed == 0);
    reached.clear();
    reached.resize(view.node_bound(), false);
    reached[removed] = true;
    reached[start] = true;
    stack.clear();
    stack.push(start);
    let mut count = 2;
    while let Some(node) = stack.pop() {
        for next in view.successors(node) {
            if !reached[next] {
                reached[next] = true;
                count += 1;
                stack.push(next);
            }
        }
    }
    count == view.node_bound()
}

#[cfg(test)]
mod tests;
