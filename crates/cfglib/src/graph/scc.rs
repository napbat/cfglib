//! Strongly connected components for any [`GraphView`].
//!
//! [`SccDecomposition::compute`] partitions a graph into its components and
//! numbers them one of two ways, which is the whole reason to pick one
//! [`SccNumbering`]: leaves first, from Tarjan's one `O(V + E)` pass, or
//! sources first, from Kosaraju's two, for consumers that must process a
//! component before everything it reaches. Both walks are iterative, so they
//! do not consume the host call stack for deeply nested code graphs.
//!
//! [`SccDecomposition::condensation`] collapses the graph to its component
//! DAG, numbering the DAG's nodes exactly as the decomposition numbers its
//! components, so a sources-first pass keeps its numbering all the way
//! through to the graph it walks.
//!
//! Within a component, [`StrongCuts::compute`] finds the nodes and edges it
//! depends on to stay strongly connected: its strong articulation points and
//! strong bridges.
//!
//! A graph that is edited between queries keeps its components current in an
//! [`IncrementalScc`] instead of decomposing again. Seeded from the
//! sources-first decomposition, it repairs the partition and a sources-first
//! order of it when told of an added or removed edge: an edge that agrees with
//! the order costs O(1), one that runs against it searches only the
//! components between its ends, and a removal re-splits only the component
//! that held the edge. Each update returns an [`SccChange`] naming the
//! components it merged or split and the stretch of the order it rewrote.

extern crate alloc;
use alloc::collections::{BTreeSet, VecDeque};
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::graph::store::{Graph, NodeId};
use crate::graph::view::{DenseId, GraphView};
use crate::identity::{NO_INDEX, check_bound, slot, word};
pub use incremental::{IncrementalScc, SccChange};
pub use strong_cuts::{StrongCutScratch, StrongCuts};

mod incremental;
mod kosaraju;
mod strong_cuts;
mod tarjan;

/// A maximal set of mutually reachable nodes, in ascending order, borrowed
/// from the [`SccDecomposition`] that holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scc<'a, N = BlockId> {
    nodes: &'a [N],
}

impl<'a, N: Copy + Ord> Scc<'a, N> {
    /// Return the nodes of this component in ascending order.
    #[must_use]
    pub fn nodes(&self) -> &'a [N] {
        self.nodes
    }

    /// Return whether this component contains one node.
    #[must_use]
    pub fn is_singleton(&self) -> bool {
        self.nodes.len() == 1
    }

    /// Return whether `node` belongs to this component.
    #[must_use]
    pub fn contains(&self, node: N) -> bool {
        self.nodes.binary_search(&node).is_ok()
    }

    /// Return whether this component holds a cycle: it has several nodes, or
    /// its one node has an edge to itself. `graph` must be the graph the
    /// component was computed from.
    #[must_use]
    pub fn is_cyclic<G>(&self, graph: &G) -> bool
    where
        G: GraphView<NodeId = N>,
    {
        is_cyclic_component(graph, self.nodes)
    }
}

/// How [`SccDecomposition::compute`] numbers the components. The partition
/// is the same either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SccNumbering {
    /// Reverse topological order of the condensation: for every edge
    /// `u -> v`, `component_index(u) >= component_index(v)`, with equality
    /// exactly when `u` and `v` share a component.
    ///
    /// Tarjan's algorithm, one pass, and the right default. Beside the
    /// result, the walk keeps one word per node, which becomes the result's
    /// component table, plus its stacks: Pearce's space-efficient form of the
    /// algorithm, in place of the index, low-link and on-stack arrays.
    LeavesFirst,
    /// Topological order of the condensation: for every edge `u -> v`,
    /// `component_index(u) <= component_index(v)`, with equality exactly when
    /// `u` and `v` share a component. Walking
    /// [`components`](SccDecomposition::components) in index order visits
    /// every component only after every component that can reach it, the
    /// order a forward closure over the condensation wants.
    ///
    /// The exact sequence, not just its topological property, is part of the
    /// contract: it is Kosaraju's classic two-pass algorithm, deterministic
    /// in the graph's own orders.
    ///
    /// 1. An iterative depth-first walk over roots in node-id order, taking
    ///    each node's successors in adjacency order, recording nodes by
    ///    finish time.
    /// 2. A walk of the reverse graph over those roots in reverse finish
    ///    order, minting one component per root that is not yet assigned.
    ///
    /// A consumer whose work budget is keyed on the component sequence — a
    /// grammar `FIRST`-set fixpoint over a symbol-dependency graph, where
    /// processing a component before its dependents is what makes one pass
    /// enough — depends on that, so it is stated rather than left to the
    /// implementation.
    SourcesFirst,
}

/// Result of strongly connected component decomposition.
///
/// The partition is a property of the graph, but the **order** of
/// [`components`](Self::components) is the [`SccNumbering`] it was computed
/// with.
///
/// Every component's nodes sit in one array, component after component, each
/// run in ascending order, with the offsets where the runs begin. However
/// many components there are, a decomposition is three allocations.
///
/// The tables span the graph's whole node bound, so a slot that holds no node
/// (one a [`Graph`] removed) belongs to no component. Both numberings treat
/// it the same way: [`component_index`](Self::component_index) and
/// [`component`](Self::component) panic for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SccDecomposition<N = BlockId> {
    nodes: Vec<N>,
    /// `offsets[i]..offsets[i + 1]` is component `i` within `nodes`.
    offsets: Vec<u32>,
    /// Each slot's component, [`NO_INDEX`] while no component has claimed it
    /// and for good when the slot holds no node.
    component_of: Vec<u32>,
}

impl<N: DenseId> SccDecomposition<N> {
    /// Compute the strongly connected components of `graph`, numbered as
    /// `numbering` says.
    ///
    /// Both numberings cost `O(V + E)`; [`SccNumbering::LeavesFirst`] takes
    /// one pass and [`SccNumbering::SourcesFirst`] two.
    ///
    /// # Panics
    ///
    /// Panics when [`node_bound`](GraphView::node_bound) exceeds the range of
    /// a dense `u32` table.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Graph, SccDecomposition, SccNumbering};
    ///
    /// // entry -> (a <-> b) -> exit
    /// let mut graph = Graph::<&str, ()>::new();
    /// let entry = graph.add_node("entry");
    /// let a = graph.add_node("a");
    /// let b = graph.add_node("b");
    /// let exit = graph.add_node("exit");
    /// graph.add_edge(entry, a, ());
    /// graph.add_edge(a, b, ());
    /// graph.add_edge(b, a, ());
    /// graph.add_edge(b, exit, ());
    ///
    /// // Sources first: the entry's component is 0, the cycle 1, the exit 2.
    /// let sources_first = SccDecomposition::compute(&graph, SccNumbering::SourcesFirst);
    /// assert_eq!(sources_first.component_index(entry), 0);
    /// assert_eq!(sources_first.component_index(a), 1);
    /// assert_eq!(sources_first.component_index(b), 1);
    /// assert_eq!(sources_first.component_index(exit), 2);
    ///
    /// // The same partition, numbered the other way round.
    /// let leaves_first = SccDecomposition::compute(&graph, SccNumbering::LeavesFirst);
    /// assert_eq!(leaves_first.component_index(entry), 2);
    /// assert_eq!(leaves_first.component_index(exit), 0);
    /// ```
    #[must_use]
    pub fn compute<G>(graph: &G, numbering: SccNumbering) -> Self
    where
        G: GraphView<NodeId = N>,
    {
        match numbering {
            SccNumbering::LeavesFirst => tarjan::leaves_first(graph),
            SccNumbering::SourcesFirst => kosaraju::sources_first(graph),
        }
    }

    /// An empty decomposition with room for `node_bound` nodes, the most any
    /// graph that size can hold; no slot has a component until a node is
    /// placed.
    fn with_node_bound(node_bound: usize) -> Self {
        check_bound(
            node_bound,
            "node bound of a strongly connected component decomposition",
        );
        let mut offsets = Vec::with_capacity(node_bound + 1);
        offsets.push(0);
        Self {
            nodes: Vec::with_capacity(node_bound),
            offsets,
            component_of: vec![NO_INDEX; node_bound],
        }
    }

    /// Close the component whose nodes were appended to `nodes` since the
    /// last one, putting them in ascending order.
    fn close_component(&mut self) {
        let start = slot(
            *self
                .offsets
                .last()
                .expect("offsets always hold the leading zero"),
        );
        if self.nodes.len() - start > 1 {
            self.nodes[start..].sort_unstable();
        }
        self.offsets.push(word(self.nodes.len()));
    }

    /// The nodes of component `index`.
    fn run(&self, index: usize) -> &[N] {
        &self.nodes[slot(self.offsets[index])..slot(self.offsets[index + 1])]
    }

    /// Return the component index containing `node`.
    ///
    /// # Panics
    ///
    /// Panics when `node` is a slot that holds no node of the decomposed
    /// graph, such as one the graph removed, and when it is beyond the graph's
    /// node bound.
    #[must_use]
    pub fn component_index(&self, node: N) -> usize {
        let index = self.component_of[node.index()];
        assert!(
            index != NO_INDEX,
            "node {} is not a node of the decomposed graph",
            node.index()
        );
        slot(index)
    }

    /// Return the component containing `node`.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of [`component_index`](Self::component_index).
    #[must_use]
    pub fn component(&self, node: N) -> Scc<'_, N> {
        Scc {
            nodes: self.run(self.component_index(node)),
        }
    }

    /// Return each component's nodes, ascending, in component order.
    pub fn components(&self) -> impl ExactSizeIterator<Item = &[N]> + DoubleEndedIterator + Clone {
        self.offsets
            .windows(2)
            .map(|run| &self.nodes[slot(run[0])..slot(run[1])])
    }

    /// Return the number of components.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Return whether the decomposition contains no components.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return whether `graph` is acyclic.
    #[must_use]
    pub fn is_dag<G>(&self, graph: &G) -> bool
    where
        G: GraphView<NodeId = N>,
    {
        self.components()
            .all(|component| !is_cyclic_component(graph, component))
    }

    /// Return a shortest cycle through `node`, or `None` when `node` lies on
    /// none.
    ///
    /// The cycle is in successor order, starts at `node`, and does not repeat
    /// it at the end: each node has an edge to the next, and the last node has
    /// an edge back to `node`. Every cycle through `node` stays inside its
    /// component, so a node of a larger component always lies on one, and a
    /// node alone in its component only through an edge to itself, which is
    /// the one-node cycle `[node]`.
    ///
    /// The search is breadth-first from `node` along successors in adjacency
    /// order, confined to the component, and stops at the first node with an
    /// edge back to `node`. The cycle's length is therefore one more than the
    /// fewest hops from a successor of `node` back to it, and the same graph
    /// gives the same cycle. [`topological_sort_or_cycle`](crate::topological_sort_or_cycle)
    /// names some cycle of a graph without a decomposition.
    ///
    /// # Cost
    ///
    /// No table spans the graph: a node is found by its position in the
    /// component's ascending run, so a component of `n_c` nodes whose nodes
    /// have `m_c` successors in all costs O((`n_c` + `m_c`) log `n_c`),
    /// however large the graph.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of
    /// [`component_index`](Self::component_index). `graph` must be the graph
    /// this decomposition was computed from.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Graph, SccDecomposition, SccNumbering};
    ///
    /// // `a -> b -> c -> a` with a shortcut `b -> a`, and `c -> d` out of it.
    /// let mut graph = Graph::<(), ()>::new();
    /// let [a, b, c, d] = [(); 4].map(|()| graph.add_node(()));
    /// graph.add_edge(a, b, ());
    /// graph.add_edge(b, c, ());
    /// graph.add_edge(c, a, ());
    /// graph.add_edge(b, a, ());
    /// graph.add_edge(c, d, ());
    ///
    /// let components = SccDecomposition::compute(&graph, SccNumbering::SourcesFirst);
    /// assert_eq!(components.cycle_through(&graph, a), Some(vec![a, b]));
    /// assert_eq!(components.cycle_through(&graph, c), Some(vec![c, a, b]));
    /// // `d` is alone in its component and has no edge to itself.
    /// assert_eq!(components.cycle_through(&graph, d), None);
    /// ```
    #[must_use]
    pub fn cycle_through<G>(&self, graph: &G, node: N) -> Option<Vec<N>>
    where
        G: GraphView<NodeId = N>,
    {
        let component = self.component_index(node);
        let members = self.run(component);
        let position = |member: N| {
            members
                .binary_search(&member)
                .expect("a node of a component is in the component's run")
        };
        // Each member's breadth-first parent, by position in `members`, and
        // `NO_INDEX` for one the search has not reached. The root is its own
        // parent, which also marks it reached.
        let root = position(node);
        let mut parent = vec![NO_INDEX; members.len()];
        parent[root] = word(root);
        let mut queue = VecDeque::from([root]);
        while let Some(current) = queue.pop_front() {
            for successor in graph.successors(members[current]) {
                if successor == node {
                    let mut cycle = vec![members[current]];
                    let mut step = current;
                    while step != root {
                        step = slot(parent[step]);
                        cycle.push(members[step]);
                    }
                    cycle.reverse();
                    return Some(cycle);
                }
                if slot(self.component_of[successor.index()]) != component {
                    continue;
                }
                let next = position(successor);
                if parent[next] == NO_INDEX {
                    parent[next] = word(current);
                    queue.push_back(next);
                }
            }
        }
        None
    }

    /// Collapse `graph` to its component DAG, with `NodeId::from_index(i)`
    /// being component `i` of this decomposition.
    ///
    /// The numbering is this decomposition's, whichever [`SccNumbering`]
    /// produced it: node `i` of the result **is** the `i`th item of
    /// [`components`](Self::components), and
    /// [`component_index`](Self::component_index) indexes the result
    /// directly, so a sources-first decomposition gives a DAG whose ascending
    /// node order is topological.
    ///
    /// There is one edge per pair of distinct components connected in
    /// `graph`, deduplicated, so parallel edges and every additional pair of
    /// nodes joining the same two components collapse into one. Nodes are
    /// read in id order and each node's successors in adjacency order, and an
    /// edge is added the first time its pair is seen, so the result is
    /// deterministic. It is acyclic by construction.
    ///
    /// Node and edge payloads are `()` deliberately. The caller already owns
    /// the components, so carrying a copy of each would allocate per
    /// component to say what [`components`](Self::components) already says;
    /// and everything a fixpoint needs beyond the partition — the in-degree of
    /// a component, the dependents to re-queue when its fact changes — falls
    /// out of the returned graph's `predecessors` and `successors`.
    ///
    /// # Examples
    ///
    /// The shape this exists for: a grammar `FIRST`-set fixpoint whose work
    /// budget is one pass, which holds only if every component is processed
    /// after everything it depends on. Kahn's algorithm over the condensation,
    /// always taking the smallest ready component, reproduces exactly
    /// ascending component order — so the pass itself can simply walk
    /// `components` from 0 upwards.
    ///
    /// ```
    /// use cfglib::{Graph, NodeId, SccDecomposition, SccNumbering};
    ///
    /// // A symbol-dependency graph: `stmt` uses `expr`, `expr` and `term` are
    /// // mutually recursive, and both use `atom`.
    /// let mut grammar = Graph::<&str, ()>::new();
    /// let stmt = grammar.add_node("stmt");
    /// let expr = grammar.add_node("expr");
    /// let term = grammar.add_node("term");
    /// let atom = grammar.add_node("atom");
    /// grammar.add_edge(stmt, expr, ());
    /// grammar.add_edge(expr, term, ());
    /// grammar.add_edge(term, expr, ());
    /// grammar.add_edge(term, atom, ());
    ///
    /// let components = SccDecomposition::compute(&grammar, SccNumbering::SourcesFirst);
    /// let dag = components.condensation(&grammar);
    ///
    /// // The cycle is one component, and the numbering is the decomposition's.
    /// assert_eq!(dag.node_bound(), components.len());
    /// assert_eq!(components.component_index(expr), components.component_index(term));
    /// assert_eq!(dag[NodeId::from_index(components.component_index(stmt))], ());
    ///
    /// // In-degrees and dependents both fall out of the DAG.
    /// let mut pending: Vec<usize> = dag
    ///     .node_ids()
    ///     .map(|component| dag.predecessors(component).count())
    ///     .collect();
    /// let mut ready: Vec<usize> = (0..dag.node_bound()).filter(|&c| pending[c] == 0).collect();
    /// let mut processed = Vec::new();
    /// while !ready.is_empty() {
    ///     let position = ready.iter().enumerate().min_by_key(|&(_, c)| *c).map(|(p, _)| p);
    ///     let component = ready.remove(position.expect("a ready component"));
    ///     processed.push(component);
    ///     for dependent in dag.successors(NodeId::from_index(component)) {
    ///         pending[dependent.index()] -= 1;
    ///         if pending[dependent.index()] == 0 {
    ///             ready.push(dependent.index());
    ///         }
    ///     }
    /// }
    ///
    /// // Sources-first numbering means ascending order already IS that order.
    /// assert_eq!(processed, (0..dag.node_bound()).collect::<Vec<_>>());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when this decomposition was not computed from `graph`: it
    /// covers a fixed node space, and one that disagrees with this graph's
    /// node bound cannot index it.
    #[must_use]
    pub fn condensation<G>(&self, graph: &G) -> Graph<(), ()>
    where
        G: GraphView<NodeId = N>,
    {
        assert!(
            self.component_of.len() == graph.node_bound(),
            "the decomposition covers {} nodes but the graph has {}: condensation takes the graph the decomposition was computed from",
            self.component_of.len(),
            graph.node_bound()
        );

        let mut condensed = Graph::with_capacity(self.len(), self.len());
        for _ in 0..self.len() {
            condensed.add_node(());
        }
        self.for_each_component_edge(graph, |from, to| {
            condensed.add_edge(NodeId::from_index(from), NodeId::from_index(to), ());
        });
        condensed
    }

    /// Report every edge of the component DAG exactly once, as component
    /// index pairs.
    ///
    /// Nodes are read in id order and each node's successors in adjacency
    /// order, and a pair is reported the first time it is seen, so the edge
    /// sequence is deterministic and shared by every consumer of the
    /// component DAG rather than derived per consumer.
    pub(crate) fn for_each_component_edge<G>(
        &self,
        graph: &G,
        mut on_edge: impl FnMut(usize, usize),
    ) where
        G: GraphView<NodeId = N>,
    {
        let mut wired = BTreeSet::new();
        for node in graph.node_ids() {
            let from = self.component_index(node);
            for successor in graph.successors(node) {
                let to = self.component_index(successor);
                if from != to && wired.insert((from, to)) {
                    on_edge(from, to);
                }
            }
        }
    }
}

/// Whether the nodes of one component hold a cycle: there are several of
/// them, or the only one has an edge to itself.
fn is_cyclic_component<G: GraphView>(graph: &G, nodes: &[G::NodeId]) -> bool {
    match *nodes {
        [node] => graph.successors(node).any(|successor| successor == node),
        _ => true,
    }
}

#[cfg(test)]
mod tests;
