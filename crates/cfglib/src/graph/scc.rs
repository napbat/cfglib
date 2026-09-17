//! Strongly connected components for any [`GraphView`].
//!
//! Both implementations are iterative — they do not consume the host call
//! stack for deeply nested code graphs — and compute the same partition. They
//! differ in how they **number** it, which is the whole reason to pick one:
//! [`tarjan_scc`] takes one `O(V + E)` pass and numbers leaves first, while
//! [`kosaraju_scc`] takes two and numbers sources first, for consumers that
//! must process a component before everything it reaches.
//!
//! The component DAG comes in the same two shapes. [`condensation`] computes
//! its own Tarjan decomposition and carries each [`Scc`] as a node payload;
//! [`condensation_of`] takes the decomposition the consumer already holds and
//! numbers the DAG's nodes exactly as that decomposition numbers its
//! components, so a sources-first pass keeps its numbering all the way through
//! to the graph it walks.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use crate::graph::store::{Graph, NodeId};
use crate::graph::view::{DenseId, GraphView};

/// A maximal set of mutually reachable nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scc<N> {
    /// Nodes in this component.
    pub nodes: BTreeSet<N>,
}

impl<N: Copy + Ord> Scc<N> {
    /// Return whether this component contains one node.
    #[must_use]
    pub fn is_singleton(&self) -> bool {
        self.nodes.len() == 1
    }

    /// Return whether `node` belongs to this component.
    #[must_use]
    pub fn contains(&self, node: N) -> bool {
        self.nodes.contains(&node)
    }
}

/// Result of strongly connected component decomposition.
///
/// The partition is a property of the graph, but the **order** of
/// [`components`](Self::components) is a property of the algorithm that
/// produced it: [`tarjan_scc`] numbers them in reverse topological order
/// (leaves first), [`kosaraju_scc`] in topological order (sources first).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SccDecomposition<N> {
    /// The components, ordered by the producing algorithm's numbering
    /// contract — see [`tarjan_scc`] and [`kosaraju_scc`].
    pub components: Vec<Scc<N>>,
    component_of: Vec<usize>,
}

impl<N: DenseId> SccDecomposition<N> {
    /// Return the component index containing `node`.
    #[must_use]
    pub fn component_index(&self, node: N) -> usize {
        self.component_of[node.index()]
    }

    /// Return the component containing `node`.
    #[must_use]
    pub fn component(&self, node: N) -> &Scc<N> {
        &self.components[self.component_index(node)]
    }

    /// Return the number of components.
    #[must_use]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Return whether the decomposition contains no components.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Return whether `graph` is acyclic.
    #[must_use]
    pub fn is_dag<G>(&self, graph: &G) -> bool
    where
        G: GraphView<NodeId = N>,
    {
        self.components.iter().all(|component| {
            if !component.is_singleton() {
                return false;
            }
            let Some(&node) = component.nodes.iter().next() else {
                return false;
            };
            !graph.successors(node).any(|successor| successor == node)
        })
    }
}

/// One frame of the explicit depth-first stack both decompositions walk on.
///
/// A frame's successors live in the walk's single arena rather than in a `Vec`
/// of its own: frames are pushed and popped strictly last in, first out, so
/// the frame on top of the stack always owns the arena's tail and a pop
/// truncates the arena back to where that frame's successors began. One
/// allocation for the whole decomposition, instead of one per node it enters.
///
/// The **numbering** each algorithm documents is untouched by construction:
/// the frame is handed the same successors, in the same adjacency order, read
/// left to right by the same advancing cursor. Only where they are stored
/// changes.
struct SccFrame<N> {
    node: N,
    /// Where this frame's successors start in the arena; the pop truncates to
    /// it.
    start: usize,
    /// The next successor to examine, as an arena index. The frame's
    /// successors are exhausted when it reaches the arena's length, since the
    /// top frame's region runs to the end.
    cursor: usize,
}

/// Push the frame for `node`, taking its successors onto the arena's tail.
fn enter<G: GraphView>(
    graph: &G,
    node: G::NodeId,
    arena: &mut Vec<G::NodeId>,
    calls: &mut Vec<SccFrame<G::NodeId>>,
) {
    let start = arena.len();
    arena.extend(graph.successors(node));
    calls.push(SccFrame {
        node,
        start,
        cursor: start,
    });
}

/// Report every edge of the component DAG exactly once, as index pairs.
///
/// Nodes are read in id order and each node's successors in adjacency order,
/// and a pair is reported the first time it is seen, so the edge sequence is
/// deterministic and shared by both condensations rather than derived twice.
fn for_each_component_edge<G: GraphView>(
    graph: &G,
    components: &SccDecomposition<G::NodeId>,
    mut on_edge: impl FnMut(usize, usize),
) {
    let mut wired = BTreeSet::new();
    for node in graph.node_ids() {
        let from = components.component_index(node);
        for successor in graph.successors(node) {
            let to = components.component_index(successor);
            if from != to && wired.insert((from, to)) {
                on_edge(from, to);
            }
        }
    }
}

/// Collapse a graph to its component DAG.
///
/// One node per strongly connected component, carrying that component's
/// [`Scc`], in the decomposition's reverse topological order (leaves
/// first); one edge per pair of distinct components connected in the
/// source graph (deduplicated). The result is acyclic by construction, so
/// topological processing, condensed traces, and cycle summaries follow
/// directly.
///
/// The decomposition is [`tarjan_scc`]'s, computed here. A consumer that
/// already holds one — in particular one numbered the other way round by
/// [`kosaraju_scc`] — wants [`condensation_of`] instead, which preserves the
/// numbering it is given.
#[must_use]
pub fn condensation<G: GraphView>(graph: &G) -> Graph<Scc<G::NodeId>, ()> {
    let components = tarjan_scc(graph);
    let mut condensed = Graph::with_capacity(components.len(), components.len());
    let ids: Vec<NodeId> = components
        .components
        .iter()
        .map(|component| condensed.add_node(component.clone()))
        .collect();

    for_each_component_edge(graph, &components, |from, to| {
        condensed.add_edge(ids[from], ids[to], ());
    });
    condensed
}

/// Collapse a graph to the component DAG of a decomposition **you** computed,
/// with `NodeId::from_index(i)` being component `i` of that decomposition.
///
/// [`condensation`] computes its own [`tarjan_scc`] and is therefore
/// leaves-first, which inverts the one numbering a budgeted forward pass wants
/// — [`kosaraju_scc`]'s sources-first order, where a component is numbered
/// only after everything that can reach it. Recovering that from
/// [`condensation`] means matching two decompositions of the same graph
/// against each other, which is both wasteful and a place for the two
/// numberings to disagree. Here the numbering is an input: whichever algorithm
/// produced `components`, node `i` of the result **is** `components.components[i]`,
/// and `components.component_index(node)` indexes the result directly.
///
/// The edges are the same as [`condensation`]'s: one per pair of distinct
/// components connected in the source graph, deduplicated, so parallel edges
/// and every additional pair of nodes joining the same two components collapse
/// into one. The result is acyclic by construction.
///
/// Node and edge payloads are `()` deliberately. The caller already owns the
/// components (it passed them in), so carrying a clone of each [`Scc`] would
/// duplicate a `BTreeSet` per component to say what
/// `components.components[i]` already says; and everything a fixpoint needs
/// beyond the partition — the in-degree of a component, the dependents to
/// re-queue when its fact changes — falls out of the returned graph's
/// `predecessors` and `successors`.
///
/// # Examples
///
/// The shape this exists for: a grammar `FIRST`-set fixpoint whose work budget
/// is one pass, which holds only if every component is processed after
/// everything it depends on. Kahn's algorithm over the condensation, always
/// taking the smallest ready component, reproduces exactly ascending component
/// order — so the pass itself can simply walk `components` from 0 upwards.
///
/// ```
/// use cfglib::{Graph, NodeId, condensation_of, kosaraju_scc};
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
/// let components = kosaraju_scc(&grammar);
/// let dag = condensation_of(&grammar, &components);
///
/// // The cycle is one component, and the numbering is the input's.
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
/// Panics when `components` was not computed from `graph`: the decomposition
/// covers a fixed node space, and one that disagrees with this graph's node
/// count cannot index it.
#[must_use]
pub fn condensation_of<G: GraphView>(
    graph: &G,
    components: &SccDecomposition<G::NodeId>,
) -> Graph<(), ()> {
    assert!(
        components.component_of.len() == graph.node_bound(),
        "the decomposition covers {} nodes but the graph has {}: condensation_of takes the decomposition OF this graph",
        components.component_of.len(),
        graph.node_bound()
    );

    let mut condensed = Graph::with_capacity(components.len(), components.len());
    for _ in 0..components.len() {
        condensed.add_node(());
    }

    for_each_component_edge(graph, components, |from, to| {
        condensed.add_edge(NodeId::from_index(from), NodeId::from_index(to), ());
    });
    condensed
}

/// Compute strongly connected components with Tarjan's algorithm, numbering
/// them in **reverse topological order of the condensation — leaves first**.
///
/// One pass, `O(V + E)`, and the right default. [`kosaraju_scc`] computes the
/// same partition with the opposite numbering (sources first) for consumers
/// that must process a component before its successors.
#[must_use]
pub fn tarjan_scc<G: GraphView>(graph: &G) -> SccDecomposition<G::NodeId> {
    let node_count = graph.node_bound();
    let mut next_index = 0_usize;
    let mut stack = Vec::new();
    let mut on_stack = vec![false; node_count];
    let mut indices = vec![usize::MAX; node_count];
    let mut lowlinks = vec![0_usize; node_count];
    let mut component_of = vec![0_usize; node_count];
    let mut components = Vec::new();
    // Every frame's successors, appended as the frame is pushed and truncated
    // away as it pops. Both are empty again when a root's walk ends, so one
    // allocation covers the whole decomposition rather than one per root.
    let mut arena: Vec<G::NodeId> = Vec::new();
    let mut calls: Vec<SccFrame<G::NodeId>> = Vec::new();

    for start in graph.node_ids() {
        if indices[start.index()] != usize::MAX {
            continue;
        }

        indices[start.index()] = next_index;
        lowlinks[start.index()] = next_index;
        next_index += 1;
        stack.push(start);
        on_stack[start.index()] = true;
        enter(graph, start, &mut arena, &mut calls);

        // Read frames through `last_mut` and copy the Copy fields out; the
        // successors are no longer among them, so no frame is ever cloned.
        while let Some(frame) = calls.last_mut() {
            let node = frame.node;
            if frame.cursor < arena.len() {
                let successor = arena[frame.cursor];
                frame.cursor += 1;

                if indices[successor.index()] == usize::MAX {
                    indices[successor.index()] = next_index;
                    lowlinks[successor.index()] = next_index;
                    next_index += 1;
                    stack.push(successor);
                    on_stack[successor.index()] = true;
                    enter(graph, successor, &mut arena, &mut calls);
                } else if on_stack[successor.index()] {
                    lowlinks[node.index()] = lowlinks[node.index()].min(indices[successor.index()]);
                }
                continue;
            }

            if lowlinks[node.index()] == indices[node.index()] {
                let mut nodes = BTreeSet::new();
                while let Some(member) = stack.pop() {
                    on_stack[member.index()] = false;
                    nodes.insert(member);
                    if member == node {
                        break;
                    }
                }

                let component_index = components.len();
                for member in &nodes {
                    component_of[member.index()] = component_index;
                }
                components.push(Scc { nodes });
            }

            let Some(finished) = calls.pop() else { break };
            arena.truncate(finished.start);
            if let Some(parent) = calls.last() {
                lowlinks[parent.node.index()] =
                    lowlinks[parent.node.index()].min(lowlinks[node.index()]);
            }
        }
    }

    SccDecomposition {
        components,
        component_of,
    }
}

/// Compute strongly connected components with Kosaraju's algorithm, numbering
/// them in **topological order of the condensation — sources first**.
///
/// # Numbering contract
///
/// The numbering is the reason this exists beside [`tarjan_scc`], which
/// computes the same partition in one pass instead of two but numbers it
/// leaves-first. Here, for every edge `u -> v` of the graph:
///
/// ```text
/// component_index(u) <= component_index(v)
/// ```
///
/// with equality exactly when `u` and `v` are in the same component. So
/// walking [`components`](SccDecomposition::components) in index order visits every
/// component only after every component that can reach it — the order a
/// forward closure over the condensation wants, and the order
/// [`condensation`] does *not* produce (it is built from [`tarjan_scc`], so
/// its nodes are leaves-first).
///
/// # Determinism
///
/// The exact sequence, not just its topological property, is part of the
/// contract: it is the classic two-pass algorithm, deterministic in the
/// graph's own orders.
///
/// 1. An iterative depth-first walk over roots in node-id order, taking each
///    node's successors in adjacency order, recording nodes by finish time.
/// 2. A walk of the reverse graph over those roots in reverse finish order,
///    minting one component per root that is not yet assigned.
///
/// A consumer whose work budget is keyed on the component sequence — a
/// grammar `FIRST`-set fixpoint over a symbol-dependency graph, where
/// processing a component before its dependents is what makes one pass
/// enough — depends on that, so it is stated rather than left to the
/// implementation.
///
/// # Examples
///
/// ```
/// use cfglib::{Graph, kosaraju_scc, tarjan_scc};
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
/// let sources_first = kosaraju_scc(&graph);
/// assert_eq!(sources_first.component_index(entry), 0);
/// assert_eq!(sources_first.component_index(a), 1);
/// assert_eq!(sources_first.component_index(b), 1);
/// assert_eq!(sources_first.component_index(exit), 2);
///
/// // The same partition, numbered the other way round.
/// let leaves_first = tarjan_scc(&graph);
/// assert_eq!(leaves_first.component_index(entry), 2);
/// assert_eq!(leaves_first.component_index(exit), 0);
/// ```
#[must_use]
pub fn kosaraju_scc<G: GraphView>(graph: &G) -> SccDecomposition<G::NodeId> {
    let node_count = graph.node_bound();
    let mut visited = vec![false; node_count];
    let mut finish_order: Vec<G::NodeId> = Vec::with_capacity(node_count);
    // One arena and one frame stack for the whole pass, on the discipline
    // `SccFrame` documents.
    let mut arena: Vec<G::NodeId> = Vec::new();
    let mut calls: Vec<SccFrame<G::NodeId>> = Vec::new();

    // Pass 1: record nodes by finish time over the forward graph.
    for start in graph.node_ids() {
        if visited[start.index()] {
            continue;
        }
        visited[start.index()] = true;
        enter(graph, start, &mut arena, &mut calls);

        // Read frames through `last_mut` and copy the Copy fields out; the
        // successors are no longer among them, so no frame is ever cloned.
        while let Some(frame) = calls.last_mut() {
            if frame.cursor < arena.len() {
                let successor = arena[frame.cursor];
                frame.cursor += 1;
                if !visited[successor.index()] {
                    visited[successor.index()] = true;
                    enter(graph, successor, &mut arena, &mut calls);
                }
                continue;
            }

            finish_order.push(frame.node);
            let Some(finished) = calls.pop() else { break };
            arena.truncate(finished.start);
        }
    }

    // Pass 2: walk the reverse graph in reverse finish order. The first root
    // is in a source component of the condensation, and every root after it
    // is in a source component of what remains — which is what makes the
    // minting order topological.
    let mut component_of = vec![usize::MAX; node_count];
    let mut components: Vec<Scc<G::NodeId>> = Vec::new();
    let mut stack = Vec::new();

    for root in finish_order.into_iter().rev() {
        if component_of[root.index()] != usize::MAX {
            continue;
        }

        let index = components.len();
        let mut nodes = BTreeSet::new();
        component_of[root.index()] = index;
        stack.push(root);
        while let Some(node) = stack.pop() {
            nodes.insert(node);
            for predecessor in graph.predecessors(node) {
                if component_of[predecessor.index()] == usize::MAX {
                    component_of[predecessor.index()] = index;
                    stack.push(predecessor);
                }
            }
        }
        components.push(Scc { nodes });
    }

    SccDecomposition {
        components,
        component_of,
    }
}

#[cfg(test)]
mod tests;
