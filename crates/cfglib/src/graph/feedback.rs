//! Linear arrangements and feedback arc sets: deterministic acyclic views of
//! cyclic graphs.
//!
//! [`topological_sort`](crate::topological_sort) gives up on the first cycle,
//! and [`find_back_edges`](crate::find_back_edges) needs a root and a
//! dominator tree, and misses the cycles of an irreducible region. A consumer
//! that wants an order over a cyclic graph — the iteration order inside a
//! recursive component, a schedule over cyclic dependencies, the edges a
//! drawing should not rank by — wants one that violates as few edges as it
//! can, and the edges it violates.
//!
//! [`linear_arrangement`] orders every live node: the components of
//! [`kosaraju_scc`] sources first, and the nodes of each cyclic component by
//! the greedy heuristic of Eades, Lin and Smyth (*A fast and effective
//! heuristic for the feedback arc set problem*, Information Processing
//! Letters 47(6), 1993). [`feedback_arc_set`] reports the edges that point
//! backward in that order, self-loops included; removing them leaves the
//! graph acyclic. With [`FeedbackConfig::minimal`], it then re-admits every
//! reported edge whose return closes no cycle, the minimality pass of
//! Cavallaro, Cutello and Pavone (*Efficient heuristics to compute minimal and
//! stable feedback arc sets*, J. Comb. Optim. 48, 2024).
//!
//! A *minimum* feedback arc set is NP-hard and is not attempted: the set is
//! small, not smallest.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::edge_view::EdgeView;
use super::epoch::EpochMarks;
use super::scc::{SccDecomposition, kosaraju_scc};
use super::view::{DenseId, GraphView};
use crate::identity::{NO_INDEX, check_bound, slot, word};

/// Options of [`feedback_arc_set`].
///
/// The field is public and the struct is `Copy`, so a consumer that stores
/// its options as data can write a literal; [`new`](Self::new) plus
/// [`with_minimal`](Self::with_minimal) are the ergonomic form. The default
/// is the linear-time set, without the minimality pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FeedbackConfig {
    /// Run the minimality pass: re-admit each edge of the set whose return
    /// closes no cycle, so that returning any edge left in the set closes
    /// one. It costs one search of the component per edge of the set,
    /// `O(f · (n + m))` for a component of `n` nodes and `m` edges whose
    /// set has `f` edges.
    pub minimal: bool,
}

impl FeedbackConfig {
    /// The linear-time set, without the minimality pass.
    #[must_use]
    pub const fn new() -> Self {
        Self { minimal: false }
    }

    /// Return this configuration with the minimality pass on or off.
    #[must_use]
    pub const fn with_minimal(mut self, minimal: bool) -> Self {
        self.minimal = minimal;
        self
    }
}

/// Order every live node so that few edges point backward.
///
/// The result is a permutation of the live nodes. The components of
/// [`kosaraju_scc`] come in its order, sources first, so every edge between
/// two components points forward. A singleton component is its node. A
/// cyclic component is ordered by the Eades–Lin–Smyth heuristic over the
/// edges inside it, self-loops aside: repeatedly peel a sink to the tail of
/// the component's run, else a source to its head, else the node whose
/// out-degree minus in-degree is largest to its head. Every edge between a
/// peeled sink or source and the nodes still left points forward, since a
/// sink ends after them and a source starts before them. The degree
/// differences of the nodes left sum to zero, so the node taken by the
/// largest one has at least as many edges to them as from them. So at most
/// half of the component's edges, self-loops aside, point backward.
///
/// Edges are counted as [`successors`](GraphView::successors) yields them, so
/// parallel edges count once each. The run is O(n + m) for `n` slots and `m`
/// edges: the candidates sit in bucket queues keyed by out-degree minus
/// in-degree.
///
/// # Determinism
///
/// Every choice is a function of node ids and adjacency order. Candidates of
/// equal priority are taken first in, first out: the buckets are filled in
/// ascending node id, and a node goes to the back of a bucket when its
/// degrees change. So ties go to the smaller node id until peeling
/// disturbs them, and the same graph always gives the same arrangement.
///
/// # Panics
///
/// Panics when `graph` has a cycle and its node bound exceeds `u32::MAX`.
///
/// # Examples
///
/// ```
/// use cfglib::{Graph, linear_arrangement};
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
/// // Only `b -> a` points backward: no order of a cycle avoids one edge.
/// assert_eq!(linear_arrangement(&graph), vec![entry, a, b, exit]);
/// ```
#[must_use]
pub fn linear_arrangement<G: GraphView>(graph: &G) -> Vec<G::NodeId> {
    arrange(graph).1
}

/// Return the edges that point backward in [`linear_arrangement`], self-loops
/// included: removing them leaves the graph acyclic.
///
/// Every edge between two strongly connected components points forward, so
/// the set holds only edges inside a cyclic component, and at most half of a
/// component's edges other than self-loops. Parallel edges are reported per
/// edge. The set is empty exactly when the graph is acyclic.
///
/// With [`FeedbackConfig::minimal`], the set is then scanned in order, and
/// each edge `u -> v` other than a self-loop is re-admitted unless `v`
/// reaches `u` over the edges admitted so far. The graph stays acyclic, and
/// the set becomes minimal: returning any one of its edges closes a cycle.
///
/// The edges come in the order [`edge_ids`](EdgeView::edge_ids) yields them,
/// ascending identity for this crate's stores; the minimality pass keeps that
/// order.
///
/// # Complexity
///
/// `O(n + m)` for `n` slots and `m` edges. With `minimal`, add
/// `O(f · (n_c + m_c))` for each component of `n_c` nodes and `m_c` edges
/// whose set has `f` edges.
///
/// # Panics
///
/// Panics when `graph` has a cycle and its node bound exceeds `u32::MAX`.
///
/// # Examples
///
/// ```
/// use cfglib::{
///     EdgeId, FeedbackConfig, FilteredEdges, Graph, feedback_arc_set, linear_arrangement,
///     topological_sort,
/// };
///
/// // A two-node cycle reached from `entry`, plus a self-loop on `b`.
/// let mut graph = Graph::<&str, ()>::new();
/// let entry = graph.add_node("entry");
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// graph.add_edge(entry, a, ());
/// graph.add_edge(a, b, ());
/// let back = graph.add_edge(b, a, ());
/// let spin = graph.add_edge(b, b, ());
/// assert_eq!(topological_sort(&graph), None);
///
/// let set = feedback_arc_set(&graph, FeedbackConfig::new());
/// assert_eq!(set, vec![back, spin]);
///
/// // Without the set the graph is acyclic, and the arrangement is a
/// // topological order of what is left.
/// let acyclic = FilteredEdges::new(&graph, |edge: EdgeId, _: &()| !set.contains(&edge));
/// assert_eq!(topological_sort(&acyclic), Some(linear_arrangement(&graph)));
///
/// // Both edges are needed, so the minimality pass keeps them.
/// let minimal = feedback_arc_set(&graph, FeedbackConfig::new().with_minimal(true));
/// assert_eq!(minimal, set);
/// ```
#[must_use]
pub fn feedback_arc_set<G: EdgeView>(graph: &G, config: FeedbackConfig) -> Vec<G::EdgeId> {
    let (components, order) = arrange(graph);
    let mut position = vec![0; graph.node_bound()];
    for (rank, node) in order.iter().enumerate() {
        position[node.index()] = rank;
    }
    let mut set: Vec<G::EdgeId> = graph
        .edge_ids()
        .filter(|&edge| {
            let edge = graph.edge(edge);
            position[edge.target().index()] <= position[edge.source().index()]
        })
        .collect();
    if config.minimal && !set.is_empty() {
        minimize(graph, &components, &mut set);
    }
    set
}

/// The decomposition and the arrangement of [`linear_arrangement`].
fn arrange<G: GraphView>(graph: &G) -> (SccDecomposition<G::NodeId>, Vec<G::NodeId>) {
    let components = kosaraju_scc(graph);
    let mut order = Vec::with_capacity(graph.node_bound());
    // An acyclic graph has no component to peel, so it never pays for the
    // buffers.
    let mut peeling = None;
    for (index, nodes) in components.components().enumerate() {
        if let [node] = *nodes {
            order.push(node);
        } else {
            peeling
                .get_or_insert_with(|| Peeling::new(graph.node_bound()))
                .arrange(graph, &components, index, nodes, &mut order);
        }
    }
    (components, order)
}

/// The list of the nodes with no out-edge left inside the component.
const SINKS: usize = 0;
/// The list of the nodes with no in-edge left and some out-edge.
const SOURCES: usize = 1;
/// The first degree bucket. A node with both kinds of edge left sits in
/// list `BUCKETS + offset + out-degree - in-degree`.
const BUCKETS: usize = 2;

/// The Eades–Lin–Smyth peeling of one cyclic component at a time.
///
/// The node-indexed tables span the whole graph and are filled one
/// component at a time, so they are allocated once for the graph. Only the
/// nodes of the component being peeled are ever filed in a list, which makes
/// "filed" the whole test for "a remaining node of this component".
struct Peeling {
    /// Out-edges left from each node to remaining nodes of its component.
    out_degree: Vec<usize>,
    /// In-edges left to each node from remaining nodes of its component.
    in_degree: Vec<usize>,
    /// The list each node is filed in, [`NO_INDEX`] outside the component
    /// being peeled and once peeled.
    list: Vec<u32>,
    /// The next node of the same list, [`NO_INDEX`] at its end.
    next: Vec<u32>,
    /// The previous node of the same list, [`NO_INDEX`] at its start.
    previous: Vec<u32>,
    /// The first node of each list, [`NO_INDEX`] for an empty one.
    head: Vec<u32>,
    /// The last node of each list, where a node is filed.
    tail: Vec<u32>,
    /// The largest in-degree of the component, so that every bucket index
    /// is positive.
    offset: usize,
    /// No bucket above this one holds a node.
    top: usize,
    /// The sinks in the order they were peeled, the reverse of their order
    /// at the component's tail.
    sinks: Vec<usize>,
}

impl Peeling {
    fn new(node_bound: usize) -> Self {
        check_bound(node_bound, "node bound of a feedback arc set");
        Self {
            out_degree: vec![0; node_bound],
            in_degree: vec![0; node_bound],
            list: vec![NO_INDEX; node_bound],
            next: vec![NO_INDEX; node_bound],
            previous: vec![NO_INDEX; node_bound],
            head: Vec::new(),
            tail: Vec::new(),
            offset: 0,
            top: BUCKETS,
            sinks: Vec::new(),
        }
    }

    /// Append the Eades–Lin–Smyth order of the cyclic component `component`,
    /// whose ascending nodes are `nodes`, to `order`.
    fn arrange<G: GraphView>(
        &mut self,
        graph: &G,
        components: &SccDecomposition<G::NodeId>,
        component: usize,
        nodes: &[G::NodeId],
        order: &mut Vec<G::NodeId>,
    ) {
        for &node in nodes {
            for successor in graph.successors(node) {
                if successor != node && components.component_index(successor) == component {
                    self.out_degree[node.index()] += 1;
                    self.in_degree[successor.index()] += 1;
                }
            }
        }
        let (max_out, max_in) = nodes.iter().fold((0, 0), |(max_out, max_in), node| {
            (
                max_out.max(self.out_degree[node.index()]),
                max_in.max(self.in_degree[node.index()]),
            )
        });
        let lists = BUCKETS + max_in + max_out + 1;
        self.head.clear();
        self.head.resize(lists, NO_INDEX);
        self.tail.clear();
        self.tail.resize(lists, NO_INDEX);
        self.offset = max_in;
        self.top = BUCKETS;
        for &node in nodes {
            self.file(node.index());
        }

        let mut remaining = nodes.len();
        self.sinks.clear();
        while remaining > 0 {
            while let Some(node) = self.take(SINKS) {
                self.peel(graph, G::NodeId::from_index(node));
                self.sinks.push(node);
                remaining -= 1;
            }
            while let Some(node) = self.take(SOURCES) {
                self.peel(graph, G::NodeId::from_index(node));
                order.push(G::NodeId::from_index(node));
                remaining -= 1;
            }
            if remaining > 0 {
                let node = self.take_highest();
                self.peel(graph, G::NodeId::from_index(node));
                order.push(G::NodeId::from_index(node));
                remaining -= 1;
            }
        }
        order.extend(
            self.sinks
                .iter()
                .rev()
                .map(|&node| G::NodeId::from_index(node)),
        );
    }

    /// The list `node` belongs in by its degrees left.
    fn key(&self, node: usize) -> usize {
        let (out_degree, in_degree) = (self.out_degree[node], self.in_degree[node]);
        if out_degree == 0 {
            SINKS
        } else if in_degree == 0 {
            SOURCES
        } else {
            BUCKETS + self.offset + out_degree - in_degree
        }
    }

    /// File `node` at the tail of the list its degrees put it in.
    fn file(&mut self, node: usize) {
        let list = self.key(node);
        self.list[node] = word(list);
        self.next[node] = NO_INDEX;
        let previous = self.tail[list];
        self.previous[node] = previous;
        match previous {
            NO_INDEX => self.head[list] = word(node),
            previous => self.next[slot(previous)] = word(node),
        }
        self.tail[list] = word(node);
        if list > self.top {
            self.top = list;
        }
    }

    /// Take `node` out of its list.
    fn unlink(&mut self, node: usize) {
        let (list, previous, next) = (slot(self.list[node]), self.previous[node], self.next[node]);
        match previous {
            NO_INDEX => self.head[list] = next,
            previous => self.next[slot(previous)] = next,
        }
        match next {
            NO_INDEX => self.tail[list] = previous,
            next => self.previous[slot(next)] = previous,
        }
        self.list[node] = NO_INDEX;
    }

    /// Take the first node of `list`, if it holds one.
    fn take(&mut self, list: usize) -> Option<usize> {
        let node = self.head[list];
        if node == NO_INDEX {
            return None;
        }
        let node = slot(node);
        self.unlink(node);
        Some(node)
    }

    /// Take the first node of the highest nonempty bucket.
    ///
    /// Called only while nodes remain and no sink or source does, so every
    /// remaining node is in a bucket at or below `top`. `top` falls only
    /// past empty buckets and rises by one per degree change, so the scans
    /// of a component cost O(buckets + edges) together.
    fn take_highest(&mut self) -> usize {
        while self.head[self.top] == NO_INDEX {
            self.top -= 1;
        }
        self.take(self.top)
            .expect("the highest nonempty bucket holds a node")
    }

    /// Remove `node` from the component and refile every remaining
    /// neighbor whose degrees changed. `node` is already unlinked, so its
    /// self-loops find it unfiled and are skipped.
    fn peel<G: GraphView>(&mut self, graph: &G, node: G::NodeId) {
        for successor in graph.successors(node) {
            let successor = successor.index();
            if self.list[successor] != NO_INDEX {
                self.in_degree[successor] -= 1;
                self.refile(successor);
            }
        }
        for predecessor in graph.predecessors(node) {
            let predecessor = predecessor.index();
            if self.list[predecessor] != NO_INDEX {
                self.out_degree[predecessor] -= 1;
                self.refile(predecessor);
            }
        }
    }

    /// Move `node` to the tail of the list its degrees now put it in, unless
    /// it is already in that list.
    fn refile(&mut self, node: usize) {
        if self.key(node) != slot(self.list[node]) {
            self.unlink(node);
            self.file(node);
        }
    }
}

/// Re-admit every edge of `set` whose return closes no cycle, scanning in
/// order, and keep the rest in order.
///
/// An edge closes a cycle only with a path back inside its own component,
/// so each search stays inside the component. The admitted edges only grow,
/// so an edge kept because its target reached its source still closes a
/// cycle at the end.
fn minimize<G: EdgeView>(
    graph: &G,
    components: &SccDecomposition<G::NodeId>,
    set: &mut Vec<G::EdgeId>,
) {
    let mut excluded = vec![false; graph.edge_bound()];
    for edge in set.iter() {
        excluded[edge.index()] = true;
    }
    let mut marks = EpochMarks::new(graph.node_bound());
    let mut stack = Vec::new();
    set.retain(|&edge| {
        let edge_ref = graph.edge(edge);
        let (source, target) = (edge_ref.source(), edge_ref.target());
        // A self-loop is a cycle on its own.
        if source == target
            || reaches(
                graph, components, &excluded, target, source, &mut marks, &mut stack,
            )
        {
            return true;
        }
        excluded[edge.index()] = false;
        false
    });
}

/// Whether `from` reaches `to`, a different node of its component, over the
/// edges not `excluded`.
fn reaches<G: EdgeView>(
    graph: &G,
    components: &SccDecomposition<G::NodeId>,
    excluded: &[bool],
    from: G::NodeId,
    to: G::NodeId,
    marks: &mut EpochMarks,
    stack: &mut Vec<G::NodeId>,
) -> bool {
    let component = components.component_index(from);
    marks.reset();
    marks.mark(from.index());
    stack.clear();
    stack.push(from);
    while let Some(node) = stack.pop() {
        for edge in graph.outgoing(node) {
            if excluded[edge.index()] {
                continue;
            }
            let next = graph.edge(edge).target();
            if next == to {
                return true;
            }
            if !marks.is_marked(next.index()) && components.component_index(next) == component {
                marks.mark(next.index());
                stack.push(next);
            }
        }
    }
    false
}

#[cfg(test)]
mod tests;
