//! Strongly connected components kept current while the caller edits the
//! edges of a graph.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use super::{NO_COMPONENT, SccDecomposition, SccFrame, enter, kosaraju_scc};
use crate::graph::view::{DenseId, GraphView};

/// The walk's word for a node it has put into a component.
const FINISHED: usize = usize::MAX;

/// Strongly connected components and a topological order of them, kept
/// current while the caller edits the edges of a graph.
///
/// [`compute`](Self::compute) seeds the structure from [`kosaraju_scc`].
/// From then on the caller edits its graph first and reports each edit right
/// after it: [`insert_edge`](Self::insert_edge) for an added edge,
/// [`insert_edges`](Self::insert_edges) for a batch of them, and
/// [`remove_edge`](Self::remove_edge) for a removed one. The graph is the only
/// adjacency: the structure keeps none of its own, reads the graph's where an
/// update needs it, and relies on the graph differing from the one it last
/// saw by exactly the edges reported. Each update returns an [`SccChange`].
///
/// # Order and identity
///
/// The order lays out the nodes, sources first, with each component's nodes
/// as one run in ascending order. A component's
/// [`position`](Self::position) is where its run starts, so positions number
/// nodes rather than components: a component of `k` nodes covers its position
/// and the `k - 1` after it, and the next component starts there. For every
/// edge `u -> v` of the graph,
///
/// ```text
/// position(component_index(u)) <= position(component_index(v))
/// ```
///
/// with equality exactly when `u` and `v` share a component.
///
/// A component's id, which [`component_index`](Self::component_index)
/// returns, is the index of its smallest node. An edit that does not touch a
/// component keeps its id, a merge keeps the smallest id of the components it
/// merges, and a split leaves the old id with the part that holds the old
/// smallest node. Every id is the index of a node, so the component that holds
/// an id from before an update is `component_index(N::from_index(id))`.
///
/// [`decomposition`](Self::decomposition) numbers the components densely in
/// this order, which meets the [`kosaraju_scc`] numbering contract. After an
/// edit it may number them differently from a fresh [`kosaraju_scc`] of the
/// same graph: a graph can have several topological orders, and both are one.
///
/// # Cost
///
/// - An inserted edge `u -> v` whose ends share a component, or whose source's
///   component comes first, costs O(1) and reads no adjacency.
/// - An edge whose target's component comes first opens a **window**: the
///   positions from the start of `v`'s component to the end of `u`'s. Every
///   cycle the edge closes lies inside it, and nothing outside it moves. One
///   Tarjan walk from `v` reads the successors of the nodes in the window
///   that `v` reaches, and of no other node. The components it finds, merged
///   where the edge closed a cycle, move behind the rest of the window, which
///   keeps its order (the shift of Marchetti-Spaccamela, Nanni and Rohnert).
///   The update costs the reached nodes and their successors, plus one write
///   per position of the window.
/// - [`insert_edges`](Self::insert_edges) repairs a whole batch at once.
///   Windows that overlap share one walk, from the targets of all their
///   edges, over the stretch they cover together; a window apart from the
///   others is repaired on its own, since no cycle and no move crosses the
///   gap between two stretches.
/// - A removed edge between two components costs O(1). One inside a
///   component runs a Tarjan walk over that component's nodes alone and lays
///   the parts out in topological order in the component's own run: the
///   component's nodes and their successors.
///
/// # Node set
///
/// The nodes are the graph's nodes at [`compute`](Self::compute), and they
/// stay fixed. A slot that held no node then, such as one a
/// [`Graph`](crate::Graph) removed, belongs to no component. A node added
/// later is not covered: an update or query that names it panics, and so does
/// an update whose walk comes upon it, so compute the structure again after
/// adding nodes. A node removed later stays in the order; once the removals of
/// its edges are reported, it is a component of its own.
///
/// # Determinism
///
/// The walks take their roots in an order fixed by the update and the order
/// before it, and successors in adjacency order, so the same graph and the
/// same sequence of updates give the same order, the same ids and the same
/// changes.
///
/// # Examples
///
/// ```
/// use cfglib::{Graph, IncrementalScc, SccChange};
///
/// // A call chain `a -> b -> c -> d`.
/// let mut graph = Graph::<(), ()>::new();
/// let [a, b, c, d] = [(); 4].map(|()| graph.add_node(()));
/// graph.add_edge(a, b, ());
/// graph.add_edge(b, c, ());
/// graph.add_edge(c, d, ());
/// let mut components = IncrementalScc::compute(&graph);
///
/// // An edge that agrees with the order changes nothing.
/// graph.add_edge(a, d, ());
/// assert_eq!(components.insert_edge(&graph, a, d), SccChange::default());
///
/// // A call from `c` back to `a` closes the cycle `a -> b -> c -> a`, and the
/// // merged component takes the id of its smallest node, `a`.
/// let back = graph.add_edge(c, a, ());
/// let change = components.insert_edge(&graph, c, a);
/// assert_eq!(change.merged, [a.index(), b.index(), c.index()]);
/// assert_eq!(components.component_index(c), a.index());
/// let exit = components.component_index(d);
/// assert!(components.position(a.index()) < components.position(exit));
///
/// // Removing that call splits the cycle into its three nodes again.
/// graph.remove_edge(back);
/// let change = components.remove_edge(&graph, c, a);
/// assert_eq!(change.split, [a.index(), b.index(), c.index()]);
///
/// // The snapshot numbers the components sources first, as `kosaraju_scc`
/// // does.
/// let snapshot = components.decomposition();
/// assert_eq!(snapshot.len(), 4);
/// assert!(snapshot.component_index(a) < snapshot.component_index(d));
/// ```
#[derive(Debug, Clone)]
pub struct IncrementalScc<N> {
    order: Order<N>,
    walk: Walk<N>,
    /// The windows of the last batch, kept so that a batch allocates nothing
    /// once the buffer has grown.
    windows: Vec<Window<N>>,
}

/// What one update of an [`IncrementalScc`] changed.
///
/// Component ids are the ones [`IncrementalScc::component_index`] returns,
/// the index of each component's smallest node. An update that changed
/// nothing returns [`SccChange::default`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct SccChange {
    /// The ids, from before the update, of every component an insertion
    /// merged with another, ascending. Each merged group appears whole,
    /// including the component whose id the group keeps, which is the
    /// smallest of its ids. Always empty after a removal.
    pub merged: Vec<usize>,
    /// The ids of the parts a removal split a component into, ascending,
    /// including the part that kept the old id. Empty when the component
    /// stayed strongly connected, and always after an insertion.
    pub split: Vec<usize>,
    /// The positions the update rewrote. Every position outside the range
    /// holds the same node, in the component with the same id, as before; an
    /// update that kept the whole order returns `0..0`.
    pub reordered: Range<usize>,
}

impl<N: DenseId> IncrementalScc<N> {
    /// Compute the components of `graph` and order them as [`kosaraju_scc`]
    /// numbers them.
    ///
    /// Right after this, [`decomposition`](Self::decomposition) equals
    /// `kosaraju_scc(graph)`. The cost is that of [`kosaraju_scc`], `O(V + E)`,
    /// and the structure keeps three words per node slot besides the order
    /// itself.
    #[must_use]
    pub fn compute<G>(graph: &G) -> Self
    where
        G: GraphView<NodeId = N>,
    {
        // Kosaraju's runs are ascending and in topological order already, so
        // its node array is the order as it stands. Its component table
        // becomes the position table: each entry is read once, as the
        // component whose first node is the id, and then overwritten.
        let SccDecomposition {
            nodes,
            offsets,
            component_of: mut position_of,
        } = kosaraju_scc(graph);
        let mut component_of = vec![NO_COMPONENT; position_of.len()];
        for (position, &node) in nodes.iter().enumerate() {
            let index = node.index();
            component_of[index] = nodes[offsets[position_of[index]]].index();
            position_of[index] = position;
        }
        let walk = Walk::with_node_bound(position_of.len());
        Self {
            order: Order {
                nodes,
                position_of,
                component_of,
            },
            walk,
            windows: Vec::new(),
        }
    }

    /// Report the edge `source -> target`, which the caller has just added to
    /// `graph`.
    ///
    /// [`merged`](SccChange::merged) names the components the edge joined
    /// into a cycle. A parallel edge or a self-loop is reported like any
    /// other, and only changes something if it runs against the order.
    ///
    /// # Panics
    ///
    /// Panics when `source` or `target` is not a node of the graph at
    /// [`compute`](Self::compute) (see [Node set](Self#node-set)).
    pub fn insert_edge<G>(&mut self, graph: &G, source: N, target: N) -> SccChange
    where
        G: GraphView<NodeId = N>,
    {
        self.insert_edges(graph, &[(source, target)])
    }

    /// Report the edges `edges`, `(source, target)` pairs that the caller
    /// has just added to `graph`, repairing the order once for the batch.
    ///
    /// Edges whose windows overlap share one walk, and each stretch of the
    /// order the batch disturbs is rewritten once (see [Cost](Self#cost)).
    /// The components are the ones reporting each edge on its own would give;
    /// the order may differ from that, and is again topological. The change
    /// covers the whole batch.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of [`insert_edge`](Self::insert_edge), for
    /// any edge of the batch.
    pub fn insert_edges<G>(&mut self, graph: &G, edges: &[(N, N)]) -> SccChange
    where
        G: GraphView<NodeId = N>,
    {
        let windows = &mut self.windows;
        windows.clear();
        for &(source, target) in edges {
            let (from, to) = (self.order.start(source), self.order.start(target));
            if to < from {
                windows.push(Window {
                    first: to,
                    last: from,
                    target,
                });
            }
        }
        let mut change = SccChange::default();
        if windows.is_empty() {
            return change;
        }

        // Group the windows that overlap. A cycle the batch closes runs down
        // only along edges against the order, so between two of its
        // components that are next to each other in the order there is one
        // such edge whose window holds both: its windows overlap in a chain,
        // and the whole cycle lies in one group. Two groups' stretches are
        // apart, and every edge between them agrees with the order, so each
        // group is repaired on its own.
        windows.sort_unstable_by_key(|window| window.first);
        let mut rewritten = None;
        let mut group = 0;
        while group < windows.len() {
            let first = windows[group].first;
            let mut last = windows[group].last;
            let mut next = group + 1;
            while next < windows.len() && windows[next].first <= last {
                last = last.max(windows[next].last);
                next += 1;
            }

            // Everything the walk reaches starts at `first` or later: an
            // edge that agrees with the order leads forward, and one against
            // it leads back only from inside its own group's stretch, which
            // for this walk is this group, whose targets start at `first` or
            // later. So only the end needs a test, and a component that starts
            // after `last` reaches no source of the group.
            let order = &self.order;
            self.walk.run(
                graph,
                windows[group..next].iter().map(|window| window.target),
                |node| order.position_of[order.component_of[node.index()]] <= last,
            );
            let end = self.order.run_end(last);
            self.order.collect_merged(&self.walk, &mut change.merged);
            self.order
                .move_reached_last(first..end, &self.walk, &mut rewritten);
            self.walk.clear();
            group = next;
        }
        change.merged.sort_unstable();
        change.reordered = rewritten.unwrap_or(0..0);
        change
    }

    /// Report that the caller has just removed one edge `source -> target`
    /// from `graph`.
    ///
    /// The graph decides what remains: when it still holds a parallel edge, or
    /// any other path that keeps the component strongly connected, the
    /// component stays whole. [`split`](SccChange::split) names the parts when
    /// it does not.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of [`insert_edge`](Self::insert_edge).
    pub fn remove_edge<G>(&mut self, graph: &G, source: N, target: N) -> SccChange
    where
        G: GraphView<NodeId = N>,
    {
        let component = self.order.component(source);
        if self.order.component(target) != component {
            return SccChange::default();
        }
        let start = self.order.position_of[component];
        let end = self.order.run_end(start);
        if end - start == 1 {
            // A self-loop of a node alone in its component.
            return SccChange::default();
        }

        let order = &self.order;
        self.walk
            .run(graph, order.nodes[start..end].iter().copied(), |node| {
                order.component_of[node.index()] == component
            });
        let mut change = SccChange::default();
        if self.walk.component_count() > 1 {
            change.split = self
                .walk
                .components()
                .map(|nodes| nodes[0].index())
                .collect();
            change.split.sort_unstable();
            let mut rewritten = None;
            self.order.place_found(start, &self.walk, &mut rewritten);
            change.reordered = rewritten.unwrap_or(0..0);
        }
        self.walk.clear();
        change
    }

    /// Return the id of `node`'s component: the index of its smallest node.
    ///
    /// # Panics
    ///
    /// Panics when `node` is not a node of the graph at
    /// [`compute`](Self::compute) (see [Node set](Self#node-set)).
    #[must_use]
    pub fn component_index(&self, node: N) -> usize {
        self.order.component(node)
    }

    /// Return where the run of `component`'s nodes starts in the order.
    ///
    /// # Panics
    ///
    /// Panics when `component` is not the id of a component, that is, not
    /// the index of the smallest node of one.
    #[must_use]
    pub fn position(&self, component: usize) -> usize {
        assert!(
            self.order.component_of.get(component) == Some(&component),
            "{component} is not the id of a component"
        );
        self.order.position_of[component]
    }

    /// Return the components numbered densely in the order, sources first.
    ///
    /// The result meets the [`kosaraju_scc`] numbering contract,
    /// `component_index(u) <= component_index(v)` on every edge `u -> v`,
    /// but may number the components differently from a fresh
    /// [`kosaraju_scc`] once an update has reordered them. Its tables span
    /// the node bound at [`compute`](Self::compute). `O(V)`.
    #[must_use]
    pub fn decomposition(&self) -> SccDecomposition<N> {
        let nodes = self.order.nodes.clone();
        let mut offsets = vec![0];
        let mut component_of = vec![NO_COMPONENT; self.order.component_of.len()];
        for (position, &node) in nodes.iter().enumerate() {
            // A run starts at its smallest node, whose index is the id.
            if position > 0 && self.order.component_of[node.index()] == node.index() {
                offsets.push(position);
            }
            component_of[node.index()] = offsets.len() - 1;
        }
        if !nodes.is_empty() {
            offsets.push(nodes.len());
        }
        SccDecomposition {
            nodes,
            offsets,
            component_of,
        }
    }
}

/// The order and the two tables that index it.
#[derive(Debug, Clone)]
struct Order<N> {
    /// Position to node: every component's nodes as one ascending run, the
    /// runs sources first.
    nodes: Vec<N>,
    /// Each slot's position in `nodes`; [`NO_COMPONENT`] for a slot that held
    /// no node at [`IncrementalScc::compute`].
    position_of: Vec<usize>,
    /// Each slot's component id, which is the index of the component's
    /// smallest node and so of the first node of its run; [`NO_COMPONENT`] as
    /// above.
    component_of: Vec<usize>,
}

impl<N: DenseId> Order<N> {
    /// The id of `node`'s component, checked.
    fn component(&self, node: N) -> usize {
        let component = self
            .component_of
            .get(node.index())
            .copied()
            .unwrap_or(NO_COMPONENT);
        assert!(
            component != NO_COMPONENT,
            "node {} is not a node of the graph the components were computed for",
            node.index()
        );
        component
    }

    /// Where the run of `node`'s component starts.
    fn start(&self, node: N) -> usize {
        self.position_of[self.component(node)]
    }

    /// Where the run that starts at `start` ends.
    fn run_end(&self, start: usize) -> usize {
        let component = self.component_of[self.nodes[start].index()];
        let mut end = start + 1;
        while end < self.nodes.len() && self.component_of[self.nodes[end].index()] == component {
            end += 1;
        }
        end
    }

    /// The ids, from before the update, of the components whose nodes make up
    /// `found`, one of the walk's components. A component found whole holds
    /// its smallest node, the one node whose id is its own index.
    fn ids_within<'a>(&'a self, found: &'a [N]) -> impl Iterator<Item = usize> + 'a {
        found
            .iter()
            .map(|node| node.index())
            .filter(|&index| self.component_of[index] == index)
    }

    /// Add to `merged` the ids of the components the walk found in one
    /// component with another, before any of them is relabeled.
    fn collect_merged(&self, walk: &Walk<N>, merged: &mut Vec<usize>) {
        for found in walk.components() {
            if self.ids_within(found).nth(1).is_some() {
                merged.extend(self.ids_within(found));
            }
        }
    }

    /// Lay out `window` again after a walk from inside it: the components it
    /// did not reach first, in the order they had, then the ones it did, in
    /// topological order. Widens `rewritten` over the positions it rewrites.
    fn move_reached_last(
        &mut self,
        window: Range<usize>,
        walk: &Walk<N>,
        rewritten: &mut Option<Range<usize>>,
    ) {
        // Compact the unreached nodes towards the start. A write lands at or
        // before the position read in the same step, so nothing is
        // overwritten before it is read.
        let mut next = window.start;
        for position in window {
            let node = self.nodes[position];
            if !walk.reached(node) {
                self.place(next, node, self.component_of[node.index()], rewritten);
                next += 1;
            }
        }
        self.place_found(next, walk, rewritten);
    }

    /// Lay out the walk's components from `next` on, in topological order,
    /// each with the id of its smallest node.
    fn place_found(
        &mut self,
        mut next: usize,
        walk: &Walk<N>,
        rewritten: &mut Option<Range<usize>>,
    ) {
        // The walk completes components leaves first.
        for found in walk.components().rev() {
            let id = found[0].index();
            for &node in found {
                self.place(next, node, id, rewritten);
                next += 1;
            }
        }
    }

    /// Put `node` at `position` in the component `id`, widening `rewritten`
    /// over the position when that changes either.
    fn place(&mut self, position: usize, node: N, id: usize, rewritten: &mut Option<Range<usize>>) {
        if self.nodes[position] != node || self.component_of[node.index()] != id {
            let range = rewritten.get_or_insert(position..position);
            range.start = range.start.min(position);
            range.end = range.end.max(position + 1);
            self.nodes[position] = node;
            self.position_of[node.index()] = position;
            self.component_of[node.index()] = id;
        }
    }
}

/// The window an edge against the order opens: the starts of its target's
/// and its source's components, and the target, where its walk starts.
#[derive(Debug, Clone, Copy)]
struct Window<N> {
    first: usize,
    last: usize,
    target: N,
}

/// Tarjan's walk over part of a graph, with its buffers kept across updates.
#[derive(Debug, Clone)]
struct Walk<N> {
    /// Per slot, Pearce's one word: 0 for a node not visited, `1..` for a
    /// node waiting for its component, holding the lowest index it has seen,
    /// and [`FINISHED`] once it is in a component. Every word is 0 between
    /// walks; [`clear`](Self::clear) resets only the nodes a walk visited.
    rindex: Vec<usize>,
    /// The visited nodes that were not the root of their component, waiting
    /// for that root to finish.
    waiting: Vec<N>,
    /// The successors of every frame on the stack, as `SccFrame` describes.
    arena: Vec<N>,
    calls: Vec<SccFrame<N>>,
    /// The components found, each an ascending run, leaves first.
    found: Vec<N>,
    /// `offsets[i]..offsets[i + 1]` is the `i`th component within `found`.
    offsets: Vec<usize>,
}

impl<N: DenseId> Walk<N> {
    fn with_node_bound(node_bound: usize) -> Self {
        Self {
            rindex: vec![0; node_bound],
            waiting: Vec::new(),
            arena: Vec::new(),
            calls: Vec::new(),
            found: Vec::new(),
            offsets: vec![0],
        }
    }

    /// Find the components of the nodes `roots` reach through nodes `within`
    /// admits, in completion order, leaves first.
    ///
    /// This is the walk of [`tarjan_scc`](super::tarjan_scc) from chosen
    /// roots, with every node outside `within` left out of the graph. A
    /// component that a root reaches is found whole, since all its nodes are
    /// reachable from any one of them, as long as `within` admits them all.
    fn run<G>(&mut self, graph: &G, roots: impl IntoIterator<Item = N>, within: impl Fn(N) -> bool)
    where
        G: GraphView<NodeId = N>,
    {
        let mut next_index = 1;
        for root in roots {
            if self.rindex[root.index()] != 0 {
                continue;
            }
            self.rindex[root.index()] = next_index;
            next_index += 1;
            enter(graph, root, &mut self.arena, &mut self.calls);

            while let Some(frame) = self.calls.last_mut() {
                let node = frame.node;
                if frame.cursor < self.arena.len() {
                    let successor = self.arena[frame.cursor];
                    frame.cursor += 1;
                    if !within(successor) {
                        continue;
                    }
                    let seen = self.rindex[successor.index()];
                    if seen == 0 {
                        self.rindex[successor.index()] = next_index;
                        next_index += 1;
                        enter(graph, successor, &mut self.arena, &mut self.calls);
                    } else if seen < self.rindex[node.index()] {
                        self.rindex[node.index()] = seen;
                        frame.root = false;
                    }
                    continue;
                }

                let Some(finished) = self.calls.pop() else {
                    break;
                };
                self.arena.truncate(finished.start);
                if finished.root {
                    // The members are the waiting nodes that hold an index no
                    // lower than the root's, as in `tarjan_scc`.
                    let first = self.found.len();
                    let root_index = self.rindex[node.index()];
                    self.rindex[node.index()] = FINISHED;
                    self.found.push(node);
                    while let Some(&member) = self.waiting.last() {
                        if self.rindex[member.index()] < root_index {
                            break;
                        }
                        self.waiting.pop();
                        self.rindex[member.index()] = FINISHED;
                        self.found.push(member);
                    }
                    self.found[first..].sort_unstable();
                    self.offsets.push(self.found.len());
                } else {
                    self.waiting.push(node);
                }
                if let Some(parent) = self.calls.last_mut()
                    && self.rindex[node.index()] < self.rindex[parent.node.index()]
                {
                    self.rindex[parent.node.index()] = self.rindex[node.index()];
                    parent.root = false;
                }
            }
        }
    }

    /// Whether the last walk put `node` into a component.
    fn reached(&self, node: N) -> bool {
        self.rindex[node.index()] == FINISHED
    }

    /// The components the last walk found, leaves first.
    fn components(&self) -> impl DoubleEndedIterator<Item = &[N]> + '_ {
        self.offsets
            .windows(2)
            .map(|run| &self.found[run[0]..run[1]])
    }

    fn component_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Forget the last walk, leaving every word of `rindex` 0 again.
    fn clear(&mut self) {
        for &node in &self.found {
            self.rindex[node.index()] = 0;
        }
        self.found.clear();
        self.offsets.truncate(1);
    }
}

#[cfg(test)]
mod tests;
