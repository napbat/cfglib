//! Strongly connected components kept current while the caller edits the
//! edges of a graph.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use super::tarjan::Tarjan;
use super::{SccDecomposition, SccNumbering};
use crate::block::BlockId;
use crate::graph::view::{DenseId, GraphView};
use crate::identity::{NO_INDEX, slot, word};

/// Strongly connected components and a topological order of them, kept
/// current while the caller edits the edges of a graph.
///
/// [`compute`](Self::compute) seeds the structure from
/// [`SccNumbering::SourcesFirst`].
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
/// position(component_id(u)) <= position(component_id(v))
/// ```
///
/// with equality exactly when `u` and `v` share a component.
///
/// A component's id, which [`component_id`](Self::component_id) returns, is
/// the index of its smallest node. An edit that does not touch a component
/// keeps its id, a merge keeps the smallest id of the components it merges,
/// and a split leaves the old id with the part that holds the old smallest
/// node. Every id is the index of a node, so the component that holds an id
/// from before an update is `component_id(N::from_index(id))`. An id is not
/// the dense index an [`SccDecomposition`] numbers a component with.
///
/// [`decomposition`](Self::decomposition) numbers the components densely in
/// this order, which meets the [`SccNumbering::SourcesFirst`] numbering
/// contract. After an edit it may number them differently from a fresh
/// sources-first decomposition of the same graph: a graph can have several
/// topological orders, and both are one.
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
/// assert_eq!(components.component_id(c), a.index());
/// let exit = components.component_id(d);
/// assert!(components.position(a.index()) < components.position(exit));
///
/// // Removing that call splits the cycle into its three nodes again.
/// graph.remove_edge(back);
/// let change = components.remove_edge(&graph, c, a);
/// assert_eq!(change.split, [a.index(), b.index(), c.index()]);
///
/// // The snapshot numbers the components sources first, as
/// // `SccNumbering::SourcesFirst` does.
/// let snapshot = components.decomposition();
/// assert_eq!(snapshot.len(), 4);
/// assert!(snapshot.component_index(a) < snapshot.component_index(d));
/// ```
#[derive(Debug, Clone)]
pub struct IncrementalScc<N = BlockId> {
    order: Order<N>,
    walk: Tarjan<N>,
    /// The windows of the last batch, kept so that a batch allocates nothing
    /// once the buffer has grown.
    windows: Vec<Window<N>>,
}

/// What one update of an [`IncrementalScc`] changed.
///
/// Component ids are the ones [`IncrementalScc::component_id`] returns,
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
    /// Compute the components of `graph` and order them as
    /// [`SccNumbering::SourcesFirst`] numbers them.
    ///
    /// Right after this, [`decomposition`](Self::decomposition) equals the
    /// sources-first [`SccDecomposition::compute`] of `graph`, at its cost,
    /// `O(V + E)`, and the structure keeps three words per node slot besides
    /// the order itself.
    ///
    /// # Panics
    ///
    /// Panics when [`node_bound`](GraphView::node_bound) exceeds the range of
    /// a dense `u32` table.
    #[must_use]
    pub fn compute<G>(graph: &G) -> Self
    where
        G: GraphView<NodeId = N>,
    {
        // The sources-first runs are ascending and in topological order
        // already, so its node array is the order as it stands. Its component
        // table becomes the position table: each entry is read once, as the
        // component whose first node is the id, and then overwritten.
        let SccDecomposition {
            nodes,
            offsets,
            component_of: mut position_of,
        } = SccDecomposition::compute(graph, SccNumbering::SourcesFirst);
        let mut component_of = vec![NO_INDEX; position_of.len()];
        for (position, &node) in nodes.iter().enumerate() {
            let index = node.index();
            let first = nodes[slot(offsets[slot(position_of[index])])];
            component_of[index] = word(first.index());
            position_of[index] = word(position);
        }
        let walk = Tarjan::with_node_bound(position_of.len());
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
                |node| slot(order.position_of[slot(order.component_of[node.index()])]) <= last,
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
        let start = slot(self.order.position_of[component]);
        let end = self.order.run_end(start);
        if end - start == 1 {
            // A self-loop of a node alone in its component.
            return SccChange::default();
        }

        let order = &self.order;
        let id = word(component);
        self.walk
            .run(graph, order.nodes[start..end].iter().copied(), |node| {
                order.component_of[node.index()] == id
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
    /// The id stays with the component across edits that do not touch it
    /// (see [Order and identity](Self#order-and-identity)); it is not a dense
    /// component index.
    ///
    /// # Panics
    ///
    /// Panics when `node` is not a node of the graph at
    /// [`compute`](Self::compute) (see [Node set](Self#node-set)).
    #[must_use]
    pub fn component_id(&self, node: N) -> usize {
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
            self.order
                .component_of
                .get(component)
                .is_some_and(|&id| slot(id) == component),
            "{component} is not the id of a component"
        );
        slot(self.order.position_of[component])
    }

    /// Return the components numbered densely in the order, sources first.
    ///
    /// The result meets the [`SccNumbering::SourcesFirst`] numbering
    /// contract, `component_index(u) <= component_index(v)` on every edge
    /// `u -> v`, but may number the components differently from a fresh
    /// sources-first decomposition once an update has reordered them. Its
    /// tables span the node bound at [`compute`](Self::compute). `O(V)`.
    #[must_use]
    pub fn decomposition(&self) -> SccDecomposition<N> {
        let nodes = self.order.nodes.clone();
        let mut offsets = vec![0];
        let mut component_of = vec![NO_INDEX; self.order.component_of.len()];
        for (position, &node) in nodes.iter().enumerate() {
            // A run starts at its smallest node, whose index is the id.
            if position > 0 && slot(self.order.component_of[node.index()]) == node.index() {
                offsets.push(word(position));
            }
            component_of[node.index()] = word(offsets.len() - 1);
        }
        if !nodes.is_empty() {
            offsets.push(word(nodes.len()));
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
    /// Each slot's position in `nodes`; [`NO_INDEX`] for a slot that held no
    /// node at [`IncrementalScc::compute`].
    position_of: Vec<u32>,
    /// Each slot's component id, which is the index of the component's
    /// smallest node and so of the first node of its run; [`NO_INDEX`] as
    /// above.
    component_of: Vec<u32>,
}

impl<N: DenseId> Order<N> {
    /// The id of `node`'s component, checked.
    fn component(&self, node: N) -> usize {
        let component = self
            .component_of
            .get(node.index())
            .copied()
            .unwrap_or(NO_INDEX);
        assert!(
            component != NO_INDEX,
            "node {} is not a node of the graph the components were computed for",
            node.index()
        );
        slot(component)
    }

    /// Where the run of `node`'s component starts.
    fn start(&self, node: N) -> usize {
        slot(self.position_of[self.component(node)])
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
            .filter(|&index| slot(self.component_of[index]) == index)
    }

    /// Add to `merged` the ids of the components the walk found in one
    /// component with another, before any of them is relabeled.
    fn collect_merged(&self, walk: &Tarjan<N>, merged: &mut Vec<usize>) {
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
        walk: &Tarjan<N>,
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
        walk: &Tarjan<N>,
        rewritten: &mut Option<Range<usize>>,
    ) {
        // The walk completes components leaves first.
        for found in walk.components().rev() {
            let id = word(found[0].index());
            for &node in found {
                self.place(next, node, id, rewritten);
                next += 1;
            }
        }
    }

    /// Put `node` at `position` in the component `id`, widening `rewritten`
    /// over the position when that changes either.
    fn place(&mut self, position: usize, node: N, id: u32, rewritten: &mut Option<Range<usize>>) {
        if self.nodes[position] != node || self.component_of[node.index()] != id {
            let range = rewritten.get_or_insert(position..position);
            range.start = range.start.min(position);
            range.end = range.end.max(position + 1);
            self.nodes[position] = node;
            self.position_of[node.index()] = word(position);
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

#[cfg(test)]
mod tests;
