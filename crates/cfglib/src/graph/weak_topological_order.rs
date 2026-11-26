//! Weak topological orders: Bourdoncle's hierarchical iteration order.
//!
//! A weak topological order lists the nodes a walk reaches so that every edge
//! either runs forward in the list or closes a loop onto the head of a
//! component that contains its source. Components nest and each one is
//! contiguous, head first. The heads cut every cycle, which makes them the
//! places where a fixpoint solve over an infinite-height lattice widens
//! (F. Bourdoncle, *Efficient chaotic iteration strategies with widenings*,
//! FMPA 1993).
//!
//! The loop nesting is the cycle nesting of Havlak and Ramalingam over one
//! depth-first walk, which Kim, Venet and Thakur (*Deterministic parallel
//! fixpoint computation*, POPL 2020) show yields Bourdoncle's order: the same
//! construction as [`LoopForest::compute_cycles`](crate::LoopForest::compute_cycles),
//! shared with it. It examines each node and edge once and costs O(m α(n)).

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use crate::graph::fanout::DenseFanout;
use crate::graph::loop_forest::cycles::CycleNesting;
use crate::graph::traverse::{Adjacency, Outgoing, by_axis};
use crate::graph::view::Direction;
use crate::graph::view::{DenseId, GraphView, RootedView, assert_in_bound};
use crate::identity::{NO_INDEX, slot, word};

/// A weak topological order of the nodes that a walk from a root reaches.
///
/// The order is hierarchical: a **component** is a nonempty contiguous run of
/// [`order`](Self::order) that starts with its **head**, and two components
/// are either disjoint or nested. For every edge `u → v` along the walk's
/// direction, either `u` comes before `v`, or `v` is the head of a component
/// that contains `u`. Every cycle therefore contains a head, and a node with
/// an edge to itself is a head whose component is that node alone.
///
/// Iterating the order with the recursive strategy — stabilize each component
/// before moving past it, checking stabilization at its head — computes a
/// fixpoint, and the heads are where a solve over an infinite-height lattice
/// widens. [`IterationOrder::WeakTopological`](crate::IterationOrder) runs the
/// generic solvers in this order.
///
/// The order depends only on the graph: the walk is depth first and takes each
/// node's neighbors in adjacency order. A component's head is the first of its
/// nodes that the walk reaches, and the components and loose nodes inside a
/// component, like those outside every component, follow reverse postorder.
///
/// # Examples
///
/// The outer loop `1 → 2 → 1` encloses the inner loop `2 → 3 → 2`:
///
/// ```
/// use cfglib::{Graph, Rooted, WeakTopologicalOrder};
///
/// let mut graph = Graph::<(), ()>::new();
/// let [entry, outer, inner, latch, exit] = [(); 5].map(|()| graph.add_node(()));
/// graph.add_edge(entry, outer, ());
/// graph.add_edge(outer, inner, ());
/// graph.add_edge(inner, latch, ());
/// graph.add_edge(latch, inner, ());
/// graph.add_edge(inner, outer, ());
/// graph.add_edge(outer, exit, ());
///
/// let order = WeakTopologicalOrder::compute(&Rooted::new(&graph, entry));
/// // entry (outer (inner latch)) exit
/// assert_eq!(order.order(), [entry, outer, inner, latch, exit]);
/// assert_eq!(order.heads().collect::<Vec<_>>(), [outer, inner]);
/// assert_eq!(order.head_of(latch), Some(inner));
/// assert_eq!(order.head_of(inner), Some(inner));
/// assert_eq!(order.head_of(exit), None);
/// assert_eq!(order.rank(exit), Some(4));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeakTopologicalOrder<N> {
    /// The nodes in order; every component is contiguous, head first.
    order: Vec<N>,
    /// Per position: one past the last position of the component headed
    /// there, or the position itself where no component starts.
    component_end: Vec<u32>,
    /// Per position: the position of the innermost head whose component
    /// contains the element there, other than the element itself; [`NO_INDEX`]
    /// outside every component.
    parent: Vec<u32>,
    /// Per dense node index: its position, [`NO_INDEX`] when the walk missed it.
    rank: Vec<u32>,
}

impl<N: DenseId> WeakTopologicalOrder<N> {
    /// The weak topological order of the nodes the root of `graph` reaches.
    ///
    /// The order follows the view's edges; a [`Reversed`](crate::Reversed)
    /// view under the [`Rooted`](crate::Rooted) gives the order of a
    /// backward problem.
    ///
    /// # Panics
    ///
    /// Panics when the root is not below the node bound of `graph`.
    #[must_use]
    pub fn compute<G>(graph: &G) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        assert_in_bound(graph, graph.root(), "root");
        build(Outgoing, graph, [graph.root()])
    }

    /// The weak topological order of the nodes that `roots` reach along
    /// `direction`, walking from each root in turn that an earlier walk
    /// missed.
    ///
    /// A walk from a later root can reach the nodes of an earlier one but not
    /// the reverse, so the later walk's nodes come first.
    pub(crate) fn from_roots<G>(
        graph: &G,
        roots: impl IntoIterator<Item = N>,
        direction: Direction,
    ) -> Self
    where
        G: GraphView<NodeId = N>,
    {
        by_axis!(direction, build(graph, roots))
    }

    /// Every node the order covers, components contiguous and head first.
    #[must_use]
    pub fn order(&self) -> &[N] {
        &self.order
    }

    /// The position of `node` in [`order`](Self::order), or `None` when the
    /// walk did not reach it.
    #[must_use]
    pub fn rank(&self, node: N) -> Option<u32> {
        self.rank
            .get(node.index())
            .copied()
            .filter(|&rank| rank != NO_INDEX)
    }

    /// Whether `node` heads a component.
    #[must_use]
    pub fn is_head(&self, node: N) -> bool {
        self.rank(node)
            .is_some_and(|rank| self.heads_at(slot(rank)))
    }

    /// Every head, in order; an enclosing head comes before the heads nested
    /// in its component.
    pub fn heads(&self) -> impl Iterator<Item = N> + '_ {
        (0..self.order.len())
            .filter(|&position| self.heads_at(position))
            .map(|position| self.order[position])
    }

    /// The head of the innermost component that contains `node`: `node`
    /// itself when it is a head, and `None` when `node` is in no component
    /// or was not reached.
    #[must_use]
    pub fn head_of(&self, node: N) -> Option<N> {
        let position = slot(self.rank(node)?);
        if self.heads_at(position) {
            return Some(node);
        }
        let parent = self.parent[position];
        (parent != NO_INDEX).then(|| self.order[slot(parent)])
    }

    /// Whether a component starts at `position`.
    pub(crate) fn heads_at(&self, position: usize) -> bool {
        slot(self.component_end[position]) > position
    }

    /// The positions of the outermost element that holds `position`: the
    /// outermost component that contains it, or the position alone when no
    /// component does.
    pub(crate) fn outermost(&self, position: usize) -> Range<usize> {
        let mut top = position;
        while self.parent[top] != NO_INDEX {
            top = slot(self.parent[top]);
        }
        top..slot(self.component_end[top]).max(top + 1)
    }

    /// For each position, the end of the outermost element that holds it.
    ///
    /// The outermost elements partition the order into consecutive runs, so
    /// one pass finds them all.
    pub(crate) fn outermost_ends(&self) -> Vec<u32> {
        let mut ends = Vec::with_capacity(self.order.len());
        let mut end = 0;
        for position in 0..self.order.len() {
            if position >= end {
                end = self.outermost(position).end;
            }
            ends.push(word(end));
        }
        ends
    }
}

fn build<G: GraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    roots: impl IntoIterator<Item = G::NodeId>,
) -> WeakTopologicalOrder<G::NodeId> {
    let nesting = CycleNesting::compute(axis, graph, roots);
    let count = nesting.len();

    // Group the elements of each component, and of the top level in row
    // `count`, in reverse postorder: compressed rows indexed by head number.
    let row_of = |number: usize| match nesting.parent_head(number) {
        NO_INDEX => count,
        head => slot(head),
    };
    let (mut start, mut members) = (Vec::new(), Vec::new());
    DenseFanout::<u32, usize>::group_into(
        count + 1,
        || {
            nesting
                .postorder()
                .iter()
                .rev()
                .map(|&number| (row_of(slot(number)), number))
        },
        &mut start,
        &mut members,
    );

    let mut order = Vec::with_capacity(count);
    let mut component_end = Vec::with_capacity(count);
    let mut parent = Vec::with_capacity(count);
    let mut rank = vec![NO_INDEX; graph.node_bound()];
    // A frame is a component's head position (`NO_INDEX` for the top level) and
    // the range of its members still to emit.
    let mut frames = vec![(NO_INDEX, slot(start[count]), slot(start[count + 1]))];
    while let Some(frame) = frames.last_mut() {
        let (head_position, next, end) = *frame;
        if next == end {
            frames.pop();
            if head_position != NO_INDEX {
                component_end[slot(head_position)] = word(order.len());
            }
            continue;
        }
        frame.1 += 1;
        let number = members[next];
        let node = nesting.node(number);
        let position = word(order.len());
        rank[node.index()] = position;
        order.push(node);
        component_end.push(position);
        parent.push(head_position);
        let number = slot(number);
        if nesting.heads(number) {
            frames.push((position, slot(start[number]), slot(start[number + 1])));
        }
    }
    WeakTopologicalOrder {
        order,
        component_end,
        parent,
        rank,
    }
}

#[cfg(test)]
mod tests;
