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
//! Construction is one depth-first walk and one bottom-up pass that collapses
//! each loop body into its head with a union-find, the loop-nesting
//! construction of Havlak and Ramalingam that Kim, Venet and Thakur
//! (*Deterministic parallel fixpoint computation*, POPL 2020) show yields
//! Bourdoncle's order. Each node and edge is examined once, except that an edge
//! entering a component below its head is examined once more for every
//! enclosing component that it also enters below the head.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use crate::graph::traverse::{Adjacency, by_axis};
use crate::graph::view::Direction;
use crate::graph::view::{DenseId, GraphView, assert_in_bound};
use crate::identity::{NO_INDEX, slot, word};
use crate::union_find::DisjointSet;

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
/// use cfglib::{Graph, Direction, WeakTopologicalOrder};
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
/// let order = WeakTopologicalOrder::compute(&graph, entry, Direction::Forward);
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
    /// The weak topological order of the nodes that `root` reaches along
    /// `direction`.
    ///
    /// # Panics
    ///
    /// Panics when `root` is not below the node bound of `graph`.
    #[must_use]
    pub fn compute<G>(graph: &G, root: N, direction: Direction) -> Self
    where
        G: GraphView<NodeId = N>,
    {
        assert_in_bound(graph, root, "root");
        Self::from_roots(graph, [root], direction)
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

/// The depth-first forest that a weak topological order is built from.
pub(crate) struct DepthFirstForest {
    /// The dense index of each node, by preorder number.
    preorder: Vec<u32>,
    /// Per dense index: the preorder number, [`NO_INDEX`] when the walk missed it.
    number: Vec<u32>,
    /// Per preorder number: the largest preorder number in its subtree.
    last: Vec<u32>,
    /// Preorder numbers in postorder.
    postorder: Vec<u32>,
}

impl DepthFirstForest {
    /// Walks depth first along `axis` from each of `roots` that an earlier
    /// walk missed, taking neighbors in adjacency order.
    fn compute<G: GraphView, A: Adjacency>(
        axis: A,
        graph: &G,
        roots: impl IntoIterator<Item = G::NodeId>,
    ) -> Self {
        let mut forest = Self {
            preorder: Vec::new(),
            number: vec![NO_INDEX; graph.node_bound()],
            last: Vec::new(),
            postorder: Vec::new(),
        };
        // A frame is a node's preorder number and the length of `pending`
        // below its own neighbors, which sit reversed so the first pops first.
        let mut frames: Vec<(u32, usize)> = Vec::new();
        let mut pending: Vec<G::NodeId> = Vec::new();
        for root in roots {
            if forest.number[root.index()] != NO_INDEX {
                continue;
            }
            forest.enter(axis, graph, root, &mut frames, &mut pending);
            while let Some(&(node, base)) = frames.last() {
                if pending.len() > base {
                    let next = pending
                        .pop()
                        .expect("a frame below the stack top is nonempty");
                    if forest.number[next.index()] == NO_INDEX {
                        forest.enter(axis, graph, next, &mut frames, &mut pending);
                    }
                } else {
                    frames.pop();
                    forest.last[slot(node)] = word(forest.preorder.len() - 1);
                    forest.postorder.push(node);
                }
            }
        }
        forest
    }

    fn enter<G: GraphView, A: Adjacency>(
        &mut self,
        axis: A,
        graph: &G,
        node: G::NodeId,
        frames: &mut Vec<(u32, usize)>,
        pending: &mut Vec<G::NodeId>,
    ) {
        let number = word(self.preorder.len());
        self.number[node.index()] = number;
        self.preorder.push(word(node.index()));
        self.last.push(number);
        let base = pending.len();
        pending.extend(axis.neighbors(graph, node));
        pending[base..].reverse();
        frames.push((number, base));
    }

    /// The node numbered `number` in preorder.
    fn node<N: DenseId>(&self, number: u32) -> N {
        N::from_index(slot(self.preorder[slot(number)]))
    }

    /// The reached nodes in reverse postorder.
    pub(crate) fn reverse_postorder<N: DenseId>(&self) -> impl ExactSizeIterator<Item = N> + '_ {
        self.postorder.iter().rev().map(|&number| self.node(number))
    }
}

/// Walks depth first along `axis` from each root that an earlier walk missed;
/// the forest the reverse-postorder iteration order reads.
pub(crate) fn depth_first_forest<G: GraphView>(
    graph: &G,
    roots: impl IntoIterator<Item = G::NodeId>,
    direction: Direction,
) -> DepthFirstForest {
    by_axis!(direction, forest_along(graph, roots))
}

fn forest_along<G: GraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    roots: impl IntoIterator<Item = G::NodeId>,
) -> DepthFirstForest {
    DepthFirstForest::compute(axis, graph, roots)
}

/// The loop nesting of a depth-first forest: which preorder numbers head a
/// component, and the head whose component absorbed each one.
struct Nesting {
    /// Per preorder number: whether it heads a component.
    head: Vec<bool>,
    /// Per preorder number: the innermost head whose component contains it,
    /// other than itself; [`NO_INDEX`] outside every component.
    parent: Vec<u32>,
}

impl Nesting {
    /// Collapses every component into its head, innermost first.
    ///
    /// A head is a node with an upstream neighbor in its own subtree, and its
    /// component is every node of that subtree that reaches it without
    /// leaving the subtree. Heads are visited by descending preorder number, so
    /// each nested component is already collapsed, and the union-find names it
    /// by its head, the member with the smallest number. An edge from outside
    /// the head's subtree into a collapsed component waits on the head's
    /// pending list until an enclosing head's subtree contains its source.
    fn compute<G: GraphView, A: Adjacency>(axis: A, graph: &G, forest: &DepthFirstForest) -> Self {
        let upstream = axis.against();
        let count = forest.preorder.len();
        let mut sets = DisjointSet::new(count);
        let mut nesting = Self {
            head: vec![false; count],
            parent: vec![NO_INDEX; count],
        };
        let mut pending_first = vec![NO_INDEX; count];
        let mut pending: Vec<(u32, u32)> = Vec::new();
        let mut body: Vec<u32> = Vec::new();
        for head in (0..count).rev() {
            let head_number = word(head);
            let last = forest.last[head];
            let inside =
                |number: u32| number != NO_INDEX && head_number <= number && number <= last;
            for source in upstream.neighbors(graph, forest.node::<G::NodeId>(head_number)) {
                let number = forest.number[source.index()];
                if inside(number) {
                    nesting.head[head] = true;
                    nesting.absorb(&mut sets, &mut body, head_number, number);
                }
            }
            while let Some(member) = body.pop() {
                let mut entry = pending_first[slot(member)];
                let sources = upstream
                    .neighbors(graph, forest.node::<G::NodeId>(member))
                    .map(|source| forest.number[source.index()])
                    .chain(core::iter::from_fn(|| {
                        (entry != NO_INDEX).then(|| {
                            let (source, next) = pending[slot(entry)];
                            entry = next;
                            source
                        })
                    }));
                let mut outside = Vec::new();
                for number in sources {
                    if inside(number) {
                        nesting.absorb(&mut sets, &mut body, head_number, number);
                    } else if number != NO_INDEX {
                        outside.push(number);
                    }
                }
                for number in outside {
                    pending.push((number, pending_first[head]));
                    pending_first[head] = word(pending.len() - 1);
                }
            }
        }
        nesting
    }

    /// Adds the component of `number` to the body of `head`, unless it is
    /// already there.
    fn absorb(&mut self, sets: &mut DisjointSet, body: &mut Vec<u32>, head: u32, number: u32) {
        let representative = word(sets.find(slot(number)));
        if representative != head {
            self.parent[slot(representative)] = head;
            sets.union_toward_min(slot(head), slot(representative));
            body.push(representative);
        }
    }
}

fn build<G: GraphView, A: Adjacency>(
    axis: A,
    graph: &G,
    roots: impl IntoIterator<Item = G::NodeId>,
) -> WeakTopologicalOrder<G::NodeId> {
    let forest = DepthFirstForest::compute(axis, graph, roots);
    let nesting = Nesting::compute(axis, graph, &forest);
    let count = forest.preorder.len();

    // Group the elements of each component, and of the top level in row
    // `count`, in reverse postorder: compressed rows indexed by head number.
    // Counting into `start[row + 2]` leaves, after the fill below advances
    // `start[row + 1]`, each row between `start[row]` and `start[row + 1]`.
    let row_of = |number: usize| match nesting.parent[number] {
        NO_INDEX => count,
        head => slot(head),
    };
    let mut start = vec![0_u32; count + 3];
    for number in 0..count {
        start[row_of(number) + 2] += 1;
    }
    for index in 2..start.len() {
        start[index] += start[index - 1];
    }
    let mut members = vec![0_u32; count];
    for &number in forest.postorder.iter().rev() {
        let row = row_of(slot(number)) + 1;
        members[slot(start[row])] = number;
        start[row] += 1;
    }

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
        let node = forest.node::<G::NodeId>(number);
        let position = word(order.len());
        rank[node.index()] = position;
        order.push(node);
        component_end.push(position);
        parent.push(head_position);
        let number = slot(number);
        if nesting.head[number] {
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
